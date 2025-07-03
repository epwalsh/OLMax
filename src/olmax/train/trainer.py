import dataclasses
import fnmatch
import functools as ft
import gc
import itertools
import logging
import math
import signal
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Generic, Iterable, Iterator, Sequence, Type, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp

from .. import distributed as dist
from .. import fs, jax_utils, nn, utils
from ..exceptions import *
from ..optim import (
    ClipByGlobalNormState,
    OptimConfig,
    extract_hyperparameter,
    extract_state,
)
from ..types import *
from .callbacks import Callback

log = logging.getLogger(__name__)

M = TypeVar("M", bound=nn.Module)
B = TypeVar("B")


@dataclass
class Trainer(Generic[M, B]):
    work_dir: Path
    """
    Local temporary working directory.
    """
    save_folder: PathOrStr
    """
    Local or remote save folder to persist checkpoints and other artifacts.
    """

    optim: OptimConfig
    loss_fun: Callable[[M, B], Array]

    mesh: dist.MeshResource = dataclasses.field(default_factory=dist.MeshResource.FSDP)

    param_dtype: DTypeLike = "float32"
    compute_dtype: DTypeLike = "bfloat16"

    callbacks: OrderedDict[str, Callback] = dataclasses.field(default_factory=OrderedDict)
    cancel_check_interval: int = 5
    gc_interval: int = 1000
    max_duration: Duration = dataclasses.field(default_factory=lambda: Duration.epochs(1))
    global_tokens_per_batch: int | Callable[[Sequence[B]], int] | None = None
    log_to_console: Sequence[str] = (
        "train/loss",
        "optim/lr",
        "system/*",
        "throughput/data loading*",
        "throughput/TPS device*",
    )
    """
    Names or glob patterns of metrics to log to the console.
    """

    # Internal bookkeeping.
    _step: int = dataclasses.field(default=0, repr=False)
    _step_this_run: int = dataclasses.field(default=0, repr=False)
    _epoch: int = dataclasses.field(default=1, repr=False)
    _metrics_per_step: OrderedDict[int, dict[str, Scalar]] = dataclasses.field(
        default_factory=OrderedDict, repr=False
    )
    _canceled: Array = dataclasses.field(
        default_factory=lambda: jax.copy_to_host_async(jnp.array(False)), repr=False
    )
    _cancel_reason: str | None = dataclasses.field(default=None, repr=False)
    _error: BaseException | None = dataclasses.field(default=None, repr=False)
    _global_train_tokens_seen: int | None = dataclasses.field(default=None, repr=False)
    _bps_average: utils.RunningAverage = dataclasses.field(
        default_factory=lambda: utils.RunningAverage(0.0), repr=False
    )

    def __post_init__(self):
        # Ensure working directory and save folder exist.
        self.work_dir = Path(self.work_dir)
        self.work_dir.mkdir(exist_ok=True, parents=True)
        if not fs.is_url(self.save_folder):
            self.save_folder = Path(fs.normalize_path(self.save_folder))
            self.save_folder.mkdir(exist_ok=True, parents=True)

        # Set pointer to self in all callbacks.
        for callback in self.callbacks.values():
            callback.trainer = self

        # Sort callbacks by (priority, name).
        # We do this for 2 reasons: (1) to respect the priority, and (2) to ensure the callback
        # order is consistent across the process group since some callbacks make distributed
        # synchronization/communication calls.
        self._sort_callbacks()

        for callback in self._iter_callbacks():
            callback.post_attach()

    def add_callback(self, name: str, callback: Callback):
        if name in self.callbacks:
            raise CallbackExistsError(f"A callback with name '{name}' already exists!")
        callback.trainer = self
        self.callbacks[name] = callback
        self._sort_callbacks()
        callback.post_attach()

    def has_callback(self, cb_class: Type[Callback]) -> bool:
        """
        Check if the trainer already has a registered instance of the given callback class.
        """
        for cb in self.callbacks.values():
            if isinstance(cb, cb_class):
                return True
        return False

    @property
    def step(self) -> int:
        return self._step

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def global_train_tokens_seen(self) -> int | None:
        return self._global_train_tokens_seen

    @property
    def is_canceled(self) -> bool:
        """
        Check if the run is canceled.

        .. warning::
            This triggers a host-device sync.
        """
        if self._error is not None:
            raise RuntimeError("An error occurred") from self._error
        return self._canceled.item()

    @property
    def training_complete(self) -> bool:
        if self._error is not None:
            raise RuntimeError("An error occurred") from self._error

        if self.step % self.cancel_check_interval == 0 and self.is_canceled:
            return True
        elif self._duration_due(self.max_duration):
            return True
        else:
            return False

    def cancel_run(self, reason: str):
        """
        Mark the run canceled. This must be called from process rank 0 during distributed training.
        It can also be called from other processes.

        :param reason: The reason for canceling.
        """
        self._cancel_reason = reason
        self._canceled = jax.copy_to_host_async(jnp.array(True))
        log.warning(f"Run canceled. Reason: {reason}")

    def record_metric(self, name: str, value: Scalar):
        self._metrics_per_step[self.step][name] = jax.copy_to_host_async(value)

    def fit(self, model: M, data_loader: Iterable[Sequence[B]]) -> tuple[M, Optim, OptState]:
        dist.barrier("pre-train-setup")

        self._step = 0
        self._step_this_run = 0
        self._epoch = 1
        self._bps_average.reset()
        self._canceled = jax.copy_to_host_async(jnp.array(False))
        self._cancel_reason = None
        self._canceling_rank = None

        # Disable automatic garbage collection.
        gc.disable()
        gc.collect()

        # Install SIGTERM + SIGINT handlers.
        og_sigterm_handler = signal.signal(signal.SIGTERM, self._handle_os_signal)
        og_sigint_handler = signal.signal(signal.SIGINT, self._handle_os_signal)

        def _shutdown(wait: bool = True):
            self._log_metrics()

            for callback in self._iter_callbacks():
                callback.close()

            # Reset garbage collection.
            gc.collect()
            gc.enable()

            # Restore original signal handlers.
            signal.signal(signal.SIGTERM, og_sigterm_handler)
            signal.signal(signal.SIGINT, og_sigint_handler)

            if wait:
                dist.barrier("post-train-loop")

        log.info("Loading first batch...")
        batches = iter(data_loader)
        first_batch = next(batches)
        batches = itertools.chain([first_batch], batches)
        num_microbatches = len(first_batch)

        param_sharding = model.get_param_shardings()

        # Partition model ahead of time for efficiency.
        params, static = eqx.partition(model, eqx.is_array)

        log.info("Initializing optimizer...")
        optim, opt_state = self.optim.build(params, num_microbatches)
        opt_state_sharding = self.mesh.get_opt_state_sharding(opt_state)

        train_batch = self._make_train_batch(
            static=static,
            optim=optim,
            param_sharding=param_sharding,
            opt_state_sharding=opt_state_sharding,
            num_microbatches=num_microbatches,
        )

        log.info("Callback order:")
        for i, callback_name in enumerate(self.callbacks.keys()):
            log.info(f"  - Callback {i+1}: {callback_name}")

        for callback in self._iter_callbacks():
            callback.pre_train()

        dist.barrier("pre-train-loop")

        log.info("Starting training...")
        train_start = time.perf_counter()
        try:
            while not self.training_complete:
                params, opt_state = self._fit_epoch(
                    train_batch=train_batch,
                    params=params,
                    opt_state=opt_state,
                    batches=batches,
                )
                batches = iter(data_loader)
        except BaseException as exc:
            log.error(f"Training failed due to:\n{exc}")
            for callback in self._iter_callbacks():
                callback.on_error(exc)
            _shutdown(wait=False)
            raise

        for callback in self._iter_callbacks():
            callback.post_train()

        # Re-combine params and state into model object.
        model = eqx.combine(params, static)

        _shutdown()
        train_end = time.perf_counter()
        log.info(
            f"Training complete. Elapsed time: {utils.format_timedelta(train_end - train_start)}"
        )

        return model, optim, opt_state

    def _fit_epoch(
        self,
        train_batch: Callable[[M, OptState, Sequence[B]], tuple[M, OptState]],
        params: M,
        opt_state: OptState,
        batches: Iterator[Sequence[B]],
    ) -> tuple[M, OptState]:
        log.info(f"Starting epoch {self.epoch}...")
        epoch_start = time.perf_counter()

        for callback in self._iter_callbacks():
            callback.pre_epoch()

        batch_start = time.perf_counter()
        while True:
            if self.training_complete:
                # Finishing before the epoch is complete.
                # Log any remaining metrics.
                self._log_metrics()
                return params, opt_state

            # Bookkeeping.
            self._step += 1
            self._step_this_run += 1
            self._metrics_per_step[self.step] = {}

            # Maybe synchronize cancellation.
            if self.step % self.cancel_check_interval == 0:
                self._synchronize_cancellation()

            for callback in self._iter_callbacks():
                callback.pre_step()

            with jax.profiler.StepTraceAnnotation("train_step", step_num=self.step):
                for callback in self._iter_callbacks():
                    callback.pre_load_batch()

                # Load next batch.
                with jax.profiler.TraceAnnotation("load_batch"):
                    batch_load_start = time.perf_counter()
                    try:
                        batch = next(batches)
                    except StopIteration:
                        break
                    batch_load_end = time.perf_counter()
                    self.record_metric(
                        "throughput/data loading time", batch_load_end - batch_load_start
                    )

                    global_train_tokens_this_batch: int | None = None
                    if isinstance(self.global_tokens_per_batch, int):
                        global_train_tokens_this_batch = self.global_tokens_per_batch
                    elif self.global_tokens_per_batch is not None:
                        global_train_tokens_this_batch = self.global_tokens_per_batch(batch)

                    for callback in self._iter_callbacks():
                        callback.post_load_batch(batch)

                # Train on batch.
                with jax.profiler.TraceAnnotation("train_batch"):
                    params, opt_state = train_batch(params, opt_state, batch)

                # More bookkeeping.
                if global_train_tokens_this_batch is not None:
                    if self._global_train_tokens_seen is None:
                        self._global_train_tokens_seen = 0
                    self._global_train_tokens_seen += global_train_tokens_this_batch
                self.record_metric(
                    "system/peak device mem usage (MiB)",
                    utils.bytes_to_mib(jax_utils.get_peak_local_device_memory_usage()),
                )

                for callback in self._iter_callbacks():
                    callback.post_train_batch()

                # Log metrics from previous step(s).
                with jax.profiler.TraceAnnotation("log_metrics"):
                    self._log_metrics(exclude={self.step})

                # Maybe run garbage collection.
                if self.step % self.gc_interval == 0:
                    gc.collect()

            for callback in self._iter_callbacks():
                callback.post_step()

            # Lastly, record throughput.
            # NOTE: this should always be called after `self._log_metrics()`, which is a host-device
            # synchronization point.
            batch_end = time.perf_counter()
            bps = 1 / (batch_end - batch_start)
            bps_avg = None if self._step_this_run < 10 else self._bps_average.update(bps)
            bps_std = (
                None
                if self._bps_average.count < 5
                else math.sqrt(self._bps_average.get_sample_variance())
            )
            self.record_metric("throughput/BPS", bps)
            if bps_avg is not None:
                self.record_metric("throughput/BPS average", bps_avg)
            if bps_std is not None:
                self.record_metric("throughput/BPS stddev", bps_std)
            if global_train_tokens_this_batch is not None:
                tps = bps * global_train_tokens_this_batch
                self.record_metric("throughput/TPS", tps)
                device_tps = tps / dist.get_global_device_count()
                self.record_metric("throughput/TPS device", device_tps)
                if bps_avg is not None:
                    tps_avg = bps_avg * global_train_tokens_this_batch
                    self.record_metric("throughput/TPS average", tps_avg)
                    device_tps_avg = tps_avg / dist.get_global_device_count()
                    self.record_metric("throughput/TPS device average", device_tps_avg)
                if bps_std is not None:
                    tps_std = bps_std * global_train_tokens_this_batch
                    self.record_metric("throughput/TPS stddev", tps_std)
                    device_tps_std = tps_std / dist.get_global_device_count()
                    self.record_metric("throughput/TPS device stddev", device_tps_std)
            batch_start = batch_end

        # Log left-over metrics.
        self._log_metrics()

        for callback in self._iter_callbacks():
            callback.post_epoch()

        epoch_end = time.perf_counter()
        log.info(
            f"Epoch {self.epoch} completed in {utils.format_timedelta(epoch_end - epoch_start)}"
        )

        self._epoch += 1
        return params, opt_state

    def _handle_os_signal(self, signalnum, stack_frame):
        del stack_frame

        signame: str | None = None
        if signalnum == signal.SIGTERM:
            signame = "SIGTERM"
        elif signalnum == signal.SIGINT:
            signame = "SIGINT"

        msg: str
        if signame is not None:
            msg = f"{signame} received"
        else:
            msg = f"Sig({signalnum}) received"

        log.warning(msg)
        self.cancel_run(msg)

    def _sort_callbacks(self):
        self.callbacks = OrderedDict(
            (
                (k, cb)
                for _, (k, cb) in sorted(
                    enumerate(self.callbacks.items()),
                    key=lambda x: (x[1][1].priority, -1 * x[0]),
                    reverse=True,
                )
            )
        )

    def _iter_callbacks(self) -> Iterable[Callback]:
        for callback in self.callbacks.values():
            if callback.enabled:
                yield callback

    def _log_metrics(self, exclude: set[int] | None = None):
        for step_to_log in list(self._metrics_per_step.keys()):
            if exclude and step_to_log in exclude:
                continue

            metrics_to_log = self._metrics_per_step.pop(step_to_log)
            log.info(
                f"[step {step_to_log:03d}]\n"
                + "\n".join(
                    [
                        f"    {name}={utils.format_scalar(metrics_to_log[name])}"
                        for name in sorted(metrics_to_log.keys())
                        if any(fnmatch.fnmatch(name, pat) for pat in self.log_to_console)
                    ]
                )
            )
            for callback in self._iter_callbacks():
                callback.log_metrics(step_to_log, metrics_to_log)

    def _synchronize_cancellation(self):
        self._canceled = jax.copy_to_host_async(dist.synchronize_array(self._canceled))

    def _make_train_batch(
        self,
        *,
        static: M,
        optim: Optim,
        param_sharding: M,
        opt_state_sharding: OptState,
        num_microbatches: int,
    ):
        @jax.named_scope("compute_loss_and_grads")
        def compute_loss_and_grads(model: M, batch: B) -> tuple[Array, M]:
            # Cast model to lower precision compute dtype.
            if self.compute_dtype != self.param_dtype:
                model_with_compute_dtype = jax_utils.cast_tree(model, self.compute_dtype)
            else:
                model_with_compute_dtype = model

            # Do forward+backward passes.
            loss, grads = eqx.filter_value_and_grad(self.loss_fun)(model_with_compute_dtype, batch)
            grads = jax.lax.with_sharding_constraint(grads, param_sharding)

            # Cast grads to param dtype.
            if self.compute_dtype != self.param_dtype:
                grads = jax_utils.cast_tree(grads, self.param_dtype)

            return loss, grads

        @jax.named_scope("step_optimizer")
        def step_optimizer(params: M, grads: M, opt_state: OptState):
            # Prepare updates.
            updates, opt_state = optim.update(grads, opt_state, params)  # pyright: ignore

            # Reinforce sharding constraints.
            updates = jax.lax.with_sharding_constraint(updates, param_sharding)
            opt_state = jax.lax.with_sharding_constraint(opt_state, opt_state_sharding)

            # Apply updates.
            params = eqx.apply_updates(params, updates)

            # Reinforce sharding constraints.
            params = jax.lax.with_sharding_constraint(params, param_sharding)

            return params, opt_state

        @ft.partial(jax.jit, donate_argnums=[0, 1, 2])
        @jax.named_scope("process_microbatch")
        def process_microbatch(
            params: M, microbatch: B, opt_state: OptState
        ) -> tuple[M, OptState, Array]:
            # Reconstruct full model object and enforce sharding constraints.
            model = eqx.combine(params, static)
            model = jax.lax.with_sharding_constraint(model, param_sharding)
            opt_state = jax.lax.with_sharding_constraint(opt_state, opt_state_sharding)

            # Compute loss and gradients.
            loss, grads = compute_loss_and_grads(model, microbatch)

            # Take optimizer step.
            params, opt_state = step_optimizer(params, grads, opt_state)

            return params, opt_state, loss

        def train_batch(params: M, opt_state: OptState, batch: Sequence[B]) -> tuple[M, OptState]:
            assert len(batch) == num_microbatches

            # Process one micro-batch at a time.
            batch_losses: list[Array] = []
            for microbatch in batch:
                params, opt_state, mb_loss = process_microbatch(params, microbatch, opt_state)
                batch_losses.append(mb_loss)

            # Reduced loss over micro-batches.
            self.record_metric("train/loss", jnp.stack(batch_losses).mean())

            # Collect learning rate.
            self.record_metric(
                "optim/lr", extract_hyperparameter(opt_state, "learning_rate").copy()
            )

            # If using gradient clipping, collect the global gradient norm.
            if (clipping_state := extract_state(opt_state, ClipByGlobalNormState)) is not None:
                self.record_metric("optim/g_norm", clipping_state.global_norm.copy())

            return params, opt_state

        return train_batch

    def _duration_due(self, duration: Duration) -> bool:
        return duration.due(step=self.step, tokens=self.global_train_tokens_seen, epoch=self.epoch)
