import dataclasses
import fnmatch
import functools as ft
import gc
import itertools
import logging
import math
import os
import re
import signal
import tempfile
import time
from collections import OrderedDict
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Callable, Generic, Iterable, Iterator, Sequence, Type, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp

from .. import data
from .. import distributed as dist
from .. import fs, jax_utils, nn, utils
from ..checkpoint.utils import AsyncSaveHandle
from ..exceptions import *
from ..optim import (
    ClipByGlobalNormState,
    OptimConfig,
    extract_hyperparameter,
    extract_state,
)
from ..types import *
from .callbacks import Callback
from .checkpointer import Checkpointer, SimpleCheckpointer

log = logging.getLogger(__name__)

M = TypeVar("M", bound=nn.Module)
B = TypeVar("B")


@dataclass
class TrainState(Generic[M]):
    step: int
    epoch: int
    global_train_tokens_seen: int | None
    params: M
    static: M
    opt_state: OptState
    data_loader_state: Any
    callbacks_state: dict[str, Any]

    @property
    def model(self) -> M:
        return eqx.combine(self.params, self.static)

    @model.setter
    def model(self, model: M):
        params, static = eqx.partition(model, eqx.is_array)
        self.params = params
        self.static = static


class TrainMetrics(StrEnum):
    loss = "train/loss"
    lr = "optim/lr"
    grad_norm = "optim/g_norm"


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
    """
    Config for the optimizer to use.
    """
    loss_fun: Callable[[M, B, dict[str, PyTree] | None], Array]
    """
    The loss function to use. Should take a model, a micro-batch, and an optional buffer cache as arguments
    and return a scalar array.
    """
    mesh: dist.MeshResource = dataclasses.field(default_factory=dist.MeshResource.FSDP)
    """
    The global mesh resource.
    """
    grad_dtype: DTypeLike = "float32"
    """
    The data type for gradients.
    """
    compute_dtype: DTypeLike = "bfloat16"
    """
    The data type to run the forward/backward pass in.
    """
    callbacks: OrderedDict[str, Callback] = dataclasses.field(default_factory=OrderedDict)
    """
    Callbacks to use.
    """
    checkpointer: Checkpointer = dataclasses.field(default_factory=SimpleCheckpointer)
    """
    The checkpointer to use for saving/restoring checkpoints.
    """
    async_checkpointing: bool = True
    """
    Whether or not to save checkpoints asynchronously.
    """
    checkpoint_interval: int | None = None
    """
    The interval in steps to save checkpoints.
    """
    cancel_check_interval: int = 5
    """
    The interval in steps to synchronize cancellation conditions.
    """
    gc_interval: int = 1000
    """
    The interval in steps to run garbage collection.
    """
    max_duration: Duration = dataclasses.field(default_factory=lambda: Duration.epochs(1))
    """
    The duration to train for.
    """
    global_tokens_per_batch: int | Callable[[Sequence[B]], int] | None = None
    """
    When training on tokens, you should set this to the number of global tokens per batch, or
    a callback that takes a batch and returns the number of global tokens in the batch.
    """
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
    save_overwrite: bool = False
    """
    If the trainer should overwrite existing checkpoints and other files.
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
    _last_checkpoint: int = dataclasses.field(default=-1, repr=False)
    _checkpoint_save_handle: AsyncSaveHandle | None = dataclasses.field(default=None, repr=False)

    def __post_init__(self):
        # Ensure working directory and save folder exist.
        self.work_dir = Path(self.work_dir)
        self.work_dir.mkdir(exist_ok=True, parents=True)
        if not fs.is_url(self.save_folder):
            self.save_folder = Path(fs.normalize_path(self.save_folder))
            self.save_folder.mkdir(exist_ok=True, parents=True)

        self.checkpointer.work_dir = self.work_dir

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

    def add_callback(self, name: str, callback: Callback) -> "Trainer":
        """
        Add a callback.
        """
        if name in self.callbacks:
            raise CallbackExistsError(f"A callback with name '{name}' already exists!")
        callback.trainer = self
        self.callbacks[name] = callback
        self._sort_callbacks()
        callback.post_attach()
        return self

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
        """
        The current training step.
        """
        return self._step

    @property
    def epoch(self) -> int:
        """
        The current training epoch.
        """
        return self._epoch

    @property
    def global_train_tokens_seen(self) -> int | None:
        """
        The number of training tokens seen globally so far.
        """
        return self._global_train_tokens_seen

    @property
    def is_canceled(self) -> bool:
        """
        If the run has been canceled.

        .. warning::
            This triggers a host-device sync.
        """
        if self._error is not None:
            raise RuntimeError("An error occurred") from self._error
        return self._canceled.item()

    @property
    def training_complete(self) -> bool:
        """
        If training is complete.
        """
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
        """
        Record a metric for logging.
        """
        self._metrics_per_step[self.step][name] = jax.copy_to_host_async(value)

    def persist_working_file(
        self, name: PathOrStr, save_overwrite: bool | None = None
    ) -> PathOrStr:
        """
        Persist a file in the :data:`work_dir` by saving/uploading it to the :data:`save_folder`.

        :param name: The name/path of the file *relative* to the :data:`work_dir`.
        :param save_overwrite: Overwrite an existing file.

        :returns: The full path/URL to the saved file.

        :raises FileNotFoundError: If the file can't be found.
        :raises FileExistsError: If the file already exists in the save folder and :data:`save_overwrite`
            is ``False``.
        """
        if save_overwrite is None:
            save_overwrite = self.save_overwrite
        if Path(name).is_relative_to(self.work_dir):
            name = Path(name).relative_to(self.work_dir)
        source = fs.join_path(self.work_dir, name)
        target = fs.join_path(self.save_folder, name)
        if source != target:
            fs.copy_file(source, target, save_overwrite=save_overwrite)
        elif not fs.file_exists(source):
            raise FileNotFoundError(source)
        return target

    def persist_working_subdir(
        self, name: PathOrStr, save_overwrite: bool | None = None
    ) -> PathOrStr:
        """
        Persist a subdirectory in the :data:`work_dir` by saving/uploading it to the :data:`save_folder`.

        :param name: The name/path of the subdirectory *relative* to the :data:`work_dir`.
        :param save_overwrite: Overwrite an existing file.

        :returns: The full path/URL to the saved file.

        :raises FileNotFoundError: If the subdirectory doesn't exist.
        :raises FileExistsError: If the any of the files already exists in the save folder and :data:`save_overwrite`
            is ``False``.
        """
        if save_overwrite is None:
            save_overwrite = self.save_overwrite
        if Path(name).is_relative_to(self.work_dir):
            name = Path(name).relative_to(self.work_dir)
        source = fs.join_path(self.work_dir, name)
        target = fs.join_path(self.save_folder, name)
        if source != target:
            fs.copy_dir(source, target, save_overwrite=save_overwrite)
        return target

    def write_file(
        self, fname: str, contents: str | bytes, save_overwrite: bool | None = None
    ) -> PathOrStr:
        """
        Write a file to the :data:`save_folder`.

        :param fname: The name of the file to write, relative to the :data:`save_folder`.
        :param contents: The contents of the file to write.
        :param save_overwrite: Overwrite an existing file.

        :returns: The full path/URL of the file.
        """
        if save_overwrite is None:
            save_overwrite = self.save_overwrite
        target = fs.join_path(self.save_folder, fname)
        mode = "wb" if isinstance(contents, bytes) else "wt"
        tmp_file = tempfile.NamedTemporaryFile(mode=mode, delete=False, dir=self.work_dir)
        tmp_path = Path(tmp_file.name)
        try:
            tmp_file.write(contents)
            tmp_file.flush()
            fs.copy_file(tmp_path, target, save_overwrite=save_overwrite)
            return target
        finally:
            tmp_path.unlink(missing_ok=True)

    def fit(
        self,
        model: M,
        data_loader: data.DataLoader,
        *,
        load_path: PathOrStr | None = None,
        buffer_cache: dict[str, PyTree] | None,
    ) -> tuple[M, Optim, OptState]:
        """
        Fit a model to a dataset.

        :returns: The trained model, its optimizer, and the optimizer state.
        """
        dist.barrier("pre-train-setup")

        self._step_this_run = 0
        self._bps_average.reset()
        self._canceled = jax.copy_to_host_async(jnp.array(False))
        self._cancel_reason = None
        self._error = None
        self._last_checkpoint = -1

        # Disable automatic garbage collection.
        gc.disable()
        gc.collect()

        # Install SIGTERM + SIGINT handlers.
        og_sigterm_handler = signal.signal(signal.SIGTERM, self._handle_os_signal)
        og_sigint_handler = signal.signal(signal.SIGINT, self._handle_os_signal)

        def _shutdown(wait: bool = True):
            self._log_metrics()
            self._maybe_wait_for_checkpoint()

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

        if load_path is None:
            load_path = self._find_latest_checkpoint()

        if load_path is not None:
            params, opt_state = self._restore_checkpoint(
                load_path,
                params=params,
                static=static,
                opt_state=opt_state,
                data_loader=data_loader,
            )
        elif self.checkpoint_interval is not None:
            # Save pre-train checkpoint.
            self._save_checkpoint(
                params=params, static=static, opt_state=opt_state, data_loader=data_loader
            )

        train_batch = self._make_train_batch(
            static=static,
            optim=optim,
            param_sharding=param_sharding,
            opt_state_sharding=opt_state_sharding,
            num_microbatches=num_microbatches,
        )

        if self.callbacks:
            log.info(
                "Callback order:\n"
                + "\n".join(
                    [
                        f"❯ Callback {i+1}: {callback_name}"
                        for i, callback_name in enumerate(self.callbacks.keys())
                    ]
                )
            )

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
                    static=static,
                    opt_state=opt_state,
                    data_loader=data_loader,
                    batches=batches,
                    buffer_cache=buffer_cache,
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

        # Maybe save a final checkpoint.
        if self.checkpoint_interval is not None:
            self._save_checkpoint(
                params=params, static=static, opt_state=opt_state, data_loader=data_loader
            )

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
        train_batch: Callable[
            [M, OptState, Sequence[B], dict[str, PyTree] | None], tuple[M, OptState]
        ],
        params: M,
        static: M,
        opt_state: OptState,
        data_loader: data.DataLoader,
        batches: Iterator[Sequence[B]],
        buffer_cache: dict[str, PyTree] | None,
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
                    params, opt_state = train_batch(params, opt_state, batch, buffer_cache)

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

            # Record throughput.
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

            # Maybe save a checkpoint.
            if self.checkpoint_interval is not None and self.step % self.checkpoint_interval == 0:
                self._save_checkpoint(
                    params=params, static=static, opt_state=opt_state, data_loader=data_loader
                )

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

            # Check for nan loss.
            if (loss := metrics_to_log.get(TrainMetrics.loss)) is not None and not math.isfinite(
                loss.item()  # type: ignore
            ):
                raise RuntimeError(f"NaN loss encountered on step {step_to_log}!")

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
        def compute_loss_and_grads(
            params: M, batch: B, buffer_cache: dict[str, PyTree] | None
        ) -> tuple[Array, M]:
            # Cast params to the compute dtype.
            params_with_compute_dtype = jax_utils.cast_tree(params, self.compute_dtype)

            # Reconstruct full model object and enforce sharding constraints.
            model = eqx.combine(params_with_compute_dtype, static)

            # Do forward+backward passes.
            loss, grads = eqx.filter_value_and_grad(self.loss_fun)(model, batch, buffer_cache)
            grads = jax.lax.with_sharding_constraint(grads, param_sharding)

            # Cast grads to the right dtype.
            grads = jax_utils.cast_tree(grads, self.grad_dtype)

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
            params: M,
            microbatch: B,
            opt_state: OptState,
            buffer_cache: dict[str, PyTree] | None,
        ) -> tuple[M, OptState, Array]:
            # Enforce sharding constraints.
            params = jax.lax.with_sharding_constraint(params, param_sharding)
            opt_state = jax.lax.with_sharding_constraint(opt_state, opt_state_sharding)

            # Compute loss and gradients.
            loss, grads = compute_loss_and_grads(params, microbatch, buffer_cache)

            # Take optimizer step.
            params, opt_state = step_optimizer(params, grads, opt_state)

            return params, opt_state, loss

        def train_batch(
            params: M,
            opt_state: OptState,
            batch: Sequence[B],
            buffer_cache: dict[str, PyTree] | None,
        ) -> tuple[M, OptState]:
            assert len(batch) == num_microbatches

            # Process one micro-batch at a time.
            batch_losses: list[Array] = []
            for microbatch in batch:
                params, opt_state, mb_loss = process_microbatch(
                    params, microbatch, opt_state, buffer_cache
                )
                batch_losses.append(mb_loss)

            # Reduced loss over micro-batches.
            self.record_metric(TrainMetrics.loss, jnp.stack(batch_losses).mean())

            # Collect learning rate.
            self.record_metric(
                TrainMetrics.lr, extract_hyperparameter(opt_state, "learning_rate").copy()
            )

            # If using gradient clipping, collect the global gradient norm.
            if (clipping_state := extract_state(opt_state, ClipByGlobalNormState)) is not None:
                self.record_metric(TrainMetrics.grad_norm, clipping_state.global_norm.copy())

            return params, opt_state

        return train_batch

    def _duration_due(self, duration: Duration) -> bool:
        return duration.due(step=self.step, tokens=self.global_train_tokens_seen, epoch=self.epoch)

    def _init_state(
        self,
        *,
        params: M,
        static: M,
        opt_state: OptState,
        data_loader: data.DataLoader,
    ) -> TrainState:
        return TrainState(
            step=self.step,
            epoch=self.epoch,
            global_train_tokens_seen=self.global_train_tokens_seen,
            params=params,
            static=static,
            opt_state=opt_state,
            data_loader_state=data_loader.get_state(),
            callbacks_state={
                name: callback.get_state() for name, callback in self.callbacks.items()
            },
        )

    def _save_checkpoint(
        self,
        *,
        params: M,
        static: M,
        opt_state: OptState,
        data_loader: data.DataLoader,
        save_overwrite: bool | None = None,
        block: bool | None = None,
    ) -> PathOrStr:
        """
        Save a checkpoint.
        """
        self._maybe_wait_for_checkpoint()

        if save_overwrite is None:
            save_overwrite = self.save_overwrite
        if block is None:
            block = not self.async_checkpointing

        step = self.step
        checkpoint_path = fs.join_path(self.save_folder, f"step{step}")
        if step == self._last_checkpoint:
            return checkpoint_path

        if not save_overwrite and not fs.dir_is_empty(checkpoint_path):
            raise FileExistsError(
                f"Checkpoint dir '{checkpoint_path}' is non-empty. Use 'save_overwrite=True' to force overwriting the dir."
            )

        log.info(f"Saving checkpoint for step {step} to '{checkpoint_path}'...")
        state = self._init_state(
            params=params,
            static=static,
            opt_state=opt_state,
            data_loader=data_loader,
        )

        if block:
            self.checkpointer.save(checkpoint_path, state, save_overwrite=self.save_overwrite)
        else:
            self._checkpoint_save_handle = self.checkpointer.save_async(
                checkpoint_path,
                state,
                save_overwrite=save_overwrite,
            )

        gc.collect()
        self._last_checkpoint = max(step, self._last_checkpoint)
        return checkpoint_path

    def _restore_checkpoint(
        self,
        dir: PathOrStr,
        *,
        params: M,
        static: M,
        opt_state: OptState,
        data_loader: data.DataLoader,
    ) -> tuple[M, OptState]:
        log.info(f"Restoring checkpoint from '{dir}'...")
        start_time = time.perf_counter()

        state = self.checkpointer.load(
            dir,
            self._init_state(
                params=params, static=static, opt_state=opt_state, data_loader=data_loader
            ),
        )

        self._step = state.step
        self._epoch = state.epoch
        self._global_train_tokens_seen = state.global_train_tokens_seen
        data_loader.load_state(state.data_loader_state)
        for name, callback in self.callbacks.items():
            if name in state.callbacks_state:
                callback.load_state(state.callbacks_state[name])

        end_time = time.perf_counter()
        log.info(f"Checkpoint restored in {utils.format_timedelta(end_time - start_time)}")

        for callback in self._iter_callbacks():
            callback.post_checkpoint_loaded(dir)

        return state.params, state.opt_state

    def _find_latest_checkpoint(self) -> PathOrStr | None:
        latest_step: int | None = None
        latest_path: PathOrStr | None = None
        for path in fs.list_directory(self.save_folder):
            name = os.path.basename(path)
            if (m := re.match(r"^step(\d+)$", name)) is not None:
                step = int(m.group(1))
                if latest_step is None or step > latest_step:
                    latest_step = step
                    latest_path = path
        return latest_path

    def _maybe_wait_for_checkpoint(self):
        if self._checkpoint_save_handle is None:
            return

        self._checkpoint_save_handle.wait_until_finished()
        self._checkpoint_save_handle.close()
        self._checkpoint_save_handle = None
