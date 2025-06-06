from __future__ import annotations

import argparse
import gc
import logging
import time
from collections import deque
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

import olmax.distributed as dist
import olmax.nn as nn
import olmax.nn.functional as F
from olmax.data.utils import generate_batches_of_sequential_tokens
from olmax.jax_utils import cast_tree, count_params, get_peak_local_device_memory_usage
from olmax.launch.beaker import BeakerRuntime
from olmax.nn.transformer.recipes import (
    Gemma2Like27BRecipe,
    Gemma3Like27BRecipe,
    LlamaLike7BRecipe,
    LlamaLike271MRecipe,
    TransformerRecipe,
)
from olmax.optim import (
    AdamWConfig,
    WarmupCosineDecaySchedule,
    clip_grads_by_global_norm,
)
from olmax.train import prepare_training_environment
from olmax.types import Array, DTypeLike
from olmax.utils import bytes_to_mib, format_scalar, prepare_cli_environment

log = logging.getLogger("main")


def train(
    recipe_name: str,
    beaker_runtime: BeakerRuntime | None = None,
    sequence_length: int | None = None,
    instances_per_device: int | None = None,
    vocab_size: int | None = None,
    param_dtype: DTypeLike = float,
    compute_dtype: DTypeLike = jax.dtypes.bfloat16,
    learning_rate: float | None = None,
    train_steps: int = 100,
    attn_window_size: int | tuple[int, int] | None = None,
    attn_implementation: Literal["xla", "cudnn"] | None = None,
    show_model: bool = False,
    running_avg_tps_count: int = 10,
    mesh_type: Literal["FSDP", "HSDP"] = "FSDP",
    trace_dir: str | None = None,
    max_grad_norm: float | None = None,
) -> tuple[float, int, int]:
    recipe: TransformerRecipe = TransformerRecipe.get_choice_class(recipe_name)
    beaker_gpu_type = None if beaker_runtime is None else beaker_runtime.node.gpu_type
    gpu_type = None if beaker_gpu_type is None else beaker_gpu_type.name.lower()

    if vocab_size is None:
        vocab_size = recipe.default_vocab_size
    if sequence_length is None:
        sequence_length = recipe.default_sequence_length
    if learning_rate is None:
        learning_rate = recipe.default_learning_rate
    if instances_per_device is None:
        batch_size_per_device = recipe.get_mbz_per_device(gpu_type or "A100")
        assert batch_size_per_device % sequence_length == 0
        instances_per_device = batch_size_per_device // sequence_length

    model_config = recipe.build_config(
        vocab_size=vocab_size,
        param_dtype=param_dtype,
        attn_window_size=attn_window_size,
        attn_implementation=attn_implementation,
    )

    batch_size_per_device = sequence_length * instances_per_device
    global_batch_size = batch_size_per_device * dist.get_global_device_count()
    global_batch_size_instances = instances_per_device * dist.get_global_device_count()
    log.info(
        f"Using global batch size of {global_batch_size:,d} tokens, "
        f"which is {global_batch_size_instances:,d} instances of length {sequence_length:,d}."
    )
    log.info(
        f"Using per-device batch size of {batch_size_per_device:,d} tokens, "
        f"which is {instances_per_device:,d} instances of length {sequence_length:,d}."
    )

    key = jax.random.PRNGKey(0)
    model_key, data_key = jax.random.split(key)

    if mesh_type == "FSDP":
        mesh_resource = dist.MeshResource.FSDP()
    elif mesh_type == "HSDP":
        mesh_resource = dist.MeshResource.HSDP(8)
    else:
        raise ValueError(mesh_type)
    log.info(f"Build mesh with axes {mesh_resource.get_mesh_axes_repr()}")

    if beaker_runtime is not None:
        beaker_runtime.set_description(
            f"OLMax {recipe_name} on {beaker_runtime.cluster_nickname}..."
        )

    log.info("Initializing model...")
    model = model_config.build(model_key, mesh_resource=mesh_resource)
    if show_model:
        print(model)
    num_params = count_params(model)
    num_non_embedding_prams = num_params - model.embedding.weight.size
    log.info(
        f"Built model with {num_params:,d} total parameters, "
        f"{num_non_embedding_prams:,d} non-embedding parameters"
    )

    log.info("Initializing optimizer...")
    optim, opt_state = AdamWConfig(
        lr=WarmupCosineDecaySchedule(
            warmup_steps=20,
            decay_steps=80,
            peak_value=learning_rate,
            init_value=learning_rate * 0.01,
            end_value=learning_rate * 0.01,
        ),
        no_decay_modules=["embedding.weight"],
    ).build(model)

    param_sharding = model.get_param_shardings()
    data_sharding = mesh_resource.get_data_sharding()
    opt_state_sharding = mesh_resource.get_opt_state_sharding(opt_state)

    @eqx.filter_value_and_grad
    def compute_loss(model: nn.Transformer, input_ids: Array, labels: Array):
        model = jax.lax.with_sharding_constraint(model, param_sharding)
        input_ids = jax.lax.with_sharding_constraint(input_ids, data_sharding)
        labels = jax.lax.with_sharding_constraint(labels, data_sharding)

        with jax.named_scope("compute_loss"):
            logits = model(input_ids)
            logits = jax.lax.with_sharding_constraint(logits, data_sharding)
            loss = F.cross_entropy_loss(logits, labels)

        return loss

    @eqx.filter_jit(donate="all")
    def train_step(
        model: nn.Transformer, input_ids: Array, labels: Array, opt_state: optax.OptState
    ) -> tuple[dict[str, Array], nn.Transformer, optax.OptState]:
        model = jax.lax.with_sharding_constraint(model, param_sharding)
        input_ids = jax.lax.with_sharding_constraint(input_ids, data_sharding)
        labels = jax.lax.with_sharding_constraint(labels, data_sharding)
        opt_state = jax.lax.with_sharding_constraint(opt_state, opt_state_sharding)

        step_metrics: dict[str, Array] = {}

        # Cast model to lower precision compute dtype.
        if compute_dtype != param_dtype:
            model_with_compute_dtype = cast_tree(model, compute_dtype)
        else:
            model_with_compute_dtype = model

        # Calculate loss and gradients.
        with jax.named_scope("compute_loss_and_grads"):
            loss, grads = compute_loss(model_with_compute_dtype, input_ids, labels)
            grads = jax.lax.with_sharding_constraint(grads, param_sharding)
            step_metrics["loss"] = jax.copy_to_host_async(loss)

        # Cast grads back to param dtype.
        if compute_dtype != param_dtype:
            grads = cast_tree(grads, param_dtype)
            grads = jax.lax.with_sharding_constraint(grads, param_sharding)

        # Maybe clip gradient norm.
        if max_grad_norm is not None:
            with jax.named_scope("clip_grads"):
                grads, g_norm = clip_grads_by_global_norm(grads, max_grad_norm)
                step_metrics["g_norm"] = jax.copy_to_host_async(g_norm)

        # Take optimizer step.
        with jax.named_scope("optim_step"):
            updates, opt_state = optim.update(grads, opt_state, model)  # pyright: ignore
            updates = jax.lax.with_sharding_constraint(updates, param_sharding)
            opt_state = jax.lax.with_sharding_constraint(opt_state, opt_state_sharding)

            model = eqx.apply_updates(model, updates)
            model = jax.lax.with_sharding_constraint(model, param_sharding)

        step_metrics["lr"] = jax.copy_to_host_async(
            opt_state.hyperparams["learning_rate"]  # pyright: ignore
        )

        return step_metrics, model, opt_state

    log.info("Starting training...")
    gc.collect()

    # Bookkeeping variables.
    step = 0
    all_steps_tps: list[float] = []
    running_avg_tps: deque[float] = deque()
    running_avg_tps_best: float = 0.0
    loss: float | None = None

    batches = generate_batches_of_sequential_tokens(
        data_key,
        vocab_size=vocab_size,
        sequence_length=sequence_length,
        global_batch_size_instances=global_batch_size_instances,
        total_batches=train_steps,
        mesh_resource=mesh_resource,
    )

    while True:
        # Bookkeeping.
        batch_start = time.monotonic()
        step += 1
        metrics_to_log: dict[str, float | int] = {}

        # Maybe start tracing.
        if step == 3 and trace_dir is not None:
            jax.profiler.start_trace(trace_dir, create_perfetto_trace=True)

        # Get batch.
        try:
            input_ids, labels = next(batches)
        except StopIteration:
            break

        # Do a step.
        array_metrics, model, opt_state = train_step(model, input_ids, labels, opt_state)
        for key, arr in array_metrics.items():
            value = arr.item()
            if key == "loss":
                loss = value
            metrics_to_log[key] = value

        # Maybe record memory metrics.
        if step % 5 == 0:
            peak_mib_in_use = int(bytes_to_mib(get_peak_local_device_memory_usage()))
            metrics_to_log["peak mem usage (MiB)"] = peak_mib_in_use

        # Maybe stop tracing.
        if step == 5 and trace_dir is not None:
            jax.profiler.stop_trace()

        # Record throughput.
        batch_end = time.monotonic()
        tps = batch_size_per_device / (batch_end - batch_start)
        metrics_to_log["TPS"] = int(tps)
        if step > 5:
            running_avg_tps.append(tps)
            all_steps_tps.append(tps)
        if len(running_avg_tps) > running_avg_tps_count:
            running_avg_tps.popleft()
        if len(running_avg_tps) >= running_avg_tps_count:
            avg_tps = sum(running_avg_tps) / len(running_avg_tps)
            running_avg_tps_best = max(running_avg_tps_best, avg_tps)

        # Log metrics.
        log.info(
            f"[step {step:03d}] "
            + ", ".join(
                f"{name} = {format_scalar(value)}" for name, value in metrics_to_log.items()
            ),
        )

    gc.collect()

    # Collect final metrics.
    assert loss is not None
    peak_mib_in_use = int(bytes_to_mib(get_peak_local_device_memory_usage()))
    tps_arr = jnp.array(all_steps_tps)
    tps_avg = int(tps_arr.mean().item())
    tps_std = int(tps_arr.std().item())

    log.info(
        f"Done.\n"
        f"❯ Best running avg throughput: {int(running_avg_tps_best):,d} TPS\n"
        f"❯ Actual avg throughput: {tps_avg:,d} += {2 * tps_std:,d} ({tps_avg - 2 * tps_std:,d}, {tps_avg + 2 * tps_std:,d}) TPS\n"
        f"❯ Peak mem usage: {peak_mib_in_use:,d} MiB\n"
        f"❯ Final loss: {loss:.4f}"
    )

    if beaker_runtime is not None:
        beaker_runtime.set_description(
            f"OLMax {recipe_name} on {beaker_runtime.cluster_nickname}: "
            f"loss = {loss:.4f}, "
            f"running best TPS = {int(running_avg_tps_best):,d}, "
            f"peak mem usage (MiB) = {peak_mib_in_use:,d}"
        )

    return loss, int(running_avg_tps_best), peak_mib_in_use


def main():
    prepare_cli_environment()

    beaker_runtime = BeakerRuntime.from_env()
    replica = None if beaker_runtime is None else beaker_runtime.replica

    parser = argparse.ArgumentParser("train_transformer")

    # Hyperparameters.
    parser.add_argument(
        "--recipe",
        choices=list(TransformerRecipe.get_known_choices().keys()),
        default=TransformerRecipe.get_choice_name(LlamaLike271MRecipe),
    )
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--vocab-size", type=int)
    parser.add_argument("--mesh-type", choices=["FSDP", "HSDP"], default="FSDP")
    parser.add_argument("--max-grad-norm", type=float)

    # Debugging.
    parser.add_argument("--show-model", action="store_true")
    parser.add_argument("--no-jit", action="store_true")
    parser.add_argument(
        "--trace", action=argparse.BooleanOptionalAction, default=beaker_runtime is not None
    )
    parser.add_argument("--trace-dir", type=str)

    # Performance.
    parser.add_argument("--no-remat", action="store_true")
    parser.add_argument("--all-gather-combine-threshold-mib", type=int)
    parser.add_argument("--reduce-scatter-combine-threshold-mib", type=int)
    parser.add_argument("--all-reduce-combine-threshold-mib", type=int)
    parser.add_argument(
        "--xla-flags", choices=["recommended", "system_default"], default="recommended"
    )
    parser.add_argument("--xla-mem-frac", type=float, default=0.95)

    # Attention settings.
    parser.add_argument("--attn-window-size", type=int)
    parser.add_argument("--attn", choices=["xla", "cudnn"])

    # Distributed settings.
    parser.add_argument("--nproc", type=int, default=1 if replica is None else replica.count)
    parser.add_argument("--proc-rank", type=int, default=None if replica is None else replica.rank)
    parser.add_argument(
        "--coordinator-address",
        type=str,
        default=None if replica is None else f"{replica.leader_node.hostname}:29400",
    )

    opts = parser.parse_args()

    trace_dir = opts.trace_dir
    if opts.trace and opts.trace_dir is None:
        if beaker_runtime is not None:
            trace_dir = beaker_runtime.workload.result_dataset_path
        else:
            raise ValueError("--trace-dir is required!")

    all_gather_combine_threshold_mib: float
    reduce_scatter_combine_threshold_mib: float
    all_reduce_combine_threshold_mib: float
    enable_pipelined_comms: bool = True
    enable_nccl_user_buffers: bool = False
    if opts.recipe == TransformerRecipe.get_choice_name(LlamaLike271MRecipe):
        all_gather_combine_threshold_mib = opts.all_gather_combine_threshold_mib or 256
        reduce_scatter_combine_threshold_mib = opts.reduce_scatter_combine_threshold_mib or 128
        all_reduce_combine_threshold_mib = opts.all_reduce_combine_threshold_mib or 256
    elif opts.recipe == TransformerRecipe.get_choice_name(LlamaLike7BRecipe):
        all_gather_combine_threshold_mib = opts.all_gather_combine_threshold_mib or 1024
        reduce_scatter_combine_threshold_mib = opts.reduce_scatter_combine_threshold_mib or 128
        all_reduce_combine_threshold_mib = opts.all_reduce_combine_threshold_mib or 1024
    elif opts.recipe == TransformerRecipe.get_choice_name(Gemma2Like27BRecipe):
        all_gather_combine_threshold_mib = opts.all_gather_combine_threshold_mib or 256
        reduce_scatter_combine_threshold_mib = opts.reduce_scatter_combine_threshold_mib or 128
        all_reduce_combine_threshold_mib = opts.all_reduce_combine_threshold_mib or 256
        enable_pipelined_comms = False
    elif opts.recipe == TransformerRecipe.get_choice_name(Gemma3Like27BRecipe):
        all_gather_combine_threshold_mib = opts.all_gather_combine_threshold_mib or 256
        reduce_scatter_combine_threshold_mib = opts.reduce_scatter_combine_threshold_mib or 128
        all_reduce_combine_threshold_mib = opts.all_reduce_combine_threshold_mib or 256
        enable_pipelined_comms = False
        enable_nccl_user_buffers = True
    else:
        raise ValueError(opts.recipe)  # need to tune for model size

    if beaker_runtime is not None:
        log.info(f"Running in Beaker on node '{beaker_runtime.node.hostname}'")

    prepare_training_environment(
        disable_jit=opts.no_jit,
        disable_remat=opts.no_remat,
        xla_mem_frac=opts.xla_mem_frac,
        xla_flags=opts.xla_flags,
        all_gather_combine_threshold_mib=all_gather_combine_threshold_mib,
        reduce_scatter_combine_threshold_mib=reduce_scatter_combine_threshold_mib,
        all_reduce_combine_threshold_mib=all_reduce_combine_threshold_mib,
        enable_pipelined_comms=enable_pipelined_comms,
        enable_nccl_user_buffers=enable_nccl_user_buffers,
        gpu_architecture=None if beaker_runtime is None else beaker_runtime.node.gpu_architecture,
    )

    if opts.nproc > 1:
        if opts.coordinator_address is None:
            raise ValueError("--coordinator-address is required for distributed training")
        if opts.proc_rank is None:
            raise ValueError("--proc-rank is required for distributed training")

        log.info("Initializing distributed backend...")
        dist.init_distributed(
            coordinator_address=opts.coordinator_address,
            num_processes=opts.nproc,
            process_id=opts.proc_rank,
        )
        log.info(
            f"Distributed backend initialized with {dist.get_global_device_count():,d} total devices "
            f"across {dist.get_process_world_size():,d} processes."
        )

    try:
        train(
            opts.recipe,
            beaker_runtime=beaker_runtime,
            instances_per_device=opts.batch_size,
            vocab_size=opts.vocab_size,
            attn_window_size=opts.attn_window_size,
            attn_implementation=opts.attn,
            show_model=opts.show_model,
            mesh_type=opts.mesh_type,
            trace_dir=trace_dir,
            max_grad_norm=opts.max_grad_norm,
        )
    finally:
        if dist.is_distributed():
            dist.teardown_distributed()


if __name__ == "__main__":
    main()
