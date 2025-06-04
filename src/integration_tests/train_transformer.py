from __future__ import annotations

import argparse
import gc
import logging
import time
from collections import deque
from typing import Literal

import equinox as eqx
import jax
import optax

import olmax.distributed as dist
import olmax.nn as nn
import olmax.nn.functional as F
import olmax.nn.transformer.recipes as recipes
from olmax.data.utils import generate_batches_of_sequential_tokens
from olmax.jax_utils import cast_tree, count_params, get_peak_local_device_memory_usage
from olmax.launch.beaker import BeakerRuntime
from olmax.train import prepare_training_environment
from olmax.types import Array, DTypeLike
from olmax.utils import bytes_to_mib, prepare_cli_environment

log = logging.getLogger("main")


def train(
    recipe: str,
    beaker_runtime: BeakerRuntime | None = None,
    sequence_length: int | None = None,
    instances_per_device: int | None = None,
    vocab_size: int = 50_304,
    param_dtype: DTypeLike = float,
    compute_dtype: DTypeLike = jax.dtypes.bfloat16,
    learning_rate: float | None = None,
    train_steps: int = 100,
    attn_window_size: int | tuple[int, int] | None = None,
    attn_implementation: Literal["xla", "cudnn"] | None = None,
    show_model: bool = False,
    running_avg_tps_count: int = 10,
    mesh_type: Literal["FSDP", "HSDP"] = "FSDP",
) -> tuple[float, int, int]:
    gpu_architecture = None if beaker_runtime is None else beaker_runtime.node.gpu_architecture

    if recipe == "271M":
        model_config = recipes.llama_like_271M(
            vocab_size,
            param_dtype=param_dtype,
            attn_window_size=attn_window_size,
            attn_implementation=attn_implementation,
        )
        if sequence_length is None:
            sequence_length = 1024
        if instances_per_device is None:
            instances_per_device = 16
            if gpu_architecture == "blackwell":
                instances_per_device *= 4
        if learning_rate is None:
            learning_rate = 1e-3
    elif recipe == "7B":
        model_config = recipes.llama_like_7B(
            vocab_size,
            param_dtype=param_dtype,
            attn_window_size=attn_window_size,
            attn_implementation=attn_implementation,
        )
        if sequence_length is None:
            sequence_length = 4096
        if instances_per_device is None:
            instances_per_device = 2
            if gpu_architecture == "blackwell":
                instances_per_device *= 4
        if learning_rate is None:
            learning_rate = 1e-4
    elif recipe == "gemma2_27B":
        model_config = recipes.gemma2_like_27B(
            vocab_size,
            param_dtype=param_dtype,
            attn_window_size=attn_window_size,
            attn_implementation=attn_implementation,
        )
        if sequence_length is None:
            sequence_length = 4096
        if instances_per_device is None:
            instances_per_device = 1
            #  if gpu_architecture == "blackwell":
            #      instances_per_device *= 4
        if learning_rate is None:
            learning_rate = 1e-5
    else:
        raise ValueError(recipe)

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
    optim = optax.adamw(learning_rate)
    opt_state = optim.init(model)  # pyright: ignore

    param_sharding = model.get_param_shardings()
    data_sharding = mesh_resource.get_data_sharding()
    opt_state_sharding = mesh_resource.get_opt_state_sharding(opt_state)

    @eqx.filter_value_and_grad
    def compute_loss(model: nn.Transformer, input_ids: Array, labels: Array):
        model = jax.lax.with_sharding_constraint(model, param_sharding)
        input_ids = jax.lax.with_sharding_constraint(input_ids, data_sharding)
        labels = jax.lax.with_sharding_constraint(labels, data_sharding)

        logits = model(input_ids)
        logits = jax.lax.with_sharding_constraint(logits, data_sharding)

        return F.cross_entropy_loss(logits, labels)

    @eqx.filter_jit(donate="all")
    def train_step(
        model: nn.Transformer, input_ids: Array, labels: Array, opt_state: optax.OptState
    ) -> tuple[Array, nn.Transformer, optax.OptState]:
        model = jax.lax.with_sharding_constraint(model, param_sharding)
        input_ids = jax.lax.with_sharding_constraint(input_ids, data_sharding)
        labels = jax.lax.with_sharding_constraint(labels, data_sharding)
        opt_state = jax.lax.with_sharding_constraint(opt_state, opt_state_sharding)

        # Cast model to lower precision compute dtype.
        if compute_dtype != param_dtype:
            model_with_compute_dtype = cast_tree(model, compute_dtype)
        else:
            model_with_compute_dtype = model

        # Calculate loss and gradients.
        loss, grads = compute_loss(model_with_compute_dtype, input_ids, labels)
        grads = jax.lax.with_sharding_constraint(grads, param_sharding)

        # Cast grads back to param dtype.
        if compute_dtype != param_dtype:
            grads = cast_tree(grads, param_dtype)
            grads = jax.lax.with_sharding_constraint(grads, param_sharding)

        # Take optimizer step.
        updates, opt_state = optim.update(grads, opt_state, model)  # pyright: ignore
        updates = jax.lax.with_sharding_constraint(updates, param_sharding)
        opt_state = jax.lax.with_sharding_constraint(opt_state, opt_state_sharding)

        model = eqx.apply_updates(model, updates)
        model = jax.lax.with_sharding_constraint(model, param_sharding)

        return loss, model, opt_state

    log.info("Starting training...")
    gc.collect()
    batch_start = time.monotonic()
    running_avg_tps: deque[float] = deque()
    running_avg_tps_best: float = 0.0
    loss: Array | None = None
    for step, (input_ids, labels) in enumerate(
        generate_batches_of_sequential_tokens(
            data_key,
            vocab_size=vocab_size,
            sequence_length=sequence_length,
            global_batch_size_instances=global_batch_size_instances,
            total_batches=train_steps,
            mesh_resource=mesh_resource,
        )
    ):
        # Do a step.
        loss, model, opt_state = train_step(model, input_ids, labels, opt_state)

        # Log progress.
        metrics: dict[str, str] = {}

        if (step + 1) % 5 == 0:
            peak_mib_in_use = int(bytes_to_mib(get_peak_local_device_memory_usage()))
            metrics["peak mem usage"] = f"{peak_mib_in_use:,d} MiB"
            metrics["loss"] = f"{loss.item():.4f}"

        batch_end = time.monotonic()
        tps = batch_size_per_device / (batch_end - batch_start)
        metrics["TPS"] = f"{int(tps):,d}"

        if step > 2:
            running_avg_tps.append(tps)
        if len(running_avg_tps) > running_avg_tps_count:
            running_avg_tps.popleft()
        if len(running_avg_tps) >= running_avg_tps_count:
            avg_tps = sum(running_avg_tps) / len(running_avg_tps)
            running_avg_tps_best = max(running_avg_tps_best, avg_tps)
        log.info(
            f"[step {step + 1:03d}] "
            + ", ".join(f"{name} = {value}" for name, value in metrics.items()),
        )
        batch_start = batch_end

    assert loss is not None
    final_loss = loss.item()
    peak_mib_in_use = int(bytes_to_mib(get_peak_local_device_memory_usage()))
    log.info(
        f"Done.\n"
        f"❯ Best throughput = {int(running_avg_tps_best):,d} TPS\n"
        f"❯ Peak mem usage = {peak_mib_in_use:,d} MiB\n"
        f"❯ Final loss = {final_loss:.4f}"
    )
    return final_loss, int(running_avg_tps_best), peak_mib_in_use


def main():
    prepare_cli_environment()

    beaker_runtime = BeakerRuntime.from_env()
    replica = None if beaker_runtime is None else beaker_runtime.replica

    parser = argparse.ArgumentParser("train_transformer")

    # Hyperparameters.
    parser.add_argument("--recipe", choices=["271M", "7B", "gemma2_27B"], default="271M")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--vocab-size", type=int, default=50_304)
    parser.add_argument("--mesh-type", choices=["FSDP", "HSDP"], default="FSDP")

    # Debugging.
    parser.add_argument("--show-model", action="store_true")
    parser.add_argument("--no-jit", action="store_true")

    # Performance.
    parser.add_argument("--no-remat", action="store_true")
    parser.add_argument("--reduce-scatter-combine-threshold-mib", type=int)
    parser.add_argument(
        "--xla-flags", choices=["recommended", "system_default"], default="recommended"
    )

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

    if beaker_runtime is not None:
        log.info(f"Running in Beaker on node '{beaker_runtime.node.hostname}'")
        beaker_runtime.set_description(
            f"OLMax {opts.recipe} on {beaker_runtime.cluster_nickname}..."
        )

    all_gather_combine_threshold_mib: float
    reduce_scatter_combine_threshold_mib: float
    all_reduce_combine_threshold_mib: float
    enabled_pipelined_comms: bool = True
    if opts.recipe == "271M":
        all_gather_combine_threshold_mib = 256
        reduce_scatter_combine_threshold_mib = opts.reduce_scatter_combine_threshold_mib or 128
        all_reduce_combine_threshold_mib = 256
    elif opts.recipe == "7B":
        all_gather_combine_threshold_mib = 1024
        reduce_scatter_combine_threshold_mib = opts.reduce_scatter_combine_threshold_mib or 128
        all_reduce_combine_threshold_mib = 1024
    elif opts.recipe == "gemma2_27B":
        all_gather_combine_threshold_mib = 256
        reduce_scatter_combine_threshold_mib = opts.reduce_scatter_combine_threshold_mib or 128
        all_reduce_combine_threshold_mib = 256
        enabled_pipelined_comms = False
    else:
        raise ValueError(opts.recipe)  # need to tune for model size

    prepare_training_environment(
        disable_jit=opts.no_jit,
        disable_remat=opts.no_remat,
        xla_flags=opts.xla_flags,
        all_gather_combine_threshold_mib=all_gather_combine_threshold_mib,
        reduce_scatter_combine_threshold_mib=reduce_scatter_combine_threshold_mib,
        all_reduce_combine_threshold_mib=all_reduce_combine_threshold_mib,
        enabled_pipelined_comms=enabled_pipelined_comms,
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
        final_loss, final_tps, peak_mem = train(
            opts.recipe,
            beaker_runtime=beaker_runtime,
            instances_per_device=opts.batch_size,
            vocab_size=opts.vocab_size,
            attn_window_size=opts.attn_window_size,
            attn_implementation=opts.attn,
            show_model=opts.show_model,
            mesh_type=opts.mesh_type,
        )
        if beaker_runtime is not None:
            beaker_runtime.set_description(
                f"OLMax {opts.recipe} on {beaker_runtime.cluster_nickname}: "
                f"loss = {final_loss:.4f}, TPS = {final_tps:,d}, mem usage (MiB) = {peak_mem:,d}"
            )
    finally:
        if dist.is_distributed():
            dist.teardown_distributed()


if __name__ == "__main__":
    main()
