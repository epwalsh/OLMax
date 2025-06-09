from __future__ import annotations

import dataclasses
import gc
import logging
import sys
import time
from collections import deque
from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

import olmax
import olmax.distributed as dist
import olmax.nn as nn
import olmax.nn.functional as F
import olmax.nn.transformer.recipes as recipes
from olmax.config import parse_config_from_args
from olmax.launch.beaker import BeakerRuntime
from olmax.types import *

log = logging.getLogger("main")
beaker_runtime: BeakerRuntime | None = None


@dataclass
class IntegrationTestConfig:
    recipe: nn.transformer.recipes.TransformerRecipe

    env: olmax.EnvConfig = dataclasses.field(
        default_factory=lambda: olmax.EnvConfig.recommended()
        if beaker_runtime is None
        else beaker_runtime.get_env_config()
    )
    mesh: dist.MeshResource = dataclasses.field(default_factory=dist.MeshResource.FSDP)
    distributed: dist.DistConfig | None = dataclasses.field(
        default_factory=lambda: None if beaker_runtime is None else beaker_runtime.get_dist_config()
    )

    steps: int = 100
    batch_size_per_device: int | None = None
    max_grad_norm: float | None = None
    param_dtype: DTypeLike = "float32"
    compute_dtype: DTypeLike = "bfloat16"

    trace_dir: str | None = dataclasses.field(
        default_factory=lambda: None
        if beaker_runtime is None
        else beaker_runtime.workload.result_dataset_path
    )
    show_config: bool = True
    show_model: bool = False
    dry_run: bool = False


def train(
    config: IntegrationTestConfig,
    running_avg_tps_count: int = 10,
) -> tuple[float, int, int]:
    recipe_name = recipes.TransformerRecipe.get_registered_name(config.recipe.__class__)
    batch_size_per_device = config.batch_size_per_device or config.recipe.get_mbz_per_device(
        None if beaker_runtime is None else beaker_runtime.node.gpu_type
    )
    instances_per_device = batch_size_per_device // config.recipe.sequence_length
    global_batch_size = batch_size_per_device * dist.get_global_device_count()
    global_batch_size_instances = instances_per_device * dist.get_global_device_count()

    log.info(
        f"Using global batch size of {global_batch_size:,d} tokens, "
        f"which is {global_batch_size_instances:,d} instances of length {config.recipe.sequence_length:,d}."
    )
    log.info(
        f"Using per-device batch size of {batch_size_per_device:,d} tokens, "
        f"which is {instances_per_device:,d} instances of length {config.recipe.sequence_length:,d}."
    )

    key = jax.random.PRNGKey(0)
    model_key, data_key = jax.random.split(key)

    log.info(f"Using mesh with axes {config.mesh.get_mesh_axes_repr()}")

    if beaker_runtime is not None and beaker_runtime.is_experiment:
        beaker_runtime.set_description(
            f"OLMax {recipe_name} on {beaker_runtime.cluster_nickname}..."
        )

    log.info("Initializing model...")
    model_config = config.recipe.build_config(param_dtype=config.param_dtype)
    model = model_config.build(model_key, mesh_resource=config.mesh)
    if config.show_model:
        log.info(model)

    num_params = olmax.jax_utils.count_params(model)
    num_non_embedding_prams = num_params - model.embedding.weight.size
    log.info(
        f"Built model with {num_params:,d} total parameters, "
        f"{num_non_embedding_prams:,d} non-embedding parameters"
    )

    log.info("Initializing optimizer...")
    optim, opt_state = olmax.optim.AdamWConfig(
        lr=olmax.optim.WarmupCosineDecaySchedule(
            warmup_steps=20,
            decay_steps=80,
            peak_value=config.recipe.learning_rate,
            init_value=config.recipe.learning_rate * 0.01,
            end_value=config.recipe.learning_rate * 0.01,
        ),
        no_decay_modules=["embedding.weight"],
    ).build(model)

    param_sharding = model.get_param_shardings()
    data_sharding = config.mesh.get_data_sharding()
    opt_state_sharding = config.mesh.get_opt_state_sharding(opt_state)

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
        if config.compute_dtype != config.param_dtype:
            model_with_compute_dtype = olmax.jax_utils.cast_tree(model, config.compute_dtype)
        else:
            model_with_compute_dtype = model

        # Calculate loss and gradients.
        with jax.named_scope("compute_loss_and_grads"):
            loss, grads = compute_loss(model_with_compute_dtype, input_ids, labels)
            grads = jax.lax.with_sharding_constraint(grads, param_sharding)
            step_metrics["loss"] = jax.copy_to_host_async(loss)

        # Cast grads back to param dtype.
        if config.compute_dtype != config.param_dtype:
            grads = olmax.jax_utils.cast_tree(grads, config.param_dtype)
            grads = jax.lax.with_sharding_constraint(grads, param_sharding)

        # Maybe clip gradient norm.
        if config.max_grad_norm is not None:
            with jax.named_scope("clip_grads"):
                grads, g_norm = olmax.optim.clip_grads_by_global_norm(grads, config.max_grad_norm)
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
    gc.disable()
    gc.collect()

    # Bookkeeping variables.
    step = 0
    all_steps_tps: list[float] = []
    running_avg_tps: deque[float] = deque()
    running_avg_tps_best: float = 0.0
    loss: float | None = None

    batches = olmax.data.utils.generate_batches_of_sequential_tokens(
        data_key,
        vocab_size=config.recipe.vocab_size,
        sequence_length=config.recipe.sequence_length,
        global_batch_size_instances=global_batch_size_instances,
        total_batches=config.steps,
        mesh_resource=config.mesh,
    )

    while True:
        # Bookkeeping.
        batch_start = time.perf_counter()
        step += 1
        metrics_to_log: dict[str, float | int] = {}

        # Maybe start tracing.
        if step == 3 and config.trace_dir is not None:
            jax.profiler.start_trace(config.trace_dir, create_perfetto_trace=True)

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
            peak_mib_in_use = int(
                olmax.utils.bytes_to_mib(olmax.jax_utils.get_peak_local_device_memory_usage())
            )
            metrics_to_log["peak mem usage (MiB)"] = peak_mib_in_use

        # Maybe stop tracing.
        if step == 5 and config.trace_dir is not None:
            jax.profiler.stop_trace()

        # Record throughput.
        batch_end = time.perf_counter()
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
                f"{name} = {olmax.utils.format_scalar(value)}"
                for name, value in metrics_to_log.items()
            ),
        )

    gc.collect()

    # Collect final metrics.
    assert loss is not None
    peak_mib_in_use = int(
        olmax.utils.bytes_to_mib(olmax.jax_utils.get_peak_local_device_memory_usage())
    )
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

    if beaker_runtime is not None and beaker_runtime.is_experiment:
        beaker_runtime.set_description(
            f"OLMax {recipe_name} on {beaker_runtime.cluster_nickname}: "
            f"loss = {loss:.4f}, "
            f"running best TPS = {int(running_avg_tps_best):,d}, "
            f"peak mem usage (MiB) = {peak_mib_in_use:,d}"
        )

    gc.enable()
    return loss, int(running_avg_tps_best), peak_mib_in_use


def main():
    recipe_names = recipes.TransformerRecipe.get_registered_names()
    if len(sys.argv) < 2 or (recipe_name := sys.argv[1]) not in recipe_names:
        print(
            f"usage: {sys.argv[0]} RECIPE_NAME [OVERRIDES...]\n\n"
            f"Where RECIPE_NAME should be one of {recipe_names}",
            file=sys.stderr,
        )
        sys.exit(1)

    config = IntegrationTestConfig(
        recipe=recipes.TransformerRecipe.get_registered_class(recipe_name)()
    )
    config.recipe.set_env_defaults(config.env)
    config = parse_config_from_args(IntegrationTestConfig, config, args=sys.argv[2:])

    if config.show_config or config.dry_run:
        log.info(config)
    if config.dry_run:
        return

    olmax.prepare_training_environment(
        jax_config=config.env.jax,
        xla_config=config.env.xla,
        nccl_config=config.env.nccl,
        cuda_config=config.env.cuda,
    )

    if config.distributed is not None:
        log.info("Initializing distributed backend...")
        dist.init_distributed(config.distributed)
        log.info(
            f"Distributed backend initialized with {dist.get_global_device_count():,d} total devices "
            f"across {dist.get_process_world_size():,d} processes."
        )

    try:
        train(config)
    finally:
        if dist.is_distributed():
            dist.teardown_distributed()


if __name__ == "__main__":
    olmax.prepare_cli_environment()
    beaker_runtime = BeakerRuntime.from_env()
    main()
