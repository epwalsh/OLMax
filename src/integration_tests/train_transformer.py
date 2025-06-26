import argparse
import dataclasses
import gc
import logging
import sys
import textwrap
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
from olmax.config import parse_config_from_args
from olmax.launch.beaker import BeakerRuntime
from olmax.nn.transformer import TransformerConfig, TransformerRecipeType
from olmax.types import *

log = logging.getLogger("main")
beaker_runtime: BeakerRuntime | None = None


@dataclass
class IntegrationTestConfig:
    model: TransformerConfig
    optim: olmax.optim.OptimConfig
    sequence_length: int
    device_microbatch_size: int
    env: olmax.EnvConfig

    mesh: dist.MeshResource = dataclasses.field(default_factory=dist.MeshResource.FSDP)
    distributed: dist.DistConfig | None = dataclasses.field(
        default_factory=lambda: None if beaker_runtime is None else beaker_runtime.get_dist_config()
    )

    steps: int = 100
    num_microbatches: int = 2
    max_grad_norm: float | None = None

    trace_dir: str | None = dataclasses.field(
        default_factory=lambda: None
        if beaker_runtime is None
        else beaker_runtime.workload.result_dataset_path
    )


def train(
    recipe_type: TransformerRecipeType,
    config: IntegrationTestConfig,
    running_avg_tps_count: int = 10,
    param_dtype: str = "float32",
    compute_dtype: str = "bfloat16",
    show_model: bool = False,
) -> tuple[float, int, int]:
    recipe_name = recipe_type.name
    batch_size_per_device = config.device_microbatch_size * config.num_microbatches
    instances_per_device = batch_size_per_device // config.sequence_length
    global_batch_size = batch_size_per_device * dist.get_global_device_count()
    global_batch_size_instances = instances_per_device * dist.get_global_device_count()

    log.info(
        f"Using global batch size of {global_batch_size:,d} tokens, "
        f"which is {global_batch_size_instances:,d} instances of length {config.sequence_length:,d}."
    )
    log.info(
        f"Using per-device batch size of {batch_size_per_device:,d} tokens over {config.num_microbatches} microbatches, "
        f"which is {instances_per_device:,d} instances of length {config.sequence_length:,d}."
    )

    key = jax.random.PRNGKey(0)
    model_key, data_key = jax.random.split(key)

    log.info(f"Using mesh with axes {config.mesh.get_mesh_axes_repr()}")

    if beaker_runtime is not None and beaker_runtime.is_experiment:
        beaker_runtime.set_description(
            f"OLMax {recipe_name} on {beaker_runtime.cluster_nickname}..."
        )

    dist.barrier("pre-init-model")
    log.info("Initializing model...")
    model = config.model.build(
        model_key,
        mesh_resource=config.mesh,
    )
    dist.barrier("post-init-model")
    if show_model:
        log.info(model)

    num_params = olmax.jax_utils.count_params(model)
    num_non_embedding_prams = num_params - model.embedding.weight.size
    log.info(
        f"Built model with {num_params:,d} total parameters, "
        f"{num_non_embedding_prams:,d} non-embedding parameters"
    )

    log.info("Initializing optimizer...")
    optim, opt_state = config.optim.build(model)

    param_sharding = model.get_param_shardings()
    data_sharding = config.mesh.get_data_sharding()
    batch_sharding = {"input_ids": data_sharding, "labels": data_sharding}
    opt_state_sharding = config.mesh.get_opt_state_sharding(opt_state)

    @jax.named_scope("compute_loss")
    def compute_loss(model: nn.Transformer, batch: dict[str, Array]):
        input_ids, labels = batch["input_ids"], batch["labels"]
        model = jax.lax.with_sharding_constraint(model, param_sharding)
        input_ids = jax.lax.with_sharding_constraint(input_ids, data_sharding)
        labels = jax.lax.with_sharding_constraint(labels, data_sharding)

        logits = model(input_ids)
        logits = jax.lax.with_sharding_constraint(logits, data_sharding)
        return F.cross_entropy_loss(logits, labels)

    @eqx.filter_jit(donate="all")
    def train_step(
        model: nn.Transformer, batch: dict[str, Array], opt_state: optax.OptState
    ) -> tuple[dict[str, Array], nn.Transformer, optax.OptState]:
        model = jax.lax.with_sharding_constraint(model, param_sharding)
        batch = jax.lax.with_sharding_constraint(batch, batch_sharding)
        opt_state = jax.lax.with_sharding_constraint(opt_state, opt_state_sharding)

        step_metrics: dict[str, Array] = {}

        # Cast model to lower precision compute dtype.
        if compute_dtype != param_dtype:
            model_with_compute_dtype = olmax.jax_utils.cast_tree(model, compute_dtype)
        else:
            model_with_compute_dtype = model

        # Calculate loss and gradients.
        loss, grads = olmax.train.utils.microbatched(
            compute_loss,
            model_with_compute_dtype,
            batch,
            num_microbatches=config.num_microbatches,
            accum_sharding=(None, param_sharding),
            accum_dtype=param_dtype,
        )
        step_metrics["loss"] = jax.copy_to_host_async(loss)

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

    dist.barrier("pre-train-loop")
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
        vocab_size=config.model.vocab_size,
        sequence_length=config.sequence_length,
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
        with jax.profiler.StepTraceAnnotation("train_step", step_num=step):
            array_metrics, model, opt_state = train_step(
                model, dict(input_ids=input_ids, labels=labels), opt_state
            )
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


def _parse_args():
    parser = argparse.ArgumentParser(
        prog=sys.argv[0],
        usage=f"{sys.argv[0]} --recipe=RECIPE [OPTIONS...] [CONFIG_OVERRIDES...]",
        description=textwrap.dedent(
            """
            Run a short transformer training integration test.

            In addition to the options listed below, you can override any field in the config using dot
            notation for nested fields. Values are parsed as YAML.
            """
        ),
        epilog=textwrap.dedent(
            f"""
            examples:
              Do a dry run to check the config before actually running anything:
              ❯ python {sys.argv[0]} --recipe=llama_like_271M --dry-run

              Run the integration test while overriding some configuration options:
              ❯ python {sys.argv[0]} --recipe=llama_like_271M \\
                  --model.scan_layers=true \\
                  --model.layer_ac_policy='{{type: nothing_saveable, prevent_cse: false}}'
            """
        ),
        formatter_class=type(  # type: ignore[arg-type]
            "CustomFormatter",
            (
                argparse.ArgumentDefaultsHelpFormatter,
                argparse.RawDescriptionHelpFormatter,
            ),
            {},
        ),
    )
    parser.add_argument(
        "--recipe",
        choices=[r.name for r in TransformerRecipeType],
        required=True,
        help="""The name of the recipe to run.""",
    )
    parser.add_argument(
        "--device-type",
        choices=[d.name for d in DeviceType],
        required=beaker_runtime is None,
        default=None if beaker_runtime is None else beaker_runtime.node.device_type,
        help="""The device type. When running on Beaker this doesn't need to specified manually as
        it will be determined automatically.""",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="""Print the config and exit. This can be useful for seeing which fields can be overridden.""",
    )
    parser.add_argument(
        "--show-model",
        action="store_true",
        help="""Print out the model structure after initialization.""",
    )

    opts, overrides = parser.parse_known_args()
    return opts, overrides


def main():
    opts, overrides = _parse_args()
    recipe_type = TransformerRecipeType(opts.recipe)
    device_type = DeviceType(opts.device_type)
    env = (
        olmax.EnvConfig.recommended() if beaker_runtime is None else beaker_runtime.get_env_config()
    )
    recipe = recipe_type.build_recipe(env, device_type)

    learning_rate: float
    if "32B" in opts.recipe or "27B" in opts.recipe:
        learning_rate = 1e-5
    elif "7B" in opts.recipe or "8B" in opts.recipe:
        learning_rate = 1e-4
    else:
        learning_rate = 1e-3

    config = IntegrationTestConfig(
        model=recipe.model,
        optim=olmax.optim.AdamWConfig(
            lr=olmax.optim.WarmupCosineDecaySchedule(
                warmup_steps=20,
                decay_steps=80,
                peak_value=learning_rate,
                init_value=learning_rate * 0.01,
                end_value=learning_rate * 0.01,
            ),
            no_decay_modules=["embedding.weight"],
        ),
        sequence_length=recipe.sequence_length,
        device_microbatch_size=recipe.device_microbatch_size,
        env=recipe.env,
    )
    config = parse_config_from_args(config, args=overrides)

    log.info(config)
    if opts.dry_run:
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
        train(recipe_type, config, show_model=opts.show_model)
    finally:
        if dist.is_distributed():
            dist.teardown_distributed()


if __name__ == "__main__":
    olmax.prepare_cli_environment()
    beaker_runtime = BeakerRuntime.from_env()
    main()
