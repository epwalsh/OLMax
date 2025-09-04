import argparse
import dataclasses
import logging
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

import jax

import olmax
import olmax.distributed as dist
import olmax.nn as nn
import olmax.nn.functional as F
from olmax.config import parse_config_from_args
from olmax.launch.beaker import BeakerRuntime
from olmax.nn.transformer import TransformerConfig, TransformerRecipeType
from olmax.train import Trainer
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

    steps: int = 100
    num_microbatches: int = 1

    param_dtype: DTypeLike = "float32"
    compute_dtype: DTypeLike = "bfloat16"

    checkpoint_interval: int | None = None
    load_path: PathOrStr | None = None

    mesh: dist.MeshResource = dataclasses.field(default_factory=dist.MeshResource.FSDP)
    distributed: dist.DistConfig | None = dataclasses.field(
        default_factory=lambda: None if beaker_runtime is None else beaker_runtime.get_dist_config()
    )

    dir: str | None = None


class DataLoader(olmax.data.DataLoader):
    def __init__(
        self,
        data_key: PRNGKeyArray,
        *,
        vocab_size: int,
        sequence_length: int,
        global_batch_size_instances: int,
        total_batches: int,
        mesh_resource: dist.MeshResource,
        num_microbatches: int,
    ):
        self.data_key = data_key
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.global_batch_size_instances = global_batch_size_instances
        self.total_batches = total_batches
        self.mesh_resource = mesh_resource
        self.num_microbatches = num_microbatches
        self.batches_processed = 0
        self.epoch = 0

    def __len__(self):
        return self.total_batches

    def __iter__(self):
        for batch in olmax.data.utils.generate_batches_of_sequential_tokens(
            jax.random.fold_in(self.data_key, self.epoch),
            vocab_size=self.vocab_size,
            sequence_length=self.sequence_length,
            global_batch_size_instances=self.global_batch_size_instances,
            total_batches=self.total_batches,
            mesh_resource=self.mesh_resource,
            num_microbatches=self.num_microbatches,
            start_batch=self.batches_processed,
        ):
            self.batches_processed += 1
            yield batch

        self.batches_processed = 0
        self.epoch += 1

    def get_state(self) -> tuple[PRNGKeyArray, int, int]:
        return (self.data_key, self.batches_processed, self.epoch)

    def load_state(self, state: tuple[PRNGKeyArray, int, int]):
        data_key, batches_processed, epoch = state
        self.data_key = data_key
        self.batches_processed = batches_processed
        self.epoch = epoch


def train(
    recipe_type: TransformerRecipeType,
    config: IntegrationTestConfig,
    show_model: bool = False,
):
    dir: Path
    trace_download_command: str | None = None
    if config.dir is not None:
        dir = Path(config.dir)
    elif (
        beaker_runtime is not None
        and (result_path := beaker_runtime.workload.result_dataset_path) is not None
    ):
        dir = Path(result_path)
        trace_download_command = f"beaker dataset fetch {beaker_runtime.workload.result_dataset_id} --output=results/ --prefix=profiler"
    else:
        dir = Path("/tmp/olmax/train")
    log.info(f"Saving results to '{dir}'")

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
        f"Using per-device batch size of {batch_size_per_device:,d} tokens, "
        f"which is {instances_per_device:,d} instances of length {config.sequence_length:,d}."
    )
    log.info(
        f"Using per-device micro-batch size of {config.device_microbatch_size:,d} tokens, "
        f"which is {instances_per_device//config.num_microbatches:,d} instances of length {config.sequence_length:,d}."
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

    param_sharding = model.get_param_shardings()

    @jax.named_scope("compute_loss")
    def loss_fun(model: nn.Transformer, batch: tuple[Array, Array]):
        input_ids, labels = batch

        # Enforce sharding constraints.
        model = jax.lax.with_sharding_constraint(model, param_sharding)
        input_ids = config.mesh.with_data_sharding_constraint(input_ids, sequence_dim=1)
        labels = config.mesh.with_data_sharding_constraint(labels, sequence_dim=1)

        # Get predicted logits.
        logits = model(input_ids)
        logits = config.mesh.with_data_sharding_constraint(logits, sequence_dim=1)

        # Compute and reduce loss.
        return F.cross_entropy_loss(logits, labels)

    trainer = Trainer(
        work_dir=dir,
        save_folder=dir,
        checkpoint_interval=config.checkpoint_interval,
        optim=config.optim,
        loss_fun=loss_fun,
        mesh=config.mesh,
        grad_dtype=config.param_dtype,
        compute_dtype=config.compute_dtype,
        max_duration=Duration.steps(config.steps),
        global_tokens_per_batch=global_batch_size,
    ).add_callback("profiler", olmax.train.callbacks.ProfilerCallback())

    data_loader = DataLoader(
        data_key,
        vocab_size=config.model.vocab_size,
        sequence_length=config.sequence_length,
        global_batch_size_instances=global_batch_size_instances,
        total_batches=config.steps,
        mesh_resource=config.mesh,
        num_microbatches=config.num_microbatches,
    )

    trainer.fit(model, data_loader, load_path=config.load_path)

    if trace_download_command is not None:
        log.info(f"To download the profiler trace, run:\n❯ {trace_download_command}")


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
    parser.add_argument(
        "--local-device-count",
        type=int,
        default=jax.local_device_count()
        if beaker_runtime is None
        else beaker_runtime.local_device_count,
        help="""The number of local devices.""",
    )
    parser.add_argument(
        "--global-device-count",
        type=int,
        default=jax.local_device_count()
        if beaker_runtime is None
        else beaker_runtime.global_device_count,
        help="""The number of global devices.""",
    )
    parser.add_argument(
        "--mesh-type",
        choices=["FSDP", "HSDP", "HSDP_with_CP"],
        default="FSDP",
        help="""The type of distributed mesh to use.""",
    )
    # Some configuration depend on others, so it's better to parse those base fields here instead of
    # as overrides.
    parser.add_argument(
        "--steps", type=int, default=100, help="""The number of steps to train for."""
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

    mesh_resource: dist.MeshResource
    if opts.mesh_type == "FSDP":
        mesh_resource = dist.MeshResource.FSDP(global_device_count=opts.global_device_count)
    elif opts.mesh_type == "HSDP":
        mesh_resource = dist.MeshResource.HSDP(
            global_device_count=opts.global_device_count, local_device_count=opts.local_device_count
        )
    elif opts.mesh_type == "HSDP_with_CP":
        mesh_resource = dist.MeshResource.HSDP_with_CP(
            global_device_count=opts.global_device_count, local_device_count=opts.local_device_count
        )
    else:
        raise ValueError(f"Unsupported mesh type '{opts.mesh_type}'")

    config = IntegrationTestConfig(
        model=recipe.model,
        optim=olmax.optim.AdamWConfig(
            lr=olmax.optim.WarmupCosineDecaySchedule(
                warmup_steps=20,
                decay_steps=opts.steps - 20,
                peak_value=learning_rate,
                init_value=learning_rate * 0.01,
                end_value=learning_rate * 0.01,
            ),
            no_decay_modules=["embedding.weight"],
        ),
        sequence_length=recipe.sequence_length,
        device_microbatch_size=recipe.device_microbatch_size,
        mesh=mesh_resource,
        env=recipe.env,
        steps=opts.steps,
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
