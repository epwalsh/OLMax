from __future__ import annotations

import argparse
import gc
import os
import time

import equinox as eqx
import jax
import optax

import olmax.distributed as dist
import olmax.nn as nn
import olmax.nn.functional as F
import olmax.nn.transformer.recipes as recipes
from olmax.data.utils import generate_batches_of_sequential_tokens
from olmax.jax_utils import cast_tree, count_params
from olmax.types import Array, DTypeLike


def main(
    recipe: str,
    sequence_length: int | None = None,
    instances_per_device: int | None = None,
    vocab_size: int = 50_304,
    param_dtype: DTypeLike = float,
    compute_dtype: DTypeLike = jax.dtypes.bfloat16,
    learning_rate: float = 1e-3,
    train_steps: int = 100,
    attn_window_size: int | tuple[int, int] | None = None,
):
    if recipe == "271M":
        model_config = recipes.llama_like_271M(
            vocab_size, param_dtype=param_dtype, attn_window_size=attn_window_size
        )
        if sequence_length is None:
            sequence_length = 1024
        if instances_per_device is None:
            instances_per_device = 16
    elif recipe == "7B":
        model_config = recipes.llama_like_7B(
            vocab_size, param_dtype=param_dtype, attn_window_size=attn_window_size
        )
        if sequence_length is None:
            sequence_length = 4096
        if instances_per_device is None:
            instances_per_device = 1
    else:
        raise ValueError(recipe)

    batch_size_per_device = sequence_length * instances_per_device

    print("========================= train integration test starting... =========================")
    key = jax.random.PRNGKey(0)
    model_key, data_key = jax.random.split(key)
    parallel_config = dist.ParallelConfig.FSDP()

    print("initializing model...")
    model = model_config.build(model_key, parallel_config=parallel_config)
    print(model)
    num_params = count_params(model)
    num_non_embedding_prams = num_params - model.embedding.weight.size
    print(
        f"Build model with {num_params:,d} total parameters, "
        f"{num_non_embedding_prams:,d} non-embedding parameters"
    )

    print("initializing optimizer...")
    optim = optax.adamw(learning_rate)
    opt_state = optim.init(model)  # pyright: ignore

    @eqx.filter_value_and_grad
    def compute_loss(model: nn.Transformer, input_ids: Array, labels: Array):
        input_ids = jax.lax.with_sharding_constraint(input_ids, parallel_config.get_data_sharding())
        labels = jax.lax.with_sharding_constraint(labels, parallel_config.get_data_sharding())
        logits = model(input_ids)
        logits = jax.lax.with_sharding_constraint(logits, parallel_config.get_data_sharding())
        loss = F.cross_entropy_loss(logits, labels)
        return loss

    @eqx.filter_jit(donate="all")
    def train_step(
        model: nn.Transformer, input_ids: Array, labels: Array, opt_state: optax.OptState
    ) -> tuple[Array, nn.Transformer, optax.OptState]:
        model = jax.lax.with_sharding_constraint(model, parallel_config.get_param_sharding())
        input_ids = jax.lax.with_sharding_constraint(input_ids, parallel_config.get_data_sharding())
        labels = jax.lax.with_sharding_constraint(labels, parallel_config.get_data_sharding())

        # Cast model to lower precision compute dtype.
        if compute_dtype != param_dtype:
            model_with_compute_dtype = cast_tree(model, compute_dtype)
        else:
            model_with_compute_dtype = model

        # Calculate loss and gradients.
        loss, grads = compute_loss(model_with_compute_dtype, input_ids, labels)
        grads = jax.lax.with_sharding_constraint(grads, parallel_config.get_param_sharding())

        # Cast grads back to param dtype.
        if compute_dtype != param_dtype:
            grads = cast_tree(grads, param_dtype)

        # Take optimizer step.
        updates, opt_state = optim.update(grads, opt_state, model)  # pyright: ignore
        model = eqx.apply_updates(model, updates)

        model = jax.lax.with_sharding_constraint(model, parallel_config.get_param_sharding())
        return loss, model, opt_state

    global_batch_size = batch_size_per_device * dist.get_global_device_count()
    per_process_batch_size = global_batch_size // dist.get_process_world_size()
    per_process_batch_size_instances = per_process_batch_size // sequence_length

    print("starting training...")
    gc.collect()
    batch_start = time.monotonic()
    for step, (input_ids, labels) in enumerate(
        generate_batches_of_sequential_tokens(
            data_key,
            local_data_parallel_rank=0,
            vocab_size=vocab_size,
            sequence_length=sequence_length,
            num_local_instances=per_process_batch_size_instances,
            total_batches=train_steps,
            parallel_config=parallel_config,
        )
    ):
        # Do a step.
        loss, model, opt_state = train_step(model, input_ids, labels, opt_state)

        # Log progress.
        metrics = {"step": step + 1, "loss": f"{loss:.4f}"}
        batch_end = time.monotonic()
        metrics["TPS"] = f"{int(batch_size_per_device / (batch_end - batch_start)):,d}"
        batch_start = batch_end
        print(", ".join(f"{name}={value}" for name, value in metrics.items()))

    print("done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("train_transformer")
    parser.add_argument("--recipe", choices=["271M", "7B"], default="271M")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no-jit", action="store_true")
    parser.add_argument("--no-remat", action="store_true")
    parser.add_argument("--xla-mem-frac", type=str, default="0.95")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--attn-window-size", type=int)
    opts = parser.parse_args()

    if opts.no_jit:
        jax.config.update("jax_disable_jit", True)

    if opts.no_remat:
        jax.config.update("jax_compiler_enable_remat_pass", False)

    if opts.debug:
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    else:
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = opts.xla_mem_frac

    main(opts.recipe, instances_per_device=opts.batch_size, attn_window_size=opts.attn_window_size)
