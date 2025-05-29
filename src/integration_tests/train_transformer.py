import time

import equinox as eqx
import jax
import optax

import olmax.distributed as dist
import olmax.nn as nn
import olmax.nn.functional as F
from olmax.data.utils import generate_batches_of_sequential_tokens
from olmax.jax_utils import cast_tree
from olmax.types import Array

VOCAB_SIZE = 50_304
#  SEQUENCE_LENGTH = 1024
SEQUENCE_LENGTH = 4096
#  BATCH_SIZE_PER_DEVICE = SEQUENCE_LENGTH * 16
BATCH_SIZE_PER_DEVICE = SEQUENCE_LENGTH * 2
#  LEARNING_RATE = 1e-3
LEARNING_RATE = 1e-4
TRAIN_STEPS = 100

PARAM_DTYPE = float
COMPUTE_DTYPE = jax.dtypes.bfloat16

NORM_CONFIG = nn.LayerNorm.Config.rms_norm(bias=False)
MODEL_CONFIG = nn.Transformer.Config(
    vocab_size=VOCAB_SIZE,
    #  d_model=1024,
    #  hidden_size=2816,
    #  num_layers=16,
    d_model=4096,
    hidden_size=11008,
    num_layers=32,
    block=nn.TransformerBlock.Config(
        attention=nn.MultiheadSelfAttention.Config(
            #  n_heads=8,
            n_heads=32,
            rope=nn.RotaryPositionalEmbedding.Config(theta=10_000),
            bias=False,
            dtype=PARAM_DTYPE,
        ),
        norm=NORM_CONFIG,
        bias=False,
        dtype=PARAM_DTYPE,
    ),
    lm_head=nn.LMHead.Config(norm=NORM_CONFIG, bias=False, dtype=PARAM_DTYPE),
    dtype=PARAM_DTYPE,
)


def main():
    print("========================= train integration test starting... =========================")
    key = jax.random.PRNGKey(0)
    model_key, data_key = jax.random.split(key)
    parallel_config = dist.ParallelConfig.FSDP()

    print("initializing model...")
    model = MODEL_CONFIG.build(model_key, parallel_config=parallel_config)
    print(model)
    num_params = jax.tree.reduce(lambda c, p: c + p.size, model, 0)
    num_non_embedding_prams = num_params - model.embedding.weight.size
    print(
        f"Build model with {num_params:,d} total parameters, "
        f"{num_non_embedding_prams:,d} non-embedding parameters"
    )

    print("initializing optimizer...")
    optim = optax.adamw(LEARNING_RATE)
    opt_state = optim.init(model)  # pyright: ignore

    @eqx.filter_value_and_grad
    def compute_loss(model: nn.Transformer, input_ids: Array, labels: Array):
        logits = model(input_ids)
        optax.softmax_cross_entropy
        return F.cross_entropy_loss(logits, labels)

    @eqx.filter_jit
    def train_step(
        model: nn.Transformer, input_ids: Array, labels: Array, opt_state: optax.OptState
    ) -> tuple[Array, nn.Transformer, optax.OptState]:
        # Cast model to lower precision compute dtype.
        if COMPUTE_DTYPE != PARAM_DTYPE:
            model_with_compute_dtype = cast_tree(model, COMPUTE_DTYPE)
        else:
            model_with_compute_dtype = model

        # Calculate loss and gradients.
        loss, grads = compute_loss(model_with_compute_dtype, input_ids, labels)

        # Cast grads back to param dtype.
        if COMPUTE_DTYPE != PARAM_DTYPE:
            grads = cast_tree(grads, PARAM_DTYPE)

        # Take optimizer step.
        updates, opt_state = optim.update(grads, opt_state, model)  # pyright: ignore
        model = eqx.apply_updates(model, updates)

        return loss, model, opt_state

    global_batch_size = BATCH_SIZE_PER_DEVICE * dist.get_global_device_count()
    per_process_batch_size = global_batch_size // dist.get_process_world_size()
    per_process_batch_size_instances = per_process_batch_size // SEQUENCE_LENGTH

    print("starting training...")
    batch_start = time.monotonic()
    for step, (input_ids, labels) in enumerate(
        generate_batches_of_sequential_tokens(
            data_key,
            local_data_parallel_rank=0,
            vocab_size=VOCAB_SIZE,
            sequence_length=SEQUENCE_LENGTH,
            num_local_instances=per_process_batch_size_instances,
            total_batches=TRAIN_STEPS,
            parallel_config=parallel_config,
        )
    ):
        # Do a step.
        loss, model, opt_state = train_step(model, input_ids, labels, opt_state)

        # Log progress.
        metrics = {"step": step + 1, "loss": f"{loss:.4f}"}
        batch_end = time.monotonic()
        metrics["TPS"] = f"{int(BATCH_SIZE_PER_DEVICE / (batch_end - batch_start)):,d}"
        batch_start = batch_end
        print(", ".join(f"{name}={value}" for name, value in metrics.items()))

    print("done.")


if __name__ == "__main__":
    main()
