import time

import equinox as eqx
import jax
import optax

import olmax.distributed as dist
import olmax.nn as nn
import olmax.nn.functional as F
from olmax.data.utils import generate_batches_of_sequential_tokens
from olmax.types import Array

VOCAB_SIZE = 50_304
SEQUENCE_LENGTH = 1024
BATCH_SIZE = SEQUENCE_LENGTH * 32
LEARNING_RATE = 1e-3
TRAIN_STEPS = 100

DTYPE = jax.dtypes.bfloat16
NORM_CONFIG = nn.LayerNorm.Config.rms_norm(bias=False)
MODEL_CONFIG = nn.Transformer.Config(
    d_model=1024,
    hidden_size=2816,
    vocab_size=VOCAB_SIZE,
    num_layers=16,
    block=nn.TransformerBlock.Config(
        attention=nn.MultiheadSelfAttention.Config(
            n_heads=8,
            rope=nn.RotaryPositionalEmbedding.Config(theta=10_000),
            bias=False,
            dtype=DTYPE,
        ),
        norm=NORM_CONFIG,
        bias=False,
        dtype=DTYPE,
    ),
    lm_head=nn.LMHead.Config(norm=NORM_CONFIG, bias=False, dtype=DTYPE),
    dtype=DTYPE,
)

LOG_INTERVAL = 5


def main():
    print("========================= train integration test starting... =========================")
    key = jax.random.PRNGKey(0)
    model_key, data_key = jax.random.split(key)
    parallel_config = dist.ParallelConfig.FSDP()

    print("initializing model...")
    model = MODEL_CONFIG.build(model_key, parallel_config=parallel_config)
    print(model)
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
        loss, grads = compute_loss(model, input_ids, labels)
        updates, opt_state = optim.update(grads, opt_state, model)  # pyright: ignore
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state

    per_device_batch_size_tokens = BATCH_SIZE // dist.get_global_device_count()
    per_process_batch_size_tokens = BATCH_SIZE // dist.get_process_world_size()
    per_process_batch_size_instances = per_process_batch_size_tokens // SEQUENCE_LENGTH

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
        metrics["TPS"] = f"{int(per_device_batch_size_tokens / (batch_end - batch_start)):,d}"
        batch_start = batch_end
        print(", ".join(f"{name}={value}" for name, value in metrics.items()))

    print("done.")


if __name__ == "__main__":
    main()
