import time

import equinox as eqx
import jax
import optax

import olmax.distributed as dist
import olmax.nn as nn
import olmax.nn.functional as F
from olmax.data.utils import generate_batches_of_sequential_tokens
from olmax.types import Array

VOCAB_SIZE = 32_000
SEQUENCE_LENGTH = 1024
BATCH_SIZE = SEQUENCE_LENGTH * 32
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
        ),
        norm=NORM_CONFIG,
        bias=False,
    ),
    lm_head=nn.LMHead.Config(norm=NORM_CONFIG, bias=False),
)
LEARNING_RATE = 1e-3
TRAIN_STEPS = 100


def main():
    print("========================= train integration test starting... =========================")
    key = jax.random.PRNGKey(0)
    model_key, data_key = jax.random.split(key)
    parallel_config = dist.ParallelConfig.FSDP()

    print("initializing model...")
    model = MODEL_CONFIG.build(model_key, parallel_config=parallel_config)
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
    start_time: float | None = None
    start_step: int = 0
    first_batch: bool = True
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
        # Bookkeeping.
        if not first_batch:
            start_time = time.monotonic()
            start_step = step

        # Do a step.
        loss, model, opt_state = train_step(model, input_ids, labels, opt_state)

        # Calculate throughput.
        tokens_per_second_per_device: int | str = "N/A"
        if start_time is not None:
            elapsed_time = time.monotonic() - start_time
            avg_time_per_batch = elapsed_time / (step - start_step)
            batches_per_second = 1 / avg_time_per_batch
            tokens_per_second_per_device = int(batches_per_second * per_device_batch_size_tokens)

        # Log progress.
        print(f"step={step+1}, loss={loss.item():.5f}, TPS={tokens_per_second_per_device}")
        first_batch = False

    print("done.")


if __name__ == "__main__":
    main()
