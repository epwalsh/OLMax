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
        jax.debug.print("logits={logits} labels={labels}", logits=logits, labels=labels)
        return F.cross_entropy_loss(logits, labels)

    @eqx.filter_jit
    def train_step(
        model: nn.Transformer, input_ids: Array, labels: Array, opt_state: optax.OptState
    ) -> tuple[Array, nn.Transformer, optax.OptState]:
        loss, grads = compute_loss(model, input_ids, labels)
        updates, opt_state = optim.update(grads, opt_state, model)  # pyright: ignore
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state

    print("starting training...")
    for step, (input_ids, labels) in enumerate(
        generate_batches_of_sequential_tokens(
            data_key,
            local_data_parallel_rank=0,
            vocab_size=VOCAB_SIZE,
            sequence_length=SEQUENCE_LENGTH,
            num_local_instances=(BATCH_SIZE // SEQUENCE_LENGTH) // dist.get_global_device_count(),
            total_batches=TRAIN_STEPS,
            parallel_config=parallel_config,
        )
    ):
        loss, model, opt_state = train_step(model, input_ids, labels, opt_state)
        print(f"step={step}, loss={loss.item():.5f}")

    print("done.")


if __name__ == "__main__":
    main()
