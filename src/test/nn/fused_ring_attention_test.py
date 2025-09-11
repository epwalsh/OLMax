import math

import jax
import jax.numpy as jnp
import pytest
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P


def test_fused_ring_attention():
    try:
        import transformer_engine.jax.attention as te_attn  # type: ignore
    except ImportError:
        pytest.skip("requires TransformerEngine")

    if jax.local_device_count() < 2:
        pytest.skip("requires multiple devices")

    print("Starting test...")

    mesh = jax.make_mesh((jax.local_device_count(),), ("context",))
    print("Running test on mesh:", mesh)

    B, S, H, H_kv, D = 1, 64, 16, 16, 8
    seq_lens = jnp.zeros(B, dtype=int) + S
    seq_descriptor = te_attn.SequenceDescriptor.from_seqlens(seq_lens)

    @jax.jit
    def run_fused_attn(q, k, v):
        return te_attn.fused_attn(
            qkv=(q, k, v),
            bias=None,
            sequence_descriptor=seq_descriptor,
            seed=None,
            attn_bias_type=te_attn.AttnBiasType.NO_BIAS,
            attn_mask_type=te_attn.AttnMaskType.CAUSAL_MASK,
            qkv_layout=te_attn.QKVLayout.BSHD_BSHD_BSHD,
            scaling_factor=1.0 / math.sqrt(D),
            dropout_probability=0.0,
            is_training=True,
            max_segments_per_seq=1,
            window_size=None,
            context_parallel_strategy=te_attn.CPStrategy.RING,
            context_parallel_causal_load_balanced=True,
            context_parallel_axis="context",
        )

    key = jax.random.PRNGKey(0)
    q_key, k_key, v_key = jax.random.split(key, 3)

    q = jax.random.normal(q_key, (B, S, H, D), dtype=jax.dtypes.bfloat16)
    k = jax.random.normal(k_key, (B, S, H_kv, D), dtype=jax.dtypes.bfloat16)
    v = jax.random.normal(v_key, (B, S, H_kv, D), dtype=jax.dtypes.bfloat16)

    q = jax.device_put(q, NamedSharding(mesh, P(None, "context")))
    k = jax.device_put(k, NamedSharding(mesh, P(None, "context")))
    v = jax.device_put(v, NamedSharding(mesh, P(None, "context")))

    att = run_fused_attn(q, k, v)
    assert att.shape == (B, S, H, D)

    print("Done")


if __name__ == "__main__":
    #  jax.config.update("jax_disable_jit", True)
    test_fused_ring_attention()
