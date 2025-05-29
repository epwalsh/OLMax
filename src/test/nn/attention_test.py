import jax
import pytest

import olmax.nn as nn


@pytest.mark.parametrize("d_model, n_heads, n_kv_heads", [(16, 4, None)])
def test_mhsa(
    d_model: int, n_heads: int, n_kv_heads: int | None, seq_len: int = 12, batch_size: int = 2
):
    key = jax.random.PRNGKey(0)
    key, batch_key = jax.random.split(key)
    batch = jax.random.normal(batch_key, (batch_size, seq_len, d_model))
    mhsa = nn.MultiheadSelfAttention(
        d_model=d_model,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        key=key,
        rope=nn.RotaryPositionalEmbedding.Config(),
    )
    out = mhsa(batch)
    assert out.shape == (batch_size, seq_len, d_model)
