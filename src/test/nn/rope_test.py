import jax

import olmax.nn as nn


def test_rope(d_model: int = 16, seq_len: int = 12, batch_size: int = 2):
    key = jax.random.PRNGKey(0)
    key, batch_key = jax.random.split(key)
    batch = jax.random.normal(batch_key, (batch_size, seq_len, d_model))
    rope = nn.RotaryPositionalEmbedding(d_model=d_model, key=key)
    out = rope(batch)
    assert out.shape == (batch_size, seq_len, d_model)
