import jax

import olmax.nn as nn


def test_rope(head_dim: int = 16, n_heads: int = 4, seq_len: int = 12, batch_size: int = 2):
    key = jax.random.PRNGKey(0)
    key, batch_key = jax.random.split(key)
    batch = jax.random.normal(batch_key, (batch_size, seq_len, n_heads, head_dim))
    rope = nn.RotaryPositionalEmbedding(head_dim=head_dim, key=key)
    out = rope(batch, head_first=False)
    assert out.shape == (batch_size, seq_len, n_heads, head_dim)
