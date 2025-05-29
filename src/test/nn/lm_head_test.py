import jax

import olmax.nn as nn


def test_lm_head(d_model: int = 16, vocab_size: int = 32, batch_size: int = 2, seq_len: int = 12):
    key = jax.random.PRNGKey(0)
    key, data_key = jax.random.split(key, 2)
    lm_head = nn.LMHead(d_model, vocab_size, key, norm=nn.LayerNorm.Config())
    batch = jax.random.normal(data_key, (batch_size, seq_len, d_model))
    out = lm_head(batch)
    assert out.shape == (batch_size, seq_len, vocab_size)
