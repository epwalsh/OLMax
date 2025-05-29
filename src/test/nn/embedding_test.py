import jax

import olmax.nn as nn


def test_embedding(
    d_model: int = 16, vocab_size: int = 1024, seq_len: int = 12, batch_size: int = 2
):
    key = jax.random.PRNGKey(0)
    key, batch_key = jax.random.split(key)
    batch = jax.random.randint(batch_key, (batch_size, seq_len), 0, vocab_size)
    emb = nn.Embedding(d_model=d_model, num_embeddings=vocab_size, key=key)
    out = emb(batch)
    assert out.shape == (batch_size, seq_len, d_model)
