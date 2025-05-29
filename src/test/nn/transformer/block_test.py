import jax

import olmax.nn as nn


def test_transformer_block(
    d_model: int = 16, hidden_size: int = 32, batch_size: int = 2, seq_len: int = 12
):
    key = jax.random.PRNGKey(0)
    key, data_key = jax.random.split(key, 2)
    block = nn.TransformerBlock(
        d_model=d_model,
        hidden_size=hidden_size,
        key=key,
        attention=nn.MultiheadSelfAttention.Config(n_heads=4),
        norm=nn.LayerNorm.Config(),
    )
    batch = jax.random.normal(data_key, (batch_size, seq_len, d_model))
    out = block(batch)
    assert out.shape == (batch_size, seq_len, d_model)


def test_reordered_norm_transformer_block(
    d_model: int = 16, hidden_size: int = 32, batch_size: int = 2, seq_len: int = 12
):
    key = jax.random.PRNGKey(0)
    key, data_key = jax.random.split(key, 2)
    block = nn.ReorderedNormTransformerBlock(  # pyright: ignore
        d_model=d_model,  # pyright: ignore
        hidden_size=hidden_size,  # pyright: ignore
        key=key,  # pyright: ignore
        attention=nn.MultiheadSelfAttention.Config(n_heads=4),
        norm=nn.LayerNorm.Config(),  # pyright: ignore
    )
    batch = jax.random.normal(data_key, (batch_size, seq_len, d_model))
    out = block(batch)
    assert out.shape == (batch_size, seq_len, d_model)
