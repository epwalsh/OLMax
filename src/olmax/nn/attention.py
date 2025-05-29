from typing import ClassVar

import equinox as eqx
import jax

from ..distributed.parallel import ParallelConfig
from ..types import Array, DTypeLike, PRNGKeyArray
from .linear import Linear
from .module import Module
from .normalization import LayerNorm, LayerNormConfig
from .rope import RotaryPositionalEmbedding, RotaryPositionalEmbeddingConfig


class MultiheadSelfAttention(Module):
    keepdims: ClassVar[int] = 2

    w_q: Linear
    w_k: Linear
    w_v: Linear
    w_out: Linear

    rope: RotaryPositionalEmbedding | None
    q_norm: LayerNorm | None
    k_norm: LayerNorm | None

    n_heads: int = eqx.field(static=True)
    n_kv_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    window_size: int | tuple[int, int] | None = eqx.field(static=True)

    def __init__(
        self,
        *,
        d_model: int,
        n_heads: int,
        key: PRNGKeyArray,
        rope: RotaryPositionalEmbeddingConfig | None = None,
        qk_norm: LayerNormConfig | None = None,
        n_kv_heads: int | None = None,
        bias: bool = True,
        window_size: int | tuple[int, int] | None = None,
        dtype: DTypeLike = float,
        parallel_config: ParallelConfig | None = None,
    ):
        super().__init__(parallel_config)
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        self.head_dim = d_model // n_heads
        self.window_size = window_size

        w_q_key, w_k_key, w_v_key, w_out_key, rope_key, q_norm_key, k_norm_key = jax.random.split(
            key, 7
        )
        self.w_q = Linear(
            d_model,
            d_model,
            w_q_key,
            bias=bias,
            dtype=dtype,
            parallel_config=parallel_config,
        )
        self.w_k = Linear(
            d_model,
            self.n_kv_heads * self.head_dim,
            w_k_key,
            bias=bias,
            dtype=dtype,
            parallel_config=parallel_config,
        )
        self.w_v = Linear(
            d_model,
            self.n_kv_heads * self.head_dim,
            w_v_key,
            bias=bias,
            dtype=dtype,
            parallel_config=parallel_config,
        )
        self.w_out = Linear(
            d_model,
            d_model,
            w_out_key,
            bias=bias,
            dtype=dtype,
            parallel_config=parallel_config,
        )
        self.rope = (
            None
            if rope is None
            else rope.build(head_dim=self.head_dim, key=rope_key, parallel_config=parallel_config)
        )
        self.q_norm = None if qk_norm is None else qk_norm.build(d_model, q_norm_key)
        self.k_norm = None if qk_norm is None else qk_norm.build(d_model, k_norm_key)

    @jax.named_scope("olmax.nn.MultiheadSelfAttention")
    def forward(self, x: Array) -> Array:
        assert x.ndim == 2  # (seq_len, d_model)

        # shape: (seq_len, n_heads * head_dim)
        q = jax.vmap(self.w_q.forward)(x)
        # shape: (seq_len, n_kv_heads * head_dim)
        k = jax.vmap(self.w_k.forward)(x)
        # shape: (seq_len, n_kv_heads * head_dim)
        v = jax.vmap(self.w_v.forward)(x)

        if self.q_norm is not None:
            q = jax.vmap(self.q_norm.forward)(q)
        if self.k_norm is not None:
            k = jax.vmap(self.k_norm.forward)(k)

        # shape: (seq_len, n_heads, head_dim)
        q = q.reshape(-1, self.n_heads, self.head_dim)
        # shape: (seq_len, n_kv_heads, head_dim)
        k = k.reshape(-1, self.n_kv_heads, self.head_dim)
        # shape: (seq_len, n_kv_heads, head_dim)
        v = v.reshape(-1, self.n_kv_heads, self.head_dim)

        if self.rope is not None:
            q = jax.vmap(self.rope.forward, 1, 1)(q)
            k = jax.vmap(self.rope.forward, 1, 1)(k)

        # shape: (seq_len, n_heads, head_dim)
        att = jax.nn.dot_product_attention(
            q, k, v, is_causal=True, local_window_size=self.window_size
        )

        # shape: (seq_len, d_model)
        att = att.reshape(-1, self.n_heads * self.head_dim)

        # shape: (seq_len, d_model)
        out = jax.vmap(self.w_out)(att)

        return out
