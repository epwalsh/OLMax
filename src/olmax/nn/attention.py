from typing import ClassVar

import equinox as eqx
import jax

from ..distributed.parallel import ParallelConfig
from ..types import Array, DTypeLike, PRNGKeyArray
from .linear import Linear
from .module import Module


class MultiheadSelfAttention(Module):
    keepdims: ClassVar[int] = 2

    w_q: Linear
    w_k: Linear
    w_v: Linear
    w_out: Linear

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

        w_q_key, w_k_key, w_v_key, w_out_key = jax.random.split(key, 4)
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

    @jax.named_scope("olmax.nn.MultiheadSelfAttention")
    def forward(self, x: Array) -> Array:
        assert x.ndim == 2  # (seq_len, d_model)

        # shape: (seq_len, n_heads, head_dim)
        q = jax.vmap(self.w_q.forward)(x).reshape(-1, self.n_heads, self.head_dim)
        # shape: (seq_len, n_kv_heads, head_dim)
        k = jax.vmap(self.w_k.forward)(x).reshape(-1, self.n_kv_heads, self.head_dim)
        # shape: (seq_len, n_kv_heads, head_dim)
        v = jax.vmap(self.w_v.forward)(x).reshape(-1, self.n_kv_heads, self.head_dim)

        # TODO: clip QKV
        # TODO: QK-norm
        # TODO: RoPE

        # shape: (seq_len, n_heads, head_dim)
        att = jax.nn.dot_product_attention(
            q, k, v, is_causal=True, local_window_size=self.window_size
        )

        # shape: (seq_len, d_model)
        att = att.reshape(-1, self.n_heads * self.head_dim)

        # shape: (seq_len, d_model)
        out = jax.vmap(self.w_out)(att)

        return out
