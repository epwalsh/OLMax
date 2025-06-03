from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Literal, Type

import equinox as eqx
import jax

from ..distributed.parallel import MeshResource
from ..types import Array, DTypeLike, PRNGKeyArray
from .linear import Linear
from .module import Module
from .normalization import LayerNorm, LayerNormConfig
from .rope import RotaryPositionalEmbedding, RotaryPositionalEmbeddingConfig


@dataclass
class MultiheadSelfAttentionConfig:
    n_heads: int
    n_kv_heads: int | None = None
    rope: RotaryPositionalEmbeddingConfig | None = None
    qk_norm: LayerNormConfig | None = None
    bias: bool = True
    window_size: int | tuple[int, int] | None = None
    implementation: Literal["xla", "cudnn"] | None = None
    dtype: DTypeLike = float

    def build(
        self,
        d_model: int,
        key: PRNGKeyArray,
        *,
        n_heads: int | None = None,
        rope: RotaryPositionalEmbeddingConfig | None = None,
        qk_norm: LayerNormConfig | None = None,
        n_kv_heads: int | None = None,
        bias: bool | None = None,
        window_size: int | tuple[int, int] | None = None,
        dtype: DTypeLike | None = None,
        implementation: Literal["xla", "cudnn"] | None = None,
        mesh_resource: MeshResource | None = None,
    ) -> MultiheadSelfAttention:
        return MultiheadSelfAttention(
            d_model=d_model,
            key=key,
            n_heads=n_heads if n_heads is not None else self.n_heads,
            n_kv_heads=n_kv_heads if n_kv_heads is not None else self.n_kv_heads,
            rope=rope if rope is not None else self.rope,
            qk_norm=qk_norm if qk_norm is not None else self.qk_norm,
            bias=bias if bias is not None else self.bias,
            window_size=window_size if window_size is not None else self.window_size,
            dtype=dtype if dtype is not None else self.dtype,
            implementation=implementation if implementation is not None else self.implementation,
            mesh_resource=mesh_resource,
        )


class MultiheadSelfAttention(Module):
    Config: ClassVar[Type[MultiheadSelfAttentionConfig]] = MultiheadSelfAttentionConfig

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
    implementation: Literal["xla", "cudnn"] | None = eqx.field(static=True)

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
        implementation: Literal["xla", "cudnn"] | None = None,
        mesh_resource: MeshResource | None = None,
    ):
        super().__init__(mesh_resource)
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        self.head_dim = d_model // n_heads
        self.window_size = window_size
        self.implementation = implementation

        w_q_key, w_k_key, w_v_key, w_out_key, rope_key, q_norm_key, k_norm_key = jax.random.split(
            key, 7
        )
        self.w_q = Linear(
            d_model,
            d_model,
            w_q_key,
            bias=bias,
            dtype=dtype,
            mesh_resource=mesh_resource,
        )
        self.w_k = Linear(
            d_model,
            self.n_kv_heads * self.head_dim,
            w_k_key,
            bias=bias,
            dtype=dtype,
            mesh_resource=mesh_resource,
        )
        self.w_v = Linear(
            d_model,
            self.n_kv_heads * self.head_dim,
            w_v_key,
            bias=bias,
            dtype=dtype,
            mesh_resource=mesh_resource,
        )
        self.w_out = Linear(
            d_model,
            d_model,
            w_out_key,
            bias=bias,
            dtype=dtype,
            mesh_resource=mesh_resource,
        )
        self.rope = (
            None
            if rope is None
            else rope.build(head_dim=self.head_dim, key=rope_key, mesh_resource=mesh_resource)
        )
        self.q_norm = (
            None
            if qk_norm is None
            else qk_norm.build(d_model, q_norm_key, mesh_resource=mesh_resource)
        )
        self.k_norm = (
            None
            if qk_norm is None
            else qk_norm.build(d_model, k_norm_key, mesh_resource=mesh_resource)
        )

    @jax.named_scope("olmax.nn.MultiheadSelfAttention")
    def __call__(self, x: Array) -> Array:
        assert x.ndim == 3  # (batch_size, seq_len, d_model)
        B, S, _ = x.shape

        # shape: (batch_size, seq_len, n_heads * head_dim)
        q = self.w_q(x)
        # shape: (batch_size, seq_len, n_kv_heads * head_dim)
        k = self.w_k(x)
        # shape: (batch_size, seq_len, n_kv_heads * head_dim)
        v = self.w_v(x)

        if self.q_norm is not None:
            q = self.q_norm(q)
        if self.k_norm is not None:
            k = self.k_norm(k)

        # shape: (batch_size, seq_len, n_heads, head_dim)
        q = q.reshape(B, S, self.n_heads, self.head_dim)
        # shape: (batch_size, seq_len, n_kv_heads, head_dim)
        k = k.reshape(B, S, self.n_kv_heads, self.head_dim)
        # shape: (batch_size, seq_len, n_kv_heads, head_dim)
        v = v.reshape(B, S, self.n_kv_heads, self.head_dim)

        if self.rope is not None:
            q = self.rope(q, head_first=False)
            k = self.rope(k, head_first=False)

        # shape: (batch_size, seq_len, n_heads, head_dim)
        att = jax.nn.dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            local_window_size=self.window_size,
            implementation=self.implementation,
        )

        # shape: (batch_size, seq_len, d_model)
        att = att.reshape(B, S, self.n_heads * self.head_dim)

        # shape: (batch_size, seq_len, d_model)
        out = self.w_out(att)

        return out
