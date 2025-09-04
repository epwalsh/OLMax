import math
import warnings
from dataclasses import dataclass
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp

from ..distributed.parallel import MeshResource
from ..jax_utils import get_cudnn_version
from ..types import Array, DTypeLike, PRNGKeyArray
from .linear import Linear
from .module import Module
from .normalization import Normalizer, NormalizerConfig
from .rope import RotaryPositionalEmbedding, RotaryPositionalEmbeddingConfig


class Attention(Module):
    pass


class MultiheadSelfAttention(Attention):
    q_proj: Linear
    k_proj: Linear
    v_proj: Linear
    o_proj: Linear
    rope: RotaryPositionalEmbedding | None
    q_norm: Normalizer | None
    k_norm: Normalizer | None

    qk_norm_headwise: bool = eqx.field(static=True)
    n_heads: int = eqx.field(static=True)
    n_kv_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    window_size: int | tuple[int, int] | None = eqx.field(static=True)
    implementation: Literal["xla", "cudnn", "te_fused"] | None = eqx.field(static=True)

    def __init__(
        self,
        *,
        d_model: int,
        n_heads: int,
        key: PRNGKeyArray,
        rope: RotaryPositionalEmbeddingConfig | None = None,
        qk_norm: NormalizerConfig | None = None,
        qk_norm_headwise: bool = False,
        n_kv_heads: int | None = None,
        head_dim: int | None = None,
        bias: bool = True,
        window_size: int | tuple[int, int] | None = None,
        dtype: DTypeLike = "float32",
        implementation: Literal["xla", "cudnn", "te_fused"] | None = None,
        mesh_resource: MeshResource | None = None,
        checkpoint_name: str | None = None,
        inference_mode: bool = False,
    ):
        super().__init__(mesh_resource, checkpoint_name, inference_mode)

        if implementation is None:
            if mesh_resource is not None and mesh_resource.cp_sharding_axis is not None:
                from olmax.te_utils import assert_te

                assert_te("context parallelism")
                implementation = "te_fused"
            elif jax.default_backend() == "gpu":
                if get_cudnn_version() is not None:
                    implementation = "cudnn"
                else:
                    warnings.warn(
                        "cuDNN not detected, falling back to slower XLA attention implementation"
                    )

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        self.head_dim = head_dim if head_dim is not None else d_model // n_heads
        self.window_size = window_size
        self.implementation = implementation

        (
            q_proj_key,
            k_proj_key,
            v_proj_key,
            o_proj_key,
            rope_key,
            q_norm_key,
            k_norm_key,
        ) = jax.random.split(key, 7)
        self.q_proj = Linear(
            d_model,
            self.n_heads * self.head_dim,
            q_proj_key,
            bias=bias,
            dtype=dtype,
            mesh_resource=mesh_resource,
            checkpoint_name=None if checkpoint_name is None else f"{checkpoint_name}.q_proj",
        )
        self.k_proj = Linear(
            d_model,
            self.n_kv_heads * self.head_dim,
            k_proj_key,
            bias=bias,
            dtype=dtype,
            mesh_resource=mesh_resource,
            checkpoint_name=None if checkpoint_name is None else f"{checkpoint_name}.k_proj",
        )
        self.v_proj = Linear(
            d_model,
            self.n_kv_heads * self.head_dim,
            v_proj_key,
            bias=bias,
            dtype=dtype,
            mesh_resource=mesh_resource,
            checkpoint_name=None if checkpoint_name is None else f"{checkpoint_name}.v_proj",
        )
        self.o_proj = Linear(
            self.n_heads * self.head_dim,
            d_model,
            o_proj_key,
            bias=bias,
            dtype=dtype,
            mesh_resource=mesh_resource,
            checkpoint_name=None if checkpoint_name is None else f"{checkpoint_name}.o_proj",
        )
        self.rope = (
            None
            if rope is None
            else rope.build(
                head_dim=self.head_dim,
                key=rope_key,
                mesh_resource=mesh_resource,
                checkpoint_name=None if checkpoint_name is None else f"{checkpoint_name}.rope",
            )
        )
        self.q_norm = (
            None
            if qk_norm is None
            else qk_norm.build(
                self.head_dim if qk_norm_headwise else self.n_heads * self.head_dim,
                q_norm_key,
                mesh_resource=mesh_resource,
                checkpoint_name=None if checkpoint_name is None else f"{checkpoint_name}.q_norm",
            )
        )
        self.k_norm = (
            None
            if qk_norm is None
            else qk_norm.build(
                self.head_dim if qk_norm_headwise else self.n_heads * self.head_dim,
                k_norm_key,
                mesh_resource=mesh_resource,
                checkpoint_name=None if checkpoint_name is None else f"{checkpoint_name}.k_norm",
            )
        )
        self.qk_norm_headwise = qk_norm_headwise

    @classmethod
    def Config(cls, **kwargs) -> "MultiheadSelfAttentionConfig":
        return MultiheadSelfAttentionConfig(**kwargs)

    @jax.named_scope("olmax.nn.MultiheadSelfAttention")
    def forward(self, x: Array) -> Array:
        assert x.ndim == 3  # (batch_size, seq_len, d_model)
        B, S, _ = x.shape

        # shape: (batch_size, seq_len, n_heads * head_dim)
        q = self.q_proj(x)
        # shape: (batch_size, seq_len, n_kv_heads * head_dim)
        k = self.k_proj(x)
        # shape: (batch_size, seq_len, n_kv_heads * head_dim)
        v = self.v_proj(x)

        if self.q_norm is not None and not self.qk_norm_headwise:
            q = self.q_norm(q)
        if self.k_norm is not None and not self.qk_norm_headwise:
            k = self.k_norm(k)

        # shape: (batch_size, seq_len, n_heads, head_dim)
        q = q.reshape(B, S, self.n_heads, self.head_dim)
        # shape: (batch_size, seq_len, n_kv_heads, head_dim)
        k = k.reshape(B, S, self.n_kv_heads, self.head_dim)
        # shape: (batch_size, seq_len, n_kv_heads, head_dim)
        v = v.reshape(B, S, self.n_kv_heads, self.head_dim)

        if self.q_norm is not None and self.qk_norm_headwise:
            q = self.q_norm(q)
        if self.k_norm is not None and self.qk_norm_headwise:
            k = self.k_norm(k)

        if self.rope is not None:
            q = self.rope(q, head_first=False)
            k = self.rope(k, head_first=False)

        # shape: (batch_size, seq_len, n_heads, head_dim)
        if self.implementation == "te_fused":
            from olmax.te_utils import assert_te

            te = assert_te("fused attention")

            with jax.ensure_compile_time_eval():
                seq_lens = jnp.zeros(B, dtype=int) + S
                seq_descriptor = te.jax.attention.SequenceDescriptor.from_seqlens(seq_lens)

            if self.training:
                att = te.jax.attention.fused_attn(
                    qkv=(q, k, v),
                    bias=None,
                    sequence_descriptor=seq_descriptor,
                    seed=None,
                    attn_bias_type=te.jax.attention.AttnBiasType.NO_BIAS,
                    attn_mask_type=te.jax.attention.AttnMaskType.CAUSAL_MASK,
                    qkv_layout=te.jax.attention.QKVLayout.BSHD_BSHD_BSHD,
                    scaling_factor=1.0 / math.sqrt(self.head_dim),
                    dropout_probability=0.0,
                    #  is_training=self.training,  # TODO: fix, might have to change back to static
                    is_training=True,
                    max_segments_per_seq=1,
                    window_size=self.window_size,
                    context_parallel_strategy=te.jax.attention.CPStrategy.DEFAULT,
                    context_parallel_causal_load_balanced=True,
                    context_parallel_axis=""
                    if self.mesh_resource is None
                    else (self.mesh_resource.cp_sharding_axis or ""),
                )
            else:
                raise NotImplementedError  # TODO: handle this
        else:
            att = jax.nn.dot_product_attention(
                q,
                k,
                v,
                is_causal=True,
                local_window_size=self.window_size,
                implementation=self.implementation,
            )

        # shape: (batch_size, seq_len, n_heads * head_dim)
        att = att.reshape(B, S, self.n_heads * self.head_dim)

        # shape: (batch_size, seq_len, d_model)
        out = self.o_proj(att)

        return out


@dataclass
class MultiheadSelfAttentionConfig:
    n_heads: int
    n_kv_heads: int | None = None
    head_dim: int | None = None
    rope: RotaryPositionalEmbeddingConfig | None = None
    qk_norm: NormalizerConfig | None = None
    qk_norm_headwise: bool = False
    bias: bool = True
    window_size: int | tuple[int, int] | None = None
    implementation: Literal["xla", "cudnn", "te_fused"] | None = None
    dtype: DTypeLike = "float32"

    def build(
        self,
        d_model: int,
        key: PRNGKeyArray,
        *,
        n_heads: int | None = None,
        head_dim: int | None = None,
        rope: RotaryPositionalEmbeddingConfig | None = None,
        qk_norm: NormalizerConfig | None = None,
        qk_norm_headwise: bool | None = None,
        n_kv_heads: int | None = None,
        bias: bool | None = None,
        window_size: int | tuple[int, int] | None = None,
        dtype: DTypeLike | None = None,
        implementation: Literal["xla", "cudnn"] | None = None,
        mesh_resource: MeshResource | None = None,
        checkpoint_name: str | None = None,
    ) -> MultiheadSelfAttention:
        return MultiheadSelfAttention(
            d_model=d_model,
            key=key,
            n_heads=n_heads if n_heads is not None else self.n_heads,
            n_kv_heads=n_kv_heads if n_kv_heads is not None else self.n_kv_heads,
            head_dim=head_dim if head_dim is not None else self.head_dim,
            rope=rope if rope is not None else self.rope,
            qk_norm=qk_norm if qk_norm is not None else self.qk_norm,
            qk_norm_headwise=qk_norm_headwise
            if qk_norm_headwise is not None
            else self.qk_norm_headwise,
            bias=bias if bias is not None else self.bias,
            window_size=window_size if window_size is not None else self.window_size,
            dtype=dtype if dtype is not None else self.dtype,
            implementation=implementation if implementation is not None else self.implementation,
            mesh_resource=mesh_resource,
            checkpoint_name=checkpoint_name,
        )
