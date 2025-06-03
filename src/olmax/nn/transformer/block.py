from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Callable, ClassVar, Type

import jax
from typing_extensions import Self

from ...distributed.parallel import MeshResource
from ...types import Array, DTypeLike, PRNGKeyArray
from ..attention import MultiheadSelfAttention, MultiheadSelfAttentionConfig
from ..mlp import GatedMLP
from ..module import Module
from ..normalization import LayerNorm, LayerNormConfig


class TransformerBlockType(StrEnum):
    default = "default"
    reordered_norm = "reordered_norm"
    gemma2 = "gemma2"

    def get_class(self) -> Type[TransformerBlock]:
        if self == self.default:
            return TransformerBlock
        elif self == self.reordered_norm:
            return ReorderedNormTransformerBlock
        elif self == self.gemma2:
            return Gemma2TransformerBlock
        else:
            raise ValueError(self)


@dataclass
class TransformerBlockConfig:
    attention: MultiheadSelfAttentionConfig
    norm: LayerNormConfig
    bias: bool = False
    dtype: DTypeLike = float
    name: TransformerBlockType = TransformerBlockType.default

    @classmethod
    def reordered_norm(cls, **kwargs) -> Self:
        return cls(name=TransformerBlockType.reordered_norm, **kwargs)

    @classmethod
    def gemma2(cls, **kwargs) -> Self:
        return cls(name=TransformerBlockType.gemma2, **kwargs)

    def build(
        self,
        d_model: int,
        hidden_size: int,
        key: PRNGKeyArray,
        attention: MultiheadSelfAttentionConfig | None = None,
        norm: LayerNormConfig | None = None,
        bias: bool | None = None,
        dtype: DTypeLike | None = None,
        mesh_resource: MeshResource | None = None,
    ) -> TransformerBlock:
        return self.name.get_class()(
            d_model,
            hidden_size,
            key,
            attention=attention if attention is not None else self.attention,
            norm=norm if norm is not None else self.norm,
            bias=bias if bias is not None else self.bias,
            dtype=dtype if dtype is not None else self.dtype,
            mesh_resource=mesh_resource,
        )


class TransformerBlock(Module):
    Config: ClassVar[Type[TransformerBlockConfig]] = TransformerBlockConfig

    mlp: GatedMLP
    mlp_norm: LayerNorm
    attention: MultiheadSelfAttention
    attention_norm: LayerNorm

    def __init__(
        self,
        d_model: int,
        hidden_size: int,
        key: PRNGKeyArray,
        attention: MultiheadSelfAttentionConfig,
        norm: LayerNormConfig,
        bias: bool = False,
        dtype: DTypeLike = float,
        mesh_resource: MeshResource | None = None,
        activation: Callable[[Array], Array] = jax.nn.silu,
    ):
        super().__init__(mesh_resource)
        mlp_key, mlp_norm_key, attention_key, attention_norm_key = jax.random.split(key, 4)
        self.mlp = GatedMLP(
            d_model,
            hidden_size,
            mlp_key,
            activation=activation,
            bias=bias,
            dtype=dtype,
            mesh_resource=mesh_resource,
        )
        self.mlp_norm = norm.build(
            d_model,
            mlp_norm_key,
            mesh_resource=mesh_resource,
        )
        self.attention = attention.build(
            d_model,
            attention_key,
            bias=bias,
            dtype=dtype,
            mesh_resource=mesh_resource,
        )
        self.attention_norm = norm.build(
            d_model,
            attention_norm_key,
            mesh_resource=mesh_resource,
        )

    @jax.named_scope("olmax.nn.TransformerBlock")
    def __call__(self, x: Array) -> Array:
        assert x.ndim == 3
        h = x + self.attention(self.attention_norm(x))
        h = h + self.mlp(self.mlp_norm(h))
        return h


class ReorderedNormTransformerBlock(TransformerBlock):
    @jax.named_scope("olmax.nn.ReorderedTransformerBlock")
    def __call__(self, x: Array) -> Array:
        assert x.ndim == 3
        h = x + self.attention_norm(self.attention(x))
        h = h + self.mlp_norm(self.mlp(h))
        return h


class Gemma2TransformerBlock(TransformerBlock):
    attention_input_norm: LayerNorm
    mlp_input_norm: LayerNorm

    def __init__(
        self,
        d_model: int,
        hidden_size: int,
        key: PRNGKeyArray,
        attention: MultiheadSelfAttentionConfig,
        norm: LayerNormConfig,
        bias: bool = False,
        dtype: DTypeLike = float,
        activation: Callable[[Array], Array] = jax.nn.gelu,
        mesh_resource: MeshResource | None = None,
    ):
        key, attn_input_norm_key, mlp_input_norm_key = jax.random.split(key, 3)
        super().__init__(
            d_model=d_model,
            hidden_size=hidden_size,
            key=key,
            attention=attention,
            norm=norm,
            bias=bias,
            dtype=dtype,
            activation=activation,
            mesh_resource=mesh_resource,
        )
        self.attention_input_norm = norm.build(
            d_model,
            attn_input_norm_key,
            mesh_resource=mesh_resource,
        )
        self.mlp_input_norm = norm.build(
            d_model,
            mlp_input_norm_key,
            mesh_resource=mesh_resource,
        )

    @jax.named_scope("olmax.nn.Gemma2TransformerBlock")
    def __call__(self, x: Array) -> Array:
        assert x.ndim == 3
        h = x + self.attention_norm(self.attention(self.attention_input_norm(x)))
        h = h + self.mlp_norm(self.mlp(self.mlp_input_norm(h)))
        return h
