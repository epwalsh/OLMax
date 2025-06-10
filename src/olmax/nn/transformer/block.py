from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import Callable, Generic, Type, TypeVar

import jax

from ...config import Registrable, required_field
from ...distributed.parallel import MeshResource
from ...types import Array, DTypeLike, PRNGKeyArray
from ..attention import MultiheadSelfAttention, MultiheadSelfAttentionConfig
from ..mlp import GatedMLP
from ..module import Module
from ..normalization import LayerNormConfig, Normalizer, NormalizerConfig


class TransformerBlock(Module):
    mlp: GatedMLP
    mlp_norm: Normalizer
    attention: MultiheadSelfAttention
    attention_norm: Normalizer

    def __init__(
        self,
        d_model: int,
        hidden_size: int,
        key: PRNGKeyArray,
        attention: MultiheadSelfAttentionConfig,
        norm: NormalizerConfig,
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

    @classmethod
    def Config(cls, **kwargs) -> TransformerBlockConfig:
        return DefaultTransformerBlockConfig(**kwargs)

    @jax.named_scope("olmax.nn.TransformerBlock")
    def __call__(self, x: Array) -> Array:
        assert x.ndim == 3
        h = x + self.attention(self.attention_norm(x))
        h = h + self.mlp(self.mlp_norm(h))
        return h


class ReorderedNormTransformerBlock(TransformerBlock):
    @classmethod
    def Config(cls, **kwargs) -> ReorderedNormTransformerBlockConfig:
        return ReorderedNormTransformerBlockConfig(**kwargs)

    @jax.named_scope("olmax.nn.ReorderedTransformerBlock")
    def __call__(self, x: Array) -> Array:
        assert x.ndim == 3
        h = x + self.attention_norm(self.attention(x))
        h = h + self.mlp_norm(self.mlp(h))
        return h


class GemmaTransformerBlock(TransformerBlock):
    attention_input_norm: Normalizer
    mlp_input_norm: Normalizer

    def __init__(
        self,
        d_model: int,
        hidden_size: int,
        key: PRNGKeyArray,
        attention: MultiheadSelfAttentionConfig,
        norm: NormalizerConfig,
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

    @classmethod
    def Config(cls, **kwargs) -> GemmaTransformerBlockConfig:
        return GemmaTransformerBlockConfig(**kwargs)

    @jax.named_scope("olmax.nn.Gemma2TransformerBlock")
    def __call__(self, x: Array) -> Array:
        assert x.ndim == 3
        h = x + self.attention_norm(self.attention(self.attention_input_norm(x)))
        h = h + self.mlp_norm(self.mlp(self.mlp_input_norm(h)))
        return h


B = TypeVar("B", bound=TransformerBlock)


@dataclass
class TransformerBlockConfig(Registrable, Generic[B]):
    attention: MultiheadSelfAttentionConfig = required_field("attention", strict=True)
    norm: NormalizerConfig = required_field("norm", strict=True)
    bias: bool = False
    dtype: DTypeLike = float

    @classmethod
    @abstractmethod
    def get_class(cls) -> Type[B]:
        raise NotImplementedError

    @classmethod
    def Default(cls, **kwargs) -> DefaultTransformerBlockConfig:
        return DefaultTransformerBlockConfig(**kwargs)

    @classmethod
    def ReorderedNorm(cls, **kwargs) -> ReorderedNormTransformerBlockConfig:
        return ReorderedNormTransformerBlockConfig(**kwargs)

    @classmethod
    def Gemma(cls, **kwargs) -> GemmaTransformerBlockConfig:
        return GemmaTransformerBlockConfig(**kwargs)

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
    ) -> B:
        return self.get_class()(
            d_model,
            hidden_size,
            key,
            attention=attention if attention is not None else self.attention,
            norm=norm if norm is not None else self.norm,
            bias=bias if bias is not None else self.bias,
            dtype=dtype if dtype is not None else self.dtype,
            mesh_resource=mesh_resource,
        )


@TransformerBlockConfig.register("default")
@dataclass
class DefaultTransformerBlockConfig(TransformerBlockConfig[TransformerBlock]):
    @classmethod
    def get_class(cls) -> Type[TransformerBlock]:
        return TransformerBlock


@TransformerBlockConfig.register("reordered_norm")
@dataclass
class ReorderedNormTransformerBlockConfig(TransformerBlockConfig[ReorderedNormTransformerBlock]):
    @classmethod
    def get_class(cls) -> Type[ReorderedNormTransformerBlock]:
        return ReorderedNormTransformerBlock


@TransformerBlockConfig.register("gemma")
@dataclass
class GemmaTransformerBlockConfig(TransformerBlockConfig[GemmaTransformerBlock]):
    @classmethod
    def get_class(cls) -> Type[GemmaTransformerBlock]:
        return GemmaTransformerBlock
