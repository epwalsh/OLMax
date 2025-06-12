from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import Callable, Generic, Type, TypeVar

import jax
from dataclass_extensions import Registrable

from ...distributed.parallel import MeshResource
from ...types import Array, DTypeLike, PRNGKeyArray
from ..attention import MultiheadSelfAttention, MultiheadSelfAttentionConfig
from ..mlp import GatedMLP
from ..module import Module
from ..normalization import LayerNormConfig, Normalizer, NormalizerConfig


class TransformerLayer(Module):
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
        block_idx: int | None = None,
        mesh_resource: MeshResource | None = None,
        checkpoint_name: str | None = None,
        activation: Callable[[Array], Array] = jax.nn.silu,
    ):
        if checkpoint_name is None:
            checkpoint_name = "block" if block_idx is None else f"block{block_idx}"

        super().__init__(mesh_resource, checkpoint_name)
        mlp_key, mlp_norm_key, attention_key, attention_norm_key = jax.random.split(key, 4)
        self.mlp = GatedMLP(
            d_model,
            hidden_size,
            mlp_key,
            activation=activation,
            bias=bias,
            dtype=dtype,
            mesh_resource=mesh_resource,
            checkpoint_name=f"{checkpoint_name}.mlp",
        )
        self.mlp_norm = norm.build(
            d_model,
            mlp_norm_key,
            mesh_resource=mesh_resource,
            checkpoint_name=f"{checkpoint_name}.mlp_norm",
        )
        self.attention = attention.build(
            d_model,
            attention_key,
            bias=bias,
            dtype=dtype,
            mesh_resource=mesh_resource,
            checkpoint_name=f"{checkpoint_name}.attention",
        )
        self.attention_norm = norm.build(
            d_model,
            attention_norm_key,
            mesh_resource=mesh_resource,
            checkpoint_name=f"{checkpoint_name}.attention_norm",
        )

    @classmethod
    def Config(cls, **kwargs) -> TransformerLayerConfig:
        return DefaultTransformerLayerConfig(**kwargs)

    @jax.named_scope("olmax.nn.TransformerLayer")
    def forward(self, x: Array) -> Array:
        assert x.ndim == 3
        h = x + self.attention(self.attention_norm(x))
        h = h + self.mlp(self.mlp_norm(h))
        return h


class ReorderedNormTransformerLayer(TransformerLayer):
    @classmethod
    def Config(cls, **kwargs) -> ReorderedNormTransformerLayerConfig:
        return ReorderedNormTransformerLayerConfig(**kwargs)

    @jax.named_scope("olmax.nn.ReorderedTransformerLayer")
    def forward(self, x: Array) -> Array:
        assert x.ndim == 3
        h = x + self.attention_norm(self.attention(x))
        h = h + self.mlp_norm(self.mlp(h))
        return h


class GemmaTransformerLayer(TransformerLayer):
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
        block_idx: int | None = None,
        dtype: DTypeLike = float,
        activation: Callable[[Array], Array] = jax.nn.gelu,
        mesh_resource: MeshResource | None = None,
        checkpoint_name: str | None = None,
    ):
        key, attn_input_norm_key, mlp_input_norm_key = jax.random.split(key, 3)
        super().__init__(
            d_model=d_model,
            hidden_size=hidden_size,
            key=key,
            attention=attention,
            norm=norm,
            bias=bias,
            block_idx=block_idx,
            dtype=dtype,
            activation=activation,
            mesh_resource=mesh_resource,
            checkpoint_name=checkpoint_name,
        )
        assert self.checkpoint_name is not None
        self.attention_input_norm = norm.build(
            d_model,
            attn_input_norm_key,
            mesh_resource=mesh_resource,
            checkpoint_name=f"{self.checkpoint_name}.attention_input_norm",
        )
        self.mlp_input_norm = norm.build(
            d_model,
            mlp_input_norm_key,
            mesh_resource=mesh_resource,
            checkpoint_name=f"{self.checkpoint_name}.mlp_input_norm",
        )

    @classmethod
    def Config(cls, **kwargs) -> GemmaTransformerLayerConfig:
        return GemmaTransformerLayerConfig(**kwargs)

    @jax.named_scope("olmax.nn.Gemma2TransformerLayer")
    def forward(self, x: Array) -> Array:
        assert x.ndim == 3
        h = x + self.attention_norm(self.attention(self.attention_input_norm(x)))
        h = h + self.mlp_norm(self.mlp(self.mlp_input_norm(h)))
        return h


B = TypeVar("B", bound=TransformerLayer)


@dataclass
class TransformerLayerConfig(Registrable, Generic[B]):
    attention: MultiheadSelfAttentionConfig
    norm: NormalizerConfig
    bias: bool = False
    dtype: DTypeLike = float

    @classmethod
    @abstractmethod
    def get_class(cls) -> Type[B]:
        raise NotImplementedError

    @classmethod
    def Default(cls, **kwargs) -> DefaultTransformerLayerConfig:
        return DefaultTransformerLayerConfig(**kwargs)

    @classmethod
    def ReorderedNorm(cls, **kwargs) -> ReorderedNormTransformerLayerConfig:
        return ReorderedNormTransformerLayerConfig(**kwargs)

    @classmethod
    def Gemma(cls, **kwargs) -> GemmaTransformerLayerConfig:
        return GemmaTransformerLayerConfig(**kwargs)

    def build(
        self,
        d_model: int,
        hidden_size: int,
        key: PRNGKeyArray,
        attention: MultiheadSelfAttentionConfig | None = None,
        norm: LayerNormConfig | None = None,
        block_idx: int | None = None,
        bias: bool | None = None,
        dtype: DTypeLike | None = None,
        mesh_resource: MeshResource | None = None,
        checkpoint_name: str | None = None,
    ) -> B:
        return self.get_class()(
            d_model=d_model,
            hidden_size=hidden_size,
            key=key,
            attention=attention if attention is not None else self.attention,
            norm=norm if norm is not None else self.norm,
            block_idx=block_idx,
            bias=bias if bias is not None else self.bias,
            dtype=dtype if dtype is not None else self.dtype,
            mesh_resource=mesh_resource,
            checkpoint_name=checkpoint_name,
        )


@TransformerLayerConfig.register("default")
@dataclass
class DefaultTransformerLayerConfig(TransformerLayerConfig[TransformerLayer]):
    @classmethod
    def get_class(cls) -> Type[TransformerLayer]:
        return TransformerLayer


@TransformerLayerConfig.register("reordered_norm")
@dataclass
class ReorderedNormTransformerLayerConfig(TransformerLayerConfig[ReorderedNormTransformerLayer]):
    @classmethod
    def get_class(cls) -> Type[ReorderedNormTransformerLayer]:
        return ReorderedNormTransformerLayer


@TransformerLayerConfig.register("gemma")
@dataclass
class GemmaTransformerLayerConfig(TransformerLayerConfig[GemmaTransformerLayer]):
    @classmethod
    def get_class(cls) -> Type[GemmaTransformerLayer]:
        return GemmaTransformerLayer
