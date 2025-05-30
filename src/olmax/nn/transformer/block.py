from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Type

import jax

from ...debug import inspect
from ...distributed.parallel import ParallelConfig
from ...types import Array, DTypeLike, PRNGKeyArray
from ..attention import MultiheadSelfAttention, MultiheadSelfAttentionConfig
from ..mlp import GatedMLP
from ..module import Module
from ..normalization import LayerNorm, LayerNormConfig


@dataclass
class TransformerBlockConfig:
    attention: MultiheadSelfAttentionConfig
    norm: LayerNormConfig
    bias: bool = False
    dtype: DTypeLike = float

    def build(
        self,
        d_model: int,
        hidden_size: int,
        key: PRNGKeyArray,
        attention: MultiheadSelfAttentionConfig | None = None,
        norm: LayerNormConfig | None = None,
        bias: bool | None = None,
        dtype: DTypeLike | None = None,
        parallel_config: ParallelConfig | None = None,
    ) -> TransformerBlock:
        return TransformerBlock(
            d_model,
            hidden_size,
            key,
            attention=attention if attention is not None else self.attention,
            norm=norm if norm is not None else self.norm,
            bias=bias if bias is not None else self.bias,
            dtype=dtype if dtype is not None else self.dtype,
            parallel_config=parallel_config,
        )


class TransformerBlock(Module):
    Config: ClassVar[Type[TransformerBlockConfig]] = TransformerBlockConfig
    keepdims: ClassVar[int] = -1

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
        parallel_config: ParallelConfig | None = None,
    ):
        super().__init__(parallel_config)
        mlp_key, mlp_norm_key, attention_key, attention_norm_key = jax.random.split(key, 4)
        self.mlp = GatedMLP(
            d_model,
            hidden_size,
            mlp_key,
            bias=bias,
            dtype=dtype,
            parallel_config=parallel_config,
        )
        self.mlp_norm = norm.build(
            d_model,
            mlp_norm_key,
            parallel_config=parallel_config,
        )
        self.attention = attention.build(
            d_model,
            attention_key,
            bias=bias,
            dtype=dtype,
            parallel_config=parallel_config,
        )
        self.attention_norm = norm.build(
            d_model,
            attention_norm_key,
            parallel_config=parallel_config,
        )

    @jax.named_scope("olmax.nn.TransformerBlock")
    def forward(self, x: Array) -> Array:
        assert x.ndim == 3
        h = x + self.attention(self.attention_norm(x))
        h = h + self.mlp(self.mlp_norm(h))
        return h


class ReorderedNormTransformerBlock(TransformerBlock):
    @jax.named_scope("olmax.nn.ReorderedTransformerBlock")
    def forward(self, x: Array) -> Array:
        assert x.ndim == 3
        h = x + self.attention_norm(self.attention(x))
        h = h + self.mlp_norm(self.mlp(h))
        return h
