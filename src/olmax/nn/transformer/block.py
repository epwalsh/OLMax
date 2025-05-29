from typing import ClassVar

import jax

from ...distributed.parallel import ParallelConfig
from ...types import Array, DTypeLike, PRNGKeyArray
from ..attention import MultiheadSelfAttention, MultiheadSelfAttentionConfig
from ..mlp import GatedMLP
from ..module import Module
from ..normalization import LayerNorm, LayerNormConfig


class TransformerBlock(Module):
    keepdims: ClassVar[int] = 2

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
        self.mlp = GatedMLP(
            d_model, hidden_size, key, bias=bias, dtype=dtype, parallel_config=parallel_config
        )
        self.mlp_norm = norm.build(
            d_model, key, bias=bias, dtype=dtype, parallel_config=parallel_config
        )
        self.attention = attention.build(
            d_model=d_model, key=key, bias=bias, dtype=dtype, parallel_config=parallel_config
        )
        self.attention_norm = norm.build(
            d_model, key, bias=bias, dtype=dtype, parallel_config=parallel_config
        )

    @jax.named_scope("olmax.nn.TransformerBlock")
    def forward(self, x: Array) -> Array:
        assert x.ndim == 2
        h = x + self.attention.forward(jax.vmap(self.attention_norm.forward)(x))
        h = h + jax.vmap(self.mlp.forward)(jax.vmap(self.mlp_norm.forward)(h))
        return h


class ReorderedNormTransformerBlock(TransformerBlock):
    @jax.named_scope("olmax.nn.ReorderedTransformerBlock")
    def forward(self, x: Array) -> Array:
        assert x.ndim == 2
        h = x + jax.vmap(self.attention_norm.forward)(self.attention.forward(x))
        h = h + jax.vmap(self.mlp_norm.forward)(jax.vmap(self.mlp.forward)(h))
        return h
