from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Type

import jax

from ..distributed.parallel import ParallelConfig
from ..types import Array, DTypeLike, PRNGKeyArray
from .linear import Linear
from .module import Module
from .normalization import LayerNorm, LayerNormConfig


@dataclass
class LMHeadConfig:
    norm: LayerNormConfig | None
    bias: bool = False
    dtype: DTypeLike = float

    def build(
        self,
        d_model: int,
        vocab_size: int,
        key: PRNGKeyArray,
        norm: LayerNormConfig | None = None,
        bias: bool | None = None,
        dtype: DTypeLike | None = None,
        parallel_config: ParallelConfig | None = None,
    ) -> LMHead:
        return LMHead(
            d_model,
            vocab_size,
            key,
            norm=norm if norm is not None else self.norm,
            bias=bias if bias is not None else self.bias,
            dtype=dtype if dtype is not None else self.dtype,
            parallel_config=parallel_config,
        )


class LMHead(Module):
    Config: ClassVar[Type[LMHeadConfig]] = LMHeadConfig

    norm: LayerNorm | None
    w_out: Linear

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        key: PRNGKeyArray,
        norm: LayerNormConfig | None,
        bias: bool = False,
        dtype: DTypeLike = float,
        parallel_config: ParallelConfig | None = None,
    ):
        super().__init__(parallel_config)
        w_out_key, norm_key = jax.random.split(key)
        self.w_out = Linear(
            d_model, vocab_size, w_out_key, bias=bias, dtype=dtype, parallel_config=parallel_config
        )
        self.norm = (
            None if norm is None else norm.build(d_model, norm_key, parallel_config=parallel_config)
        )

    @jax.named_scope("olmax.nn.LMHead")
    def forward(self, x: Array) -> Array:
        if self.norm is not None:
            x = self.norm.forward(x)
        return self.w_out.forward(x)
