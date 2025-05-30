from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar, Sequence, Type

import equinox as eqx
import jax
from typing_extensions import Self

from ..distributed.parallel import ParallelConfig
from ..types import Array, DTypeLike, PRNGKeyArray
from .functional import layer_norm, rms_norm
from .init import ones, zeros
from .module import Module


class LayerNormType(StrEnum):
    default = "default"
    rms = "rms"

    def get_class(self) -> Type[LayerNorm]:
        if self == self.default:
            return LayerNorm
        elif self == self.rms:
            return RMSNorm
        else:
            raise ValueError(self)


@dataclass
class LayerNormConfig:
    name: LayerNormType = LayerNormType.default
    eps: float = 1e-5
    dtype: DTypeLike = float
    elementwise_affine: bool = True
    bias: bool = True

    @classmethod
    def rms_norm(cls, **kwargs) -> Self:
        return cls(name=LayerNormType.rms, **kwargs)

    def build(
        self,
        shape: int | Sequence[int],
        key: PRNGKeyArray,
        *,
        eps: float | None = None,
        elementwise_affine: bool | None = None,
        bias: bool | None = None,
        dtype: DTypeLike | None = None,
        parallel_config: ParallelConfig | None = None,
    ) -> LayerNorm:
        return self.name.get_class()(
            shape,
            key,
            eps=eps if eps is not None else self.eps,
            elementwise_affine=elementwise_affine
            if elementwise_affine is not None
            else self.elementwise_affine,
            bias=bias if bias is not None else self.bias,
            dtype=dtype if dtype is not None else self.dtype,
            parallel_config=parallel_config,
        )


class LayerNorm(Module):
    keepdims: ClassVar[int] = 1
    Config: ClassVar[Type[LayerNormConfig]] = LayerNormConfig

    shape: tuple[int, ...] = eqx.field(static=True)
    weight: Array | None
    bias: Array | None
    eps: float = eqx.field(static=True)

    def __init__(
        self,
        shape: int | Sequence[int],
        key: PRNGKeyArray,
        *,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        dtype: DTypeLike = float,
        parallel_config: ParallelConfig | None = None,
    ):
        super().__init__(parallel_config)
        if isinstance(shape, int):
            shape = (shape,)
        else:
            shape = tuple(shape)

        self.eps = eps
        self.shape = shape

        wkey, bkey = jax.random.split(key)
        self.weight = (
            None
            if not elementwise_affine
            else ones(
                wkey,
                shape,
                dtype=dtype,
                sharding=None if parallel_config is None else parallel_config.get_param_sharding(),
            )
        )
        self.bias = (
            None
            if not (elementwise_affine and bias)
            else zeros(
                bkey,
                shape,
                dtype=dtype,
                sharding=None if parallel_config is None else parallel_config.get_param_sharding(),
            )
        )

    @jax.named_scope("olmax.nn.LayerNorm")
    def forward(self, x: Array) -> Array:
        return layer_norm(x, weight=self.weight, bias=self.bias, eps=self.eps)


class RMSNorm(LayerNorm):
    @jax.named_scope("olmax.nn.RMSNorm")
    def forward(self, x: Array) -> Array:
        return rms_norm(x, weight=self.weight, bias=self.bias, eps=self.eps)
