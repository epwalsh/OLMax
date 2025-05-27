from abc import abstractmethod
from dataclasses import dataclass
from typing import Sequence

import equinox as eqx
import jax
from jax.sharding import NamedSharding

from ..distributed.parallel import ParallelConfig
from ..types import Array, DTypeLike, PRNGKeyArray
from .functional import layer_norm, rms_norm
from .init import ones, zeros
from .module import Module, ModuleSharding


@dataclass
class NormSharding(ModuleSharding):
    @property
    @abstractmethod
    def weight_sharding(self) -> NamedSharding:
        raise NotImplementedError

    @property
    @abstractmethod
    def bias_sharding(self) -> NamedSharding:
        raise NotImplementedError


@dataclass
class DefaultNormSharding(NormSharding):
    @property
    def weight_sharding(self) -> NamedSharding:
        return self.global_config.get_dp_param_sharding()

    @property
    def bias_sharding(self) -> NamedSharding:
        return self.global_config.get_dp_param_sharding()


class LayerNorm(Module):
    shape: tuple[int, ...] = eqx.field(static=True)
    weight: Array | None
    bias: Array | None
    eps: float = eqx.field(static=True)

    def __init__(
        self,
        shape: int | Sequence[int],
        key: PRNGKeyArray,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        dtype: DTypeLike = float,
        sharding: NamedSharding | None = None,
    ):
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
            else ones(wkey, shape, dtype=dtype, sharding=sharding)
        )
        self.bias = (
            None
            if not (elementwise_affine and bias)
            else zeros(bkey, shape, dtype=dtype, sharding=sharding)
        )

    @jax.named_scope("olmax.nn.LayerNorm")
    def forward(self, x: Array) -> Array:
        if x.shape != self.shape:
            raise ValueError(
                "`LayerNorm(shape)(x)` must satisfy the invariant `shape == x.shape`.\n"
                f"Received `shape={self.shape} and `x.shape={x.shape}`. You might need "
                "to replace `layer_norm(x)` with `jax.vmap(layer_norm)(x)`.\n"
            )

        return layer_norm(x, weight=self.weight, bias=self.bias, eps=self.eps)

    @classmethod
    def DefaultSharding(cls, global_config: ParallelConfig) -> DefaultNormSharding:
        return DefaultNormSharding(global_config)


class RMSNorm(Module):
    shape: tuple[int, ...] = eqx.field(static=True)
    weight: Array | None
    bias: Array | None
    eps: float = eqx.field(static=True)

    def __init__(
        self,
        shape: int | Sequence[int],
        key: PRNGKeyArray,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        dtype: DTypeLike = float,
        sharding: NamedSharding | None = None,
    ):
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
            else ones(wkey, shape, dtype=dtype, sharding=sharding)
        )
        self.bias = (
            None
            if not (elementwise_affine and bias)
            else zeros(bkey, shape, dtype=dtype, sharding=sharding)
        )

    @jax.named_scope("olmax.nn.RMSNorm")
    def forward(self, x: Array) -> Array:
        if x.shape != self.shape:
            raise ValueError(
                "`RMSNorm(shape)(x)` must satisfy the invariant `shape == x.shape`.\n"
                f"Received `shape={self.shape} and `x.shape={x.shape}`. You might need "
                "to replace `rms_norm(x)` with `jax.vmap(rms_norm)(x)`.\n"
            )

        return rms_norm(x, weight=self.weight, bias=self.bias, eps=self.eps)

    @classmethod
    def DefaultSharding(cls, global_config: ParallelConfig) -> DefaultNormSharding:
        return DefaultNormSharding(global_config)
