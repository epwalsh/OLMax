from abc import abstractmethod
from dataclasses import dataclass
from typing import Callable

import jax

from ..distributed.parallel import ParallelConfig
from ..types import Array, DTypeLike, PRNGKeyArray
from .linear import DefaultLinearSharding, Linear, LinearSharding
from .module import Module, ModuleSharding


@dataclass
class GatedMLPSharding(ModuleSharding):
    @property
    @abstractmethod
    def w1_sharding(self) -> LinearSharding:
        raise NotImplementedError

    @property
    @abstractmethod
    def w2_sharding(self) -> LinearSharding:
        raise NotImplementedError

    @property
    @abstractmethod
    def w3_sharding(self) -> LinearSharding:
        raise NotImplementedError


@dataclass
class DefaultGatedMLPSharding(ModuleSharding):
    @property
    def w1_sharding(self) -> LinearSharding:
        return DefaultLinearSharding(self.global_config)

    @property
    def w2_sharding(self) -> LinearSharding:
        return DefaultLinearSharding(self.global_config)

    @property
    def w3_sharding(self) -> LinearSharding:
        return DefaultLinearSharding(self.global_config)


class GatedMLP(Module):
    w1: Linear
    w2: Linear
    w3: Linear
    activation: Callable[[Array], Array]

    def __init__(
        self,
        d_model: int,
        hidden_size: int,
        key: PRNGKeyArray,
        activation: Callable[[Array], Array] = jax.nn.silu,
        bias: bool = True,
        dtype: DTypeLike = float,
        sharding: GatedMLPSharding | None = None,
    ):
        self.w1 = Linear(
            d_model,
            hidden_size,
            key,
            bias=bias,
            dtype=dtype,
            sharding=None if sharding is None else sharding.w1_sharding,
        )
        self.w2 = Linear(
            hidden_size,
            d_model,
            key,
            bias=bias,
            dtype=dtype,
            sharding=None if sharding is None else sharding.w2_sharding,
        )
        self.w3 = Linear(
            d_model,
            hidden_size,
            key,
            bias=bias,
            dtype=dtype,
            sharding=None if sharding is None else sharding.w3_sharding,
        )
        self.activation = activation

    @jax.named_scope("olmax.nn.GatedMLP")
    def forward(self, x: Array) -> Array:
        return self.w2(self.activation(self.w1(x)) * self.w3(x))

    @classmethod
    def DefaultSharding(cls, global_config: ParallelConfig) -> DefaultGatedMLPSharding:
        return DefaultGatedMLPSharding(global_config)
