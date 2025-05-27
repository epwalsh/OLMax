import functools as ft
from abc import abstractmethod
from dataclasses import dataclass
from typing import Callable

import equinox as eqx
import jax
from jax.sharding import NamedSharding

from ..distributed.parallel import ParallelConfig
from ..types import Array, DTypeLike, PRNGKeyArray
from .functional import linear
from .init import truncated_normal
from .module import Module, ModuleSharding


@dataclass
class LinearSharding(ModuleSharding):
    @property
    def all_gather_axis(self) -> str | None:
        raise NotImplementedError

    @property
    @abstractmethod
    def weight_sharding(self) -> NamedSharding:
        raise NotImplementedError

    @property
    @abstractmethod
    def bias_sharding(self) -> NamedSharding:
        raise NotImplementedError


@dataclass
class DefaultLinearSharding(LinearSharding):
    @property
    def all_gather_axis(self) -> str | None:
        return self.global_config.get_dp_param_sharding_axis_name()

    @property
    def weight_sharding(self) -> NamedSharding:
        return self.global_config.get_dp_param_sharding()

    @property
    def bias_sharding(self) -> NamedSharding:
        return self.global_config.get_dp_param_sharding()


class Linear(Module):
    weight: Array
    bias: Array | None
    sharding: LinearSharding | None = eqx.field(static=True)
    _fwd_internal: Callable[[Array, Array, Array | None], Array] = eqx.field(
        static=True, repr=False
    )

    def __init__(
        self,
        in_size: int,
        out_size: int,
        key: PRNGKeyArray,
        bias: bool = True,
        sharding: LinearSharding | None = None,
        dtype: DTypeLike = float,
    ):
        wkey, bkey = jax.random.split(key)
        self.weight = truncated_normal(
            wkey,
            (out_size, in_size),
            sharding=None if sharding is None else sharding.weight_sharding,
            dtype=dtype,
        )
        self.bias = (
            None
            if not bias
            else truncated_normal(
                bkey,
                (out_size,),
                sharding=None if sharding is None else sharding.bias_sharding,
                dtype=dtype,
            )
        )
        self.sharding = sharding
        self._fwd_internal = linear

        if self.sharding is not None and self.sharding.all_gather_axis is not None:
            all_gather_axis: str = self.sharding.all_gather_axis

            @ft.partial(
                jax.remat,  # pyright: ignore
                policy=lambda op, *_, **__: str(op) != "all_gather",  # pyright: ignore
            )
            def forward_fsdp(x: Array, weight: Array, bias: Array | None) -> Array:
                weight = jax.lax.all_gather(weight, all_gather_axis, tiled=True)
                bias = (
                    None
                    if bias is None
                    else jax.lax.all_gather(bias, all_gather_axis, tiled=True)
                )

                return linear(x, weight, bias)

            self._fwd_internal = forward_fsdp  # type: ignore

    @jax.named_scope("olmax.nn.Linear")
    def forward(self, x: Array) -> Array:
        return self._fwd_internal(x, self.weight, self.bias)

    @classmethod
    def DefaultSharding(cls, global_config: ParallelConfig) -> DefaultLinearSharding:
        return DefaultLinearSharding(global_config)
