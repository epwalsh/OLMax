from typing import Sequence

import equinox as eqx
import jax

from ..distributed.parallel import ParallelConfig
from ..types import Array, DTypeLike, PRNGKeyArray
from .functional import layer_norm, rms_norm
from .init import ones, zeros
from .module import Module


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
        if x.shape != self.shape:
            raise ValueError(
                "`LayerNorm(shape)(x)` must satisfy the invariant `shape == x.shape`.\n"
                f"Received `shape={self.shape} and `x.shape={x.shape}`. You might need "
                "to replace `layer_norm(x)` with `jax.vmap(layer_norm)(x)`.\n"
            )

        return layer_norm(x, weight=self.weight, bias=self.bias, eps=self.eps)


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
        parallel_config: ParallelConfig | None = None,
    ):
        super().__init__()
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

    @jax.named_scope("olmax.nn.RMSNorm")
    def forward(self, x: Array) -> Array:
        if x.shape != self.shape:
            raise ValueError(
                "`RMSNorm(shape)(x)` must satisfy the invariant `shape == x.shape`.\n"
                f"Received `shape={self.shape} and `x.shape={x.shape}`. You might need "
                "to replace `rms_norm(x)` with `jax.vmap(rms_norm)(x)`.\n"
            )

        return rms_norm(x, weight=self.weight, bias=self.bias, eps=self.eps)
