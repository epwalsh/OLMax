from typing import Callable, ClassVar

import equinox as eqx
import jax

from ..debug import inspect
from ..distributed.parallel import ParallelConfig, TPStyle
from ..types import Array, DTypeLike, PRNGKeyArray
from .linear import Linear
from .module import Module


class GatedMLP(Module):
    keepdims: ClassVar[int] = -1

    w1: Linear
    w2: Linear
    w3: Linear
    activation: Callable[[Array], Array] = eqx.field(static=True)

    def __init__(
        self,
        d_model: int,
        hidden_size: int,
        key: PRNGKeyArray,
        activation: Callable[[Array], Array] = jax.nn.silu,
        bias: bool = True,
        dtype: DTypeLike = float,
        parallel_config: ParallelConfig | None = None,
    ):
        super().__init__(parallel_config)
        tp_enabled = parallel_config is not None and parallel_config.tp is not None
        w1_key, w2_key, w3_key = jax.random.split(key, 3)
        self.w1 = Linear(
            d_model,
            hidden_size,
            w1_key,
            bias=bias,
            dtype=dtype,
            parallel_config=parallel_config,
            tp_style=TPStyle.colwise if tp_enabled else None,
        )
        self.w2 = Linear(
            hidden_size,
            d_model,
            w2_key,
            bias=bias,
            dtype=dtype,
            parallel_config=parallel_config,
            tp_style=TPStyle.rowwise if tp_enabled else None,
        )
        self.w3 = Linear(
            d_model,
            hidden_size,
            w3_key,
            bias=bias,
            dtype=dtype,
            parallel_config=parallel_config,
            tp_style=TPStyle.colwise if tp_enabled else None,
        )
        self.activation = activation

    @jax.named_scope("olmax.nn.GatedMLP")
    def forward(self, x: Array) -> Array:
        h = jax.vmap(self.activation)(self.w1(x)) * self.w3(x)
        inspect(h, "h1")
        h = self.w2(h)
        inspect(h, "h2")
        return h
