from typing import Callable

import jax

from ..distributed.parallel import ParallelConfig
from ..types import Array, DTypeLike, PRNGKeyArray
from .linear import Linear
from .module import Module


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
        parallel_config: ParallelConfig | None = None,
    ):
        super().__init__()
        self.w1 = Linear(
            d_model,
            hidden_size,
            key,
            bias=bias,
            dtype=dtype,
            parallel_config=parallel_config,  # TODO: handle TP
        )
        self.w2 = Linear(
            hidden_size,
            d_model,
            key,
            bias=bias,
            dtype=dtype,
            parallel_config=parallel_config,  # TODO: handle TP
        )
        self.w3 = Linear(
            d_model,
            hidden_size,
            key,
            bias=bias,
            dtype=dtype,
            parallel_config=parallel_config,  # TODO: handle TP
        )
        self.activation = activation

    @jax.named_scope("olmax.nn.GatedMLP")
    def forward(self, x: Array) -> Array:
        return self.w2(self.activation(self.w1(x)) * self.w3(x))
