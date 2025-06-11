from typing import Callable

import equinox as eqx
import jax

from ..distributed.parallel import MeshResource, TPStyle
from ..jax_utils import vmap_multiple
from ..types import Array, DTypeLike, PRNGKeyArray
from .linear import Linear
from .module import Module


class GatedMLP(Module):
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
        mesh_resource: MeshResource | None = None,
        checkpoint_name: str | None = None,
    ):
        super().__init__(mesh_resource, checkpoint_name)
        tp_enabled = mesh_resource is not None and mesh_resource.tp is not None
        if tp_enabled and bias:
            raise ValueError(
                f"bias=True is not allowed with tensor parallelism in {self.__class__.__name__}"
            )
        w1_key, w2_key, w3_key = jax.random.split(key, 3)
        self.w1 = Linear(
            d_model,
            hidden_size,
            w1_key,
            bias=bias,
            dtype=dtype,
            mesh_resource=mesh_resource,
            checkpoint_name=None if checkpoint_name is None else f"{checkpoint_name}.w1",
            tp_style=TPStyle.colwise if tp_enabled else None,
        )
        self.w2 = Linear(
            hidden_size,
            d_model,
            w2_key,
            bias=bias,
            dtype=dtype,
            mesh_resource=mesh_resource,
            checkpoint_name=None if checkpoint_name is None else f"{checkpoint_name}.w2",
            tp_style=TPStyle.rowwise if tp_enabled else None,
        )
        self.w3 = Linear(
            d_model,
            hidden_size,
            w3_key,
            bias=bias,
            dtype=dtype,
            mesh_resource=mesh_resource,
            checkpoint_name=None if checkpoint_name is None else f"{checkpoint_name}.w3",
            tp_style=TPStyle.colwise if tp_enabled else None,
        )
        self.activation = activation

    @jax.named_scope("olmax.nn.GatedMLP")
    def forward(self, x: Array) -> Array:
        return self.w2(
            vmap_multiple(self.activation, x.ndim - 1)(self.w1(x)) * self.w3(x),
        )
