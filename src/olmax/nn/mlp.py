from typing import Callable

import equinox as eqx
import jax

from ..distributed.parallel import MeshResource, TPStyle
from ..jax_utils import vmap_multiple
from ..types import Array, DTypeLike, PRNGKeyArray
from .linear import Linear
from .module import Module


class GatedMLP(Module):
    gate_proj: Linear
    down_proj: Linear
    up_proj: Linear
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
        gate_proj_key, down_proj_key, up_proj_key = jax.random.split(key, 3)
        self.gate_proj = Linear(
            d_model,
            hidden_size,
            gate_proj_key,
            bias=bias,
            dtype=dtype,
            mesh_resource=mesh_resource,
            checkpoint_name=None if checkpoint_name is None else f"{checkpoint_name}.gate_proj",
            tp_style=TPStyle.colwise if tp_enabled else None,
        )
        self.down_proj = Linear(
            hidden_size,
            d_model,
            down_proj_key,
            bias=bias,
            dtype=dtype,
            mesh_resource=mesh_resource,
            checkpoint_name=None if checkpoint_name is None else f"{checkpoint_name}.down_proj",
            tp_style=TPStyle.rowwise if tp_enabled else None,
        )
        self.up_proj = Linear(
            d_model,
            hidden_size,
            up_proj_key,
            bias=bias,
            dtype=dtype,
            mesh_resource=mesh_resource,
            checkpoint_name=None if checkpoint_name is None else f"{checkpoint_name}.up_proj",
            tp_style=TPStyle.colwise if tp_enabled else None,
        )
        self.activation = activation

    @jax.named_scope("olmax.nn.GatedMLP")
    def forward(self, x: Array) -> Array:
        return self.down_proj(
            vmap_multiple(self.activation, x.ndim - 1)(self.gate_proj(x)) * self.up_proj(x),
        )
