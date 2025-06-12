from __future__ import annotations

from dataclasses import dataclass

import jax

from ..distributed.parallel import MeshResource
from ..types import Array, DTypeLike, PRNGKeyArray
from .linear import Linear
from .module import Module


@dataclass
class LMHeadConfig:
    bias: bool = False
    dtype: DTypeLike = float

    def build(
        self,
        d_model: int,
        vocab_size: int,
        key: PRNGKeyArray,
        bias: bool | None = None,
        dtype: DTypeLike | None = None,
        mesh_resource: MeshResource | None = None,
        checkpoint_name: str | None = None,
    ) -> LMHead:
        return LMHead(
            d_model,
            vocab_size,
            key,
            bias=bias if bias is not None else self.bias,
            dtype=dtype if dtype is not None else self.dtype,
            mesh_resource=mesh_resource,
            checkpoint_name=checkpoint_name,
        )


class LMHead(Module):
    w_out: Linear

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        key: PRNGKeyArray,
        bias: bool = False,
        dtype: DTypeLike = float,
        mesh_resource: MeshResource | None = None,
        checkpoint_name: str | None = None,
    ):
        super().__init__(mesh_resource, checkpoint_name)
        self.w_out = Linear(
            d_model, vocab_size, key, bias=bias, dtype=dtype, mesh_resource=mesh_resource
        )

    @classmethod
    def Config(cls, **kwargs) -> LMHeadConfig:
        return LMHeadConfig(**kwargs)

    @jax.named_scope("olmax.nn.LMHead")
    def forward(self, x: Array) -> Array:
        return self.w_out(x).astype(float)
