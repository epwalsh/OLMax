from __future__ import annotations

from dataclasses import dataclass

import jax

from ..distributed.parallel import MeshResource
from ..types import Array, DTypeLike, PRNGKeyArray
from .linear import Linear
from .module import Module
from .normalization import Normalizer, NormalizerConfig


@dataclass
class LMHeadConfig:
    norm: NormalizerConfig | None
    bias: bool = False
    dtype: DTypeLike = float

    def build(
        self,
        d_model: int,
        vocab_size: int,
        key: PRNGKeyArray,
        norm: NormalizerConfig | None = None,
        bias: bool | None = None,
        dtype: DTypeLike | None = None,
        mesh_resource: MeshResource | None = None,
    ) -> LMHead:
        return LMHead(
            d_model,
            vocab_size,
            key,
            norm=norm if norm is not None else self.norm,
            bias=bias if bias is not None else self.bias,
            dtype=dtype if dtype is not None else self.dtype,
            mesh_resource=mesh_resource,
        )


class LMHead(Module):
    norm: Normalizer | None
    w_out: Linear

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        key: PRNGKeyArray,
        norm: NormalizerConfig | None,
        bias: bool = False,
        dtype: DTypeLike = float,
        mesh_resource: MeshResource | None = None,
    ):
        super().__init__(mesh_resource)
        w_out_key, norm_key = jax.random.split(key)
        self.w_out = Linear(
            d_model, vocab_size, w_out_key, bias=bias, dtype=dtype, mesh_resource=mesh_resource
        )
        self.norm = (
            None if norm is None else norm.build(d_model, norm_key, mesh_resource=mesh_resource)
        )

    @classmethod
    def Config(cls, **kwargs) -> LMHeadConfig:
        return LMHeadConfig(**kwargs)

    @jax.named_scope("olmax.nn.LMHead")
    def __call__(self, x: Array) -> Array:
        if self.norm is not None:
            x = self.norm(x)
        return self.w_out(x).astype(float)
