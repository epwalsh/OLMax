from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import Generic, Sequence, Type, TypeVar

import equinox as eqx
import jax

from ..config import RegistrableConfig
from ..distributed.parallel import MeshResource
from ..types import Array, DTypeLike, PRNGKeyArray
from .functional import layer_norm, rms_norm
from .init import ones, zeros
from .module import Module


class Normalizer(Module):
    shape: tuple[int, ...] = eqx.field(static=True)
    weight: Array | None
    bias: Array | None
    eps: float = eqx.field(static=True)

    def __init__(
        self,
        shape: int | Sequence[int],
        key: PRNGKeyArray,
        *,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        dtype: DTypeLike = float,
        mesh_resource: MeshResource | None = None,
    ):
        super().__init__(mesh_resource)
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
                sharding=None if mesh_resource is None else mesh_resource.get_param_sharding(),
            )
        )
        self.bias = (
            None
            if not (elementwise_affine and bias)
            else zeros(
                bkey,
                shape,
                dtype=dtype,
                sharding=None if mesh_resource is None else mesh_resource.get_param_sharding(),
            )
        )


class LayerNorm(Normalizer):
    @classmethod
    def Config(cls, **kwargs) -> LayerNormConfig:
        return LayerNormConfig(**kwargs)

    @jax.named_scope("olmax.nn.LayerNorm")
    def __call__(self, x: Array) -> Array:
        return layer_norm(x, weight=self.weight, bias=self.bias, eps=self.eps)


class RMSNorm(Normalizer):
    @classmethod
    def Config(cls, **kwargs) -> RMSNormConfig:
        return RMSNormConfig(**kwargs)

    @jax.named_scope("olmax.nn.RMSNorm")
    def __call__(self, x: Array) -> Array:
        return rms_norm(x, weight=self.weight, bias=self.bias, eps=self.eps)


N = TypeVar("N", bound=Normalizer)


@dataclass
class NormalizerConfig(RegistrableConfig, Generic[N]):
    eps: float = 1e-5
    dtype: DTypeLike = float
    elementwise_affine: bool = True
    bias: bool = True

    @classmethod
    @abstractmethod
    def get_class(cls) -> Type[N]:
        raise NotImplementedError

    @classmethod
    def LayerNorm(cls, **kwargs) -> LayerNormConfig:
        return LayerNormConfig(**kwargs)

    @classmethod
    def RMSNorm(cls, **kwargs) -> RMSNormConfig:
        return RMSNormConfig(**kwargs)

    def build(
        self,
        shape: int | Sequence[int],
        key: PRNGKeyArray,
        *,
        eps: float | None = None,
        elementwise_affine: bool | None = None,
        bias: bool | None = None,
        dtype: DTypeLike | None = None,
        mesh_resource: MeshResource | None = None,
    ) -> N:
        return self.get_class()(
            shape,
            key,
            eps=eps if eps is not None else self.eps,
            elementwise_affine=elementwise_affine
            if elementwise_affine is not None
            else self.elementwise_affine,
            bias=bias if bias is not None else self.bias,
            dtype=dtype if dtype is not None else self.dtype,
            mesh_resource=mesh_resource,
        )


@NormalizerConfig.register("layer_norm")
@dataclass
class LayerNormConfig(NormalizerConfig[LayerNorm]):
    @classmethod
    def get_class(cls) -> Type[LayerNorm]:
        return LayerNorm


@NormalizerConfig.register("rms_norm")
@dataclass
class RMSNormConfig(NormalizerConfig[RMSNorm]):
    @classmethod
    def get_class(cls) -> Type[RMSNorm]:
        return RMSNorm
