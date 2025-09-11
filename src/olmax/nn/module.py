import functools as ft
import typing
from abc import abstractmethod
from typing import Iterable, Type, TypeVar

import equinox as eqx
import jax
from jax.ad_checkpoint import checkpoint_name as ckpt_name
from typing_extensions import Self

from ..activation_checkpointing import ActivationCheckpointingPolicy
from ..distributed.parallel import MeshResource
from ..types import Array, PyTree

M = TypeVar("M", bound="Module")


class Module(eqx.Module):
    """
    Abstract base class for ``nn`` modules. This is just an extension of :class:`equinox.Module`.
    """

    mesh_resource: MeshResource | None = eqx.field(static=True, repr=False)
    checkpoint_name: str | None = eqx.field(static=True)
    """A name to assign to the output the module for activation checkpointing."""
    inference_mode: bool = eqx.field(static=True)

    def __init__(
        self,
        mesh_resource: MeshResource | None = None,
        checkpoint_name: str | None = None,
        inference_mode: bool = False,
    ):
        self.mesh_resource = mesh_resource
        self.checkpoint_name = checkpoint_name
        self.inference_mode = inference_mode

    @property
    def training(self) -> bool:
        """
        Whether the module is in training mode or not. This is determined by checking the
        ``inference_mode`` flag of the module and all its children recursively.
        """
        return not self.inference_mode

    def __call__(self, *args, **kwargs):
        out = self.forward(*args, **kwargs)
        if self.checkpoint_name is not None:
            out = jax.tree.map(
                ft.partial(ckpt_name, name=self.checkpoint_name), out, is_leaf=eqx.is_array
            )
        return out

    @classmethod
    def inject_ac_policy(cls: Type[M], policy: ActivationCheckpointingPolicy) -> Type[M]:
        """
        Create a new subclass with the given activation checkpointing policy applied.
        """

        def remat_call(self_, *args, **kwargs):
            return policy.wrap(cls.__call__)(self_, *args, **kwargs)

        return typing.cast(Type[M], type(f"Remat{cls.__name__}", (cls,), {"__call__": remat_call}))

    @abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def eval(self) -> Self:
        """
        Returns a copy of the module with all inference flags set to ``True`` recursively.

        .. important::
            Unlike the equivalent in Pytorch this does not modify the module in-place!
        """
        return self.train(False)

    def train(self, mode: bool = True) -> Self:
        """
        Returns a copy of the module with all inference flags set to ``False`` recursively.

        .. important::
            Unlike the equivalent in Pytorch this does not modify the module in-place!
        """
        out = jax.tree.map(lambda x: x, self)
        for m in out.modules():
            object.__setattr__(m, "inference_mode", not mode)
        return out

    def get_checkpoint_names(self) -> Iterable[str]:
        """
        Get all registered activation checkpointing names, recursively.
        """
        if self.checkpoint_name is not None:
            yield self.checkpoint_name
        for child in self.children(recurse=True):
            if child.checkpoint_name is not None:
                yield child.checkpoint_name

    def parameters(self) -> list[Array]:
        """
        Get all parameters, recursively.
        """
        return [x for x in jax.tree.flatten(self)[0] if eqx.is_array(x)]

    def children(self, recurse: bool = False) -> Iterable["Module"]:
        """
        Returns an iterator of children modules.
        """

        def flatten_children(root) -> Iterable[Module]:
            for v in eqx.tree_flatten_one_level(root)[0]:
                if isinstance(v, Module):
                    yield v
                if recurse and not eqx.is_array_like(v):
                    yield from flatten_children(v)

        yield from flatten_children(self)

    def modules(self) -> Iterable["Module"]:
        yield self
        for child in self.children(recurse=True):
            yield child

    def get_param_partitions(self) -> PyTree:
        if self.mesh_resource is None:
            return None
        else:
            return jax.tree.map(
                lambda a: a.sharding.spec
                if eqx.is_array(a)
                else self.mesh_resource.get_replicated_partition(),
                self,
            )

    def get_param_shardings(self) -> PyTree:
        if self.mesh_resource is None:
            return None
        else:
            return jax.tree.map(
                lambda a: a.sharding
                if eqx.is_array(a)
                else self.mesh_resource.get_replicated_sharding(),
                self,
            )
