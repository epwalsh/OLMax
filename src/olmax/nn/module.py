from __future__ import annotations

from abc import abstractmethod
from typing import Iterable

import equinox as eqx
import jax
from jax.ad_checkpoint import checkpoint_name as ckpt_name
from typing_extensions import Self

from ..distributed.parallel import MeshResource
from ..types import Array, PyTree


class Module(eqx.Module):
    """
    Abstract base class for ``nn`` modules. This is just an extension of :class:`equinox.Module`.
    """

    mesh_resource: MeshResource | None = eqx.field(static=True, repr=False)
    checkpoint_name: str | None = eqx.field(static=True)
    """A name to assign to the output the module for activation checkpointing."""

    def __init__(
        self, mesh_resource: MeshResource | None = None, checkpoint_name: str | None = None
    ):
        self.mesh_resource = mesh_resource
        self.checkpoint_name = checkpoint_name

    def __call__(self, *args, **kwargs):
        out = self.forward(*args, **kwargs)
        if self.checkpoint_name is not None:
            if eqx.is_array(out):
                out = ckpt_name(out, self.checkpoint_name)
            else:
                raise ValueError(
                    f"Expected an array for the output of module '{self.__class__.__name__}' "
                    f"with assigned checkpoint name '{self.checkpoint_name}', but got a {type(out)}. "
                    f"You'll have to override the '__call__()' method to handle assigning a checkpoint "
                    f"to this type of output."
                )

        return out

    @abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def eval(self) -> Self:
        """
        Returns a copy of the module with all inference flags set to ``True`` recursively.

        .. important::
            Unlike the equivalent in Pytorch this does not modify the module in-place!
        """
        return eqx.nn.inference_mode(self, value=True)

    def train(self) -> Self:
        """
        Returns a copy of the module with all inference flags set to ``False`` recursively.

        .. important::
            Unlike the equivalent in Pytorch this does not modify the module in-place!
        """
        return eqx.nn.inference_mode(self, value=False)

    def get_checkpoint_names(self) -> Iterable[str]:
        if self.checkpoint_name is not None:
            yield self.checkpoint_name
        for child in self.children(recurse=True):
            if child.checkpoint_name is not None:
                yield child.checkpoint_name

    def parameters(self) -> list[Array]:
        return jax.tree.flatten(self)[0]

    def children(self, recurse: bool = False) -> Iterable[Module]:
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

    def get_param_partitions(self) -> PyTree:
        if self.mesh_resource is None:
            return None
        else:
            return jax.tree.map(lambda a: a.sharding.spec, self)

    def get_param_shardings(self) -> PyTree:
        if self.mesh_resource is None:
            return None
        else:
            return jax.tree.map(lambda a: a.sharding, self)
