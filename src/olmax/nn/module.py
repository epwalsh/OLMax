from abc import abstractmethod

import equinox as eqx
import jax
from typing_extensions import Self

from ..distributed.parallel import MeshResource
from ..types import Array, PyTree


class Module(eqx.Module):
    """
    Abstract base class for ``nn`` modules. This is just an extension of :class:`equinox.Module`.
    """

    mesh_resource: MeshResource | None = eqx.field(static=True, repr=False)

    def __init__(self, mesh_resource: MeshResource | None = None):
        self.mesh_resource = mesh_resource

    @abstractmethod
    def __call__(self, *args, **kwargs):
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

    def parameters(self) -> list[Array]:
        return jax.tree.flatten(self)[0]

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
