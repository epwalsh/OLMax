from abc import abstractmethod

import equinox as eqx
import jax
from typing_extensions import Self

from ..distributed.parallel import ParallelConfig
from ..types import Array, PyTree


class Module(eqx.Module):
    """
    Abstract base class for ``nn`` modules. This is just an extension of :class:`equinox.Module`.
    """

    parallel_config: ParallelConfig | None = eqx.field(static=True, repr=False)

    def __init__(self, parallel_config: ParallelConfig | None = None):
        self.parallel_config = parallel_config

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
        if self.parallel_config is None:
            return None
        else:
            return jax.tree.map(lambda a: a.sharding.spec, self)
