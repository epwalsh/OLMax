from abc import abstractmethod
from dataclasses import dataclass

import equinox as eqx
from typing_extensions import Self

from ..distributed.parallel import ParallelConfig


@dataclass
class ModuleSharding:
    """
    Defines how a module should be sharded.
    """

    global_config: ParallelConfig


class Module(eqx.Module):
    """
    Abstract base class for ``nn`` modules. This is just an extension of :class:`equinox.Module`.
    """

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

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
