from abc import abstractmethod

import equinox as eqx
from typing_extensions import Self


class Module(eqx.Module):
    """
    Abstract base class for ``nn`` modules. This is just an extension of :class:`equinox.Module`.
    """

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
