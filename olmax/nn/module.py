from abc import abstractmethod
from typing import Callable

import equinox as eqx
import jax
from typing_extensions import Self


class Module(eqx.Module):
    """
    Abstract base class for ``nn`` modules. This is just an extension of :class:`equinox.Module`.
    """

    forward_batch: Callable | None = eqx.field(static=True, repr=False)

    def __init__(self):
        self.forward_batch = None
        #  self.forward_batch = lambda self_, *args, **kwargs: jax.vmap(self_.forward)(
        #      *args, **kwargs
        #  )

    def __call__(self, *args, **kwargs):
        if self.forward_batch is not None:
            return self.forward_batch(self, *args, **kwargs)
        else:
            return jax.vmap(self.forward)(*args, **kwargs)

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
