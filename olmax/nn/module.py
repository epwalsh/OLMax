from abc import abstractmethod
from typing import Callable

import equinox as eqx
import jax
from typing_extensions import Self

from ..distributed.parallel import ParallelConfig


class Module(eqx.Module):
    """
    Abstract base class for ``nn`` modules. This is just an extension of :class:`equinox.Module`.
    """

    parallel_config: ParallelConfig | None = eqx.field(static=True, repr=False)
    forward_batch: Callable | None = eqx.field(static=True, repr=False)

    def __init__(self, parallel_config: ParallelConfig | None = None):
        self.parallel_config = parallel_config
        self.forward_batch = None

    def __call__(self, *args, **kwargs):
        if self.forward_batch is not None:
            return self.forward_batch(self, *args, **kwargs)
        elif (
            self.parallel_config is not None
            and (dp_sharding := self.parallel_config.get_data_sharding()) is not None
        ):
            # Default data-parallel implementation (like FSDP or DDP)
            args = jax.lax.with_sharding_constraint(args, dp_sharding)
            kwargs = jax.lax.with_sharding_constraint(kwargs, dp_sharding)
            out = jax.vmap(self.forward)(*args, **kwargs)
            out = jax.lax.with_sharding_constraint(out, dp_sharding)
            return out
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
