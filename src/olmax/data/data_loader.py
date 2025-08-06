from abc import abstractmethod
from collections.abc import Iterable
from typing import Generic, Sequence, TypeVar

from ..types import *

B = TypeVar("B")
S = TypeVar("S")


class DataLoader(Generic[B, S], Iterable[Sequence[B]]):
    """
    An abstract base class for data loaders used by the :class:`~olmax.train.Trainer`.
    """

    def __len__(self) -> int:
        """
        Should returns the total number of batches in an epoch if known, otherwise a :class:`TypeError` is raised.
        """
        raise TypeError(
            f"total length (number of batches) is unknown for {self.__class__.__name__}"
        )

    @abstractmethod
    def get_state(self) -> S:
        """
        Get state for checkpointing.
        """
        raise NotImplementedError

    @abstractmethod
    def load_state(self, state: S):
        """
        Load state from :meth:`get_state()` to restore the data loader's state.
        """
        raise NotImplementedError
