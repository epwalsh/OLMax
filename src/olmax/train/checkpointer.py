import dataclasses
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from .. import checkpoint, fs
from ..types import *

if TYPE_CHECKING:
    from .trainer import TrainState


class Checkpointer(ABC):
    @abstractmethod
    def save(self, dir: PathOrStr, state: "TrainState", save_overwrite: bool = False):
        """
        Save train state to the ``dir``.
        """
        raise NotImplementedError

    @abstractmethod
    def load(self, dir: PathOrStr, state: "TrainState") -> "TrainState":
        """
        Load train state from the ``dir``.
        """
        raise NotImplementedError


class SimpleCheckpointer(Checkpointer):
    def save(self, dir: PathOrStr, state: "TrainState", save_overwrite: bool = False):
        # TODO: handle remote dirs
        # TODO: save to temp folder first to avoid corrupted checkpoints
        if fs.is_url(dir):
            raise NotImplementedError
        checkpoint.save(dir, self._get_state_dict(state), block=True, force=save_overwrite)

    def load(self, dir: PathOrStr, state: "TrainState") -> "TrainState":
        # TODO: handle remote dirs
        if fs.is_url(dir):
            raise NotImplementedError
        state_dict = checkpoint.restore(dir, self._get_state_dict(state))
        state.data_loader.load_state(state_dict.pop("data_loader"))
        return dataclasses.replace(state, **state_dict)

    def _get_state_dict(self, state: "TrainState") -> dict[str, Any]:
        return {
            "step": state.step,
            "epoch": state.epoch,
            "global_train_tokens_seen": state.global_train_tokens_seen,
            "model": state.model,
            "opt_state": state.opt_state,
            "data_loader": state.data_loader.get_state(),
        }
