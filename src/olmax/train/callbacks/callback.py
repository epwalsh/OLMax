import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

from dataclass_extensions import Registrable

from ...types import *

if TYPE_CHECKING:
    from ..trainer import Trainer, TrainState


@dataclass
class Callback(Registrable):
    """
    Trainer callback base class.

    Callbacks can be used to modify and extend the behavior of the trainer loop.
    This module contains a number of useful :class:`Callback` implementations, but you can
    always add your own.
    """

    priority: ClassVar[int] = 0
    """
    Priority of the callback. Determines the order in which callbacks run relative to each other.
    The higher the priority, the earlier a callback runs.
    """

    enabled: bool = True
    """
    Set to false to disable the callback.
    """

    _trainer: Any = dataclasses.field(repr=False, default=None)

    @property
    def trainer(self) -> "Trainer":
        """
        The trainer that the callback is attached to.
        """
        assert self._trainer is not None
        return self._trainer

    @trainer.setter
    def trainer(self, trainer: "Trainer"):
        self._trainer = trainer

    @property
    def train_state(self) -> "TrainState":
        """
        The current train state. This can only be called during :meth:`Trainer.fit()`, otherwise
        a runtime error is raised.
        """
        return self.trainer.state

    @property
    def step(self) -> int:
        """
        The current training step.
        """
        return self.trainer.step

    def get_state(self) -> Any:
        """
        Get the state to save for checkpointing.
        """
        return {}

    def load_state(self, state: Any):
        """
        Load state from :meth:`get_state()`.
        """
        del state

    def post_attach(self):
        """
        Called right after the callback is attached to the :class:`~olmo_core.train.Trainer`.
        """
        pass

    def pre_train(self):
        """
        Runs before the training loop starts.
        """
        pass

    def pre_epoch(self):
        """
        Runs before the start of a new epoch.
        """
        pass

    def pre_step(self):
        """
        Runs before anything else in a train step.
        """
        pass

    def pre_load_batch(self):
        """
        Runs right before the next batch is fetched from the data loader.
        """
        pass

    def post_load_batch(self, batch: Any):
        """
        Runs right after a training batch is fetched from the data loader.
        """
        del batch

    def post_train_batch(self):
        """
        Runs after a training batch is processed.
        """
        pass

    def post_step(self):
        """
        Runs after a complete step (potentially including evals and checkpointing).
        """
        pass

    def post_checkpoint_saved(self, path: PathOrStr):
        """
        Called when a checkpoint is successfully saved.

        :param path: The path/URL to the checkpoint.
        """
        del path

    def post_checkpoint_loaded(self, path: PathOrStr):
        """
        Called when a checkpoint is successfully loaded.

        :param path: The path/URL to the checkpoint.
        """
        del path

    def log_metrics(self, step: int, metrics: dict[str, Scalar]):
        """
        Called when metrics have been gathered for a given step (possibly a previous step).
        """
        del step, metrics

    def post_epoch(self):
        """
        Runs at the end of a complete epoch.
        """
        pass

    def post_train(self):
        """
        Runs after the training loop successfully completes.
        """
        pass

    def on_error(self, exc: BaseException):
        """
        Called when the training loop exits with an error.
        """
        del exc

    def close(self):
        """
        The callback method called. The trainer will always attempt to call this.
        """
