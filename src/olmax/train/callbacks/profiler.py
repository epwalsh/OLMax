import dataclasses
from dataclasses import dataclass

import jax

from ...types import *
from .callback import Callback


@Callback.register("profiler")
@dataclass
class ProfilerCallback(Callback):
    skip_first: int = 3
    """
    The number of steps to skip before starting to trace.
    """
    active: int = 3
    """
    The number of steps to trace.
    """
    _step_count: int = dataclasses.field(default=0, repr=False)
    _is_active: bool = dataclasses.field(default=False, repr=False)

    @property
    def is_active(self) -> bool:
        return self._is_active

    def pre_step(self):
        self._step_count += 1
        if self._step_count == (self.skip_first + 1):
            jax.profiler.start_trace(self.trainer.work_dir, create_perfetto_trace=True)
            self._is_active = True

    def post_step(self):
        if self._step_count == (self.skip_first + 1 + self.active):
            jax.profiler.stop_trace()
            self._is_active = False

    def close(self):
        if self.is_active:
            jax.profiler.stop_trace()
            self._is_active = False
