import os
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import optax
from dataclass_extensions import Dataclass
from jaxtyping import Array, ArrayLike, DTypeLike, PRNGKeyArray

__all__ = [
    "Array",
    "ArrayLike",
    "PRNGKeyArray",
    "DTypeLike",
    "PyTree",
    "Specs",
    "Scalar",
    "Optim",
    "OptState",
    "GPUArchitecture",
    "DeviceType",
    "PathOrStr",
    "Dataclass",
    "Duration",
    "DurationUnit",
]

PyTree = Any
Specs = Any
Scalar = float | int | Array
PathOrStr = os.PathLike | str
Optim = optax.GradientTransformation | optax.MultiSteps
OptState = optax.OptState


class GPUArchitecture(StrEnum):
    blackwell = "blackwell"
    hopper = "hopper"
    ampere = "ampere"


class DeviceType(StrEnum):
    NVIDIA_H100 = "NVIDIA_H100"
    NVIDIA_A100_80GB = "NVIDIA_A100_80GB"
    NVIDIA_A100_40GB = "NVIDIA_A100_40GB"
    NVIDIA_L4 = "NVIDIA_L4"
    NVIDIA_RTX_A5000 = "NVIDIA_RTX_A5000"
    NVIDIA_RTX_A6000 = "NVIDIA_RTX_A6000"
    NVIDIA_RTX_8000 = "NVIDIA_RTX_8000"
    NVIDIA_T4 = "NVIDIA_T4"
    NVIDIA_P100 = "NVIDIA_P100"
    NVIDIA_P4 = "NVIDIA_P4"
    NVIDIA_V100 = "NVIDIA_V100"
    NVIDIA_L40 = "NVIDIA_L40"
    NVIDIA_L40S = "NVIDIA_L40S"
    NVIDIA_B200 = "NVIDIA_B200"


class DurationUnit(StrEnum):
    """
    Units that can be used to define a :class:`Duration`.
    """

    steps = "steps"
    """
    Steps (batches).
    """
    epochs = "epochs"
    """
    Epochs.
    """
    tokens = "tokens"
    """
    Tokens.
    """


@dataclass
class Duration:
    value: int
    """
    The value of the duration.
    """
    unit: DurationUnit
    """
    The unit associated with the :data:`value`.
    """

    @classmethod
    def steps(cls, steps: int) -> "Duration":
        """
        Define a duration from a number of steps.
        """
        return cls(value=steps, unit=DurationUnit.steps)

    @classmethod
    def epochs(cls, epochs: int) -> "Duration":
        """
        Define a duration from a number of epochs.
        """
        return cls(value=epochs, unit=DurationUnit.epochs)

    @classmethod
    def tokens(cls, tokens: int) -> "Duration":
        """
        Define a duration from a number of tokens.
        """
        return cls(value=tokens, unit=DurationUnit.tokens)

    def due(self, *, step: int | None = None, tokens: int | None, epoch: int | None = None) -> bool:
        """
        Check if the duration is due.
        """
        if self.unit == DurationUnit.steps:
            assert step is not None
            return step >= self.value
        elif self.unit == DurationUnit.tokens:
            assert tokens is not None
            return tokens >= self.value
        elif self.unit == DurationUnit.epochs:
            assert epoch is not None
            return epoch > self.value
        else:
            raise NotImplementedError
