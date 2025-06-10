import os
from enum import StrEnum
from typing import Any, ClassVar, Protocol

from jaxtyping import Array, ArrayLike, DTypeLike, PRNGKeyArray

__all__ = [
    "Array",
    "ArrayLike",
    "PRNGKeyArray",
    "DTypeLike",
    "PyTree",
    "Specs",
    "Scalar",
    "GPUArchitecture",
    "GPUType",
    "PathOrStr",
    "Dataclass",
]

PyTree = Any
Specs = Any
Scalar = float | int | Array
PathOrStr = os.PathLike | str


class Dataclass(Protocol):
    __dataclass_fields__: ClassVar[dict[str, Any]]


class GPUArchitecture(StrEnum):
    blackwell = "blackwell"
    hopper = "hopper"
    ampere = "ampere"


class GPUType(StrEnum):
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
