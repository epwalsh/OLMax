import os
from enum import StrEnum
from typing import Any

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
    "PathOrStr",
]

PyTree = Any
Specs = Any
Scalar = float | int | Array
PathOrStr = os.PathLike | str


class GPUArchitecture(StrEnum):
    blackwell = "blackwell"
    hopper = "hopper"
    ampere = "ampere"
