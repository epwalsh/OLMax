import os
from typing import Any

from jaxtyping import Array, ArrayLike, DTypeLike, PRNGKeyArray

__all__ = [
    "Array",
    "ArrayLike",
    "PRNGKeyArray",
    "DTypeLike",
    "PyTree",
    "Specs",
    "PathOrStr",
]

PyTree = Any
Specs = Any
PathOrStr = os.PathLike | str
