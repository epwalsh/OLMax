import os
from typing import Any

from jaxtyping import Array, ArrayLike, DTypeLike, PRNGKeyArray

__all__ = [
    "Array",
    "ArrayLike",
    "PRNGKeyArray",
    "DTypeLike",
    "PyTree",
    "PathOrStr",
]

PathOrStr = os.PathLike | str
PyTree = Any
