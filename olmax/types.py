import os
from typing import Any

__all__ = ["PathOrStr", "PyTree"]

PathOrStr = os.PathLike | str
PyTree = Any
