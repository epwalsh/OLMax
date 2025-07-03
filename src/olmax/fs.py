import re

from .types import PathOrStr


def normalize_path(path: PathOrStr) -> str:
    """
    Normalize a path/URL.

    :param path: The path/URL to normalize.
    """
    return str(path).rstrip("/").replace("file://", "")


def is_url(path: PathOrStr) -> bool:
    """
    Check if a path is a URL.

    :param path: Path-like object to check.
    """
    path = normalize_path(path)
    return re.match(r"[a-z0-9]+://.*", str(path)) is not None


def copy_file(source: PathOrStr, target: PathOrStr):
    """
    Copy a file from ``source`` to ``target``.
    """
    del source, target
    raise NotImplementedError


def copy_dir(source: PathOrStr, target: PathOrStr):
    """
    Copy a directory from ``source`` to ``target``.
    """
    del source, target
    raise NotImplementedError
