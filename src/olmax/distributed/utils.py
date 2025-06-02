import os
from pathlib import Path
from typing import Sequence, TypeVar

import jax
import jax.experimental.multihost_utils as multihost_utils
import jax.numpy as jnp

from .. import fs
from ..types import PathOrStr

SHARED_FS_DIRS_ENV_VAR = "OLMAX_SHARED_FS_DIRS"

_DIST_INITIALIZED = False


def init_distributed(
    coordinator_address: str | None = None,
    num_processes: int | None = None,
    process_id: int | None = None,
    local_device_ids: int | Sequence[int] | None = None,
    cluster_detection_method: str | None = None,
    initialization_timeout: int = 300,
    coordinator_bind_address: str | None = None,
    slice_index: int | None = None,
):
    global _DIST_INITIALIZED
    if _DIST_INITIALIZED:
        raise RuntimeError("a distributed backend has already been initialized!")
    jax.distributed.initialize(
        coordinator_address=coordinator_address,
        num_processes=num_processes,
        process_id=process_id,
        local_device_ids=local_device_ids,
        cluster_detection_method=cluster_detection_method,
        initialization_timeout=initialization_timeout,
        coordinator_bind_address=coordinator_bind_address,
        slice_index=slice_index,
    )
    _DIST_INITIALIZED = True


def teardown_distributed():
    global _DIST_INITIALIZED
    jax.distributed.shutdown()
    _DIST_INITIALIZED = False


def get_process_world_size() -> int:
    return jax.process_count()


def get_process_rank() -> int:
    return jax.process_index()


_PROCESS_FS_RANK_CACHE: dict[int, dict[Path, int]] = {}


def _get_process_filesystem_rank(dir: Path, process_world_size: int) -> int:
    global _PROCESS_FS_RANK_CACHE

    if (cache := _PROCESS_FS_RANK_CACHE.get(process_world_size)) is None:
        cache = {}
        _PROCESS_FS_RANK_CACHE[process_world_size] = cache

    # Direct cache hit.
    if dir in cache:
        return cache[dir]

    def cache_result(rank: int) -> int:
        assert cache is not None
        cache[dir] = rank
        for parent in dir.parents:
            cache[parent] = rank
        return rank

    # Check if any parent is already cached.
    for parent in dir.parents:
        if (rank := cache.get(parent)) is not None:
            return cache_result(rank)

    # No cache hit, check env var.
    if (shared_dirs_var := os.environ.get(SHARED_FS_DIRS_ENV_VAR)) is not None:
        shared_dirs = shared_dirs_var.split(":")
        for shared_dir_str in shared_dirs:
            shared_dir = Path(shared_dir_str).resolve()

            if dir == shared_dir:
                return cache_result(get_process_rank())

            for parent in dir.parents:
                if parent == shared_dir:
                    return cache_result(get_process_rank())

    return cache_result(0)


def get_process_filesystem_rank(dir: PathOrStr) -> int:
    f"""
    Get the rank of the current process modulo the number of processes that have write access to
    the given directory.

    This will check to see if the ``dir`` or any of its parents are listed in the environment variable
    '{SHARED_FS_DIRS_ENV_VAR}', which should be a colon-separated list of directories that are accessible
    by all processes. Otherwise this will assume that only the current process can access to the ``dir``
    and return 0 for all processes.
    """
    if fs.is_url(dir):
        raise ValueError("expected a local directory, not a URL")
    dir = Path(fs.normalize_path(dir)).resolve()
    return _get_process_filesystem_rank(dir, get_process_world_size())


def get_global_device_count() -> int:
    return jax.device_count()


def get_local_device_count() -> int:
    return jax.local_device_count()


def get_local_devices(process_rank: int | None = None) -> list[jax.Device]:
    return jax.local_devices(process_rank)


def is_distributed() -> bool:
    return get_process_world_size() > 1


def barrier(name: str):
    if is_distributed():
        multihost_utils.sync_global_devices(name)


V = TypeVar("V", bool, int, float)


def synchronize_value(value: V) -> V:
    if not is_distributed():
        return value
    arr = jnp.array(value)
    arr = multihost_utils.broadcast_one_to_all(arr)
    return type(value)(arr.item())


def get_reduce_divide_factor(axis_size: int) -> float:
    factor: int = 1
    while axis_size % factor == 0 and axis_size / factor > factor:
        factor *= 2
    return float(factor)
