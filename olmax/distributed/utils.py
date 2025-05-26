from typing import Sequence, TypeVar

import jax
import jax.experimental.multihost_utils as multihost_utils
import jax.numpy as jnp

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


def get_process_world_size():
    return jax.process_count()


def get_process_rank():
    return jax.process_index()


def get_global_device_count():
    return jax.device_count()


def is_distributed():
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
