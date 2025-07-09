import threading
from dataclasses import dataclass
from typing import Any, Callable, Protocol, TypeVar

import jax
import numpy as np
import orbax.checkpoint as ocp
from jaxtyping import Array

from .. import distributed as dist
from ..types import PathOrStr, PyTree

CheckpointMetadata = Any
T = TypeVar("T")


class AsyncSaveHandle(Protocol):
    def done(self) -> bool:
        """
        Check if the checkpoint has finished saving.
        """
        raise NotImplementedError

    def wait(self):
        """
        Blocks until the checkpoint is finished saving.
        """
        raise NotImplementedError

    def close(self):
        """
        Close any resources.
        """
        raise NotImplementedError


@dataclass
class OrbaxAsyncSaveHandle:
    """
    Async save handle.
    """

    checkpointer: ocp.AsyncCheckpointer
    done_event: threading.Event

    def done(self) -> bool:
        """
        Check if the checkpoint has finished saving.
        """
        return self.done_event.is_set()

    def wait(self):
        """
        Blocks until the checkpoint is finished saving.
        """
        self.checkpointer.wait_until_finished()

    def close(self):
        """
        Close any resources.
        """
        self.checkpointer.close()


def save(
    dir: PathOrStr, state: PyTree, *, block: bool = True, force: bool = False
) -> AsyncSaveHandle | None:
    """
    Save a checkpoint from a PyTree.
    """
    done_event = threading.Event()

    def done_callback():
        done_event.set()

    checkpointer = _get_checkpointer(post_save_callback=done_callback)
    checkpointer.save(dir, state, force=force)
    save_handle = OrbaxAsyncSaveHandle(checkpointer=checkpointer, done_event=done_event)

    if block:
        save_handle.wait()
        save_handle.close()
        return None
    else:
        return save_handle


def restore(
    dir: PathOrStr,
    state: T,
    enable_single_replica_restoring: bool | None = None,
) -> T:
    """
    Restore a checkpoint saved via :func:`save()`.

    .. important::
        This does not modify the state in-place, so if you're passing a model as the ``state``
        to load, you should assign the model to the output: ``model = restore(dir, model)``.

    :param dir: The checkpoint directory to restore from.
    :param state: The state to restore.
    :param enable_single_replica_restoring: If ``True``, read the checkpoint only
        on a single replica's hosts and do broadcasting. This should significantly
        improve the loading time at scale. Only valid in a distributed environment.
    """
    if enable_single_replica_restoring is None:
        enable_single_replica_restoring = dist.is_distributed()
    elif enable_single_replica_restoring and not dist.is_distributed():
        raise ValueError(
            "'enable_single_replica_restoring=True' is only valid in a distributed environment"
        )

    restore_args = make_restore_args(
        state, enable_single_replica_restoring=enable_single_replica_restoring
    )

    with _get_checkpointer() as checkpointer:
        result = checkpointer.restore(dir, item=state, restore_args=restore_args)

    return result


def restore_from_metadata(
    dir: PathOrStr,
    metadata: CheckpointMetadata | None = None,
    enable_single_replica_restoring: bool | None = None,
) -> Any:
    """
    Restore a checkpoint saved via :func:`save()` from metadata only.

    :param dir: The checkpoint directory to restore from.
    :param metadata: The metadata to restore from.
    :param enable_single_replica_restoring: If ``True``, read the checkpoint only
        on a single replica's hosts and do broadcasting. This should significantly
        improve the loading time at scale. Only valid in a distributed environment.
    """
    if enable_single_replica_restoring is None:
        enable_single_replica_restoring = dist.is_distributed()
    elif enable_single_replica_restoring and not dist.is_distributed():
        raise ValueError(
            "'enable_single_replica_restoring=True' is only valid in a distributed environment"
        )

    with _get_checkpointer() as checkpointer:
        if metadata is None:
            metadata = checkpointer.metadata(dir)
        state = jax.tree.map(ocp.utils.to_shape_dtype_struct, metadata)
        restore_args = make_restore_args(
            state, enable_single_replica_restoring=enable_single_replica_restoring
        )
        result = checkpointer.restore(dir, item=state, restore_args=restore_args)

    return result


def get_metadata(dir: PathOrStr) -> CheckpointMetadata:
    """
    Get metadata about a checkpoint saved via :func:`save()`.
    """
    with _get_checkpointer() as checkpointer:
        return checkpointer.metadata(dir)


def _get_checkpointer(
    post_save_callback: Callable[[], None] | None = None
) -> ocp.AsyncCheckpointer:
    return ocp.AsyncCheckpointer(
        ocp.PyTreeCheckpointHandler(use_ocdbt=True, use_zarr3=True),
        async_options=ocp.options.AsyncOptions(post_finalization_callback=post_save_callback),
    )


def is_checkpoint_metadata(state: PyTree | CheckpointMetadata) -> bool:
    for leaf in jax.tree.leaves(state):
        if isinstance(leaf, ocp.metadata.Metadata):
            return True
    return False


def make_restore_args(data: T, enable_single_replica_restoring: bool = False) -> T:
    def _make_restore_args(x: Any) -> ocp.RestoreArgs | None:
        if not isinstance(x, Array):
            return None

        if not enable_single_replica_restoring:
            return ocp.type_handlers.ArrayRestoreArgs(sharding=x.sharding)

        if not isinstance(x.sharding, jax.sharding.NamedSharding):
            raise RuntimeError(
                "Restoring a checkpoint with 'enable_single_replica_restoring=True' requires all arrays to use NamedSharding"
            )

        pspec = x.sharding.spec
        mesh = x.sharding.mesh
        assert mesh.devices is not None
        replica_axis_index = 0
        replica_devices = _replica_devices(mesh.devices, replica_axis_index)
        replica_mesh = jax.sharding.Mesh(replica_devices, mesh.axis_names)
        single_replica_sharding = jax.sharding.NamedSharding(replica_mesh, pspec)

        return ocp.type_handlers.SingleReplicaArrayRestoreArgs(
            sharding=jax.sharding.NamedSharding(mesh, pspec),
            single_replica_sharding=single_replica_sharding,
            global_shape=x.shape,
            dtype=x.dtype,
        )

    return jax.tree.map(_make_restore_args, data)


def _replica_devices(device_array: np.ndarray, replica_axis_idx: int):
    """
    Returns the devices from the replica that current host belongs to.
    Replicas are assumed to be restricted to the first axis.

    :param device_array: Devices of the mesh that can be obtained by 'mesh.devices'
    :param replica_axis_idx: Axis dimension along which replica is taken

    :returns: Devices inside the replica that current host is in.
    """
    idx = _find_idx(device_array, replica_axis_idx)
    replica_result = np.take(device_array, idx, axis=replica_axis_idx)
    return np.expand_dims(replica_result, axis=replica_axis_idx)


def _find_idx(array: np.ndarray, replica_axis_idx: int):
    """Returns the index along given dimension that the current host belongs to."""
    idx = None
    for idx, val in np.ndenumerate(array):
        if val.process_index == jax.process_index():
            return idx[replica_axis_idx]
    raise RuntimeError("failed to find host index")
