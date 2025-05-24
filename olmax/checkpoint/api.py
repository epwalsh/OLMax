from dataclasses import dataclass
from typing import Any

import jax
import orbax.checkpoint as ocp

from ..types import PathOrStr, PyTree

CheckpointMetadata = Any


@dataclass
class AsyncSaveHandle:
    """
    Async save handle.
    """

    checkpointer: ocp.AsyncCheckpointer

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

    def __del__(self):
        self.close()


def save(
    dir: PathOrStr, state: PyTree, *, block: bool = True, force: bool = False
) -> AsyncSaveHandle | None:
    """
    Save a checkpoint from a PyTree.
    """
    checkpointer = _get_checkpointer()
    checkpointer.save(dir, state, force=force)
    save_handle = AsyncSaveHandle(checkpointer)
    if block:
        save_handle.wait()
        save_handle.close()
    else:
        return save_handle


def restore(dir: PathOrStr, state: PyTree | CheckpointMetadata) -> PyTree:
    """
    Restore a checkpoint saved via :func:`save()`.

    .. important::
        This does not modify the state in-place, so if you're passing a model as the ``state``
        to load, you should assign the model to the output: ``model = restore(dir, model)``.
    """
    checkpointer = _get_checkpointer()
    if _is_checkpoint_metadata(state):
        state = jax.tree_util.tree_map(ocp.utils.to_shape_dtype_struct, state)
    result = checkpointer.restore(dir, state)
    checkpointer.close()
    return result


def get_metadata(dir: PathOrStr) -> CheckpointMetadata:
    """
    Get metadata about a checkpoint saved via :func:`save()`.
    """
    checkpointer = _get_checkpointer()
    result = checkpointer.metadata(dir)
    checkpointer.close()
    return result


def _get_checkpointer() -> ocp.AsyncCheckpointer:
    return ocp.StandardCheckpointer()


def _is_checkpoint_metadata(state: PyTree | CheckpointMetadata) -> bool:
    for leaf in jax.tree.leaves(state):
        if isinstance(leaf, ocp.metadata.Metadata):
            return True
    return False
