import dataclasses
import hashlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

import jax
import numpy as np
import orbax.checkpoint as ocp

from .. import distributed as dist
from .. import fs, jax_utils
from ..types import *

if TYPE_CHECKING:
    from .trainer import TrainState


class Checkpointer(ABC):
    def __init__(self):
        self._work_dir: Path | None = None

    @abstractmethod
    def save(self, dir: PathOrStr, state: "TrainState", save_overwrite: bool = False):
        """
        Save train state to the ``dir``.
        """
        raise NotImplementedError

    @abstractmethod
    def load(self, dir: PathOrStr, state: "TrainState") -> "TrainState":
        """
        Load train state from the ``dir``.
        """
        raise NotImplementedError

    @property
    def work_dir(self) -> Path:
        if self._work_dir is None:
            raise RuntimeError("work_dir hasn't been set yet!")
        return self._work_dir

    @work_dir.setter
    def work_dir(self, work_dir: PathOrStr):
        self._work_dir = Path(work_dir)


class SimpleCheckpointer(Checkpointer):
    def __init__(self, enable_single_replica_array_restore: bool = False):
        super().__init__()
        self.enable_single_replica_array_restore = enable_single_replica_array_restore

        handler_registry = ocp.DefaultCheckpointHandlerRegistry()
        json_handler = ocp.JsonCheckpointHandler()
        pytree_handler = ocp.PyTreeCheckpointHandler(use_ocdbt=True, use_zarr3=True)
        handler_registry.add("trainer", ocp.args.JsonSave, json_handler)
        handler_registry.add("trainer", ocp.args.JsonRestore, json_handler)
        handler_registry.add("data_loader", ocp.args.PyTreeSave, pytree_handler)
        handler_registry.add("data_loader", ocp.args.PyTreeRestore, pytree_handler)
        handler_registry.add("params", ocp.args.PyTreeSave, pytree_handler)
        handler_registry.add("params", ocp.args.PyTreeRestore, pytree_handler)
        handler_registry.add("static", ocp.args.PyTreeSave, pytree_handler)
        handler_registry.add("static", ocp.args.PyTreeRestore, pytree_handler)
        handler_registry.add("opt_state", ocp.args.PyTreeSave, pytree_handler)
        handler_registry.add("opt_state", ocp.args.PyTreeRestore, pytree_handler)
        self.handler_registry = handler_registry

        if self.enable_single_replica_array_restore:
            array_handler = ocp.type_handlers.SingleReplicaArrayHandler(
                replica_axis_index=0,
                broadcast_memory_limit_bytes=1024 * 1024 * 1000,  # 1000 MB limit
            )
            ocp.type_handlers.register_type_handler(jax.Array, array_handler, override=True)

    def save(self, dir: PathOrStr, state: "TrainState", save_overwrite: bool = False):
        checkpointer = self._get_checkpointer()
        with fs.get_tempdir_for(dir, work_dir=self.work_dir, save_overwrite=save_overwrite) as wd:
            checkpointer.save(
                wd,
                args=self._get_checkpoint_save_args(state),
                force=True,
            )

    def load(self, dir: PathOrStr, state: "TrainState") -> "TrainState":
        local_dir: Path
        if fs.is_url(dir):
            sha256_hash = hashlib.sha256()
            sha256_hash.update(fs.normalize_path(dir).encode())
            dirname_hash = sha256_hash.hexdigest()
            local_dir = self.work_dir / "checkpointer" / dirname_hash
            with fs.get_tempdir_for(local_dir, save_overwrite=True) as wd:
                if dist.get_process_filesystem_rank(local_dir) == 0:
                    fs.copy_dir(dir, wd)
        else:
            local_dir = Path(dir)

        checkpointer = self._get_checkpointer()
        try:
            result = checkpointer.restore(local_dir, self._get_checkpoint_restore_args(state))

            # HACK: orbax will commit all arrays to devices, including previously uncommitted single
            # device arrays, which might break things when some of those arrays are not meant to be committed.
            # So we force un-commit those here.
            opt_state = jax_utils.uncommit_single_device_arrays(result["opt_state"])
            params = jax_utils.uncommit_single_device_arrays(result["params"])
            state.data_loader.load_state(
                jax_utils.uncommit_single_device_arrays(result["data_loader"])
            )
            return dataclasses.replace(
                state, params=params, opt_state=opt_state, **result["trainer"]
            )
        finally:
            if fs.is_url(dir) and dist.get_process_filesystem_rank(local_dir) == 0:
                fs.clear_directory(local_dir)

    def _get_checkpointer(self) -> ocp.Checkpointer:
        checkpointer = ocp.Checkpointer(
            ocp.CompositeCheckpointHandler(handler_registry=self.handler_registry)
        )
        return checkpointer

    def _get_checkpoint_save_args(self, state: "TrainState") -> ocp.args.Composite:
        return ocp.args.Composite(
            trainer=ocp.args.JsonSave(  # pyright: ignore
                {  # pyright: ignore
                    "step": state.step,
                    "epoch": state.epoch,
                    "global_train_tokens_seen": state.global_train_tokens_seen,
                }
            ),
            data_loader=ocp.args.PyTreeSave(state.data_loader.get_state()),  # pyright: ignore
            params=ocp.args.PyTreeSave(state.params),  # pyright: ignore
            static=ocp.args.PyTreeSave(state.static),  # pyright: ignore
            opt_state=ocp.args.PyTreeSave(state.opt_state),  # pyright: ignore
        )

    def _get_checkpoint_restore_args(self, state: "TrainState") -> ocp.args.Composite:
        return ocp.args.Composite(
            trainer=ocp.args.JsonRestore(  # pyright: ignore
                {  # pyright: ignore
                    "step": state.step,
                    "epoch": state.epoch,
                    "global_train_tokens_seen": state.global_train_tokens_seen,
                }
            ),
            data_loader=self._make_pytree_restore_args(  # pyright: ignore
                state.data_loader.get_state()
            ),
            params=self._make_pytree_restore_args(state.params),  # pyright: ignore
            opt_state=self._make_pytree_restore_args(state.opt_state),  # pyright: ignore
        )

    def _make_pytree_restore_args(self, data) -> ocp.args.PyTreeRestore:
        return ocp.args.PyTreeRestore(
            item=data,  # pyright: ignore
            restore_args=jax.tree.map(self._make_array_restore_args, data),  # pyright: ignore
        )

    def _make_array_restore_args(self, data) -> ocp.ArrayRestoreArgs | None:
        if not isinstance(data, Array):
            return None
        elif self.enable_single_replica_array_restore:
            #  return ocp.type_handlers.SingleReplicaArrayRestoreArgs(sharding=data.sharding)
            assert isinstance(data.sharding, jax.sharding.NamedSharding)
            pspec = data.sharding.spec
            mesh = data.sharding.mesh
            replica_axis_index = 0
            assert mesh.devices is not None
            replica_devices = _replica_devices(mesh.devices, replica_axis_index)
            replica_mesh = jax.sharding.Mesh(replica_devices, mesh.axis_names)
            single_replica_sharding = jax.sharding.NamedSharding(replica_mesh, pspec)
            return ocp.type_handlers.SingleReplicaArrayRestoreArgs(
                sharding=jax.sharding.NamedSharding(mesh, pspec),
                single_replica_sharding=single_replica_sharding,
                global_shape=data.shape,
                dtype=data.dtype,
            )
        else:
            return ocp.ArrayRestoreArgs(sharding=data.sharding)


def _replica_devices(device_array: np.ndarray, replica_axis_idx: int):
    """Returns the devices from the replica that current host belongs to."""
    idx = _find_idx(device_array, replica_axis_idx)
    replica_result = np.take(device_array, idx, axis=replica_axis_idx)
    return np.expand_dims(replica_result, axis=replica_axis_idx)


def _find_idx(array: np.ndarray, replica_axis_idx: int):
    """Returns the index along given dimension that the current host belongs to."""
    idx = None
    for idx, val in np.ndenumerate(array):
        if val.process_index == jax.process_index():
            break
    return idx[replica_axis_idx]  # type: ignore
