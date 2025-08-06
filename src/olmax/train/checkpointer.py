import dataclasses
import hashlib
import logging
import os
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import jax
import orbax.checkpoint as ocp

from .. import distributed as dist
from .. import fs, jax_utils, utils
from ..checkpoint import utils as checkpoint_utils
from ..types import *

if TYPE_CHECKING:
    from .trainer import TrainState


log = logging.getLogger(__name__)


class Checkpointer(ABC):
    def __init__(self):
        self._work_dir: Path | None = None

    @abstractmethod
    def save(
        self,
        dir: PathOrStr,
        state: "TrainState",
        save_overwrite: bool = False,
    ):
        """
        Save train state to the ``dir`` synchronously.
        """
        raise NotImplementedError

    @abstractmethod
    def save_async(
        self,
        dir: PathOrStr,
        state: "TrainState",
        save_overwrite: bool = False,
    ) -> checkpoint_utils.AsyncSaveHandle:
        """
        Save train state to the ``dir`` asynchronously.
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


class SimpleCheckpointerAsyncSaveHandle:
    def __init__(
        self,
        checkpointer: ocp.AsyncCheckpointer,
        done_event: threading.Event,
        finalizer_thread: threading.Thread,
    ):
        self.checkpointer = checkpointer
        self.done_event = done_event
        self.finalizer_thread = finalizer_thread

    def is_done(self) -> bool:
        if self.done_event.is_set():
            self.checkpointer.wait_until_finished()
            return True
        else:
            return False

    def wait_until_finished(self):
        self.checkpointer.wait_until_finished()

    def close(self):
        self.finalizer_thread.join()
        self.checkpointer.close()


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

    def save(
        self,
        dir: PathOrStr,
        state: "TrainState",
        save_overwrite: bool = False,
    ):
        local_dir = self.work_dir / os.path.basename(dir)
        checkpointer = self._get_checkpointer()
        checkpointer.save(
            local_dir, args=self._get_checkpoint_save_args(state), force=save_overwrite
        )

        if fs.normalize_path(local_dir) != fs.normalize_path(dir):
            log.info(f"Copying checkpoint from '{local_dir}' to '{dir}'...")
            start_time = time.perf_counter()

            if dist.get_process_filesystem_rank(local_dir) == 0:
                fs.copy_dir(local_dir, dir, save_overwrite=save_overwrite)

            dist.barrier(f"checkpointer-post-upload-{os.path.basename(dir)}")

            end_time = time.perf_counter()
            log.info(
                f"Copied checkpoint for from '{local_dir}' to '{dir}' "
                f"in {utils.format_timedelta(end_time - start_time)}."
            )

    def save_async(
        self,
        dir: PathOrStr,
        state: "TrainState",
        save_overwrite: bool = False,
    ) -> checkpoint_utils.AsyncSaveHandle:
        local_dir = self.work_dir / os.path.basename(dir)
        done_event = threading.Event()
        checkpointer = self._get_async_checkpointer()
        checkpointer.save(dir, args=self._get_checkpoint_save_args(state), force=save_overwrite)
        finalizer_thread = threading.Thread(
            target=self._finalize_async_checkpoint,
            kwargs=dict(
                checkpointer=checkpointer,
                done_event=done_event,
                local_dir=local_dir,
                final_dir=dir,
                save_overwrite=save_overwrite,
            ),
        )
        finalizer_thread.start()

        return SimpleCheckpointerAsyncSaveHandle(checkpointer, done_event, finalizer_thread)

    def load(self, dir: PathOrStr, state: "TrainState") -> "TrainState":
        local_dir: Path
        if fs.is_url(dir):
            sha256_hash = hashlib.sha256()
            sha256_hash.update(fs.normalize_path(dir).encode())
            dirname_hash = sha256_hash.hexdigest()
            local_dir = self.work_dir / "checkpointer" / dirname_hash
            with fs.get_temp_dir_for_target_dir(local_dir, save_overwrite=True) as wd:
                if dist.get_process_filesystem_rank(local_dir) == 0:
                    fs.copy_dir(dir, wd)
        else:
            local_dir = Path(dir)

        checkpointer = self._get_checkpointer()
        try:
            result = checkpointer.restore(local_dir, self._get_checkpoint_restore_args(state))
            updates = dict(callbacks_state=result["callbacks_state"], **result["trainer"])
            # HACK: orbax will commit all arrays to devices, including previously uncommitted single
            # device arrays, which might break things in compiled regions when those arrays are not
            # meant to be committed. So we force un-commit those here.
            updates["params"] = jax_utils.uncommit_single_device_arrays(result["params"])
            updates["opt_state"] = jax_utils.uncommit_single_device_arrays(result["opt_state"])
            updates["data_loader_state"] = jax_utils.uncommit_single_device_arrays(
                result["data_loader_state"]
            )

            state = dataclasses.replace(state, **updates)
        finally:
            if fs.is_url(dir) and dist.get_process_filesystem_rank(local_dir) == 0:
                fs.clear_directory(local_dir)

        return state

    def _get_checkpointer(self) -> ocp.Checkpointer:
        handler = ocp.CompositeCheckpointHandler(handler_registry=self.handler_registry)
        return ocp.Checkpointer(handler)

    def _get_async_checkpointer(
        self, done_callback: Callable[[], None] | None = None
    ) -> ocp.AsyncCheckpointer:
        handler = ocp.CompositeCheckpointHandler(handler_registry=self.handler_registry)
        return ocp.AsyncCheckpointer(
            handler,
            async_options=ocp.options.AsyncOptions(post_finalization_callback=done_callback),
        )

    def _finalize_async_checkpoint(
        self,
        *,
        checkpointer: ocp.AsyncCheckpointer,
        done_event: threading.Event,
        local_dir: PathOrStr,
        final_dir: PathOrStr,
        save_overwrite: bool,
    ):
        checkpointer.wait_until_finished()
        if fs.normalize_path(local_dir) != fs.normalize_path(final_dir):
            if dist.get_process_filesystem_rank(local_dir) == 0:
                log.info(f"Copying checkpoint from '{local_dir}' to '{final_dir}'...")
                start_time = time.perf_counter()
                fs.copy_dir(local_dir, final_dir, save_overwrite=save_overwrite)
                end_time = time.perf_counter()
                log.info(
                    f"Copied checkpoint for from '{local_dir}' to '{final_dir}' "
                    f"in {utils.format_timedelta(end_time - start_time)}."
                )
        done_event.set()

    def _get_checkpoint_save_args(self, state: "TrainState") -> ocp.args.Composite:
        return ocp.args.Composite(
            trainer=ocp.args.JsonSave(  # pyright: ignore
                {  # pyright: ignore
                    "step": state.step,
                    "epoch": state.epoch,
                    "global_train_tokens_seen": state.global_train_tokens_seen,
                }
            ),
            params=ocp.args.PyTreeSave(state.params),  # pyright: ignore
            static=ocp.args.PyTreeSave(state.static),  # pyright: ignore
            opt_state=ocp.args.PyTreeSave(state.opt_state),  # pyright: ignore
            data_loader_state=ocp.args.PyTreeSave(state.data_loader_state),  # pyright: ignore
            callbacks_state=ocp.args.PyTreeSave(state.callbacks_state),  # pyright: ignore
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
            params=self._make_pytree_restore_args(state.params),  # pyright: ignore
            opt_state=self._make_pytree_restore_args(state.opt_state),  # pyright: ignore
            data_loader_state=self._make_pytree_restore_args(  # pyright: ignore
                state.data_loader_state
            ),
            callbacks_state=self._make_pytree_restore_args(  # pyright: ignore
                state.callbacks_state
            ),
        )

    def _make_pytree_restore_args(self, data) -> ocp.args.PyTreeRestore:
        return ocp.args.PyTreeRestore(
            item=data,  # pyright: ignore
            restore_args=checkpoint_utils.make_restore_args(data),  # pyright: ignore
        )
