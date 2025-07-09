import dataclasses
import hashlib
import threading
from abc import ABC, abstractmethod
from contextlib import ExitStack
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import jax
import orbax.checkpoint as ocp

from .. import distributed as dist
from .. import fs, jax_utils
from ..checkpoint import utils as checkpoint_utils
from ..types import *

if TYPE_CHECKING:
    from .trainer import TrainState


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
        done_callback: Callable[[], None] | None = None,
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
        checkpointer = self._get_checkpointer()
        with fs.get_temp_dir_for_target_dir(
            dir, work_dir=self.work_dir, save_overwrite=save_overwrite
        ) as wd:
            checkpointer.save(wd, args=self._get_checkpoint_save_args(state), force=True)

    def save_async(
        self,
        dir: PathOrStr,
        state: "TrainState",
        save_overwrite: bool = False,
        done_callback: Callable[[], None] | None = None,
    ) -> checkpoint_utils.AsyncSaveHandle:
        stack = ExitStack()
        wd = stack.enter_context(
            fs.get_temp_dir_for_target_dir(
                dir, work_dir=self.work_dir, save_overwrite=save_overwrite
            )
        )
        done_event = threading.Event()

        def final_done_callback():
            stack.close()
            if done_callback is not None:
                done_callback()
            done_event.set()

        checkpointer = self._get_async_checkpointer(final_done_callback)
        checkpointer.save(wd, args=self._get_checkpoint_save_args(state), force=True)
        return checkpoint_utils.OrbaxAsyncSaveHandle(
            checkpointer=checkpointer, done_event=done_event
        )

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

            # HACK: orbax will commit all arrays to devices, including previously uncommitted single
            # device arrays, which might break things in compiled regions when those arrays are not
            # meant to be committed. So we force un-commit those here.
            opt_state = jax_utils.uncommit_single_device_arrays(result["opt_state"])
            params = jax_utils.uncommit_single_device_arrays(result["params"])
            data_loader_state = jax_utils.uncommit_single_device_arrays(result["data_loader"])

            state.data_loader.load_state(data_loader_state)
            state = dataclasses.replace(
                state, params=params, opt_state=opt_state, **result["trainer"]
            )
        finally:
            if fs.is_url(dir) and dist.get_process_filesystem_rank(local_dir) == 0:
                fs.clear_directory(local_dir)

        return state

    def _get_checkpointer(self) -> ocp.Checkpointer:
        handler = ocp.CompositeCheckpointHandler(handler_registry=self.handler_registry)
        return ocp.Checkpointer(handler)

    def _get_async_checkpointer(
        self, done_callback: Callable[[], None] | None
    ) -> ocp.AsyncCheckpointer:
        handler = ocp.CompositeCheckpointHandler(handler_registry=self.handler_registry)
        return ocp.AsyncCheckpointer(
            handler,
            async_options=ocp.options.AsyncOptions(post_finalization_callback=done_callback),
        )

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
            restore_args=checkpoint_utils.make_restore_args(data),  # pyright: ignore
        )
