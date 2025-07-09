import dataclasses
import hashlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

import orbax.checkpoint as ocp

from .. import distributed as dist
from .. import fs
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
    def save(self, dir: PathOrStr, state: "TrainState", save_overwrite: bool = False):
        ocp.StandardCheckpointHandler
        checkpointer = ocp.Checkpointer(ocp.CompositeCheckpointHandler())
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

        checkpointer = ocp.Checkpointer(ocp.CompositeCheckpointHandler())
        try:
            result = checkpointer.restore(local_dir, self._get_checkpoint_restore_args(state))
            state.data_loader.load_state(result["data_loader"])
            return dataclasses.replace(
                state, params=result["params"], opt_state=result["opt_state"], **result["trainer"]
            )
        finally:
            if fs.is_url(dir) and dist.get_process_filesystem_rank(local_dir) == 0:
                fs.clear_directory(local_dir)

    def _get_checkpoint_save_args(self, state: "TrainState") -> ocp.args.Composite:
        return ocp.args.Composite(
            trainer=ocp.args.JsonSave(  # pyright: ignore
                {  # pyright: ignore
                    "step": state.step,
                    "epoch": state.epoch,
                    "global_train_tokens_seen": state.global_train_tokens_seen,
                }
            ),
            data_loader=ocp.args.StandardSave(state.data_loader.get_state()),  # pyright: ignore
            params=ocp.args.StandardSave(state.params),  # pyright: ignore
            static=ocp.args.StandardSave(state.static),  # pyright: ignore
            opt_state=ocp.args.StandardSave(state.opt_state),  # pyright: ignore
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
            data_loader=ocp.args.StandardRestore(state.data_loader.get_state()),  # pyright: ignore
            params=ocp.args.StandardRestore(state.params),  # pyright: ignore
            static=ocp.args.StandardRestore(state.static),  # pyright: ignore
            opt_state=ocp.args.StandardRestore(state.opt_state),  # pyright: ignore
        )
