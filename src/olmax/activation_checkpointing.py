import dataclasses
import fnmatch
import logging
import typing
from abc import abstractmethod
from dataclasses import dataclass
from typing import Callable, Iterable, TypeVar

import equinox as eqx
import jax
from dataclass_extensions import Registrable

from .utils import log_once

log = logging.getLogger(__name__)
F = TypeVar("F", bound=Callable)


@dataclass(unsafe_hash=True)
class ActivationCheckpointingPolicy(Registrable):
    """
    Defines an activation checkpointing (rematerialization) policy.
    """

    prevent_cse: bool = True

    @classmethod
    def no_policy(cls) -> "NoPolicy":
        return NoPolicy()

    @abstractmethod
    def get_policy(self) -> Callable[..., bool]:
        raise NotImplementedError

    def wrap(self, fun: F) -> F:
        """
        Wrap a function for activation checkpointing with the given policy.
        """
        return typing.cast(
            F, eqx.filter_checkpoint(fun, prevent_cse=self.prevent_cse, policy=self.get_policy())
        )


@ActivationCheckpointingPolicy.register("default", default=True)
@dataclass
class NoPolicy(ActivationCheckpointingPolicy):
    """
    A no-op. No checkpointing is applied. In theory this is equivalent to :class:`EverythingSaveable`,
    except this doesn't apply ``jax.checkpoint()`` at all, so it can be useful for debugging.
    """

    def get_policy(self) -> Callable[..., bool]:
        # Could return anything from here because we don't actually use this.
        return jax.checkpoint_policies.everything_saveable

    def wrap(self, fun: F) -> F:
        return fun


@ActivationCheckpointingPolicy.register("everything_saveable")
@dataclass
class EverythingSaveable(ActivationCheckpointingPolicy):
    """
    Everything is saved. This is essentially the same as :class:`NoPolicy`.
    """

    def get_policy(self) -> Callable[..., bool]:
        return jax.checkpoint_policies.everything_saveable


@ActivationCheckpointingPolicy.register("nothing_saveable")
@dataclass
class NothingSaveable(ActivationCheckpointingPolicy):
    """
    Nothing is saved, everything is recomputed. This is equivalent to calling ``jax.checkpoint()``
    without an explicit policy.
    """

    def get_policy(self) -> Callable[..., bool]:
        return jax.checkpoint_policies.nothing_saveable


@ActivationCheckpointingPolicy.register("dots_saveable")
@dataclass
class DotsSaveable(ActivationCheckpointingPolicy):
    """
    Only dot operations are saved, everything else is recomputed.
    """

    def get_policy(self) -> Callable[..., bool]:
        return jax.checkpoint_policies.dots_saveable


@ActivationCheckpointingPolicy.register("dots_with_no_batch_dims_saveable")
@dataclass
class DotsWithNoBatchDimsSaveable(ActivationCheckpointingPolicy):
    """
    Only certain dot operations are saved according to a heuristic which is generally useful
    for transformers. Everything else is recomputed.
    """

    def get_policy(self) -> Callable[..., bool]:
        return jax.checkpoint_policies.dots_with_no_batch_dims_saveable


@dataclass
class NamedCheckpointPolicy(ActivationCheckpointingPolicy):
    names: list[str] = dataclasses.field(default_factory=list)
    _resolved_names: list[str] | None = dataclasses.field(default=None, repr=False)

    def resolve_names(self, actual_names: Iterable[str]):
        self._resolved_names = []
        actual_names = list(actual_names)
        actual_names_set = set(actual_names)
        for name in self.names:
            has_match = False
            if name in actual_names_set:
                has_match = False
                self._resolved_names.append(name)
            else:
                for actual_name in actual_names:
                    if fnmatch.fnmatch(actual_name, name):
                        has_match = True
                        self._resolved_names.append(actual_name)
            if not has_match:
                raise ValueError(
                    f"checkpoint name pattern '{name}' does not match any named activations: {actual_names_set}"
                )

    def _get_names(self) -> list[str]:
        if self._resolved_names is not None:
            return self._resolved_names
        elif not any(["*" in name for name in self.names]):
            self._resolved_names = self.names
            return self.names
        else:
            raise RuntimeError("You must call `.resolve_names()` before applying the policy")


@ActivationCheckpointingPolicy.register("save_anything_except_these_names")
@dataclass
class SaveAnythingExceptTheseNames(NamedCheckpointPolicy):
    """
    Save any values (not just named ones) excluding the names given. Everything else is recomputed.
    """

    def get_policy(self) -> Callable[..., bool]:
        names = self._get_names()
        if names:
            names_str = "\n❯ ".join(names)
            log_once(log, f"Will save all activations except for:\n❯ {names_str}")
        return jax.checkpoint_policies.save_anything_except_these_names(*names)


@ActivationCheckpointingPolicy.register("save_any_names_but_these")
@dataclass
class SaveAnyNamesButThese(NamedCheckpointPolicy):
    """
    Save only named values, excluding the names given. Everything else is recomputed.
    """

    def get_policy(self) -> Callable[..., bool]:
        names = self._get_names()
        if names:
            names_str = "\n❯ ".join(names)
            log_once(log, f"Will save all named activations except for:\n❯ {names_str}")
        return jax.checkpoint_policies.save_any_names_but_these(*names)


@ActivationCheckpointingPolicy.register("save_only_these_names")
@dataclass
class SaveOnlyTheseNames(NamedCheckpointPolicy):
    """
    Save only named values, and only among the names given. Everything else is recomputed.
    """

    def get_policy(self) -> Callable[..., bool]:
        names = self._get_names()
        if names:
            names_str = "\n❯ ".join(names)
            log_once(log, f"Will save these named activations:\n❯ {names_str}")
        return jax.checkpoint_policies.save_only_these_names(*names)
