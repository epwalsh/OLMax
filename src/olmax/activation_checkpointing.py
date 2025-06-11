from __future__ import annotations

import dataclasses
import fnmatch
import logging
import typing
from abc import abstractmethod
from dataclasses import dataclass
from typing import Callable, Iterable, TypeVar

import jax
from dataclass_extensions import Registrable

log = logging.getLogger(__name__)
F = TypeVar("F", bound=Callable)


@dataclass
class ActivationCheckpointingPolicy(Registrable):
    """
    Defines an activation checkpointing (rematerialization) policy.
    """

    @classmethod
    def no_policy(cls) -> NoPolicy:
        return NoPolicy()

    @abstractmethod
    def get_policy(self) -> Callable[..., bool]:
        raise NotImplementedError

    def wrap(self, fun: F, static_argnums: int | tuple[int, ...] = ()) -> F:
        """
        Wrap a function for activation checkpointing with the given policy.
        """
        return typing.cast(
            F, jax.checkpoint(fun, static_argnums=static_argnums, policy=self.get_policy())
        )


@ActivationCheckpointingPolicy.register("default", default=True)
@dataclass
class NoPolicy(ActivationCheckpointingPolicy):
    """
    A no-op. No policy is applied.
    """

    def get_policy(self) -> Callable[..., bool]:
        # Could return anything from here because we don't actually use this.
        return jax.checkpoint_policies.everything_saveable

    def wrap(self, fun: F, static_argnums: int | tuple[int, ...] = ()) -> F:
        del static_argnums
        return fun


@ActivationCheckpointingPolicy.register("everything_saveable")
@dataclass
class EverythingSaveable(ActivationCheckpointingPolicy):
    """
    Everything is saved.
    """

    def get_policy(self) -> Callable[..., bool]:
        return jax.checkpoint_policies.everything_saveable


@ActivationCheckpointingPolicy.register("nothing_saveable")
@dataclass
class NothingSaveable(ActivationCheckpointingPolicy):
    """
    Nothing is saved, everything is recomputed.
    """

    def get_policy(self) -> Callable[..., bool]:
        return jax.checkpoint_policies.nothing_saveable


@ActivationCheckpointingPolicy.register("dots_saveable")
@dataclass
class DotsSaveable(ActivationCheckpointingPolicy):
    """
    Only dot operations are saved.
    """

    def get_policy(self) -> Callable[..., bool]:
        return jax.checkpoint_policies.dots_saveable


@ActivationCheckpointingPolicy.register("dots_with_no_batch_dims_saveable")
@dataclass
class DotsWithNoBatchDimsSaveable(ActivationCheckpointingPolicy):
    """
    Only certain dot operations are saved according to a heuristic which is generally useful
    for transformers.
    """

    def get_policy(self) -> Callable[..., bool]:
        return jax.checkpoint_policies.dots_with_no_batch_dims_saveable


@dataclass
class NamedCheckpointPolicy(ActivationCheckpointingPolicy):
    names: list[str]
    _resolved_names: list[str] | None = dataclasses.field(default=None, repr=False)

    def resolve_names(self, actual_names: Iterable[str]):
        self._resolved_names = []
        actual_names_set = set(actual_names)
        for name in self.names:
            has_match = False
            if name in actual_names_set:
                self._resolved_names.append(name)
            elif "*" in name:
                for actual_name in actual_names:
                    if fnmatch.fnmatch(actual_name, name):
                        has_match = True
                        self._resolved_names.append(actual_name)
            if not has_match:
                raise ValueError(
                    f"checkpoint name pattern '{name}' does not match any named activations: {actual_names_set}"
                )


@ActivationCheckpointingPolicy.register("save_anything_except_these_names")
@dataclass
class SaveAnythingExceptTheseNames(NamedCheckpointPolicy):
    """
    Save any values (not just named ones) excluding the names given.
    """

    def get_policy(self) -> Callable[..., bool]:
        names = self._resolved_names or self.names
        if names:
            names_str = "\n❯ ".join(names)
            log.info(f"Will save all activations except for:\n❯ {names_str}")
        return jax.checkpoint_policies.save_anything_except_these_names(*names)


@ActivationCheckpointingPolicy.register("save_any_names_but_these")
@dataclass
class SaveAnyNamesButThese(NamedCheckpointPolicy):
    """
    Save only named values, excluding the names given.
    """

    def get_policy(self) -> Callable[..., bool]:
        names = self._resolved_names or self.names
        if names:
            names_str = "\n❯ ".join(names)
            log.info(f"Will save all named activations except for:\n❯ {names_str}")
        return jax.checkpoint_policies.save_any_names_but_these(*names)


@ActivationCheckpointingPolicy.register("save_only_these_names")
@dataclass
class SaveOnlyTheseNames(NamedCheckpointPolicy):
    """
    Save only named values, and only among the names given.
    """

    def get_policy(self) -> Callable[..., bool]:
        names = self._resolved_names or self.names
        if names:
            names_str = "\n❯ ".join(names)
            log.info(f"Will save these named activations:\n❯ {names_str}")
        return jax.checkpoint_policies.save_only_these_names(*names)
