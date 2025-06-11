from __future__ import annotations

import typing
from abc import abstractmethod
from dataclasses import dataclass
from typing import Callable, TypeVar

import jax
from dataclass_extensions import Registrable

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


@ActivationCheckpointingPolicy.register("save_anything_except_these_names")
@dataclass
class SaveAnythingExceptTheseNames(ActivationCheckpointingPolicy):
    """
    Save any values (not just named ones) excluding the names given.
    """

    names: list[str]

    def get_policy(self) -> Callable[..., bool]:
        return jax.checkpoint_policies.save_anything_except_these_names(*self.names)


@ActivationCheckpointingPolicy.register("save_any_names_but_these")
@dataclass
class SaveAnyNamesButThese(ActivationCheckpointingPolicy):
    """
    Save only named values, excluding the names given.
    """

    names: list[str]

    def get_policy(self) -> Callable[..., bool]:
        return jax.checkpoint_policies.save_any_names_but_these(*self.names)


@ActivationCheckpointingPolicy.register("save_only_these_names")
@dataclass
class SaveOnlyTheseNames(ActivationCheckpointingPolicy):
    """
    Save only named values, and only among the names given.
    """

    names: list[str]

    def get_policy(self) -> Callable[..., bool]:
        return jax.checkpoint_policies.save_only_these_names(*self.names)
