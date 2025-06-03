import functools as ft
from typing import Callable, TypeVar

import equinox as eqx
import jax
import jax.core
import jax.numpy as jnp

from .types import Array, DTypeLike, PyTree, Specs

F = TypeVar("F", bound=Callable)


def vmap_multiple(fun: F, n: int) -> F:
    for _ in range(n):
        fun = jax.vmap(fun)
    return fun


T = TypeVar("T", bound=PyTree)


def cast_tree(tree: T, dtype: DTypeLike) -> T:
    return jax.tree.map(lambda x: x.astype(dtype), tree)


@eqx.filter_jit(donate="all")
def count_params(tree: PyTree) -> int:
    return jax.tree.reduce(lambda c, p: c + p.size, tree, 0)


def with_optional_sharding_contraint(tree: T, sharding: Specs, cond: Callable[[Array], bool]) -> T:
    return jax.tree.map(
        ft.partial(_with_optional_sharding_contraint, sharding=sharding, cond=cond), tree
    )


def _with_optional_sharding_contraint(
    leaf: Array, sharding: Specs, cond: Callable[[Array], bool]
) -> Array:
    if cond(leaf):
        return jax.lax.with_sharding_constraint(leaf, sharding)
    else:
        return leaf


def is_in_jit():
    return isinstance(jnp.zeros((), dtype=jnp.float32), jax.core.Tracer)
