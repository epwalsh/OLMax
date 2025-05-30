from typing import Callable, TypeVar

import equinox as eqx
import jax

from .types import DTypeLike, PyTree

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
