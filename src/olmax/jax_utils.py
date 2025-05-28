from typing import Callable, TypeVar

import jax

F = TypeVar("F", bound=Callable)


def vmap_multiple(fun: F, n: int) -> F:
    for _ in range(n):
        fun = jax.vmap(fun)
    return fun
