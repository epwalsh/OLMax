import functools as ft
from typing import TypeVar

import equinox as eqx
import jax

from ..types import Array, PyTree

T = TypeVar("T", bound=PyTree)


@eqx.filter_jit(donate="all")
def sgd_step(model: T, grads: T, *, lr: float) -> T:
    return jax.tree.map(ft.partial(_sgd_single, lr=lr), model, grads)


def _sgd_single(p: Array, g: Array, *, lr: float) -> Array:
    p = p - lr * g
    return p
