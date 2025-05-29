import functools as ft

import jax
import jax.numpy as jnp

from ..types import ArrayLike, PyTree


def _allclose(a: ArrayLike, b: ArrayLike, rtol: float = 1e-05, atol: float = 1e-08) -> bool:
    return jnp.allclose(a, b, rtol=rtol, atol=atol).item()


def allclose(a: PyTree, b: PyTree, rtol: float = 1e-05, atol: float = 1e-08) -> bool:
    partial_allclose = ft.partial(_allclose, rtol=rtol, atol=atol)
    return jax.tree.all(jax.tree.map(partial_allclose, a, b))
