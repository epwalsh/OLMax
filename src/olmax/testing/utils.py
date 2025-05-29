import functools as ft

import jax
import jax.numpy as jnp

from ..types import PyTree


def allclose(a: PyTree, b: PyTree, rtol: float = 1e-05, atol: float = 1e-08) -> bool:
    jnp_allclose = ft.partial(jnp.allclose, rtol=rtol, atol=atol)
    return jax.tree.all(jax.tree.map(jnp_allclose, a, b))
