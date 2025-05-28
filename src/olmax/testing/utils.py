import jax
import jax.numpy as jnp

from ..types import PyTree


def allclose(a: PyTree, b: PyTree) -> bool:
    return jax.tree.all(jax.tree.map(jnp.allclose, a, b))
