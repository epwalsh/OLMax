import jax
import jax.numpy as jnp

from ..jax_utils import get_global_norm
from ..types import Array, PyTree


def clip_grads_by_global_norm(grads: PyTree, max_norm: float) -> tuple[PyTree, Array]:
    g_norm = get_global_norm(grads)
    g_norm = jnp.maximum(max_norm, g_norm)
    grads = jax.tree.map(lambda t: (t / g_norm) * max_norm, grads)
    return grads, g_norm
