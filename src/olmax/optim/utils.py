from typing import Any, NamedTuple, Type, TypeVar

import jax
import jax.numpy as jnp
import optax

from ..jax_utils import get_global_norm
from ..types import Array, PyTree


class ClipByGlobalNormState(NamedTuple):
    global_norm: Array


def clip_grads_by_global_norm(grads: PyTree, max_norm: float) -> tuple[PyTree, Array]:
    g_norm = get_global_norm(grads)
    g_norm_clamped = jnp.maximum(max_norm, g_norm)
    grads = jax.tree.map(lambda t: (t / g_norm_clamped) * max_norm, grads)
    return grads, g_norm


def clip_grads_by_global_norm_transform(max_norm: float) -> optax.GradientTransformation:
    """
    Like :func:`optax.clip_by_global_norm` but tracks the global norm in its state.
    """

    def init_fn(params):
        del params
        return ClipByGlobalNormState(global_norm=jnp.zeros([], jnp.float32))

    def update_fn(updates, state, params=None):
        del params, state
        g_norm = get_global_norm(updates)
        g_norm_clamped = jnp.maximum(max_norm, g_norm)
        updates = jax.tree.map(lambda t: (t / g_norm_clamped) * max_norm, updates)
        return updates, ClipByGlobalNormState(global_norm=jax.copy_to_host_async(g_norm))

    return optax.GradientTransformation(init_fn, update_fn)


S = TypeVar("S")


def extract_state(opt_state: Any, state_class: Type[S]) -> S | None:
    if isinstance(opt_state, state_class):
        return opt_state
    elif isinstance(opt_state, optax.MultiStepsState):
        return extract_state(opt_state.inner_opt_state, state_class)
    elif isinstance(opt_state, tuple):
        for state in opt_state:
            if (value := extract_state(state, state_class)) is not None:
                return value
    return None


def extract_hyperparameter(opt_state: Any, hparam: str) -> Any:
    state = extract_state(opt_state, optax.InjectStatefulHyperparamsState)
    if state is not None:
        return jax.copy_to_host_async(state.hyperparams[hparam])
    raise KeyError(hparam)
