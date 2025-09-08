import functools as ft
from typing import Callable, Sequence, TypeVar

import equinox as eqx
import jax
import jax.core
import jax.numpy as jnp
import numpy as np

from .types import Array, ArrayLike, DTypeLike, PRNGKeyArray, PyTree, Specs

F = TypeVar("F", bound=Callable)


def vmap_multiple(fun: F, n: int) -> F:
    for _ in range(n):
        fun = jax.vmap(fun)
    return fun


T = TypeVar("T", bound=PyTree)


def cast_tree(tree: T, dtype: DTypeLike) -> T:
    return jax.tree.map(lambda x: x.astype(dtype), tree)


def zeros_like_tree(tree: T, dtype: DTypeLike | None = None) -> T:
    return jax.tree.map(lambda x: jnp.zeros_like(x, dtype=dtype), tree)


def uncommit_single_device_arrays(tree: T) -> T:
    def uncommit(x):
        if (
            isinstance(x, Array)
            and isinstance(x.sharding, jax.sharding.SingleDeviceSharding)
            and x.committed
        ):
            return jax.numpy.array(np.array(x))
        else:
            return x

    return jax.tree.map(uncommit, tree)


@eqx.filter_jit(donate="all")
def count_params(tree: PyTree) -> tuple[int, int]:
    """
    Get the total number of params and the total size in bytes of those params.
    """
    return jax.tree.reduce(
        lambda c, p: (
            c[0] + (p.size if eqx.is_array(p) else 0),
            c[1] + (p.size * p.dtype.itemsize if eqx.is_array(p) else 0),
        ),
        tree,
        (0, 0),
    )


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


def get_peak_local_device_memory_usage() -> int:
    peak_bytes_in_use = 0
    for device in jax.local_devices():
        memory_stats = device.memory_stats()
        peak_bytes_in_use = max(peak_bytes_in_use, memory_stats.get("peak_bytes_in_use", 0))
    return peak_bytes_in_use


def get_global_norm(tree: PyTree) -> Array:
    """Compute the global norm across a nested structure of tensors."""
    return jnp.sqrt(sum(jnp.sum(abs_sq(x)) for x in jax.tree.leaves(tree)))


def abs_sq(x: Array) -> Array:
    """
    Returns the squared absolute value of a (maybe complex) array.

    For real ``x``, JAX generates the same HLO from this, ``jnp.square(x)``, ``x * x``,
    or ``x**2``.
    """
    if not isinstance(x, (np.ndarray, jnp.ndarray)):
        raise ValueError(f"`abs_sq` accepts only NDarrays, got: {x}.")
    return (x.conj() * x).real


def get_cudnn_version() -> int | None:
    cuda_versions = jax._src.lib.cuda_versions  # pyright: ignore
    if cuda_versions is None:
        return None
    return cuda_versions.cudnn_get_version()


def shaped_rng_split(key, split_shape: int | Sequence[int] = 2) -> PRNGKeyArray:
    if isinstance(split_shape, int):
        num_splits = split_shape
        split_shape = (num_splits,) + key.shape
    else:
        num_splits = int(np.prod(split_shape))
        split_shape = tuple(split_shape) + key.shape

    if num_splits == 1:
        return jnp.reshape(key, split_shape)

    unshaped = maybe_rng_split(key, num_splits)
    return jnp.reshape(unshaped, split_shape)


def maybe_rng_split(key: PRNGKeyArray | None, num: int = 2) -> ArrayLike:
    """Splits a random key into multiple random keys. If the key is None, then it replicates the None. Also handles
    num == 1 case"""
    if key is None:
        return [None] * num  # type: ignore
    elif num == 1:
        return jnp.reshape(key, (1,) + key.shape)
    else:
        return jax.random.split(key, num)
