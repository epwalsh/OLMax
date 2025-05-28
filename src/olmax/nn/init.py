import functools as ft

import jax
from jax.sharding import NamedSharding
from jaxtyping import Array, DTypeLike, PRNGKeyArray


@ft.partial(
    jax.jit, static_argnums=(1,), static_argnames=("shape", "dtype", "sharding")
)
def truncated_normal(
    key: PRNGKeyArray,
    shape: tuple[int, ...],
    *,
    dtype: DTypeLike = float,
    sharding: NamedSharding | None = None,
    stddev: float = 0.02,
    lower: float = -3.0,
    upper: float = 3.0,
) -> Array:
    out = jax.nn.initializers.truncated_normal(stddev=stddev, lower=lower, upper=upper)(
        key, shape, dtype=dtype
    )
    if sharding is not None:
        out = jax.lax.with_sharding_constraint(out, sharding)
    return out


@ft.partial(
    jax.jit, static_argnums=(1,), static_argnames=("shape", "dtype", "sharding")
)
def zeros(
    key: PRNGKeyArray,
    shape: tuple[int, ...],
    *,
    dtype: DTypeLike = float,
    sharding: NamedSharding | None = None,
) -> Array:
    out = jax.nn.initializers.zeros(key, shape, dtype=dtype)
    if sharding is not None:
        out = jax.lax.with_sharding_constraint(out, sharding)
    return out


@ft.partial(
    jax.jit, static_argnums=(1,), static_argnames=("shape", "dtype", "sharding")
)
def ones(
    key: PRNGKeyArray,
    shape: tuple[int, ...],
    *,
    dtype: DTypeLike = float,
    sharding: NamedSharding | None = None,
) -> Array:
    out = jax.nn.initializers.ones(key, shape, dtype=dtype)
    if sharding is not None:
        out = jax.lax.with_sharding_constraint(out, sharding)
    return out
