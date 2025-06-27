from typing import Callable, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp

from ..jax_utils import zeros_like_tree
from ..types import *

M = TypeVar("M", bound=eqx.Module)
T = TypeVar("T")


def microbatched(
    fun: Callable[[M, dict[str, Array]], T],
    model: M,
    batch: dict[str, Array],
    *,
    num_microbatches: int,
    accum_sharding: Specs | None = None,
    accum_dtype: DTypeLike = jnp.float32,
    divide_factor: ArrayLike | None = None,
) -> T:
    batch_size = next(iter(batch.values())).shape[0]
    assert (
        batch_size % num_microbatches == 0
    ), "Batch size isn't divided evenly by num_microbatches."
    microbatch_size = batch_size // num_microbatches
    if divide_factor is None:
        divide_factor = num_microbatches

    def get_microbatch(batch: dict[str, Array], idx: int) -> dict[str, Array]:
        """Fetch microbatch slice from possibly-packed input data."""
        offset = idx * microbatch_size
        length = microbatch_size
        starts = {k: [offset] + [0] * (b.ndim - 1) for k, b in batch.items()}
        limits = {k: [length] + list(b.shape[1:]) for k, b in batch.items()}
        return {k: jax.lax.dynamic_slice(b, starts[k], limits[k]) for k, b in batch.items()}

    @jax.named_scope("per_microbatch_fun")
    def per_microbatch_fun(model: M, microbatch: dict[str, Array]):
        """Compute the micro-batch loss."""
        out = fun(model, microbatch)
        return jax.tree.map(lambda x: (x / divide_factor).astype(accum_dtype), out)

    @jax.named_scope("per_microbatch_train_step")
    def per_microbatch_train_step(loop_cnt: int, state: tuple[M, T]) -> tuple[M, T]:
        """Run a training step on a micro-batch."""
        model, accum = state
        microbatch = get_microbatch(batch, loop_cnt)
        result = per_microbatch_fun(model, microbatch)
        accum = jax.tree.map(jnp.add, accum, result)
        if accum_sharding is not None:
            accum = jax.lax.with_sharding_constraint(accum, accum_sharding)
        return (model, accum)

    accum_shape = eqx.filter_eval_shape(fun, model, batch)
    accum = zeros_like_tree(accum_shape, accum_dtype)
    if accum_sharding is not None:
        accum = jax.lax.with_sharding_constraint(accum, accum_sharding)

    model, accum = jax.lax.fori_loop(0, num_microbatches, per_microbatch_train_step, (model, accum))
    return accum
