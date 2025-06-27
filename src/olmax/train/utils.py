from typing import Any, Callable, ParamSpec, Sequence, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp

from ..jax_utils import zeros_like_tree
from ..types import *

Args = ParamSpec("Args")
R = TypeVar("R")
T = TypeVar("T")


def microbatched(
    fun: Callable[Args, R],
    microbatches: Sequence[tuple[Any, ...]],
    accum_sharding: Specs | None = None,
    accum_dtype: DTypeLike = jnp.float32,
    divide_factor: ArrayLike | None = None,
    **kwargs,
) -> R:
    if divide_factor is None:
        divide_factor = len(microbatches)

    @jax.named_scope("per_microbatch_fun")
    def per_microbatch_fun(*mb_args: Args.args, **mb_kwargs: Args.kwargs) -> R:
        """Compute the micro-batch loss."""
        out = fun(*mb_args, **mb_kwargs)
        return jax.tree.map(lambda x: (x / divide_factor).astype(accum_dtype), out)

    @jax.named_scope("per_microbatch_train_step")
    def per_microbatch_train_step(loop_cnt: int, accum: R) -> R:
        """Run a training step on a micro-batch."""
        result = per_microbatch_fun(*microbatches[loop_cnt], **kwargs)
        accum = jax.tree.map(jnp.add, accum, result)
        if accum_sharding is not None:
            accum = jax.lax.with_sharding_constraint(accum, accum_sharding)
        return accum

    accum_shape = eqx.filter_eval_shape(fun, microbatches[0], **kwargs)
    accum = zeros_like_tree(accum_shape, accum_dtype)
    if accum_sharding is not None:
        accum = jax.lax.with_sharding_constraint(accum, accum_sharding)

    return jax.lax.fori_loop(0, len(microbatches), per_microbatch_train_step, accum)


def split_into_microbatches(b: Array, num_microbatches: int) -> list[Array]:
    b = b.reshape(-1, num_microbatches, *b.shape[1:])
    return [mb.squeeze(1) for mb in jnp.split(b, num_microbatches, axis=1)]
