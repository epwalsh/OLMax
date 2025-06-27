from typing import Any, Callable, ParamSpec, TypeVar

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
    *args,
    num_microbatches: int,
    accum_sharding: Specs | None = None,
    accum_dtype: DTypeLike = jnp.float32,
    divide_factor: ArrayLike | None = None,
    **kwargs,
) -> R:
    batch_size = _infer_batch_size(args, kwargs)
    if batch_size % num_microbatches != 0:
        raise ValueError("Batch size isn't divided evenly by num_microbatches.")
    microbatch_size = batch_size // num_microbatches
    if divide_factor is None:
        divide_factor = num_microbatches

    @jax.named_scope("per_microbatch_fun")
    def per_microbatch_fun(*mb_args: Args.args, **mb_kwargs: Args.kwargs) -> R:
        """Compute the micro-batch loss."""
        out = fun(*mb_args, **mb_kwargs)
        return jax.tree.map(lambda x: (x / divide_factor).astype(accum_dtype), out)

    @jax.named_scope("per_microbatch_train_step")
    def per_microbatch_train_step(loop_cnt: int, accum: R) -> R:
        """Run a training step on a micro-batch."""
        mb_args, mb_kwargs = _slice_inputs_for_microbatch(args, kwargs, loop_cnt, microbatch_size)
        result = per_microbatch_fun(*mb_args, **mb_kwargs)
        accum = jax.tree.map(jnp.add, accum, result)
        if accum_sharding is not None:
            accum = jax.lax.with_sharding_constraint(accum, accum_sharding)
        return accum

    accum_shape = eqx.filter_eval_shape(fun, *args, **kwargs)
    accum = zeros_like_tree(accum_shape, accum_dtype)
    if accum_sharding is not None:
        accum = jax.lax.with_sharding_constraint(accum, accum_sharding)

    return jax.lax.fori_loop(0, num_microbatches, per_microbatch_train_step, accum)


def _infer_batch_size(args: tuple[Any, ...], kwargs: dict[str, Any]) -> int:
    for x in args:
        if isinstance(x, Array):
            return x.shape[0]
    for x in kwargs.values():
        if isinstance(x, Array):
            return x.shape[0]
    raise ValueError("Couldn't determine batch size from arguments")


def _slice_for_microbatch(x: T, idx: int, microbatch_size: int) -> T:
    if isinstance(x, Array):
        offset = idx * microbatch_size
        length = microbatch_size
        starts = [offset] + [0] * (x.ndim - 1)
        limits = [length] + list(x.shape[1:])
        return jax.lax.dynamic_slice(x, starts, limits)
    else:
        return x


def _slice_inputs_for_microbatch(
    args: tuple[Any, ...], kwargs: dict[str, Any], idx: int, microbatch_size: int
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    args = tuple(_slice_for_microbatch(x, idx, microbatch_size) for x in args)
    kwargs = {k: _slice_for_microbatch(x, idx, microbatch_size) for k, x in kwargs.items()}
    return args, kwargs
