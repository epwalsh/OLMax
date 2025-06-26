from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp

from ..jax_utils import cast_tree, zeros_like_tree
from ..types import *


def microbatched(
    loss_fn: Callable[[PyTree, dict[str, Array]], Array],
    model: PyTree,
    batch: dict[str, Array],
    *,
    num_microbatches: int,
    param_sharding: Specs | None = None,
    batch_sharding: Specs | None = None,
    acc_dtype: DTypeLike = jnp.float32,
    divide_factor: ArrayLike | None = None,
) -> tuple[Array, Array]:
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

    @eqx.filter_value_and_grad
    @jax.named_scope("per_microbatch_loss_fn")
    def per_microbatch_loss_fn(model: PyTree, microbatch: dict[str, Array]):
        """Compute the micro-batch loss."""
        if param_sharding is not None:
            model = jax.lax.with_sharding_constraint(model, param_sharding)
        if batch_sharding is not None:
            microbatch = jax.lax.with_sharding_constraint(microbatch, batch_sharding)
        return loss_fn(model, microbatch) / divide_factor

    @jax.named_scope("per_microbatch_train_step")
    def per_microbatch_train_step(
        loop_cnt: int,
        state: tuple[Array, Array],
    ) -> tuple[Array, Array]:
        """Run a training step on a micro-batch."""
        loss_acc, grad_acc = state
        microbatch = get_microbatch(batch, loop_cnt)

        loss, grads = per_microbatch_loss_fn(model, microbatch)
        grads = cast_tree(grads, acc_dtype)
        if param_sharding is not None:
            grads = jax.lax.with_sharding_constraint(grads, param_sharding)

        loss_acc = loss_acc + loss
        grad_acc = jax.tree.map(jnp.add, grad_acc, grads)
        if param_sharding is not None:
            grad_acc = jax.lax.with_sharding_constraint(grad_acc, param_sharding)

        return loss_acc, grad_acc

    loss_acc = jnp.array(0.0)
    grad_acc = zeros_like_tree(model, acc_dtype)
    if param_sharding is not None:
        grad_acc = jax.lax.with_sharding_constraint(grad_acc, param_sharding)

    loss_acc, grad_acc = jax.lax.fori_loop(
        0, num_microbatches, per_microbatch_train_step, (loss_acc, grad_acc)
    )
    return loss_acc, grad_acc
