import equinox as eqx
import jax
import jax.numpy as jnp

import olmax.nn as nn
from olmax.testing.utils import allclose
from olmax.train.utils import microbatched
from olmax.types import *


def test_microbatched():
    key = jax.random.PRNGKey(42)
    model_key, x_key, y_key = jax.random.split(key, 3)
    batch_size, microbatch_size, in_size, out_size = 6, 2, 4, 4
    num_microbatches = batch_size // microbatch_size

    model = nn.Linear(in_size, out_size, model_key)
    x = jax.random.normal(x_key, (batch_size, in_size))
    y = jax.random.normal(y_key, (batch_size, out_size))

    @eqx.filter_value_and_grad
    def _loss_fn(model: nn.Linear, x: Array, y: Array) -> Array:
        return jnp.mean(jnp.sum((model(x) - y) ** 2, axis=-1))

    @jax.jit
    def _get_loss_and_grads(model, x: Array, y: Array):
        return _loss_fn(model, x, y)

    @jax.jit
    def _get_microbatched_loss_and_grads(model, x: Array, y: Array):
        return microbatched(_get_loss_and_grads, model, x, y, num_microbatches=num_microbatches)

    full_loss, full_grads = _get_loss_and_grads(model, x, y)
    loss_acc, grad_acc = _get_microbatched_loss_and_grads(model, x, y)
    assert allclose(full_loss, loss_acc)
    assert allclose(full_grads, grad_acc)
