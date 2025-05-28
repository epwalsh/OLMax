import logging

import equinox as eqx
import jax
import jax.numpy as jnp

import olmax.distributed as dist
import olmax.nn as nn
from olmax.testing.distributed import run_distributed_test
from olmax.testing.utils import allclose
from olmax.types import Array, PRNGKeyArray


def _get_batch(
    key: PRNGKeyArray, batch_size: int, in_size: int, out_size: int
) -> tuple[Array, Array]:
    inputs_key, targets_key = jax.random.split(key)
    return jax.random.normal(inputs_key, (batch_size, in_size)), jax.random.normal(
        targets_key, (batch_size, out_size)
    )


def _get_loss(model: nn.Linear, batch: tuple[Array, Array]) -> Array:
    inputs, targets = batch
    predictions = model(inputs)
    return jnp.mean(jnp.sum((predictions - targets) ** 2, axis=-1))


@jax.jit
@eqx.filter_value_and_grad
def _get_loss_and_grads(model: nn.Linear, batch: tuple[Array, Array]) -> Array:
    return _get_loss(model, batch)


def test_linear():
    key = jax.random.PRNGKey(0)
    key, batch_key = jax.random.split(key)
    batch = _get_batch(batch_key, 2, 4, 4)
    linear = nn.Linear(4, 4, key=key)
    loss, grads = _get_loss_and_grads(linear, batch)
    assert loss is not None
    assert grads is not None


def _run_linear_with_fsdp():
    key = jax.random.PRNGKey(0)

    parallel_config = dist.ParallelConfig.FSDP()

    key, batch_key = jax.random.split(key)
    full_batch = _get_batch(batch_key, 2, 4, 4)
    fsdp_batch = jax.device_put(full_batch, parallel_config.get_dp_sharding())

    full_linear = nn.Linear(4, 4, key=key)
    fsdp_linear = nn.Linear(4, 4, key=key, parallel_config=parallel_config)

    assert allclose(full_linear.weight, fsdp_linear.weight)
    assert allclose(full_linear.bias, fsdp_linear.bias)

    full_loss, full_grads = _get_loss_and_grads(full_linear, full_batch)
    fsdp_loss, fsdp_grads = _get_loss_and_grads(fsdp_linear, fsdp_batch)
    assert allclose(full_loss, fsdp_loss)
    assert allclose(full_grads.weight, fsdp_grads.weight)
    assert allclose(full_grads.bias, fsdp_grads.bias)


def test_linear_with_fsdp():
    run_distributed_test(_run_linear_with_fsdp, num_processes=1, devices_per_process=2)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    run_distributed_test(_run_linear_with_fsdp)
