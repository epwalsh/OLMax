import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

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


def test_linear_3d():
    batch_size, seq_len, in_size, out_size = 2, 12, 8, 4
    key = jax.random.PRNGKey(0)
    key, batch_key = jax.random.split(key)
    inputs = jax.random.normal(batch_key, (batch_size, seq_len, in_size))
    linear = nn.Linear(in_size, out_size, key=key)
    out = linear(inputs)
    assert out.shape == (batch_size, seq_len, out_size)


def _run_linear_data_parallel(parallel_config: dist.ParallelConfig):
    in_size, out_size, batch_size = (
        2 * dist.get_global_device_count(),
        2 * dist.get_global_device_count(),
        2 * dist.get_global_device_count(),
    )
    key = jax.random.PRNGKey(0)

    key, batch_key = jax.random.split(key)
    full_batch = _get_batch(batch_key, batch_size, in_size, out_size)
    dist_batch = jax.device_put(full_batch, parallel_config.get_data_sharding())

    full_linear = nn.Linear(in_size, out_size, key=key)
    dist_linear = nn.Linear(in_size, out_size, key=key, parallel_config=parallel_config)

    assert allclose(full_linear.weight, dist_linear.weight)
    assert allclose(full_linear.bias, dist_linear.bias)

    full_loss, full_grads = _get_loss_and_grads(full_linear, full_batch)
    dist_loss, dist_grads = _get_loss_and_grads(dist_linear, dist_batch)
    assert allclose(full_loss, dist_loss)
    assert allclose(full_grads.weight, dist_grads.weight)
    assert allclose(full_grads.bias, dist_grads.bias)


@pytest.mark.parametrize(
    "parallel_config",
    [
        pytest.param(dist.ParallelConfig.FSDP(), id="FSDP"),
        pytest.param(dist.ParallelConfig.DDP(), id="DDP"),
        pytest.param(dist.ParallelConfig.HSDP(2, 2), id="HSDP"),
    ],
)
def test_linear_data_parallel(parallel_config: dist.ParallelConfig):
    run_distributed_test(
        _run_linear_data_parallel,
        num_processes=1,
        devices_per_process=parallel_config.get_min_device_count(),
        args=(parallel_config,),
    )


if __name__ == "__main__":
    jax.config.update("jax_num_cpu_devices", 2)
    jax.config.update("jax_disable_jit", True)

    parallel_config = dist.ParallelConfig.FSDP()

    key = jax.random.PRNGKey(0)
    in_size, out_size, batch_size = 4, 12, 4

    key, batch_key = jax.random.split(key)
    batch = _get_batch(batch_key, batch_size, in_size, out_size)
    batch = jax.device_put(batch, parallel_config.get_data_sharding())

    model = nn.Linear(in_size, out_size, key=key, parallel_config=parallel_config)
    loss, grads = _get_loss_and_grads(model, batch)
    print(loss)
