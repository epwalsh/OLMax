import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import olmax.distributed as dist
import olmax.nn as nn
from olmax.testing.distributed import run_distributed_test
from olmax.testing.utils import allclose
from olmax.types import Array, PRNGKeyArray


def _get_norm(
    key: PRNGKeyArray,
    dim: int,
    norm_type: str,
    parallel_config: dist.ParallelConfig | None = None,
) -> nn.LayerNorm | nn.RMSNorm:
    if norm_type == "LayerNorm":
        return nn.LayerNorm(dim, key, parallel_config=parallel_config)
    elif norm_type == "RMSNorm":
        return nn.RMSNorm(dim, key, parallel_config=parallel_config)
    else:
        raise ValueError(norm_type)


def _get_batch(key: PRNGKeyArray, batch_size: int, dim: int) -> tuple[Array, Array]:
    inputs_key, targets_key = jax.random.split(key)
    return jax.random.normal(inputs_key, (batch_size, dim)), jax.random.normal(
        targets_key, (batch_size, dim)
    )


def _get_loss(model: nn.Module, batch: tuple[Array, Array]) -> Array:
    inputs, targets = batch
    predictions = model(inputs)
    return jnp.mean(jnp.sum((predictions - targets) ** 2, axis=-1))


@jax.jit
@eqx.filter_value_and_grad
def _get_loss_and_grads(model: nn.Module, batch: tuple[Array, Array]) -> Array:
    return _get_loss(model, batch)


@pytest.mark.parametrize("norm_type", ["LayerNorm", "RMSNorm"])
def test_norm(norm_type: str):
    key = jax.random.PRNGKey(0)
    key, batch_key = jax.random.split(key)
    batch = _get_batch(batch_key, 2, 4)
    norm = _get_norm(key, 4, norm_type)
    loss, grads = _get_loss_and_grads(norm, batch)
    assert loss is not None
    assert grads is not None


def _run_norm_data_parallel(parallel_config: dist.ParallelConfig, norm_type: str):
    dim, batch_size = (
        4 * dist.get_global_device_count(),
        2 * dist.get_global_device_count(),
    )
    key = jax.random.PRNGKey(0)

    key, batch_key = jax.random.split(key)
    full_batch = _get_batch(batch_key, batch_size, dim)
    dist_batch = jax.device_put(full_batch, parallel_config.get_data_sharding())

    full_norm = _get_norm(key, dim, norm_type)
    dist_norm = _get_norm(key, dim, norm_type, parallel_config=parallel_config)

    assert allclose(full_norm.weight, dist_norm.weight)
    assert allclose(full_norm.bias, dist_norm.bias)

    full_loss, full_grads = _get_loss_and_grads(full_norm, full_batch)
    dist_loss, dist_grads = _get_loss_and_grads(dist_norm, dist_batch)
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
@pytest.mark.parametrize("norm_type", ["LayerNorm", "RMSNorm"])
def test_norm_data_parallel(parallel_config: dist.ParallelConfig, norm_type: str):
    run_distributed_test(
        _run_norm_data_parallel,
        num_processes=1,
        devices_per_process=parallel_config.get_min_device_count(),
        args=(parallel_config, norm_type),
    )
