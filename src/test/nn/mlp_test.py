import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import olmax.distributed as dist
import olmax.nn as nn
from olmax.testing.distributed import run_distributed_test
from olmax.testing.utils import allclose
from olmax.types import Array, PRNGKeyArray


def _get_batch(key: PRNGKeyArray, batch_size: int, d_model: int) -> tuple[Array, Array]:
    inputs_key, targets_key = jax.random.split(key)
    return jax.random.normal(inputs_key, (batch_size, d_model)), jax.random.normal(
        targets_key, (batch_size, d_model)
    )


def _get_loss(model: nn.Module, batch: tuple[Array, Array]) -> Array:
    inputs, targets = batch
    predictions = model(inputs)
    return jnp.mean(jnp.sum((predictions - targets) ** 2, axis=-1))


@jax.jit
@eqx.filter_value_and_grad
def _get_loss_and_grads(model: nn.Module, batch: tuple[Array, Array]) -> Array:
    return _get_loss(model, batch)


def test_mlp(d_model: int = 4, hidden_size: int = 8, batch_size: int = 2):
    key = jax.random.PRNGKey(0)
    key, batch_key = jax.random.split(key)
    batch = _get_batch(batch_key, batch_size, d_model)
    mlp = nn.GatedMLP(d_model, hidden_size, key=key)
    loss, grads = _get_loss_and_grads(mlp, batch)
    assert loss is not None
    assert grads is not None


def _run_mlp_parallel(parallel_config: dist.ParallelConfig):
    d_model, hidden_size, batch_size = (
        2 * dist.get_global_device_count(),
        4 * dist.get_global_device_count(),
        2 * dist.get_global_device_count(),
    )
    key = jax.random.PRNGKey(0)

    key, batch_key = jax.random.split(key)
    full_batch = _get_batch(batch_key, batch_size, d_model)
    dist_batch = jax.device_put(full_batch, parallel_config.get_data_sharding())

    full_mlp = nn.GatedMLP(d_model, hidden_size, key=key)
    dist_mlp = nn.GatedMLP(d_model, hidden_size, key=key, parallel_config=parallel_config)

    assert allclose(full_mlp.w1.weight, dist_mlp.w1.weight)
    assert allclose(full_mlp.w2.weight, dist_mlp.w2.weight)
    assert allclose(full_mlp.w3.weight, dist_mlp.w3.weight)

    full_loss, full_grads = _get_loss_and_grads(full_mlp, full_batch)
    dist_loss, dist_grads = _get_loss_and_grads(dist_mlp, dist_batch)
    assert allclose(full_loss, dist_loss), f"{full_loss} != {dist_loss}"
    assert allclose(full_grads.w1.weight, dist_grads.w1.weight)
    assert allclose(full_grads.w1.bias, dist_grads.w1.bias)


@pytest.mark.parametrize(
    "parallel_config",
    [
        pytest.param(dist.ParallelConfig.FSDP(), id="FSDP"),
        pytest.param(dist.ParallelConfig.DDP(), id="DDP"),
        pytest.param(dist.ParallelConfig.HSDP(2, 2), id="HSDP"),
    ],
)
def test_mlp_data_parallel(parallel_config: dist.ParallelConfig):
    run_distributed_test(
        _run_mlp_parallel,
        num_processes=1,
        devices_per_process=parallel_config.get_min_device_count(),
        args=(parallel_config,),
    )


@pytest.mark.parametrize(
    "parallel_config",
    [
        pytest.param(
            dist.ParallelConfig(dp=dist.DataParallelConfig.FSDP(), tp=dist.TensorParallelConfig(2)),
            id="FSDP+TP",
        ),
        pytest.param(
            dist.ParallelConfig(tp=dist.TensorParallelConfig(2)),
            id="TP",
        ),
    ],
)
def test_mlp_tensor_parallel(parallel_config: dist.ParallelConfig):
    run_distributed_test(
        _run_mlp_parallel,
        num_processes=1,
        devices_per_process=parallel_config.get_min_device_count(),
        args=(parallel_config,),
    )


if __name__ == "__main__":
    jax.config.update("jax_num_cpu_devices", 2)
    jax.config.update("jax_disable_jit", True)

    parallel_config = dist.ParallelConfig(tp=dist.TensorParallelConfig(2))

    d_model, hidden_size, batch_size = (
        2 * dist.get_global_device_count(),
        4 * dist.get_global_device_count(),
        2 * dist.get_global_device_count(),
    )
    key = jax.random.PRNGKey(0)

    key, batch_key = jax.random.split(key)
    full_batch = _get_batch(batch_key, batch_size, d_model)
    dist_batch = jax.device_put(full_batch, parallel_config.get_data_sharding())

    full_mlp = nn.GatedMLP(d_model, hidden_size, key=key)
    dist_mlp = nn.GatedMLP(d_model, hidden_size, key=key, parallel_config=parallel_config)

    full_loss, _ = _get_loss_and_grads(full_mlp, full_batch)
    dist_loss, _ = _get_loss_and_grads(dist_mlp, dist_batch)
    print(full_loss, dist_loss)
