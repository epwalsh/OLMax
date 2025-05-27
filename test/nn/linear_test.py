import functools as ft

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

import olmax.distributed as dist
import olmax.nn as nn
from olmax.testing.distributed import run_distributed_test
from olmax.testing.utils import allclose
from olmax.types import Array, PRNGKeyArray


def get_batch(
    key: PRNGKeyArray, batch_size: int, in_size: int, out_size: int
) -> tuple[Array, Array]:
    inputs_key, targets_key = jax.random.split(key)
    return jax.random.normal(inputs_key, (batch_size, in_size)), jax.random.normal(
        targets_key, (batch_size, out_size)
    )


def _run_linear_with_fsdp():
    key = jax.random.PRNGKey(0)

    parallel_config = dist.ParallelConfig.FSDP()
    shard_axis = parallel_config.get_data_sharding_axis_name()
    assert shard_axis is not None
    mesh = parallel_config.get_data_mesh()

    key, batch_key = jax.random.split(key)
    full_batch = get_batch(batch_key, 2, 4, 4)
    fsdp_batch = jax.device_put(full_batch, parallel_config.get_data_sharding())

    full_linear = nn.Linear(4, 4, key=key)
    fsdp_linear = nn.Linear(
        4, 4, key=key, sharding=nn.Linear.DefaultSharding(parallel_config)
    )

    assert allclose(full_linear.weight, fsdp_linear.weight)
    assert allclose(full_linear.bias, fsdp_linear.bias)

    def get_local_loss(model: nn.Linear, batch: tuple[Array, Array]):
        inputs, targets = batch
        predictions = jax.vmap(model)(inputs)
        return jnp.mean(jnp.sum((predictions - targets) ** 2, axis=-1))

    def get_full_loss(model: nn.Linear, batch: tuple[Array, Array]):
        return get_local_loss(model, batch)

    @ft.partial(jax.shard_map, mesh=mesh, in_specs=P(shard_axis), out_specs=P())
    def get_fsdp_loss(model: nn.Linear, batch: tuple[Array, Array]):
        local_loss = get_local_loss(model, batch)
        return jax.lax.pmean(local_loss, shard_axis)

    full_loss = get_full_loss(full_linear, full_batch)
    fsdp_loss = get_fsdp_loss(fsdp_linear, fsdp_batch)  # pyright: ignore
    assert allclose(full_loss, fsdp_loss)

    full_grads = jax.jit(jax.grad(get_full_loss))(full_linear, full_batch)
    fsdp_grads = jax.jit(jax.grad(get_fsdp_loss))(fsdp_linear, fsdp_batch)
    assert allclose(full_grads, fsdp_grads)


def test_linear_with_fsdp():
    run_distributed_test(_run_linear_with_fsdp)
