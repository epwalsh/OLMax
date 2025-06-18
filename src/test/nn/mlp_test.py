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


def _run_mlp_parallel(mesh_resource: dist.MeshResource):
    d_model, hidden_size, batch_size = (
        2 * dist.get_global_device_count(),
        4 * dist.get_global_device_count(),
        2 * dist.get_global_device_count(),
    )
    key = jax.random.PRNGKey(0)

    key, batch_key = jax.random.split(key)
    full_batch = _get_batch(batch_key, batch_size, d_model)
    dist_batch = jax.device_put(full_batch, mesh_resource.get_data_sharding())

    full_mlp = nn.GatedMLP(d_model, hidden_size, key=key, bias=mesh_resource.tp is None)
    dist_mlp = nn.GatedMLP(
        d_model,
        hidden_size,
        key=key,
        bias=mesh_resource.tp is None,
        mesh_resource=mesh_resource,
    )

    assert allclose(full_mlp.up_proj.weight, dist_mlp.up_proj.weight)
    assert allclose(full_mlp.down_proj.weight, dist_mlp.down_proj.weight)
    assert allclose(full_mlp.gate_proj.weight, dist_mlp.gate_proj.weight)

    full_loss, full_grads = _get_loss_and_grads(full_mlp, full_batch)
    dist_loss, dist_grads = _get_loss_and_grads(dist_mlp, dist_batch)
    assert allclose(full_loss, dist_loss), f"{full_loss} != {dist_loss}"
    assert allclose(full_grads.gate_proj.weight, dist_grads.gate_proj.weight)
    assert allclose(full_grads.gate_proj.bias, dist_grads.gate_proj.bias)


@pytest.mark.parametrize(
    "mesh_resource",
    [
        pytest.param(dist.MeshResource.FSDP(), id="FSDP"),
        pytest.param(dist.MeshResource.DDP(), id="DDP"),
        pytest.param(dist.MeshResource.HSDP(2, 2), id="HSDP"),
    ],
)
def test_mlp_data_parallel(mesh_resource: dist.MeshResource):
    run_distributed_test(
        _run_mlp_parallel,
        num_processes=1,
        devices_per_process=mesh_resource.get_min_device_count(),
        args=(mesh_resource,),
    )


@pytest.mark.parametrize(
    "mesh_resource",
    [
        pytest.param(
            dist.MeshResource(dp=dist.DataParallelConfig.FSDP(), tp=dist.TensorParallelConfig(2)),
            id="FSDP+TP",
        ),
        pytest.param(
            dist.MeshResource(tp=dist.TensorParallelConfig(2)),
            id="TP",
        ),
    ],
)
def test_mlp_tensor_parallel(mesh_resource: dist.MeshResource):
    run_distributed_test(
        _run_mlp_parallel,
        num_processes=1,
        devices_per_process=mesh_resource.get_min_device_count(),
        args=(mesh_resource,),
    )


if __name__ == "__main__":
    jax.config.update("jax_num_cpu_devices", 2)
    jax.config.update("jax_disable_jit", True)

    mesh_resource = dist.MeshResource(tp=dist.TensorParallelConfig(2))

    d_model, hidden_size, batch_size = (8, 32, 2)
    key = jax.random.PRNGKey(0)

    key, batch_key = jax.random.split(key)
    full_batch = _get_batch(batch_key, batch_size, d_model)
    dist_batch = jax.device_put(full_batch, mesh_resource.get_data_sharding())

    full_mlp = nn.GatedMLP(d_model, hidden_size, key=key, bias=False)
    dist_mlp = nn.GatedMLP(d_model, hidden_size, key=key, bias=False, mesh_resource=mesh_resource)
    assert allclose(full_mlp.gate_proj.weight, dist_mlp.gate_proj.weight)
    assert allclose(full_mlp.down_proj.weight, dist_mlp.down_proj.weight)
    assert allclose(full_mlp.up_proj.weight, dist_mlp.up_proj.weight)

    full_preds = jax.jit(full_mlp)(full_batch[0])
    dist_preds = jax.jit(dist_mlp)(dist_batch[0])
    print(full_preds)
    print(dist_preds)

    full_loss, _ = _get_loss_and_grads(full_mlp, full_batch)
    dist_loss, _ = _get_loss_and_grads(dist_mlp, dist_batch)
    print(full_loss, dist_loss)
