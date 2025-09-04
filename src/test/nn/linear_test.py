import functools as ft

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


@eqx.filter_jit
@eqx.filter_value_and_grad
def _get_loss_and_grads(model: nn.Linear, batch: tuple[Array, Array]) -> Array:
    inputs, targets = batch
    predictions = model(inputs)
    return jnp.mean(jnp.sum((predictions - targets) ** 2, axis=-1))


def test_linear():
    key = jax.random.PRNGKey(0)
    key, batch_key = jax.random.split(key)
    batch = _get_batch(batch_key, 2, 4, 4)
    linear = nn.Linear(4, 4, key=key)
    assert linear.training
    loss, grads = _get_loss_and_grads(linear, batch)
    assert loss is not None
    assert grads is not None
    assert not linear.eval().training


def test_linear_3d():
    batch_size, seq_len, in_size, out_size = 2, 12, 8, 4
    key = jax.random.PRNGKey(0)
    key, batch_key = jax.random.split(key)
    inputs = jax.random.normal(batch_key, (batch_size, seq_len, in_size))
    linear = nn.Linear(in_size, out_size, key=key)
    out = linear(inputs)
    assert out.shape == (batch_size, seq_len, out_size)


def _run_linear_data_parallel(mesh_resource: dist.MeshResource):
    in_size, out_size, batch_size = (
        2 * dist.get_global_device_count(),
        2 * dist.get_global_device_count(),
        2 * dist.get_global_device_count(),
    )
    key = jax.random.PRNGKey(0)

    key, batch_key = jax.random.split(key)
    full_batch = _get_batch(batch_key, batch_size, in_size, out_size)
    dist_batch = jax.device_put(full_batch, mesh_resource.get_data_sharding_for(full_batch[0]))

    full_linear = nn.Linear(in_size, out_size, key=key)
    dist_linear = nn.Linear(in_size, out_size, key=key, mesh_resource=mesh_resource)

    assert allclose(full_linear.weight, dist_linear.weight)
    assert allclose(full_linear.bias, dist_linear.bias)

    full_loss, full_grads = _get_loss_and_grads(full_linear, full_batch)
    dist_loss, dist_grads = _get_loss_and_grads(dist_linear, dist_batch)
    assert allclose(full_loss, dist_loss)
    assert allclose(full_grads.weight, dist_grads.weight)
    assert allclose(full_grads.bias, dist_grads.bias)


@pytest.mark.parametrize(
    "mesh_resource",
    [
        pytest.param(dist.MeshResource.FSDP(2), id="FSDP"),
        pytest.param(dist.MeshResource.DDP(2), id="DDP"),
        pytest.param(dist.MeshResource.HSDP(shard_degree=2, global_device_count=4), id="HSDP"),
    ],
)
def test_linear_data_parallel(mesh_resource: dist.MeshResource):
    run_distributed_test(
        _run_linear_data_parallel,
        num_processes=1,
        devices_per_process=mesh_resource.size,
        args=(mesh_resource,),
    )


def _run_linear_manual_sharding(mesh_resource: dist.MeshResource):
    assert isinstance(mesh_resource.fsdp_sharding_axis, str)

    in_size, out_size, batch_size = (
        2 * dist.get_global_device_count(),
        2 * dist.get_global_device_count(),
        2 * dist.get_global_device_count(),
    )
    key = jax.random.PRNGKey(0)

    key, batch_key = jax.random.split(key)
    full_batch = _get_batch(batch_key, batch_size, in_size, out_size)
    dist_batch = jax.device_put(full_batch, mesh_resource.get_data_sharding_for(full_batch[0]))

    full_linear = nn.Linear(in_size, out_size, key=key)
    dist_linear = nn.Linear(in_size, out_size, key=key, mesh_resource=mesh_resource)

    assert allclose(full_linear.weight, dist_linear.weight)
    assert allclose(full_linear.bias, dist_linear.bias)

    @jax.jit
    @ft.partial(
        mesh_resource.shard_map,
        in_specs=(
            dist_linear.get_param_partitions(),
            (
                mesh_resource.get_data_partition_for(full_batch[0]),
                mesh_resource.get_data_partition_for(full_batch[1]),
            ),
        ),
        out_specs=(mesh_resource.get_replicated_partition(), dist_linear.get_param_partitions()),
    )
    def get_dist_loss_and_grads(
        model: nn.Linear, batch: tuple[Array, Array]
    ) -> tuple[Array, Array]:
        assert isinstance(mesh_resource.fsdp_sharding_axis, str)
        loss, grads = _get_loss_and_grads(
            mesh_resource.all_gather(model, mesh_resource.fsdp_sharding_axis), batch
        )

        for axis in mesh_resource.axis_names:
            loss = jax.lax.pmean(loss, axis)

        grads = mesh_resource.reduce_scatter(grads, mesh_resource.fsdp_sharding_axis)

        if mesh_resource.has_axis("fsdp_replicate"):
            # NOTE: JAX will have already ensured the grads are replicated across this axis
            # (by summation via psum), but we need that to be an average so we just divide here by the
            # size of that axis.
            #
            # So instead of doing this:
            #  grads = mesh_resource.all_reduce(grads, dist.MeshAxesNames.DP.replicate)
            #
            # We just do this:
            grads = jax.tree.map(lambda x: x / mesh_resource.axis_size("fsdp_replicate"), grads)

        return loss, grads

    full_loss, full_grads = _get_loss_and_grads(full_linear, full_batch)
    dist_loss, dist_grads = get_dist_loss_and_grads(dist_linear, dist_batch)

    assert allclose(full_loss, dist_loss)
    assert allclose(
        full_grads.weight, dist_grads.weight
    ), f"full_grads: {full_grads.weight}\ndist_grads: {dist_grads.weight}"
    assert allclose(full_grads.bias, dist_grads.bias)


@pytest.mark.parametrize(
    "mesh_resource",
    [
        pytest.param(dist.MeshResource.FSDP(2), id="FSDP"),
        pytest.param(dist.MeshResource.HSDP(shard_degree=4, global_device_count=8), id="HSDP"),
    ],
)
def test_linear_manual_sharding(mesh_resource: dist.MeshResource):
    run_distributed_test(
        _run_linear_manual_sharding,
        num_processes=1,
        devices_per_process=mesh_resource.size,
        args=(mesh_resource,),
    )


def main():
    mesh_resource = dist.MeshResource.HSDP(2, 2)

    key = jax.random.PRNGKey(0)
    in_size, out_size, batch_size = (
        2 * dist.get_global_device_count(),
        2 * dist.get_global_device_count(),
        2 * dist.get_global_device_count(),
    )
    print(f"{in_size=}, {out_size}")

    key, batch_key = jax.random.split(key)
    batch = _get_batch(batch_key, batch_size, in_size, out_size)
    batch = jax.device_put(batch, mesh_resource.get_data_sharding_for(batch[0]))

    model = nn.Linear(in_size, out_size, key=key, mesh_resource=mesh_resource)

    @jax.jit
    @ft.partial(
        mesh_resource.shard_map,
        in_specs=(
            model.get_param_partitions(),
            (
                mesh_resource.get_data_partition_for(batch[0]),
                mesh_resource.get_data_partition_for(batch[1]),
            ),
        ),
        out_specs=(mesh_resource.get_replicated_partition(), model.get_param_partitions()),
    )
    def get_dist_loss_and_grads(
        model: nn.Linear, batch: tuple[Array, Array]
    ) -> tuple[Array, Array]:
        loss, grads = _get_loss_and_grads(
            mesh_resource.all_gather(model, dist.MeshAxesNames.DP.shard), batch
        )

        for axis in (dist.MeshAxesNames.DP.replicate, dist.MeshAxesNames.DP.shard):
            if mesh_resource.has_axis(axis):
                loss = jax.lax.pmean(loss, axis)

        if mesh_resource.has_axis(dist.MeshAxesNames.DP.shard):
            grads = mesh_resource.reduce_scatter(grads, dist.MeshAxesNames.DP.shard)

        if mesh_resource.has_axis(dist.MeshAxesNames.DP.replicate):
            #  grads = mesh_resource.all_reduce(grads, dist.MeshAxesNames.DP.replicate)
            grads = jax.tree.map(
                lambda x: x / mesh_resource.axis_size(dist.MeshAxesNames.DP.replicate), grads
            )

        return loss, grads

    print(jax.make_jaxpr(get_dist_loss_and_grads)(model, batch))

    loss, grads = get_dist_loss_and_grads(model, batch)
    print(loss)
    print(grads.weight)

    #  jax.debug.visualize_array_sharding(model.weight)
    #  jax.debug.visualize_array_sharding(grads.weight)


if __name__ == "__main__":
    jax.config.update("jax_num_cpu_devices", 4)
    #  jax.config.update("jax_disable_jit", True)
    main()
