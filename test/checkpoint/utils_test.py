import jax
import jax.numpy as jnp
import pytest
from jax.sharding import NamedSharding
from jaxtyping import Array, PRNGKeyArray

import olmax.checkpoint.utils as checkpoint_utils
import olmax.distributed as dist
import olmax.nn as nn
from olmax.testing.distributed import run_distributed_test


class Linear(nn.Module):
    weight: Array
    bias: Array

    def __init__(
        self,
        in_size: int,
        out_size: int,
        key: PRNGKeyArray,
        sharding: NamedSharding | None = None,
    ):
        wkey, bkey = jax.random.split(key)
        weight = jax.random.normal(wkey, (out_size, in_size))
        bias = jax.random.normal(bkey, (out_size,))
        if sharding is not None:
            weight = jax.device_put(weight, sharding)
            bias = jax.device_put(bias, sharding)
        self.weight = weight
        self.bias = bias

    def __call__(self, x: Array) -> Array:
        return self.weight @ x + self.bias


@pytest.mark.parametrize("block", [True, False])
def test_save_and_restore(tmp_path, block: bool):
    checkpoint_dir = tmp_path / "checkpoint"

    model = Linear(2, 3, key=jax.random.PRNGKey(0))

    # Save checkpoint.
    handle = checkpoint_utils.save(checkpoint_dir, model, block=block)
    if not block:
        assert handle is not None
        handle.wait()
        handle.close()
        assert handle.done()

    # Get metadata about checkpoint.
    metadata = checkpoint_utils.get_metadata(checkpoint_dir)

    # Restore from metadata.
    checkpoint_utils.restore(checkpoint_dir, metadata)

    # Restore from model.
    model2 = Linear(2, 3, key=jax.random.PRNGKey(1))
    model2 = checkpoint_utils.restore(checkpoint_dir, model2)

    # Check that both model's are now equal.
    assert jnp.allclose(model.weight, model2.weight).item()
    assert jnp.allclose(model.bias, model2.bias).item()


def _run_save_and_restore_distributed(
    checkpoint_dir, save_sharding: NamedSharding, load_sharding: NamedSharding
):
    in_size = dist.get_global_device_count() * 4
    out_size = dist.get_global_device_count() * 2

    # Initialize sharded model.
    model = Linear(
        in_size,
        out_size,
        key=jax.random.PRNGKey(0),
        sharding=save_sharding,
    )

    # Save checkpoint.
    checkpoint_utils.save(checkpoint_dir, model)

    # Restore from model.
    model2 = Linear(
        in_size,
        out_size,
        key=jax.random.PRNGKey(1),
        sharding=load_sharding,
    )
    model2 = checkpoint_utils.restore(checkpoint_dir, model2)

    # Check that both model's are now equal.
    assert jnp.allclose(model.weight, model2.weight).item()
    assert jnp.allclose(model.bias, model2.bias).item()


def _run_save_and_restore_distributed_fsdp(checkpoint_dir):
    _run_save_and_restore_distributed(
        checkpoint_dir,
        dist.get_fsdp_sharding(),
        dist.get_fsdp_sharding(),
    )


def test_save_and_restore_distributed_fsdp(tmp_path):
    checkpoint_dir = tmp_path / "checkpoint"
    run_distributed_test(
        _run_save_and_restore_distributed_fsdp, world_size=2, args=(checkpoint_dir,)
    )


def _run_save_and_restore_distributed_hsdp(checkpoint_dir):
    assert dist.get_global_device_count() == 4
    _run_save_and_restore_distributed(
        checkpoint_dir,
        dist.get_hsdp_sharding(2),
        dist.get_hsdp_sharding(2),
    )


def test_save_and_restore_distributed_hsdp(tmp_path):
    checkpoint_dir = tmp_path / "checkpoint"
    run_distributed_test(
        _run_save_and_restore_distributed_hsdp, world_size=4, args=(checkpoint_dir,)
    )


def _run_save_and_restore_distributed_fsdp_to_hsdp(checkpoint_dir):
    assert dist.get_global_device_count() == 4
    _run_save_and_restore_distributed(
        checkpoint_dir,
        dist.get_fsdp_sharding(),
        dist.get_hsdp_sharding(2),
    )


def test_save_and_restore_distributed_fsdp_to_hsdp(tmp_path):
    checkpoint_dir = tmp_path / "checkpoint"
    run_distributed_test(
        _run_save_and_restore_distributed_fsdp_to_hsdp,
        world_size=4,
        args=(checkpoint_dir,),
    )
