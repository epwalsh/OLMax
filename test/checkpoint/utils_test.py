import jax
import jax.numpy as jnp
import pytest
from jax.sharding import NamedSharding

import olmax.checkpoint.utils as checkpoint_utils
import olmax.distributed as dist
import olmax.nn as nn
from olmax.testing.distributed import run_distributed_test


@pytest.mark.parametrize("block", [True, False])
def test_save_and_restore(tmp_path, block: bool):
    checkpoint_dir = tmp_path / "checkpoint"

    model = nn.Linear(2, 3, key=jax.random.PRNGKey(0))

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
    model2 = nn.Linear(2, 3, key=jax.random.PRNGKey(1))
    model2 = checkpoint_utils.restore(checkpoint_dir, model2)

    # Check that both model's are now equal.
    assert jnp.allclose(model.weight, model2.weight).item()
    assert model.bias is not None
    assert jnp.allclose(model.bias, model2.bias).item()


def _run_save_and_restore_distributed(
    checkpoint_dir, save_sharding: NamedSharding, load_sharding: NamedSharding
):
    in_size = dist.get_global_device_count() * 4
    out_size = dist.get_global_device_count() * 2

    # Initialize sharded model.
    model = nn.Linear(
        in_size,
        out_size,
        key=jax.random.PRNGKey(0),
        weight_sharding=save_sharding,
        bias_sharding=save_sharding,
    )

    # Save checkpoint.
    checkpoint_utils.save(checkpoint_dir, model)

    # Restore from model.
    model2 = nn.Linear(
        in_size,
        out_size,
        key=jax.random.PRNGKey(1),
        weight_sharding=load_sharding,
        bias_sharding=load_sharding,
    )
    model2 = checkpoint_utils.restore(checkpoint_dir, model2)

    # Check that both model's are now equal.
    assert jnp.allclose(model.weight, model2.weight).item()
    assert model.bias is not None
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
