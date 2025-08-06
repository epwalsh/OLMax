import jax
import pytest

import olmax.checkpoint.utils as checkpoint_utils
import olmax.distributed as dist
import olmax.nn as nn
from olmax.testing.distributed import run_distributed_test
from olmax.testing.utils import allclose


@pytest.mark.parametrize("block", [True, False])
def test_save_and_restore(tmp_path, block: bool):
    checkpoint_dir = tmp_path / "checkpoint"

    model = nn.Linear(2, 3, key=jax.random.PRNGKey(0))

    # Save checkpoint.
    handle = checkpoint_utils.save(checkpoint_dir, model, block=block)
    if not block:
        assert handle is not None
        handle.wait_until_finished()
        handle.close()

    # Get metadata about checkpoint.
    metadata = checkpoint_utils.get_metadata(checkpoint_dir)

    # Restore from metadata.
    checkpoint_utils.restore_from_metadata(checkpoint_dir, metadata)

    # Restore from model.
    model2 = nn.Linear(2, 3, key=jax.random.PRNGKey(1))
    model2 = checkpoint_utils.restore(checkpoint_dir, model2)

    # Check that both model's are now equal.
    assert allclose(model, model2)


def _run_save_and_restore_distributed(
    checkpoint_dir,
    save_sharding: dist.MeshResource,
    load_sharding: dist.MeshResource,
):
    in_size = dist.get_global_device_count() * 4
    out_size = dist.get_global_device_count() * 2

    # Initialize sharded model.
    model = nn.Linear(
        in_size,
        out_size,
        key=jax.random.PRNGKey(0),
        mesh_resource=save_sharding,
    )

    # Save checkpoint.
    checkpoint_utils.save(checkpoint_dir, model)

    # Restore from model.
    model2 = nn.Linear(
        in_size,
        out_size,
        key=jax.random.PRNGKey(1),
        mesh_resource=load_sharding,
    )
    model2 = checkpoint_utils.restore(checkpoint_dir, model2)

    # Check that both model's are now equal.
    assert allclose(model.weight, model2.weight)
    assert model.bias is not None
    assert allclose(model.bias, model2.bias)


def _run_save_and_restore_distributed_fsdp(checkpoint_dir):
    _run_save_and_restore_distributed(
        checkpoint_dir,
        dist.MeshResource.FSDP(),
        dist.MeshResource.FSDP(),
    )


def test_save_and_restore_distributed_fsdp(tmp_path):
    checkpoint_dir = tmp_path / "checkpoint"
    run_distributed_test(_run_save_and_restore_distributed_fsdp, args=(checkpoint_dir,))


def _run_save_and_restore_distributed_hsdp(checkpoint_dir):
    assert dist.get_global_device_count() == 4
    _run_save_and_restore_distributed(
        checkpoint_dir,
        dist.MeshResource.HSDP(2),
        dist.MeshResource.HSDP(2),
    )


def test_save_and_restore_distributed_hsdp(tmp_path):
    checkpoint_dir = tmp_path / "checkpoint"
    run_distributed_test(
        _run_save_and_restore_distributed_hsdp, num_processes=4, args=(checkpoint_dir,)
    )


def _run_save_and_restore_distributed_fsdp_to_hsdp(checkpoint_dir):
    assert dist.get_global_device_count() == 4
    _run_save_and_restore_distributed(
        checkpoint_dir,
        dist.MeshResource.FSDP(),
        dist.MeshResource.HSDP(2),
    )


def test_save_and_restore_distributed_fsdp_to_hsdp(tmp_path):
    checkpoint_dir = tmp_path / "checkpoint"
    run_distributed_test(
        _run_save_and_restore_distributed_fsdp_to_hsdp,
        num_processes=4,
        args=(checkpoint_dir,),
    )
