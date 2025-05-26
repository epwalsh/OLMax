import olmax.distributed.utils as dist_utils
from olmax.testing.distributed import run_distributed_test


def run_test_utils():
    dist_utils.barrier("test init")
    assert dist_utils.is_distributed()
    assert dist_utils.get_process_world_size() > 1
    assert 0 <= dist_utils.get_process_rank() < dist_utils.get_process_world_size()
    assert dist_utils.get_global_device_count() >= dist_utils.get_process_world_size()
    assert dist_utils.get_local_device_count() < dist_utils.get_global_device_count()

    assert (
        dist_utils.synchronize_value(
            True if dist_utils.get_process_rank() == 0 else False
        )
        is True
    )
    assert (
        dist_utils.synchronize_value(2.0 if dist_utils.get_process_rank() == 0 else 0.0)
        == 2.0
    )
    assert (
        dist_utils.synchronize_value(-1 if dist_utils.get_process_rank() == 0 else 0)
        == -1
    )

    # NOTE: relies on the SHARED_FS_DIRS_ENV_VAR which is set by 'run_distributed_test'.
    assert dist_utils.get_process_filesystem_rank(".") == dist_utils.get_process_rank()


def test_utils():
    run_distributed_test(run_test_utils, world_size=2)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    test_utils()
