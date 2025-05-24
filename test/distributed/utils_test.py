import olmax.distributed.utils as dist
from olmax.testing.distributed import run_distributed_test


def run_test_utils():
    dist.barrier("test init")
    assert dist.is_distributed()
    assert dist.get_process_world_size() > 1
    assert 0 <= dist.get_process_rank() < dist.get_process_world_size()

    assert (
        dist.synchronize_value(True if dist.get_process_rank() == 0 else False) is True
    )
    assert dist.synchronize_value(2.0 if dist.get_process_rank() == 0 else 0.0) == 2.0
    assert dist.synchronize_value(-1 if dist.get_process_rank() == 0 else 0) == -1


def test_utils():
    run_distributed_test(run_test_utils, world_size=2)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    test_utils()
