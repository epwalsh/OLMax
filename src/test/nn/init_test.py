import jax

import olmax.distributed as dist
import olmax.nn as nn
from olmax.testing.distributed import run_distributed_test
from olmax.testing.utils import allclose


def test_truncated_normal():
    key = jax.random.PRNGKey(0)
    x = nn.init.truncated_normal(key, (2, 4))
    assert x.shape == (2, 4)


def _run_truncated_normal_mp():
    pc = dist.MeshResource.HSDP(2)
    # given the same key, should get same result regardless of how we shard
    key = jax.random.PRNGKey(0)
    x_full = nn.init.truncated_normal(key, (4, 8))
    x_sharded = nn.init.truncated_normal(key, (4, 8), sharding=pc.get_param_sharding())
    assert allclose(x_full, x_sharded)


def test_truncated_normal_mp():
    run_distributed_test(_run_truncated_normal_mp, num_processes=1, devices_per_process=2)
