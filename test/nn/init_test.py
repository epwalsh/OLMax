import jax
import jax.numpy as jnp

import olmax.distributed as dist
import olmax.nn as nn
from olmax.testing.distributed import run_distributed_test


def test_truncated_normal():
    key = jax.random.PRNGKey(0)
    x = nn.init.truncated_normal(key, (2, 4))
    assert x.shape == (2, 4)


def _run_truncated_normal_distributed():
    # given the same key, should get same result regardless of how we shard
    key = jax.random.PRNGKey(0)
    x_full = nn.init.truncated_normal(key, (4, 8))
    x_sharded = nn.init.truncated_normal(
        key, (4, 8), sharding=dist.get_hsdp_sharding(2)
    )
    assert jnp.allclose(x_full, x_sharded).item()


def test_truncated_normal_distributed():
    run_distributed_test(_run_truncated_normal_distributed)
