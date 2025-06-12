import jax
import jax.numpy as jnp

import olmax.nn as nn
import olmax.nn.functional as F
from olmax.testing.utils import allclose
from olmax.types import *


def test_cross_entropy_loss():
    inputs = [[-1.24, 0.79], [2.91, -0.23]]
    labels = [0, 1]
    loss1 = F.cross_entropy_loss(jnp.array([inputs]), jnp.array([labels]))
    loss2 = F.cross_entropy_loss(jnp.array([inputs + [[0.27, 1.3]]]), jnp.array([labels + [-100]]))
    assert allclose(
        loss1,
        loss2,
    ), f"{loss1} != {loss2}"


def test_fused_cross_entropy_loss():
    inputs = [[-1.24, 0.79], [2.91, -0.23], [0.27, 1.3]]
    labels = [0, 1, -100]

    expected_loss = F.cross_entropy_loss(jnp.array([inputs]), jnp.array([labels]))
    loss, z_loss = F.fused_cross_entropy_loss(jnp.array([inputs]), jnp.array([labels]))

    assert allclose(expected_loss, loss)
    assert (z_loss >= 0).all().item()


def test_scan_module():
    def build_linear(key: PRNGKeyArray) -> nn.Linear:
        return nn.Linear(4, 4, key)

    key1, key2, data_key = jax.random.split(jax.random.key(0), 3)
    combined_key = jnp.stack([key1, key2])

    linear1 = build_linear(key1)
    linear2 = build_linear(key2)
    stacked_linear = jax.vmap(build_linear)(key=combined_key)

    x = jax.random.normal(data_key, (2, 4))

    out = linear2(linear1(x))
    scanned_out = F.scan_module(stacked_linear, x)
    assert allclose(out, scanned_out)
