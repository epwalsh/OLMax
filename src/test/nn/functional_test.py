import jax.numpy as jnp

import olmax.nn.functional as F
from olmax.testing.utils import allclose


def test_cross_entropy_loss():
    assert allclose(
        F.cross_entropy_loss(jnp.array([0.58, 0.79]), jnp.array([0, 1])),
        F.cross_entropy_loss(jnp.array([0.58, 0.79, 1.3]), jnp.array([0, 1, -100])),
    )
