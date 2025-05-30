import jax.numpy as jnp

import olmax.nn.functional as F
from olmax.testing.utils import allclose


def test_cross_entropy_loss():
    inputs = [[-1.24, 0.79], [2.91, -0.23]]
    labels = [0, 1]
    loss1 = F.cross_entropy_loss(jnp.array([inputs]), jnp.array([labels]))
    loss2 = F.cross_entropy_loss(jnp.array([inputs + [[0.27, 1.3]]]), jnp.array([labels + [-100]]))
    assert allclose(
        loss1,
        loss2,
    ), f"{loss1} != {loss2}"
