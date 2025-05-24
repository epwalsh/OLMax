import jax
import jax.numpy as jnp
import pytest
from jaxtyping import Array, PRNGKeyArray

import olmax.checkpoint as checkpoint
import olmax.nn as nn


@pytest.mark.parametrize("block", [True, False])
def test_save_and_restore(tmp_path, block: bool):
    checkpoint_dir = tmp_path / "checkpoint"

    class Linear(nn.Module):
        weight: Array
        bias: Array

        def __init__(self, in_size: int, out_size: int, key: PRNGKeyArray):
            wkey, bkey = jax.random.split(key)
            self.weight = jax.random.normal(wkey, (out_size, in_size))
            self.bias = jax.random.normal(bkey, (out_size,))

        def __call__(self, x: Array) -> Array:
            return self.weight @ x + self.bias

    model = Linear(2, 3, key=jax.random.PRNGKey(0))

    # Save checkpoint.
    handle = checkpoint.save(checkpoint_dir, model, block=block)
    if not block:
        assert handle is not None
        handle.wait()
        handle.close()

    # Get metadata about checkpoint.
    metadata = checkpoint.get_metadata(checkpoint_dir)

    # Restore from metadata.
    checkpoint.restore(checkpoint_dir, metadata)

    # Restore from model.
    model2 = Linear(2, 3, key=jax.random.PRNGKey(1))
    model2 = checkpoint.restore(checkpoint_dir, model2)

    # Check that both model's are now equal.
    assert jnp.allclose(model.weight, model2.weight).item()
    assert jnp.allclose(model.bias, model2.bias).item()
