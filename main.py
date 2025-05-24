import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
from jaxtyping import Array, PRNGKeyArray

import olmax.checkpoint as checkpoint
import olmax.nn as nn


class Linear(nn.Module):
    weight: Array
    bias: Array

    def __init__(self, in_size: int, out_size: int, key: PRNGKeyArray):
        wkey, bkey = jax.random.split(key)
        self.weight = jax.random.normal(wkey, (out_size, in_size))
        self.bias = jax.random.normal(bkey, (out_size,))

    def __call__(self, x: Array) -> Array:
        return self.weight @ x + self.bias


@eqx.filter_value_and_grad
def compute_loss(model: Linear, x: Array, y: Array) -> Array:
    pred_y = jax.vmap(model)(x)
    return jnp.mean((y - pred_y) ** 2)


if __name__ == "__main__":
    batch_size, in_size, out_size = 32, 2, 3
    checkpoint_dir = "/tmp/olmax-checkpoint"

    model = Linear(in_size, out_size, key=jax.random.PRNGKey(0))
    print(model)

    checkpoint.save(checkpoint_dir, model, force=True)
    metadata = checkpoint.get_metadata(checkpoint_dir)
    checkpoint.restore(checkpoint_dir, metadata)
    checkpoint.restore(checkpoint_dir, model)

    optim = optax.adamw(1e-4)
    opt_state = optim.init(model)

    @eqx.filter_jit
    def make_step(model: Linear, x: Array, y: Array, opt_state):
        loss, grads = compute_loss(model, x, y)
        updates, opt_state = optim.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state

    x = jnp.zeros((batch_size, in_size))
    y = jnp.zeros((batch_size, out_size))
    loss, model, opt_state = make_step(model, x, y, opt_state)
    print(loss.item())
