from typing import Callable, cast

import equinox as eqx
import jax
import jax.numpy as jnp

import olmax.nn as nn
from olmax.optim.config import (
    AdamWConfig,
    ConstantSchedule,
    OptimConfig,
    WarmupStableDecaySchedule,
)
from olmax.optim.utils import (
    ClipByGlobalNormState,
    extract_hyperparameter,
    extract_state,
)
from olmax.types import *


def test_warmup_stable_decay_schedule():
    scheduler = WarmupStableDecaySchedule(
        warmup_steps=10, stable_steps=10, decay_steps=10, peak_value=1.0
    ).build()
    scheduler = cast(Callable[[int], float], scheduler)
    results = list(scheduler(step) for step in range(31))
    assert results[0] == 0.0
    assert results[5] == 0.5
    assert results[10] == 1.0
    assert results[20] == 1.0
    assert results[25] == 0.5
    assert results[30] == 0.0


def test_build_weight_decay_mask():
    key = jax.random.PRNGKey(0)
    model = nn.GatedMLP(4, 8, key)
    weight_decay_mask = OptimConfig.build_weight_decay_mask(
        model, ["gate_proj.bias", "down_proj.bias", "up_proj.bias"]
    )
    assert weight_decay_mask.gate_proj.weight is True
    assert weight_decay_mask.gate_proj.bias is False
    assert weight_decay_mask.down_proj.weight is True
    assert weight_decay_mask.down_proj.bias is False
    assert weight_decay_mask.up_proj.weight is True
    assert weight_decay_mask.up_proj.bias is False


@jax.jit
@eqx.filter_value_and_grad
def _get_loss_and_grads(model: nn.Module, x: Array, y: Array):
    preds = model(x)
    return jnp.mean(jnp.sum((preds - y) ** 2, axis=-1))


def test_adamw_with_microbatches_and_clipping():
    key = jax.random.PRNGKey(0)
    model_key, x_key, y_key = jax.random.split(key, 3)

    model = nn.GatedMLP(4, 8, model_key)
    optim, opt_state = AdamWConfig(
        lr=ConstantSchedule(value=1e-2),
        max_grad_norm=1.0,
    ).build(model, 2)

    for i in range(2):
        x_key = jax.random.fold_in(x_key, i)
        y_key = jax.random.fold_in(y_key, i)
        x = jax.random.normal(x_key, (2, 4))
        y = jax.random.normal(y_key, (2, 4))

        _, grads = _get_loss_and_grads(model, x, y)
        updates, opt_state = optim.update(grads, opt_state, model)  # pyright: ignore
        model = eqx.apply_updates(model, updates)

    lr = extract_hyperparameter(opt_state, "learning_rate")
    assert isinstance(lr, Array)
    clipping_state = extract_state(opt_state, ClipByGlobalNormState)
    assert clipping_state is not None
    assert clipping_state.global_norm > 0


if __name__ == "__main__":
    test_adamw_with_microbatches_and_clipping()
