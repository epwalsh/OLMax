from typing import Callable, cast

import jax

import olmax.nn as nn
from olmax.optim.config import (
    AdamWConfig,
    ConstantSchedule,
    OptimConfig,
    WarmupStableDecaySchedule,
)


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


def main():
    key = jax.random.PRNGKey(0)
    model = nn.GatedMLP(4, 8, key)
    optim, opt_state = AdamWConfig(
        lr=ConstantSchedule(value=1e-2),
        #  lr=WarmupStableDecaySchedule(warmup_steps=10, stable_steps=10, decay_steps=10, peak_value=1e-2)
    ).build(model)
    print(optim)
    print(opt_state)


if __name__ == "__main__":
    main()
