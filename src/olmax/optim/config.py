from __future__ import annotations

import dataclasses
import fnmatch
import logging
import warnings
from abc import abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import jax
import optax

from ..config import Registrable
from ..types import PyTree

log = logging.getLogger(__name__)


@dataclass
class Schedule(Registrable):
    @abstractmethod
    def build(self) -> optax.ScalarOrSchedule:
        raise NotImplementedError


@Schedule.register("constant")
@dataclass
class ConstantSchedule(Schedule):
    value: float = 0.0

    def build(self) -> optax.ScalarOrSchedule:
        return self.value


@Schedule.register("warmup_cosine_decay")
@dataclass
class WarmupCosineDecaySchedule(Schedule):
    warmup_steps: int = 0
    decay_steps: int = 0
    peak_value: float = 0.0
    init_value: float = 0.0
    end_value: float = 0.0

    def build(self) -> optax.ScalarOrSchedule:
        return optax.warmup_cosine_decay_schedule(
            init_value=self.init_value,
            peak_value=self.peak_value,
            warmup_steps=self.warmup_steps,
            decay_steps=self.decay_steps
            + self.warmup_steps,  # optax treats decay_steps as total steps
            end_value=self.end_value,
        )


@Schedule.register("warmup_stable_decay")
@dataclass
class WarmupStableDecaySchedule(Schedule):
    warmup_steps: int = 0
    stable_steps: int = 0
    decay_steps: int = 0
    peak_value: float = 0.0
    init_value: float = 0.0
    end_value: float = 0.0

    def build(self) -> optax.ScalarOrSchedule:
        return optax.join_schedules(
            [
                optax.warmup_constant_schedule(
                    init_value=self.init_value,
                    peak_value=self.peak_value,
                    warmup_steps=self.warmup_steps,
                ),
                optax.linear_schedule(
                    init_value=self.peak_value,
                    end_value=self.end_value,
                    transition_steps=self.decay_steps,
                ),
            ],
            [self.warmup_steps + self.stable_steps],
        )


@dataclass
class OptimConfig(Registrable):
    @classmethod
    def AdamW(cls, *args, **kwargs) -> AdamWConfig:
        return AdamWConfig(*args, **kwargs)

    @abstractmethod
    def build(self, model: PyTree) -> tuple[optax.GradientTransformation, optax.OptState]:
        raise NotImplementedError

    @classmethod
    def build_weight_decay_mask(cls, model: PyTree, no_decay_modules: list[str]) -> PyTree:
        not_decaying: dict[str, list[str]] = defaultdict(list)

        def should_decay(key_path: tuple[Any, ...], _: Any) -> bool:
            name_parts = []
            for kp in key_path:
                if isinstance(kp, jax.tree_util.GetAttrKey):
                    name_parts.append(kp.name)
                elif isinstance(kp, jax.tree_util.SequenceKey):
                    name_parts.append(str(kp.idx))
                else:
                    raise ValueError(kp)
            name = ".".join(name_parts)
            for pattern in no_decay_modules:
                if fnmatch.fnmatch(name, pattern):
                    not_decaying[pattern].append(name)
                    return False
            return True

        result = jax.tree.map_with_path(should_decay, model)

        for pattern in no_decay_modules:
            if pattern not in not_decaying:
                warnings.warn(f"no decay pattern '{pattern}' did not match any parameters")
            else:
                for name in not_decaying[pattern]:
                    log.info(f"Weight decay will not be applied to parameter '{name}'")

        return result


@OptimConfig.register("adamw")
@dataclass
class AdamWConfig(OptimConfig):
    lr: Schedule = dataclasses.field(default_factory=ConstantSchedule)
    b1: float = 0.9
    b2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 1e-4
    no_decay_modules: list[str] | None = None

    def build(self, model: PyTree) -> tuple[optax.GradientTransformation, optax.OptState]:
        weight_decay_mask: PyTree | None = None
        if self.no_decay_modules:
            weight_decay_mask = self.build_weight_decay_mask(model, self.no_decay_modules)

        optim = optax.inject_hyperparams(optax.adamw, static_args=("mask",))(
            learning_rate=self.lr.build(),
            b1=self.b1,
            b2=self.b2,
            eps=self.eps,
            weight_decay=self.weight_decay,
            mask=weight_decay_mask,
        )

        return optim, optim.init(model)
