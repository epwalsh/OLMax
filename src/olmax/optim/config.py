from __future__ import annotations

import fnmatch
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any

import jax
import optax

from ..config import RegistrableConfig
from ..types import PyTree


@dataclass
class Schedule(RegistrableConfig):
    @abstractmethod
    def build(self) -> optax.ScalarOrSchedule:
        raise NotImplementedError


@Schedule.register_subclass("constant")
@dataclass
class ConstantSchedule:
    value: float

    def build(self) -> optax.ScalarOrSchedule:
        return self.value


@Schedule.register_subclass("warmup_cosine_decay")
@dataclass
class WarmupCosineDecaySchedule(Schedule):
    warmup_steps: int
    decay_steps: int
    peak_value: float
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


@Schedule.register_subclass("warmup_stable_decay")
@dataclass
class WarmupStableDecaySchedule(Schedule):
    warmup_steps: int
    stable_steps: int
    decay_steps: int
    peak_value: float
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
class OptimConfig(RegistrableConfig):
    @classmethod
    def AdamW(cls, *args, **kwargs) -> AdamWConfig:
        return AdamWConfig(*args, **kwargs)

    @abstractmethod
    def build(self, model: PyTree) -> tuple[optax.GradientTransformation, optax.OptState]:
        raise NotImplementedError

    @classmethod
    def build_weight_decay_mask(cls, model: PyTree, no_decay_modules: list[str]) -> PyTree:
        def should_decay(key_path: tuple[Any, ...], _: Any) -> bool:
            name = ".".join([kp.name for kp in key_path])
            for pattern in no_decay_modules:
                if fnmatch.fnmatch(name, pattern):
                    return False
            return True

        return jax.tree.map_with_path(should_decay, model)


@OptimConfig.register_subclass("adamw")
@dataclass
class AdamWConfig(OptimConfig):
    lr: Schedule
    b1: float = 0.9
    b2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 1e-4
    max_grad_norm: float | None = None
    no_decay_modules: list[str] | None = None

    def build(self, model: PyTree) -> tuple[optax.GradientTransformation, optax.OptState]:
        weight_decay_mask: PyTree | None = None
        if self.no_decay_modules:
            weight_decay_mask = self.build_weight_decay_mask(model, self.no_decay_modules)

        components: list[optax.GradientTransformation] = []
        if self.max_grad_norm is not None:
            components.append(optax.clip_by_global_norm(self.max_grad_norm))

        components.append(
            optax.adamw(
                self.lr.build(),
                b1=self.b1,
                b2=self.b2,
                eps=self.eps,
                weight_decay=self.weight_decay,
                mask=weight_decay_mask,
            )
        )

        optim = optax.chain(*components)

        return optim, optim.init(model)
