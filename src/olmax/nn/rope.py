from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Type

import equinox as eqx
import jax
import jax.numpy as jnp

from ..caches import cache_clears
from ..distributed.parallel import ParallelConfig
from ..types import Array, DTypeLike, PRNGKeyArray
from .module import Module

internal_rope_embedding_cache: dict[tuple[int, DTypeLike], tuple[Array, Array]] = {}
cache_clears.append(internal_rope_embedding_cache.clear)


@dataclass
class RotaryPositionalEmbeddingConfig:
    theta: float = 10_000.0
    dtype: DTypeLike = float

    def build(
        self,
        *,
        head_dim: int,
        key: PRNGKeyArray,
        theta: float | None = None,
        dtype: DTypeLike | None = None,
    ) -> RotaryPositionalEmbedding:
        return RotaryPositionalEmbedding(
            head_dim=head_dim,
            key=key,
            theta=theta if theta is not None else self.theta,
            dtype=dtype if dtype is not None else self.dtype,
        )


class RotaryPositionalEmbedding(Module):
    Config: ClassVar[Type[RotaryPositionalEmbeddingConfig]] = RotaryPositionalEmbeddingConfig
    keepdims: ClassVar[int] = 2

    head_dim: int = eqx.field(static=True)
    theta: float = eqx.field(static=True, default=10_000.0)
    dtype: DTypeLike = eqx.field(static=True, default=float)

    def __init__(
        self,
        *,
        head_dim: int,
        key: PRNGKeyArray,
        theta: float = 10_000.0,
        dtype: DTypeLike = float,
        parallel_config: ParallelConfig | None = None,
    ):
        del key  # unused
        super().__init__(parallel_config)
        self.head_dim = head_dim
        self.theta = theta
        self.dtype = dtype

    @staticmethod
    def rotate_half(x: Array) -> Array:
        d_2 = x.shape[-1] // 2
        return jnp.concatenate([-x[..., d_2:], x[..., :d_2]], axis=-1)

    @staticmethod
    def precompute_freqs_cis(
        head_dim: int, end: int, theta: float, dtype: DTypeLike
    ) -> tuple[Array, Array]:
        freqs = 1.0 / (theta ** (jnp.arange(0.0, head_dim, 2)[jnp.newaxis, :] / head_dim))

        t = jnp.arange(float(end))
        freqs_outer = jnp.outer(t, freqs)

        # Assign the type at the very end to minimize the loss of precision.
        return jnp.cos(freqs_outer).astype(dtype), jnp.sin(freqs_outer).astype(dtype)

    @jax.named_scope("olmax.nn.RotaryPositionalEmbedding")
    def forward(self, x: Array) -> Array:
        seq_len, head_dim = x.shape
        og_dtype = x.dtype
        if head_dim != self.head_dim:
            raise ValueError(
                f"x.shape[-1] must match self.head_dim, but {x.shape[-1]} != {self.head_dim}"
            )

        x = x.astype(self.dtype)

        with jax.ensure_compile_time_eval():
            cache_key = (head_dim, self.dtype)
            if cache_key not in internal_rope_embedding_cache:
                internal_rope_embedding_cache[cache_key] = self.precompute_freqs_cis(
                    head_dim, seq_len, self.theta, self.dtype
                )

            freqs_cos, freqs_sin = internal_rope_embedding_cache[cache_key]
            freqs_seq_len, _ = freqs_cos.shape
            if seq_len > freqs_seq_len:
                internal_rope_embedding_cache[cache_key] = self.precompute_freqs_cis(
                    head_dim, seq_len, self.theta, self.dtype
                )
                freqs_cos, freqs_sin = internal_rope_embedding_cache[cache_key]

            freqs_cos = freqs_cos[:seq_len]
            freqs_sin = freqs_sin[:seq_len]

        freqs_cos = jnp.tile(freqs_cos, (1, 2))
        freqs_sin = jnp.tile(freqs_sin, (1, 2))

        rotate_x = self.rotate_half(x)
        x_rope = (x * freqs_cos) + (rotate_x * freqs_sin)

        return x_rope.astype(og_dtype)
