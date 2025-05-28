from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
from jax._src.dtypes import TypePromotionError

from ..caches import cache_clears
from ..distributed.parallel import ParallelConfig
from ..types import Array, DTypeLike, PRNGKeyArray
from .module import Module

internal_rope_embedding_cache: dict[tuple[int, DTypeLike], tuple[Array, Array]] = {}
cache_clears.append(internal_rope_embedding_cache.clear)


class RotaryPositionalEmbedding(Module):
    keepdims: ClassVar[int] = 2

    d_model: int = eqx.field(static=True)
    theta: float = eqx.field(static=True, default=10_000.0)
    dtype: DTypeLike = eqx.field(static=True, default=float)

    def __init__(
        self,
        *,
        d_model: int,
        theta: float = 10_000.0,
        key: PRNGKeyArray,
        dtype: DTypeLike = float,
        parallel_config: ParallelConfig | None = None,
    ):
        del key  # unused
        super().__init__(parallel_config)
        self.d_model = d_model
        self.theta = theta
        self.dtype = dtype

    @staticmethod
    def rotate_half(x: Array) -> Array:
        d_2 = x.shape[-1] // 2
        return jnp.concatenate([-x[..., d_2:], x[..., :d_2]], axis=-1)

    @staticmethod
    def precompute_freqs_cis(
        d_model: int, end: int, theta: float, dtype: DTypeLike
    ) -> tuple[Array, Array]:
        freqs = 1.0 / (theta ** (jnp.arange(0.0, d_model, 2)[jnp.newaxis, :] / d_model))

        t = jnp.arange(float(end))
        freqs_outer = jnp.outer(t, freqs)

        # we assign the type at the very end to minimize the loss of precision
        return jnp.cos(freqs_outer).astype(dtype), jnp.sin(freqs_outer).astype(dtype)

    @jax.named_scope("olmax.nn.RotaryPositionalEmbedding")
    def forward(self, x: Array) -> Array:
        seq_len, d_model = x.shape
        if d_model != self.d_model:
            raise ValueError(
                f"x.shape[-1] must match self.d_model, " f"but {x.shape[-1]} != {self.d_model}"
            )

        with jax.ensure_compile_time_eval():
            cache_key = (d_model, self.dtype)
            if cache_key not in internal_rope_embedding_cache:
                internal_rope_embedding_cache[cache_key] = self.precompute_freqs_cis(
                    d_model, seq_len, self.theta, self.dtype
                )

            freqs_cos, freqs_sin = internal_rope_embedding_cache[cache_key]
            freqs_seq_len, _ = freqs_cos.shape
            if seq_len > freqs_seq_len:
                internal_rope_embedding_cache[cache_key] = self.precompute_freqs_cis(
                    d_model, seq_len, self.theta, self.dtype
                )
                freqs_cos, freqs_sin = internal_rope_embedding_cache[cache_key]

            freqs_cos = freqs_cos[:seq_len]
            freqs_sin = freqs_sin[:seq_len]

        freqs_cos = jnp.tile(freqs_cos, (1, 2))
        freqs_sin = jnp.tile(freqs_sin, (1, 2))

        rotate_x = self.rotate_half(x)
        try:
            x_rope = (x * freqs_cos) + (rotate_x * freqs_sin)
        except TypePromotionError as e:
            inp_dtype = jnp.dtype(x.dtype)
            rope_dtype = jnp.dtype(self.dtype)
            raise TypePromotionError(
                f"The type of the passed value differs from the type "
                f"of the rotary embeddings ({inp_dtype} != {rope_dtype}), thus leading "
                f"to a conflict when numpy_dtype_promotion is set to strict. To avoid "
                f"this error, either initialiaze RoPE module with {inp_dtype} "
                f"dtype, or explicitly cast the input argument to {rope_dtype}."
            ) from e

        return x_rope.astype(x.dtype)
