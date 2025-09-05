from dataclasses import dataclass
from typing import ClassVar, Type

import equinox as eqx
import jax
import jax.numpy as jnp

from ..caches import cache_clears
from ..distributed.parallel import MeshResource
from ..types import Array, DTypeLike, PRNGKeyArray
from .module import Module

internal_rope_embedding_cache: dict[tuple[int, float, DTypeLike], tuple[Array, Array]] = {}
cache_clears.append(internal_rope_embedding_cache.clear)


@dataclass
class RotaryPositionalEmbeddingConfig:
    theta: float = 10_000.0
    dtype: DTypeLike = "float32"

    def build(
        self,
        head_dim: int,
        key: PRNGKeyArray,
        *,
        theta: float | None = None,
        dtype: DTypeLike | None = None,
        mesh_resource: MeshResource | None = None,
        checkpoint_name: str | None = None,
    ) -> "RotaryPositionalEmbedding":
        return RotaryPositionalEmbedding(
            head_dim=head_dim,
            key=key,
            theta=theta if theta is not None else self.theta,
            dtype=dtype if dtype is not None else self.dtype,
            mesh_resource=mesh_resource,
            checkpoint_name=checkpoint_name,
        )


class RotaryPositionalEmbedding(Module):
    Config: ClassVar[Type[RotaryPositionalEmbeddingConfig]] = RotaryPositionalEmbeddingConfig

    head_dim: int = eqx.field(static=True)
    theta: float = eqx.field(static=True, default=10_000.0)
    dtype: DTypeLike = eqx.field(static=True, default=float)

    def __init__(
        self,
        head_dim: int,
        key: PRNGKeyArray,
        *,
        theta: float = 10_000.0,
        dtype: DTypeLike = "float32",
        mesh_resource: MeshResource | None = None,
        checkpoint_name: str | None = None,
    ):
        del key  # unused
        super().__init__(mesh_resource, checkpoint_name)
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
    def forward(self, x: Array, head_first: bool = False) -> Array:
        assert x.ndim == 4
        head_dim = 3 if head_first else 2
        x = jax.vmap(self._apply_rope, head_dim, head_dim)(x)
        return x

    def _apply_rope(self, x: Array) -> Array:
        _, S, H = x.shape
        og_dtype = x.dtype
        if H != self.head_dim:
            raise ValueError(
                f"x.shape[-1] must match self.head_dim, but {x.shape[-1]} != {self.head_dim}"
            )

        x = x.astype(self.dtype)

        with jax.ensure_compile_time_eval():
            cache_key = (H, self.theta, self.dtype)
            if cache_key not in internal_rope_embedding_cache:
                internal_rope_embedding_cache[cache_key] = self.precompute_freqs_cis(
                    H, S, self.theta, self.dtype
                )

            freqs_cos, freqs_sin = internal_rope_embedding_cache[cache_key]
            freqs_seq_len, _ = freqs_cos.shape
            if S > freqs_seq_len:
                internal_rope_embedding_cache[cache_key] = self.precompute_freqs_cis(
                    H, S, self.theta, self.dtype
                )
                freqs_cos, freqs_sin = internal_rope_embedding_cache[cache_key]

            freqs_cos = freqs_cos[:, :S]
            freqs_sin = freqs_sin[:, :S]
            if self.mesh_resource is not None and self.mesh_resource.cp_sharding_axis is not None:
                freqs_cos = self.mesh_resource.reorder_cp_array_for_causal_load_balancing(
                    freqs_cos, 1
                )
                freqs_cos = self.mesh_resource.with_data_sharding_constraint(
                    freqs_cos, batch_dim=None, sequence_dim=1
                )
                freqs_sin = self.mesh_resource.reorder_cp_array_for_causal_load_balancing(
                    freqs_sin, 1
                )
                freqs_sin = self.mesh_resource.with_data_sharding_constraint(
                    freqs_sin, batch_dim=None, sequence_dim=1
                )

        freqs_cos = jnp.tile(jnp.expand_dims(freqs_cos, 0), (1, 2))
        freqs_sin = jnp.tile(jnp.expand_dims(freqs_sin, 0), (1, 2))

        rotate_x = self.rotate_half(x)
        x_rope = (x * freqs_cos) + (rotate_x * freqs_sin)

        return x_rope.astype(og_dtype)
