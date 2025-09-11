import functools as ft
from dataclasses import dataclass
from typing import ClassVar, Type

import equinox as eqx
import jax
import jax.numpy as jnp

from ..distributed.parallel import MeshResource
from ..types import Array, DTypeLike, PRNGKeyArray, PyTree
from .module import Module


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

    @jax.named_scope("olmax.nn.RotaryPositionalEmbedding")
    def forward(
        self, x: Array, *, head_first: bool = False, buffer_cache: dict[str, PyTree] | None = None
    ) -> Array:
        assert x.ndim == 4
        head_dim = 3 if head_first else 2
        seq_len = x.shape[1]
        cache_key = self.get_buffer_cache_key(seq_len)
        if buffer_cache is None or cache_key not in buffer_cache:
            freqs_cos, freqs_sin = self.get_freqs_cis(seq_len)
        else:
            freqs_cos, freqs_sin = buffer_cache[cache_key]
        x = jax.vmap(self._apply_rope, (head_dim, None, None), head_dim)(x, freqs_cos, freqs_sin)
        return x

    def get_freqs_cis(self, sequence_length: int) -> tuple[Array, Array]:
        return precompute_freqs_cis(
            head_dim=self.head_dim,
            end=sequence_length,
            theta=self.theta,
            dtype=self.dtype,
            mesh_resource=self.mesh_resource,
        )

    def get_buffer_cache_key(self, sequence_length: int) -> str:
        return f"RoPE_freqs_cis,H={self.head_dim},S={sequence_length},theta={int(self.theta)},dtype={self.dtype}"

    def update_buffer_cache(self, cache: dict[str, PyTree], sequence_length: int):
        key = self.get_buffer_cache_key(sequence_length)
        if key not in cache or cache[key][0].shape[0] < sequence_length:
            cache[key] = self.get_freqs_cis(sequence_length)

    def _apply_rope(self, x: Array, freqs_cos: Array, freqs_sin: Array) -> Array:
        H = x.shape[-1]
        og_dtype = x.dtype
        if H != self.head_dim:
            raise ValueError(
                f"x.shape[-1] must match self.head_dim, but {x.shape[-1]} != {self.head_dim}"
            )

        x = x.astype(self.dtype)

        freqs_cos = jnp.tile(jnp.expand_dims(freqs_cos, 0), (1, 2))
        freqs_sin = jnp.tile(jnp.expand_dims(freqs_sin, 0), (1, 2))

        rotate_x = self.rotate_half(x)
        x_rope = (x * freqs_cos) + (rotate_x * freqs_sin)

        return x_rope.astype(og_dtype)


@ft.partial(jax.jit, static_argnames=("head_dim", "end", "dtype", "mesh_resource"))
def precompute_freqs_cis(
    *,
    head_dim: int,
    end: int,
    theta: float,
    dtype: DTypeLike,
    mesh_resource: MeshResource | None = None,
) -> tuple[Array, Array]:
    freqs = 1.0 / (theta ** (jnp.arange(0.0, head_dim, 2)[jnp.newaxis, :] / head_dim))

    t = jnp.arange(float(end))
    freqs_outer = jnp.outer(t, freqs)

    # Assign the type at the very end to minimize the loss of precision.
    freqs_cos, freqs_sin = jnp.cos(freqs_outer).astype(dtype), jnp.sin(freqs_outer).astype(dtype)

    if mesh_resource is not None and mesh_resource.cp_sharding_axis is not None:
        freqs_cos = mesh_resource.reorder_cp_array_for_causal_load_balancing(freqs_cos, 1)
        freqs_cos = mesh_resource.with_data_sharding_constraint(
            freqs_cos, batch_dim=None, sequence_dim=1
        )
        freqs_sin = mesh_resource.reorder_cp_array_for_causal_load_balancing(freqs_sin, 1)
        freqs_sin = mesh_resource.with_data_sharding_constraint(
            freqs_sin, batch_dim=None, sequence_dim=1
        )

    return freqs_cos, freqs_sin
