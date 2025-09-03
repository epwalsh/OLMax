import functools as ft
from typing import Generator

import jax
import jax.numpy as jnp
from jax import random
from jax.sharding import NamedSharding

from ..distributed.parallel import MeshResource
from ..types import Array, PRNGKeyArray


@ft.partial(jax.jit, static_argnames=("sharding",))
def _randomize_start_offsets(
    *,
    microbatch: Array,
    vocab_size: int,
    key: PRNGKeyArray,
    sharding: NamedSharding | None,
) -> tuple[Array, Array]:
    num_instances, sequence_length = microbatch.shape
    start_offsets = random.randint(key, (num_instances, 1), 0, vocab_size - sequence_length - 1)
    inputs = microbatch + start_offsets
    targets = inputs + 1
    if sharding is not None:
        inputs = jax.lax.with_sharding_constraint(inputs, sharding)
        targets = jax.lax.with_sharding_constraint(targets, sharding)
    return inputs, targets


def generate_batches_of_sequential_tokens(
    key: PRNGKeyArray,
    *,
    vocab_size: int,
    sequence_length: int,
    global_batch_size_instances: int,
    total_batches: int,
    start_batch: int = 0,
    mesh_resource: MeshResource | None = None,
    num_microbatches: int = 1,
) -> Generator[list[tuple[Array, Array]], None, None]:
    if global_batch_size_instances % num_microbatches != 0:
        raise ValueError("global batch size must be divisible by the number of micro-batches")
    global_microbatch_size_instances = global_batch_size_instances // num_microbatches
    microbatch = (
        jnp.arange(0, sequence_length).reshape(1, -1).repeat(global_microbatch_size_instances, 0)
    )
    sharding = None if mesh_resource is None else mesh_resource.get_data_sharding_for(microbatch)
    for batch_idx in range(start_batch, total_batches):
        batch_key = jax.random.fold_in(key, batch_idx)
        batch = []
        for microbatch_idx in range(num_microbatches):
            microbatch_key = jax.random.fold_in(batch_key, microbatch_idx)
            batch.append(
                _randomize_start_offsets(
                    microbatch=microbatch,
                    vocab_size=vocab_size,
                    key=microbatch_key,
                    sharding=sharding,
                )
            )
        yield batch
