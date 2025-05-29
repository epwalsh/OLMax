import functools as ft
from typing import Generator

import jax
import jax.numpy as jnp
from jax import random
from jax.sharding import NamedSharding

from ..distributed.parallel import ParallelConfig
from ..types import Array, PRNGKeyArray


@ft.partial(jax.jit, static_argnames=("sharding",))
def _randomize_start_offsets(
    *,
    batch: Array,
    vocab_size: int,
    key: PRNGKeyArray,
    sharding: NamedSharding | None,
) -> tuple[Array, Array]:
    num_instances, sequence_length = batch.shape
    start_offsets = random.randint(key, (num_instances, 1), 0, vocab_size - sequence_length - 1)
    inputs = batch + start_offsets
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
    num_local_instances: int,
    total_batches: int,
    local_data_parallel_rank: int = 0,
    start_batch: int = 0,
    parallel_config: ParallelConfig | None = None,
) -> Generator[tuple[Array, Array], None, None]:
    key = jax.random.fold_in(key, local_data_parallel_rank)
    batch = jnp.arange(0, sequence_length).reshape(1, -1).repeat(num_local_instances, 0)
    for batch_idx in range(start_batch, total_batches):
        batch_key = jax.random.fold_in(key, batch_idx)
        yield (
            _randomize_start_offsets(
                batch=batch,
                vocab_size=vocab_size,
                key=batch_key,
                sharding=None if parallel_config is None else parallel_config.get_data_sharding(),
            )
        )
