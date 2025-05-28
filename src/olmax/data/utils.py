from typing import Generator

import jax
import jax.numpy as jnp
from jax import random
from jaxtyping import Array, PRNGKeyArray


@jax.jit
def _randomize_start_offsets(
    *,
    batch: Array,
    vocab_size: int,
    key: PRNGKeyArray,
) -> Array:
    num_instances, sequence_length = batch.shape
    start_offsets = random.randint(key, (num_instances, 1), 0, vocab_size - sequence_length)
    return batch + start_offsets


def generate_batches_of_sequential_tokens(
    *,
    global_seed: int,
    local_data_parallel_rank: int,
    vocab_size: int,
    sequence_length: int,
    num_local_instances: int,
    total_batches: int,
) -> Generator[Array, None, None]:
    key = random.key(global_seed + local_data_parallel_rank)
    batch = jnp.arange(0, sequence_length).reshape(1, -1).repeat(num_local_instances, 0)
    for _ in range(total_batches):
        key, subkey = random.split(key)
        yield (
            _randomize_start_offsets(
                batch=batch,
                vocab_size=vocab_size,
                key=subkey,
            )
        )
