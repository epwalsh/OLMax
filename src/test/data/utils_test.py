import jax
import pytest

from olmax.data.utils import generate_batches_of_sequential_tokens


@pytest.mark.parametrize("num_microbatches", [1, 2])
def test_generate_batches_of_sequential_tokens(num_microbatches: int):
    key = jax.random.PRNGKey(0)
    for batch in generate_batches_of_sequential_tokens(
        key,
        vocab_size=32,
        sequence_length=4,
        global_batch_size_instances=2 * num_microbatches,
        total_batches=4,
        num_microbatches=num_microbatches,
    ):
        assert len(batch) == num_microbatches
        for mb in batch:
            input_ids, labels = mb
            assert input_ids.shape == (2, 4)
            assert labels.shape == (2, 4)
