import jax

from olmax.data.utils import generate_batches_of_sequential_tokens


def test_generate_batches_of_sequential_tokens():
    key = jax.random.PRNGKey(0)
    for batch in generate_batches_of_sequential_tokens(
        key,
        vocab_size=32,
        sequence_length=4,
        global_batch_size_instances=2,
        total_batches=4,
    ):
        assert batch[0].shape == (2, 4)
        assert batch[1].shape == (2, 4)
