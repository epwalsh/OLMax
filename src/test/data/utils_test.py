from olmax.data.utils import generate_batches_of_sequential_tokens


def test_generate_batches_of_sequential_tokens():
    for batch in generate_batches_of_sequential_tokens(
        global_seed=2132,
        local_data_parallel_rank=0,
        vocab_size=32,
        sequence_length=4,
        num_local_instances=2,
        total_batches=4,
    ):
        assert batch.shape == (2, 4)
