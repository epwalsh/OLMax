import equinox as eqx
import jax
import pytest

import olmax.distributed as dist
import olmax.nn as nn
from olmax.testing.distributed import run_distributed_test
from olmax.testing.utils import allclose
from olmax.types import Array, PRNGKeyArray


def _get_batch(
    key: PRNGKeyArray, batch_size: int, seq_len: int, vocab_size: int
) -> tuple[Array, Array]:
    inputs_key, targets_key = jax.random.split(key)
    return jax.random.randint(inputs_key, (batch_size, seq_len), 0, vocab_size), jax.random.randint(
        targets_key, (batch_size, seq_len), 0, vocab_size
    )


def _get_model(
    key: PRNGKeyArray,
    d_model: int = 16,
    hidden_size: int = 32,
    vocab_size: int = 128,
    num_layers: int = 4,
    mesh_resource: dist.MeshResource | None = None,
) -> nn.Transformer:
    return nn.Transformer(
        key=key,
        d_model=d_model,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        num_layers=num_layers,
        block=nn.TransformerBlock.Config(
            attention=nn.MultiheadSelfAttention.Config(n_heads=4),
            norm=nn.LayerNorm.Config(),
        ),
        norm=nn.LayerNorm.Config(),
        lm_head=nn.LMHead.Config(),
        mesh_resource=mesh_resource,
    )


def _get_loss(model: nn.Module, batch: tuple[Array, Array]) -> Array:
    inputs, targets = batch
    logits = model(inputs)
    return nn.functional.cross_entropy_loss(logits, targets)


@jax.jit
@eqx.filter_value_and_grad
def _get_loss_and_grads(model: nn.Module, batch: tuple[Array, Array]) -> Array:
    return _get_loss(model, batch)


def test_transformer(
    d_model: int = 16,
    hidden_size: int = 32,
    vocab_size: int = 128,
    num_layers: int = 4,
    batch_size: int = 2,
    seq_len: int = 12,
):
    key = jax.random.PRNGKey(0)
    key, data_key = jax.random.split(key, 2)
    model = _get_model(
        key=key,
        d_model=d_model,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        num_layers=num_layers,
    )
    batch = _get_batch(data_key, batch_size, seq_len, vocab_size)
    out = model(batch[0])
    assert out.shape == (batch_size, seq_len, vocab_size)


def _run_transformer_parallel(
    mesh_resource: dist.MeshResource,
    d_model: int = 16,
    hidden_size: int = 32,
    vocab_size: int = 128,
    num_layers: int = 4,
    batch_size: int | None = None,
    seq_len: int = 12,
):
    key = jax.random.PRNGKey(0)
    if batch_size is None:
        batch_size = 2 * dist.get_global_device_count()

    key, batch_key = jax.random.split(key)
    full_batch = _get_batch(batch_key, batch_size, seq_len, vocab_size)
    dist_batch = jax.device_put(full_batch, mesh_resource.get_data_sharding())

    full_model = _get_model(
        key=key,
        d_model=d_model,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        num_layers=num_layers,
    )
    dist_model = _get_model(
        key=key,
        d_model=d_model,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        num_layers=num_layers,
        mesh_resource=mesh_resource,
    )

    assert allclose(full_model.embedding.weight, dist_model.embedding.weight)

    full_loss, full_grads = _get_loss_and_grads(full_model, full_batch)
    dist_loss, dist_grads = _get_loss_and_grads(dist_model, dist_batch)
    assert allclose(full_loss, dist_loss)
    assert allclose(full_grads.parameters(), dist_grads.parameters(), rtol=1e-3, atol=1e-5)


@pytest.mark.parametrize(
    "mesh_resource",
    [
        pytest.param(dist.MeshResource.FSDP(), id="FSDP"),
        pytest.param(dist.MeshResource.DDP(), id="DDP"),
        pytest.param(dist.MeshResource.HSDP(2, 2), id="HSDP"),
    ],
)
def test_transformer_data_parallel(mesh_resource: dist.MeshResource):
    run_distributed_test(
        _run_transformer_parallel,
        num_processes=1,
        devices_per_process=mesh_resource.get_min_device_count(),
        args=(mesh_resource,),
    )


def main(
    mesh_resource: dist.MeshResource,
    d_model: int = 8,
    hidden_size: int = 16,
    vocab_size: int = 32,
    num_layers: int = 2,
    batch_size: int | None = None,
    seq_len: int = 12,
):
    key = jax.random.PRNGKey(0)
    batch_size = 2 * dist.get_global_device_count()

    key, batch_key = jax.random.split(key)
    full_batch = _get_batch(batch_key, batch_size, seq_len, vocab_size)
    dist_batch = jax.device_put(full_batch, mesh_resource.get_data_sharding())

    dist_model = _get_model(
        key=key,
        d_model=d_model,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        num_layers=num_layers,
        mesh_resource=mesh_resource,
    )
    jax.debug.visualize_array_sharding(dist_model.blocks[0].mlp.w1.weight)

    dist_loss, dist_grad = _get_loss_and_grads(dist_model, dist_batch)
    print(dist_loss)
    jax.debug.visualize_array_sharding(dist_grad.blocks[0].mlp.w1.weight)


if __name__ == "__main__":
    jax.config.update("jax_num_cpu_devices", 2)
    jax.config.update("jax_disable_jit", False)

    main(dist.MeshResource.FSDP())
