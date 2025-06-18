import functools as ft
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding

from ..jax_utils import vmap_multiple
from ..types import Array
from .module import Module


def _linear_single(x: Array, weight: Array, bias: Array | None = None) -> Array:
    x = weight @ x
    if bias is not None:
        x = x + bias
    return x


def linear(
    x: Array, weight: Array, bias: Array | None = None, psum_axis: str | None = None
) -> Array:
    out = vmap_multiple(lambda xi: _linear_single(xi, weight, bias), x.ndim - 1)(x)
    if psum_axis is not None:
        out = jax.lax.psum(out, psum_axis)
    return out


def layer_norm(
    x: Array, weight: Array | None = None, bias: Array | None = None, eps: float = 1e-5
) -> Array:
    orig_dtype = x.dtype
    with jax.numpy_dtype_promotion("standard"):
        dtype = jnp.result_type(x.dtype, jnp.float32)

    x = x.astype(dtype)
    mean = jnp.mean(x, axis=-1, keepdims=True)
    variance = jnp.var(x, axis=-1, keepdims=True)
    inv = jax.lax.rsqrt(variance + eps)
    out = (x - mean) * inv

    if weight is not None:
        out = weight.astype(dtype) * out
    if bias is not None:
        out = out + bias.astype(dtype)

    return out.astype(orig_dtype)


def rms_norm(
    x: Array, weight: Array | None = None, bias: Array | None = None, eps: float = 1e-5
) -> Array:
    orig_dtype = x.dtype
    with jax.numpy_dtype_promotion("standard"):
        dtype = jnp.result_type(x.dtype, jnp.float32)

    x = x.astype(dtype)
    inv_rms = jax.lax.rsqrt(jnp.mean(x**2, axis=-1, keepdims=True) + eps)
    out = inv_rms * x

    if weight is not None:
        out = weight.astype(dtype) * out
    if bias is not None:
        out = out + bias.astype(dtype)

    return out.astype(orig_dtype)


def cross_entropy_loss(
    logits: Array,
    labels: Array,
    *,
    ignore_index: int = -100,
    reduction: Literal["sum", "mean", "none"] = "mean",
) -> Array:
    n_classes = logits.shape[-1]
    labels_one_hot = jax.nn.one_hot(labels, n_classes)
    where = jnp.expand_dims(labels != ignore_index, -1)

    log_probs = jax.nn.log_softmax(logits, -1, where)
    loss = -(labels_one_hot * log_probs).sum(-1, where=where)

    if reduction == "sum":
        return loss.sum()
    elif reduction == "mean":
        return loss.mean(where=where.squeeze(-1))
    elif reduction == "none":
        return loss  # pyright: ignore
    else:
        raise ValueError(reduction)


def fused_cross_entropy_loss(
    logits: Array,
    labels: Array,
    *,
    ignore_index: int = -100,
    reduction: Literal["sum", "mean", "none"] = "mean",
) -> tuple[Array, Array]:
    """
    Computes CE loss along with the log softmax normalizer (Z loss) term.
    """
    n_classes = logits.shape[-1]
    labels_one_hot = jax.nn.one_hot(labels, n_classes)
    where = jnp.expand_dims(labels != ignore_index, -1)

    log_normalizer = jax.nn.logsumexp(logits, -1, where, keepdims=True)
    log_probs = logits - log_normalizer
    loss = -(labels_one_hot * log_probs).sum(-1, where=where)

    if reduction == "sum":
        loss = loss.sum()
    elif reduction == "mean":
        loss = loss.mean(where=where.squeeze(-1))
    elif reduction == "none":
        pass
    else:
        raise ValueError(reduction)

    return loss, log_normalizer * where


def _apply_module(
    carry: tuple[Module, Array],
    params: Module,
    input_sharding: NamedSharding | None = None,
    output_sharding: NamedSharding | None = None,
) -> tuple[tuple[Module, Array], None]:
    static, x = carry
    if input_sharding is not None:
        x = jax.lax.with_sharding_constraint(x, input_sharding)
    m = eqx.combine(params, static, is_leaf=eqx.is_array)
    y = m(x)
    if output_sharding is not None:
        y = jax.lax.with_sharding_constraint(y, output_sharding)
    return (static, y), None


def scan_module(
    m: Module,
    x: Array,
    input_sharding: NamedSharding | None = None,
    output_sharding: NamedSharding | None = None,
) -> Array:
    params, static = eqx.partition(m, eqx.is_array)
    carry = (static, x)
    carry, _ = jax.lax.scan(
        ft.partial(
            _apply_module,
            input_sharding=input_sharding,
            output_sharding=output_sharding,
        ),
        carry,
        params,
    )
    _, y = carry
    return y
