import functools as ft
from typing import Literal

import jax
import jax.numpy as jnp

from ..jax_utils import vmap_multiple
from ..types import Array


@jax.jit
def _linear_single(x: Array, weight: Array, bias: Array | None = None) -> Array:
    x = weight @ x
    if bias is not None:
        x = x + bias
    return x


@ft.partial(jax.jit, static_argnums=(3,), static_argnames=("psum_axis",))
def linear(
    x: Array, weight: Array, bias: Array | None = None, psum_axis: str | None = None
) -> Array:
    #  print(weight.shape)
    #  jax.debug.inspect_array_sharding(x, callback=lambda s: print("x:", s))
    #  jax.debug.inspect_array_sharding(weight, callback=lambda s: print("weight:", s))
    #  jax.debug.visualize_array_sharding(weight)
    out = vmap_multiple(lambda xi: _linear_single(xi, weight, bias), x.ndim - 1)(x)
    if psum_axis is not None:
        out = jax.lax.psum(out, psum_axis)
    return out


@jax.jit
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


@jax.jit
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


@ft.partial(jax.jit, static_argnames=("reduction",))
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


@ft.partial(jax.jit, static_argnames=("reduction",))
def cross_entropy_loss_and_log_normalizer(
    logits: Array,
    labels: Array,
    *,
    ignore_index: int = -100,
    reduction: Literal["sum", "mean", "none"] = "mean",
) -> tuple[Array, Array]:
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
