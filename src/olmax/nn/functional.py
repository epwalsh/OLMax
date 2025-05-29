import functools as ft
from typing import Literal

import jax
import jax.numpy as jnp

from ..types import Array


@jax.jit
def linear(x: Array, weight: Array, bias: Array | None = None) -> Array:
    x = weight @ x
    if bias is not None:
        x = x + bias
    #  jax.debug.inspect_array_sharding(x, callback=lambda s: print("linear out:", s))
    return x


@jax.jit
def layer_norm(
    x: Array, weight: Array | None = None, bias: Array | None = None, eps: float = 1e-5
) -> Array:
    orig_dtype = x.dtype
    with jax.numpy_dtype_promotion("standard"):
        dtype = jnp.result_type(x.dtype, jnp.float32)

    x = x.astype(dtype)
    mean = jnp.mean(x, keepdims=True)
    variance = jnp.var(x, keepdims=True)
    variance = jnp.maximum(0.0, variance)
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
    inv_rms = jax.lax.rsqrt(jnp.mean(x**2) + eps)
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
    loss = (labels_one_hot * log_probs).sum(-1, where=where)

    if reduction == "sum":
        return loss.sum()
    elif reduction == "mean":
        return loss.mean()
    elif reduction == "none":
        return loss  # pyright: ignore
    else:
        raise ValueError(reduction)
