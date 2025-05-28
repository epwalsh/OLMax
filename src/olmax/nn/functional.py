import jax
import jax.numpy as jnp

from ..types import Array


def linear(x: Array, weight: Array, bias: Array | None = None) -> Array:
    x = weight @ x
    if bias is not None:
        x = x + bias
    return x


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
