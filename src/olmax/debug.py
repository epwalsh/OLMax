import jax

from .types import Array


def inspect(x: Array, name: str):
    jax.debug.print("{name}: shape={shape}", name=name, shape=x.shape)
    jax.debug.inspect_array_sharding(x, callback=lambda s: print(f"{name}: sharding={s}"))
