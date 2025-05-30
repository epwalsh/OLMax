import jax

from .types import Array


def inspect(x: Array, name: str):
    shape = x.shape
    jax.debug.inspect_array_sharding(
        x, callback=lambda sharding: print(f"{name}: {shape=}, {sharding=}")
    )
