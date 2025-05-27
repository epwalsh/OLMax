import jax
from jax.sharding import NamedSharding

from ..types import Array, PRNGKeyArray
from .init import truncated_normal
from .module import Module


class Linear(Module):
    weight: Array
    bias: Array | None

    def __init__(
        self,
        in_size: int,
        out_size: int,
        key: PRNGKeyArray,
        bias: bool = True,
        weight_sharding: NamedSharding | None = None,
        bias_sharding: NamedSharding | None = None,
    ):
        wkey, bkey = jax.random.split(key)
        self.weight = truncated_normal(
            wkey, (out_size, in_size), sharding=weight_sharding
        )
        self.bias = (
            None
            if not bias
            else truncated_normal(bkey, (out_size,), sharding=bias_sharding)
        )

    @jax.named_scope("olmax.nn.Linear")
    def __call__(self, x: Array) -> Array:
        x = self.weight @ x
        if self.bias is not None:
            x = x + self.bias
        return x
