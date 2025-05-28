import jax

from ..distributed.parallel import ParallelConfig
from ..types import Array, DTypeLike, PRNGKeyArray
from .functional import linear
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
        dtype: DTypeLike = float,
        parallel_config: ParallelConfig | None = None,
    ):
        super().__init__(parallel_config)

        wkey, bkey = jax.random.split(key)
        self.weight = truncated_normal(
            wkey,
            (out_size, in_size),
            sharding=None
            if parallel_config is None
            else parallel_config.get_param_sharding(),
            dtype=dtype,
        )
        self.bias = (
            None
            if not bias
            else truncated_normal(
                bkey,
                (out_size,),
                sharding=None
                if parallel_config is None
                else parallel_config.get_param_sharding(),
                dtype=dtype,
            )
        )

    @jax.named_scope("olmax.nn.Linear")
    def forward(self, x: Array) -> Array:
        return linear(x, self.weight, self.bias)
