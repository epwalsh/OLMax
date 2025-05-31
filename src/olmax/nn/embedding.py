import jax

from ..distributed.parallel import ParallelConfig
from ..types import Array, DTypeLike, PRNGKeyArray
from .init import truncated_normal
from .module import Module


class Embedding(Module):
    weight: Array

    def __init__(
        self,
        d_model: int,
        num_embeddings: int,
        key: PRNGKeyArray,
        *,
        dtype: DTypeLike = float,
        parallel_config: ParallelConfig | None = None,
    ):
        super().__init__(parallel_config)
        self.weight = truncated_normal(
            key,
            (num_embeddings, d_model),
            sharding=None if parallel_config is None else parallel_config.get_param_sharding(),
            dtype=dtype,
        )

    @jax.named_scope("olmax.nn.Embedding")
    def __call__(self, x: Array) -> Array:
        return jax.vmap(lambda idx: self.weight[idx])(x)
