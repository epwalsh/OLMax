import jax

from ..distributed.parallel import MeshResource
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
        mesh_resource: MeshResource | None = None,
    ):
        super().__init__(mesh_resource)
        self.weight = truncated_normal(
            key,
            (num_embeddings, d_model),
            sharding=None if mesh_resource is None else mesh_resource.get_param_sharding(),
            dtype=dtype,
        )

    @jax.named_scope("olmax.nn.Embedding")
    def __call__(self, x: Array) -> Array:
        return jax.vmap(lambda idx: self.weight[idx])(x)
