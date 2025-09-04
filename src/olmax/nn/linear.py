import equinox as eqx
import jax

from ..distributed.parallel import MeshResource, TPStyle
from ..types import Array, DTypeLike, PRNGKeyArray
from .functional import linear
from .init import truncated_normal
from .module import Module


class Linear(Module):
    weight: Array
    bias: Array | None
    tp_style: TPStyle | None = eqx.field(static=True)

    def __init__(
        self,
        in_size: int,
        out_size: int,
        key: PRNGKeyArray,
        bias: bool = True,
        dtype: DTypeLike = "float32",
        mesh_resource: MeshResource | None = None,
        checkpoint_name: str | None = None,
        tp_style: TPStyle | None = None,
    ):
        super().__init__(mesh_resource, checkpoint_name)

        # Notes on tensor parallelism.
        # ============================
        #
        # nn.Linear(input) = input * weight^T + bias
        #
        # * With colwise we shard weight/bias on dim 0 (output dimension), output is kept sharded
        #   on this dimension, input is assumed to be replicated.
        # * With rowwise we shard weight on dim 1 (input dimension), input is also sharded on its
        #   corresponding dimension (it's last dimension), and output is summed over that dimension
        #   across the TP group and returned replicated.
        self.tp_style = tp_style

        wkey, bkey = jax.random.split(key)
        self.weight = truncated_normal(
            wkey,
            (out_size, in_size),
            sharding=None
            if mesh_resource is None
            else mesh_resource.get_param_sharding_for(
                2,
                fsdp_sharding_dim=1 if tp_style == TPStyle.colwise else 0,
                tp_sharding_dim=0
                if tp_style == TPStyle.colwise
                else (1 if tp_style == TPStyle.rowwise else None),
            ),
            dtype=dtype,
        )
        self.bias = (
            None
            if not bias
            else truncated_normal(
                bkey,
                (out_size,),
                sharding=None
                if mesh_resource is None
                else mesh_resource.get_param_sharding_for(
                    1,
                    fsdp_sharding_dim=0,
                    tp_sharding_dim=0 if tp_style == TPStyle.colwise else None,
                ),
                dtype=dtype,
            )
        )

    @jax.named_scope("olmax.nn.Linear")
    def forward(self, x):
        if (pc := self.mesh_resource) is None or self.tp_style is None:
            out = linear(x, self.weight, self.bias)
            return out
        elif self.tp_style == TPStyle.colwise:
            return pc.shard_map(
                linear,
                (
                    pc.get_data_partition_for(x),
                    pc.get_param_partition_for(
                        self.weight, fsdp_sharding_dim=None, tp_sharding_dim=0
                    ),
                    None
                    if self.bias is None
                    else pc.get_param_partition_for(
                        self.bias, fsdp_sharding_dim=None, tp_sharding_dim=0
                    ),
                ),
                pc.get_data_partition_for(x, tp_sharding_dim=-1),
            )(x, self.weight, self.bias)
        elif self.tp_style == TPStyle.rowwise:
            assert pc.tp_sharding_axis is not None
            out = pc.shard_map(
                linear,
                (
                    pc.get_data_partition_for(x, tp_sharding_dim=-1),
                    pc.get_param_partition_for(
                        self.weight, fsdp_sharding_dim=None, tp_sharding_dim=1
                    ),
                    None,
                    None,
                ),
                pc.get_data_partition_for(x),
            )(x, self.weight, self.bias, pc.tp_sharding_axis)
            return out
        else:
            raise ValueError(self.tp_style)
