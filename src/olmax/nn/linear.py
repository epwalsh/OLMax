from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from ..distributed.parallel import ParallelConfig, TPStyle
from ..jax_utils import vmap_multiple
from ..types import Array, DTypeLike, PRNGKeyArray
from .functional import batched_linear, batched_linear_rowwise_tp, linear
from .init import truncated_normal
from .module import Module


class Linear(Module):
    keepdims: ClassVar[int] = 1

    weight: Array
    bias: Array | None
    tp_style: TPStyle | None = eqx.field(static=True)

    def __init__(
        self,
        in_size: int,
        out_size: int,
        key: PRNGKeyArray,
        bias: bool = True,
        dtype: DTypeLike = float,
        parallel_config: ParallelConfig | None = None,
        tp_style: TPStyle | None = None,
    ):
        super().__init__(parallel_config)

        # Notes on tensor parallelism.
        # ============================
        #
        # nn.Linear(input) = input * weight^T + bias
        #
        # * With colwise we shard weight/bias on dim 0 (output dimension), output is kept sharded
        #   on this dimension, input is assumed to be replicated.
        # * With rowwise we shard weight on dim 1 (input dimension), input is also sharded on this
        #   dimension, and output is replicated.
        self.tp_style = tp_style

        wkey, bkey = jax.random.split(key)
        self.weight = truncated_normal(
            wkey,
            (out_size, in_size),
            sharding=None
            if parallel_config is None
            else parallel_config.get_param_sharding(
                dp_sharding_axis=1 if tp_style == TPStyle.colwise else 0,
                tp_sharding_axis=0
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
                if parallel_config is None
                else parallel_config.get_param_sharding(
                    dp_sharding_axis=0,
                    tp_sharding_axis=0 if tp_style == TPStyle.colwise else None,
                ),
                dtype=dtype,
            )
        )

    @jax.named_scope("olmax.nn.Linear")
    def __call__(self, x):
        if (pc := self.parallel_config) is None:
            return batched_linear(x, self.weight, self.bias)
        elif self.tp_style is None:
            return jax.shard_map(
                batched_linear,
                mesh=pc.get_data_mesh(),
                in_specs=(pc.get_data_partition(), None, None),
                out_specs=pc.get_data_partition(),
            )(
                x, self.weight, self.bias  # pyright: ignore
            )
        elif self.tp_style == TPStyle.colwise:
            return jax.shard_map(
                batched_linear,
                mesh=pc.get_data_mesh(),
                in_specs=(
                    pc.get_data_partition(),
                    pc.get_param_partition(dp_sharding_axis=None, tp_sharding_axis=0),
                    None
                    if self.bias is None
                    else pc.get_param_partition(dp_sharding_axis=None, tp_sharding_axis=0),
                ),
                out_specs=pc.get_data_partition(tp_sharding_axis=-1, ndim=x.ndim),
            )(
                x, self.weight, self.bias  # pyright: ignore
            )
        elif self.tp_style == TPStyle.rowwise:
            out = jax.shard_map(
                batched_linear_rowwise_tp,
                mesh=pc.get_data_mesh(),
                in_specs=(
                    pc.get_data_partition(tp_sharding_axis=-1, ndim=x.ndim),
                    pc.get_param_partition(dp_sharding_axis=None, tp_sharding_axis=1),
                    None,
                ),
                out_specs=pc.get_data_partition(),
            )(
                x, self.weight, self.bias  # pyright: ignore
            )
            return out
        else:
            raise ValueError(self.tp_style)

    @jax.named_scope("olmax.nn.Linear")
    def forward(self, x: Array) -> Array:
        #  assert x.ndim == 1
        weight, bias = self.weight, self.bias

        # Maybe set parameter sharding constraints for tensor parallelism.
        # Refer to notes above about how the sharding is determined.
        if self.tp_style is not None:
            assert self.parallel_config is not None
            if self.tp_style == TPStyle.colwise:
                weight = jax.lax.with_sharding_constraint(
                    weight,
                    self.parallel_config.get_param_sharding(
                        dp_sharding_axis=None, tp_sharding_axis=0
                    ),
                )
                bias = jax.lax.with_sharding_constraint(
                    bias,
                    self.parallel_config.get_param_sharding(
                        dp_sharding_axis=None, tp_sharding_axis=0
                    ),
                )
            elif self.tp_style == TPStyle.rowwise:
                x = jax.lax.with_sharding_constraint(
                    x,
                    self.parallel_config.get_param_sharding(tp_sharding_axis=-1, ndim=x.ndim),
                )
                weight = jax.lax.with_sharding_constraint(
                    weight,
                    self.parallel_config.get_param_sharding(
                        dp_sharding_axis=None, tp_sharding_axis=1
                    ),
                )
            else:
                raise ValueError(f"unexpected tp_style '{self.tp_style}'")

        out = linear(x, weight, bias)

        # Maybe set output sharding constraints for tensor parallelism.
        # Refer to notes above about how the sharding is determined.
        if self.tp_style == TPStyle.colwise:
            assert self.parallel_config is not None
            out = jax.lax.with_sharding_constraint(
                out, self.parallel_config.get_param_sharding(tp_sharding_axis=-1, ndim=out.ndim)
            )

        return out
