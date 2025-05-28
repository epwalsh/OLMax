import functools as ft
from typing import Callable

import equinox as eqx
import jax
from jax.sharding import PartitionSpec as P
from typing_extensions import Self

from ..distributed.parallel import ParallelConfig
from ..types import Array, DTypeLike, PRNGKeyArray
from .functional import linear
from .init import truncated_normal
from .module import Module


class Linear(Module):
    weight: Array
    bias: Array | None
    linear_fn: Callable[[Array, Array, Array | None], Array] = eqx.field(
        static=True,
        repr=False,
        default=linear,
    )

    def __init__(
        self,
        in_size: int,
        out_size: int,
        key: PRNGKeyArray,
        bias: bool = True,
        dtype: DTypeLike = float,
        parallel_config: ParallelConfig | None = None,
    ):
        super().__init__()

        wkey, bkey = jax.random.split(key)
        self.weight = truncated_normal(
            wkey,
            (out_size, in_size),
            sharding=None
            if parallel_config is None
            else parallel_config.get_dp_param_sharding(),
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
                else parallel_config.get_dp_param_sharding(),
                dtype=dtype,
            )
        )
        self.linear_fn = linear

        if (
            parallel_config is not None
            and (all_gather_axis := parallel_config.get_dp_param_sharding_axis_name())
            is not None
        ):
            mesh = parallel_config.get_dp_param_mesh()

            @ft.partial(
                jax.remat,  # pyright: ignore
                policy=lambda op, *_, **__: str(op) != "all_gather",  # pyright: ignore
            )
            def linear_fsdp(x: Array, weight: Array, bias: Array | None) -> Array:
                weight = jax.lax.all_gather(weight, all_gather_axis, tiled=True)
                bias = (
                    None
                    if bias is None
                    else jax.lax.all_gather(bias, all_gather_axis, tiled=True)
                )
                return linear(x, weight, bias)

            self.linear_fn = linear_fsdp  # type: ignore

            @ft.partial(
                jax.shard_map,
                mesh=mesh,
                in_specs=P(all_gather_axis),
                out_specs=P(all_gather_axis),
            )
            def batch_linear_fsdp(self_: Self, x: Array) -> Array:
                return jax.vmap(self_.forward)(x)

            self.forward_batch = batch_linear_fsdp

    @jax.named_scope("olmax.nn.Linear")
    def forward(self, x: Array) -> Array:
        return self.linear_fn(x, self.weight, self.bias)
