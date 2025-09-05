import dataclasses
import functools as ft
from dataclasses import dataclass
from enum import StrEnum
from typing import Callable, TypeVar, cast

import jax
import optax
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from typing_extensions import Self

from ..types import Array, PyTree, Specs
from . import utils as dist_utils

F = TypeVar("F", bound=Callable)
T = TypeVar("T", bound=PyTree)


class TPStyle(StrEnum):
    colwise = "colwise"
    rowwise = "rowwise"


@dataclass(frozen=True)
class DataParallelConfig:
    replicate_degree: int = -1
    shard_degree: int = 0

    @classmethod
    def FSDP(cls) -> Self:
        return cls(replicate_degree=0, shard_degree=-1)

    @classmethod
    def HSDP(cls, shard_degree: int, replicate_degree: int = -1) -> Self:
        if shard_degree < 1:
            raise ValueError("expected 'shard_degree' > 1")
        return cls(replicate_degree=replicate_degree, shard_degree=shard_degree)

    @classmethod
    def DDP(cls) -> Self:
        return cls(replicate_degree=-1, shard_degree=0)


@dataclass(frozen=True)
class TensorParallelConfig:
    degree: int


@dataclass(frozen=True)
class ContextParallelConfig:
    degree: int
    use_dp_shard_axis: bool = False
    """
    If True, use the data parallel shard axis for context parallelism. Otherwise use a separate axis
    just for CP.
    """


@dataclass(frozen=True)
class PipelineParallelConfig:
    degree: int


@dataclass(frozen=True)
class ExpertParallelConfig:
    degree: int


@dataclass(frozen=True)
class MeshResource:
    axes: tuple[tuple[int, str, AxisType | None], ...]
    """
    List of ``(axis_size, axis_name, axis_type)`` tuples.
    """

    batch_sharding_axis: str | tuple[str, ...] | None
    """
    The axis name(s) along which data should be sharded.
    """

    fsdp_sharding_axis: str | tuple[str, ...] | None
    """
    The axis name(s) along which parameters should be sharded for FSDP.
    """

    tp_sharding_axis: str | None = None
    """
    The axis name(s) along which parameters should be sharded for TP.
    """

    cp_sharding_axis: str | None = None
    """
    The axis name(s) along which sequences should be sharded for CP.
    """

    @classmethod
    def FSDP(cls, global_device_count: int | None = None) -> Self:
        return cls(
            axes=(
                (
                    global_device_count
                    if global_device_count is not None
                    else dist_utils.get_global_device_count(),
                    "fsdp",
                    AxisType.Auto,
                ),
            ),
            batch_sharding_axis="fsdp",
            fsdp_sharding_axis="fsdp",
            tp_sharding_axis=None,
            cp_sharding_axis=None,
        )

    @classmethod
    def HSDP(
        cls,
        shard_degree: int | None = None,
        global_device_count: int | None = None,
        local_device_count: int | None = None,
    ) -> Self:
        if shard_degree is None or shard_degree < 0:
            shard_degree = (
                local_device_count
                if local_device_count is not None
                else dist_utils.get_local_device_count()
            )
        else:
            assert shard_degree > 0
        replicate_degree = (
            global_device_count
            if global_device_count is not None
            else dist_utils.get_global_device_count()
        ) // shard_degree
        return cls(
            axes=(
                (replicate_degree, "fsdp_replicate", AxisType.Auto),
                (shard_degree, "fsdp_shard", AxisType.Auto),
            ),
            batch_sharding_axis=("fsdp_replicate", "fsdp_shard"),
            fsdp_sharding_axis="fsdp_shard",
            tp_sharding_axis=None,
            cp_sharding_axis=None,
        )

    @classmethod
    def FSDP_with_CP(
        cls,
        global_device_count: int | None = None,
    ) -> Self:
        return cls(
            axes=(
                (
                    global_device_count
                    if global_device_count is not None
                    else dist_utils.get_global_device_count(),
                    "context",
                    AxisType.Auto,
                ),
            ),
            batch_sharding_axis=None,
            fsdp_sharding_axis="context",
            tp_sharding_axis=None,
            cp_sharding_axis="context",
        )

    @classmethod
    def HSDP_with_CP(
        cls,
        shard_degree: int | None = None,
        global_device_count: int | None = None,
        local_device_count: int | None = None,
    ) -> Self:
        return dataclasses.replace(
            cls.HSDP(
                shard_degree=shard_degree,
                global_device_count=global_device_count,
                local_device_count=local_device_count,
            ),
            batch_sharding_axis="fsdp_replicate",
            cp_sharding_axis="fsdp_shard",
        )

    @classmethod
    def DDP(cls, global_device_count: int | None = None) -> Self:
        return cls(
            axes=(
                (
                    global_device_count
                    if global_device_count is not None
                    else dist_utils.get_global_device_count(),
                    "data",
                    AxisType.Auto,
                ),
            ),
            batch_sharding_axis="data",
            fsdp_sharding_axis=None,
            tp_sharding_axis=None,
            cp_sharding_axis=None,
        )

    @ft.cached_property
    def axis_shapes(self) -> tuple[int, ...]:
        return tuple(axis_size for axis_size, *_ in self.axes)

    @ft.cached_property
    def axis_names(self) -> tuple[str, ...]:
        return tuple(axis_name for _, axis_name, _ in self.axes)

    @ft.cached_property
    def axis_types(self) -> tuple[AxisType, ...]:
        return tuple(
            axis_type if axis_type is not None else AxisType.Auto for _, _, axis_type in self.axes
        )

    @ft.cached_property
    def data_parallel_size(self) -> int:
        size = self.size
        for axis in (self.cp_sharding_axis, self.tp_sharding_axis):
            if axis is not None:
                size //= self.axis_size(axis)
        return size

    @ft.cached_property
    def size(self) -> int:
        """
        The size of the mesh, i.e. the total number of devices.
        """
        size = 1
        for axis_size in self.axis_shapes:
            size *= axis_size
        return size

    def axis_size(self, axis: int | str) -> int:
        if isinstance(axis, int):
            return self.axis_shapes[axis]
        else:
            axis_index = self.axis_names.index(axis)
            return self.axis_shapes[axis_index]

    def has_axis(self, axis: str) -> bool:
        return axis in self.axis_names

    def get_partition_for(
        self, x: Array | tuple[int, ...] | int, axes: dict[int, str | tuple[str, ...] | None]
    ) -> P:
        mesh = self.get_mesh()
        ndim: int
        if isinstance(x, Array):
            ndim = x.ndim
        elif isinstance(x, int):
            ndim = x
        elif isinstance(x, tuple):
            ndim = len(x)
        else:
            raise TypeError(f"unexpected type for 'x': {type(x)}")
        partitions: list[str | tuple[str, ...] | None] = [None] * ndim
        for dim, axis_spec in axes.items():
            if dim < 0:
                dim = ndim + dim
            assert 0 <= dim < ndim
            assert partitions[dim] is None  # no duplicates
            if isinstance(axis_spec, tuple):
                axis_spec = tuple([a for a in axis_spec if a in mesh.shape]) or None
            elif isinstance(axis_spec, str) and axis_spec not in mesh.shape:
                axis_spec = None
            partitions[dim] = axis_spec
        return P(*partitions)

    def get_sharding_for(
        self, x: Array | tuple[int, ...] | int, axes: dict[int, str | tuple[str, ...] | None]
    ) -> NamedSharding:
        mesh = self.get_mesh()
        return NamedSharding(mesh, self.get_partition_for(x, axes))

    def get_replicated_partition(self) -> P:
        return P()

    def get_replicated_sharding(self) -> NamedSharding:
        return NamedSharding(self.get_mesh(), P())

    def get_param_partition_for(
        self,
        x: Array | tuple[int, ...] | int,
        fsdp_sharding_dim: int | None = 0,
        tp_sharding_dim: int | None = None,
    ) -> P:
        axes: dict[int, str | tuple[str, ...] | None] = {}
        if fsdp_sharding_dim is not None and self.fsdp_sharding_axis is not None:
            axes[fsdp_sharding_dim] = self.fsdp_sharding_axis
        if tp_sharding_dim is not None and self.tp_sharding_axis is not None:
            axes[tp_sharding_dim] = self.tp_sharding_axis
        return self.get_partition_for(x, axes)

    def get_param_sharding_for(
        self,
        x: Array | tuple[int, ...] | int,
        fsdp_sharding_dim: int | None = 0,
        tp_sharding_dim: int | None = None,
    ) -> NamedSharding:
        mesh = self.get_mesh()
        return NamedSharding(
            mesh,
            self.get_param_partition_for(
                x, fsdp_sharding_dim=fsdp_sharding_dim, tp_sharding_dim=tp_sharding_dim
            ),
        )

    def get_data_partition_for(
        self,
        x: Array | tuple[int, ...],
        batch_dim: int | None = 0,
        sequence_dim: int | None = None,
        tp_sharding_dim: int | None = None,
    ) -> P:
        axes: dict[int, str | tuple[str, ...] | None] = {}
        if batch_dim is not None and self.batch_sharding_axis is not None:
            axes[batch_dim] = self.batch_sharding_axis
        if sequence_dim is not None and self.cp_sharding_axis is not None:
            axes[sequence_dim] = self.cp_sharding_axis
        if tp_sharding_dim is not None and self.tp_sharding_axis is not None:
            axes[tp_sharding_dim] = self.tp_sharding_axis
        return self.get_partition_for(x, axes)

    def get_data_sharding_for(
        self,
        x: Array | tuple[int, ...],
        batch_dim: int | None = 0,
        sequence_dim: int | None = None,
        tp_sharding_dim: int | None = None,
    ) -> NamedSharding:
        mesh = self.get_mesh()
        return NamedSharding(
            mesh,
            self.get_data_partition_for(
                x, batch_dim=batch_dim, sequence_dim=sequence_dim, tp_sharding_dim=tp_sharding_dim
            ),
        )

    def with_data_sharding_constraint(
        self,
        x: Array,
        batch_dim: int | None = 0,
        sequence_dim: int | None = None,
        tp_sharding_dim: int | None = None,
    ) -> Array:
        return jax.lax.with_sharding_constraint(
            x,
            self.get_data_sharding_for(
                x, batch_dim=batch_dim, sequence_dim=sequence_dim, tp_sharding_dim=tp_sharding_dim
            ),
        )

    def reorder_cp_array_for_causal_load_balancing(self, x: Array, sequence_dim: int) -> Array:
        from olmax.te_utils import assert_te

        if self.cp_sharding_axis is None:
            raise RuntimeError(
                "'reorder_cp_array_for_causal_load_balancing' can only be used with a CP dimension in the mesh"
            )
        te = assert_te("context parallelism")
        return te.jax.attention.reorder_causal_load_balancing(
            x,
            te.jax.attention.ReorderStrategy.DualChunkSwap,
            self.axis_size(self.cp_sharding_axis),
            sequence_dim,
        )

    def get_opt_state_sharding(self, opt_state: optax.OptState) -> optax.OptState:
        return jax.tree.map(
            lambda a: a.sharding if a.ndim > 0 else self.get_replicated_sharding(), opt_state
        )

    def get_mesh(self) -> Mesh:
        return make_mesh(self.axis_shapes, self.axis_names, self.axis_types)

    def shard_map(self, fun: F, in_specs: Specs, out_specs: Specs | None) -> F:
        return cast(
            F, jax.shard_map(fun, mesh=self.get_mesh(), in_specs=in_specs, out_specs=out_specs)
        )

    def all_gather(self, tree: T, axis: str) -> T:
        def _all_gather(x: Array) -> Array:
            if x.ndim > 0:
                return jax.lax.all_gather(x, axis_name=axis, tiled=True)
            else:
                return x

        return jax.tree.map(_all_gather, tree)

    def reduce_scatter(self, tree: T, axis: str) -> T:
        axis_size = self.axis_size(axis)
        divide_factor = dist_utils.get_reduce_divide_factor(axis_size)

        def _reduce_scatter(x: Array) -> Array:
            x = jax.lax.psum_scatter(x / divide_factor, axis_name=axis, tiled=True)
            return x * divide_factor / axis_size

        return jax.tree.map(_reduce_scatter, tree)

    def all_reduce(self, tree: T, axis: str) -> T:
        axis_size = self.axis_size(axis)
        divide_factor = dist_utils.get_reduce_divide_factor(axis_size)

        def _all_reduce(x: Array) -> Array:
            x = jax.lax.psum(x / divide_factor, axis_name=axis)
            return x * divide_factor / axis_size

        return jax.tree.map(_all_reduce, tree)

    def get_mesh_axes_repr(self) -> str:
        axes_repr = []
        for axis_size, axis_name, _ in self.axes:
            axes_repr.append(f"{axis_name} x {axis_size}")
        return f"({', '.join(axes_repr)},)"

    def set_mesh(self):
        jax.sharding.set_mesh(self.get_mesh())


@ft.cache
def make_mesh(
    axis_shapes: tuple[int, ...],
    axis_names: tuple[str, ...],
    axis_types: tuple[AxisType, ...] | None = None,
) -> Mesh:
    assert len(axis_shapes) == len(axis_names)
    return jax.make_mesh(axis_shapes, axis_names, axis_types=axis_types)
