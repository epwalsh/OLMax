import dataclasses
import functools as ft
from dataclasses import dataclass
from enum import StrEnum
from typing import Callable, ClassVar, Type, TypeVar, cast

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


class MeshAxesNames:
    class DP:
        replicate = "dp_replicate"
        shard = "dp_shard"

    class PP:
        shard = "pp_shard"

    class CP:
        shard = "cp_shard"

    class TP:
        shard = "tp_shard"


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


@dataclass(frozen=True)
class PipelineParallelConfig:
    degree: int


@dataclass(frozen=True)
class ExpertParallelConfig:
    degree: int


@dataclass(frozen=True)
class MeshResource:
    MeshAxesNames: ClassVar[Type[MeshAxesNames]] = MeshAxesNames

    dp: DataParallelConfig = dataclasses.field(default_factory=DataParallelConfig)
    tp: TensorParallelConfig | None = None
    #  cp: ContextParallelConfig | None = None
    #  pp: PipelineParallelConfig | None = None
    #  ep: ExpertParallelConfig | None = None

    @classmethod
    def FSDP(cls) -> Self:
        return cls(dp=DataParallelConfig.FSDP())

    @classmethod
    def HSDP(cls, shard_degree: int, replicate_degree: int = -1) -> Self:
        return cls(dp=DataParallelConfig.HSDP(shard_degree, replicate_degree=replicate_degree))

    @classmethod
    def DDP(cls) -> Self:
        return cls(dp=DataParallelConfig.DDP())

    def get_min_device_count(self) -> int:
        devices = 1
        if (d := self.dp.replicate_degree) > 0:
            devices *= d
        if (d := self.dp.shard_degree) != 0:
            devices *= d if d > 0 else 2
        if self.tp is not None and (d := self.tp.degree) != 0:
            devices *= d if d > 0 else 2
        return devices

    def get_mesh(self) -> Mesh:
        axis_shapes, axis_names, axis_types = self._get_mesh_axes(
            dist_utils.get_global_device_count()
        )
        return make_mesh(axis_shapes, axis_names, axis_types)

    def get_partition_for(
        self, x: Array | tuple[int, ...], axes: dict[int, str | tuple[str, ...] | None]
    ) -> P:
        mesh = self.get_mesh()
        ndim = x.ndim if isinstance(x, Array) else len(x)
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
        self, x: Array | tuple[int, ...], axes: dict[int, str | tuple[str, ...] | None]
    ) -> NamedSharding:
        mesh = self.get_mesh()
        return NamedSharding(mesh, self.get_partition_for(x, axes))

    def get_param_partition_for(
        self,
        x: Array | tuple[int, ...],
        dp_sharding_axis: int | None = 0,
        tp_sharding_axis: int | None = None,
    ) -> P:
        axes: dict[int, str | tuple[str, ...] | None] = {}
        if dp_sharding_axis is not None:
            axes[dp_sharding_axis] = MeshAxesNames.DP.shard
        if tp_sharding_axis is not None:
            axes[tp_sharding_axis] = MeshAxesNames.TP.shard
        return self.get_partition_for(x, axes)

    def get_data_partition_for(
        self,
        x: Array | tuple[int, ...],
        dp_sharding_axis: int = 0,
        tp_sharding_axis: int | None = None,
    ) -> P:
        axes: dict[int, str | tuple[str, ...] | None] = {}
        if dp_sharding_axis is not None:
            axes[dp_sharding_axis] = (MeshAxesNames.DP.replicate, MeshAxesNames.DP.shard)
        if tp_sharding_axis is not None:
            axes[tp_sharding_axis] = MeshAxesNames.TP.shard
        return self.get_partition_for(x, axes)

    def get_replicated_partition(self) -> P:
        return P()

    def get_replicated_sharding(self) -> NamedSharding:
        return NamedSharding(self.get_mesh(), P())

    def get_opt_state_sharding(self, opt_state: optax.OptState) -> optax.OptState:
        return jax.tree.map(
            lambda a: a.sharding if a.ndim > 0 else self.get_replicated_sharding(), opt_state
        )

    def shard_map(self, fun: F, in_specs: Specs, out_specs: Specs | None) -> F:
        return cast(
            F, jax.shard_map(fun, mesh=self.get_mesh(), in_specs=in_specs, out_specs=out_specs)
        )

    def all_gather(self, tree: T, axis: str) -> T:
        return jax.tree.map(ft.partial(jax.lax.all_gather, axis_name=axis, tiled=True), tree)

    def reduce_scatter(self, tree: T, axis: str) -> T:
        axis_size = self.axis_size(axis)
        divide_factor = dist_utils.get_reduce_divide_factor(axis_size)

        def reduce_scatter(x: Array) -> Array:
            x = jax.lax.psum_scatter(x / divide_factor, axis_name=axis, tiled=True)
            return x * divide_factor / axis_size

        return jax.tree.map(reduce_scatter, tree)

    def all_reduce(self, tree: T, axis: str) -> T:
        axis_size = self.axis_size(axis)
        divide_factor = dist_utils.get_reduce_divide_factor(axis_size)

        def all_reduce(x: Array) -> Array:
            x = jax.lax.psum(x / divide_factor, axis_name=axis)
            return x * divide_factor / axis_size

        return jax.tree.map(all_reduce, tree)

    def has_axis(self, axis: str) -> bool:
        return axis in self.get_mesh().shape

    def axis_size(self, axis: str) -> int:
        return self.get_mesh().shape[axis]

    def get_param_partition(
        self,
        dp_sharding_axis: int | None = 0,
        tp_sharding_axis: int | None = None,
        ndim: int | None = None,
    ) -> P:
        if tp_sharding_axis is not None and self.tp is None:
            raise ValueError("'tp_sharding_axis' is only valid when tensor parallelism is enabled")

        if dp_sharding_axis is not None and dp_sharding_axis < 0:
            if ndim is None:
                raise ValueError("using negative offset axes requires specifying ndim")
            dp_sharding_axis = ndim + dp_sharding_axis

        if tp_sharding_axis is not None and tp_sharding_axis < 0:
            if ndim is None:
                raise ValueError("using negative offset axes requires specifying ndim")
            tp_sharding_axis = ndim + tp_sharding_axis

        mesh = self.get_mesh()
        partitions: list[str | tuple[str, ...] | None] = []
        if tp_sharding_axis is not None:
            assert MeshAxesNames.TP.shard in mesh.shape
            if dp_sharding_axis is not None and MeshAxesNames.DP.shard in mesh.shape:
                partitions.extend([None] * (max(dp_sharding_axis, tp_sharding_axis) + 1))
                if dp_sharding_axis == tp_sharding_axis:
                    partitions[dp_sharding_axis] = (
                        MeshAxesNames.DP.shard,
                        MeshAxesNames.TP.shard,
                    )
                else:
                    partitions[dp_sharding_axis] = MeshAxesNames.DP.shard
                    partitions[tp_sharding_axis] = MeshAxesNames.TP.shard
            else:
                partitions.extend([None] * tp_sharding_axis)
                partitions.append(MeshAxesNames.TP.shard)
        elif dp_sharding_axis is not None and MeshAxesNames.DP.shard in mesh.shape:
            partitions.extend([None] * dp_sharding_axis)
            partitions.append(MeshAxesNames.DP.shard)

        return P(*partitions)

    def get_param_sharding(
        self,
        dp_sharding_axis: int | None = 0,
        tp_sharding_axis: int | None = None,
        ndim: int | None = None,
    ) -> NamedSharding:
        mesh = self.get_mesh()
        return NamedSharding(
            mesh,
            self.get_param_partition(
                dp_sharding_axis=dp_sharding_axis, tp_sharding_axis=tp_sharding_axis, ndim=ndim
            ),
        )

    def get_param_replication(self):
        mesh = self.get_mesh()
        return NamedSharding(mesh, P())

    def get_data_partition(
        self,
        dp_sharding_axis: int = 0,
        tp_sharding_axis: int | None = None,
        ndim: int | None = None,
    ) -> P:
        if tp_sharding_axis is not None and self.tp is None:
            raise ValueError("'tp_sharding_axis' is only valid when tensor parallelism is enabled")

        if dp_sharding_axis is not None and dp_sharding_axis < 0:
            if ndim is None:
                raise ValueError("using negative offset axes requires specifying ndim")
            dp_sharding_axis = ndim + dp_sharding_axis

        if tp_sharding_axis is not None and tp_sharding_axis < 0:
            if ndim is None:
                raise ValueError("using negative offset axes requires specifying ndim")
            tp_sharding_axis = ndim + tp_sharding_axis

        mesh = self.get_mesh()
        partitions: list[str | tuple[str, ...] | None] = []
        if MeshAxesNames.DP.replicate in mesh.shape and MeshAxesNames.DP.shard in mesh.shape:
            partitions.extend([None] * dp_sharding_axis)
            partitions.append((MeshAxesNames.DP.replicate, MeshAxesNames.DP.shard))
        elif MeshAxesNames.DP.replicate in mesh.shape:
            partitions.extend([None] * dp_sharding_axis)
            partitions.append(MeshAxesNames.DP.replicate)
        elif MeshAxesNames.DP.shard in mesh.shape:
            partitions.extend([None] * dp_sharding_axis)
            partitions.append(MeshAxesNames.DP.shard)

        if tp_sharding_axis is not None:
            assert tp_sharding_axis > dp_sharding_axis
            assert MeshAxesNames.TP.shard in mesh.shape
            partitions.extend([None] * (1 + tp_sharding_axis - len(partitions)))
            partitions[tp_sharding_axis] = MeshAxesNames.TP.shard

        return P(*partitions)

    def get_data_sharding(self, sharding_axis: int = 0) -> NamedSharding:
        mesh = self.get_mesh()
        return NamedSharding(mesh, self.get_data_partition(dp_sharding_axis=sharding_axis))

    def _validate_dp_degrees(self, dp_device_ws: int) -> tuple[int, int]:
        dp_replicate_degree = self.dp.replicate_degree or 1
        dp_shard_degree = self.dp.shard_degree or 1

        # Make DP shard/replicate degrees concrete.
        if dp_shard_degree < 0 and dp_replicate_degree < 0:
            raise ValueError(
                f"Only one of '{self.__class__.__name__}.dp.shard_degree' or "
                f"'{self.__class__.__name__}.dp.replicate_degree' can be set to '-1'"
            )

        if dp_shard_degree > 0 and dp_device_ws % dp_shard_degree != 0:
            raise ValueError(
                f"'{self.__class__.__name__}.dp.shard_degree' ({dp_shard_degree}) must "
                f"divide into the data parallel device world size ({dp_device_ws})"
            )

        if dp_replicate_degree > 0 and dp_device_ws % dp_replicate_degree != 0:
            raise ValueError(
                f"'{self.__class__.__name__}.dp.replicate_degree' ({dp_replicate_degree}) must "
                f"divide into the data parallel device world size ({dp_device_ws})"
            )

        if dp_replicate_degree < 0:
            assert dp_shard_degree > 0
            dp_replicate_degree = dp_device_ws // dp_shard_degree

        if dp_shard_degree < 0:
            assert dp_replicate_degree > 0
            dp_shard_degree = dp_device_ws // dp_replicate_degree

        return dp_replicate_degree, dp_shard_degree

    @ft.cache
    def _get_mesh_axes(
        self, device_count: int
    ) -> tuple[tuple[int, ...], tuple[str, ...], tuple[AxisType, ...]]:
        dp_device_ws = device_count
        dp_shard_degree = self.dp.shard_degree or 1
        dp_replicate_degree = self.dp.replicate_degree or 1

        # Validate TP degree and adjust DP device world size accordingly.
        if self.tp is not None:
            if self.tp.degree < 1 or dp_device_ws % self.tp.degree != 0:
                raise ValueError(
                    f"'{self.__class__.__name__}.tp.degree' must be at least 1 and divide into the data parallel device world size"
                )
            dp_device_ws //= self.tp.degree

        # Make DP shard/replicate degrees concrete.
        dp_replicate_degree, dp_shard_degree = self._validate_dp_degrees(dp_device_ws)
        assert dp_shard_degree > 0
        assert dp_replicate_degree > 0

        # Build up mesh axes.
        axis_shapes: list[int] = []
        axis_names: list[str] = []
        axis_types: list[AxisType] = []

        # Data parallel.
        if dp_replicate_degree > 1:
            axis_shapes.append(dp_replicate_degree)
            axis_names.append(MeshAxesNames.DP.replicate)
            axis_types.append(AxisType.Auto)
        if dp_shard_degree > 1:
            axis_shapes.append(dp_shard_degree)
            axis_names.append(MeshAxesNames.DP.shard)
            axis_types.append(AxisType.Auto)

        # Tensor parallel, the inner-most axis.
        if self.tp is not None:
            axis_shapes.append(self.tp.degree)
            axis_names.append(MeshAxesNames.TP.shard)
            axis_types.append(AxisType.Auto)

        return tuple(axis_shapes), tuple(axis_names), tuple(axis_types)

    def get_mesh_axes_repr(self) -> str:
        axes = self._get_mesh_axes(dist_utils.get_global_device_count())
        axes_repr = []
        for axis_size, axis_name, _ in zip(*axes):
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
