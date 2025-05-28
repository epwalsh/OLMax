import dataclasses
import functools as ft
from dataclasses import dataclass
from enum import StrEnum

import jax
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from typing_extensions import Self

from . import utils as dist_utils


class TPStyle(StrEnum):
    colwise = "colwise"
    rowwise = "rowwise"


class MeshAxisNames:
    class DP:
        replicate = "dp_replicate"
        shard = "dp_shard"

    class PP:
        shard = "pp_split"

    class CP:
        shard = "cp_split"

    class TP:
        shard = "tp_split"


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
class ParallelConfig:
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

    def get_param_sharding(
        self, dp_sharding_axis: int | None = 0, tp_sharding_axis: int | None = None
    ) -> NamedSharding:
        if tp_sharding_axis is not None and self.tp is None:
            raise ValueError("'tp_sharding_axis' is only valid when tensor parallelism is enabled")

        mesh = self.get_param_mesh()
        partitions: list[str | tuple[str, ...] | None] = []
        if tp_sharding_axis is not None:
            assert MeshAxisNames.TP.shard in mesh.shape
            if dp_sharding_axis is not None and MeshAxisNames.DP.shard in mesh.shape:
                partitions.extend([None] * (max(dp_sharding_axis, tp_sharding_axis) + 1))
                if dp_sharding_axis == tp_sharding_axis:
                    partitions[dp_sharding_axis] = (
                        MeshAxisNames.DP.shard,
                        MeshAxisNames.TP.shard,
                    )
                else:
                    partitions[dp_sharding_axis] = MeshAxisNames.DP.shard
                    partitions[tp_sharding_axis] = MeshAxisNames.TP.shard
            else:
                partitions.extend([None] * tp_sharding_axis)
                partitions.append(MeshAxisNames.TP.shard)
        elif dp_sharding_axis is not None and MeshAxisNames.DP.shard in mesh.shape:
            partitions.extend([None] * dp_sharding_axis)
            partitions.append(MeshAxisNames.DP.shard)

        return NamedSharding(mesh, P(*partitions))

    def get_data_sharding(self, sharding_axis: int = 0) -> NamedSharding:
        mesh = self.get_data_mesh()
        partitions: list[str | tuple[str, ...] | None] = []
        if MeshAxisNames.DP.replicate in mesh.shape and MeshAxisNames.DP.shard in mesh.shape:
            partitions.extend([None] * sharding_axis)
            partitions.append((MeshAxisNames.DP.replicate, MeshAxisNames.DP.shard))
        elif MeshAxisNames.DP.replicate in mesh.shape:
            partitions.extend([None] * sharding_axis)
            partitions.append(MeshAxisNames.DP.replicate)
        elif MeshAxisNames.DP.shard in mesh.shape:
            partitions.extend([None] * sharding_axis)
            partitions.append(MeshAxisNames.DP.shard)
        return NamedSharding(mesh, P(*partitions))

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
    def _get_param_mesh_axes(self, device_count: int) -> tuple[tuple[int, ...], tuple[str, ...]]:
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

        # Data parallel.
        if dp_replicate_degree > 1:
            axis_shapes.append(dp_replicate_degree)
            axis_names.append(MeshAxisNames.DP.replicate)
        if dp_shard_degree > 1:
            axis_shapes.append(dp_shard_degree)
            axis_names.append(MeshAxisNames.DP.shard)

        # Tensor parallel, the inner-most axis.
        if self.tp is not None:
            axis_shapes.append(self.tp.degree)
            axis_names.append(MeshAxisNames.TP.shard)

        return tuple(axis_shapes), tuple(axis_names)

    def get_param_mesh(self) -> Mesh:
        axis_shapes, axis_names = self._get_param_mesh_axes(dist_utils.get_global_device_count())
        return make_mesh(tuple(axis_shapes), tuple(axis_names))

    def get_data_mesh(self) -> Mesh:
        return self.get_param_mesh()


@ft.cache
def make_mesh(axis_shapes: tuple[int, ...], axis_names: tuple[str, ...]) -> Mesh:
    assert len(axis_shapes) == len(axis_names)
    return jax.make_mesh(axis_shapes, axis_names)


def _get_fsdp_mesh(global_device_count: int) -> Mesh:
    return make_mesh((global_device_count,), (MeshAxisNames.DP.shard,))


def get_fsdp_mesh() -> Mesh:
    return _get_fsdp_mesh(dist_utils.get_global_device_count())


def _get_fsdp_sharding(global_device_count: int, sharding_axis: int) -> NamedSharding:
    partitions = [None] * sharding_axis + [MeshAxisNames.DP.shard]
    return NamedSharding(_get_fsdp_mesh(global_device_count), P(*partitions))


def get_fsdp_sharding(sharding_axis: int = 0) -> NamedSharding:
    return _get_fsdp_sharding(dist_utils.get_global_device_count(), sharding_axis)


def _get_hsdp_mesh(global_device_count: int, shard_degree: int) -> Mesh:
    return make_mesh(
        (global_device_count // shard_degree, shard_degree),
        (
            MeshAxisNames.DP.replicate,
            MeshAxisNames.DP.shard,
        ),
    )


def get_hsdp_mesh(shard_degree: int) -> Mesh:
    assert dist_utils.get_global_device_count() % shard_degree == 0
    return _get_hsdp_mesh(dist_utils.get_global_device_count(), shard_degree)


def _get_hsdp_sharding(
    global_device_count: int, shard_degree: int, sharding_axis: int
) -> NamedSharding:
    partitions = [None] * sharding_axis + [MeshAxisNames.DP.shard]
    return NamedSharding(_get_hsdp_mesh(global_device_count, shard_degree), P(*partitions))


def get_hsdp_sharding(shard_degree: int, sharding_axis: int = 0) -> NamedSharding:
    return _get_hsdp_sharding(dist_utils.get_global_device_count(), shard_degree, sharding_axis)


def _get_ddp_mesh(global_device_count: int) -> Mesh:
    return make_mesh((global_device_count,), (MeshAxisNames.DP.replicate,))


def get_ddp_mesh() -> Mesh:
    return _get_ddp_mesh(dist_utils.get_global_device_count())


def _get_ddp_sharding(global_device_count: int) -> NamedSharding:
    return NamedSharding(_get_ddp_mesh(global_device_count), P())


def get_ddp_sharding() -> NamedSharding:
    return _get_ddp_sharding(dist_utils.get_global_device_count())
