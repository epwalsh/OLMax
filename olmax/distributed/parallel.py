import functools as ft

import jax
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from . import utils as dist_utils


class MeshAxisNames:
    class FSDP:
        shard = "fsdp_shard"

    class HSDP:
        replicate = "hsdp_replicate"
        shard = "hsdp_shard"


@ft.cache
def _get_fsdp_mesh(global_device_count: int) -> Mesh:
    return jax.make_mesh((global_device_count,), (MeshAxisNames.FSDP.shard,))


def get_fsdp_mesh() -> Mesh:
    return _get_fsdp_mesh(dist_utils.get_global_device_count())


@ft.cache
def _get_fsdp_sharding(global_device_count: int, sharding_axis: int) -> NamedSharding:
    partitions = [None] * sharding_axis + [MeshAxisNames.FSDP.shard]
    return NamedSharding(_get_fsdp_mesh(global_device_count), P(*partitions))


def get_fsdp_sharding(sharding_axis: int = 0) -> NamedSharding:
    return _get_fsdp_sharding(dist_utils.get_global_device_count(), sharding_axis)


@ft.cache
def _get_hsdp_mesh(global_device_count: int, shard_degree: int) -> Mesh:
    return jax.make_mesh(
        (global_device_count // shard_degree, shard_degree),
        (
            MeshAxisNames.HSDP.replicate,
            MeshAxisNames.HSDP.shard,
        ),
    )


def get_hsdp_mesh(shard_degree: int) -> Mesh:
    assert dist_utils.get_global_device_count() % shard_degree == 0
    return _get_hsdp_mesh(dist_utils.get_global_device_count(), shard_degree)


@ft.cache
def _get_hsdp_sharding(
    global_device_count: int, shard_degree: int, sharding_axis: int
) -> NamedSharding:
    partitions = [None] * sharding_axis + [MeshAxisNames.HSDP.shard]
    return NamedSharding(
        _get_hsdp_mesh(global_device_count, shard_degree), P(*partitions)
    )


def get_hsdp_sharding(shard_degree: int, sharding_axis: int = 0) -> NamedSharding:
    return _get_hsdp_sharding(
        dist_utils.get_global_device_count(), shard_degree, sharding_axis
    )
