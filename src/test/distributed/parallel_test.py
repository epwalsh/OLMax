from jax.sharding import AxisType

from olmax.distributed.parallel import (
    DataParallelConfig,
    MeshAxesNames,
    MeshResource,
    TensorParallelConfig,
)


def test_mesh_resource_fsdp():
    config = MeshResource.FSDP()
    assert config.get_min_device_count() == 2
    assert config._get_mesh_axes(4) == (
        (4,),
        (MeshAxesNames.DP.shard,),
        (AxisType.Auto,),
    )


def test_mesh_resource_hsdp():
    config = MeshResource.HSDP(2)
    assert config.get_min_device_count() == 2
    assert config._get_mesh_axes(4) == (
        (2, 2),
        (MeshAxesNames.DP.replicate, MeshAxesNames.DP.shard),
        (AxisType.Auto, AxisType.Auto),
    )


def test_mesh_resource_ddp():
    config = MeshResource.DDP()
    assert config.get_min_device_count() == 1
    assert config._get_mesh_axes(4) == (
        (4,),
        (MeshAxesNames.DP.replicate,),
        (AxisType.Auto,),
    )


def test_mesh_resource_tp():
    config = MeshResource(tp=TensorParallelConfig(2))
    assert config.get_min_device_count() == 2
    assert config._get_mesh_axes(2) == (
        (2,),
        (MeshAxesNames.TP.shard,),
        (AxisType.Auto,),
    )


def test_mesh_resource_tp_with_fsdp():
    mesh_resource = MeshResource(dp=DataParallelConfig.FSDP(), tp=TensorParallelConfig(2))
    assert mesh_resource.get_min_device_count() == 4
    assert mesh_resource._get_mesh_axes(4) == (
        (2, 2),
        (
            MeshAxesNames.DP.shard,
            MeshAxesNames.TP.shard,
        ),
        (AxisType.Auto, AxisType.Auto),
    )
