from jax.sharding import AxisType

from olmax.distributed.parallel import (
    DataParallelConfig,
    MeshAxesNames,
    ParallelConfig,
    TensorParallelConfig,
)


def test_parallel_config_fsdp():
    config = ParallelConfig.FSDP()
    assert config.get_min_device_count() == 2
    assert config._get_mesh_axes(4) == (
        (4,),
        (MeshAxesNames.DP.shard,),
        (AxisType.Auto,),
    )


def test_parallel_config_hsdp():
    config = ParallelConfig.HSDP(2)
    assert config.get_min_device_count() == 2
    assert config._get_mesh_axes(4) == (
        (2, 2),
        (MeshAxesNames.DP.replicate, MeshAxesNames.DP.shard),
        (AxisType.Auto, AxisType.Auto),
    )


def test_parallel_config_ddp():
    config = ParallelConfig.DDP()
    assert config.get_min_device_count() == 1
    assert config._get_mesh_axes(4) == (
        (4,),
        (MeshAxesNames.DP.replicate,),
        (AxisType.Auto,),
    )


def test_parallel_config_tp():
    config = ParallelConfig(tp=TensorParallelConfig(2))
    assert config.get_min_device_count() == 2
    assert config._get_mesh_axes(2) == (
        (2,),
        (MeshAxesNames.TP.shard,),
        (AxisType.Auto,),
    )


def test_parallel_config_tp_with_fsdp():
    config = ParallelConfig(dp=DataParallelConfig.FSDP(), tp=TensorParallelConfig(2))
    assert config.get_min_device_count() == 4
    assert config._get_mesh_axes(4) == (
        (2, 2),
        (
            MeshAxesNames.DP.shard,
            MeshAxesNames.TP.shard,
        ),
        (AxisType.Auto, AxisType.Auto),
    )
