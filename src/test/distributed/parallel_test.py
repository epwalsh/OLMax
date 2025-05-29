from jax.sharding import AxisType

from olmax.distributed.parallel import (
    DataParallelConfig,
    MeshAxisNames,
    ParallelConfig,
    TensorParallelConfig,
)


def test_parallel_config_fsdp():
    config = ParallelConfig.FSDP()
    assert config.get_min_device_count() == 2
    assert config._get_param_mesh_axes(4) == (
        (4,),
        (MeshAxisNames.DP.shard,),
        (AxisType.Auto,),
    )


def test_parallel_config_hsdp():
    config = ParallelConfig.HSDP(2)
    assert config.get_min_device_count() == 2
    assert config._get_param_mesh_axes(4) == (
        (2, 2),
        (MeshAxisNames.DP.replicate, MeshAxisNames.DP.shard),
        (AxisType.Auto, AxisType.Auto),
    )


def test_parallel_config_ddp():
    config = ParallelConfig.DDP()
    assert config.get_min_device_count() == 1
    assert config._get_param_mesh_axes(4) == (
        (4,),
        (MeshAxisNames.DP.replicate,),
        (AxisType.Auto,),
    )


def test_parallel_config_tp():
    config = ParallelConfig(tp=TensorParallelConfig(2))
    assert config.get_min_device_count() == 2
    assert config._get_param_mesh_axes(2) == (
        (2,),
        (MeshAxisNames.TP.shard,),
        (AxisType.Auto,),
    )


def test_parallel_config_tp_with_fsdp():
    config = ParallelConfig(dp=DataParallelConfig.FSDP(), tp=TensorParallelConfig(2))
    assert config.get_min_device_count() == 4
    assert config._get_param_mesh_axes(4) == (
        (2, 2),
        (
            MeshAxisNames.DP.shard,
            MeshAxisNames.TP.shard,
        ),
        (AxisType.Auto, AxisType.Auto),
    )
