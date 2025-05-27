from olmax.distributed.parallel import (
    DataParallelConfig,
    MeshAxisNames,
    ParallelConfig,
    TensorParallelConfig,
)


def test_parallel_config_fsdp():
    config = ParallelConfig.FSDP()
    assert config._get_param_mesh_axes(4) == (
        (4,),
        (MeshAxisNames.FSDP.shard,),
    )


def test_parallel_config_hsdp():
    config = ParallelConfig.HSDP(2)
    assert config._get_param_mesh_axes(4) == (
        (2, 2),
        (MeshAxisNames.HSDP.replicate, MeshAxisNames.HSDP.shard),
    )


def test_parallel_config_ddp():
    config = ParallelConfig.DDP()
    assert config._get_param_mesh_axes(4) == (
        (4,),
        (MeshAxisNames.DDP.replicate,),
    )


def test_parallel_config_tp_with_fsdp():
    config = ParallelConfig(dp=DataParallelConfig.FSDP(), tp=TensorParallelConfig(2))
    assert config._get_param_mesh_axes(4) == (
        (2, 2),
        (
            MeshAxisNames.FSDP.shard,
            MeshAxisNames.TP.split,
        ),
    )
