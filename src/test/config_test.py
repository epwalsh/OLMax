from olmax.config import CUDAConfig, EnvConfig, parse_config_from_args


def test_parse_config_from_args():
    cfg = parse_config_from_args(
        EnvConfig,
        EnvConfig(cuda=CUDAConfig(device_max_connections=1)),
        args=["--xla.python_client_mem_fraction=0.8"],
    )
    assert cfg.xla.python_client_mem_fraction == 0.8
    assert cfg.cuda.device_max_connections == 1
