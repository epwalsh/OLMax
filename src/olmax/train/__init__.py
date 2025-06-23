import logging
import os
from typing import Literal

from ..env import CUDAConfig, JAXConfig, NCCLConfig, XLAConfig
from ..jax_utils import get_cudnn_version
from ..types import *

log = logging.getLogger(__name__)


def prepare_training_environment(
    *,
    xla_config: Literal["recommended", "system_default"] | XLAConfig = "recommended",
    nccl_config: Literal["recommended", "system_default"] | NCCLConfig = "recommended",
    cuda_config: Literal["recommended", "system_default"] | CUDAConfig = "recommended",
    jax_config: Literal["recommended", "system_default"] | JAXConfig = "recommended",
    gpu_architecture: GPUArchitecture | None = None,
):
    # See:
    # - https://github.com/NVIDIA/JAX-Toolbox/blob/main/rosetta/docs/GPU_performance.md
    # - https://docs.jax.dev/en/latest/gpu_performance_tips.html

    if isinstance(xla_config, XLAConfig):
        xla_config.apply()
    elif xla_config == "recommended":
        XLAConfig.recommended(gpu_architecture).apply()
    elif xla_config != "system_default":
        raise ValueError(xla_config)

    if isinstance(nccl_config, NCCLConfig):
        nccl_config.apply()
    elif nccl_config == "recommended":
        NCCLConfig.recommended(gpu_architecture).apply()
    elif nccl_config != "system_default":
        raise ValueError(nccl_config)

    if isinstance(cuda_config, CUDAConfig):
        cuda_config.apply()
    elif cuda_config == "recommended":
        NCCLConfig.recommended(gpu_architecture).apply()
    elif cuda_config != "system_default":
        raise ValueError(cuda_config)

    if isinstance(jax_config, JAXConfig):
        jax_config.apply()
    elif jax_config == "recommended":
        NCCLConfig.recommended(gpu_architecture).apply()
    elif jax_config != "system_default":
        raise ValueError(jax_config)

    all_xla_env_vars = []
    for name, value in os.environ.items():
        if name == "XLA_FLAGS":
            for flag in value.strip().split(" "):
                all_xla_env_vars.append(flag.replace("--", "", 1))
        elif name.startswith("XLA_"):
            all_xla_env_vars.append(f"{name}={value}")
    log.info("XLA environment:\n❯ " + "\n❯ ".join(all_xla_env_vars))

    all_cuda_env_vars = []
    for name, value in os.environ.items():
        if name.startswith("CUDA_"):
            all_cuda_env_vars.append(f"{name}={value}")
    log.info("CUDA environment:\n❯ " + "\n❯ ".join(all_cuda_env_vars))

    log.info(f"cuDNN version: {get_cudnn_version()}")

    all_nccl_env_vars = []
    for name, value in os.environ.items():
        if name.startswith("NCCL_"):
            all_nccl_env_vars.append(f"{name}={value}")
    log.info("NCCL environment:\n❯ " + "\n❯ ".join(all_nccl_env_vars))
