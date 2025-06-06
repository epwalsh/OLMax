import logging
import os
from typing import Literal

from ..utils import mib_to_bytes, set_env_var

log = logging.getLogger(__name__)


def prepare_training_environment(
    *,
    xla_mem_frac: float = 0.95,
    xla_flags: Literal["recommended", "system_default"] = "recommended",
    all_gather_combine_threshold_mib: float = 256,
    reduce_scatter_combine_threshold_mib: float = 128,
    all_reduce_combine_threshold_mib: float = 256,
    enable_pipelined_comms: bool = True,
    enable_nccl_user_buffers: bool = False,  # uses more memory
    disable_jit: bool = False,
    disable_remat: bool = False,
    gpu_architecture: Literal["hopper", "blackwell", "ampere"] | None = None,
):
    assert 0 <= xla_mem_frac <= 1.0

    # See:
    # - https://github.com/NVIDIA/JAX-Toolbox/blob/main/rosetta/docs/GPU_performance.md
    # - https://docs.jax.dev/en/latest/gpu_performance_tips.html

    if xla_flags == "recommended":
        xla_flags_ = [
            "--xla_gpu_enable_latency_hiding_scheduler=true",
            f"--xla_gpu_enable_pipelined_all_gather={str(enable_pipelined_comms).lower()}",
            f"--xla_gpu_enable_pipelined_reduce_scatter={str(enable_pipelined_comms).lower()}",
            f"--xla_gpu_enable_pipelined_all_reduce={str(enable_pipelined_comms).lower()}",
            "--xla_gpu_enable_all_gather_combine_by_dim=false",
            "--xla_gpu_enable_reduce_scatter_combine_by_dim=false",
            f"--xla_gpu_all_gather_combine_threshold_bytes={mib_to_bytes(all_gather_combine_threshold_mib)}",
            f"--xla_gpu_reduce_scatter_combine_threshold_bytes={mib_to_bytes(reduce_scatter_combine_threshold_mib)}",
            f"--xla_gpu_all_reduce_combine_threshold_bytes={mib_to_bytes(all_reduce_combine_threshold_mib)}",
            f"--xla_gpu_enable_nccl_user_buffers={str(enable_nccl_user_buffers).lower()}",
            "--xla_gpu_enable_nccl_comm_splitting=true",
            "--xla_gpu_enable_nccl_per_stream_comms=false",
            #  "--xla_gpu_enable_while_loop_double_buffering=true",
            #  "--xla_gpu_enable_command_buffer=",
            #  "--xla_gpu_enable_triton_gemm=false",
        ]
        #  if gpu_architecture == "blackwell":
        #      xla_flags_.append("--xla_gpu_enable_command_buffer=FUSION,CUSTOM_CALL")
        set_env_var("XLA_FLAGS", " ".join(xla_flags_), override=True)
    elif xla_flags != "system_default":
        raise ValueError(
            f"invalid value for 'xla_flags', expected one of 'recommended', 'system_default', but got '{xla_flags}'"
        )

    set_env_var("XLA_PYTHON_CLIENT_MEM_FRACTION", f"{round(xla_mem_frac, 2):.2f}", override=True)

    set_env_var("NCCL_LL128_BUFFSIZE", "-2")
    set_env_var("NCCL_LL_BUFFSIZE", "-2")
    set_env_var("NCCL_PROTO", "SIMPLE,LL,LL128")

    if gpu_architecture != "blackwell":
        set_env_var("CUDA_DEVICE_MAX_CONNECTIONS", "1")

    import jax

    if disable_jit:
        jax.config.update("jax_disable_jit", True)

    if disable_remat:
        jax.config.update("jax_compiler_enable_remat_pass", False)

    all_xla_env_vars = []
    for name, value in os.environ.items():
        if name == "XLA_FLAGS":
            for flag in value.strip().split(" "):
                all_xla_env_vars.append(flag.replace("--", "", 1))
        elif name.startswith("XLA_"):
            all_xla_env_vars.append(f"{name}={value}")
    log.info("XLA environment:\n- " + "\n- ".join(all_xla_env_vars))

    all_cuda_env_vars = []
    for name, value in os.environ.items():
        if name.startswith("CUDA_"):
            all_cuda_env_vars.append(f"{name}={value}")
    log.info("CUDA environment:\n- " + "\n- ".join(all_cuda_env_vars))

    all_nccl_env_vars = []
    for name, value in os.environ.items():
        if name.startswith("NCCL_"):
            all_nccl_env_vars.append(f"{name}={value}")
    log.info("NCCL environment:\n- " + "\n- ".join(all_nccl_env_vars))
