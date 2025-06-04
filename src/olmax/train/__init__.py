from typing import Literal

from ..utils import mib_to_bytes, set_env_var


def prepare_training_environment(
    xla_mem_frac: float = 0.95,
    all_gather_combine_threshold_mib: float = 256,
    reduce_scatter_combine_threshold_mib: float = 128,
    all_reduce_combine_threshold_mib: float = 256,
    disable_jit: bool = False,
    disable_remat: bool = False,
    gpu_architecture: Literal["hopper", "blackwell", "ampere"] | None = None,
):
    assert 0 <= xla_mem_frac <= 1.0

    # See:
    #  - https://github.com/NVIDIA/JAX-Toolbox/blob/main/rosetta/docs/GPU_performance.md
    #  - https://docs.jax.dev/en/latest/gpu_performance_tips.html
    xla_flags = [
        "--xla_gpu_enable_latency_hiding_scheduler=true",
        "--xla_gpu_enable_while_loop_double_buffering=true",
        "--xla_gpu_enable_pipelined_all_gather=true",
        "--xla_gpu_enable_pipelined_reduce_scatter=true",
        "--xla_gpu_enable_pipelined_all_reduce=true",
        "--xla_gpu_enable_all_gather_combine_by_dim=false",
        "--xla_gpu_enable_reduce_scatter_combine_by_dim=false",
        f"--xla_gpu_all_gather_combine_threshold_bytes={mib_to_bytes(all_gather_combine_threshold_mib)}",
        f"--xla_gpu_reduce_scatter_combine_threshold_bytes={mib_to_bytes(reduce_scatter_combine_threshold_mib)}",
        f"--xla_gpu_all_reduce_combine_threshold_bytes={mib_to_bytes(all_reduce_combine_threshold_mib)}",
        #  "--xla_gpu_enable_nccl_user_buffers=true",  # takes up more memory
        #  "--xla_gpu_enable_command_buffer=",
    ]
    if gpu_architecture == "blackwell":
        xla_flags.append("--xla_gpu_enable_command_buffer=FUSION,CUSTOM_CALL")
    set_env_var("XLA_FLAGS", " ".join(xla_flags), override=True)
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
