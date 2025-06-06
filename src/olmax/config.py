from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import Type, TypeVar

import draccus
import jax

from .types import *
from .utils import bytes_to_mib, mib_to_bytes, set_env_var

RegistrableConfig = draccus.ChoiceRegistry


C = TypeVar("C", bound="_EnvConfig")


@dataclass
class _EnvConfig:
    @classmethod
    @abstractmethod
    def recommend(cls: Type[C], gpu_architecture: GPUArchitecture | None, **overrides) -> C:
        raise NotImplementedError

    @abstractmethod
    def apply(self):
        """
        Apply the settings.
        """
        raise NotImplementedError


@dataclass
class XLAConfig(_EnvConfig):
    """
    XLA environment configuration.
    """

    # See here for good defaults:
    # - https://github.com/NVIDIA/JAX-Toolbox/blob/main/rosetta/docs/GPU_performance.md
    # - https://docs.jax.dev/en/latest/gpu_performance_tips.html

    python_client_mem_fraction: float = 0.95

    gpu_enable_latency_hiding_scheduler: bool = True

    gpu_enable_pipelined_all_gather: bool = True
    gpu_enable_pipelined_reduce_scatter: bool = True
    gpu_enable_pipelined_all_reduce: bool = True

    gpu_enable_all_gather_combine_by_dim: bool = False
    gpu_enable_reduce_scatter_combine_by_dim: bool = False

    gpu_all_gather_combine_threshold_bytes: int | None = None
    gpu_reduce_scatter_combine_threshold_bytes: int | None = None
    gpu_all_reduce_combine_threshold_bytes: int | None = None

    gpu_enable_nccl_user_buffers: bool = False
    gpu_enable_nccl_comm_splitting: bool = True
    gpu_enable_nccl_per_stream_comms: bool = False

    gpu_enable_while_loop_double_buffering: bool = True
    gpu_enable_command_buffer: str | None = None

    gpu_enable_triton_gemm: bool = False

    @classmethod
    def recommended(cls, gpu_architecture: GPUArchitecture | None, **overrides) -> XLAConfig:
        if gpu_architecture == GPUArchitecture.blackwell:
            return cls(
                #  gpu_enable_command_buffer="FUSION,CUSTOM_CALL",
                **overrides
            )
        return cls(**overrides)

    @property
    def gpu_all_gather_combine_threshold_mib(self) -> float | None:
        if self.gpu_all_gather_combine_threshold_bytes is None:
            return None
        else:
            return bytes_to_mib(self.gpu_all_gather_combine_threshold_bytes)

    @gpu_all_gather_combine_threshold_mib.setter
    def gpu_all_gather_combine_threshold_mib(self, value: float | None):
        if value is None:
            self.gpu_all_gather_combine_threshold_bytes = None
        else:
            self.gpu_all_gather_combine_threshold_bytes = mib_to_bytes(value)

    @property
    def gpu_reduce_scatter_combine_threshold_mib(self) -> float | None:
        if self.gpu_reduce_scatter_combine_threshold_bytes is None:
            return None
        else:
            return bytes_to_mib(self.gpu_reduce_scatter_combine_threshold_bytes)

    @gpu_reduce_scatter_combine_threshold_mib.setter
    def gpu_reduce_scatter_combine_threshold_mib(self, value: float | None):
        if value is None:
            self.gpu_reduce_scatter_combine_threshold_bytes = None
        else:
            self.gpu_reduce_scatter_combine_threshold_bytes = mib_to_bytes(value)

    @property
    def gpu_all_reduce_combine_threshold_mib(self) -> float | None:
        if self.gpu_all_reduce_combine_threshold_bytes is None:
            return None
        else:
            return bytes_to_mib(self.gpu_all_reduce_combine_threshold_bytes)

    @gpu_all_reduce_combine_threshold_mib.setter
    def gpu_all_reduce_combine_threshold_mib(self, value: float | None):
        if value is None:
            self.gpu_all_reduce_combine_threshold_bytes = None
        else:
            self.gpu_all_reduce_combine_threshold_bytes = mib_to_bytes(value)

    def _get_env_flags(self) -> list[str]:
        flags = []
        for name, value in self.__dict__.items():
            if name == "python_client_mem_fraction":
                continue

            value_str: str | None = None
            if isinstance(value, bool):
                value_str = str(value).lower()
            elif isinstance(value, str):
                value_str = value
            elif isinstance(value, int):
                value_str = str(value)
            elif value is not None:
                raise ValueError(value)

            if value_str is not None:
                flags.append(f"--xla_{name}={value_str}")

        return flags

    def apply(self):
        assert 0 <= self.python_client_mem_fraction <= 1.0
        set_env_var(
            "XLA_PYTHON_CLIENT_MEM_FRACTION",
            f"{round(self.python_client_mem_fraction, 2):.2f}",
            override=True,
        )
        set_env_var("XLA_FLAGS", " ".join(self._get_env_flags()), override=True)


@dataclass
class NCCLConfig(_EnvConfig):
    LL128_BUFFSIZE: int = -2
    LL_BUFFSIZE: int = -2
    PROTO: str = "SIMPLE,LL,LL128"

    @classmethod
    def recommend(cls, gpu_architecture: GPUArchitecture | None, **overrides) -> NCCLConfig:
        del gpu_architecture
        return cls(**overrides)

    def _get_env_vars(self) -> list[tuple[str, str]]:
        env_vars = []
        for name, value in self.__dict__.items():
            value_str: str | None = None
            if isinstance(value, str):
                value_str = value
            elif isinstance(value, int):
                value_str = str(value)
            elif value is not None:
                raise ValueError(value)

            if value_str is not None:
                env_vars.append((f"NCCL_{name.upper()}", value_str))

        return env_vars

    def apply(self):
        for name, value in self._get_env_vars():
            set_env_var(name, value, override=True)


@dataclass
class CUDAConfig(_EnvConfig):
    device_max_connections: int | None

    @classmethod
    def recommend(cls, gpu_architecture: GPUArchitecture | None, **overrides) -> CUDAConfig:
        if gpu_architecture != GPUArchitecture.blackwell:
            return cls(device_max_connections=1, **overrides)
        return cls(**overrides)

    def _get_env_vars(self) -> list[tuple[str, str]]:
        env_vars = []
        for name, value in self.__dict__.items():
            value_str: str | None = None
            if isinstance(value, str):
                value_str = value
            elif isinstance(value, int):
                value_str = str(value)
            elif value is not None:
                raise ValueError(value)

            if value_str is not None:
                env_vars.append((f"CUDA_{name.upper()}", value_str))

        return env_vars

    def apply(self):
        for name, value in self._get_env_vars():
            set_env_var(name, value, override=True)


@dataclass
class JAXConfig(_EnvConfig):
    disable_jit: bool | None = None
    compiler_enable_remat_pass: bool | None = None

    @classmethod
    def recommend(cls, gpu_architecture: GPUArchitecture | None, **overrides) -> JAXConfig:
        del gpu_architecture
        return cls(**overrides)

    def apply(self):
        for name, value in self.__dict__.items():
            if value is None:
                continue
            jax.config.update(f"jax_{name}", value)
