from __future__ import annotations

import dataclasses
import json
import os
import sys
import tempfile
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generator, Sequence, Type, TypeVar

import draccus
import jax

from .types import *
from .utils import bytes_to_mib, mib_to_bytes, set_env_var

draccus.encode.register(type(float), lambda x, _=None: x.__name__)
draccus.decode.register(DTypeLike, lambda r, _: r)

RegistrableConfig = draccus.ChoiceRegistry


C = TypeVar("C")


def encode(
    data: Any,
    *,
    exclude_none: bool = False,
    exclude_private_fields: bool = False,
    json_safe: bool = False,
    recurse: bool = True,
) -> dict[str, Any]:
    """
    Convert into a regular Python dictionary.

    :param exclude_none: Don't include values that are ``None``.
    :param exclude_private_fields: Don't include private fields.
    :param json_safe: Output only JSON-safe types.
    :param recurse: Recurse into fields that are also configs/dataclasses.
    """

    def iter_fields(d) -> Generator[tuple[str, Any], None, None]:
        for field in dataclasses.fields(d):
            value = getattr(d, field.name)
            if exclude_none and value is None:
                continue
            elif exclude_private_fields and field.name.startswith("_"):
                continue
            else:
                yield (field.name, value)

    def as_dict(d: Any, recurse: bool = True) -> Any:
        if dataclasses.is_dataclass(d):
            if recurse:
                out = {k: as_dict(v) for k, v in iter_fields(d)}
            else:
                out = {k: v for k, v in iter_fields(d)}
            if isinstance(d, RegistrableConfig):
                out["type"] = d.get_choice_name(d.__class__)
            return out
        elif isinstance(d, dict):
            return {k: as_dict(v) for k, v in d.items()}
        elif isinstance(d, (list, tuple, set)):
            if json_safe:
                return [as_dict(x) for x in d]
            else:
                return d.__class__((as_dict(x) for x in d))
        elif d is None or isinstance(d, (float, int, bool, str)):
            return d
        elif json_safe:
            if hasattr(d, "__name__"):
                return d.__name__
            else:
                return str(d)
        else:
            return d

    return as_dict(data, recurse=recurse)


def _clean_opts(opts: Sequence[str]) -> list[str]:
    return [_clean_opt(s) for s in opts]


def _clean_opt(arg: str) -> str:
    if arg in {"-h", "--help"}:
        return arg
    if "=" not in arg:
        arg = f"{arg}=True"
    name, val = arg.split("=", 1)
    name = name.strip("-").replace("-", "_")
    return f"--{name}={val}"


def parse_config_from_args(
    config_class: Type[C],
    default: C | PathOrStr,
    *,
    args: Sequence[str] | None = None,
    prog: str | None = None,
) -> C:
    """
    Parse a config dataclass from command-line args.
    """

    # NOTE: a default is required because otherwise draccus won't respect default factory functions
    # when trying to override a single field.
    defaults_path: PathOrStr
    if isinstance(default, (str, Path)):
        defaults_path = default
    else:
        with tempfile.NamedTemporaryFile("w+t", delete=False) as tmp_file:
            json_safe = encode(default, json_safe=True)
            json.dump(json_safe, tmp_file)
            defaults_path = tmp_file.name

    if args is None and sys.argv:
        args = sys.argv[1:]

    if prog is None and sys.argv:
        prog = sys.argv[0]

    if args:
        args = _clean_opts(args)

    try:
        return draccus.parse(
            config_class=config_class, config_path=defaults_path, args=args, prog=prog
        )
    finally:
        if defaults_path is not None:
            os.remove(defaults_path)


@dataclass
class _EnvBaseConfig:
    @classmethod
    @abstractmethod
    def recommended(
        cls: Type[C], gpu_architecture: GPUArchitecture | None = None, **overrides
    ) -> C:
        raise NotImplementedError

    @abstractmethod
    def apply(self):
        """
        Apply the settings.
        """
        raise NotImplementedError


@dataclass
class XLAConfig(_EnvBaseConfig):
    """
    XLA environment configuration.

    See here for good defaults:
    - https://github.com/NVIDIA/JAX-Toolbox/blob/main/rosetta/docs/GPU_performance.md
    - https://docs.jax.dev/en/latest/gpu_performance_tips.html
    """

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
    gpu_enable_nccl_per_stream_comms: bool | None = None

    gpu_enable_while_loop_double_buffering: bool = True
    gpu_enable_command_buffer: str | None = None

    gpu_enable_triton_gemm: bool = False

    @classmethod
    def recommended(cls, gpu_architecture: GPUArchitecture | None = None, **overrides) -> XLAConfig:
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
class NCCLConfig(_EnvBaseConfig):
    LL128_buffsize: int = -2
    LL_buffsize: int = -2
    proto: str = "Simple,LL,LL128"
    debug: str = "WARN"
    tuner_config_path: str | None = None
    shimnet_guest_config_checker_config_file: str | None = None

    @classmethod
    def recommended(
        cls, gpu_architecture: GPUArchitecture | None = None, **overrides
    ) -> NCCLConfig:
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
class CUDAConfig(_EnvBaseConfig):
    device_max_connections: int | None = None

    @classmethod
    def recommended(
        cls, gpu_architecture: GPUArchitecture | None = None, **overrides
    ) -> CUDAConfig:
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
class JAXConfig(_EnvBaseConfig):
    disable_jit: bool | None = None
    compiler_enable_remat_pass: bool | None = None

    @classmethod
    def recommended(cls, gpu_architecture: GPUArchitecture | None = None, **overrides) -> JAXConfig:
        del gpu_architecture
        return cls(**overrides)

    def apply(self):
        for name, value in self.__dict__.items():
            if value is None:
                continue
            jax.config.update(f"jax_{name}", value)


@dataclass
class EnvConfig(_EnvBaseConfig):
    xla: XLAConfig = dataclasses.field(default_factory=XLAConfig)
    jax: JAXConfig = dataclasses.field(default_factory=JAXConfig)
    nccl: NCCLConfig = dataclasses.field(default_factory=NCCLConfig)
    cuda: CUDAConfig = dataclasses.field(default_factory=CUDAConfig)

    @classmethod
    def recommended(cls, gpu_architecture: GPUArchitecture | None = None) -> EnvConfig:  # type: ignore[override]
        return cls(
            xla=XLAConfig.recommended(gpu_architecture),
            jax=JAXConfig.recommended(gpu_architecture),
            nccl=NCCLConfig.recommended(gpu_architecture),
            cuda=CUDAConfig.recommended(gpu_architecture),
        )

    def apply(self):
        self.xla.apply()
        self.jax.apply()
        self.nccl.apply()
        self.cuda.apply()


def _main():
    from rich import print

    cfg = parse_config_from_args(EnvConfig, EnvConfig(cuda=CUDAConfig(device_max_connections=1)))
    print(cfg)


if __name__ == "__main__":
    _main()
