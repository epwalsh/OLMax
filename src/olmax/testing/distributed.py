import logging
import multiprocessing as mp
import os
import random
import socket
import sys
import tempfile
from collections import deque
from pathlib import Path
from typing import Any, Callable, Literal, Optional

import jax
import pytest

from .. import distributed as dist

log = logging.getLogger(__name__)


_PORT_MIN = 29500
_PORT_MAX = 30000


def _initialize_ports() -> deque[int]:
    ports = list(range(_PORT_MIN, _PORT_MAX))
    random.Random().shuffle(ports)
    return deque(ports)


_PORTS = _initialize_ports()


def _port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        return s.connect_ex((host, port)) == 0


def _get_next_port() -> int:
    global _PORTS
    port = _PORTS[0]
    _PORTS.rotate()
    return port


def _find_open_port(host: str = "127.0.0.1") -> int:
    port = _get_next_port()
    attempts = 0
    while _port_in_use(host, port):
        port += _get_next_port()
        attempts += 1
        if attempts >= 10:
            raise RuntimeError("failed to find an open port")
    return port


def _init_process(
    *,
    process_rank: int,
    num_processes: int,
    log_from_all_ranks: bool,
    func: Callable,
    func_args: Optional[tuple[Any, ...]] = None,
    func_kwargs: Optional[dict[str, Any]] = None,
    primary_addr: str = "127.0.0.1",
    primary_port: int = 29500,
    backend: Literal["gpu", "tpu", "cpu"] | str | None = None,
    devices_per_process: int | None = None,
):
    if devices_per_process is not None:
        if backend == "cpu":
            os.environ[
                "XLA_FLAGS"
            ] = f"--xla_force_host_platform_device_count={devices_per_process}"
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        elif backend == "gpu":
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = f"{process_rank}"

    os.environ[
        dist.SHARED_FS_DIRS_ENV_VAR
    ] = f"{Path.home()}:{Path(tempfile.gettempdir()).resolve()}"

    old_log_record_factory = logging.getLogRecordFactory()

    def log_record_factory(*args, **kwargs) -> logging.LogRecord:
        record = old_log_record_factory(*args, **kwargs)
        setattr(record, "rank", process_rank)
        return record

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter(
            "[rank %(rank)s] %(asctime)s:%(name)s:%(lineno)s:%(levelname)s: %(message)s"
        )
    )
    logging.setLogRecordFactory(log_record_factory)

    if log_from_all_ranks or process_rank == 0:
        logging.basicConfig(level=logging.INFO, handlers=[handler], force=True)

    if num_processes > 1:
        log.info("Initializing distributed backend...")
        dist.init_distributed(
            coordinator_address=f"{primary_addr}:{primary_port}",
            num_processes=num_processes,
            process_id=process_rank,
        )

    log.info("Starting test...")
    try:
        func(*(func_args or []), **(func_kwargs or {}))
    except BaseException as e:
        log.exception(f"{e}")
        raise
    finally:
        if num_processes > 1:
            dist.teardown_distributed()


def run_distributed_test(
    func: Callable,
    num_processes: int = 2,
    devices_per_process: int | None = None,
    log_from_all_ranks: bool = False,
    start_method: Optional[str] = "spawn",
    args: Optional[tuple[Any, ...]] = None,
    kwargs: Optional[dict[str, Any]] = None,
    primary_addr: str = "127.0.0.1",
    primary_port: Optional[int] = None,
    backend: Literal["gpu", "tpu", "cpu"] | str | None = None,
    timeout: int = 60,
):
    """
    This runs the `func` in a simulated distributed environment.
    """
    if backend is None:
        backend = jax.default_backend()

    total_devices_needed = num_processes
    if devices_per_process is not None:
        total_devices_needed *= devices_per_process
    if backend != "cpu" and jax.device_count(backend) < total_devices_needed:
        pytest.skip(f"Requires at least {total_devices_needed} {backend} devices")

    # Check if we can run the test directly.
    if num_processes == 1 and (
        devices_per_process is None or devices_per_process == jax.device_count()
    ):
        func(*(args or []), **(kwargs or {}))
        return

    ctx = mp.get_context(method=start_method)

    if primary_port is None:
        primary_port = _find_open_port(host=primary_addr)

    log.info(f"Running distributed test on port {primary_port}...")

    with ctx.Pool(processes=num_processes) as pool:
        results = []
        for rank in range(num_processes):
            result = pool.apply_async(
                _init_process,
                [],
                dict(
                    process_rank=rank,
                    num_processes=num_processes,
                    devices_per_process=devices_per_process,
                    log_from_all_ranks=log_from_all_ranks,
                    func=func,
                    func_args=args,
                    func_kwargs=kwargs,
                    primary_addr=primary_addr,
                    primary_port=primary_port,
                    backend=backend,
                ),
            )
            results.append(result)

        for result in results:
            result.get(timeout=timeout)
