from __future__ import annotations

import argparse
import functools as ft
import logging
import os
import sys
from dataclasses import dataclass

from beaker import Beaker, BeakerGpuType
from gantry.api import launch_experiment

from ..config import EnvConfig
from ..distributed import DistConfig
from ..types import *
from ..utils import prepare_cli_environment

log = logging.getLogger(__name__)

B200_CLUSTERS = {""}


@dataclass
class BeakerWorkloadInfo:
    id: str
    job_id: str
    task_id: str | None
    result_dataset_id: str | None
    result_dataset_path: str | None

    @classmethod
    def from_env(cls) -> BeakerWorkloadInfo:
        return cls(
            id=os.environ["BEAKER_WORKLOAD_ID"],
            task_id=os.environ.get("BEAKER_TASK_ID"),
            job_id=os.environ["BEAKER_JOB_ID"],
            result_dataset_id=os.environ.get("BEAKER_RESULT_DATASET_ID"),
            result_dataset_path=os.environ.get("RESULTS_DIR"),
        )

    @ft.cached_property
    def url(self) -> str:
        with Beaker.from_env() as beaker:
            workload = beaker.workload.get(self.id)
            return beaker.workload.url(workload)


@dataclass
class BeakerResourcesInfo:
    gpu_count: int
    cpu_count: int

    @classmethod
    def from_env(cls) -> BeakerResourcesInfo:
        return cls(
            gpu_count=int(os.environ["BEAKER_ASSIGNED_GPU_COUNT"]),
            cpu_count=int(os.environ["BEAKER_ASSIGNED_CPU_COUNT"]),
        )


@dataclass
class BeakerNodeInfo:
    id: str
    hostname: str

    @classmethod
    def from_env(cls) -> BeakerNodeInfo:
        return cls(
            id=os.environ["BEAKER_NODE_ID"],
            hostname=os.environ["BEAKER_NODE_HOSTNAME"],
        )

    @ft.cached_property
    def gpu_type(self) -> GPUType | None:
        with Beaker.from_env() as beaker:
            node = beaker.node.get(self.id)
            try:
                beaker_gpu_type = BeakerGpuType(node.node_resources.gpu_type)
            except ValueError:
                return None
        return GPUType(beaker_gpu_type.name)

    @property
    def gpu_architecture(self) -> GPUArchitecture | None:
        gpu_type = self.gpu_type
        if gpu_type is None:
            return None
        elif "H100" in gpu_type.name:
            return GPUArchitecture.hopper
        elif "B200" in gpu_type.name:
            return GPUArchitecture.blackwell
        elif "A100" in gpu_type.name:
            return GPUArchitecture.ampere
        else:
            raise ValueError(f"unexpected GPU type {gpu_type}")


@dataclass
class BeakerReplicaInfo:
    rank: int
    count: int
    leader_job_id: str
    leader_node: BeakerNodeInfo

    @classmethod
    def from_env(cls) -> BeakerReplicaInfo | None:
        if "BEAKER_REPLICA_RANK" not in os.environ:
            return None

        return cls(
            rank=int(os.environ["BEAKER_REPLICA_RANK"]),
            count=int(os.environ["BEAKER_REPLICA_COUNT"]),
            leader_job_id=os.environ["BEAKER_LEADER_REPLICA_JOB_ID"],
            leader_node=BeakerNodeInfo(
                id=os.environ["BEAKER_LEADER_REPLICA_NODE_ID"],
                hostname=os.environ["BEAKER_LEADER_REPLICA_HOSTNAME"],
            ),
        )


@dataclass
class BeakerRuntime:
    workload: BeakerWorkloadInfo
    resources: BeakerResourcesInfo
    node: BeakerNodeInfo
    replica: BeakerReplicaInfo | None

    @classmethod
    def from_env(cls) -> BeakerRuntime | None:
        if "BEAKER_WORKLOAD_ID" not in os.environ:
            return None

        beaker_runtime = cls(
            workload=BeakerWorkloadInfo.from_env(),
            resources=BeakerResourcesInfo.from_env(),
            node=BeakerNodeInfo.from_env(),
            replica=BeakerReplicaInfo.from_env(),
        )

        info = [
            f"Running in Beaker on node '{beaker_runtime.node.hostname}'",
            f"❯ Workload: {beaker_runtime.workload.url}",
        ]
        if (gpu_count := beaker_runtime.resources.gpu_count) > 0 and (
            gpu_type := beaker_runtime.node.gpu_type
        ) is not None:
            arch = beaker_runtime.node.gpu_architecture
            info.append(
                f"❯ Resources: {gpu_count} {gpu_type.replace('_', ' ')} GPU(s) ({arch} architecture)"
            )
        if (replica := beaker_runtime.replica) is not None:
            info.append(f"❯ Replicas: {replica.count}")

        log.info("\n".join(info))

        return beaker_runtime

    @property
    def cluster_nickname(self) -> str:
        return self.node.hostname.split("-")[0]

    @property
    def is_experiment(self) -> bool:
        return self.workload.task_id is not None

    def set_description(self, description: str):
        if self.replica is not None and self.replica.rank != 0:
            return

        with Beaker.from_env() as beaker:
            workload = beaker.workload.get(self.workload.id)
            beaker.workload.update(workload, description=description)

    def get_dist_config(self) -> DistConfig | None:
        if self.replica is None:
            return None
        else:
            return DistConfig(
                coordinator_address=f"{self.replica.leader_node.hostname}:29400",
                num_processes=self.replica.count,
                process_id=self.replica.rank,
            )

    def get_env_config(self) -> EnvConfig:
        print("yeup!!!!!!")
        env = EnvConfig.recommended(self.node.gpu_architecture)
        print(self.replica)
        print(self.node.hostname)
        if self.replica is not None and "augusta" in self.node.hostname:
            env.nccl.proto = "Simple,LL128"
            env.nccl.tuner_config_path = "/var/lib/tcpxo/lib64/a3plus_tuner_config_ll128.textproto"
            env.nccl.shimnet_guest_config_checker_config_file = (
                "/var/lib/tcpxo/lib64/a3plus_guest_config_ll128.textproto"
            )
        return env


def _parse_args():
    parser = argparse.ArgumentParser(
        "olmax.launch.beaker", usage="python -m olmax.launch.beaker [OPTIONS...] -- [CMD...]"
    )
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--gpus-per-node", type=int, default=8)
    parser.add_argument("--gpu-type", type=str, choices=["h100", "b200"])
    parser.add_argument("--cluster", type=str, nargs="*")
    parser.add_argument("--hostname", type=str, nargs="*")
    parser.add_argument("--allow-dirty", action="store_true")

    if len(sys.argv) < 3 or "--" not in sys.argv:
        parser.print_help()
        sys.exit(1)

    sep_index = sys.argv.index("--")
    args = sys.argv[1:sep_index]
    command = sys.argv[sep_index + 1 :]
    opts = parser.parse_args(args)
    return opts, tuple(command)


def main():
    prepare_cli_environment()
    opts, command = _parse_args()
    is_multi_node = opts.nodes > 1
    launch_experiment(
        command,
        priority="high",
        yes=True,
        timeout=-1,
        gpus=opts.gpus_per_node,
        gpu_types=None if not opts.gpu_type else (opts.gpu_type,),
        clusters=opts.cluster,
        hostnames=opts.hostname,
        beaker_image="petew/olmax",
        env_vars=["PYTHONUNBUFFERED=1", "FORCE_COLOR=1"],
        env_secrets=["BEAKER_TOKEN=PETEW_BEAKER_TOKEN"],
        allow_dirty=opts.allow_dirty,
        install="pip install -e '.[all]'",
        replicas=opts.nodes if is_multi_node else None,
        leader_selection=is_multi_node,
        host_networking=is_multi_node,
        propagate_failure=is_multi_node,
        propagate_preemption=is_multi_node,
        synchronized_start_timeout="5m" if is_multi_node else None,
    )


if __name__ == "__main__":
    main()
