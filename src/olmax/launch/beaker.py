from __future__ import annotations

import argparse
import functools as ft
import os
import sys
from dataclasses import dataclass
from typing import Literal

from beaker import Beaker, BeakerGpuType
from gantry.api import launch_experiment

from ..utils import prepare_cli_environment, set_env_var

B200_CLUSTERS = {""}


@dataclass
class BeakerWorkloadInfo:
    id: str
    task_id: str
    job_id: str
    result_dataset_id: str | None
    result_dataset_path: str | None

    @classmethod
    def from_env(cls) -> BeakerWorkloadInfo:
        return cls(
            id=os.environ["BEAKER_WORKLOAD_ID"],
            task_id=os.environ["BEAKER_TASK_ID"],
            job_id=os.environ["BEAKER_JOB_ID"],
            result_dataset_id=os.environ.get("BEAKER_RESULT_DATASET_ID"),
            result_dataset_path=os.environ.get("RESULTS_DIR"),
        )


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
    def gpu_type(self) -> BeakerGpuType | None:
        with Beaker.from_env() as beaker:
            node = beaker.node.get(self.id)
            try:
                return BeakerGpuType(node.node_resources.gpu_type)
            except ValueError:
                return None

    @property
    def gpu_architecture(self) -> Literal["hopper", "blackwell", "ampere"] | None:
        gpu_type = self.gpu_type
        if gpu_type is None:
            return None
        elif "H100" in gpu_type.name:
            return "hopper"
        elif "B200" in gpu_type.name:
            return "blackwell"
        elif "A100" in gpu_type.name:
            return "ampere"
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
        if "BEAKER_WORKLOAD_ID" not in os.environ or "BEAKER_TASK_ID" not in os.environ:
            return None

        return cls(
            workload=BeakerWorkloadInfo.from_env(),
            resources=BeakerResourcesInfo.from_env(),
            node=BeakerNodeInfo.from_env(),
            replica=BeakerReplicaInfo.from_env(),
        )

    @property
    def cluster_nickname(self) -> str:
        return self.node.hostname.split("-")[0]

    def set_description(self, description: str):
        if self.replica is not None and self.replica.rank != 0:
            return

        with Beaker.from_env() as beaker:
            workload = beaker.workload.get(self.workload.id)
            beaker.workload.update(workload, description=description)

    def set_env_vars(self):
        multi_node = self.replica is not None and self.replica.count > 1

        if "titan" in self.node.hostname:
            #  set_env_var("NCCL_P2P_NET_CHUNKSIZE", "131072")
            pass
        elif "jupiter" in self.node.hostname:
            #  set_env_var("NCCL_IB_HCA", "^=mlx5_bond_0")
            #  set_env_var("NCCL_SOCKET_IFNAME", "ib")
            pass
        elif "pluto" in self.node.hostname:
            set_env_var("NCCL_IB_HCA", "^=mlx5_1,mlx5_2")
        elif "augusta" in self.node.hostname and multi_node:
            # See https://beaker-docs.apps.allenai.org/compute/augusta.html#distributed-workloads
            # NOTE: This path var must be set prior to launching Python
            #  set_env_var(
            #      "LD_LIBRARY_PATH",
            #      "/var/lib/tcpxo/lib64:" + os.environ.get("LD_LIBRARY_PATH", ""),
            #      override=True,
            #  )
            set_env_var("NCCL_CROSS_NIC", "0")
            set_env_var("NCCL_ALGO", "Ring,Tree")
            set_env_var("NCCL_PROTO", "Simple,LL128")

            set_env_var("NCCL_MIN_NCHANNELS", "4")
            set_env_var("NCCL_P2P_NET_CHUNKSIZE", "524288")
            set_env_var("NCCL_P2P_PCI_CHUNKSIZE", "524288")
            set_env_var("NCCL_P2P_NVL_CHUNKSIZE", "1048576")

            set_env_var("NCCL_FASTRAK_NUM_FLOWS", "2")
            set_env_var("NCCL_FASTRAK_ENABLE_CONTROL_CHANNEL", "0")
            #  set_env_var("NCCL_BUFFSIZE", "8388608")
            set_env_var("NCCL_FASTRAK_USE_SNAP", "1")
            #  set_env_var("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
            set_env_var("NCCL_NET_GDR_LEVEL", "PIX")
            set_env_var("NCCL_FASTRAK_ENABLE_HOTPATH_LOGGING", "0")
            set_env_var("NCCL_FASTRAK_PLUGIN_ACCEPT_TIMEOUT_MS", "600000")
            set_env_var("NCCL_NVLS_ENABLE", "0")
            set_env_var("NCCL_USE_SNAP", "1")
            set_env_var("NCCL_FASTRAK_USE_LLCM", "1")
            set_env_var("NCCL_FASTRAK_LLCM_DEVICE_DIRECTORY", "/dev/aperture_devices")
            set_env_var("NCCL_TUNER_PLUGIN", "libnccl-tuner.so")
            set_env_var(
                "NCCL_TUNER_CONFIG_PATH", "/var/lib/tcpxo/lib64/a3plus_tuner_config_ll128.textproto"
            )
            set_env_var(
                "NCCL_SHIMNET_GUEST_CONFIG_CHECKER_CONFIG_FILE",
                "/var/lib/tcpxo/lib64/a3plus_guest_config_ll128.textproto",
            )
            set_env_var("NCCL_FASTRAK_CTRL_DEV", "enp0s12")
            set_env_var(
                "NCCL_FASTRAK_IFNAME",
                "enp6s0,enp7s0,enp13s0,enp14s0,enp134s0,enp135s0,enp141s0,enp142s0",
            )
            set_env_var("NCCL_SOCKET_IFNAME", "enp0s12")
            set_env_var("NCCL_DEBUG_SUBSYS", "INIT,NET")


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
        gpu_types=(opts.gpu_type,),
        clusters=opts.cluster,
        hostnames=opts.hostname,
        beaker_image="petew/olmax",
        env_vars=["PYTHONUNBUFFERED=1", "NCCL_DEBUG=info", "FORCE_COLOR=1"],
        env_secrets=["BEAKER_TOKEN=PETEW_BEAKER_TOKEN"],
        allow_dirty=opts.allow_dirty,
        install="./src/scripts/beaker/setup_env.sh",
        replicas=opts.nodes if is_multi_node else None,
        leader_selection=is_multi_node,
        host_networking=is_multi_node,
        propagate_failure=is_multi_node,
        propagate_preemption=is_multi_node,
        synchronized_start_timeout="5m" if is_multi_node else None,
    )


if __name__ == "__main__":
    main()
