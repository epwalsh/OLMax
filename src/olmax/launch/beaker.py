import os
from dataclasses import dataclass

from typing_extensions import Self


@dataclass
class BeakerWorkloadInfo:
    id: str
    task_id: str
    job_id: str
    result_dataset_id: str | None

    @classmethod
    def from_env(cls) -> Self:
        return cls(
            id=os.environ["BEAKER_WORKLOAD_ID"],
            task_id=os.environ["BEAKER_TASK_ID"],
            job_id=os.environ["BEAKER_JOB_ID"],
            result_dataset_id=os.environ.get("BEAKER_RESULT_DATASET_ID"),
        )


@dataclass
class BeakerResourcesInfo:
    gpu_count: int
    cpu_count: int

    @classmethod
    def from_env(cls) -> Self:
        return cls(
            gpu_count=int(os.environ["BEAKER_ASSIGNED_GPU_COUNT"]),
            cpu_count=int(os.environ["BEAKER_ASSIGNED_CPU_COUNT"]),
        )


@dataclass
class BeakerNodeInfo:
    id: str
    hostname: str

    @classmethod
    def from_env(cls) -> Self:
        return cls(
            id=os.environ["BEAKER_NODE_ID"],
            hostname=os.environ["BEAKER_NODE_HOSTNAME"],
        )


@dataclass
class BeakerReplicaInfo:
    rank: int
    count: int
    leader_job_id: str
    leader_node: BeakerNodeInfo

    @classmethod
    def from_env(cls) -> Self | None:
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
    def from_env(cls) -> Self | None:
        if "BEAKER_WORKLOAD_ID" not in os.environ:
            return None

        return cls(
            workload=BeakerWorkloadInfo.from_env(),
            resources=BeakerResourcesInfo.from_env(),
            node=BeakerNodeInfo.from_env(),
            replica=BeakerReplicaInfo.from_env(),
        )
