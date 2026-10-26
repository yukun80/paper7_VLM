"""M7 CUDA microbatch 同步遥测、allocator 身份与 fail-closed 资源门。"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Mapping

import torch

from oa_groundrag.artifacts.identity import canonical_json
from oa_groundrag.artifacts.io import atomic_write_json, first_symlink_component
from oa_groundrag.data.rs_general.io import read_json, read_jsonl
from oa_groundrag.vlm.errors import ModelError, ReasonCode


CUDA_TELEMETRY_SCHEMA = "rs_vlm.cuda_microbatch_telemetry.v1"
CUDA_TELEMETRY_IDENTITY_SCHEMA = "rs_vlm.cuda_microbatch_telemetry_identity.v1"
CUDA_STEP_TELEMETRY_SCHEMA = "rs_vlm.cuda_step_telemetry.v1"
CUDA_FAILURE_SCHEMA = "rs_vlm.cuda_resource_failure.v1"

ALLOCATOR_NATIVE = "native"
ALLOCATOR_EXPANDABLE = "expandable_segments"
ALLOCATOR_EXPANDABLE_MICRO_CACHE = (
    "expandable_segments_post_backward_empty_cache"
)
MICRO_CACHE_NONE = "none"
MICRO_CACHE_POST_BACKWARD = "post_backward_empty_cache"
POST_ZERO_GRAD_GROWTH_LIMIT_BYTES = 256 * 1024 * 1024

_ALLOCATOR_ENV_NAMES = ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF")


@dataclass(frozen=True)
class CudaTelemetryPolicy:
    schema_version: str
    allocator_profile: str
    microbatch_cache_policy: str
    min_cuda_free_bytes: int
    max_microbatches: int
    synchronize_cuda: bool
    resource_profile_identity: Mapping[str, Any]

    def layout_identity(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "allocator_profile": self.allocator_profile,
            "microbatch_cache_policy": self.microbatch_cache_policy,
            "min_cuda_free_bytes": self.min_cuda_free_bytes,
            "max_microbatches": self.max_microbatches,
            "synchronize_cuda": self.synchronize_cuda,
            "resource_profile_identity": dict(self.resource_profile_identity),
        }


def allocator_environment_identity(profile: str) -> dict[str, Any]:
    """在创建 CUDA allocator 前验证环境变量与所选 profile 一致。"""

    values = {name: os.environ.get(name) for name in _ALLOCATOR_ENV_NAMES}
    present = {name: value for name, value in values.items() if value not in {None, ""}}
    if profile == ALLOCATOR_NATIVE:
        valid = not present
    elif profile in {ALLOCATOR_EXPANDABLE, ALLOCATOR_EXPANDABLE_MICRO_CACHE}:
        normalized = {
            name: str(value).replace(" ", "").lower()
            for name, value in present.items()
        }
        valid = (
            len(normalized) == 1
            and next(iter(normalized.values())) == "expandable_segments:true"
        )
    else:
        valid = False
    if not valid:
        raise ModelError(
            ReasonCode.CHECKPOINT_INCOMPATIBLE,
            "allocator 环境与 M7 v4 profile 不匹配",
            details={"profile": profile, "environment": values},
        )
    return {
        "profile": profile,
        "environment": values,
    }


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    """追加单条 durable JSONL；每条 microbatch 独立落盘以保留 OOM 前证据。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(canonical_json(dict(row)))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


class CudaMicrobatchTelemetry:
    """同步采样 allocator/device 状态，并在越过冻结阈值时立即停止。"""

    def __init__(
        self,
        *,
        policy: CudaTelemetryPolicy,
        device: torch.device,
    ) -> None:
        if (
            policy.schema_version != CUDA_TELEMETRY_SCHEMA
            or device.type != "cuda"
            or device.index not in {None, 0}
            or policy.min_cuda_free_bytes <= 0
            or policy.max_microbatches <= 0
            or policy.synchronize_cuda is not True
            or policy.microbatch_cache_policy
            not in {MICRO_CACHE_NONE, MICRO_CACHE_POST_BACKWARD}
        ):
            raise ModelError(
                ReasonCode.TYPE_MISMATCH,
                "M7 CUDA telemetry policy/device 非法",
            )
        if (
            policy.allocator_profile == ALLOCATOR_EXPANDABLE_MICRO_CACHE
        ) != (
            policy.microbatch_cache_policy == MICRO_CACHE_POST_BACKWARD
        ):
            raise ModelError(
                ReasonCode.TYPE_MISMATCH,
                "allocator profile 与 microbatch cache policy 不匹配",
            )
        self.policy = policy
        self.device = device
        self.output_root: Path | None = None
        self.completed_microbatches = 0
        self.current: dict[str, Any] | None = None
        self._initial_num_alloc_retries: int | None = None
        self._initial_num_ooms: int | None = None

    @property
    def microbatch_cache_enabled(self) -> bool:
        return self.policy.microbatch_cache_policy == MICRO_CACHE_POST_BACKWARD

    def layout_identity(self) -> dict[str, Any]:
        return self.policy.layout_identity()

    def _paths(self) -> dict[str, Path]:
        if self.output_root is None:
            raise AssertionError("CUDA telemetry 尚未 start")
        return {
            "identity": self.output_root / "cuda_resource_identity.json",
            "micro": self.output_root / "cuda_resource_telemetry.jsonl",
            "step": self.output_root / "cuda_step_telemetry.jsonl",
            "active": self.output_root / "cuda_active_microbatch.json",
            "failure": self.output_root / "cuda_resource_failure.json",
        }

    def _runtime_identity(self) -> dict[str, Any]:
        environment = allocator_environment_identity(
            self.policy.allocator_profile
        )
        properties = torch.cuda.get_device_properties(self.device)
        allocator_backend = torch.cuda.memory.get_allocator_backend()
        if allocator_backend != "native":
            raise ModelError(
                ReasonCode.CHECKPOINT_INCOMPATIBLE,
                "M7 v4 只允许 PyTorch native CUDA allocator backend",
                details={"allocator_backend": allocator_backend},
            )
        return {
            "schema_version": CUDA_TELEMETRY_IDENTITY_SCHEMA,
            "policy": self.policy.layout_identity(),
            "allocator_environment": environment,
            "allocator_backend": allocator_backend,
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "device_index": 0 if self.device.index is None else self.device.index,
            "device_name": properties.name,
            "device_total_memory_bytes": int(properties.total_memory),
        }

    def start(self, output_root: Path, *, completed_microbatches: int) -> None:
        if self.output_root is not None:
            raise AssertionError("CUDA telemetry 不能重复 start")
        if (
            isinstance(completed_microbatches, bool)
            or not isinstance(completed_microbatches, int)
            or not 0 <= completed_microbatches <= self.policy.max_microbatches
        ):
            raise ModelError(
                ReasonCode.CHECKPOINT_INCOMPATIBLE,
                "CUDA telemetry 已完成 microbatch 计数非法",
            )
        root = Path(output_root)
        linked = first_symlink_component(root)
        if linked is not None or not root.is_dir() or root.is_symlink():
            raise ModelError(
                ReasonCode.OUTPUT_LINK,
                "CUDA telemetry output_root 必须是 trainer 已创建的普通目录",
            )
        self.output_root = root
        paths = self._paths()
        runtime_identity = self._runtime_identity()
        if completed_microbatches == 0:
            if any(path.exists() or path.is_symlink() for path in paths.values()):
                raise ModelError(
                    ReasonCode.OUTPUT_EXISTS,
                    "fresh CUDA telemetry root 已有资源日志",
                )
            atomic_write_json(paths["identity"], runtime_identity)
            micro_rows: list[dict[str, Any]] = []
            step_rows: list[dict[str, Any]] = []
        else:
            if paths["failure"].exists() or paths["failure"].is_symlink():
                raise ModelError(
                    ReasonCode.CHECKPOINT_INCOMPATIBLE,
                    "存在 CUDA failure evidence 的根禁止恢复",
                )
            if read_json(paths["identity"]) != runtime_identity:
                raise ModelError(
                    ReasonCode.CHECKPOINT_INCOMPATIBLE,
                    "CUDA allocator/device identity 与恢复运行不一致",
                )
            micro_rows = read_jsonl(paths["micro"])
            step_rows = read_jsonl(paths["step"])
            if (
                len(micro_rows) != completed_microbatches
                or not step_rows
                or micro_rows[-1].get("sequence_index")
                != completed_microbatches - 1
                or read_json(paths["active"]).get("status") != "completed"
            ):
                raise ModelError(
                    ReasonCode.CHECKPOINT_INCOMPATIBLE,
                    "CUDA telemetry 与 sample trace/cursor 不一致",
                )
        self.completed_microbatches = completed_microbatches
        stats = torch.cuda.memory_stats(self.device)
        self._initial_num_alloc_retries = int(stats.get("num_alloc_retries", 0))
        self._initial_num_ooms = int(stats.get("num_ooms", 0))
        if completed_microbatches == 0:
            atomic_write_json(
                paths["active"],
                {"schema_version": CUDA_TELEMETRY_SCHEMA, "status": "completed"},
            )

    def _snapshot(self, stage: str) -> dict[str, Any]:
        torch.cuda.synchronize(self.device)
        stats = torch.cuda.memory_stats(self.device)
        free_bytes, total_bytes = torch.cuda.mem_get_info(self.device)
        snapshot = {
            "stage": stage,
            "allocated_bytes": int(torch.cuda.memory_allocated(self.device)),
            "reserved_bytes": int(torch.cuda.memory_reserved(self.device)),
            "active_bytes": int(stats.get("active_bytes.all.current", 0)),
            "inactive_split_bytes": int(
                stats.get("inactive_split_bytes.all.current", 0)
            ),
            "device_free_bytes": int(free_bytes),
            "device_total_bytes": int(total_bytes),
            "device_used_bytes": int(total_bytes - free_bytes),
            "num_alloc_retries": int(stats.get("num_alloc_retries", 0)),
            "num_ooms": int(stats.get("num_ooms", 0)),
        }
        if snapshot["device_free_bytes"] < self.policy.min_cuda_free_bytes:
            raise ModelError(
                ReasonCode.CUDA_REQUIRED,
                "M7 CUDA 同步采样点设备余量低于 2 GiB",
                details={"snapshot": snapshot},
            )
        if (
            self._initial_num_alloc_retries is not None
            and snapshot["num_alloc_retries"]
            != self._initial_num_alloc_retries
        ):
            raise ModelError(
                ReasonCode.CUDA_REQUIRED,
                "M7 CUDA allocator retry 计数增加",
                details={"snapshot": snapshot},
            )
        if (
            self._initial_num_ooms is not None
            and snapshot["num_ooms"] != self._initial_num_ooms
        ):
            raise ModelError(
                ReasonCode.CUDA_REQUIRED,
                "M7 CUDA allocator OOM 计数增加",
                details={"snapshot": snapshot},
            )
        return snapshot

    def begin(self, metadata: Mapping[str, Any]) -> None:
        if self.current is not None:
            raise AssertionError("上一个 CUDA microbatch 尚未完成")
        if self.completed_microbatches >= self.policy.max_microbatches:
            raise ModelError(
                ReasonCode.ASSET_LIMIT_EXCEEDED,
                "CUDA telemetry 超过配置的 microbatch 上限",
            )
        row = {
            "schema_version": CUDA_TELEMETRY_SCHEMA,
            "status": "running",
            "sequence_index": self.completed_microbatches,
            **dict(metadata),
        }
        self.current = row
        atomic_write_json(self._paths()["active"], row)
        self.current["pre_forward"] = self._snapshot("pre_forward")
        atomic_write_json(self._paths()["active"], self.current)

    def after_forward(self) -> None:
        if self.current is None:
            raise AssertionError("CUDA telemetry 未 begin")
        self.current["post_forward"] = self._snapshot("post_forward")
        atomic_write_json(self._paths()["active"], self.current)

    def after_backward(self) -> None:
        if self.current is None:
            raise AssertionError("CUDA telemetry 未 begin")
        self.current["post_backward"] = self._snapshot("post_backward")
        atomic_write_json(self._paths()["active"], self.current)

    def complete_after_release(self) -> None:
        if self.current is None:
            raise AssertionError("CUDA telemetry 未 begin")
        if self.microbatch_cache_enabled:
            torch.cuda.empty_cache()
        self.current["post_release"] = self._snapshot("post_release")
        self.current["status"] = "completed"
        _append_jsonl(self._paths()["micro"], self.current)
        atomic_write_json(self._paths()["active"], self.current)
        self.completed_microbatches += 1
        self.current = None

    def record_post_zero_grad(self, *, optimizer_step: int) -> None:
        snapshot = self._snapshot("post_zero_grad")
        row = {
            "schema_version": CUDA_STEP_TELEMETRY_SCHEMA,
            "optimizer_step": optimizer_step,
            "completed_microbatches": self.completed_microbatches,
            "snapshot": snapshot,
        }
        paths = self._paths()
        prior = read_jsonl(paths["step"]) if paths["step"].exists() else []
        if prior and int(prior[-1]["optimizer_step"]) >= optimizer_step:
            raise ModelError(
                ReasonCode.CHECKPOINT_INCOMPATIBLE,
                "CUDA step telemetry optimizer step 未严格递增",
            )
        baseline = next(
            (item for item in prior if item.get("optimizer_step") == 1),
            None,
        )
        if optimizer_step in {20, 100}:
            if baseline is None:
                raise ModelError(
                    ReasonCode.CHECKPOINT_INCOMPATIBLE,
                    "CUDA step telemetry 缺少 step-1 稳态基线",
                )
            limit = (
                int(baseline["snapshot"]["allocated_bytes"])
                + POST_ZERO_GRAD_GROWTH_LIMIT_BYTES
            )
            if snapshot["allocated_bytes"] > limit:
                raise ModelError(
                    ReasonCode.CUDA_REQUIRED,
                    "step 20/100 post-zero-grad allocated 超过 step-1 + 256 MiB",
                    details={
                        "optimizer_step": optimizer_step,
                        "allocated_bytes": snapshot["allocated_bytes"],
                        "limit_bytes": limit,
                    },
                )
        _append_jsonl(paths["step"], row)

    def persist_failure(
        self,
        error: BaseException,
        *,
        last_completed_optimizer_step: int,
        last_completed_microbatches: int,
    ) -> None:
        """OOM 后只使用 CPU 已存状态写证据，绝不再调用 CUDA。"""

        if self.output_root is None:
            return
        paths = self._paths()
        failure = {
            "schema_version": CUDA_FAILURE_SCHEMA,
            "exception_type": type(error).__name__,
            "exception_message": str(error),
            "last_completed_optimizer_step": last_completed_optimizer_step,
            "last_completed_microbatches": last_completed_microbatches,
            "last_planned_microbatch": self.current,
            "resource_policy": self.policy.layout_identity(),
        }
        if paths["failure"].exists() or paths["failure"].is_symlink():
            return
        atomic_write_json(paths["failure"], failure)
