"""Unified Demo task/suite runner；只编排现有 capability，不实现模型数学。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from io import BytesIO
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence
from uuid import uuid4

import numpy as np
from PIL import Image

from oa_groundrag.artifacts.directory import AtomicArtifactDirectory
from oa_groundrag.artifacts.identity import (
    canonical_json,
    sha256_bytes,
    sha256_file,
    sha256_text,
)
from oa_groundrag.artifacts.io import first_symlink_component
from oa_groundrag.data.oa_auxseg.dataset import collate_benchmark_samples
from oa_groundrag.grounding.evidence import render_evidence_image, render_mask_overlay
from oa_groundrag.runtime.config import build_unified_runtime
from oa_groundrag.runtime.contracts import (
    InMemorySpatialInput,
    RegionSource,
    UnifiedInferenceError,
    UnifiedRequest,
    UnifiedResponse,
    UnifiedTask,
)
from oa_groundrag.runtime.inference import UnifiedInferenceRuntime
from oa_groundrag.runtime.providers import (
    GroundedEvidenceResult,
    SpatialResult,
    TextRAGResult,
)
from oa_groundrag.segmentation.config import load_runtime_config
from oa_groundrag.segmentation.data import prepare_collated_batch

from .access import DemoAuthorizedSpatialInput, DemoInferenceAccess
from .catalog import (
    BenchmarkCatalog,
    BenchmarkRecord,
    LoadedBenchmarkSample,
)
from .config import DemoConfig
from .previews import InputChannelPreview


DEMO_RUN_MANIFEST_SCHEMA = "oa_groundrag.unified_demo.run_manifest.v2"
DEMO_RUN_SUMMARY_SCHEMA = "oa_groundrag.unified_demo.run_summary.v2"
DEMO_VIEWER_SCHEMA = "oa_groundrag.unified_demo.viewer.v2"
DEMO_SPATIAL_SNAPSHOT_SCHEMA = "oa_groundrag.unified_demo.spatial_snapshot.v1"
KNOWLEDGE_ONLY_MODE = "Knowledge QA / No Benchmark Payload"
WAITING_FOR_CANDIDATE = "WAITING_FOR_CANDIDATE"

TASK_ORDER = (
    UnifiedTask.VLM_ONLY,
    UnifiedTask.SEGMENT_ONLY,
    UnifiedTask.REGION_UNDERSTANDING,
    UnifiedTask.SEGMENT_AND_UNDERSTAND,
    UnifiedTask.KNOWLEDGE_QA,
    UnifiedTask.REGION_INTERPRETATION,
)

FRESH_SPATIAL_TASKS = frozenset({
    UnifiedTask.SEGMENT_ONLY,
    UnifiedTask.SEGMENT_AND_UNDERSTAND,
})
PAYLOAD_TASKS = frozenset({
    UnifiedTask.VLM_ONLY,
    UnifiedTask.SEGMENT_ONLY,
    UnifiedTask.REGION_UNDERSTANDING,
    UnifiedTask.SEGMENT_AND_UNDERSTAND,
})


class DemoRunnerError(RuntimeError):
    """Demo preflight、输入 staging 或 suite artifact 合同失败。"""


class DemoCandidateKind(str, Enum):
    CANDIDATE = "CANDIDATE"
    EXPLICIT_GLOBAL = "EXPLICIT_GLOBAL"


@dataclass(frozen=True)
class DemoCandidateSelection:
    """绑定一次 spatial snapshot 的 Demo-only 区域选择。"""

    snapshot_id: str
    kind: DemoCandidateKind
    candidate_region_id: int | None = None

    def __post_init__(self) -> None:
        if not self.snapshot_id.startswith("dss_"):
            raise DemoRunnerError("candidate selection snapshot_id 非法")
        if self.kind is DemoCandidateKind.CANDIDATE:
            if (
                isinstance(self.candidate_region_id, bool)
                or not isinstance(self.candidate_region_id, int)
                or self.candidate_region_id < 0
            ):
                raise DemoRunnerError("candidate selection 要求精确非负 candidate ID")
        elif self.candidate_region_id is not None:
            raise DemoRunnerError("explicit global selection 禁止 candidate ID")

    @property
    def token(self) -> str:
        suffix = (
            "global"
            if self.kind is DemoCandidateKind.EXPLICIT_GLOBAL
            else f"candidate:{self.candidate_region_id}"
        )
        return f"{self.snapshot_id}:{suffix}"

    @classmethod
    def from_token(cls, value: str) -> "DemoCandidateSelection":
        if not isinstance(value, str) or not value:
            raise DemoRunnerError("candidate selection token 为空")
        parts = value.split(":")
        if len(parts) == 2 and parts[1] == "global":
            return cls(parts[0], DemoCandidateKind.EXPLICIT_GLOBAL)
        if len(parts) == 3 and parts[1] == "candidate" and parts[2].isdigit():
            return cls(parts[0], DemoCandidateKind.CANDIDATE, int(parts[2]))
        raise DemoRunnerError("candidate selection token 格式非法")


@dataclass(frozen=True)
class DemoSpatialSnapshot:
    """同一次已发布 spatial result 的精确 Demo replay 句柄。"""

    snapshot_id: str
    benchmark_identity: Mapping[str, Any]
    sample_id: str
    split: str
    source: str
    run_id: str
    source_task: UnifiedTask
    task_root: Path
    spatial_root: Path
    spatial_input: InMemorySpatialInput
    spatial_result: SpatialResult
    spatial_ledger: tuple[Mapping[str, Any], ...]
    candidate_previews: tuple[Mapping[str, Any], ...]
    global_mask_path: Path
    global_mask_sha256: str
    global_overlay_path: Path | None
    global_overlay_sha256: str | None
    optical_channel_previews: tuple[InputChannelPreview, ...]
    auxiliary_channel_previews: tuple[InputChannelPreview, ...]
    test_receipt_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": DEMO_SPATIAL_SNAPSHOT_SCHEMA,
            "snapshot_id": self.snapshot_id,
            "benchmark_identity": dict(self.benchmark_identity),
            "sample_id": self.sample_id,
            "split": self.split,
            "source": self.source,
            "run_id": self.run_id,
            "source_task": self.source_task.value,
            "task_root": str(self.task_root),
            "spatial_root": str(self.spatial_root),
            "candidate_count": len(self.spatial_result.candidates),
            "candidates": [dict(value) for value in self.candidate_previews],
            "global_mask_sha256": self.global_mask_sha256,
            "global_overlay_sha256": self.global_overlay_sha256,
            "spatial_ledger": [dict(value) for value in self.spatial_ledger],
            "test_receipt_id": self.test_receipt_id,
            "qualitative_demo_only": True,
            "formal_evaluation": False,
        }


@dataclass(frozen=True)
class DemoTaskResult:
    task: UnifiedTask
    status: str
    task_root: str | None
    viewer_path: str
    response: Mapping[str, Any] | None
    failure: Mapping[str, Any] | None
    pending: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task.value,
            "status": self.status,
            "task_root": self.task_root,
            "viewer_path": self.viewer_path,
            "response": None if self.response is None else dict(self.response),
            "failure": None if self.failure is None else dict(self.failure),
            "pending": None if self.pending is None else dict(self.pending),
        }


@dataclass(frozen=True)
class DemoRunSummary:
    run_id: str
    run_root: Path
    sample_id: str | None
    split: str | None
    source: str | None
    data_mode: str
    tasks: tuple[DemoTaskResult, ...]
    sealed_test_accessed: bool
    access_receipt_id: str | None
    input_scope: str
    benchmark_payload_consumed: bool
    benchmark_payload_loaded_this_run: bool
    active_snapshot_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "run_root": str(self.run_root),
            "sample_id": self.sample_id,
            "split": self.split,
            "source": self.source,
            "data_mode": self.data_mode,
            "tasks": [task.to_dict() for task in self.tasks],
            "sealed_test_accessed": self.sealed_test_accessed,
            "access_receipt_id": self.access_receipt_id,
            "input_scope": self.input_scope,
            "benchmark_payload_consumed": self.benchmark_payload_consumed,
            "benchmark_payload_loaded_this_run": self.benchmark_payload_loaded_this_run,
            "active_snapshot_id": self.active_snapshot_id,
            "engineering_demo": True,
            "formal_evaluation": False,
            "scientific_acceptance": False,
        }


@dataclass
class _ObservationState:
    spatial: SpatialResult | None = None
    spatial_replayed: bool = False
    evidence: GroundedEvidenceResult | None = None
    rag: TextRAGResult | None = None
    raw_visual: str | None = None
    raw_knowledge: str | None = None

    def reset(self) -> None:
        self.spatial = None
        self.spatial_replayed = False
        self.evidence = None
        self.rag = None
        self.raw_visual = None
        self.raw_knowledge = None


@dataclass(frozen=True)
class _SnapshotCapture:
    snapshot_id: str
    task: UnifiedTask
    task_relative: str
    spatial_input: InMemorySpatialInput
    spatial_result: SpatialResult
    spatial_ledger: tuple[Mapping[str, Any], ...]
    viewer: Mapping[str, Any]
    optical_channel_previews: tuple[InputChannelPreview, ...]
    auxiliary_channel_previews: tuple[InputChannelPreview, ...]
    test_receipt_id: str | None


class _ObservedSpatial:
    def __init__(self, target: Any, state: _ObservationState) -> None:
        self.target = target
        self.state = state
        self.replay: DemoSpatialSnapshot | None = None

    def use_replay(self, snapshot: DemoSpatialSnapshot | None) -> None:
        self.replay = snapshot

    @staticmethod
    def _copy_replay_assets(snapshot: DemoSpatialSnapshot, output_dir: Path) -> None:
        source = snapshot.spatial_root
        if (
            first_symlink_component(source) is not None
            or not source.is_dir()
            or source.is_symlink()
            or output_dir.exists()
            or output_dir.is_symlink()
        ):
            raise DemoRunnerError("spatial replay source/destination 合同非法")
        expected = {str(row["path"]): str(row["sha256"]) for row in snapshot.spatial_ledger}
        actual: dict[str, str] = {}
        for path in sorted(source.rglob("*")):
            if path.is_dir():
                if path.is_symlink():
                    raise DemoRunnerError("spatial replay source 含 symlink directory")
                continue
            if (
                not path.is_file()
                or path.is_symlink()
                or first_symlink_component(path) is not None
            ):
                raise DemoRunnerError("spatial replay source 含非法文件")
            relative = path.relative_to(source).as_posix()
            actual[relative] = sha256_file(path)
        if actual != expected:
            raise DemoRunnerError("spatial replay source SHA-256 漂移")
        output_dir.mkdir(parents=True)
        for relative, expected_sha in sorted(expected.items()):
            source_path = source / relative
            destination = output_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_path, destination)
            if sha256_file(destination) != expected_sha:
                raise DemoRunnerError("spatial replay 未保持逐字节一致")

    def infer(self, request: UnifiedRequest, *, output_dir: Path) -> SpatialResult:
        if self.replay is None:
            result = self.target.infer(request, output_dir=output_dir)
            self.state.spatial_replayed = False
        else:
            spatial_input = request.spatial_input
            if (
                spatial_input is None
                or spatial_input.sample_id != self.replay.sample_id
                or spatial_input.split != self.replay.split
                or spatial_input.source != self.replay.source
            ):
                raise DemoRunnerError("spatial replay 与 UnifiedRequest sample 身份不一致")
            self._copy_replay_assets(self.replay, output_dir)
            result = self.replay.spatial_result
            self.state.spatial_replayed = True
        self.state.spatial = result
        return result

    def release(self) -> None:
        self.target.release()


class _ObservedShared:
    def __init__(self, target: Any, state: _ObservationState) -> None:
        self.target = target
        self.state = state

    def describe(self, request: UnifiedRequest) -> str:
        return self.target.describe(request)

    def generate_visual(self, messages: Sequence[Mapping[str, Any]]) -> str:
        value = self.target.generate_visual(messages)
        self.state.raw_visual = value
        return value

    def generate_text(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        packet: Mapping[str, Any],
    ) -> str:
        value = self.target.generate_text(messages, packet=packet)
        self.state.raw_knowledge = value
        return value

    def release(self) -> None:
        self.target.release()

    def runtime_metadata(self) -> Mapping[str, Any]:
        return getattr(self.target, "runtime_metadata", lambda: {})()


class _ObservedEvidence:
    def __init__(self, target: Any, state: _ObservationState) -> None:
        self.target = target
        self.state = state

    def build(self, *args: Any, **kwargs: Any) -> GroundedEvidenceResult:
        result = self.target.build(*args, **kwargs)
        self.state.evidence = result
        return result

    def parse_observation(self, *args: Any, **kwargs: Any) -> Mapping[str, Any]:
        return self.target.parse_observation(*args, **kwargs)


class _ObservedRAG:
    def __init__(self, target: Any, state: _ObservationState) -> None:
        self.target = target
        self.state = state

    def retrieve(self, *args: Any, **kwargs: Any) -> TextRAGResult:
        result = self.target.retrieve(*args, **kwargs)
        self.state.rag = result
        return result

    def parse_generation(self, *args: Any, **kwargs: Any) -> Mapping[str, Any]:
        return self.target.parse_generation(*args, **kwargs)

    def release(self) -> None:
        self.target.release()


class UnifiedDemoRunner:
    """同一样本的 Single Task / fixed-order Task Suite 执行器。"""

    def __init__(
        self,
        config: DemoConfig,
        catalog: BenchmarkCatalog,
        *,
        runtime: UnifiedInferenceRuntime | None = None,
    ) -> None:
        self.config = config
        self.catalog = catalog
        base_runtime = runtime or build_unified_runtime(config.unified)
        self._observation = _ObservationState()
        self._observed_spatial = _ObservedSpatial(
            base_runtime.spatial,
            self._observation,
        )
        self.runtime = UnifiedInferenceRuntime(
            spatial=self._observed_spatial,
            shared_mllm=_ObservedShared(base_runtime.shared_mllm, self._observation),
            evidence=_ObservedEvidence(base_runtime.evidence, self._observation),
            text_rag=_ObservedRAG(base_runtime.text_rag, self._observation),
            router=base_runtime.router,
        )
        self.spatial_config = load_runtime_config(config.unified.spatial.config_path)
        self._active_snapshot: DemoSpatialSnapshot | None = None

    @staticmethod
    def canonical_tasks(tasks: Sequence[UnifiedTask | str]) -> tuple[UnifiedTask, ...]:
        try:
            selected = tuple(
                value if isinstance(value, UnifiedTask) else UnifiedTask(value)
                for value in tasks
            )
        except (TypeError, ValueError) as error:
            raise DemoRunnerError("Task Suite 含未知 UnifiedTask") from error
        if not selected or len(selected) != len(set(selected)):
            raise DemoRunnerError("Task Suite 必须非空且不重复")
        return tuple(task for task in TASK_ORDER if task in selected)

    def candidate_choices(self, sample_id: str, *, split: str) -> tuple[int, ...]:
        """仅返回最近一次同 sample spatial result 的精确 ID。"""

        snapshot = self._active_snapshot
        if snapshot is None or (snapshot.split, snapshot.sample_id) != (split, sample_id):
            return ()
        return tuple(candidate.region_id for candidate in snapshot.spatial_result.candidates)

    @property
    def active_snapshot(self) -> DemoSpatialSnapshot | None:
        return self._active_snapshot

    def clear_spatial_snapshot(self) -> None:
        """切换样本或开始新 spatial run 时使旧 candidate token 立即失效。"""

        self._active_snapshot = None

    def resolve_candidate_selection(
        self,
        token: str,
        *,
        sample_id: str,
        split: str,
    ) -> DemoCandidateSelection:
        selection = DemoCandidateSelection.from_token(token)
        snapshot = self._active_snapshot
        if (
            snapshot is None
            or selection.snapshot_id != snapshot.snapshot_id
            or snapshot.sample_id != sample_id
            or snapshot.split != split
            or dict(snapshot.benchmark_identity) != dict(self.catalog.identity)
        ):
            raise DemoRunnerError("candidate selection 不属于当前 sample 的有效 spatial result")
        if selection.kind is DemoCandidateKind.CANDIDATE and not any(
            candidate.region_id == selection.candidate_region_id
            for candidate in snapshot.spatial_result.candidates
        ):
            raise DemoRunnerError("candidate selection ID 不属于当前 spatial result")
        return selection

    def candidate_ui_payload(
        self,
        *,
        sample_id: str,
        split: str,
    ) -> dict[str, Any]:
        snapshot = self._active_snapshot
        if snapshot is None or (snapshot.split, snapshot.sample_id) != (split, sample_id):
            return {
                "snapshot": None,
                "options": [],
                "gallery_tokens": [],
            }
        previews = sorted(
            (dict(value) for value in snapshot.candidate_previews),
            key=lambda value: int(value["candidate_id"]),
        )
        options: list[dict[str, Any]] = []
        gallery_tokens: list[str] = []
        for preview in previews:
            selection = DemoCandidateSelection(
                snapshot.snapshot_id,
                DemoCandidateKind.CANDIDATE,
                int(preview["candidate_id"]),
            )
            options.append({
                "kind": DemoCandidateKind.CANDIDATE.value,
                "token": selection.token,
                "candidate_id": int(preview["candidate_id"]),
                "bbox_xyxy": list(preview["bbox_xyxy"]),
                "area_pixels": int(preview["area_pixels"]),
                "confidence": float(preview["confidence"]),
                "mask_path": str(preview["mask_path"]),
                "overlay_path": str(preview["overlay_path"]),
            })
            gallery_tokens.append(selection.token)
        global_selection = DemoCandidateSelection(
            snapshot.snapshot_id,
            DemoCandidateKind.EXPLICIT_GLOBAL,
        )
        options.append({
            "kind": DemoCandidateKind.EXPLICIT_GLOBAL.value,
            "token": global_selection.token,
            "candidate_id": None,
            "bbox_xyxy": None,
            "area_pixels": None,
            "confidence": None,
            "mask_path": str(snapshot.global_mask_path),
            "overlay_path": (
                None
                if snapshot.global_overlay_path is None
                else str(snapshot.global_overlay_path)
            ),
        })
        return {
            "snapshot": snapshot.to_dict(),
            "options": options,
            "gallery_tokens": gallery_tokens,
        }

    def candidate_preview(
        self,
        selection: DemoCandidateSelection,
        *,
        sample_id: str,
        split: str,
    ) -> tuple[str | None, str | None, Mapping[str, Any]]:
        validated = self.resolve_candidate_selection(
            selection.token,
            sample_id=sample_id,
            split=split,
        )
        assert self._active_snapshot is not None
        snapshot = self._active_snapshot
        if validated.kind is DemoCandidateKind.EXPLICIT_GLOBAL:
            if sha256_file(snapshot.global_mask_path) != snapshot.global_mask_sha256:
                raise DemoRunnerError("global mask preview SHA-256 漂移")
            if (
                snapshot.global_overlay_path is not None
                and snapshot.global_overlay_sha256 is not None
                and sha256_file(snapshot.global_overlay_path)
                != snapshot.global_overlay_sha256
            ):
                raise DemoRunnerError("global overlay preview SHA-256 漂移")
            return (
                str(snapshot.global_mask_path),
                None if snapshot.global_overlay_path is None else str(snapshot.global_overlay_path),
                {
                    "snapshot_id": snapshot.snapshot_id,
                    "sample_id": snapshot.sample_id,
                    "requested_source": RegionSource.OA_AUXSEG_CANDIDATE.value,
                    "selection_intent": DemoCandidateKind.EXPLICIT_GLOBAL.value,
                    "candidate_id": None,
                    "explicit_global_confirmed": True,
                },
            )
        preview = next(
            value
            for value in snapshot.candidate_previews
            if int(value["candidate_id"]) == validated.candidate_region_id
        )
        if (
            sha256_file(Path(str(preview["mask_path"]))) != preview["mask_sha256"]
            or sha256_file(Path(str(preview["overlay_path"])))
            != preview["overlay_sha256"]
        ):
            raise DemoRunnerError("candidate preview SHA-256 漂移")
        return (
            str(preview["mask_path"]),
            str(preview["overlay_path"]),
            {
                key: value
                for key, value in preview.items()
                if key not in {"mask_path", "overlay_path"}
            },
        )

    @staticmethod
    def validate_and_stage_user_mask(
        source: Path | str,
        destination: Path,
        *,
        expected_size: tuple[int, int],
    ) -> dict[str, Any]:
        """严格验证 PNG-L/0-255/尺寸/非空后逐字节复制，不转换、不修复。"""

        path = Path(os.path.abspath(Path(source)))
        if (
            first_symlink_component(path) is not None
            or not path.is_file()
            or path.is_symlink()
            or path.stat().st_nlink != 1
        ):
            raise DemoRunnerError(f"user mask 必须是普通单链接文件：{path}")
        if path.suffix.lower() != ".png":
            raise DemoRunnerError("user mask 必须使用 .png 后缀")
        payload = path.read_bytes()
        if not payload or len(payload) > 64 * 1024 * 1024:
            raise DemoRunnerError("user mask 文件为空或超过 64 MiB")
        try:
            with Image.open(BytesIO(payload)) as probe:
                probe.verify()
            with Image.open(BytesIO(payload)) as image:
                if image.format != "PNG" or image.mode != "L":
                    raise DemoRunnerError("user mask 必须是 PNG-L，禁止隐式模式转换")
                if image.size != expected_size:
                    raise DemoRunnerError(
                        f"user mask 尺寸必须为 {expected_size}，实际 {image.size}"
                    )
                image.load()
                values = np.asarray(image, dtype=np.uint8)
        except DemoRunnerError:
            raise
        except Exception as error:
            raise DemoRunnerError("user mask 不是严格合法 PNG") from error
        unique = set(np.unique(values).tolist())
        if not unique.issubset({0, 255}) or 255 not in unique:
            raise DemoRunnerError("user mask 只能含 0/255 且必须非空")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() or destination.is_symlink():
            raise DemoRunnerError(f"user mask staging 目标已存在：{destination}")
        destination.write_bytes(payload)
        if destination.read_bytes() != payload:
            raise DemoRunnerError("user mask staging 未保持逐字节一致")
        return {
            "source_size_bytes": len(payload),
            "source_sha256": sha256_bytes(payload),
            "staged_sha256": sha256_file(destination),
            "image_mode": "L",
            "pixel_values": sorted(unique),
            "nonempty": True,
            "byte_preserving_copy": True,
        }

    def _spatial_input(self, loaded: LoadedBenchmarkSample) -> InMemorySpatialInput:
        prepared = prepare_collated_batch(
            collate_benchmark_samples([loaded.model_sample])
        )
        kwargs = {
            "batch": prepared.model,
            "optical_image": loaded.optical_image,
            "sample_id": loaded.record.sample_id,
            "source": loaded.record.source,
            "split": loaded.record.split,
        }
        if loaded.record.split == "test":
            if loaded.test_receipt is None:
                raise DemoRunnerError("test inference 缺少先行 receipt")
            return DemoAuthorizedSpatialInput(
                **kwargs,
                receipt_id=loaded.test_receipt.receipt_id,
            )
        return InMemorySpatialInput(**kwargs)

    @staticmethod
    def _replay_spatial_input(
        snapshot: DemoSpatialSnapshot,
        *,
        receipt_id: str | None,
    ) -> InMemorySpatialInput:
        value = snapshot.spatial_input
        kwargs = {
            "batch": value.batch,
            "optical_image": value.optical_image,
            "sample_id": snapshot.sample_id,
            "source": snapshot.source,
            "split": snapshot.split,
        }
        if snapshot.split == "test":
            if receipt_id is None:
                raise DemoRunnerError("test spatial replay 缺少新 INFERENCE receipt")
            return DemoAuthorizedSpatialInput(**kwargs, receipt_id=receipt_id)
        return InMemorySpatialInput(**kwargs)

    @staticmethod
    def _stage_channel_previews(
        writer: AtomicArtifactDirectory,
        *,
        optical: Sequence[InputChannelPreview],
        auxiliary: Sequence[InputChannelPreview],
    ) -> dict[str, list[dict[str, Any]]]:
        assert writer.staging is not None
        result: dict[str, list[dict[str, Any]]] = {"optical": [], "auxiliary": []}
        for group, previews in (("optical", optical), ("auxiliary", auxiliary)):
            for ordinal, preview in enumerate(previews):
                digest = sha256_text(
                    f"{preview.modality}\0{preview.channel_index}\0{preview.channel_name}"
                )[:16]
                relative = f"inputs/channel_previews/{group}/{ordinal:03d}_{digest}.png"
                path = writer.path(relative)
                path.parent.mkdir(parents=True, exist_ok=True)
                preview.image.save(path, format="PNG", optimize=False)
                result[group].append({
                    **preview.to_dict(),
                    "asset": relative,
                    "caption": preview.caption,
                })
        return result

    @staticmethod
    def _spatial_ledger(root: Path) -> tuple[Mapping[str, Any], ...]:
        rows: list[Mapping[str, Any]] = []
        for path in sorted(root.rglob("*")):
            if path.is_dir():
                if path.is_symlink():
                    raise DemoRunnerError("spatial output 含 symlink directory")
                continue
            if (
                not path.is_file()
                or path.is_symlink()
                or first_symlink_component(path) is not None
            ):
                raise DemoRunnerError("spatial output 含非法文件")
            rows.append({
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
        if not rows:
            raise DemoRunnerError("spatial output 为空，不能创建 candidate snapshot")
        return tuple(rows)

    def _build_request(
        self,
        *,
        task: UnifiedTask,
        request_id: str,
        instruction: str,
        optical_path: Path | None,
        user_mask_path: Path | None,
        spatial_input: InMemorySpatialInput | None,
        candidate_region_id: int | None,
        region_interpretation_source: RegionSource,
    ) -> UnifiedRequest:
        if task is UnifiedTask.VLM_ONLY:
            if optical_path is None:
                raise DemoRunnerError("VLM_ONLY 缺少 optical input")
            return UnifiedRequest(
                task=task,
                request_id=request_id,
                instruction=instruction,
                images=(optical_path,),
                include_audit=True,
            )
        if task is UnifiedTask.SEGMENT_ONLY:
            if spatial_input is None:
                raise DemoRunnerError("SEGMENT_ONLY 缺少 spatial input")
            return UnifiedRequest(
                task=task,
                request_id=request_id,
                spatial_input=spatial_input,
                include_audit=True,
            )
        if task is UnifiedTask.REGION_UNDERSTANDING:
            if optical_path is None or user_mask_path is None:
                raise DemoRunnerError(
                    "REGION_UNDERSTANDING 必须上传独立 user/demo mask；Reference/GT 禁止代用"
                )
            return UnifiedRequest(
                task=task,
                request_id=request_id,
                instruction=instruction,
                images=(optical_path,),
                user_mask=user_mask_path,
                region_source=RegionSource.USER_MASK,
                include_audit=True,
            )
        if task is UnifiedTask.SEGMENT_AND_UNDERSTAND:
            if spatial_input is None:
                raise DemoRunnerError("SEGMENT_AND_UNDERSTAND 缺少 spatial input")
            return UnifiedRequest(
                task=task,
                request_id=request_id,
                instruction=instruction,
                spatial_input=spatial_input,
                region_source=RegionSource.OA_AUXSEG_GLOBAL,
                include_audit=True,
            )
        if task is UnifiedTask.KNOWLEDGE_QA:
            return UnifiedRequest(
                task=task,
                request_id=request_id,
                instruction=instruction,
                include_audit=True,
            )
        if task is UnifiedTask.REGION_INTERPRETATION:
            if region_interpretation_source is RegionSource.USER_MASK:
                if optical_path is None or user_mask_path is None:
                    raise DemoRunnerError("USER_MASK interpretation 要求独立上传 mask")
                return UnifiedRequest(
                    task=task,
                    request_id=request_id,
                    instruction=instruction,
                    images=(optical_path,),
                    user_mask=user_mask_path,
                    region_source=RegionSource.USER_MASK,
                    include_audit=True,
                )
            if region_interpretation_source is not RegionSource.OA_AUXSEG_CANDIDATE:
                raise DemoRunnerError("REGION_INTERPRETATION 只允许 USER_MASK/OA_AUXSEG_CANDIDATE")
            if spatial_input is None:
                raise DemoRunnerError("candidate interpretation 缺少有效 spatial snapshot")
            return UnifiedRequest(
                task=task,
                request_id=request_id,
                instruction=instruction,
                spatial_input=spatial_input,
                region_source=RegionSource.OA_AUXSEG_CANDIDATE,
                candidate_region_id=candidate_region_id,
                include_audit=True,
            )
        raise AssertionError(task)

    @staticmethod
    def _ledger_rows(root: Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(root).as_posix()
            if relative in {"manifest.json", "SHA256SUMS.jsonl"}:
                continue
            rows.append({
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
        return rows

    @staticmethod
    def _relative(root: Path, path: Path | None) -> str | None:
        if path is None:
            return None
        return path.relative_to(root).as_posix()

    def _viewer(
        self,
        *,
        writer: AtomicArtifactDirectory,
        task: UnifiedTask,
        request: UnifiedRequest | None,
        task_relative: str | None,
        optical_path: Path | None,
        input_preview_assets: Mapping[str, Sequence[Mapping[str, Any]]],
        response: UnifiedResponse | None,
        failure: Mapping[str, Any] | None,
        status: str,
        pending: Mapping[str, Any] | None,
        candidate_selection: DemoCandidateSelection | None,
        benchmark_payload_consumed_by_task: bool,
        effective_instruction: str,
        instruction_source: str,
    ) -> tuple[str, dict[str, Any]]:
        assert writer.staging is not None
        run_staging = writer.staging
        asset_root = writer.path(f"viewer_assets/{task.value.lower()}")
        asset_root.mkdir(parents=True, exist_ok=True)
        spatial = self._observation.spatial
        evidence = self._observation.evidence
        rag = self._observation.rag
        overlay_reference: str | None = None
        probability_reference: str | None = None
        global_mask_reference: str | None = None
        candidate_previews: list[dict[str, Any]] = []
        if spatial is not None:
            optical = render_evidence_image(spatial.optical_image)
            mask = np.asarray(spatial.global_mask, dtype=bool)
            global_mask_path = asset_root / "global_mask.png"
            Image.fromarray(mask.astype(np.uint8) * 255).save(
                global_mask_path,
                format="PNG",
            )
            global_mask_reference = self._relative(run_staging, global_mask_path)
            overlay_path = asset_root / "predicted_overlay.png"
            render_mask_overlay(optical, mask).save(overlay_path, format="PNG")
            overlay_reference = self._relative(run_staging, overlay_path)
            probability = np.asarray(spatial.mask_probability, dtype=np.float32)
            if probability.ndim == 2 and np.isfinite(probability).all():
                probability_path = asset_root / "mask_probability.png"
                Image.fromarray(
                    (np.clip(probability, 0.0, 1.0) * 255.0).astype(np.uint8),
                ).save(probability_path, format="PNG")
                probability_reference = self._relative(run_staging, probability_path)
            candidate_root = asset_root / "candidates"
            for candidate in sorted(spatial.candidates, key=lambda value: value.region_id):
                candidate_mask = np.asarray(candidate.mask, dtype=bool)
                if candidate_mask.shape != mask.shape:
                    raise DemoRunnerError("SpatialCandidate mask 与 global mask 尺寸不一致")
                candidate_root.mkdir(parents=True, exist_ok=True)
                mask_path = (
                    None
                    if task_relative is None or candidate.mask_reference is None
                    else writer.path(f"{task_relative}/{candidate.mask_reference}")
                )
                if mask_path is None or not mask_path.is_file():
                    mask_path = candidate_root / f"region_{candidate.region_id}_mask.png"
                    Image.fromarray(candidate_mask.astype(np.uint8) * 255).save(
                        mask_path,
                        format="PNG",
                    )
                overlay_path = candidate_root / f"region_{candidate.region_id}_overlay.png"
                render_mask_overlay(optical, candidate_mask).save(
                    overlay_path,
                    format="PNG",
                )
                candidate_previews.append({
                    "candidate_id": candidate.region_id,
                    "bbox_xyxy": list(candidate.bbox_xyxy),
                    "centroid_xy": list(candidate.centroid_xy),
                    "area_pixels": candidate.area_pixels,
                    "confidence": candidate.confidence,
                    "mask_path": self._relative(run_staging, mask_path),
                    "overlay_path": self._relative(run_staging, overlay_path),
                    "mask_sha256": sha256_file(mask_path),
                    "overlay_sha256": sha256_file(overlay_path),
                    "source": "OA_AUXSEG_SPATIAL_RESULT",
                    "reference_or_gt_mask": False,
                })
        grounded_assets: dict[str, str | None] = {}
        for role, name in (
            ("full_optical", "optical_full.png"),
            ("binary_mask", "binary_mask.png"),
            ("context_crop", "context_crop.png"),
        ):
            path = None if task_relative is None else writer.path(
                f"{task_relative}/grounded/{name}"
            )
            grounded_assets[role] = (
                self._relative(run_staging, path)
                if path is not None and path.is_file()
                else None
            )
        response_row = None if response is None else response.to_dict()
        trace = (
            list((failure or {}).get("completed_trace", []))
            if response_row is None
            else response_row.get("trace", [])
        )
        provider_order = [
            {"provider": row.get("provider"), "operation": row.get("operation")}
            for row in trace
            if row.get("event") == "provider_call_started"
        ]
        cuda_peaks = [
            int(row.get("cuda_peak_allocated_bytes", 0))
            for row in trace
            if row.get("event") == "provider_call_completed"
        ]
        packet = None if rag is None else dict(rag.packet)
        if request is None:
            execution_plan = {
                "needs_spatial": True,
                "needs_shared_mllm": True,
                "needs_region": True,
                "needs_rag": True,
                "region_source": RegionSource.OA_AUXSEG_CANDIDATE.value,
                "response_kind": "REGION_INTERPRETATION",
            }
        else:
            execution_plan = (
                self.runtime.router.route(request).to_dict()
                if response is None or response.execution_plan is None
                else response.execution_plan.to_dict()
            )
        needs_spatial = bool(execution_plan["needs_spatial"])
        direct_visual = task is UnifiedTask.VLM_ONLY
        optical_reference = self._relative(run_staging, optical_path)
        if task is UnifiedTask.KNOWLEDGE_QA:
            optical_reference = None
        candidate_decision: dict[str, Any]
        if pending is not None:
            candidate_decision = {
                "selection_intent": "PENDING_USER_SELECTION",
                "requested_source": RegionSource.OA_AUXSEG_CANDIDATE.value,
                "effective_source": None,
                "fallback_reason": None,
                "runtime_invoked": False,
                **dict(pending),
            }
        elif candidate_selection is not None:
            candidate_decision = {
                "selection_intent": candidate_selection.kind.value,
                "snapshot_id": candidate_selection.snapshot_id,
                "candidate_id": candidate_selection.candidate_region_id,
                "explicit_global_confirmed": (
                    candidate_selection.kind is DemoCandidateKind.EXPLICIT_GLOBAL
                ),
                "requested_source": RegionSource.OA_AUXSEG_CANDIDATE.value,
                "effective_source": (
                    None
                    if response is None or response.region_selection is None
                    else response.region_selection.effective_source.value
                ),
                "fallback_reason": (
                    None
                    if response is None
                    or response.region_selection is None
                    or response.region_selection.status.value != "FALLBACK_GLOBAL"
                    else response.region_selection.reason.value
                ),
                "runtime_invoked": True,
            }
        else:
            candidate_decision = {
                "selection_intent": None,
                "runtime_invoked": response is not None,
            }
        viewer = {
            "schema_version": DEMO_VIEWER_SCHEMA,
            "task": task.value,
            "status": status,
            "request_preview": {
                "effective_instruction": effective_instruction,
                "instruction_source": instruction_source,
                "benchmark_payload_consumed_by_task": benchmark_payload_consumed_by_task,
            },
            "input_preview": {
                "spatial_expert_inputs": {
                    "full_optical": optical_reference if needs_spatial else None,
                    "active_modalities": [] if spatial is None else list(spatial.active_modalities),
                    "optical_channel_previews": (
                        list(input_preview_assets.get("optical", ()))
                        if needs_spatial and benchmark_payload_consumed_by_task
                        else []
                    ),
                    "auxiliary_channel_previews": (
                        list(input_preview_assets.get("auxiliary", ()))
                        if needs_spatial and benchmark_payload_consumed_by_task
                        else []
                    ),
                    "auxiliary_preview_is_model_input": False,
                },
                "direct_mllm_visual_inputs": {
                    "full_optical": optical_reference if direct_visual else None,
                },
                "mllm_formal_grounded_inputs": {
                    **grounded_assets,
                    "formal_roles": [] if evidence is None else list(
                        evidence.metadata.get("formal_model_input_roles", [])
                    ),
                    "auxiliary_modalities_formally_consumed": False,
                },
                "reference_or_gt_mask_consumed": False,
                "benchmark_payload_consumed_by_task": benchmark_payload_consumed_by_task,
            },
            "spatial_result": {
                "predicted_mask": (
                    global_mask_reference
                    if response is None or response.mask_reference is None
                    else (
                        f"{task_relative}/{response.mask_reference}"
                        if task_relative is not None
                        and not Path(response.mask_reference).is_absolute()
                        else response.mask_reference
                    )
                ),
                "mask_probability": probability_reference,
                "overlay": overlay_reference,
                "no_target": spatial.no_target if response is None and spatial is not None else (
                    None if response is None else response.no_target
                ),
                "no_target_score": (
                    spatial.no_target_score
                    if response is None and spatial is not None
                    else (None if response is None else response.no_target_score)
                ),
                "candidate_count": len(candidate_previews),
                "candidates": (
                    [candidate.public_dict() for candidate in spatial.candidates]
                    if spatial is not None
                    else []
                ),
                "candidate_previews": candidate_previews,
            },
            "grounded_understanding": {
                "programmatic_facts": None if evidence is None else dict(evidence.program_facts),
                "pass1_structured_observation": None if response is None else response.region_observation,
                "raw_pass1": self._observation.raw_visual,
                "limitations": [] if response is None else list(response.limitations),
            },
            "knowledge_augmentation": {
                "evidence_packet": packet,
                "pass2_interpretation": None if response is None else response.knowledge_output,
                "raw_pass2": self._observation.raw_knowledge,
                "citations": [] if response is None else [dict(value) for value in response.citations],
            },
            "execution_trace": {
                "execution_plan": execution_plan,
                "provider_call_order": provider_order,
                "candidate_fallback": None if response is None or response.region_selection is None else response.region_selection.to_dict(),
                "candidate_decision": candidate_decision,
                "spatial_result_replayed": self._observation.spatial_replayed,
                "cuda_peak_allocated_bytes": max(cuda_peaks, default=0),
                "limitations": [] if response is None else list(response.limitations),
                "trace": trace,
            },
            "response": response_row,
            "failure": None if failure is None else dict(failure),
            "pending": None if pending is None else dict(pending),
            "qualitative_demo_only": True,
            "formal_acceptance": False,
            "scientific_acceptance": False,
        }
        relative = f"viewer/{task.value.lower()}.json"
        writer.write_json(relative, viewer)
        return relative, viewer

    def _snapshot_from_capture(
        self,
        *,
        target: Path,
        run_id: str,
        capture: _SnapshotCapture,
        sample_id: str,
        split: str,
        source: str,
    ) -> DemoSpatialSnapshot:
        task_root = target / capture.task_relative
        spatial_root = task_root / "spatial"
        if not spatial_root.is_dir() or spatial_root.is_symlink():
            raise DemoRunnerError("已发布 spatial snapshot root 不存在或非法")
        candidate_previews: list[Mapping[str, Any]] = []
        for value in capture.viewer["spatial_result"]["candidate_previews"]:
            preview = dict(value)
            for key in ("mask_path", "overlay_path"):
                path = target / str(preview[key])
                if (
                    first_symlink_component(path) is not None
                    or not path.is_file()
                    or path.is_symlink()
                ):
                    raise DemoRunnerError("已发布 candidate preview 资产非法")
                expected_sha = str(preview[f"{key.removesuffix('_path')}_sha256"])
                if sha256_file(path) != expected_sha:
                    raise DemoRunnerError("已发布 candidate preview SHA-256 漂移")
                preview[key] = str(path)
            preview["snapshot_id"] = capture.snapshot_id
            preview["run_id"] = run_id
            preview["sample_id"] = sample_id
            preview["split"] = split
            preview["source_task"] = capture.task.value
            candidate_previews.append(preview)
        global_mask_path = spatial_root / "global_mask.png"
        spatial_hashes = {
            str(value["path"]): str(value["sha256"])
            for value in capture.spatial_ledger
        }
        global_mask_sha256 = spatial_hashes.get("global_mask.png")
        if global_mask_sha256 is None:
            raise DemoRunnerError("spatial snapshot ledger 缺少 global_mask.png")
        overlay_value = capture.viewer["spatial_result"].get("overlay")
        global_overlay_path = None if overlay_value is None else target / str(overlay_value)
        for path in (global_mask_path, global_overlay_path):
            if path is None:
                continue
            if (
                first_symlink_component(path) is not None
                or not path.is_file()
                or path.is_symlink()
            ):
                raise DemoRunnerError("已发布 global spatial preview 资产非法")
        global_overlay_sha256 = (
            None
            if global_overlay_path is None
            else sha256_file(global_overlay_path)
        )
        return DemoSpatialSnapshot(
            snapshot_id=capture.snapshot_id,
            benchmark_identity=self.catalog.identity,
            sample_id=sample_id,
            split=split,
            source=source,
            run_id=run_id,
            source_task=capture.task,
            task_root=task_root,
            spatial_root=spatial_root,
            spatial_input=capture.spatial_input,
            spatial_result=capture.spatial_result,
            spatial_ledger=capture.spatial_ledger,
            candidate_previews=tuple(candidate_previews),
            global_mask_path=global_mask_path,
            global_mask_sha256=global_mask_sha256,
            global_overlay_path=global_overlay_path,
            global_overlay_sha256=global_overlay_sha256,
            optical_channel_previews=capture.optical_channel_previews,
            auxiliary_channel_previews=capture.auxiliary_channel_previews,
            test_receipt_id=capture.test_receipt_id,
        )

    def run(
        self,
        *,
        record: BenchmarkRecord | None,
        tasks: Sequence[UnifiedTask | str],
        instructions: Mapping[UnifiedTask | str, str] | None = None,
        user_mask: Path | str | None = None,
        candidate_selection: DemoCandidateSelection | None = None,
        region_interpretation_source: RegionSource = RegionSource.OA_AUXSEG_CANDIDATE,
        data_mode: str = "Demo / Qualitative Exploration",
    ) -> DemoRunSummary:
        selected = self.canonical_tasks(tasks)
        if data_mode != "Demo / Qualitative Exploration":
            raise DemoRunnerError(
                "Unified Demo 只接受 Demo / Qualitative Exploration；"
                f"旧 Frozen data_mode 已退役：{data_mode}"
            )
        pure_knowledge = selected == (UnifiedTask.KNOWLEDGE_QA,)
        has_fresh_spatial = any(task in FRESH_SPATIAL_TASKS for task in selected)
        has_candidate_interpretation = (
            UnifiedTask.REGION_INTERPRETATION in selected
            and region_interpretation_source is RegionSource.OA_AUXSEG_CANDIDATE
        )
        candidate_pending = has_candidate_interpretation and (
            has_fresh_spatial or candidate_selection is None
        )
        needs_payload = any(task in PAYLOAD_TASKS for task in selected) or (
            UnifiedTask.REGION_INTERPRETATION in selected
            and region_interpretation_source is RegionSource.USER_MASK
        )
        replay_snapshot: DemoSpatialSnapshot | None = None
        if has_fresh_spatial:
            self.clear_spatial_snapshot()
        elif has_candidate_interpretation and candidate_selection is not None:
            snapshot = self._active_snapshot
            if snapshot is None:
                raise DemoRunnerError("candidate interpretation 前必须先运行 Spatial Task")
            sample_id = snapshot.sample_id if record is None else record.sample_id
            split = snapshot.split if record is None else record.split
            candidate_selection = self.resolve_candidate_selection(
                candidate_selection.token,
                sample_id=sample_id,
                split=split,
            )
            replay_snapshot = snapshot
        elif candidate_selection is not None and not pure_knowledge:
            raise DemoRunnerError("candidate selection 只能用于 OA_AUXSEG_CANDIDATE interpretation")

        consumes_visual_payload = needs_payload or replay_snapshot is not None
        if needs_payload and record is None:
            raise DemoRunnerError("视觉任务必须绑定当前 Benchmark Browser sample")
        if replay_snapshot is not None:
            if record is None:
                raise DemoRunnerError("candidate replay 必须绑定当前 selection metadata")
            if (
                record.sample_id != replay_snapshot.sample_id
                or record.split != replay_snapshot.split
                or record.source != replay_snapshot.source
            ):
                raise DemoRunnerError("candidate replay 与当前 sample identity 不一致")
        loaded: LoadedBenchmarkSample | None = None
        replay_receipt = None
        if needs_payload:
            assert record is not None
            loaded = self.catalog.load(
                record,
                action="INFERENCE",
                model_normalization=self.spatial_config.normalization,
            )
        elif replay_snapshot is not None and replay_snapshot.split == "test":
            controller = self.catalog.access_controller
            if controller is None:
                raise DemoRunnerError("test replay 缺少 Demo access controller")
            replay_receipt = controller.issue(
                sample_id=replay_snapshot.sample_id,
                action="INFERENCE",
            )

        run_id = (
            "demo_"
            + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            + "_"
            + uuid4().hex
        )
        target = self.config.demo_root / "runs" / run_id
        task_results: list[DemoTaskResult] = []
        snapshot_capture: _SnapshotCapture | None = None
        prompt_overrides: dict[UnifiedTask, str] = {}
        for key, value in (instructions or {}).items():
            task = key if isinstance(key, UnifiedTask) else UnifiedTask(key)
            if task not in selected:
                raise DemoRunnerError(f"{task.value} instruction override 不属于当前 task set")
            if not isinstance(value, str) or not value.strip():
                raise DemoRunnerError(f"{task.value} instruction 必须是非空字符串")
            prompt_overrides[task] = value.strip()

        context_record = record if consumes_visual_payload else None
        summary_data_mode = KNOWLEDGE_ONLY_MODE if pure_knowledge else data_mode
        input_scope = (
            "KNOWLEDGE_ONLY"
            if pure_knowledge
            else (
                "SPATIAL_REPLAY"
                if replay_snapshot is not None and not needs_payload
                else ("BENCHMARK_VISUAL" if needs_payload else "PENDING_SPATIAL_SELECTION")
            )
        )
        receipt = (
            loaded.test_receipt
            if loaded is not None
            else replay_receipt
        )
        sealed_test_accessed = bool(
            consumes_visual_payload
            and context_record is not None
            and context_record.split == "test"
        )

        with AtomicArtifactDirectory(target) as writer:
            optical_path: Path | None = None
            optical_previews: tuple[InputChannelPreview, ...] = ()
            auxiliary_previews: tuple[InputChannelPreview, ...] = ()
            if loaded is not None:
                optical_image = loaded.optical_image
                optical_previews = loaded.optical_channel_previews
                auxiliary_previews = loaded.auxiliary_channel_previews
            elif replay_snapshot is not None:
                optical_image = render_evidence_image(
                    replay_snapshot.spatial_result.optical_image
                )
                optical_previews = replay_snapshot.optical_channel_previews
                auxiliary_previews = replay_snapshot.auxiliary_channel_previews
            else:
                optical_image = None
            if optical_image is not None:
                optical_path = writer.path("inputs/optical_full.png")
                optical_path.parent.mkdir(parents=True, exist_ok=True)
                optical_image.save(optical_path, format="PNG", optimize=False)
                input_preview_assets = self._stage_channel_previews(
                    writer,
                    optical=optical_previews,
                    auxiliary=auxiliary_previews,
                )
            else:
                input_preview_assets = {"optical": [], "auxiliary": []}

            staged_mask: Path | None = None
            mask_audit: Mapping[str, Any] | None = None
            needs_user_mask = (
                UnifiedTask.REGION_UNDERSTANDING in selected
                or (
                    UnifiedTask.REGION_INTERPRETATION in selected
                    and region_interpretation_source is RegionSource.USER_MASK
                )
            )
            if needs_user_mask and user_mask is not None:
                if optical_image is None:
                    raise DemoRunnerError("user mask task 缺少 optical image")
                staged_mask = writer.path("inputs/user_mask.png")
                mask_audit = self.validate_and_stage_user_mask(
                    user_mask,
                    staged_mask,
                    expected_size=optical_image.size,
                )

            spatial_input: InMemorySpatialInput | None = None
            if has_fresh_spatial:
                if loaded is None:
                    raise DemoRunnerError("fresh spatial task 缺少 Benchmark payload")
                spatial_input = self._spatial_input(loaded)
            elif replay_snapshot is not None:
                spatial_input = self._replay_spatial_input(
                    replay_snapshot,
                    receipt_id=None if receipt is None else receipt.receipt_id,
                )

            requests: dict[UnifiedTask, tuple[UnifiedRequest, str, str, str]] = {}
            for ordinal, task in enumerate(selected):
                instruction = prompt_overrides.get(task, self.config.defaults.prompts[task])
                instruction_source = (
                    "USER_OVERRIDE" if task in prompt_overrides else "CONFIG_DEFAULT"
                )
                if task is UnifiedTask.REGION_INTERPRETATION and candidate_pending:
                    continue
                candidate_id = (
                    candidate_selection.candidate_region_id
                    if task is UnifiedTask.REGION_INTERPRETATION
                    and candidate_selection is not None
                    and candidate_selection.kind is DemoCandidateKind.CANDIDATE
                    else None
                )
                request = self._build_request(
                    task=task,
                    request_id=f"{run_id}_{task.value.lower()}",
                    instruction=instruction,
                    optical_path=optical_path,
                    user_mask_path=staged_mask,
                    spatial_input=spatial_input,
                    candidate_region_id=candidate_id,
                    region_interpretation_source=region_interpretation_source,
                )
                task_relative = f"tasks/{ordinal:02d}_{task.value.lower()}"
                requests[task] = (
                    request,
                    task_relative,
                    instruction,
                    instruction_source,
                )
            access_context = (
                None
                if receipt is None
                else DemoInferenceAccess(
                    receipt=receipt,
                    demo_root=self.config.demo_root,
                    benchmark_identity=self.catalog.identity,
                    config_sha256=self.config.config_sha256,
                )
            )
            # Suite 先完成全部合同与 routing preflight；任何失败时尚未调用 provider。
            for preflight_task, values in requests.items():
                request, _relative, _instruction, _source = values
                self.runtime.plan(
                    request,
                    access_context=(
                        None if preflight_task is UnifiedTask.KNOWLEDGE_QA else access_context
                    ),
                )

            for task in selected:
                if task is UnifiedTask.REGION_INTERPRETATION and candidate_pending:
                    self._observation.reset()
                    pending_spatial = (
                        snapshot_capture.spatial_result
                        if snapshot_capture is not None
                        else (
                            None
                            if self._active_snapshot is None
                            else self._active_snapshot.spatial_result
                        )
                    )
                    self._observation.spatial = pending_spatial
                    candidate_count = (
                        0 if pending_spatial is None else len(pending_spatial.candidates)
                    )
                    if pending_spatial is None:
                        reason = "SPATIAL_RESULT_REQUIRED"
                    elif candidate_count == 0:
                        reason = "NO_CANDIDATES_EXPLICIT_GLOBAL_REQUIRED"
                    elif has_fresh_spatial:
                        reason = "NEW_SPATIAL_RESULT_REQUIRES_NEW_SELECTION"
                    else:
                        reason = "CANDIDATE_SELECTION_REQUIRED"
                    pending = {
                        "reason": reason,
                        "candidate_count": candidate_count,
                        "required_action": (
                            "Select a candidate or explicitly confirm global mask, then run REGION_INTERPRETATION"
                        ),
                    }
                    instruction = prompt_overrides.get(
                        task,
                        self.config.defaults.prompts[task],
                    )
                    viewer_relative, _viewer = self._viewer(
                        writer=writer,
                        task=task,
                        request=None,
                        task_relative=None,
                        optical_path=optical_path,
                        input_preview_assets=input_preview_assets,
                        response=None,
                        failure=None,
                        status=WAITING_FOR_CANDIDATE,
                        pending=pending,
                        candidate_selection=None,
                        benchmark_payload_consumed_by_task=False,
                        effective_instruction=instruction,
                        instruction_source=(
                            "USER_OVERRIDE" if task in prompt_overrides else "CONFIG_DEFAULT"
                        ),
                    )
                    task_results.append(DemoTaskResult(
                        task=task,
                        status=WAITING_FOR_CANDIDATE,
                        task_root=None,
                        viewer_path=viewer_relative,
                        response=None,
                        failure=None,
                        pending=pending,
                    ))
                    continue

                request, task_relative, instruction, instruction_source = requests[task]
                self._observation.reset()
                response: UnifiedResponse | None = None
                failure: Mapping[str, Any] | None = None
                use_replay = (
                    task is UnifiedTask.REGION_INTERPRETATION
                    and replay_snapshot is not None
                    and region_interpretation_source is RegionSource.OA_AUXSEG_CANDIDATE
                )
                self._observed_spatial.use_replay(replay_snapshot if use_replay else None)
                try:
                    response = self.runtime.infer(
                        request,
                        output_root=writer.path(task_relative),
                        access_context=(
                            None if task is UnifiedTask.KNOWLEDGE_QA else access_context
                        ),
                    )
                    status = "SUCCESS"
                except UnifiedInferenceError as error:
                    failure = error.to_dict()
                    status = "FAILED"
                finally:
                    self._observed_spatial.use_replay(None)
                viewer_relative, viewer = self._viewer(
                    writer=writer,
                    task=task,
                    request=request,
                    task_relative=task_relative,
                    optical_path=optical_path,
                    input_preview_assets=input_preview_assets,
                    response=response,
                    failure=failure,
                    status=status,
                    pending=None,
                    candidate_selection=(
                        candidate_selection
                        if task is UnifiedTask.REGION_INTERPRETATION
                        else None
                    ),
                    benchmark_payload_consumed_by_task=(
                        task is not UnifiedTask.KNOWLEDGE_QA
                    ),
                    effective_instruction=instruction,
                    instruction_source=instruction_source,
                )
                task_results.append(DemoTaskResult(
                    task=task,
                    status=status,
                    task_root=task_relative,
                    viewer_path=viewer_relative,
                    response=None if response is None else response.to_dict(),
                    failure=failure,
                ))
                if (
                    status == "SUCCESS"
                    and task in FRESH_SPATIAL_TASKS
                    and self._observation.spatial is not None
                    and spatial_input is not None
                    and not self._observation.spatial_replayed
                ):
                    spatial_root = writer.path(f"{task_relative}/spatial")
                    spatial_ledger = self._spatial_ledger(spatial_root)
                    snapshot_payload = {
                        "run_id": run_id,
                        "task": task.value,
                        "sample_id": self._observation.spatial.sample_id,
                        "split": self._observation.spatial.split,
                        "source": self._observation.spatial.source,
                        "benchmark_identity": self.catalog.identity,
                        "spatial_ledger": [dict(value) for value in spatial_ledger],
                        "candidates": [
                            candidate.public_dict()
                            for candidate in self._observation.spatial.candidates
                        ],
                    }
                    snapshot_id = "dss_" + sha256_text(
                        canonical_json(snapshot_payload)
                    )[:40]
                    snapshot_capture = _SnapshotCapture(
                        snapshot_id=snapshot_id,
                        task=task,
                        task_relative=task_relative,
                        spatial_input=spatial_input,
                        spatial_result=self._observation.spatial,
                        spatial_ledger=spatial_ledger,
                        viewer=viewer,
                        optical_channel_previews=optical_previews,
                        auxiliary_channel_previews=auxiliary_previews,
                        test_receipt_id=None if receipt is None else receipt.receipt_id,
                    )

            assert writer.staging is not None
            active_snapshot_id = (
                snapshot_capture.snapshot_id
                if snapshot_capture is not None
                else (
                    replay_snapshot.snapshot_id
                    if replay_snapshot is not None
                    else None
                )
            )
            if snapshot_capture is not None:
                writer.write_json("viewer/spatial_snapshot.json", {
                    "schema_version": DEMO_SPATIAL_SNAPSHOT_SCHEMA,
                    "snapshot_id": snapshot_capture.snapshot_id,
                    "run_id": run_id,
                    "sample_id": snapshot_capture.spatial_result.sample_id,
                    "split": snapshot_capture.spatial_result.split,
                    "source": snapshot_capture.spatial_result.source,
                    "source_task": snapshot_capture.task.value,
                    "source_task_root": snapshot_capture.task_relative,
                    "spatial_ledger": [
                        dict(value) for value in snapshot_capture.spatial_ledger
                    ],
                    "candidate_previews": list(
                        snapshot_capture.viewer["spatial_result"]["candidate_previews"]
                    ),
                    "reference_or_gt_mask_consumed": False,
                    "qualitative_demo_only": True,
                    "formal_evaluation": False,
                })
            summary_payload = {
                "schema_version": DEMO_RUN_SUMMARY_SCHEMA,
                "run_id": run_id,
                "sample_id": None if context_record is None else context_record.sample_id,
                "split": None if context_record is None else context_record.split,
                "source": None if context_record is None else context_record.source,
                "data_mode": summary_data_mode,
                "selection_context_mode": data_mode,
                "selection_context_ignored": pure_knowledge,
                "input_scope": input_scope,
                "benchmark_payload_consumed": consumes_visual_payload,
                "benchmark_payload_loaded_this_run": loaded is not None,
                "selected_tasks": [task.value for task in selected],
                "task_results": [item.to_dict() for item in task_results],
                "active_snapshot_id": active_snapshot_id,
                "user_mask_audit": mask_audit,
                "reference_or_gt_mask_consumed": False,
                "selection_unchanged": False,
                "frozen_evaluation_name": None,
                "frozen_ordinal": None,
                "frozen_baseline_record_id": None,
                "formal_evaluation": False,
                "qualitative_demo_only": True,
            }
            writer.write_json("run_summary.json", summary_payload)
            ledger = self._ledger_rows(writer.staging)
            writer.write_jsonl("SHA256SUMS.jsonl", ledger)
            writer.write_json("manifest.json", {
                "schema_version": DEMO_RUN_MANIFEST_SCHEMA,
                "run_id": run_id,
                "sample_id": None if context_record is None else context_record.sample_id,
                "split": None if context_record is None else context_record.split,
                "source": None if context_record is None else context_record.source,
                "data_mode": summary_data_mode,
                "selection_context_mode": data_mode,
                "selection_context_ignored": pure_knowledge,
                "input_scope": input_scope,
                "benchmark_payload_consumed": consumes_visual_payload,
                "benchmark_payload_loaded_this_run": loaded is not None,
                "task_count": len(task_results),
                "success_count": sum(item.status == "SUCCESS" for item in task_results),
                "failure_count": sum(item.status == "FAILED" for item in task_results),
                "pending_count": sum(
                    item.status == WAITING_FOR_CANDIDATE for item in task_results
                ),
                "payload_count": len(ledger),
                "ledger_root_sha256": sha256_text(canonical_json(ledger)),
                "benchmark_identity": self.catalog.identity,
                "config_sha256": self.config.config_sha256,
                "access_receipt_id": None if receipt is None else receipt.receipt_id,
                "sealed_test_accessed": sealed_test_accessed,
                "blind_or_sealed_evaluation_property": (
                    False if sealed_test_accessed else None
                ),
                "active_snapshot_id": active_snapshot_id,
                "selection_unchanged": False,
                "frozen_evaluation_name": None,
                "frozen_baseline_record_id": None,
                "engineering_demo": True,
                "formal_evaluation": False,
                "formal_acceptance": False,
                "scientific_acceptance": False,
            })
            writer.publish()

        if snapshot_capture is not None:
            self._active_snapshot = self._snapshot_from_capture(
                target=target,
                run_id=run_id,
                capture=snapshot_capture,
                sample_id=snapshot_capture.spatial_result.sample_id,
                split=snapshot_capture.spatial_result.split,
                source=snapshot_capture.spatial_result.source,
            )
        active_snapshot_id = (
            self._active_snapshot.snapshot_id
            if self._active_snapshot is not None
            and not pure_knowledge
            and context_record is not None
            and self._active_snapshot.sample_id == context_record.sample_id
            and self._active_snapshot.split == context_record.split
            else None
        )
        return DemoRunSummary(
            run_id=run_id,
            run_root=target,
            sample_id=None if context_record is None else context_record.sample_id,
            split=None if context_record is None else context_record.split,
            source=None if context_record is None else context_record.source,
            data_mode=summary_data_mode,
            tasks=tuple(task_results),
            sealed_test_accessed=sealed_test_accessed,
            access_receipt_id=None if receipt is None else receipt.receipt_id,
            input_scope=input_scope,
            benchmark_payload_consumed=consumes_visual_payload,
            benchmark_payload_loaded_this_run=loaded is not None,
            active_snapshot_id=active_snapshot_id,
        )

    def load_viewer(self, run_root: Path | str, task: UnifiedTask | str) -> Mapping[str, Any]:
        root = Path(os.path.abspath(Path(run_root)))
        try:
            root.relative_to(self.config.demo_root / "runs")
        except ValueError as error:
            raise DemoRunnerError("run root 不属于当前 Demo root") from error
        selected = task if isinstance(task, UnifiedTask) else UnifiedTask(task)
        path = root / "viewer" / f"{selected.value.lower()}.json"
        if (
            first_symlink_component(path) is not None
            or not path.is_file()
            or path.is_symlink()
        ):
            raise DemoRunnerError(f"viewer sidecar 不存在或含链接：{path}")
        return json.loads(path.read_text(encoding="utf-8"))
