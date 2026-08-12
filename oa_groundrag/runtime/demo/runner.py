"""Unified Demo task/suite runner；只编排现有 capability，不实现模型数学。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
import json
import os
from pathlib import Path
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
    FrozenEvaluationItem,
    LoadedBenchmarkSample,
)
from .config import DemoConfig


DEMO_RUN_MANIFEST_SCHEMA = "oa_groundrag.unified_demo.run_manifest.v1"
DEMO_VIEWER_SCHEMA = "oa_groundrag.unified_demo.viewer.v1"

TASK_ORDER = (
    UnifiedTask.VLM_ONLY,
    UnifiedTask.SEGMENT_ONLY,
    UnifiedTask.REGION_UNDERSTANDING,
    UnifiedTask.SEGMENT_AND_UNDERSTAND,
    UnifiedTask.KNOWLEDGE_QA,
    UnifiedTask.REGION_INTERPRETATION,
)


class DemoRunnerError(RuntimeError):
    """Demo preflight、输入 staging 或 suite artifact 合同失败。"""


@dataclass(frozen=True)
class DemoTaskResult:
    task: UnifiedTask
    status: str
    task_root: str
    viewer_path: str
    response: Mapping[str, Any] | None
    failure: Mapping[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task.value,
            "status": self.status,
            "task_root": self.task_root,
            "viewer_path": self.viewer_path,
            "response": None if self.response is None else dict(self.response),
            "failure": None if self.failure is None else dict(self.failure),
        }


@dataclass(frozen=True)
class DemoRunSummary:
    run_id: str
    run_root: Path
    sample_id: str
    split: str
    source: str
    data_mode: str
    tasks: tuple[DemoTaskResult, ...]
    sealed_test_accessed: bool
    access_receipt_id: str | None

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
            "engineering_demo": True,
            "formal_evaluation": False,
            "scientific_acceptance": False,
        }


@dataclass
class _ObservationState:
    spatial: SpatialResult | None = None
    evidence: GroundedEvidenceResult | None = None
    rag: TextRAGResult | None = None
    raw_visual: str | None = None
    raw_knowledge: str | None = None

    def reset(self) -> None:
        self.spatial = None
        self.evidence = None
        self.rag = None
        self.raw_visual = None
        self.raw_knowledge = None


class _ObservedSpatial:
    def __init__(self, target: Any, state: _ObservationState) -> None:
        self.target = target
        self.state = state

    def infer(self, request: UnifiedRequest, *, output_dir: Path) -> SpatialResult:
        result = self.target.infer(request, output_dir=output_dir)
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
        self.runtime = UnifiedInferenceRuntime(
            spatial=_ObservedSpatial(base_runtime.spatial, self._observation),
            shared_mllm=_ObservedShared(base_runtime.shared_mllm, self._observation),
            evidence=_ObservedEvidence(base_runtime.evidence, self._observation),
            text_rag=_ObservedRAG(base_runtime.text_rag, self._observation),
            router=base_runtime.router,
        )
        self.spatial_config = load_runtime_config(config.unified.spatial.config_path)
        self._candidate_cache: dict[tuple[str, str], tuple[int, ...]] = {}

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

        return self._candidate_cache.get((split, sample_id), ())

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

    def _build_request(
        self,
        *,
        task: UnifiedTask,
        request_id: str,
        instruction: str,
        optical_path: Path,
        user_mask_path: Path | None,
        spatial_input: InMemorySpatialInput,
        candidate_region_id: int | None,
        region_interpretation_source: RegionSource,
    ) -> UnifiedRequest:
        if task is UnifiedTask.VLM_ONLY:
            return UnifiedRequest(
                task=task,
                request_id=request_id,
                instruction=instruction,
                images=(optical_path,),
                include_audit=True,
            )
        if task is UnifiedTask.SEGMENT_ONLY:
            return UnifiedRequest(
                task=task,
                request_id=request_id,
                spatial_input=spatial_input,
                include_audit=True,
            )
        if task is UnifiedTask.REGION_UNDERSTANDING:
            if user_mask_path is None:
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
                if user_mask_path is None:
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
        request: UnifiedRequest,
        task_relative: str,
        optical_path: Path,
        response: UnifiedResponse | None,
        failure: Mapping[str, Any] | None,
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
        if spatial is not None:
            optical = render_evidence_image(spatial.optical_image)
            mask = np.asarray(spatial.global_mask, dtype=bool)
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
        grounded_assets: dict[str, str | None] = {}
        for role, name in (
            ("full_optical", "optical_full.png"),
            ("binary_mask", "binary_mask.png"),
            ("context_crop", "context_crop.png"),
        ):
            path = writer.path(f"{task_relative}/grounded/{name}")
            grounded_assets[role] = self._relative(run_staging, path) if path.is_file() else None
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
        viewer = {
            "schema_version": DEMO_VIEWER_SCHEMA,
            "task": task.value,
            "status": "FAILED" if response is None else "SUCCESS",
            "input_preview": {
                "spatial_expert_inputs": {
                    "full_optical": self._relative(run_staging, optical_path),
                    "active_modalities": [] if spatial is None else list(spatial.active_modalities),
                },
                "mllm_formal_grounded_inputs": {
                    **grounded_assets,
                    "formal_roles": [] if evidence is None else list(
                        evidence.metadata.get("formal_model_input_roles", [])
                    ),
                    "auxiliary_modalities_formally_consumed": False,
                },
                "reference_or_gt_mask_consumed": False,
            },
            "spatial_result": {
                "predicted_mask": None if response is None or response.mask_reference is None else (
                    f"{task_relative}/{response.mask_reference}"
                    if not Path(response.mask_reference).is_absolute()
                    else response.mask_reference
                ),
                "mask_probability": probability_reference,
                "overlay": overlay_reference,
                "no_target": None if response is None else response.no_target,
                "no_target_score": None if response is None else response.no_target_score,
                "candidate_count": 0 if response is None else len(response.candidate_regions),
                "candidates": [] if response is None else [dict(value) for value in response.candidate_regions],
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
                "execution_plan": (
                    self.runtime.router.route(request).to_dict()
                    if response is None or response.execution_plan is None
                    else response.execution_plan.to_dict()
                ),
                "provider_call_order": provider_order,
                "candidate_fallback": None if response is None or response.region_selection is None else response.region_selection.to_dict(),
                "cuda_peak_allocated_bytes": max(cuda_peaks, default=0),
                "limitations": [] if response is None else list(response.limitations),
                "trace": trace,
            },
            "response": response_row,
            "failure": None if failure is None else dict(failure),
            "qualitative_demo_only": True,
            "formal_acceptance": False,
            "scientific_acceptance": False,
        }
        relative = f"viewer/{task.value.lower()}.json"
        writer.write_json(relative, viewer)
        return relative, viewer

    def run(
        self,
        *,
        record: BenchmarkRecord,
        tasks: Sequence[UnifiedTask | str],
        instructions: Mapping[UnifiedTask | str, str] | None = None,
        user_mask: Path | str | None = None,
        candidate_region_id: int | None = None,
        region_interpretation_source: RegionSource = RegionSource.OA_AUXSEG_CANDIDATE,
        data_mode: str = "Demo / Qualitative Exploration",
        frozen_item: FrozenEvaluationItem | None = None,
    ) -> DemoRunSummary:
        selected = self.canonical_tasks(tasks)
        if data_mode not in {
            "Demo / Qualitative Exploration",
            "Frozen Evaluation / Read Only",
        }:
            raise DemoRunnerError(f"未知 data_mode：{data_mode}")
        if data_mode == "Frozen Evaluation / Read Only":
            if frozen_item is None or frozen_item.sample_id != record.sample_id:
                raise DemoRunnerError("Frozen inference 必须绑定当前只读 selection item")
        elif frozen_item is not None:
            raise DemoRunnerError("Benchmark Browser run 不得伪装 Frozen selection")
        loaded = self.catalog.load(
            record,
            action="INFERENCE",
            model_normalization=self.spatial_config.normalization,
        )
        run_id = (
            "demo_"
            + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            + "_"
            + uuid4().hex
        )
        target = self.config.demo_root / "runs" / run_id
        task_results: list[DemoTaskResult] = []
        observed_candidates: tuple[int, ...] | None = None
        with AtomicArtifactDirectory(target) as writer:
            optical_path = writer.path("inputs/optical_full.png")
            optical_path.parent.mkdir(parents=True, exist_ok=True)
            loaded.optical_image.save(optical_path, format="PNG", optimize=False)
            staged_mask: Path | None = None
            mask_audit: Mapping[str, Any] | None = None
            if user_mask is not None:
                staged_mask = writer.path("inputs/user_mask.png")
                mask_audit = self.validate_and_stage_user_mask(
                    user_mask,
                    staged_mask,
                    expected_size=loaded.optical_image.size,
                )
            spatial_input = self._spatial_input(loaded)
            prompt_overrides: dict[UnifiedTask, str] = {}
            for key, value in (instructions or {}).items():
                task = key if isinstance(key, UnifiedTask) else UnifiedTask(key)
                if not isinstance(value, str) or not value.strip():
                    raise DemoRunnerError(f"{task.value} instruction 必须是非空字符串")
                prompt_overrides[task] = value.strip()
            requests: list[tuple[UnifiedTask, UnifiedRequest, str]] = []
            for ordinal, task in enumerate(selected):
                instruction = prompt_overrides.get(task, self.config.defaults.prompts[task])
                request = self._build_request(
                    task=task,
                    request_id=f"{run_id}_{task.value.lower()}",
                    instruction=instruction,
                    optical_path=optical_path,
                    user_mask_path=staged_mask,
                    spatial_input=spatial_input,
                    candidate_region_id=candidate_region_id,
                    region_interpretation_source=region_interpretation_source,
                )
                task_relative = f"tasks/{ordinal:02d}_{task.value.lower()}"
                requests.append((task, request, task_relative))
            access_context = (
                None
                if loaded.test_receipt is None
                else DemoInferenceAccess(
                    receipt=loaded.test_receipt,
                    demo_root=self.config.demo_root,
                    benchmark_identity=self.catalog.identity,
                    config_sha256=self.config.config_sha256,
                )
            )
            # Suite 先完成全部合同与 routing preflight；任何失败时尚未调用 provider。
            for _task, request, _relative in requests:
                self.runtime.plan(request, access_context=access_context)
            for task, request, task_relative in requests:
                self._observation.reset()
                response: UnifiedResponse | None = None
                failure: Mapping[str, Any] | None = None
                try:
                    response = self.runtime.infer(
                        request,
                        output_root=writer.path(task_relative),
                        access_context=access_context,
                    )
                    status = "SUCCESS"
                except UnifiedInferenceError as error:
                    failure = error.to_dict()
                    status = "FAILED"
                if self._observation.spatial is not None:
                    observed_candidates = tuple(
                        candidate.region_id
                        for candidate in self._observation.spatial.candidates
                    )
                viewer_relative, _viewer = self._viewer(
                    writer=writer,
                    task=task,
                    request=request,
                    task_relative=task_relative,
                    optical_path=optical_path,
                    response=response,
                    failure=failure,
                )
                task_results.append(DemoTaskResult(
                    task=task,
                    status=status,
                    task_root=task_relative,
                    viewer_path=viewer_relative,
                    response=None if response is None else response.to_dict(),
                    failure=failure,
                ))
            assert writer.staging is not None
            summary_payload = {
                "schema_version": "oa_groundrag.unified_demo.run_summary.v1",
                "run_id": run_id,
                "sample_id": record.sample_id,
                "split": record.split,
                "source": record.source,
                "data_mode": data_mode,
                "selected_tasks": [task.value for task in selected],
                "task_results": [item.to_dict() for item in task_results],
                "user_mask_audit": mask_audit,
                "reference_or_gt_mask_consumed": False,
                "selection_unchanged": data_mode == "Frozen Evaluation / Read Only",
                "frozen_evaluation_name": None if frozen_item is None else frozen_item.evaluation_name,
                "frozen_ordinal": None if frozen_item is None else frozen_item.ordinal,
                "frozen_baseline_record_id": (
                    None if frozen_item is None else frozen_item.baseline_record.get("record_id")
                ),
                "formal_evaluation": False,
                "qualitative_demo_only": True,
            }
            writer.write_json("run_summary.json", summary_payload)
            ledger = self._ledger_rows(writer.staging)
            writer.write_jsonl("SHA256SUMS.jsonl", ledger)
            writer.write_json("manifest.json", {
                "schema_version": DEMO_RUN_MANIFEST_SCHEMA,
                "run_id": run_id,
                "sample_id": record.sample_id,
                "split": record.split,
                "source": record.source,
                "data_mode": data_mode,
                "task_count": len(task_results),
                "success_count": sum(item.status == "SUCCESS" for item in task_results),
                "failure_count": sum(item.status == "FAILED" for item in task_results),
                "payload_count": len(ledger),
                "ledger_root_sha256": sha256_text(canonical_json(ledger)),
                "benchmark_identity": self.catalog.identity,
                "config_sha256": self.config.config_sha256,
                "access_receipt_id": None if loaded.test_receipt is None else loaded.test_receipt.receipt_id,
                "sealed_test_accessed": record.split == "test",
                "blind_or_sealed_evaluation_property": False if record.split == "test" else None,
                "selection_unchanged": data_mode == "Frozen Evaluation / Read Only",
                "frozen_evaluation_name": None if frozen_item is None else frozen_item.evaluation_name,
                "frozen_baseline_record_id": (
                    None if frozen_item is None else frozen_item.baseline_record.get("record_id")
                ),
                "engineering_demo": True,
                "formal_evaluation": False,
                "formal_acceptance": False,
                "scientific_acceptance": False,
            })
            writer.publish()
        if observed_candidates is not None:
            self._candidate_cache[(record.split, record.sample_id)] = observed_candidates
        return DemoRunSummary(
            run_id=run_id,
            run_root=target,
            sample_id=record.sample_id,
            split=record.split,
            source=record.source,
            data_mode=data_mode,
            tasks=tuple(task_results),
            sealed_test_accessed=record.split == "test",
            access_receipt_id=None if loaded.test_receipt is None else loaded.test_receipt.receipt_id,
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
