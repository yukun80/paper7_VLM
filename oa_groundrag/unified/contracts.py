"""OA-GroundRAG P0 unified inference 的稳定公共合同。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Mapping, Sequence


UNIFIED_REQUEST_SCHEMA = "oa_groundrag.unified_request.v1"
UNIFIED_RESPONSE_SCHEMA = "oa_groundrag.unified_response.v1"
UNIFIED_FAILURE_SCHEMA = "oa_groundrag.unified_failure.v1"


class UnifiedTask(StrEnum):
    VLM_ONLY = "VLM_ONLY"
    SEGMENT_ONLY = "SEGMENT_ONLY"
    REGION_UNDERSTANDING = "REGION_UNDERSTANDING"
    SEGMENT_AND_UNDERSTAND = "SEGMENT_AND_UNDERSTAND"
    KNOWLEDGE_QA = "KNOWLEDGE_QA"
    REGION_INTERPRETATION = "REGION_INTERPRETATION"


class RegionSource(StrEnum):
    NONE = "NONE"
    USER_MASK = "USER_MASK"
    OA_AUXSEG_GLOBAL = "OA_AUXSEG_GLOBAL"
    OA_AUXSEG_CANDIDATE = "OA_AUXSEG_CANDIDATE"
    GT_MASK = "GT_MASK"


class ResponseKind(StrEnum):
    TEXT = "TEXT"
    SPATIAL = "SPATIAL"
    REGION_OBSERVATION = "REGION_OBSERVATION"
    KNOWLEDGE_ANSWER = "KNOWLEDGE_ANSWER"
    REGION_INTERPRETATION = "REGION_INTERPRETATION"


class RegionSelectionStatus(StrEnum):
    DIRECT = "DIRECT"
    CANDIDATE_SELECTED = "CANDIDATE_SELECTED"
    FALLBACK_GLOBAL = "FALLBACK_GLOBAL"


class RegionSelectionReason(StrEnum):
    NONE = "NONE"
    CANDIDATE_ID_MISSING = "CANDIDATE_ID_MISSING"
    CANDIDATE_ID_NOT_FOUND = "CANDIDATE_ID_NOT_FOUND"
    NO_CANDIDATES = "NO_CANDIDATES"


class UnifiedReasonCode(StrEnum):
    INVALID_TASK = "INVALID_TASK"
    REQUEST_CONTRACT_INVALID = "REQUEST_CONTRACT_INVALID"
    FORBIDDEN_REGION_SOURCE = "FORBIDDEN_REGION_SOURCE"
    TEST_OR_SEALED_PATH_FORBIDDEN = "TEST_OR_SEALED_PATH_FORBIDDEN"
    OUTPUT_EXISTS = "OUTPUT_EXISTS"
    OUTPUT_LINK = "OUTPUT_LINK"
    ARTIFACT_IDENTITY_MISMATCH = "ARTIFACT_IDENTITY_MISMATCH"
    SPATIAL_PROVIDER_FAILED = "SPATIAL_PROVIDER_FAILED"
    EVIDENCE_PROVIDER_FAILED = "EVIDENCE_PROVIDER_FAILED"
    SHARED_MLLM_FAILED = "SHARED_MLLM_FAILED"
    TEXT_RAG_FAILED = "TEXT_RAG_FAILED"
    INVALID_MODEL_OUTPUT = "INVALID_MODEL_OUTPUT"
    ARTIFACT_WRITE_FAILED = "ARTIFACT_WRITE_FAILED"


class UnifiedInferenceError(RuntimeError):
    """统一层稳定错误；provider 失败不会触发另一 capability。"""

    def __init__(
        self,
        reason_code: UnifiedReasonCode,
        message: str,
        *,
        task: UnifiedTask | None = None,
        provider: str | None = None,
        completed_trace: Sequence[Mapping[str, Any]] = (),
        cause_type: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        self.reason_code = reason_code
        self.task = task
        self.provider = provider
        self.completed_trace = tuple(dict(row) for row in completed_trace)
        self.cause_type = cause_type
        self.details = dict(details or {})
        super().__init__(f"{reason_code.value}: {message}")
        self.message = message

    def with_trace(
        self,
        trace: Sequence[Mapping[str, Any]],
    ) -> "UnifiedInferenceError":
        return UnifiedInferenceError(
            self.reason_code,
            self.message,
            task=self.task,
            provider=self.provider,
            completed_trace=trace,
            cause_type=self.cause_type,
            details=self.details,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": UNIFIED_FAILURE_SCHEMA,
            "reason_code": self.reason_code.value,
            "task": None if self.task is None else self.task.value,
            "provider": self.provider,
            "message": self.message,
            "cause_type": self.cause_type,
            "details": dict(self.details),
            "completed_trace": [dict(row) for row in self.completed_trace],
        }


def reject_test_or_sealed_path(path: Path | str, *, label: str) -> Path:
    """拒绝词法/解析路径组件中的 test 或 sealed，且不打开目标文件。"""

    value = Path(path)
    for part in value.resolve(strict=False).parts:
        if part.lower() in {"test", "sealed"}:
            raise UnifiedInferenceError(
                UnifiedReasonCode.TEST_OR_SEALED_PATH_FORBIDDEN,
                f"{label} 指向 test/sealed 路径：{value}",
            )
    return value


@dataclass(frozen=True)
class BenchmarkSampleRef:
    split: str
    sample_id: str

    def __post_init__(self) -> None:
        if self.split not in {"train", "val"}:
            raise UnifiedInferenceError(
                UnifiedReasonCode.TEST_OR_SEALED_PATH_FORBIDDEN,
                "Unified Runtime 的 benchmark split 只允许 train/val",
            )
        if not isinstance(self.sample_id, str) or not self.sample_id.strip():
            raise UnifiedInferenceError(
                UnifiedReasonCode.REQUEST_CONTRACT_INVALID,
                "spatial_input.sample_id 必须是非空字符串",
            )

    def to_dict(self) -> dict[str, str]:
        return {"kind": "benchmark_sample", "split": self.split, "sample_id": self.sample_id}


@dataclass(frozen=True)
class InMemorySpatialInput:
    """Python API 专用；batch 仍是既有 OAAuxSegBatch，不在此模块导入 torch。"""

    batch: Any
    optical_image: Any
    sample_id: str
    source: str
    split: str

    def __post_init__(self) -> None:
        if self.split not in {"train", "val"}:
            raise UnifiedInferenceError(
                UnifiedReasonCode.TEST_OR_SEALED_PATH_FORBIDDEN,
                "in-memory spatial split 只允许 train/val",
            )
        if not self.sample_id or not self.source:
            raise UnifiedInferenceError(
                UnifiedReasonCode.REQUEST_CONTRACT_INVALID,
                "in-memory spatial input 缺少 sample_id/source",
            )


SpatialInput = BenchmarkSampleRef | InMemorySpatialInput


@dataclass(frozen=True)
class AuxiliaryViewInput:
    evidence_id: str
    name: str
    image: Path
    alignment: str
    coverage: float | None = None
    unit: str | None = None
    sign_convention: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.evidence_id, str) or not self.evidence_id.strip():
            raise UnifiedInferenceError(
                UnifiedReasonCode.REQUEST_CONTRACT_INVALID,
                "auxiliary view 的 evidence_id 必须为非空字符串",
            )
        if not isinstance(self.name, str) or not self.name.strip():
            raise UnifiedInferenceError(
                UnifiedReasonCode.REQUEST_CONTRACT_INVALID,
                "auxiliary view 的 name 必须为非空字符串",
            )
        if not isinstance(self.image, Path):
            raise UnifiedInferenceError(
                UnifiedReasonCode.REQUEST_CONTRACT_INVALID,
                "auxiliary view image 必须是 Path",
            )
        if not isinstance(self.alignment, str) or self.alignment not in {
            "registered", "approximate", "unregistered", "unknown",
        }:
            raise UnifiedInferenceError(
                UnifiedReasonCode.REQUEST_CONTRACT_INVALID,
                f"auxiliary view alignment 非法：{self.alignment!r}",
            )
        if self.coverage is not None and (
            isinstance(self.coverage, bool)
            or not isinstance(self.coverage, (int, float))
            or not 0.0 <= float(self.coverage) <= 1.0
        ):
            raise UnifiedInferenceError(
                UnifiedReasonCode.REQUEST_CONTRACT_INVALID,
                "auxiliary view coverage 必须位于 [0,1] 或为 null",
            )
        for label, value in (("unit", self.unit), ("sign_convention", self.sign_convention)):
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise UnifiedInferenceError(
                    UnifiedReasonCode.REQUEST_CONTRACT_INVALID,
                    f"auxiliary view {label} 必须是非空字符串或 null",
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "name": self.name,
            "image": str(self.image),
            "alignment": self.alignment,
            "coverage": self.coverage,
            "unit": self.unit,
            "sign_convention": self.sign_convention,
        }


@dataclass(frozen=True)
class UnifiedRequest:
    task: UnifiedTask
    request_id: str
    instruction: str | None = None
    images: tuple[Path, ...] = ()
    user_mask: Path | None = None
    spatial_input: SpatialInput | None = None
    region_source: RegionSource = RegionSource.NONE
    candidate_region_id: int | None = None
    auxiliary_views: tuple[AuxiliaryViewInput, ...] = ()
    include_audit: bool = False
    schema_version: str = UNIFIED_REQUEST_SCHEMA

    def __post_init__(self) -> None:
        if not isinstance(self.task, UnifiedTask):
            raise UnifiedInferenceError(
                UnifiedReasonCode.INVALID_TASK,
                f"未知 task：{self.task!r}",
            )
        if not isinstance(self.region_source, RegionSource):
            raise UnifiedInferenceError(
                UnifiedReasonCode.FORBIDDEN_REGION_SOURCE,
                f"未知 region_source：{self.region_source!r}",
                task=self.task,
            )
        if self.instruction is not None and not isinstance(self.instruction, str):
            raise UnifiedInferenceError(
                UnifiedReasonCode.REQUEST_CONTRACT_INVALID,
                "instruction 必须是字符串或 null",
                task=self.task,
            )
        if self.schema_version != UNIFIED_REQUEST_SCHEMA:
            raise UnifiedInferenceError(
                UnifiedReasonCode.REQUEST_CONTRACT_INVALID,
                f"只支持 request schema {UNIFIED_REQUEST_SCHEMA}",
                task=self.task,
            )
        if not isinstance(self.request_id, str) or not self.request_id.strip():
            raise UnifiedInferenceError(
                UnifiedReasonCode.REQUEST_CONTRACT_INVALID,
                "request_id 必须是非空字符串",
                task=self.task,
            )
        if self.candidate_region_id is not None and (
            isinstance(self.candidate_region_id, bool)
            or not isinstance(self.candidate_region_id, int)
            or self.candidate_region_id < 0
        ):
            raise UnifiedInferenceError(
                UnifiedReasonCode.REQUEST_CONTRACT_INVALID,
                "candidate_region_id 必须是 >= 0 的整数或 null",
                task=self.task,
            )
        if not isinstance(self.include_audit, bool):
            raise UnifiedInferenceError(
                UnifiedReasonCode.REQUEST_CONTRACT_INVALID,
                "include_audit 必须是 bool",
                task=self.task,
            )
        evidence_ids = [view.evidence_id for view in self.auxiliary_views]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise UnifiedInferenceError(
                UnifiedReasonCode.REQUEST_CONTRACT_INVALID,
                "auxiliary_views.evidence_id 不允许重复",
                task=self.task,
            )
        if self.auxiliary_views and self.task not in {
            UnifiedTask.REGION_UNDERSTANDING,
            UnifiedTask.SEGMENT_AND_UNDERSTAND,
            UnifiedTask.REGION_INTERPRETATION,
        }:
            raise UnifiedInferenceError(
                UnifiedReasonCode.REQUEST_CONTRACT_INVALID,
                "auxiliary_views P0 只允许 region task 记录元数据",
                task=self.task,
            )
        self.validate_task_contract()

    @property
    def normalized_instruction(self) -> str | None:
        if self.instruction is None:
            return None
        value = self.instruction.strip()
        return value or None

    def _require_instruction(self) -> None:
        if self.normalized_instruction is None:
            raise UnifiedInferenceError(
                UnifiedReasonCode.REQUEST_CONTRACT_INVALID,
                f"{self.task.value} 要求非空 instruction",
                task=self.task,
            )

    def validate_task_contract(self) -> None:
        if self.region_source is RegionSource.GT_MASK:
            raise UnifiedInferenceError(
                UnifiedReasonCode.FORBIDDEN_REGION_SOURCE,
                "GT_MASK 只允许 evaluation-side 使用",
                task=self.task,
            )
        if self.task is UnifiedTask.VLM_ONLY:
            self._require_instruction()
            if not self.images or self.spatial_input is not None or self.user_mask is not None:
                self._invalid("VLM_ONLY 要求 image(s)，且禁止 spatial/user_mask")
            if self.region_source is not RegionSource.NONE or self.candidate_region_id is not None:
                self._invalid("VLM_ONLY 的 region_source 必须为 NONE")
        elif self.task is UnifiedTask.SEGMENT_ONLY:
            if self.spatial_input is None or self.images or self.user_mask is not None:
                self._invalid("SEGMENT_ONLY 只接受 spatial_input")
            if self.region_source is not RegionSource.NONE or self.candidate_region_id is not None:
                self._invalid("SEGMENT_ONLY 的 region_source 必须为 NONE")
        elif self.task is UnifiedTask.REGION_UNDERSTANDING:
            self._require_instruction()
            if len(self.images) != 1 or self.user_mask is None or self.spatial_input is not None:
                self._invalid("REGION_UNDERSTANDING 要求单 image + user_mask")
            if self.region_source is not RegionSource.USER_MASK or self.candidate_region_id is not None:
                self._invalid("REGION_UNDERSTANDING 的 region_source 必须为 USER_MASK")
        elif self.task is UnifiedTask.SEGMENT_AND_UNDERSTAND:
            self._require_instruction()
            if self.spatial_input is None or self.user_mask is not None or self.images:
                self._invalid("SEGMENT_AND_UNDERSTAND 只接受 spatial_input + instruction")
            if self.region_source is not RegionSource.OA_AUXSEG_GLOBAL or self.candidate_region_id is not None:
                self._invalid("SEGMENT_AND_UNDERSTAND 必须使用 OA_AUXSEG_GLOBAL")
        elif self.task is UnifiedTask.KNOWLEDGE_QA:
            self._require_instruction()
            if self.images or self.user_mask is not None or self.spatial_input is not None:
                self._invalid("KNOWLEDGE_QA 禁止 image/mask/spatial")
            if self.region_source is not RegionSource.NONE or self.candidate_region_id is not None:
                self._invalid("KNOWLEDGE_QA 的 region_source 必须为 NONE")
        elif self.task is UnifiedTask.REGION_INTERPRETATION:
            self._require_instruction()
            if self.region_source is RegionSource.USER_MASK:
                if len(self.images) != 1 or self.user_mask is None or self.spatial_input is not None:
                    self._invalid("USER_MASK interpretation 要求单 image + user_mask")
                if self.candidate_region_id is not None:
                    self._invalid("USER_MASK interpretation 禁止 candidate_region_id")
            elif self.region_source is RegionSource.OA_AUXSEG_CANDIDATE:
                if self.spatial_input is None or self.user_mask is not None or self.images:
                    self._invalid("candidate interpretation 只接受 spatial_input")
            else:
                raise UnifiedInferenceError(
                    UnifiedReasonCode.FORBIDDEN_REGION_SOURCE,
                    "REGION_INTERPRETATION 只允许 USER_MASK/OA_AUXSEG_CANDIDATE",
                    task=self.task,
                )

    def _invalid(self, message: str) -> None:
        raise UnifiedInferenceError(
            UnifiedReasonCode.REQUEST_CONTRACT_INVALID,
            message,
            task=self.task,
        )

    def validate_paths(self) -> None:
        for index, image in enumerate(self.images):
            reject_test_or_sealed_path(image, label=f"images[{index}]")
        if self.user_mask is not None:
            reject_test_or_sealed_path(self.user_mask, label="user_mask")
        for index, view in enumerate(self.auxiliary_views):
            reject_test_or_sealed_path(view.image, label=f"auxiliary_views[{index}].image")

    def to_dict(self) -> dict[str, Any]:
        if isinstance(self.spatial_input, BenchmarkSampleRef):
            spatial: Mapping[str, Any] | None = self.spatial_input.to_dict()
        elif self.spatial_input is None:
            spatial = None
        else:
            spatial = {
                "kind": "in_memory",
                "sample_id": self.spatial_input.sample_id,
                "source": self.spatial_input.source,
                "split": self.spatial_input.split,
            }
        return {
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "task": self.task.value,
            "instruction": self.instruction,
            "images": [str(path) for path in self.images],
            "user_mask": None if self.user_mask is None else str(self.user_mask),
            "spatial_input": spatial,
            "region_source": self.region_source.value,
            "candidate_region_id": self.candidate_region_id,
            "auxiliary_views": [view.to_dict() for view in self.auxiliary_views],
            "include_audit": self.include_audit,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "UnifiedRequest":
        if not isinstance(value, Mapping):
            raise UnifiedInferenceError(
                UnifiedReasonCode.REQUEST_CONTRACT_INVALID,
                "request 顶层必须是对象",
            )
        row = dict(value)
        fields = {
            "schema_version", "request_id", "task", "instruction", "images",
            "user_mask", "spatial_input", "region_source", "candidate_region_id",
            "auxiliary_views", "include_audit",
        }
        unknown = set(row) - fields
        if unknown:
            raise UnifiedInferenceError(
                UnifiedReasonCode.REQUEST_CONTRACT_INVALID,
                f"request 含未知字段：{sorted(unknown)}",
            )
        try:
            task = UnifiedTask(row.get("task"))
        except (TypeError, ValueError) as error:
            raise UnifiedInferenceError(
                UnifiedReasonCode.INVALID_TASK,
                f"未知 task：{row.get('task')!r}",
                cause_type=type(error).__name__,
            ) from error
        try:
            source = RegionSource(row.get("region_source", RegionSource.NONE.value))
        except (TypeError, ValueError) as error:
            raise UnifiedInferenceError(
                UnifiedReasonCode.FORBIDDEN_REGION_SOURCE,
                f"未知 region_source：{row.get('region_source')!r}",
                task=task,
            ) from error
        images_value = row.get("images", [])
        if not isinstance(images_value, list) or not all(isinstance(item, str) and item for item in images_value):
            raise UnifiedInferenceError(
                UnifiedReasonCode.REQUEST_CONTRACT_INVALID,
                "images 必须是路径字符串列表",
                task=task,
            )
        spatial_value = row.get("spatial_input")
        spatial: SpatialInput | None = None
        if spatial_value is not None:
            if not isinstance(spatial_value, Mapping) or set(spatial_value) != {"kind", "split", "sample_id"}:
                raise UnifiedInferenceError(
                    UnifiedReasonCode.REQUEST_CONTRACT_INVALID,
                    "CLI spatial_input 只接受 benchmark_sample 合同",
                    task=task,
                )
            if spatial_value.get("kind") != "benchmark_sample":
                raise UnifiedInferenceError(
                    UnifiedReasonCode.REQUEST_CONTRACT_INVALID,
                    "CLI spatial_input.kind 只允许 benchmark_sample",
                    task=task,
                )
            if not all(
                isinstance(spatial_value.get(name), str) and spatial_value[name]
                for name in ("split", "sample_id")
            ):
                raise UnifiedInferenceError(
                    UnifiedReasonCode.REQUEST_CONTRACT_INVALID,
                    "spatial_input.split/sample_id 必须是非空字符串",
                    task=task,
                )
            spatial = BenchmarkSampleRef(
                split=spatial_value["split"],
                sample_id=spatial_value["sample_id"],
            )
        auxiliaries_value = row.get("auxiliary_views", [])
        if not isinstance(auxiliaries_value, list):
            raise UnifiedInferenceError(
                UnifiedReasonCode.REQUEST_CONTRACT_INVALID,
                "auxiliary_views 必须是列表",
                task=task,
            )
        auxiliary_views: list[AuxiliaryViewInput] = []
        auxiliary_fields = {
            "evidence_id", "name", "image", "alignment", "coverage", "unit", "sign_convention",
        }
        for index, item in enumerate(auxiliaries_value):
            if not isinstance(item, Mapping) or set(item) != auxiliary_fields:
                raise UnifiedInferenceError(
                    UnifiedReasonCode.REQUEST_CONTRACT_INVALID,
                    f"auxiliary_views[{index}] 字段不匹配",
                    task=task,
                )
            if not all(
                isinstance(item.get(name), str) and item[name]
                for name in ("evidence_id", "name", "image", "alignment")
            ):
                raise UnifiedInferenceError(
                    UnifiedReasonCode.REQUEST_CONTRACT_INVALID,
                    f"auxiliary_views[{index}] 的 ID/name/image/alignment 必须是非空字符串",
                    task=task,
                )
            coverage = item["coverage"]
            if coverage is not None and (
                isinstance(coverage, bool) or not isinstance(coverage, (int, float))
            ):
                raise UnifiedInferenceError(
                    UnifiedReasonCode.REQUEST_CONTRACT_INVALID,
                    f"auxiliary_views[{index}].coverage 必须是数值或 null",
                    task=task,
                )
            if any(
                item[name] is not None and not isinstance(item[name], str)
                for name in ("unit", "sign_convention")
            ):
                raise UnifiedInferenceError(
                    UnifiedReasonCode.REQUEST_CONTRACT_INVALID,
                    f"auxiliary_views[{index}] unit/sign_convention 必须是字符串或 null",
                    task=task,
                )
            auxiliary_views.append(AuxiliaryViewInput(
                evidence_id=item["evidence_id"],
                name=item["name"],
                image=Path(item["image"]),
                alignment=item["alignment"],
                coverage=None if coverage is None else float(coverage),
                unit=item["unit"],
                sign_convention=item["sign_convention"],
            ))
        instruction = row.get("instruction")
        if instruction is not None and not isinstance(instruction, str):
            raise UnifiedInferenceError(
                UnifiedReasonCode.REQUEST_CONTRACT_INVALID,
                "instruction 必须是字符串或 null",
                task=task,
            )
        user_mask = row.get("user_mask")
        if user_mask is not None and not isinstance(user_mask, str):
            raise UnifiedInferenceError(
                UnifiedReasonCode.REQUEST_CONTRACT_INVALID,
                "user_mask 必须是路径字符串或 null",
                task=task,
            )
        return cls(
            schema_version=row.get("schema_version", UNIFIED_REQUEST_SCHEMA),
            request_id=row.get("request_id", ""),
            task=task,
            instruction=instruction,
            images=tuple(Path(item) for item in images_value),
            user_mask=None if user_mask is None else Path(user_mask),
            spatial_input=spatial,
            region_source=source,
            candidate_region_id=row.get("candidate_region_id"),
            auxiliary_views=tuple(auxiliary_views),
            include_audit=row.get("include_audit", False),
        )


@dataclass(frozen=True)
class ExecutionPlan:
    needs_spatial: bool
    needs_shared_mllm: bool
    needs_region: bool
    needs_rag: bool
    region_source: RegionSource
    response_kind: ResponseKind

    def to_dict(self) -> dict[str, Any]:
        return {
            "needs_spatial": self.needs_spatial,
            "needs_shared_mllm": self.needs_shared_mllm,
            "needs_region": self.needs_region,
            "needs_rag": self.needs_rag,
            "region_source": self.region_source.value,
            "response_kind": self.response_kind.value,
        }


@dataclass(frozen=True)
class RegionSelection:
    requested_source: RegionSource
    effective_source: RegionSource
    status: RegionSelectionStatus
    reason: RegionSelectionReason = RegionSelectionReason.NONE
    requested_candidate_id: int | None = None
    selected_candidate_id: int | None = None
    candidate_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested_source": self.requested_source.value,
            "effective_source": self.effective_source.value,
            "status": self.status.value,
            "reason": self.reason.value,
            "requested_candidate_id": self.requested_candidate_id,
            "selected_candidate_id": self.selected_candidate_id,
            "candidate_count": self.candidate_count,
        }


@dataclass(frozen=True)
class UnifiedResponse:
    request_id: str
    task: UnifiedTask
    response_kind: ResponseKind
    text: str | None = None
    mask_reference: str | None = None
    mask_probability_reference: str | None = None
    no_target: bool | None = None
    no_target_score: float | None = None
    candidate_regions: tuple[Mapping[str, Any], ...] = ()
    region_selection: RegionSelection | None = None
    region_observation: Mapping[str, Any] | None = None
    knowledge_output: Mapping[str, Any] | None = None
    citations: tuple[Mapping[str, Any], ...] = ()
    limitations: tuple[str, ...] = ()
    execution_plan: ExecutionPlan | None = None
    trace: tuple[Mapping[str, Any], ...] = ()
    status: str = "SUCCESS"
    schema_version: str = UNIFIED_RESPONSE_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "task": self.task.value,
            "status": self.status,
            "response_kind": self.response_kind.value,
            "text": self.text,
            "mask_reference": self.mask_reference,
            "mask_probability_reference": self.mask_probability_reference,
            "no_target": self.no_target,
            "no_target_score": self.no_target_score,
            "candidate_regions": [dict(item) for item in self.candidate_regions],
            "region_selection": None if self.region_selection is None else self.region_selection.to_dict(),
            "region_observation": None if self.region_observation is None else dict(self.region_observation),
            "knowledge_output": None if self.knowledge_output is None else dict(self.knowledge_output),
            "citations": [dict(item) for item in self.citations],
            "limitations": list(self.limitations),
        }
        if self.execution_plan is not None:
            row["execution_plan"] = self.execution_plan.to_dict()
            row["trace"] = [dict(item) for item in self.trace]
        return row
