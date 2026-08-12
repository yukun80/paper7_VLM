"""Grounded Evidence Interface 的版本化公共数据合同。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Any, Mapping, Sequence

import numpy as np

from oa_groundrag.vlm.errors import ContractError, ReasonCode


CONFIG_SCHEMA_VERSION = "rs_vlm.config.v2"
EVIDENCE_SCHEMA_VERSION = "rs_vlm.evidence.v1"
MODEL_OUTPUT_SCHEMA_VERSION = "rs_vlm.model_output.v1"
PREDICTION_SCHEMA_VERSION = "rs_vlm.prediction.v1"
FAILURE_SCHEMA_VERSION = "rs_vlm.failure.v1"
CHECKPOINT_SCHEMA_VERSION = "rs_vlm.checkpoint.v1"
RUN_MANIFEST_SCHEMA_VERSION = "rs_vlm.run_manifest.v1"
SAMPLE_TRACE_SCHEMA_VERSION = "rs_vlm.sample_trace.v1"
GATE_B_PROTOCOL_SCHEMA_VERSION = "rs_vlm.gate_b_protocol.v1"
GATE_B_SELECTION_SCHEMA_VERSION = "rs_vlm.gate_b_selection.v1"
GATE_B_GENERATION_SCHEMA_VERSION = "rs_vlm.gate_b_generation.v1"
GATE_B_REPORT_SCHEMA_VERSION = "rs_vlm.gate_b_report.v1"

SUPPORTED_MANIFEST_SCHEMA = "rs_generaldesc.manifest.v1"
SUPPORTED_CANONICAL_SCHEMA = "rs_generaldesc.canonical.v1"
SUPPORTED_AUXSEG_INFERENCE_SCHEMA = "oa_auxseg_inference_v6"


class DataMode(StrEnum):
    EXTERNAL_GENERIC = "external_generic"


class MaskMode(StrEnum):
    EXTERNAL_GENERIC = "external_generic"
    GT_MASK = "gt_mask"
    FIXED_PREDICTED_MASK = "fixed_predicted_mask"
    END_TO_END_PREDICTED_MASK = "end_to_end_predicted_mask"


class TargetStatus(StrEnum):
    TARGET_PRESENT = "target_present"
    NO_TARGET = "no_target"


class EvidenceSufficiency(StrEnum):
    SUFFICIENT = "sufficient"
    LIMITED = "limited"
    INSUFFICIENT = "insufficient"


class SelectionMode(StrEnum):
    GLOBAL = "global"
    ALL = "all"
    REGION_ID = "region_id"
    BBOX = "bbox"
    CLICK = "click"
    AREA_RANK = "area_rank"
    LOCATION = "location"
    NUMBERED_OVERLAY = "numbered_overlay"


class AlignmentStatus(StrEnum):
    REGISTERED = "registered"
    APPROXIMATE = "approximate"
    UNREGISTERED = "unregistered"
    UNKNOWN = "unknown"


def _binary_mask(value: Any, *, location: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2 or array.size == 0:
        raise ContractError(
            ReasonCode.MASK_INVALID,
            f"{location}: mask 必须是非空二维数组",
        )
    if array.dtype == np.bool_:
        mask = array.copy()
    elif np.issubdtype(array.dtype, np.integer):
        unique = np.unique(array)
        if not set(int(item) for item in unique).issubset({0, 1}):
            raise ContractError(
                ReasonCode.MASK_INVALID,
                f"{location}: 整数 mask 只能包含 0/1",
            )
        mask = array.astype(bool, copy=True)
    else:
        raise ContractError(
            ReasonCode.MASK_INVALID,
            f"{location}: mask dtype 必须是 bool 或 0/1 整数",
        )
    mask.setflags(write=False)
    return mask


def _finite_number(
    value: Any,
    *,
    location: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(
            ReasonCode.TYPE_MISMATCH,
            f"{location}: 必须是数值",
        )
    result = float(value)
    if not math.isfinite(result):
        raise ContractError(
            ReasonCode.NONFINITE_NUMBER,
            f"{location}: 必须是有限数",
        )
    if minimum is not None and result < minimum:
        raise ContractError(
            ReasonCode.TYPE_MISMATCH,
            f"{location}: 必须 >= {minimum}",
        )
    if maximum is not None and result > maximum:
        raise ContractError(
            ReasonCode.TYPE_MISMATCH,
            f"{location}: 必须 <= {maximum}",
        )
    return result


@dataclass(frozen=True)
class RegionCandidate:
    """已有候选区域；mask 是唯一空间真值，几何字段由下游重算。"""

    region_id: str
    mask: np.ndarray
    confidence: float | None = None
    provenance: str = "candidate_region"

    def __post_init__(self) -> None:
        if not isinstance(self.region_id, str) or not self.region_id.strip():
            raise ContractError(
                ReasonCode.TYPE_MISMATCH,
                "region_id 必须是非空字符串",
            )
        object.__setattr__(
            self,
            "mask",
            _binary_mask(self.mask, location=f"region[{self.region_id}].mask"),
        )
        if not bool(self.mask.any()):
            raise ContractError(
                ReasonCode.MASK_INVALID,
                f"region[{self.region_id}]: candidate mask 不能为空",
            )
        if self.confidence is not None:
            object.__setattr__(
                self,
                "confidence",
                _finite_number(
                    self.confidence,
                    location=f"region[{self.region_id}].confidence",
                    minimum=0.0,
                    maximum=1.0,
                ),
            )
        if self.provenance not in {
            "candidate_region",
            "oa_auxseg_inference_v6",
        }:
            raise ContractError(
                ReasonCode.INVALID_ENUM,
                f"region[{self.region_id}]: 非法 provenance",
            )


@dataclass(frozen=True)
class RegionInventory:
    """global mask 与已有 candidate-region mask 的不可扩张清单。"""

    sample_id: str
    mask_mode: MaskMode
    global_mask: np.ndarray
    candidates: tuple[RegionCandidate, ...]
    source_identity: str

    def __post_init__(self) -> None:
        if not isinstance(self.sample_id, str) or not self.sample_id.strip():
            raise ContractError(
                ReasonCode.TYPE_MISMATCH,
                "sample_id 必须是非空字符串",
            )
        if self.mask_mode is MaskMode.EXTERNAL_GENERIC:
            raise ContractError(
                ReasonCode.EXTERNAL_MASK_FORBIDDEN,
                "external_generic 不能创建 RegionInventory",
            )
        global_mask = _binary_mask(
            self.global_mask,
            location=f"inventory[{self.sample_id}].global_mask",
        )
        object.__setattr__(self, "global_mask", global_mask)
        if not isinstance(self.candidates, tuple):
            object.__setattr__(self, "candidates", tuple(self.candidates))
        ids: set[str] = set()
        for candidate in self.candidates:
            if not isinstance(candidate, RegionCandidate):
                raise ContractError(
                    ReasonCode.TYPE_MISMATCH,
                    "candidates 必须是 RegionCandidate",
                )
            if candidate.region_id in ids:
                raise ContractError(
                    ReasonCode.REGION_ID_INVALID,
                    f"重复 region_id：{candidate.region_id}",
                )
            ids.add(candidate.region_id)
            if candidate.mask.shape != global_mask.shape:
                raise ContractError(
                    ReasonCode.MASK_SHAPE_MISMATCH,
                    f"region[{candidate.region_id}] 与 global mask shape 不一致",
                )
            if bool(np.any(candidate.mask & ~global_mask)):
                raise ContractError(
                    ReasonCode.MASK_INVALID,
                    f"region[{candidate.region_id}] 超出已有 global mask",
                )
        if not bool(global_mask.any()) and self.candidates:
            raise ContractError(
                ReasonCode.MASK_INVALID,
                "empty global mask 不允许包含 candidate regions",
            )
        if not isinstance(self.source_identity, str) or not self.source_identity:
            raise ContractError(
                ReasonCode.TYPE_MISMATCH,
                "source_identity 必须是非空字符串",
            )


@dataclass(frozen=True)
class SelectionRequest:
    mode: SelectionMode
    region_id: str | None = None
    bbox_xyxy: tuple[float, float, float, float] | None = None
    click_xy: tuple[float, float] | None = None
    area_rank: int | None = None
    location: str | None = None
    numbered_response: str | None = None
    numbered_selector_frozen: bool | None = None

    def __post_init__(self) -> None:
        expected = {
            SelectionMode.GLOBAL: set(),
            SelectionMode.ALL: set(),
            SelectionMode.REGION_ID: {"region_id"},
            SelectionMode.BBOX: {"bbox_xyxy"},
            SelectionMode.CLICK: {"click_xy"},
            SelectionMode.AREA_RANK: {"area_rank"},
            SelectionMode.LOCATION: {"location"},
            SelectionMode.NUMBERED_OVERLAY: {
                "numbered_response",
                "numbered_selector_frozen",
            },
        }[self.mode]
        values = {
            "region_id": self.region_id,
            "bbox_xyxy": self.bbox_xyxy,
            "click_xy": self.click_xy,
            "area_rank": self.area_rank,
            "location": self.location,
            "numbered_response": self.numbered_response,
            "numbered_selector_frozen": self.numbered_selector_frozen,
        }
        populated = {key for key, value in values.items() if value is not None}
        if populated != expected:
            raise ContractError(
                ReasonCode.UNKNOWN_FIELD,
                f"selection mode={self.mode.value} 只允许字段 {sorted(expected)}",
            )
        if self.area_rank is not None and (
            isinstance(self.area_rank, bool)
            or not isinstance(self.area_rank, int)
            or self.area_rank < 1
        ):
            raise ContractError(
                ReasonCode.TYPE_MISMATCH,
                "area_rank 必须是从 1 开始的整数",
            )
        if self.location is not None and self.location not in {
            "northwest",
            "north",
            "northeast",
            "west",
            "center",
            "east",
            "southwest",
            "south",
            "southeast",
        }:
            raise ContractError(
                ReasonCode.INVALID_ENUM,
                f"非法 3x3 location：{self.location}",
            )
        if (
            self.mode is SelectionMode.NUMBERED_OVERLAY
            and self.numbered_selector_frozen is not True
        ):
            raise ContractError(
                ReasonCode.MODEL_IDENTITY_MISMATCH,
                "numbered overlay selector 必须是冻结模型",
            )


@dataclass(frozen=True)
class SelectedRegion:
    sample_id: str
    target_status: TargetStatus
    selection_mode: SelectionMode
    region_id: str | None
    mask: np.ndarray | None
    source_identity: str

    def __post_init__(self) -> None:
        if self.target_status is TargetStatus.NO_TARGET:
            if self.region_id is not None or self.mask is not None:
                raise ContractError(
                    ReasonCode.MASK_INVALID,
                    "no_target 不允许 region_id 或 mask",
                )
            return
        if self.region_id is None or self.mask is None:
            raise ContractError(
                ReasonCode.MASK_INVALID,
                "target_present 必须有 region_id 和 mask",
            )
        object.__setattr__(
            self,
            "mask",
            _binary_mask(self.mask, location="SelectedRegion.mask"),
        )
        if not bool(self.mask.any()):
            raise ContractError(
                ReasonCode.MASK_INVALID,
                "target_present mask 不能为空",
            )


@dataclass(frozen=True)
class AuxiliaryView:
    """可展示的辅助证据及其科学边界；不接受内部权重或 feature。"""

    evidence_id: str
    name: str
    image: Any
    alignment: AlignmentStatus
    coverage: float | None
    unit: str | None
    sign_convention: str | None

    def __post_init__(self) -> None:
        for name, value in (
            ("evidence_id", self.evidence_id),
            ("name", self.name),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ContractError(
                    ReasonCode.TYPE_MISMATCH,
                    f"AuxiliaryView.{name} 必须是非空字符串",
                )
        if self.coverage is not None:
            object.__setattr__(
                self,
                "coverage",
                _finite_number(
                    self.coverage,
                    location=f"aux[{self.name}].coverage",
                    minimum=0.0,
                    maximum=1.0,
                ),
            )
        for name, value in (
            ("unit", self.unit),
            ("sign_convention", self.sign_convention),
        ):
            if value is not None and (
                not isinstance(value, str) or not value.strip()
            ):
                raise ContractError(
                    ReasonCode.TYPE_MISMATCH,
                    f"AuxiliaryView.{name} 必须是非空字符串或 null",
                )


@dataclass(frozen=True)
class EvidenceAsset:
    evidence_id: str
    role: str
    relative_path: str

    def __post_init__(self) -> None:
        if not self.evidence_id or not self.role:
            raise ContractError(
                ReasonCode.TYPE_MISMATCH,
                "EvidenceAsset id/role 必须非空",
            )
        path = PurePosixPath(self.relative_path)
        if (
            path.is_absolute()
            or not path.parts
            or any(part in {"", ".", ".."} for part in path.parts)
        ):
            raise ContractError(
                ReasonCode.PATH_ESCAPE,
                f"非法 evidence 相对路径：{self.relative_path}",
            )


@dataclass(frozen=True)
class EvidenceBundle:
    schema_version: str
    sample_id: str
    mask_mode: MaskMode
    target_status: TargetStatus
    selection: Mapping[str, Any]
    deterministic_facts: Mapping[str, Any]
    assets: tuple[EvidenceAsset, ...]
    auxiliary_metadata: tuple[Mapping[str, Any], ...]
    limitations: tuple[str, ...]
    excluded_internal_signals: tuple[str, ...]
    rag_context: tuple[Any, ...]

    def __post_init__(self) -> None:
        if self.schema_version != EVIDENCE_SCHEMA_VERSION:
            raise ContractError(
                ReasonCode.INVALID_ENUM,
                f"不支持 evidence schema：{self.schema_version}",
            )
        if self.rag_context:
            raise ContractError(
                ReasonCode.RAG_FORBIDDEN,
                "Phase 3 evidence 的 rag_context 必须为空",
            )
        ids = [asset.evidence_id for asset in self.assets]
        if len(ids) != len(set(ids)):
            raise ContractError(
                ReasonCode.EVIDENCE_REFERENCE_INVALID,
                "evidence_id 不能重复",
            )
        if self.target_status is TargetStatus.NO_TARGET:
            forbidden_roles = {"mask_overlay", "context_crop"}
            if any(asset.role in forbidden_roles for asset in self.assets):
                raise ContractError(
                    ReasonCode.MASK_INVALID,
                    "no_target 不允许 overlay/crop",
                )
            if self.deterministic_facts:
                raise ContractError(
                    ReasonCode.FORBIDDEN_MODEL_FACT,
                    "no_target 不允许几何事实",
                )

    @property
    def evidence_ids(self) -> frozenset[str]:
        return frozenset(asset.evidence_id for asset in self.assets)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "sample_id": self.sample_id,
            "mask_mode": self.mask_mode.value,
            "target_status": self.target_status.value,
            "selection": dict(self.selection),
            "deterministic_facts": dict(self.deterministic_facts),
            "assets": [
                {
                    "evidence_id": asset.evidence_id,
                    "role": asset.role,
                    "relative_path": asset.relative_path,
                }
                for asset in self.assets
            ],
            "auxiliary_metadata": [
                dict(value) for value in self.auxiliary_metadata
            ],
            "limitations": list(self.limitations),
            "excluded_internal_signals": list(
                self.excluded_internal_signals
            ),
            "rag_context": [],
        }


@dataclass(frozen=True)
class Claim:
    text: str
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True)
class StructuredModelOutput:
    schema_version: str
    target_status: TargetStatus
    description: str
    answer: str | None
    evidence_sufficiency: EvidenceSufficiency
    claims: tuple[Claim, ...]
    limitations: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "target_status": self.target_status.value,
            "description": self.description,
            "answer": self.answer,
            "evidence_sufficiency": self.evidence_sufficiency.value,
            "claims": [
                {
                    "text": claim.text,
                    "evidence_ids": list(claim.evidence_ids),
                }
                for claim in self.claims
            ],
            "limitations": list(self.limitations),
        }


def ensure_unique_strings(
    value: Any,
    *,
    location: str,
    allow_empty: bool = True,
) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise ContractError(
            ReasonCode.TYPE_MISMATCH,
            f"{location}: 必须是非空字符串列表",
        )
    normalized = tuple(item.strip() for item in value)
    if not allow_empty and not normalized:
        raise ContractError(
            ReasonCode.TYPE_MISMATCH,
            f"{location}: 不能为空",
        )
    if len(normalized) != len(set(normalized)):
        raise ContractError(
            ReasonCode.TYPE_MISMATCH,
            f"{location}: 不允许重复",
        )
    return normalized
