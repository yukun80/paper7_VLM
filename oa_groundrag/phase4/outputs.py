"""严格结构化模型输出、prediction 与 failure 合同。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Iterable, Mapping

from oa_groundrag.phase3.common import canonical_json

from .contracts import (
    FAILURE_SCHEMA_VERSION,
    MODEL_OUTPUT_SCHEMA_VERSION,
    PREDICTION_SCHEMA_VERSION,
    Claim,
    EvidenceSufficiency,
    MaskMode,
    StructuredModelOutput,
    TargetStatus,
    ensure_unique_strings,
)
from .errors import ContractError, PredictionError, ReasonCode


_FORBIDDEN_GEOMETRY_PATTERNS = (
    r"\bbbox\b",
    r"\bbounding box\b",
    r"\bcentroid\b",
    r"\barea[_ ]pixels\b",
    r"\barea ratio\b",
    r"\bperimeter\b",
    r"\bcompactness\b",
    r"\belongation\b",
    r"\b(?:northwest|northeast|southwest|southeast)\b",
)


def _reject_model_geometry_language(value: str, *, location: str) -> None:
    lowered = value.lower()
    if any(re.search(pattern, lowered) for pattern in _FORBIDDEN_GEOMETRY_PATTERNS):
        raise ContractError(
            ReasonCode.FORBIDDEN_MODEL_FACT,
            f"{location}: 模型文本不得重写程序化几何事实",
        )


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key: {key}")
        result[key] = value
    return result


def _invalid_constant(token: str) -> None:
    raise ValueError(f"non-finite constant: {token}")


def parse_model_output(
    value: str | Mapping[str, Any],
    *,
    valid_evidence_ids: Iterable[str],
) -> StructuredModelOutput:
    if isinstance(value, str):
        try:
            row = json.loads(
                value,
                object_pairs_hook=_unique_object,
                parse_constant=_invalid_constant,
            )
        except (json.JSONDecodeError, ValueError) as error:
            raise ContractError(
                ReasonCode.INVALID_MODEL_OUTPUT,
                "模型输出不是严格 JSON",
                details={"error": str(error)},
            ) from error
    elif isinstance(value, Mapping):
        row = dict(value)
    else:
        raise ContractError(
            ReasonCode.TYPE_MISMATCH,
            "模型输出必须是 JSON 字符串或对象",
        )
    expected = {
        "schema_version",
        "target_status",
        "description",
        "answer",
        "evidence_sufficiency",
        "claims",
        "limitations",
    }
    unknown = set(row) - expected
    missing = expected - set(row)
    if unknown:
        raise ContractError(
            ReasonCode.FORBIDDEN_MODEL_FACT,
            f"模型输出含禁止/未知字段：{sorted(unknown)}",
        )
    if missing:
        raise ContractError(
            ReasonCode.INVALID_MODEL_OUTPUT,
            f"模型输出缺少字段：{sorted(missing)}",
        )
    if row["schema_version"] != MODEL_OUTPUT_SCHEMA_VERSION:
        raise ContractError(
            ReasonCode.INVALID_MODEL_OUTPUT,
            "模型输出 schema_version 不匹配",
        )
    try:
        target_status = TargetStatus(row["target_status"])
        sufficiency = EvidenceSufficiency(row["evidence_sufficiency"])
    except (TypeError, ValueError) as error:
        raise ContractError(
            ReasonCode.INVALID_ENUM,
            "模型输出 enum 非法",
        ) from error
    description = row["description"]
    if not isinstance(description, str) or not description.strip():
        raise ContractError(
            ReasonCode.TYPE_MISMATCH,
            "description 必须是非空字符串",
        )
    _reject_model_geometry_language(description, location="description")
    answer = row["answer"]
    if answer is not None and (
        not isinstance(answer, str) or not answer.strip()
    ):
        raise ContractError(
            ReasonCode.TYPE_MISMATCH,
            "answer 必须是非空字符串或 null",
        )
    if answer is not None:
        _reject_model_geometry_language(answer, location="answer")
    evidence_ids = frozenset(str(item) for item in valid_evidence_ids)
    claims_value = row["claims"]
    if not isinstance(claims_value, list):
        raise ContractError(
            ReasonCode.TYPE_MISMATCH,
            "claims 必须是列表",
        )
    claims: list[Claim] = []
    for index, claim_value in enumerate(claims_value):
        if (
            not isinstance(claim_value, dict)
            or set(claim_value) != {"text", "evidence_ids"}
        ):
            raise ContractError(
                ReasonCode.UNKNOWN_FIELD,
                f"claims[{index}] 只允许 text/evidence_ids",
            )
        text = claim_value["text"]
        if not isinstance(text, str) or not text.strip():
            raise ContractError(
                ReasonCode.TYPE_MISMATCH,
                f"claims[{index}].text 必须非空",
            )
        _reject_model_geometry_language(
            text,
            location=f"claims[{index}].text",
        )
        claim_ids = ensure_unique_strings(
            claim_value["evidence_ids"],
            location=f"claims[{index}].evidence_ids",
            allow_empty=False,
        )
        unknown_ids = sorted(set(claim_ids) - evidence_ids)
        if unknown_ids:
            raise ContractError(
                ReasonCode.EVIDENCE_REFERENCE_INVALID,
                f"claims[{index}] 引用了未知 evidence：{unknown_ids}",
            )
        claims.append(Claim(text=text.strip(), evidence_ids=claim_ids))
    limitations = ensure_unique_strings(
        row["limitations"],
        location="limitations",
    )
    return StructuredModelOutput(
        schema_version=MODEL_OUTPUT_SCHEMA_VERSION,
        target_status=target_status,
        description=description.strip(),
        answer=None if answer is None else answer.strip(),
        evidence_sufficiency=sufficiency,
        claims=tuple(claims),
        limitations=limitations,
    )


def serialize_model_output(value: StructuredModelOutput) -> str:
    return canonical_json(value.to_dict())


def prediction_row(
    *,
    record_id: str,
    parent_id: str,
    logical_role: str,
    task_family: str,
    mask_mode: MaskMode,
    model_output: StructuredModelOutput,
    reference_responses: Iterable[str],
    evidence_ids: Iterable[str],
    provenance: Mapping[str, Any],
    counterfactual: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if any(
        not isinstance(value, str) or not value.strip()
        for value in (record_id, parent_id, logical_role, task_family)
    ):
        raise PredictionError(
            ReasonCode.PREDICTION_INVALID,
            "prediction id/role/task 必须是非空字符串",
        )
    if mask_mode is MaskMode.EXTERNAL_GENERIC:
        raise PredictionError(
            ReasonCode.EXTERNAL_MASK_FORBIDDEN,
            "mask-grounded prediction_row 不接受 external_generic",
        )
    reference_values = ensure_unique_strings(
        list(reference_responses),
        location="prediction.reference_responses",
        allow_empty=False,
    )
    evidence_values = ensure_unique_strings(
        list(evidence_ids),
        location="prediction.evidence_ids",
    )
    parsed = parse_model_output(
        model_output.to_dict(),
        valid_evidence_ids=evidence_values,
    )
    if parsed != model_output:
        raise PredictionError(
            ReasonCode.PREDICTION_INVALID,
            "model_output 不是 parser 规范化后的严格对象",
        )
    if not isinstance(provenance, Mapping):
        raise PredictionError(
            ReasonCode.TYPE_MISMATCH,
            "prediction provenance 必须是对象",
        )
    if counterfactual is not None and not isinstance(counterfactual, Mapping):
        raise PredictionError(
            ReasonCode.TYPE_MISMATCH,
            "prediction counterfactual 必须是对象或 null",
        )
    return {
        "schema_version": PREDICTION_SCHEMA_VERSION,
        "record_id": record_id,
        "parent_id": parent_id,
        "logical_role": logical_role,
        "task_family": task_family,
        "mask_mode": mask_mode.value,
        "generated_text": serialize_model_output(model_output),
        "model_output": model_output.to_dict(),
        "reference_responses": list(reference_values),
        "evidence_ids": list(evidence_values),
        "provenance": dict(provenance),
        "counterfactual": None if counterfactual is None else dict(counterfactual),
    }


def generic_prediction_row(
    *,
    record_id: str,
    parent_id: str,
    logical_role: str,
    task_family: str,
    generated_text: str,
    reference_responses: Iterable[str],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        not record_id
        or not parent_id
        or not isinstance(generated_text, str)
        or not generated_text.strip()
    ):
        raise PredictionError(
            ReasonCode.PREDICTION_INVALID,
            "generic prediction id/text 不能为空",
        )
    if any(
        not isinstance(value, str) or not value.strip()
        for value in (record_id, parent_id, logical_role, task_family)
    ):
        raise PredictionError(
            ReasonCode.PREDICTION_INVALID,
            "generic prediction role/task 不能为空",
        )
    reference_values = ensure_unique_strings(
        list(reference_responses),
        location="generic_prediction.reference_responses",
        allow_empty=False,
    )
    if not isinstance(provenance, Mapping):
        raise PredictionError(
            ReasonCode.TYPE_MISMATCH,
            "generic prediction provenance 必须是对象",
        )
    return {
        "schema_version": PREDICTION_SCHEMA_VERSION,
        "record_id": record_id,
        "parent_id": parent_id,
        "logical_role": logical_role,
        "task_family": task_family,
        "mask_mode": MaskMode.EXTERNAL_GENERIC.value,
        "generated_text": generated_text.strip(),
        "model_output": None,
        "reference_responses": list(reference_values),
        "evidence_ids": [],
        "provenance": dict(provenance),
        "counterfactual": None,
    }


def failure_row(
    *,
    record_id: str,
    parent_id: str,
    stage: str,
    code: ReasonCode,
    message: str,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": FAILURE_SCHEMA_VERSION,
        "record_id": record_id,
        "parent_id": parent_id,
        "stage": stage,
        "reason_code": code.value,
        "message": message,
        "details": dict(details or {}),
    }


# Stage 4 v2 使用独立 schema；不得改变上方 generic/Gate B v1 parser 的语义。
REGION_OUTPUT_SCHEMA_VERSION = "rs_vlm.mask_grounded_region_output.v2"
REGION_PREDICTION_SCHEMA_VERSION = "rs_vlm.mask_grounded_region_prediction.v1"
REGION_FAILURE_SCHEMA_VERSION = "rs_vlm.mask_grounded_region_failure.v1"
REGION_PROVENANCE_SCHEMA_VERSION = "rs_vlm.mask_grounded_region_provenance.v1"


class RegionEvidenceSufficiency(StrEnum):
    SUFFICIENT = "sufficient"
    LIMITED = "limited"
    INSUFFICIENT = "insufficient"
    NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True)
class RegionDescriptionOutput:
    schema_version: str
    target_status: TargetStatus
    target_appearance: Mapping[str, str]
    target_morphology: Mapping[str, str]
    surrounding_environment: Mapping[str, tuple[str, ...]]
    region_context_contrast: Mapping[str, Any]
    possible_confusers: tuple[str, ...]
    evidence_sufficiency: RegionEvidenceSufficiency
    short_summary: str
    limitations: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "target_status": self.target_status.value,
            "target_appearance": dict(self.target_appearance),
            "target_morphology": dict(self.target_morphology),
            "surrounding_environment": {
                key: list(value) for key, value in self.surrounding_environment.items()
            },
            "region_context_contrast": {
                key: list(value) if isinstance(value, tuple) else value
                for key, value in self.region_context_contrast.items()
            },
            "possible_confusers": list(self.possible_confusers),
            "evidence_sufficiency": self.evidence_sufficiency.value,
            "short_summary": self.short_summary,
            "limitations": list(self.limitations),
        }


_REGION_TOP_LEVEL = {
    "schema_version", "target_status", "target_appearance", "target_morphology",
    "surrounding_environment", "region_context_contrast", "possible_confusers",
    "evidence_sufficiency", "short_summary", "limitations",
}
_APPEARANCE_FIELDS = {
    "tone", "texture", "vegetation_or_exposure", "homogeneity", "boundary_visibility",
}
_MORPHOLOGY_FIELDS = {"shape", "fragmentation", "qualitative_orientation"}
_SURROUNDING_FIELDS = {
    "land_cover", "nearby_objects", "visible_terrain_context", "human_disturbance",
}
_CONTRAST_FIELDS = {
    "tone_contrast", "texture_contrast", "vegetation_contrast", "boundary_transition", "adjacency",
}

_FORBIDDEN_CLAIM_PATTERNS: Mapping[str, tuple[str, ...]] = {
    "event_time": (r"发生时间", r"\b(?:occurred|happened)\s+(?:in|on|at)\b"),
    "trigger_cause": (r"(?:降雨|地震|施工|工程活动).{0,8}(?:导致|触发|引起)", r"\btriggered by\b", r"\bcaused by\b"),
    "precise_motion": (r"(?:位移|速度).{0,8}\d", r"\b\d+(?:\.\d+)?\s*(?:mm|cm|m)/(?:day|year|s)\b"),
    "movement_direction": (r"(?:滑动|运动|位移)方向", r"\bmovement direction\b"),
    "stability": (r"(?:稳定|不稳定)性?(?:等级|状态|为)", r"\bstability (?:class|level|is)\b"),
    "risk_level": (r"(?:危险性|风险)(?:等级|为|是)", r"\b(?:risk|hazard) (?:class|level|is)\b"),
    "disaster_scale": (r"灾害规模(?:等级|为|是)", r"\bdisaster scale\b"),
    "actual_threat": (r"(?:威胁|危及).{0,8}(?:人员|道路|建筑)", r"\bthreatens? (?:people|roads?|buildings?)\b"),
    "field_geology": (r"(?:地层|岩性|内部结构)(?:为|是|由)", r"\b(?:stratigraphy|lithology|internal structure) is\b"),
}
_UNCERTAINTY_MARKERS = (
    "无法判断", "不能判断", "无法确定", "不能确定", "证据不足", "不可见", "未提供",
    "cannot determine", "cannot infer", "insufficient evidence", "not visible", "unknown",
)


def detect_forbidden_region_claims(value: Any) -> tuple[str, ...]:
    """检测肯定式禁区结论；明确写成 limitation 的不确定陈述允许保留。"""

    texts: list[str] = []

    def collect(child: Any) -> None:
        if isinstance(child, str):
            texts.append(child)
        elif isinstance(child, Mapping):
            for nested in child.values():
                collect(nested)
        elif isinstance(child, (list, tuple)):
            for nested in child:
                collect(nested)

    collect(value)
    violations: set[str] = set()
    for text in texts:
        lowered = text.lower()
        uncertain = any(marker in lowered for marker in _UNCERTAINTY_MARKERS)
        if uncertain:
            continue
        for code, patterns in _FORBIDDEN_CLAIM_PATTERNS.items():
            if any(re.search(pattern, lowered, flags=re.IGNORECASE) for pattern in patterns):
                violations.add(code)
    return tuple(sorted(violations))


def _region_mapping(value: Any, *, fields: set[str], location: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(ReasonCode.TYPE_MISMATCH, f"{location}: 必须是对象")
    row = dict(value)
    unknown, missing = set(row) - fields, fields - set(row)
    if unknown:
        raise ContractError(ReasonCode.UNKNOWN_FIELD, f"{location}: 未知字段 {sorted(unknown)}")
    if missing:
        raise ContractError(ReasonCode.INVALID_MODEL_OUTPUT, f"{location}: 缺少字段 {sorted(missing)}")
    return row


def _region_text(value: Any, *, location: str, reject_program_geometry: bool = True) -> str:
    if not isinstance(value, str) or not value.strip() or len(value.strip()) > 2000:
        raise ContractError(ReasonCode.TYPE_MISMATCH, f"{location}: 必须是长度受限的非空字符串")
    text = " ".join(value.split())
    if reject_program_geometry:
        _reject_model_geometry_language(text, location=location)
    return text


def _region_list(value: Any, *, location: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ContractError(ReasonCode.TYPE_MISMATCH, f"{location}: 必须是列表")
    result = tuple(_region_text(item, location=f"{location}[]") for item in value)
    if len(result) != len(set(result)) or len(result) > 64:
        raise ContractError(ReasonCode.INVALID_MODEL_OUTPUT, f"{location}: 列表重复或过长")
    return result


def parse_region_model_output(value: str | Mapping[str, Any]) -> RegionDescriptionOutput:
    if isinstance(value, str):
        try:
            parsed = json.loads(value, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
        except (json.JSONDecodeError, ValueError) as error:
            raise ContractError(
                ReasonCode.INVALID_MODEL_OUTPUT,
                "Stage 4 v2 输出不是严格 JSON",
                details={"error": str(error)},
            ) from error
    elif isinstance(value, Mapping):
        parsed = dict(value)
    else:
        raise ContractError(ReasonCode.TYPE_MISMATCH, "Stage 4 v2 输出必须是 JSON 字符串或对象")
    row = _region_mapping(parsed, fields=_REGION_TOP_LEVEL, location="$")
    if row["schema_version"] != REGION_OUTPUT_SCHEMA_VERSION:
        raise ContractError(ReasonCode.INVALID_MODEL_OUTPUT, "Stage 4 v2 schema_version 不匹配")
    try:
        target_status = TargetStatus(row["target_status"])
        sufficiency = RegionEvidenceSufficiency(row["evidence_sufficiency"])
    except (TypeError, ValueError) as error:
        raise ContractError(ReasonCode.INVALID_ENUM, "Stage 4 v2 enum 非法") from error
    appearance_row = _region_mapping(row["target_appearance"], fields=_APPEARANCE_FIELDS, location="$.target_appearance")
    morphology_row = _region_mapping(row["target_morphology"], fields=_MORPHOLOGY_FIELDS, location="$.target_morphology")
    surrounding_row = _region_mapping(row["surrounding_environment"], fields=_SURROUNDING_FIELDS, location="$.surrounding_environment")
    contrast_row = _region_mapping(row["region_context_contrast"], fields=_CONTRAST_FIELDS, location="$.region_context_contrast")
    appearance = {key: _region_text(appearance_row[key], location=f"$.target_appearance.{key}") for key in sorted(_APPEARANCE_FIELDS)}
    morphology = {
        key: _region_text(
            morphology_row[key],
            location=f"$.target_morphology.{key}",
            # 定性方向是 Stage 4 人工/VLM 字段；这里只豁免自然语言方向，schema 仍无
            # bbox、centroid、数值 elongation 等程序几何入口。
            reject_program_geometry=key != "qualitative_orientation",
        )
        for key in sorted(_MORPHOLOGY_FIELDS)
    }
    surrounding = {key: _region_list(surrounding_row[key], location=f"$.surrounding_environment.{key}") for key in sorted(_SURROUNDING_FIELDS)}
    contrast: dict[str, Any] = {}
    for key in sorted(_CONTRAST_FIELDS):
        contrast[key] = (
            _region_list(contrast_row[key], location=f"$.region_context_contrast.{key}")
            if key == "adjacency"
            else _region_text(contrast_row[key], location=f"$.region_context_contrast.{key}")
        )
    confusers = _region_list(row["possible_confusers"], location="$.possible_confusers")
    summary = _region_text(row["short_summary"], location="$.short_summary")
    limitations = _region_list(row["limitations"], location="$.limitations")
    if target_status is TargetStatus.NO_TARGET:
        if any(text != "not_applicable" for text in appearance.values()) or any(
            text != "not_applicable" for text in morphology.values()
        ):
            raise ContractError(ReasonCode.INVALID_MODEL_OUTPUT, "no-target 的 target 字段必须为 not_applicable")
        if any(surrounding.values()) or confusers or contrast["adjacency"]:
            raise ContractError(ReasonCode.INVALID_MODEL_OUTPUT, "no-target 的区域列表必须为空")
        if any(contrast[key] != "not_applicable" for key in contrast if key != "adjacency"):
            raise ContractError(ReasonCode.INVALID_MODEL_OUTPUT, "no-target 的 contrast 字段必须为 not_applicable")
        if sufficiency not in {RegionEvidenceSufficiency.INSUFFICIENT, RegionEvidenceSufficiency.NOT_APPLICABLE}:
            raise ContractError(ReasonCode.INVALID_MODEL_OUTPUT, "no-target evidence_sufficiency 非法")
        if not limitations:
            raise ContractError(ReasonCode.INVALID_MODEL_OUTPUT, "no-target 必须解释 empty/no-target limitation")
    result = RegionDescriptionOutput(
        schema_version=REGION_OUTPUT_SCHEMA_VERSION,
        target_status=target_status,
        target_appearance=appearance,
        target_morphology=morphology,
        surrounding_environment=surrounding,
        region_context_contrast=contrast,
        possible_confusers=confusers,
        evidence_sufficiency=sufficiency,
        short_summary=summary,
        limitations=limitations,
    )
    violations = detect_forbidden_region_claims(result.to_dict())
    if violations:
        raise ContractError(
            ReasonCode.FORBIDDEN_CLAIM,
            f"Stage 4 v2 含不受图像支持的肯定式结论：{list(violations)}",
        )
    return result


def serialize_region_model_output(value: RegionDescriptionOutput) -> str:
    parsed = parse_region_model_output(value.to_dict())
    return canonical_json(parsed.to_dict())


def region_provenance_row(
    *,
    record_id: str,
    asset_manifest_sha256: str,
    asset_identity_sha256: str,
    representation_mode: str,
    formal_model_input_roles: Iterable[str],
    prompt_sha256: str,
    generation: Mapping[str, Any],
) -> dict[str, Any]:
    roles = tuple(formal_model_input_roles)
    if "audit_overlay" in roles:
        raise PredictionError(ReasonCode.ASSET_ROLE_LEAKAGE, "provenance formal roles 禁止 audit_overlay")
    return {
        "schema_version": REGION_PROVENANCE_SCHEMA_VERSION,
        "record_id": record_id,
        "asset_manifest_sha256": asset_manifest_sha256,
        "asset_identity_sha256": asset_identity_sha256,
        "representation_mode": representation_mode,
        "formal_model_input_roles": list(roles),
        "prompt_sha256": prompt_sha256,
        "generation": dict(generation),
    }


def region_prediction_row(
    *,
    record_id: str,
    parent_id: str,
    output: RegionDescriptionOutput,
    provenance: Mapping[str, Any],
    counterfactual_group_id: str | None = None,
) -> dict[str, Any]:
    if not record_id or not parent_id or provenance.get("schema_version") != REGION_PROVENANCE_SCHEMA_VERSION:
        raise PredictionError(ReasonCode.PREDICTION_INVALID, "Region prediction identity/provenance 非法")
    parsed = parse_region_model_output(output.to_dict())
    return {
        "schema_version": REGION_PREDICTION_SCHEMA_VERSION,
        "record_id": record_id,
        "parent_id": parent_id,
        "model_output": parsed.to_dict(),
        "generated_text": canonical_json(parsed.to_dict()),
        "provenance": dict(provenance),
        "counterfactual_group_id": counterfactual_group_id,
    }


def region_failure_row(
    *,
    record_id: str,
    parent_id: str,
    stage: str,
    code: ReasonCode,
    message: str,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": REGION_FAILURE_SCHEMA_VERSION,
        "record_id": record_id,
        "parent_id": parent_id,
        "stage": stage,
        "reason_code": code.value,
        "message": message,
        "details": dict(details or {}),
    }
