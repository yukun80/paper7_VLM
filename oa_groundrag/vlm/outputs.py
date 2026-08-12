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


def _region_target_status(value: str | TargetStatus) -> TargetStatus:
    """规范化模板所用 target status；模板和严格 parser 使用同一枚举。"""

    try:
        return value if isinstance(value, TargetStatus) else TargetStatus(value)
    except (TypeError, ValueError) as error:
        raise ContractError(
            ReasonCode.INVALID_ENUM,
            "Stage 4 v2 模板 target_status 非法",
        ) from error


def region_output_template(
    target_status: str | TargetStatus,
) -> dict[str, Any]:
    """返回可通过严格 parser 的独立空模板，仅供人工编辑器和生成失败回退。

    正式模型消息不得包含这份完整答案，否则 greedy decoding 容易逐字复制模板。这里的
    字符串是待专家替换的保守占位值，不含 bbox、面积或质心等程序事实。每次调用均返回
    新对象，避免 UI 编辑时污染后续记录。
    """

    status = _region_target_status(target_status)
    if status is TargetStatus.NO_TARGET:
        return {
            "schema_version": REGION_OUTPUT_SCHEMA_VERSION,
            "target_status": status.value,
            "target_appearance": {
                "tone": "not_applicable",
                "texture": "not_applicable",
                "vegetation_or_exposure": "not_applicable",
                "homogeneity": "not_applicable",
                "boundary_visibility": "not_applicable",
            },
            "target_morphology": {
                "shape": "not_applicable",
                "fragmentation": "not_applicable",
                "qualitative_orientation": "not_applicable",
            },
            "surrounding_environment": {
                "land_cover": [],
                "nearby_objects": [],
                "visible_terrain_context": [],
                "human_disturbance": [],
            },
            "region_context_contrast": {
                "tone_contrast": "not_applicable",
                "texture_contrast": "not_applicable",
                "vegetation_contrast": "not_applicable",
                "boundary_transition": "not_applicable",
                "adjacency": [],
            },
            "possible_confusers": [],
            "evidence_sufficiency": RegionEvidenceSufficiency.INSUFFICIENT.value,
            "short_summary": "二值 mask 未指定可描述的目标区域。",
            "limitations": ["空 mask 不提供目标区域，无法生成目标外观或形态描述。"],
        }
    return {
        "schema_version": REGION_OUTPUT_SCHEMA_VERSION,
        "target_status": status.value,
        "target_appearance": {
            "tone": "无法判断",
            "texture": "无法判断",
            "vegetation_or_exposure": "无法判断",
            "homogeneity": "无法判断",
            "boundary_visibility": "无法判断",
        },
        "target_morphology": {
            "shape": "无法判断",
            "fragmentation": "无法判断",
            "qualitative_orientation": "无法判断",
        },
        "surrounding_environment": {
            "land_cover": [],
            "nearby_objects": [],
            "visible_terrain_context": [],
            "human_disturbance": [],
        },
        "region_context_contrast": {
            "tone_contrast": "无法判断",
            "texture_contrast": "无法判断",
            "vegetation_contrast": "无法判断",
            "boundary_transition": "无法判断",
            "adjacency": [],
        },
        "possible_confusers": [],
        "evidence_sufficiency": RegionEvidenceSufficiency.INSUFFICIENT.value,
        "short_summary": "mask 指定区域尚待依据当前影像核验。",
        "limitations": ["当前影像不足以支持更具体的视觉描述。"],
    }


def region_output_contract(
    target_status: str | TargetStatus,
) -> dict[str, Any]:
    """返回供 VLM 阅读的 Stage 4 v2 字段、类型、枚举和目标状态合同。

    合同只描述结构，不携带一份可直接提交的答案。保守空模板由
    :func:`region_output_template` 独立提供给人工编辑器和失败回退，避免 VLM 在
    greedy decoding 下把模板复制成看似合法、实则没有视觉信息的草稿。
    """

    status = _region_target_status(target_status)

    def string_fields(fields: Iterable[str]) -> dict[str, dict[str, Any]]:
        return {
            field: {"type": "string", "non_empty": True} for field in fields
        }

    def list_fields(fields: Iterable[str]) -> dict[str, dict[str, Any]]:
        return {
            field: {
                "type": "array",
                "items": "non_empty_string",
                "unique": True,
            }
            for field in fields
        }

    evidence_values = [item.value for item in RegionEvidenceSufficiency]
    if status is TargetStatus.NO_TARGET:
        evidence_values = [
            RegionEvidenceSufficiency.INSUFFICIENT.value,
            RegionEvidenceSufficiency.NOT_APPLICABLE.value,
        ]
    return {
        "type": "object",
        "additional_properties": False,
        "required": sorted(_REGION_TOP_LEVEL),
        "properties": {
            "schema_version": {
                "type": "string",
                "const": REGION_OUTPUT_SCHEMA_VERSION,
            },
            "target_status": {"type": "string", "const": status.value},
            "target_appearance": {
                "type": "object",
                "additional_properties": False,
                "required": sorted(_APPEARANCE_FIELDS),
                "properties": string_fields(sorted(_APPEARANCE_FIELDS)),
            },
            "target_morphology": {
                "type": "object",
                "additional_properties": False,
                "required": sorted(_MORPHOLOGY_FIELDS),
                "properties": string_fields(sorted(_MORPHOLOGY_FIELDS)),
            },
            "surrounding_environment": {
                "type": "object",
                "additional_properties": False,
                "required": sorted(_SURROUNDING_FIELDS),
                "properties": list_fields(sorted(_SURROUNDING_FIELDS)),
            },
            "region_context_contrast": {
                "type": "object",
                "additional_properties": False,
                "required": sorted(_CONTRAST_FIELDS),
                "properties": {
                    **string_fields(sorted(_CONTRAST_FIELDS - {"adjacency"})),
                    "adjacency": {
                        "type": "array",
                        "items": "non_empty_string",
                        "unique": True,
                    },
                },
            },
            "possible_confusers": {
                "type": "array",
                "items": "non_empty_string",
                "unique": True,
            },
            "evidence_sufficiency": {
                "type": "string",
                "enum": evidence_values,
                "values_must_remain_ascii_and_must_not_be_translated": True,
            },
            "short_summary": {"type": "string", "non_empty": True},
            "limitations": {
                "type": "array",
                "items": "non_empty_string",
                "unique": True,
                "min_items": 1 if status is TargetStatus.NO_TARGET else 0,
            },
        },
        "target_specific_rules": (
            {
                "target_scalar_fields": "all_exactly_not_applicable",
                "region_arrays": "all_empty",
                "limitations": "must_explain_empty_or_no_target_mask",
            }
            if status is TargetStatus.NO_TARGET
            else {
                "unknown_scalar_value": "无法判断",
                "not_visible_array_value": [],
                "insufficient_evidence": "explain_in_limitations",
            }
        ),
    }

_FORBIDDEN_CLAIM_PATTERNS: Mapping[str, tuple[str, ...]] = {
    "event_time": (r"发生时间", r"\b(?:occurred|happened)\s+(?:in|on|at)\b"),
    "trigger_cause": (
        r"(?:降雨|暴雨|强降雨|地震|施工|工程活动).{0,8}(?:导致|触发|引起)",
        r"\btriggered by\b",
        r"\bcaused by\b",
    ),
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
_UNCERTAINTY_BREAKERS = ("但", "但是", "然而", "不过", "but", "however", "yet")


def _claim_is_locally_negated(text: str, start: int) -> bool:
    """只豁免紧邻禁区断言的无法判断表述，防止用转折词夹带肯定结论。"""

    prefix = text[max(0, start - 64):start]
    positions = [
        (prefix.rfind(marker), marker)
        for marker in _UNCERTAINTY_MARKERS
        if prefix.rfind(marker) >= 0
    ]
    if not positions:
        return False
    position, marker = max(positions, key=lambda item: item[0])
    between = prefix[position + len(marker):]
    return not any(breaker in between for breaker in _UNCERTAINTY_BREAKERS)


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
        for code, patterns in _FORBIDDEN_CLAIM_PATTERNS.items():
            for pattern in patterns:
                for match in re.finditer(pattern, lowered, flags=re.IGNORECASE):
                    if not _claim_is_locally_negated(lowered, match.start()):
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


class RegionDraftQualityStatus(StrEnum):
    """单次模型草稿的确定性语义信息量状态；不进入专家最终 annotation。"""

    INFORMATIVE = "informative"
    LIMITED_BUT_SPECIFIC = "limited_but_specific"
    LOW_INFORMATION = "low_information"
    NOT_APPLICABLE_NO_TARGET = "not_applicable_no_target"


@dataclass(frozen=True)
class RegionDraftQualityAssessment:
    """可重算的草稿质量诊断，用于区分 JSON 合法与视觉描述有信息。"""

    status: RegionDraftQualityStatus
    issues: tuple[str, ...]
    metrics: Mapping[str, int | bool]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "issues": list(self.issues),
            "metrics": dict(self.metrics),
        }


_DRAFT_PLACEHOLDER_VALUES = frozenset({
    "无法判断", "不能判断", "无法确定", "不能确定", "不可见", "不确定",
    "unknown", "not_applicable", "not applicable", "cannot determine",
    "cannot infer", "not visible",
})
_DRAFT_GENERIC_LIMITATION_VALUES = frozenset({
    "当前影像不足以支持更具体的视觉描述。",
    "当前影像证据不足。",
    "证据不足。",
    "无法判断。",
    "the current image is insufficient.",
    "insufficient evidence.",
    "cannot determine.",
})
_SPECIFIC_VISIBILITY_MARKERS = (
    "目标过小", "区域过小", "尺寸过小", "分辨率", "像素", "模糊", "遮挡",
    "阴影", "云", "低对比", "对比度低", "边界不清", "裁剪过窄", "长宽比",
    "不可分辨", "无法看清", "too small", "low resolution", "blur", "occlusion",
    "shadow", "cloud", "low contrast", "narrow crop", "not discernible",
)


def _draft_text_is_placeholder(value: str) -> bool:
    normalized = " ".join(value.strip().lower().split())
    if normalized in _DRAFT_PLACEHOLDER_VALUES:
        return True
    placeholder_prefixes = tuple(
        marker for marker in _DRAFT_PLACEHOLDER_VALUES
        if marker not in {"not_applicable", "not applicable"}
    )
    return (
        normalized.startswith(placeholder_prefixes)
        and not any(marker in normalized for marker in _SPECIFIC_VISIBILITY_MARKERS)
    )


def assess_region_draft_quality(
    value: str | Mapping[str, Any] | RegionDescriptionOutput,
) -> RegionDraftQualityAssessment:
    """重算单条草稿的视觉信息量，不把 schema valid 误写成 informative。

    target 草稿必须至少给出目标观察（或具体可见性限制）、场景环境和区域—环境
    对比（或具体不可判断原因）。no-target 的目标拒绝是合同要求，不因与空模板一致
    而判为低信息。
    """

    parsed = value if isinstance(value, RegionDescriptionOutput) else parse_region_model_output(value)
    row = parsed.to_dict()
    template = region_output_template(parsed.target_status)
    template_match = canonical_json(row) == canonical_json(template)
    summary_is_default = row["short_summary"] == template["short_summary"]
    target_scalars = [
        *row["target_appearance"].values(),
        *row["target_morphology"].values(),
    ]
    contrast_scalars = [
        row["region_context_contrast"][key]
        for key in sorted(_CONTRAST_FIELDS - {"adjacency"})
    ]
    target_observations = sum(
        not _draft_text_is_placeholder(text) for text in target_scalars
    )
    contrast_observations = sum(
        not _draft_text_is_placeholder(text) for text in contrast_scalars
    ) + len(row["region_context_contrast"]["adjacency"])
    environment_items = sum(
        len(items) for items in row["surrounding_environment"].values()
    )
    placeholder_scalars = sum(
        _draft_text_is_placeholder(text)
        for text in (*target_scalars, *contrast_scalars)
    )
    generic_limitation = template["limitations"][0]
    generic_limitations = [
        text
        for text in row["limitations"]
        if text.strip().lower() in _DRAFT_GENERIC_LIMITATION_VALUES
        or text == generic_limitation
    ]
    specific_limitations = [
        text
        for text in row["limitations"]
        if text != generic_limitation
        and any(marker in text.lower() for marker in _SPECIFIC_VISIBILITY_MARKERS)
    ]
    metrics: dict[str, int | bool] = {
        "template_match": template_match,
        "summary_is_default": summary_is_default,
        "target_observation_count": target_observations,
        "environment_item_count": environment_items,
        "contrast_observation_count": contrast_observations,
        "specific_limitation_count": len(specific_limitations),
        "generic_limitation_count": len(generic_limitations),
        "placeholder_scalar_count": placeholder_scalars,
        "descriptive_scalar_count": target_observations + contrast_observations,
        "descriptive_scalar_total": len(target_scalars) + len(contrast_scalars),
        "scene_array_item_count": environment_items,
    }
    if parsed.target_status is TargetStatus.NO_TARGET:
        return RegionDraftQualityAssessment(
            status=RegionDraftQualityStatus.NOT_APPLICABLE_NO_TARGET,
            issues=(),
            metrics=metrics,
        )

    issues: list[str] = []
    if template_match:
        issues.append("template_copy")
    if summary_is_default:
        issues.append("default_summary")
    if target_observations == 0 and not specific_limitations:
        issues.append("missing_target_observation_or_specific_limitation")
    if environment_items == 0:
        issues.append("missing_environment_observation")
    if contrast_observations == 0 and not specific_limitations:
        issues.append("missing_contrast_observation_or_specific_limitation")
    if row["limitations"] and len(generic_limitations) == len(row["limitations"]):
        issues.append("generic_limitation_only")
    if issues:
        return RegionDraftQualityAssessment(
            status=RegionDraftQualityStatus.LOW_INFORMATION,
            issues=tuple(issues),
            metrics=metrics,
        )
    limited = (
        placeholder_scalars > 0
        or bool(specific_limitations)
        or parsed.evidence_sufficiency is not RegionEvidenceSufficiency.SUFFICIENT
    )
    return RegionDraftQualityAssessment(
        status=(
            RegionDraftQualityStatus.LIMITED_BUT_SPECIFIC
            if limited
            else RegionDraftQualityStatus.INFORMATIVE
        ),
        issues=(),
        metrics=metrics,
    )


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
