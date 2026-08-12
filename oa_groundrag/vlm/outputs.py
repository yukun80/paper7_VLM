"""Shared VLM 的严格结构化输出、prediction 与 failure 合同。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Iterable, Mapping

from oa_groundrag.artifacts.identity import canonical_json

from oa_groundrag.grounding.contracts import (
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
from oa_groundrag.vlm.errors import ContractError, PredictionError, ReasonCode


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
