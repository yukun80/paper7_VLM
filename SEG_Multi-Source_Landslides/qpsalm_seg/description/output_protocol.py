#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Raw parse, schema validation and separately reported deterministic repair."""

from __future__ import annotations

from dataclasses import dataclass
import copy
from functools import lru_cache
import json
import math
import re
from typing import Any

try:
    from jsonschema import Draft202012Validator
except ModuleNotFoundError:  # 固定输出协议提供等价内置校验，避免包导入阶段失败。
    Draft202012Validator = None  # type: ignore[assignment,misc]

from qpsalm_seg.paths import REPO_ROOT

from .json_protocol import strict_json_loads


OUTPUT_SCHEMA_VERSION = "qpsalm_description_output_v1"


@dataclass
class ParsedDescription:
    raw_text: str
    parsed: dict[str, Any] | None
    parse_errors: tuple[str, ...]
    schema_valid: bool
    repaired: dict[str, Any]
    repair_actions: tuple[str, ...]


@dataclass(frozen=True)
class _SchemaIssue:
    path: tuple[str, ...]
    message: str


@lru_cache(maxsize=1)
def _schema() -> dict[str, Any]:
    path = REPO_ROOT / "configs/qpsalm_description_output_v1.schema.json"
    return strict_json_loads(path.read_text(encoding="utf-8"))


def _matches_type(value: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "number":
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        )
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "null":
        return value is None
    raise ValueError(f"unsupported built-in JSON schema type: {expected}")


def _builtin_schema_issues(
    value: Any,
    schema: dict[str, Any],
    path: tuple[str, ...] = (),
) -> list[_SchemaIssue]:
    """Validate the keyword subset used by qpsalm_description_output_v1."""
    issues: list[_SchemaIssue] = []
    expected_type = schema.get("type")
    if isinstance(expected_type, str) and not _matches_type(value, expected_type):
        return [_SchemaIssue(path, f"expected {expected_type}, got {type(value).__name__}")]
    # 固定 output schema 使用条件约束阻止 absent 样本继续声明滑坡属性。
    # `if` 本身只决定分支，其不匹配原因不能泄漏为 validation error。
    for subschema in schema.get("allOf") or []:
        if not isinstance(subschema, dict):
            raise ValueError("built-in JSON schema allOf members must be objects")
        issues.extend(_builtin_schema_issues(value, subschema, path))
    condition = schema.get("if")
    if condition is not None:
        if not isinstance(condition, dict):
            raise ValueError("built-in JSON schema if must be an object")
        branch_name = "then" if not _builtin_schema_issues(value, condition, path) else "else"
        branch = schema.get(branch_name)
        if branch is not None:
            if not isinstance(branch, dict):
                raise ValueError(f"built-in JSON schema {branch_name} must be an object")
            issues.extend(_builtin_schema_issues(value, branch, path))
    if "const" in schema and value != schema["const"]:
        issues.append(_SchemaIssue(path, f"value must equal {schema['const']!r}"))
    if "enum" in schema and value not in schema["enum"]:
        issues.append(_SchemaIssue(path, f"value is not one of {schema['enum']!r}"))
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            issues.append(_SchemaIssue(path, f"value is below minimum {schema['minimum']}"))
        if "maximum" in schema and value > schema["maximum"]:
            issues.append(_SchemaIssue(path, f"value is above maximum {schema['maximum']}"))
    if isinstance(value, str):
        if "minLength" in schema and len(value) < int(schema["minLength"]):
            issues.append(_SchemaIssue(path, f"string is shorter than minLength {schema['minLength']}"))
        if "maxLength" in schema and len(value) > int(schema["maxLength"]):
            issues.append(_SchemaIssue(path, f"string is longer than maxLength {schema['maxLength']}"))
    if isinstance(value, dict):
        properties = dict(schema.get("properties") or {})
        for field in schema.get("required") or []:
            if field not in value:
                issues.append(_SchemaIssue((*path, str(field)), "required property is missing"))
        if schema.get("additionalProperties") is False:
            for field in sorted(set(value) - set(properties)):
                issues.append(_SchemaIssue((*path, str(field)), "additional property is not allowed"))
        for field, field_schema in properties.items():
            if field in value:
                issues.extend(
                    _builtin_schema_issues(value[field], field_schema, (*path, str(field)))
                )
    return issues


def _validation_errors(value: dict[str, Any]) -> list[str]:
    schema = _schema()
    if Draft202012Validator is not None:
        validator = Draft202012Validator(schema)
        return [
            f"schema:{'.'.join(str(item) for item in error.absolute_path)}:{error.message}"
            for error in sorted(
                validator.iter_errors(value),
                key=lambda error: [str(item) for item in error.absolute_path],
            )
        ]
    return [
        f"schema:{'.'.join(issue.path)}:{issue.message}"
        for issue in sorted(_builtin_schema_issues(value, schema), key=lambda issue: issue.path)
    ]


def _exact_json_object(text: str) -> dict[str, Any]:
    value = strict_json_loads(text.strip())
    if not isinstance(value, dict):
        raise ValueError("description output root must be a JSON object")
    return value


def _extract_json_for_repair(text: str) -> dict[str, Any]:
    """Lenient extraction is analysis-only and never establishes raw validity."""
    stripped = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        stripped = fenced.group(1).strip()
    try:
        value = strict_json_loads(stripped)
    except json.JSONDecodeError:
        start, end = stripped.find("{"), stripped.rfind("}")
        if start < 0 or end <= start:
            raise
        value = strict_json_loads(stripped[start:end + 1])
    if not isinstance(value, dict):
        raise ValueError("description output root must be a JSON object")
    return value


def _enum(value: Any, allowed: set[str], default: str, field: str, actions: list[str]) -> str:
    normalized = str(value) if value is not None else default
    if normalized not in allowed:
        actions.append(f"{field}:invalid_to_{default}")
        return default
    return normalized


def deterministic_repair(value: dict[str, Any] | None) -> tuple[dict[str, Any], tuple[str, ...]]:
    source = copy.deepcopy(value) if isinstance(value, dict) else {}
    actions: list[str] = []
    if not isinstance(value, dict):
        actions.append("root:missing_to_default")
    region_source = source.get("region") if isinstance(source.get("region"), dict) else {}
    evidence_source = source.get("evidence") if isinstance(source.get("evidence"), dict) else {}
    repaired = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "target_status": _enum(
            source.get("target_status"), {"present", "absent", "uncertain"},
            "uncertain", "target_status", actions,
        ),
        "region": {
            "location": _enum(region_source.get("location"), {
                "upper_left", "upper_center", "upper_right", "center_left", "center",
                "center_right", "lower_left", "lower_center", "lower_right", "distributed",
                "unknown", "unavailable",
            }, "unavailable", "region.location", actions),
            "size_class": _enum(region_source.get("size_class"), {
                "tiny", "small", "medium", "large", "extensive", "unknown", "unavailable",
            }, "unavailable", "region.size_class", actions),
            "shape": _enum(region_source.get("shape"), {
                "compact", "elongated", "branching", "fragmented", "irregular", "unknown", "unavailable",
            }, "unavailable", "region.shape", actions),
            "elongation": _enum(region_source.get("elongation"), {
                "low", "moderate", "high", "unknown", "unavailable",
            }, "unavailable", "region.elongation", actions),
            "compactness": _enum(region_source.get("compactness"), {
                "compact", "moderate", "dispersed", "unknown", "unavailable",
            }, "unavailable", "region.compactness", actions),
            "fragmentation": _enum(region_source.get("fragmentation"), {
                "single", "few_components", "many_components", "highly_fragmented", "unknown", "unavailable",
            }, "unavailable", "region.fragmentation", actions),
        },
        "evidence": {
            "surface_observation": str(evidence_source.get("surface_observation") or "unavailable"),
            "terrain_support": _enum(evidence_source.get("terrain_support"), {
                "supports", "does_not_support", "insufficient_evidence", "unknown", "unavailable",
            }, "unavailable", "evidence.terrain_support", actions),
            "sar_support": _enum(evidence_source.get("sar_support"), {
                "supports", "does_not_support", "insufficient_evidence", "unknown", "unavailable",
            }, "unavailable", "evidence.sar_support", actions),
            "deformation_support": _enum(evidence_source.get("deformation_support"), {
                "supports", "does_not_support", "insufficient_evidence", "unknown", "unavailable",
            }, "unavailable", "evidence.deformation_support", actions),
            "surrounding_context": str(evidence_source.get("surrounding_context") or "unavailable"),
            "evidence_sufficiency": _enum(evidence_source.get("evidence_sufficiency"), {
                "sufficient", "partial", "insufficient", "unavailable",
            }, "unavailable", "evidence.evidence_sufficiency", actions),
        },
        "summary": str(source.get("summary") or "No reliable description is available."),
    }
    confidence = source.get("confidence")
    if (
        isinstance(confidence, (int, float))
        and not isinstance(confidence, bool)
        and math.isfinite(float(confidence))
    ):
        clipped = min(1.0, max(0.0, float(confidence)))
        if clipped != float(confidence):
            actions.append("confidence:clipped")
        repaired["confidence"] = clipped
    elif isinstance(confidence, (int, float)) and not isinstance(confidence, bool):
        actions.append("confidence:nonfinite_removed")
    if source.get("schema_version") != OUTPUT_SCHEMA_VERSION:
        actions.append("schema_version:reset")
    if repaired["target_status"] == "absent":
        # Repair 只执行 schema 可推导的拒绝语义，不借助 GT 或图像补写内容。
        for field, current in repaired["region"].items():
            if current != "unavailable":
                repaired["region"][field] = "unavailable"
                actions.append(f"region.{field}:absent_to_unavailable")
        for field in ("terrain_support", "sar_support", "deformation_support"):
            if repaired["evidence"][field] not in {
                "insufficient_evidence", "unavailable",
            }:
                repaired["evidence"][field] = "unavailable"
                actions.append(f"evidence.{field}:absent_to_unavailable")
        if repaired["evidence"]["evidence_sufficiency"] not in {
            "insufficient", "unavailable",
        }:
            repaired["evidence"]["evidence_sufficiency"] = "unavailable"
            actions.append("evidence.evidence_sufficiency:absent_to_unavailable")
    return repaired, tuple(actions)


def parse_description_output(raw_text: str) -> ParsedDescription:
    errors: list[str] = []
    try:
        parsed = _exact_json_object(raw_text)
    except Exception as exc:
        parsed = None
        errors.append(f"json_parse:{type(exc).__name__}:{exc}")
    if parsed is not None:
        errors.extend(_validation_errors(parsed))
    repair_source = parsed
    extracted_for_repair = False
    if repair_source is None:
        try:
            repair_source = _extract_json_for_repair(raw_text)
            extracted_for_repair = True
        except Exception:
            repair_source = None
    repaired, actions = deterministic_repair(repair_source)
    if extracted_for_repair:
        actions = ("root:extracted_wrapper_for_repair", *actions)
    return ParsedDescription(
        raw_text=raw_text,
        parsed=parsed,
        parse_errors=tuple(errors),
        schema_valid=parsed is not None and not errors,
        repaired=repaired,
        repair_actions=actions,
    )
