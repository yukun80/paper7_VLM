#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Raw parse, schema validation and separately reported deterministic repair."""

from __future__ import annotations

from dataclasses import dataclass
import copy
import json
import re
from typing import Any

from jsonschema import Draft202012Validator

from qpsalm_seg.paths import REPO_ROOT


OUTPUT_SCHEMA_VERSION = "qpsalm_description_output_v1"


@dataclass
class ParsedDescription:
    raw_text: str
    parsed: dict[str, Any] | None
    parse_errors: tuple[str, ...]
    schema_valid: bool
    repaired: dict[str, Any]
    repair_actions: tuple[str, ...]


def _schema() -> dict[str, Any]:
    path = REPO_ROOT / "configs/qpsalm_description_output_v1.schema.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        stripped = fenced.group(1).strip()
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        start, end = stripped.find("{"), stripped.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(stripped[start:end + 1])
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
    if isinstance(confidence, (int, float)):
        clipped = min(1.0, max(0.0, float(confidence)))
        if clipped != float(confidence):
            actions.append("confidence:clipped")
        repaired["confidence"] = clipped
    if source.get("schema_version") != OUTPUT_SCHEMA_VERSION:
        actions.append("schema_version:reset")
    return repaired, tuple(actions)


def parse_description_output(raw_text: str) -> ParsedDescription:
    errors: list[str] = []
    try:
        parsed = _extract_json(raw_text)
    except Exception as exc:
        parsed = None
        errors.append(f"json_parse:{type(exc).__name__}:{exc}")
    if parsed is not None:
        validator = Draft202012Validator(_schema())
        errors.extend(
            f"schema:{'.'.join(str(value) for value in error.absolute_path)}:{error.message}"
            for error in sorted(validator.iter_errors(parsed), key=lambda error: list(error.absolute_path))
        )
    repaired, actions = deterministic_repair(parsed)
    return ParsedDescription(
        raw_text=raw_text,
        parsed=parsed,
        parse_errors=tuple(errors),
        schema_valid=parsed is not None and not errors,
        repaired=repaired,
        repair_actions=actions,
    )
