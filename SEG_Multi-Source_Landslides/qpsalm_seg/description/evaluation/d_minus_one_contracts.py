"""Reusable file and zero-shot contracts for the D-1 acceptance workflow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from qpsalm_seg.paths import resolve_project_path

from ..protocols.io import sha256_file, strict_json_loads
from .zero_shot import (
    DESCRIPTION_BUILDER_VERSION,
    ZERO_SHOT_INPUT_PROTOCOL,
    ZERO_SHOT_MODEL_IDENTITY_PROTOCOL,
    ZERO_SHOT_PROTOCOL,
    build_zero_shot_input_audit,
    build_zero_shot_model_identity,
)


def load_d_minus_one_report(path: Path, label: str) -> dict[str, Any]:
    """Load one strict JSON report with a D-1-specific diagnostic."""
    if not path.is_file():
        raise FileNotFoundError(f"D-1 缺少 {label}: {path}")
    try:
        value = strict_json_loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"D-1 {label} 不是合法 JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"D-1 {label} 必须为 JSON object: {path}")
    return value


def bound_file_matches(path_value: Any, sha_value: Any) -> bool:
    """Return whether a file exists and matches its exact SHA-256 binding."""
    path = resolve_project_path(str(path_value or ""))
    return bool(
        path is not None
        and path.is_file()
        and isinstance(sha_value, str)
        and len(sha_value) == 64
        and sha256_file(path) == sha_value
    )


def read_strict_jsonl(path: Path) -> list[dict[str, Any]] | None:
    """Read a strict object-only JSONL artifact, returning None on corruption."""
    try:
        rows = [
            strict_json_loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError):
        return None
    return rows if all(isinstance(row, dict) for row in rows) else None


def validate_zero_shot_report(report: dict[str, Any]) -> dict[str, Any]:
    """Rebuild zero-shot input selection and revalidate every bound file."""
    input_audit = dict(report.get("input_audit") or {})
    model_audit = dict(report.get("model_audit") or {})
    num_samples = int(report.get("num_samples", 0))
    raw_path = resolve_project_path(str(report.get("raw_generations") or ""))
    raw_rows = (
        read_strict_jsonl(raw_path)
        if raw_path is not None and raw_path.is_file()
        else None
    )
    current_input: dict[str, Any] | None = None
    selected_rows: list[dict[str, Any]] | None = None
    try:
        selected_rows, current_input = build_zero_shot_input_audit(
            input_audit.get("benchmark_root") or "",
            str(input_audit.get("split") or ""),
            int(input_audit.get("requested_max_samples", 0)),
            int(input_audit.get("sampling_seed", -1)),
        )
    except (FileNotFoundError, KeyError, RuntimeError, TypeError, ValueError):
        current_input = None
        selected_rows = None
    model_dir = resolve_project_path(str(model_audit.get("model_dir") or ""))
    current_model: dict[str, Any] | None = None
    try:
        if model_dir is not None and model_dir.is_dir():
            current_model = build_zero_shot_model_identity(model_dir)
    except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError):
        current_model = None
    raw_ids = (
        [str(row.get("sample_id") or "") for row in raw_rows]
        if raw_rows is not None else []
    )
    selected_ids = (
        [str(row.get("sample_id") or "") for row in selected_rows]
        if selected_rows is not None else []
    )
    reported_checks = dict(report.get("checks") or {})
    checks = {
        "protocol_current": report.get("protocol") == ZERO_SHOT_PROTOCOL,
        "reported_engineering_valid": (
            report.get("status") == "engineering-valid"
            and not (report.get("errors") or [])
            and bool(reported_checks)
            and all(value is True for value in reported_checks.values())
        ),
        "population_is_exactly_64": num_samples == 64,
        "input_budget_is_exactly_64": (
            input_audit.get("requested_max_samples") == 64
            and input_audit.get("selected_samples") == 64
        ),
        "input_protocol_current": (
            input_audit.get("protocol") == ZERO_SHOT_INPUT_PROTOCOL
        ),
        "input_builder_current": (
            input_audit.get("builder_version") == DESCRIPTION_BUILDER_VERSION
        ),
        "input_rebuild_matches": (
            current_input is not None and input_audit == current_input
        ),
        "raw_generations_bound": bound_file_matches(
            report.get("raw_generations"),
            report.get("raw_generations_sha256"),
        ),
        "raw_generation_population_matches": (
            raw_rows is not None
            and len(raw_rows) == num_samples
            and raw_ids == selected_ids
            and len(raw_ids) == len(set(raw_ids))
            and all(raw_ids)
            and all(
                bool(str(row.get("prediction") or "").strip())
                for row in raw_rows
            )
        ),
        "model_identity_protocol_current": (
            model_audit.get("protocol")
            == ZERO_SHOT_MODEL_IDENTITY_PROTOCOL
        ),
        "model_files_exactly_bound": (
            current_model is not None and model_audit == current_model
        ),
        "statistics_seed_matches_sampling": (
            report.get("statistics_seed") == input_audit.get("sampling_seed")
        ),
        "no_region_capability_claim": (
            report.get("region_capability_claimed") is False
        ),
    }
    errors = [name for name, passed in checks.items() if not passed]
    return {
        "protocol": "qpsalm_d_minus_one_zero_shot_revalidation_v1",
        "status": "engineering-valid" if not errors else "engineering-invalid",
        "checks": checks,
        "errors": errors,
    }
