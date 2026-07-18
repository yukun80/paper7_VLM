#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared strict contracts for M7 full-val retention evidence."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from qpsalm_seg.engine.evaluator import (
    SAMPLE_IDENTITY_FIELDS,
    SAMPLE_IDENTITY_PROTOCOL,
    validate_segmentation_prediction_population,
)
from qpsalm_seg.paths import resolve_project_path

from ..protocols.io import sha256_file, strict_json_loads


RETENTION_GATE_PROTOCOL = (
    "qpsalm_segdesc_retention_v22_run_completion_bound"
)
RETENTION_EVAL_BINDING_PROTOCOL = "qpsalm_segdesc_retention_eval_binding_v1"
BASELINE_CHECKPOINT_REPLAY_PROTOCOL = (
    "qpsalm_segdesc_baseline_checkpoint_replay_v2_eval_config_bound"
)
M7_RETENTION_SEED_GATE_PROTOCOL = (
    "qpsalm_segdesc_retention_seed_gate_v18_run_completion_bound"
)
M7_RUN_LOCAL_CONFIG_FIELDS = {
    "seed",
    "output_dir",
    "d4_final_acceptance_gate",
    "m6_acceptance_gate",
}
SEGMENTATION_METRIC_INPUT_POPULATION_PROTOCOL = (
    "qpsalm_segmentation_metric_input_population_v1_target_valid_bound"
)
SEGMENTATION_METRIC_INPUT_FIELDS = (
    "sample_id",
    "parent_sample_id",
    "shape",
    "target_sha256",
    "valid_sha256",
)


def json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = strict_json_loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} 不是可读 JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} 顶层必须是 object: {path}")
    return payload


def bound_file(
    path_ref: Any,
    expected_sha256: Any,
    *,
    label: str,
) -> tuple[Path, str]:
    if not isinstance(path_ref, str) or not path_ref.strip():
        raise ValueError(f"{label} 缺少 path")
    expected = str(expected_sha256 or "")
    if len(expected) != 64:
        raise ValueError(f"{label} 缺少 SHA-256")
    path = resolve_project_path(path_ref)
    if path is None or not path.is_file():
        raise ValueError(f"{label} 文件不存在: {path_ref}")
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(
            f"{label} SHA-256 已漂移: expected={expected} observed={observed}"
        )
    return path.resolve(strict=False), observed


def finite_float(value: Any, *, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} 不是数值: {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} 必须有限: {value!r}")
    return result


def integer(value: Any, *, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} 不能是 bool")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} 不是整数: {value!r}") from exc
    if str(value).strip() not in {str(result), f"+{result}"} and not isinstance(value, int):
        raise ValueError(f"{label} 不是精确整数: {value!r}")
    return result


def positive_dice(report: dict[str, Any]) -> float:
    return finite_float(
        (((report.get("metrics") or {}).get("positive_only") or {}).get("dice")),
        label="eval positive Dice",
    )


def sample_population(report: dict[str, Any]) -> dict[str, Any]:
    value = (report.get("coverage") or {}).get("sample_population") or {}
    if not isinstance(value, dict):
        raise ValueError("eval sample_population 必须是 object")
    return value


def validate_population(
    population: dict[str, Any],
    *,
    expected_count: int,
    label: str,
) -> dict[str, Any]:
    errors: list[str] = []
    if population.get("protocol") != SAMPLE_IDENTITY_PROTOCOL:
        errors.append("protocol")
    if tuple(population.get("fields") or ()) != tuple(SAMPLE_IDENTITY_FIELDS):
        errors.append("fields")
    if population.get("complete") is not True:
        errors.append("complete")
    if population.get("unique") is not True:
        errors.append("unique")
    if integer(population.get("num_records", -1), label=f"{label}.num_records") != expected_count:
        errors.append("num_records")
    if integer(
        population.get("num_unique_sample_ids", -1),
        label=f"{label}.num_unique_sample_ids",
    ) != expected_count:
        errors.append("num_unique_sample_ids")
    population_sha256 = str(population.get("sha256") or "")
    if len(population_sha256) != 64:
        errors.append("sha256")
    if errors:
        raise ValueError(f"{label} population 非法: {errors}")
    return {
        "protocol": SAMPLE_IDENTITY_PROTOCOL,
        "fields": list(SAMPLE_IDENTITY_FIELDS),
        "sha256": population_sha256,
        "num_records": expected_count,
    }


def validate_eval_report_matches_gate(
    report: dict[str, Any],
    gate: dict[str, Any],
    *,
    prefix: str,
) -> None:
    expected_count = integer(gate[f"{prefix}_num_samples"], label=f"{prefix}_num_samples")
    observed_count = integer(
        (report.get("coverage") or {}).get("num_samples", -1),
        label=f"{prefix} report num_samples",
    )
    expected_population = gate[f"{prefix}_sample_population"]
    if not isinstance(expected_population, dict):
        raise ValueError(f"{prefix}_sample_population 必须是 object")
    observed_population = sample_population(report)
    expected_threshold = finite_float(gate[f"{prefix}_threshold"], label=f"{prefix}_threshold")
    observed_threshold = finite_float(report.get("threshold", 0.5), label=f"{prefix} report threshold")
    expected_dice = finite_float(
        gate[f"{prefix}_positive_dice"], label=f"{prefix}_positive_dice"
    )
    observed_dice = positive_dice(report)
    prediction_population = validate_segmentation_prediction_population(
        report.get("prediction_population")
    )
    expected_prediction_sha256 = str(
        gate.get(f"{prefix}_prediction_population_sha256") or ""
    )
    if (
        observed_count != expected_count
        or observed_population != expected_population
        or abs(observed_threshold - expected_threshold) > 1.0e-12
        or abs(observed_dice - expected_dice) > 1.0e-12
        or prediction_population.get("sha256")
        != expected_prediction_sha256
        or abs(
            finite_float(
                prediction_population.get("threshold"),
                label=f"{prefix} prediction population threshold",
            )
            - observed_threshold
        ) > 1.0e-12
    ):
        raise ValueError(f"{prefix} eval report 与 retention gate 内容不一致")
