#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bind baseline replay, metric inputs, and joint evaluation artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from qpsalm_seg.engine.evaluator import validate_segmentation_prediction_population
from qpsalm_seg.paths import resolve_project_path

from ..protocols.io import canonical_sha256
from .retention_contracts import (
    BASELINE_CHECKPOINT_REPLAY_PROTOCOL,
    RETENTION_EVAL_BINDING_PROTOCOL,
    RETENTION_GATE_PROTOCOL,
    SEGMENTATION_METRIC_INPUT_FIELDS,
    SEGMENTATION_METRIC_INPUT_POPULATION_PROTOCOL,
    finite_float,
    integer,
    json_object,
    sample_population,
    sha256_file,
    validate_eval_report_matches_gate,
)


def segmentation_metric_input_population(
    prediction_population: dict[str, Any],
) -> dict[str, Any]:
    """Bind the exact target/valid bytes used to compute segmentation metrics."""

    validated = validate_segmentation_prediction_population(
        prediction_population
    )
    rows = [
        {field: row[field] for field in SEGMENTATION_METRIC_INPUT_FIELDS}
        for row in validated["rows"]
    ]
    rows.sort(key=lambda row: str(row["sample_id"]))
    return {
        "protocol": SEGMENTATION_METRIC_INPUT_POPULATION_PROTOCOL,
        "fields": list(SEGMENTATION_METRIC_INPUT_FIELDS),
        "num_records": len(rows),
        "sha256": canonical_sha256(rows),
    }


def build_baseline_checkpoint_replay_audit(
    frozen_report: dict[str, Any],
    replay_report: dict[str, Any],
    *,
    baseline_binding: dict[str, Any],
    segmentation_migration: dict[str, Any],
    replay_report_path: str | Path,
) -> dict[str, Any]:
    """Prove the frozen baseline metrics replay from the declared checkpoint."""

    frozen_prediction = validate_segmentation_prediction_population(
        frozen_report.get("prediction_population")
    )
    replay_prediction = validate_segmentation_prediction_population(
        replay_report.get("prediction_population")
    )
    replay_path = resolve_project_path(replay_report_path) or Path(replay_report_path)
    if not replay_path.is_file():
        raise FileNotFoundError(f"baseline replay report 不存在: {replay_path}")
    replay_from_file = json_object(replay_path, label="baseline replay report")
    if replay_from_file != replay_report:
        raise ValueError("baseline replay 内存结果与原子发布文件不一致")
    source_sha256 = str(segmentation_migration.get("source_sha256") or "")
    bound_checkpoint_sha256 = str(baseline_binding.get("checkpoint_sha256") or "")
    frozen_count = integer(
        (frozen_report.get("coverage") or {}).get("num_samples", -1),
        label="frozen baseline count",
    )
    replay_count = integer(
        (replay_report.get("coverage") or {}).get("num_samples", -1),
        label="replayed baseline count",
    )
    frozen_step = integer(
        frozen_report.get("checkpoint_step", -1), label="frozen baseline step"
    )
    replay_step = integer(
        replay_report.get("checkpoint_step", -1), label="replayed baseline step"
    )
    bound_step = integer(
        baseline_binding.get("checkpoint_step", -1), label="bound baseline step"
    )
    migration_step = integer(
        segmentation_migration.get("source_step", -1),
        label="segmentation migration source step",
    )
    frozen_evidence = {
        "metrics": frozen_report.get("metrics"),
        "metrics_original_size": frozen_report.get("metrics_original_size"),
        "threshold_sweep": frozen_report.get("threshold_sweep"),
    }
    replay_evidence = {
        "metrics": replay_report.get("metrics"),
        "metrics_original_size": replay_report.get("metrics_original_size"),
        "threshold_sweep": replay_report.get("threshold_sweep"),
    }
    checks = {
        "same_segmentation_checkpoint": bool(
            len(source_sha256) == 64
            and source_sha256 == bound_checkpoint_sha256
        ),
        "same_checkpoint_step": (
            frozen_step == replay_step == bound_step == migration_step
        ),
        "same_sample_count": frozen_count > 0 and frozen_count == replay_count,
        "same_sample_population": (
            sample_population(frozen_report) == sample_population(replay_report)
        ),
        "same_threshold": abs(
            finite_float(frozen_report.get("threshold"), label="frozen threshold")
            - finite_float(replay_report.get("threshold"), label="replay threshold")
        ) <= 1.0e-12,
        "same_prediction_population": (
            frozen_prediction == replay_prediction
        ),
        "same_metric_evidence": frozen_evidence == replay_evidence,
    }
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise ValueError(f"baseline checkpoint replay 不一致: {failures}")
    return {
        "protocol": BASELINE_CHECKPOINT_REPLAY_PROTOCOL,
        "passed": True,
        "checks": checks,
        "checkpoint_sha256": source_sha256,
        "checkpoint_step": bound_step,
        "segmentation_source_identity": {
            name: segmentation_migration.get(name)
            for name in (
                "source_sha256",
                "source_format",
                "source_step",
                "allowed_prefixes",
            )
        },
        "num_samples": frozen_count,
        "sample_population_sha256": str(
            sample_population(frozen_report).get("sha256") or ""
        ),
        "prediction_population_sha256": frozen_prediction["sha256"],
        "metric_evidence_sha256": canonical_sha256(frozen_evidence),
        "frozen_report": {
            "path": str(baseline_binding.get("eval_report") or ""),
            "sha256": str(baseline_binding.get("eval_report_sha256") or ""),
        },
        "replay_report": {
            "path": str(replay_path.resolve(strict=False)),
            "sha256": sha256_file(replay_path),
            "bytes": int(replay_path.stat().st_size),
        },
    }


def bind_joint_evaluation_report(
    gate: dict[str, Any],
    *,
    eval_report_path: str | Path,
    checkpoint_path: str | Path,
) -> dict[str, Any]:
    """Finalize a retention gate only after binding its raw full-val report."""
    if gate.get("protocol") != RETENTION_GATE_PROTOCOL:
        raise ValueError("retention gate protocol 不支持 report binding")
    report_path = resolve_project_path(eval_report_path) or Path(eval_report_path)
    checkpoint = resolve_project_path(checkpoint_path) or Path(checkpoint_path)
    if not report_path.is_file() or not checkpoint.is_file():
        raise FileNotFoundError("retention report binding 的 report/checkpoint 不存在")
    report = json_object(report_path, label="joint segmentation eval report")
    validate_eval_report_matches_gate(report, gate, prefix="joint")
    checkpoint_sha256 = sha256_file(checkpoint)
    declared_checkpoint_sha256 = str(gate.get("joint_checkpoint_sha256") or "")
    if declared_checkpoint_sha256 != checkpoint_sha256:
        raise ValueError("joint checkpoint SHA-256 与 retention gate 不一致")
    finalized = dict(gate)
    finalized["joint_eval_binding"] = {
        "protocol": RETENTION_EVAL_BINDING_PROTOCOL,
        "eval_report": str(report_path.resolve(strict=False)),
        "eval_report_sha256": sha256_file(report_path),
        "checkpoint": str(checkpoint.resolve(strict=False)),
        "checkpoint_sha256": checkpoint_sha256,
        "split": str(gate.get("split") or ""),
        "sample_population_sha256": str(
            (gate.get("joint_sample_population") or {}).get("sha256") or ""
        ),
    }
    finalized["formal_report_binding_complete"] = True
    finalized["passed"] = bool(
        finalized.get("scientific_gate_eligible") is True
        and finalized.get("preliminary_passed") is True
    )
    return finalized
