#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""M4/M7 multi-seed comparison workflows.

用途：集中 MGRR paired gate 与 full-val retention 三种子聚合的原子发布。
推荐调用：由 compare_description_runs/compare_segdesc_retention 薄入口调用。
输入：source-bound eval、retrieval、expert 或 retention gate artifacts。
输出：自验证的 comparison gate JSON。
写入行为：只写显式 output，不修改源运行目录。
工作流阶段：M4/M7 scientific comparison orchestration。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from qpsalm_seg.paths import resolve_project_path

from ..evaluation.comparison import (
    compare_description_seeds,
    validate_m4_seed_gate,
)
from ..evaluation.retention import (
    aggregate_m7_retention_seed_gates,
    validate_m7_retention_seed_gate,
)
from ..protocols.io import atomic_write_json


def _publish(
    report: dict[str, Any], output: str, validator: Callable[[Path], Any]
) -> dict[str, Any]:
    destination = resolve_project_path(output) or Path(output)
    candidate = destination.with_name(f".{destination.name}.candidate")
    try:
        atomic_write_json(candidate, report)
        validator(candidate)
        candidate.replace(destination)
    finally:
        candidate.unlink(missing_ok=True)
    return report


def run_description_seed_comparison(
    *,
    baseline_dirs: list[str],
    candidate_dirs: list[str],
    seeds: list[int],
    bridge_benchmark: str,
    baseline_retrieval_dirs: list[str],
    candidate_retrieval_dirs: list[str],
    baseline_expert_reports: list[str],
    candidate_expert_reports: list[str],
    output: str,
) -> dict[str, Any]:
    report = compare_description_seeds(
        baseline_dirs,
        candidate_dirs,
        seeds=seeds,
        bridge_benchmark=bridge_benchmark,
        baseline_retrieval_dirs=baseline_retrieval_dirs,
        candidate_retrieval_dirs=candidate_retrieval_dirs,
        baseline_expert_reports=baseline_expert_reports,
        candidate_expert_reports=candidate_expert_reports,
    )
    return _publish(report, output, validate_m4_seed_gate)


def run_retention_seed_comparison(
    *, retention_gates: list[str], seeds: list[int], output: str
) -> dict[str, Any]:
    report = aggregate_m7_retention_seed_gates(retention_gates, seeds=seeds)
    return _publish(report, output, validate_m7_retention_seed_gate)
