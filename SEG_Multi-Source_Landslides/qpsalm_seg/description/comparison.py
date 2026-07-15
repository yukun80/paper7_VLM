#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Paired Small-seed comparison for crop-only, masked pooling and MGRR."""

from __future__ import annotations

import json
from pathlib import Path
import hashlib
from typing import Any

from qpsalm_seg.paths import resolve_project_path

from .metrics import paired_bootstrap_delta_ci


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_expert_binding(
    expert: dict[str, Any],
    evaluation_dir: str | Path,
    *,
    label: str,
) -> None:
    root = (resolve_project_path(evaluation_dir) or Path(evaluation_dir)).resolve(
        strict=False
    )
    observed_root = (resolve_project_path(str(expert.get("eval_dir") or "")) or Path(
        str(expert.get("eval_dir") or "")
    )).resolve(strict=False)
    if observed_root != root:
        raise ValueError(f"{label} expert report 与 eval directory 不匹配")
    generation = root / "raw_generations.jsonl"
    report = root / "eval_report.json"
    if (
        str(expert.get("raw_generations_sha256") or "") != _sha256(generation)
        or str(expert.get("eval_report_sha256") or "") != _sha256(report)
    ):
        raise ValueError(f"{label} expert report 的 frozen eval 指纹已失效")


def _rows(
    directory: str | Path,
    *,
    require_complete_generation: bool = False,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    root = resolve_project_path(directory) or Path(directory)
    rows = {
        str(row["sample_id"]): row
        for row in (
            json.loads(line)
            for line in (root / "raw_generations.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    report = json.loads((root / "eval_report.json").read_text(encoding="utf-8"))
    if require_complete_generation and report.get("protocol") != "qpsalm_description_evaluation_v3":
        raise ValueError(
            f"正式 paired gate 需要 qpsalm_description_evaluation_v3：{root}；"
            "请用当前 evaluator 重跑评估"
        )
    if require_complete_generation and not bool(
        (report.get("generation_coverage") or {}).get("complete")
    ):
        raise ValueError(
            f"正式 paired gate 需要全量 generation：{root}；"
            "请用 --max-generate-samples 0 重跑评估"
        )
    return rows, report


def _score(row: dict[str, Any]) -> float:
    metrics = row.get("raw_metrics") or {}
    if metrics.get("raw_field_accuracy") is not None:
        return float(metrics["raw_field_accuracy"])
    return float(metrics.get("caption_token_f1") or 0.0)


def _claim_rate(rows: list[dict[str, Any]]) -> float:
    unsupported = sum(int((row.get("raw_metrics") or {}).get("unsupported_claims") or 0) for row in rows)
    claims = sum(int((row.get("raw_metrics") or {}).get("factual_claims") or 0) for row in rows)
    return unsupported / max(claims, 1)


def _counterfactual_gate(report: dict[str, Any]) -> dict[str, Any]:
    values = report.get("counterfactual_sensitivity") or {}

    def upper(mode: str, metric: str) -> float | None:
        value = (((values.get(mode) or {}).get(metric) or {}).get("high"))
        return float(value) if value is not None else None

    required_modes = (
        "shuffled_mask", "region_swap", "cross_parent_modality_swap",
        "modality_removal",
    )
    coverage = {
        mode: bool(
            int((values.get(mode) or {}).get("requested") or 0) > 0
            and int((values.get(mode) or {}).get("n") or 0)
            >= int((values.get(mode) or {}).get("requested") or 0)
            and (values.get(mode) or {}).get("coverage_complete") is True
        )
        for mode in required_modes
    }
    checks = {
        "counterfactual_coverage_complete": all(coverage.values()),
        "shuffled_mask_degrades_target_score": (
            upper("shuffled_mask", "paired_target_score_delta_ci") is not None
            and upper("shuffled_mask", "paired_target_score_delta_ci") < 0
        ),
        "region_swap_degrades_target_score": (
            upper("region_swap", "paired_target_score_delta_ci") is not None
            and upper("region_swap", "paired_target_score_delta_ci") < 0
        ),
        "cross_parent_swap_degrades_target_score": (
            upper("cross_parent_modality_swap", "paired_target_score_delta_ci") is not None
            and upper("cross_parent_modality_swap", "paired_target_score_delta_ci") < 0
        ),
        "modality_removal_reduces_factual_claims": (
            upper("modality_removal", "paired_factual_claim_count_delta_ci") is not None
            and upper("modality_removal", "paired_factual_claim_count_delta_ci") < 0
        ),
    }
    return {
        "checks": checks,
        "coverage_by_mode": coverage,
        "passed": all(checks.values()),
        "counterfactual_sensitivity": values,
    }


def compare_description_run_pair(
    baseline_dir: str | Path,
    candidate_dir: str | Path,
    *,
    seed: int,
    unsupported_noninferiority: float,
    baseline_retrieval_dir: str | Path,
    candidate_retrieval_dir: str | Path,
    baseline_expert_report: str | Path,
    candidate_expert_report: str | Path,
) -> dict[str, Any]:
    baseline, _baseline_report = _rows(
        baseline_dir, require_complete_generation=True
    )
    candidate, _candidate_report = _rows(
        candidate_dir, require_complete_generation=True
    )
    shared = sorted(set(baseline) & set(candidate))
    if set(baseline) != set(candidate):
        raise ValueError(
            f"paired description samples 不一致: baseline={len(baseline)} "
            f"candidate={len(candidate)} shared={len(shared)}"
        )
    baseline_rows = [baseline[value] for value in shared]
    candidate_rows = [candidate[value] for value in shared]
    baseline_expert_path = resolve_project_path(baseline_expert_report) or Path(baseline_expert_report)
    candidate_expert_path = resolve_project_path(candidate_expert_report) or Path(candidate_expert_report)
    baseline_expert = json.loads(baseline_expert_path.read_text(encoding="utf-8"))
    candidate_expert = json.loads(candidate_expert_path.read_text(encoding="utf-8"))
    if baseline_expert.get("protocol") != "qpsalm_expert_region_factuality_v1" or candidate_expert.get("protocol") != "qpsalm_expert_region_factuality_v1":
        raise ValueError("正式 MGRR gate 需要 qpsalm_expert_region_factuality_v1 报告")
    _validate_expert_binding(
        baseline_expert, baseline_dir, label="baseline"
    )
    _validate_expert_binding(
        candidate_expert, candidate_dir, label="candidate"
    )
    baseline_parent = baseline_expert.get("per_parent_scores") or {}
    candidate_parent = candidate_expert.get("per_parent_scores") or {}
    if set(baseline_parent) != set(candidate_parent):
        raise ValueError("baseline/candidate ERFS parent 集合不一致")
    expert_parents = sorted(baseline_parent)
    ci = paired_bootstrap_delta_ci(
        [float(baseline_parent[value]) for value in expert_parents],
        [float(candidate_parent[value]) for value in expert_parents],
        seed=seed,
        samples=10000,
    )
    automatic_baseline_ufcr = _claim_rate(baseline_rows)
    automatic_candidate_ufcr = _claim_rate(candidate_rows)
    baseline_ufcr = float(baseline_expert.get("expert_unsupported_claim_rate") or 0.0)
    candidate_ufcr = float(candidate_expert.get("expert_unsupported_claim_rate") or 0.0)
    _unused_rows, baseline_retrieval = _rows(baseline_retrieval_dir)
    _unused_rows, candidate_retrieval = _rows(candidate_retrieval_dir)
    baseline_r1 = (baseline_retrieval.get("same_image_retrieval") or {}).get("mean_r1")
    candidate_r1 = (candidate_retrieval.get("same_image_retrieval") or {}).get("mean_r1")
    baseline_retrieval_parent = (
        (baseline_retrieval.get("same_image_retrieval") or {}).get("per_parent_mean_r1") or {}
    )
    candidate_retrieval_parent = (
        (candidate_retrieval.get("same_image_retrieval") or {}).get("per_parent_mean_r1") or {}
    )
    if not baseline_retrieval_parent or set(baseline_retrieval_parent) != set(candidate_retrieval_parent):
        raise ValueError(
            "正式 MGRR gate 需要相同 parent 的 per_parent_mean_r1；请用当前 evaluator 重跑 DIOR"
        )
    retrieval_parents = sorted(baseline_retrieval_parent)
    retrieval_ci = paired_bootstrap_delta_ci(
        [float(baseline_retrieval_parent[value]) for value in retrieval_parents],
        [float(candidate_retrieval_parent[value]) for value in retrieval_parents],
        seed=seed + 104729,
        samples=10000,
    )
    retrieval_improved = (
        baseline_r1 is not None and candidate_r1 is not None
        and float(candidate_r1) > float(baseline_r1)
        and retrieval_ci["low"] is not None
        and float(retrieval_ci["low"]) > 0
    )
    counterfactual_gate = _counterfactual_gate(_candidate_report)
    passed = (
        ci["low"] is not None
        and float(ci["low"]) > 0
        and retrieval_improved
        and candidate_ufcr <= baseline_ufcr + float(unsupported_noninferiority)
        and counterfactual_gate["passed"]
    )
    return {
        "seed": seed,
        "num_paired_samples": len(shared),
        "num_expert_parents": len(expert_parents),
        "expert_region_factuality_delta_ci": ci,
        "baseline_unsupported_claim_rate": baseline_ufcr,
        "candidate_unsupported_claim_rate": candidate_ufcr,
        "automatic_proxy_baseline_unsupported_claim_rate": automatic_baseline_ufcr,
        "automatic_proxy_candidate_unsupported_claim_rate": automatic_candidate_ufcr,
        "unsupported_noninferiority": unsupported_noninferiority,
        "baseline_same_image_r1": baseline_r1,
        "candidate_same_image_r1": candidate_r1,
        "num_retrieval_parents": len(retrieval_parents),
        "same_image_r1_delta_ci": retrieval_ci,
        "retrieval_improved": retrieval_improved,
        "counterfactual_gate": counterfactual_gate,
        "passed": passed,
    }


def compare_description_seeds(
    baseline_dirs: list[str],
    candidate_dirs: list[str],
    *,
    seeds: list[int],
    unsupported_noninferiority: float,
    baseline_retrieval_dirs: list[str],
    candidate_retrieval_dirs: list[str],
    baseline_expert_reports: list[str],
    candidate_expert_reports: list[str],
) -> dict[str, Any]:
    if not (
        len(baseline_dirs) == len(candidate_dirs) == len(seeds)
        == len(baseline_retrieval_dirs) == len(candidate_retrieval_dirs)
        == len(baseline_expert_reports) == len(candidate_expert_reports)
    ):
        raise ValueError("description/retrieval baseline/candidate/seeds 数量必须一致")
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError("正式 MGRR gate 必须提供三个不同 seed 的配对运行")
    pairs = [
        compare_description_run_pair(
            baseline, candidate, seed=seed,
            unsupported_noninferiority=unsupported_noninferiority,
            baseline_retrieval_dir=baseline_retrieval,
            candidate_retrieval_dir=candidate_retrieval,
            baseline_expert_report=baseline_expert,
            candidate_expert_report=candidate_expert,
        )
        for baseline, candidate, seed, baseline_retrieval, candidate_retrieval, baseline_expert, candidate_expert in zip(
            baseline_dirs, candidate_dirs, seeds,
            baseline_retrieval_dirs, candidate_retrieval_dirs,
            baseline_expert_reports, candidate_expert_reports,
        )
    ]
    required = 2
    passed = sum(int(value["passed"]) for value in pairs)
    return {
        "protocol": "qpsalm_description_seed_gate_v2",
        "pairs": pairs,
        "num_passed": passed,
        "required_passed": required,
        "passed_2_of_3_gate": passed >= 2,
        "passed_fraction_gate": passed >= required,
    }
