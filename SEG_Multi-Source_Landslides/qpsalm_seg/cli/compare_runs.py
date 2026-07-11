#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""比较两个 QPSALM 训练运行的指标。

用途：比较 baseline/candidate 的 overall、模态分组、loss 和 proposal diagnostics。
推荐运行命令：PYTHONPATH=SEG_Multi-Source_Landslides python -m
qpsalm_seg.cli.compare_runs --baseline-summary outputs/BASELINE
--candidate-summary outputs/CANDIDATE --output outputs/comparison.json
主要输入：两个 run_summary.json 或包含 run_summary.json 的运行目录。
主要输出：comparison JSON。
写入行为：只写可选 --output，不修改 checkpoint 或 benchmark。
所属流程：SANE/QMEF/PMRD preset 消融比较。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from qpsalm_seg.paths import resolve_repo_path


METRIC_KEYS = ["dice", "iou", "precision", "recall", "loss", "n"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two QPSALM run summaries.")
    parser.add_argument("--baseline-summary", required=True)
    parser.add_argument("--candidate-summary", required=True)
    parser.add_argument("--baseline-name", default="baseline")
    parser.add_argument("--candidate-name", default="candidate")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def resolve_summary_path(path_ref: str | Path) -> Path:
    path = resolve_repo_path(path_ref)
    if path is None:
        raise FileNotFoundError(path_ref)
    if path.is_dir():
        path = path / "run_summary.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def read_run_summary(path_ref: str | Path) -> dict[str, Any]:
    path = resolve_summary_path(path_ref)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"run summary 必须是 JSON object: {path}")
    data["_summary_path"] = str(path)
    return data


def choose_metric_block(summary: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """优先比较 eval；没有 eval 时退回 validation。"""
    eval_block = summary.get("eval")
    if isinstance(eval_block, dict) and eval_block.get("overall"):
        return "eval", eval_block
    val_block = summary.get("validation")
    if isinstance(val_block, dict) and val_block.get("overall"):
        return "validation", val_block
    return "none", {}


def numeric_delta(base: Any, cand: Any) -> float | None:
    if isinstance(base, (int, float)) and isinstance(cand, (int, float)):
        return float(cand) - float(base)
    return None


def compare_metric_dict(base: dict[str, Any] | None, cand: dict[str, Any] | None) -> dict[str, Any]:
    """比较两个 metric dict，保留 baseline/candidate/delta。"""
    base = base or {}
    cand = cand or {}
    keys = sorted(set(base) | set(cand))
    out: dict[str, Any] = {}
    for key in keys:
        if key not in METRIC_KEYS and not isinstance(base.get(key, cand.get(key)), (int, float)):
            continue
        out[key] = {
            "baseline": base.get(key),
            "candidate": cand.get(key),
            "delta": numeric_delta(base.get(key), cand.get(key)),
        }
    return out


def compare_group_maps(base: dict[str, Any] | None, cand: dict[str, Any] | None) -> dict[str, Any]:
    base = base or {}
    cand = cand or {}
    groups = sorted(set(base) | set(cand))
    return {
        group: compare_metric_dict(
            base.get(group) if isinstance(base.get(group), dict) else None,
            cand.get(group) if isinstance(cand.get(group), dict) else None,
        )
        for group in groups
    }


def compare_loss_components(base: dict[str, Any] | None, cand: dict[str, Any] | None) -> dict[str, Any]:
    base = base or {}
    cand = cand or {}
    keys = sorted(set(base) | set(cand))
    out = {}
    for key in keys:
        out[key] = {
            "baseline": base.get(key),
            "candidate": cand.get(key),
            "delta": numeric_delta(base.get(key), cand.get(key)),
        }
    return out


def proposal_summary_from_block(block: dict[str, Any]) -> dict[str, Any]:
    diagnostics = block.get("proposal_diagnostics") if isinstance(block, dict) else None
    if not isinstance(diagnostics, dict):
        return {}
    summary = diagnostics.get("summary")
    return summary if isinstance(summary, dict) else {}


def proposal_primary_deltas(base_summary: dict[str, Any], cand_summary: dict[str, Any]) -> dict[str, Any]:
    base_overall = base_summary.get("overall") if isinstance(base_summary.get("overall"), dict) else {}
    cand_overall = cand_summary.get("overall") if isinstance(cand_summary.get("overall"), dict) else {}
    fields = [
        "mean_selected_matches_best",
        "mean_selected_query_dice",
        "mean_best_query_dice",
        "mean_dice_gap_selected_minus_best",
        "mean_selected_relevance_logit",
        "mean_best_relevance_logit",
        "mean_best_query_relevance_rank",
        "mean_relevance_gap_selected_minus_best",
        "mean_final_dice",
        "mean_final_iou",
    ]
    out: dict[str, Any] = {}
    for field in fields:
        if field in base_overall or field in cand_overall:
            out[field] = {
                "baseline": base_overall.get(field),
                "candidate": cand_overall.get(field),
                "delta": numeric_delta(base_overall.get(field), cand_overall.get(field)),
            }
    return out


def compare_run_summaries(
    baseline_summary: dict[str, Any],
    candidate_summary: dict[str, Any],
    baseline_name: str = "baseline",
    candidate_name: str = "candidate",
) -> dict[str, Any]:
    base_source, base_block = choose_metric_block(baseline_summary)
    cand_source, cand_block = choose_metric_block(candidate_summary)
    overall = compare_metric_dict(base_block.get("overall"), cand_block.get("overall"))
    proposal_summary_base = proposal_summary_from_block(base_block)
    proposal_summary_cand = proposal_summary_from_block(cand_block)
    proposal_summary_delta = compare_group_maps(proposal_summary_base, proposal_summary_cand)
    dice_delta = (overall.get("dice") or {}).get("delta")
    iou_delta = (overall.get("iou") or {}).get("delta")
    proposal_deltas = proposal_primary_deltas(proposal_summary_base, proposal_summary_cand)
    return {
        "baseline_name": baseline_name,
        "candidate_name": candidate_name,
        "baseline_summary": baseline_summary.get("_summary_path") or baseline_summary.get("run_dir"),
        "candidate_summary": candidate_summary.get("_summary_path") or candidate_summary.get("run_dir"),
        "metric_sources": {
            "baseline": base_source,
            "candidate": cand_source,
        },
        "acceptance": {
            "baseline_pipeline_ready": (baseline_summary.get("acceptance") or {}).get("research_pipeline_ready"),
            "candidate_pipeline_ready": (candidate_summary.get("acceptance") or {}).get("research_pipeline_ready"),
        },
        "overall": overall,
        "primary_deltas": {
            "dice": dice_delta,
            "iou": iou_delta,
            "proposal_selected_matches_best": (
                proposal_deltas.get("mean_selected_matches_best") or {}
            ).get("delta"),
            "proposal_selected_query_dice": (
                proposal_deltas.get("mean_selected_query_dice") or {}
            ).get("delta"),
        },
        "canonical_combos": compare_group_maps(
            base_block.get("canonical_combos") if isinstance(base_block, dict) else None,
            cand_block.get("canonical_combos") if isinstance(cand_block, dict) else None,
        ),
        "raw_combos": compare_group_maps(
            base_block.get("raw_combos") if isinstance(base_block, dict) else None,
            cand_block.get("raw_combos") if isinstance(cand_block, dict) else None,
        ),
        "loss_components": compare_loss_components(
            base_block.get("loss_components") if isinstance(base_block, dict) else None,
            cand_block.get("loss_components") if isinstance(cand_block, dict) else None,
        ),
        "proposal_diagnostics": {
            "summary": proposal_summary_delta,
            "primary_deltas": proposal_deltas,
        },
    }


def write_json(path_ref: str | Path, payload: dict[str, Any]) -> Path:
    path = resolve_repo_path(path_ref)
    if path is None:
        raise FileNotFoundError(path_ref)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def main() -> None:
    args = parse_args()
    baseline = read_run_summary(args.baseline_summary)
    candidate = read_run_summary(args.candidate_summary)
    comparison = compare_run_summaries(
        baseline,
        candidate,
        baseline_name=args.baseline_name,
        candidate_name=args.candidate_name,
    )
    if args.output:
        output_path = write_json(args.output, comparison)
        comparison["output"] = str(output_path)
    print(json.dumps(comparison, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
