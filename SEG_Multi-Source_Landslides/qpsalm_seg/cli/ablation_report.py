#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build paired instruction/visual-evidence ablation evidence from eval reports.

用途：比较同一 checkpoint 的 normal、instruction ablation 和 visual ablation 评估结果。
推荐命令：PYTHONPATH=SEG_Multi-Source_Landslides python -m
qpsalm_seg.cli.ablation_report --normal EVAL_NORMAL --instruction-shuffled EVAL_SHUFFLED
--instruction-fixed-generic EVAL_FIXED --instruction-no-semantic EVAL_NO_SEMANTIC
--visual-shuffled EVAL_VIS_SHUFFLED --visual-text-only EVAL_TEXT_ONLY
--visual-remove terrain=EVAL_REMOVE_TERRAIN --output outputs/ablation_evidence.json
主要输入：每个 eval 目录中的 eval_report.json 与 eval_manifest.json。
主要输出：逐样本成对指标退化、instruction sensitivity 退化和准入结论 JSON。
写入行为：只写 --output，不修改 checkpoint、eval 结果或 benchmark。
所属流程：Qwen/condition/visual evidence 真实性验证和三 seed 正式实验之前的科学门槛。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

from qpsalm_seg.paths import resolve_project_path
from qpsalm_seg.schema import MODALITY_FAMILIES
from qpsalm_seg.description.protocols.io import strict_json_loads


REPORT_FORMAT = "qpsalm_ablation_evidence_v1"
PAIRED_FIELDS = (
    "final_dice",
    "final_iou",
    "selected_is_matched",
    "matched_relevance_rank_score",
    "component_recall",
    "proposal_union_dice",
)
INSTRUCTION_FIELDS = (
    "instruction_contrast_ratio_16",
    "paired_prediction_difference_rate",
    "no_target_empty_prediction_rate",
    "no_target_mean_unmatched_rejection",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build strict paired QPSALM ablation evidence.")
    parser.add_argument("--normal", required=True)
    parser.add_argument("--instruction-shuffled", required=True)
    parser.add_argument("--instruction-fixed-generic", required=True)
    parser.add_argument("--instruction-no-semantic", required=True)
    parser.add_argument("--visual-shuffled", required=True)
    parser.add_argument("--visual-text-only", required=True)
    parser.add_argument(
        "--visual-remove",
        action="append",
        required=True,
        metavar="FAMILY=EVAL",
        help="至少提供一个 remove:<family> 评估目录，可重复。",
    )
    parser.add_argument("--image-text-delta", default=None, help="可选策略报告，不参与退化门槛。")
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _json(path: Path) -> dict[str, Any]:
    value = strict_json_loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON 顶层必须是 object: {path}")
    return value


def load_eval_bundle(path_ref: str | Path) -> dict[str, Any]:
    path = resolve_project_path(path_ref) or Path(path_ref)
    report_path = path / "eval_report.json" if path.is_dir() else path
    manifest_path = report_path.parent / "eval_manifest.json"
    if not report_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            f"ablation 需要同目录 eval_report.json/eval_manifest.json: {report_path.parent}"
        )
    report, manifest = _json(report_path), _json(manifest_path)
    records = ((report.get("proposal_diagnostics") or {}).get("records") or [])
    if not isinstance(records, list) or not records:
        raise ValueError(f"eval report 缺少 proposal_diagnostics.records: {report_path}")
    by_sample = {}
    for record in records:
        sample_id = str(record.get("sample_id") or "")
        if not sample_id or sample_id in by_sample:
            raise ValueError(f"ablation sample_id 缺失或重复: {report_path} sample={sample_id!r}")
        by_sample[sample_id] = record
    return {
        "path": str(report_path),
        "report": report,
        "manifest": manifest,
        "records": by_sample,
    }


def _condition(bundle: dict[str, Any]) -> tuple[str, str]:
    config = bundle["manifest"].get("resolved_config") or {}
    return str(config.get("instruction_ablation") or "normal"), str(config.get("visual_ablation") or "normal")


def _assert_condition(bundle: dict[str, Any], instruction: str, visual: str) -> None:
    observed = _condition(bundle)
    if observed != (instruction, visual):
        raise ValueError(
            f"ablation condition 不匹配: report={bundle['path']} expected={(instruction, visual)} "
            f"observed={observed}"
        )


def _assert_comparable(normal: dict[str, Any], candidate: dict[str, Any]) -> None:
    fields = ("checkpoint", "checkpoint_step", "split", "preset")
    mismatched = {
        field: (normal["manifest"].get(field), candidate["manifest"].get(field))
        for field in fields
        if normal["manifest"].get(field) != candidate["manifest"].get(field)
    }
    if mismatched:
        raise ValueError(f"ablation reports 不是同一模型/数据协议: {mismatched}")
    normal_ids, candidate_ids = set(normal["records"]), set(candidate["records"])
    if normal_ids != candidate_ids:
        raise ValueError(
            f"ablation sample coverage 不一致: normal={len(normal_ids)} candidate={len(candidate_ids)}"
        )


def _number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def paired_comparison(
    normal: dict[str, Any],
    candidate: dict[str, Any],
    *,
    family_filter: str | None = None,
    min_delta: float = 0.0,
) -> dict[str, Any]:
    _assert_comparable(normal, candidate)
    sample_ids = sorted(normal["records"])
    if family_filter is not None:
        sample_ids = [
            sample_id
            for sample_id in sample_ids
            if family_filter in set(str(normal["records"][sample_id].get("family_combo") or "").split("+"))
        ]
    if not sample_ids:
        raise ValueError(f"ablation 没有 family={family_filter!r} 的成对样本")
    metric_deltas = {}
    for field in PAIRED_FIELDS:
        pairs = [
            (
                _number(normal["records"][sample_id].get(field)),
                _number(candidate["records"][sample_id].get(field)),
            )
            for sample_id in sample_ids
        ]
        pairs = [(left, right) for left, right in pairs if left is not None and right is not None]
        if pairs:
            normal_mean = mean(left for left, _right in pairs)
            candidate_mean = mean(right for _left, right in pairs)
            metric_deltas[field] = {
                "normal": normal_mean,
                "ablation": candidate_mean,
                "normal_minus_ablation": normal_mean - candidate_mean,
                "n": len(pairs),
            }
    if not metric_deltas:
        raise ValueError("ablation 没有可比较的逐样本指标")
    composite_delta = mean(value["normal_minus_ablation"] for value in metric_deltas.values())
    return {
        "n": len(sample_ids),
        "family_filter": family_filter,
        "metrics": metric_deltas,
        "composite_delta": composite_delta,
        "passed": composite_delta > float(min_delta),
    }


def instruction_comparison(
    normal: dict[str, Any],
    candidate: dict[str, Any],
    *,
    min_delta: float,
) -> dict[str, Any]:
    paired = paired_comparison(normal, candidate, min_delta=min_delta)
    normal_summary = normal["report"].get("instruction_sensitivity") or {}
    candidate_summary = candidate["report"].get("instruction_sensitivity") or {}
    sensitivity = {}
    for field in INSTRUCTION_FIELDS:
        left, right = _number(normal_summary.get(field)), _number(candidate_summary.get(field))
        if left is not None and right is not None:
            sensitivity[field] = {
                "normal": left,
                "ablation": right,
                "normal_minus_ablation": left - right,
            }
    if not sensitivity:
        raise ValueError("instruction ablation 缺少 paired/no-target sensitivity 指标")
    sensitivity_delta = mean(value["normal_minus_ablation"] for value in sensitivity.values())
    combined_delta = mean((paired["composite_delta"], sensitivity_delta))
    return {
        **paired,
        "instruction_sensitivity": sensitivity,
        "instruction_sensitivity_delta": sensitivity_delta,
        "combined_delta": combined_delta,
        "passed": combined_delta > float(min_delta),
    }


def _parse_removals(values: list[str]) -> dict[str, str]:
    output = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--visual-remove 必须是 FAMILY=EVAL: {value!r}")
        family, path = value.split("=", 1)
        family = family.strip()
        if family not in MODALITY_FAMILIES or family in output or not path.strip():
            raise ValueError(f"非法或重复 visual remove: {value!r}")
        output[family] = path.strip()
    return output


def build_ablation_report(
    normal: dict[str, Any],
    instruction_bundles: dict[str, dict[str, Any]],
    visual_bundles: dict[str, tuple[dict[str, Any], str | None]],
    *,
    strategy_bundle: dict[str, Any] | None = None,
    min_delta: float = 0.0,
) -> dict[str, Any]:
    _assert_condition(normal, "normal", "normal")
    expected_instruction = {"shuffled", "fixed-generic", "no-semantic"}
    if set(instruction_bundles) != expected_instruction:
        raise ValueError(f"instruction ablation 不完整: {sorted(instruction_bundles)}")
    instruction_results = {}
    for name, bundle in sorted(instruction_bundles.items()):
        _assert_condition(bundle, name, "normal")
        instruction_results[name] = instruction_comparison(
            normal, bundle, min_delta=min_delta
        )
    required_visual = {"shuffled", "text-only"}
    if not required_visual.issubset(visual_bundles) or not any(
        name.startswith("remove:") for name in visual_bundles
    ):
        raise ValueError("visual ablation 必须包含 shuffled、text-only 和至少一个 remove:<family>")
    visual_results = {}
    for name, (bundle, family_filter) in sorted(visual_bundles.items()):
        _assert_condition(bundle, "normal", name)
        visual_results[name] = paired_comparison(
            normal, bundle, family_filter=family_filter, min_delta=min_delta
        )
    strategy = None
    if strategy_bundle is not None:
        _assert_condition(strategy_bundle, "normal", "image-text-delta")
        strategy = paired_comparison(normal, strategy_bundle, min_delta=-float("inf"))
        strategy.pop("passed", None)
    all_checks = [
        value["passed"] for value in (*instruction_results.values(), *visual_results.values())
    ]
    return {
        "format": REPORT_FORMAT,
        "normal_report": normal["path"],
        "checkpoint": normal["manifest"].get("checkpoint"),
        "checkpoint_step": normal["manifest"].get("checkpoint_step"),
        "split": normal["manifest"].get("split"),
        "preset": normal["manifest"].get("preset"),
        "min_delta": float(min_delta),
        "instruction": instruction_results,
        "visual": visual_results,
        "image_text_delta_strategy": strategy,
        "acceptance": {
            "passed": bool(all_checks) and all(all_checks),
            "num_checks": len(all_checks),
            "num_passed": sum(all_checks),
            "criterion": "normal composite evidence score must exceed every ablation",
        },
    }


def _write(path_ref: str, payload: dict[str, Any]) -> Path:
    path = resolve_project_path(path_ref) or Path(path_ref)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(
                payload, ensure_ascii=False, indent=2, allow_nan=False
            ) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def main() -> None:
    args = parse_args()
    normal = load_eval_bundle(args.normal)
    instruction = {
        "shuffled": load_eval_bundle(args.instruction_shuffled),
        "fixed-generic": load_eval_bundle(args.instruction_fixed_generic),
        "no-semantic": load_eval_bundle(args.instruction_no_semantic),
    }
    visual = {
        "shuffled": (load_eval_bundle(args.visual_shuffled), None),
        "text-only": (load_eval_bundle(args.visual_text_only), None),
    }
    for family, path in _parse_removals(args.visual_remove).items():
        visual[f"remove:{family}"] = (load_eval_bundle(path), family)
    strategy = load_eval_bundle(args.image_text_delta) if args.image_text_delta else None
    report = build_ablation_report(
        normal,
        instruction,
        visual,
        strategy_bundle=strategy,
        min_delta=args.min_delta,
    )
    output = _write(args.output, report)
    print(json.dumps({"output": str(output), **report["acceptance"]}, ensure_ascii=False))
    if not report["acceptance"]["passed"]:
        raise SystemExit("ablation evidence gate failed")


if __name__ == "__main__":
    main()
