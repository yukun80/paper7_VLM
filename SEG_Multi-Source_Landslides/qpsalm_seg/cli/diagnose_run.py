#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""诊断 QPSALM 运行结果中的低精度原因。

脚本作用：读取 run_summary.json，自动检查验证覆盖、threshold sweep、proposal
选择、modality gate、precision/recall 结构和 checkpoint 产物，输出可操作建议。
主要输入：一个 run 目录或 run_summary.json。
主要输出：诊断 JSON，可选写入 diagnose_report.json。
是否改写原始数据：只在指定 --output 时写诊断报告，不改 checkpoint 或 benchmark。
典型用法：python -m qpsalm_seg.cli.diagnose_run --run outputs/.../baseline。
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from qpsalm_seg.data import CANONICAL_MODALITIES, resolve_repo_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose a QPSALM run summary.")
    parser.add_argument("--run", required=True, help="Path to run directory or run_summary.json.")
    parser.add_argument("--output", default=None, help="Optional diagnose_report.json path.")
    parser.add_argument("--low-dice", type=float, default=0.45)
    parser.add_argument("--low-iou", type=float, default=0.30)
    parser.add_argument("--low-recall", type=float, default=0.40)
    parser.add_argument("--threshold-delta", type=float, default=0.03)
    parser.add_argument("--query-collapse-frac", type=float, default=0.80)
    parser.add_argument("--gate-collapse-weight", type=float, default=0.85)
    parser.add_argument("--query-gate-collapse-peak", type=float, default=0.85)
    parser.add_argument("--weak-evidence-score-gap", type=float, default=0.05)
    parser.add_argument("--empty-fp-area", type=float, default=128.0)
    return parser.parse_args()


def default_diagnose_args(**overrides: Any) -> argparse.Namespace:
    """返回与 CLI 默认值一致的诊断参数，便于 run_phase1 自动生成报告。"""
    values: dict[str, Any] = {
        "low_dice": 0.45,
        "low_iou": 0.30,
        "low_recall": 0.40,
        "threshold_delta": 0.03,
        "query_collapse_frac": 0.80,
        "gate_collapse_weight": 0.85,
        "query_gate_collapse_peak": 0.85,
        "weak_evidence_score_gap": 0.05,
        "empty_fp_area": 128.0,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def resolve_summary_path(path_ref: str | Path) -> Path:
    path = resolve_repo_path(path_ref)
    if path is None:
        raise FileNotFoundError(path_ref)
    if path.is_dir():
        path = path / "run_summary.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON 顶层必须是 object: {path}")
    return payload


def metric_block(summary: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """优先诊断 eval，其次 validation_best，再其次 validation。"""
    for name in ["eval", "validation_best", "validation"]:
        block = summary.get(name)
        if isinstance(block, dict) and isinstance(block.get("overall"), dict):
            return name, block
    return "none", {}


def issue(severity: str, code: str, message: str, evidence: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "severity": severity,
        "code": code,
        "message": message,
        "evidence": evidence or {},
    }


def numeric(value: Any, default: float = 0.0) -> float:
    return float(value) if isinstance(value, (int, float)) else default


def check_artifacts(summary: dict[str, Any]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    acceptance = summary.get("acceptance") if isinstance(summary.get("acceptance"), dict) else {}
    if not acceptance.get("phase1_smoke_ready"):
        issues.append(issue("error", "phase1_not_ready", "phase1_smoke_ready 不是 true。", acceptance))
    checkpoint = summary.get("checkpoint_best") if isinstance(summary.get("checkpoint_best"), dict) else {}
    if not checkpoint.get("exists"):
        issues.append(issue("warning", "missing_best_checkpoint", "缺少 checkpoint_best.pt，最终 eval 可能不是最佳验证点。", checkpoint))
    artifacts = summary.get("artifacts") if isinstance(summary.get("artifacts"), dict) else {}
    validation_best = artifacts.get("validation_best") if isinstance(artifacts.get("validation_best"), dict) else {}
    if not validation_best.get("exists"):
        issues.append(issue("warning", "missing_validation_best", "缺少 validation_best.json，无法追踪最佳验证指标。", validation_best))
    return issues


def check_validation_coverage(block_name: str, block: dict[str, Any], config: dict[str, Any]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    overall = block.get("overall") if isinstance(block.get("overall"), dict) else {}
    n = numeric(overall.get("n"))
    max_val_samples = config.get("max_val_samples")
    max_val_batches = config.get("max_val_batches")
    if isinstance(max_val_batches, (int, float)) and int(max_val_batches) > 0:
        issues.append(
            issue(
                "warning",
                "validation_truncated_by_batches",
                "验证集被 max_val_batches 截断，分组指标可能不代表完整多源验证集。",
                {"metric_block": block_name, "n": n, "max_val_batches": max_val_batches},
            )
        )
    if isinstance(max_val_samples, (int, float)) and max_val_samples > 0 and n < 0.8 * float(max_val_samples):
        issues.append(
            issue(
                "warning",
                "validation_sample_count_low",
                "实际评估样本数明显少于 max_val_samples，需确认是否被 batch 或 index 截断。",
                {"metric_block": block_name, "n": n, "max_val_samples": max_val_samples},
            )
        )
    canonical = block.get("canonical_combos") if isinstance(block.get("canonical_combos"), dict) else {}
    if len(canonical) <= 1:
        issues.append(
            issue(
                "warning",
                "limited_modality_coverage",
                "当前指标只覆盖很少的 canonical modality combo，不能代表完整多源能力。",
                {"num_canonical_combos": len(canonical), "groups": sorted(canonical.keys())},
            )
        )
    target_area_bins = block.get("target_area_px_bins") if isinstance(block.get("target_area_px_bins"), dict) else {}
    if not target_area_bins:
        issues.append(
            issue(
                "info",
                "missing_target_area_strata",
                "缺少 target area 分层指标，难以判断小滑坡斑块是否是主要误差来源。",
                {"metric_block": block_name},
            )
        )
    elif len(target_area_bins) <= 1:
        issues.append(
            issue(
                "info",
                "limited_target_area_strata",
                "当前验证只覆盖一个目标面积分层，小目标/大目标泛化结论不充分。",
                {"metric_block": block_name, "groups": sorted(target_area_bins.keys())},
            )
        )
    gsd_tokens = block.get("gsd_tokens") if isinstance(block.get("gsd_tokens"), dict) else {}
    if not gsd_tokens:
        issues.append(
            issue(
                "info",
                "missing_gsd_strata",
                "缺少 GSD 分层指标，无法判断尺度差异对分割性能的影响。",
                {"metric_block": block_name},
            )
        )
    elif all(str(name).endswith("=unknown") for name in gsd_tokens):
        issues.append(
            issue(
                "info",
                "gsd_unknown_or_missing",
                "当前验证样本的 GSD 全部为 unknown，scale-aware 模块只能依赖弱文本/数据集线索。",
                {"metric_block": block_name, "groups": sorted(gsd_tokens.keys())},
            )
        )
    ground_area_bins = block.get("ground_area_m2_bins") if isinstance(block.get("ground_area_m2_bins"), dict) else {}
    if ground_area_bins and all(str(name).endswith("=unknown") for name in ground_area_bins):
        issues.append(
            issue(
                "info",
                "ground_area_unknown",
                "缺少可估算地面面积的样本，无法进行真实尺度下的小/大滑坡性能比较。",
                {"metric_block": block_name, "groups": sorted(ground_area_bins.keys())},
            )
        )
    return issues


def check_metric_shape(block_name: str, block: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    overall = block.get("overall") if isinstance(block.get("overall"), dict) else {}
    dice = numeric(overall.get("dice"))
    iou = numeric(overall.get("iou"))
    precision = numeric(overall.get("precision"))
    recall = numeric(overall.get("recall"))
    if dice < args.low_dice:
        issues.append(issue("warning", "low_dice", "Dice 偏低，需要继续改进模型或阈值。", {"block": block_name, "dice": dice}))
    if iou < args.low_iou:
        issues.append(issue("warning", "low_iou", "IoU 偏低，需要检查召回、边界和假阳性。", {"block": block_name, "iou": iou}))
    if recall < args.low_recall:
        issues.append(issue("warning", "low_recall", "Recall 偏低，预测 mask 可能偏小或阈值偏高。", {"block": block_name, "recall": recall, "precision": precision}))
    if precision > recall + 0.15:
        issues.append(issue("info", "precision_recall_imbalance", "Precision 明显高于 recall，模型偏保守。", {"precision": precision, "recall": recall}))
    return issues


def check_threshold_sweep(block: dict[str, Any], args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[str]]:
    issues: list[dict[str, Any]] = []
    recommendations: list[str] = []
    overall = block.get("overall") if isinstance(block.get("overall"), dict) else {}
    sweep = block.get("threshold_sweep") if isinstance(block.get("threshold_sweep"), dict) else {}
    best = sweep.get("best_by_dice") if isinstance(sweep.get("best_by_dice"), dict) else {}
    current_threshold = numeric(block.get("threshold"), 0.5)
    current_dice = numeric(overall.get("dice"))
    best_threshold = numeric(best.get("threshold"), current_threshold)
    best_dice = numeric(best.get("dice"), current_dice)
    if best and best_dice >= current_dice + float(args.threshold_delta):
        issues.append(
            issue(
                "info",
                "threshold_can_improve_dice",
                "threshold sweep 显示调阈值可提升 Dice。",
                {
                    "current_threshold": current_threshold,
                    "current_dice": current_dice,
                    "best_threshold": best_threshold,
                    "best_dice": best_dice,
                },
            )
        )
        recommendations.append(f"尝试设置 EVAL_THRESHOLD={best_threshold:.2f} 重新 eval；若稳定，再作为默认推理阈值。")
    elif not sweep:
        issues.append(issue("info", "missing_threshold_sweep", "报告中缺少 threshold_sweep，无法判断阈值校准收益。"))
    per_group = sweep.get("best_by_dice_per_group") if isinstance(sweep.get("best_by_dice_per_group"), dict) else {}
    canonical_thresholds: dict[str, float] = {}
    for group, values in per_group.items():
        if not isinstance(group, str) or not group.startswith("canonical_combo=") or not isinstance(values, dict):
            continue
        threshold = values.get("threshold")
        if isinstance(threshold, (int, float)):
            canonical_thresholds[group] = float(threshold)
    if len(set(canonical_thresholds.values())) >= 2:
        issues.append(
            issue(
                "info",
                "combo_specific_thresholds",
                "不同 canonical modality combo 的最佳阈值不同，可能需要按模态组合校准。",
                canonical_thresholds,
            )
        )
        recommendations.append("查看 threshold_sweep.csv 的 canonical_combo 行，判断是否需要为不同模态组合报告 calibrated metrics。")
    return issues, recommendations


def proposal_records(block: dict[str, Any]) -> list[dict[str, Any]]:
    diagnostics = block.get("proposal_diagnostics") if isinstance(block.get("proposal_diagnostics"), dict) else {}
    records = diagnostics.get("records")
    return [item for item in records if isinstance(item, dict)] if isinstance(records, list) else []


def mean_numeric_field(records: list[dict[str, Any]], field: str) -> float | None:
    values = [numeric(row.get(field)) for row in records if isinstance(row.get(field), (int, float))]
    if not values:
        return None
    return sum(values) / len(values)


def check_proposals(block: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    records = proposal_records(block)
    if not records:
        return [issue("info", "missing_proposal_records", "缺少 proposal_diagnostics.records，无法分析 query 选择。")]
    selected = [int(row.get("selected_query")) for row in records if isinstance(row.get("selected_query"), int)]
    if selected:
        counts = Counter(selected)
        query, count = counts.most_common(1)[0]
        frac = count / max(1, len(selected))
        if frac >= float(args.query_collapse_frac):
            issues.append(
                issue(
                    "warning",
                    "query_selection_collapse",
                    "selected query 过度集中，multi-mask proposal 多样性不足。",
                    {"top_query": query, "fraction": frac, "num_records": len(selected)},
                )
            )
    mean_match = mean_numeric_field(records, "selected_matches_best")
    if mean_match is not None:
        if mean_match < 0.70:
            issues.append(
                issue(
                    "warning",
                    "combined_selector_misses_best_query",
                    "combined selector 经常没有选择 Dice 最好的 proposal。",
                    {"mean_selected_matches_best": mean_match},
                )
            )
    condition_match = mean_numeric_field(records, "condition_top_matches_best")
    if condition_match is not None and condition_match < 0.70:
        issues.append(
            issue(
                "info",
                "condition_verifier_misses_best_query",
                "condition verifier 单独打分时经常没有把 Dice 最优 query 排在第一。",
                {"mean_condition_top_matches_best": condition_match},
            )
        )
    return issues


def check_evidence_verifier(block: dict[str, Any], config: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    """检查 evidence verifier 是否记录完整，以及是否能偏向更好的 proposal。"""
    issues: list[dict[str, Any]] = []
    use_evidence = bool(config.get("use_evidence_reasoning", True))
    selection_weight = numeric(config.get("selection_evidence_weight"))
    cls_weight = numeric(config.get("evidence_cls_weight"))
    rank_weight = numeric(config.get("evidence_ranking_loss_weight"))
    if not use_evidence and selection_weight <= 0 and cls_weight <= 0 and rank_weight <= 0:
        return issues
    records = proposal_records(block)
    if not records:
        return issues
    has_evidence = any(isinstance(row.get("selected_evidence_score"), (int, float)) for row in records)
    if not has_evidence:
        issues.append(
            issue(
                "warning",
                "missing_evidence_scores",
                "当前配置启用了 evidence reasoning/selection/loss，但 proposal diagnostics 缺少 evidence score 字段。",
                {
                    "use_evidence_reasoning": use_evidence,
                    "selection_evidence_weight": selection_weight,
                    "evidence_cls_weight": cls_weight,
                    "evidence_ranking_loss_weight": rank_weight,
                },
            )
        )
        return issues
    gaps = [
        numeric(row.get("evidence_score_gap_selected_minus_best"))
        for row in records
        if isinstance(row.get("evidence_score_gap_selected_minus_best"), (int, float))
    ]
    if gaps:
        mean_gap = sum(gaps) / len(gaps)
        if mean_gap < -float(args.weak_evidence_score_gap):
            issues.append(
                issue(
                    "warning",
                    "evidence_verifier_penalizes_best_query",
                    "evidence scorer 平均更偏向 selected query 而不是 Dice 最优 query，可能拖累 proposal selection。",
                    {"mean_evidence_score_gap_selected_minus_best": mean_gap, "num_records": len(gaps)},
                )
            )
        elif abs(mean_gap) <= float(args.weak_evidence_score_gap):
            issues.append(
                issue(
                    "info",
                    "weak_evidence_verifier_separation",
                    "evidence scorer 对 selected/best query 区分度较弱，当前可能只是弱 verifier。",
                    {"mean_evidence_score_gap_selected_minus_best": mean_gap, "num_records": len(gaps)},
                )
            )
    evidence_match = mean_numeric_field(records, "evidence_top_matches_best")
    if evidence_match is not None and evidence_match < 0.70:
        issues.append(
            issue(
                "info",
                "evidence_verifier_misses_best_query",
                "evidence verifier 单独打分时经常没有把 Dice 最优 query 排在第一。",
                {"mean_evidence_top_matches_best": evidence_match},
            )
        )
    return issues


def check_visual_evidence_verifier(
    block: dict[str, Any],
    config: dict[str, Any],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    """检查 visual evidence verifier 是否记录完整，以及是否能偏向更好的 proposal。"""
    issues: list[dict[str, Any]] = []
    use_visual = bool(config.get("use_visual_evidence", True))
    selection_weight = numeric(config.get("selection_visual_evidence_weight"))
    cls_weight = numeric(config.get("visual_evidence_cls_weight"))
    rank_weight = numeric(config.get("visual_evidence_ranking_loss_weight"))
    if not use_visual and selection_weight <= 0 and cls_weight <= 0 and rank_weight <= 0:
        return issues
    records = proposal_records(block)
    if not records:
        return issues
    has_visual = any(isinstance(row.get("selected_visual_evidence_score"), (int, float)) for row in records)
    if not has_visual:
        issues.append(
            issue(
                "warning",
                "missing_visual_evidence_scores",
                "当前配置启用了 visual evidence verifier，但 proposal diagnostics 缺少 visual evidence score 字段。",
                {
                    "use_visual_evidence": use_visual,
                    "selection_visual_evidence_weight": selection_weight,
                    "visual_evidence_cls_weight": cls_weight,
                    "visual_evidence_ranking_loss_weight": rank_weight,
                },
            )
        )
        return issues
    gaps = [
        numeric(row.get("visual_evidence_score_gap_selected_minus_best"))
        for row in records
        if isinstance(row.get("visual_evidence_score_gap_selected_minus_best"), (int, float))
    ]
    if gaps:
        mean_gap = sum(gaps) / len(gaps)
        if mean_gap < -float(args.weak_evidence_score_gap):
            issues.append(
                issue(
                    "warning",
                    "visual_evidence_verifier_penalizes_best_query",
                    "visual evidence verifier 平均更偏向 selected query 而不是 Dice 最优 query，可能拖累 proposal selection。",
                    {"mean_visual_evidence_score_gap_selected_minus_best": mean_gap, "num_records": len(gaps)},
                )
            )
        elif abs(mean_gap) <= float(args.weak_evidence_score_gap):
            issues.append(
                issue(
                    "info",
                    "weak_visual_evidence_verifier_separation",
                    "visual evidence verifier 对 selected/best query 区分度较弱，当前可能只是弱 verifier。",
                    {"mean_visual_evidence_score_gap_selected_minus_best": mean_gap, "num_records": len(gaps)},
                )
            )
    visual_match = mean_numeric_field(records, "visual_evidence_top_matches_best")
    if visual_match is not None and visual_match < 0.70:
        issues.append(
            issue(
                "info",
                "visual_evidence_verifier_misses_best_query",
                "visual evidence verifier 单独打分时经常没有把 Dice 最优 query 排在第一。",
                {"mean_visual_evidence_top_matches_best": visual_match},
            )
        )
    return issues


def check_empty_false_positives(block: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    """检查 negative/empty GT 样本是否产生大量假阳性面积。"""
    diagnostics = block.get("proposal_diagnostics") if isinstance(block.get("proposal_diagnostics"), dict) else {}
    summary = diagnostics.get("summary") if isinstance(diagnostics.get("summary"), dict) else {}
    issues: list[dict[str, Any]] = []
    for group, payload in summary.items():
        if not isinstance(payload, dict):
            continue
        target_area = numeric(payload.get("mean_target_area"))
        final_area = numeric(payload.get("mean_final_mask_area"))
        selected_area = numeric(payload.get("mean_selected_mask_area"))
        n = numeric(payload.get("n"))
        if n <= 0 or target_area > 1.0:
            continue
        if final_area >= float(args.empty_fp_area) or selected_area >= float(args.empty_fp_area):
            issues.append(
                issue(
                    "warning",
                    "empty_mask_false_positives",
                    "空 GT 样本仍预测出明显前景面积，会直接拉低 precision。",
                    {
                        "group": group,
                        "n": n,
                        "mean_target_area": target_area,
                        "mean_final_mask_area": final_area,
                        "mean_selected_mask_area": selected_area,
                    },
                )
            )
    return issues


def check_gates(block: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    gates = block.get("modality_gate_summary") if isinstance(block.get("modality_gate_summary"), dict) else {}
    for group, payload in gates.items():
        if not isinstance(payload, dict):
            continue
        weights = payload.get("mean_weights") if isinstance(payload.get("mean_weights"), dict) else {}
        active = payload.get("mean_active") if isinstance(payload.get("mean_active"), dict) else {}
        active_count = sum(numeric(active.get(name)) for name in CANONICAL_MODALITIES)
        if active_count <= 1.2:
            continue
        top_name = None
        top_weight = -1.0
        for name in CANONICAL_MODALITIES:
            value = numeric(weights.get(name))
            if value > top_weight:
                top_name = name
                top_weight = value
        if top_weight >= float(args.gate_collapse_weight):
            issues.append(
                issue(
                    "info",
                    "modality_gate_collapse",
                    "多模态样本中 gate 过度偏向单一模态，需要确认是否合理。",
                    {"group": group, "top_modality": top_name, "top_weight": top_weight, "active_count": active_count},
                )
            )
    return issues


def check_query_modality_gates(block: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    """检查 query-level modality attention 是否存在，以及多源组合中是否塌缩。"""
    issues: list[dict[str, Any]] = []
    query_summary = block.get("query_modality_summary") if isinstance(block.get("query_modality_summary"), dict) else {}
    if not query_summary:
        return [issue("info", "missing_query_modality_summary", "缺少 query_modality_summary，无法判断 query 是否按 proposal 选择模态证据。")]
    for group, payload in query_summary.items():
        if not isinstance(group, str) or not group.startswith("canonical_combo=") or not isinstance(payload, dict):
            continue
        combo = group.split("=", 1)[1]
        if "+" not in combo:
            continue
        peak = numeric(payload.get("mean_peak"))
        weights = payload.get("mean_query_weights") if isinstance(payload.get("mean_query_weights"), dict) else {}
        top_name = None
        top_weight = -1.0
        for name in CANONICAL_MODALITIES:
            value = numeric(weights.get(name))
            if value > top_weight:
                top_name = name
                top_weight = value
        if peak >= float(args.query_gate_collapse_peak):
            issues.append(
                issue(
                    "info",
                    "query_modality_attention_collapse",
                    "多源组合中 query-level modality attention 过度集中，需确认是否只依赖单一证据源。",
                    {
                        "group": group,
                        "mean_peak": peak,
                        "top_modality": top_name,
                        "top_weight": top_weight,
                    },
                )
            )
    return issues


def build_recommendations(issues: list[dict[str, Any]]) -> list[str]:
    codes = {item["code"] for item in issues}
    recs: list[str] = []
    if "validation_truncated_by_batches" in codes or "limited_modality_coverage" in codes:
        recs.append("使用 MAX_VAL_BATCHES=0 并确认 index cache 为 round_robin_canonical_combo，重新生成完整多源验证报告。")
    if "low_recall" in codes or "precision_recall_imbalance" in codes:
        recs.append("优先检查 threshold_sweep；若低阈值有效，调低 EVAL_THRESHOLD；若仍低 recall，增大 Tversky beta 或 foreground BCE 权重。")
    if "query_selection_collapse" in codes:
        recs.append("考虑降低 condition/proposal 分类权重或加入 query 多样性约束，避免所有样本集中到同一 mask token。")
    if "empty_mask_false_positives" in codes:
        recs.append("启用 empty mask suppression，并用更高阈值 sweep 检查 negative-aware 样本的误报面积是否下降。")
    if "combined_selector_misses_best_query" in codes:
        recs.append("对比 proposal_diagnostics.csv 中 proposal/condition/evidence/visual_evidence 的 top query 命中率，调整各 SELECTION_*_WEIGHT。")
    if "condition_verifier_misses_best_query" in codes:
        recs.append("提高 condition ranking 监督质量，或降低 SELECTION_CONDITION_WEIGHT，必要时启用 FINAL_FOREGROUND_GATE_WEIGHT。")
    if "missing_evidence_scores" in codes:
        recs.append("重新运行 eval/summary，确认新版本 proposal_diagnostics 已包含 evidence score 字段。")
    if "evidence_verifier_penalizes_best_query" in codes or "weak_evidence_verifier_separation" in codes:
        recs.append("检查 proposal_diagnostics.csv 中 evidence_score_gap_selected_minus_best；必要时调低 SELECTION_EVIDENCE_WEIGHT 或调高 EVIDENCE_RANKING_LOSS_WEIGHT 做 ablation。")
    if "evidence_verifier_misses_best_query" in codes:
        recs.append("查看 evidence_top_matches_best 和 best_query_evidence_rank，判断 evidence verifier 是弱监督不足还是 selection 权重过高。")
    if "missing_visual_evidence_scores" in codes:
        recs.append("重新运行 eval/summary，确认 proposal_diagnostics 已包含 visual evidence score 字段。")
    if "visual_evidence_verifier_penalizes_best_query" in codes or "weak_visual_evidence_verifier_separation" in codes:
        recs.append("检查 proposal_diagnostics.csv 中 visual_evidence_score_gap_selected_minus_best；必要时调低 SELECTION_VISUAL_EVIDENCE_WEIGHT 或调高 VISUAL_EVIDENCE_RANKING_LOSS_WEIGHT 做 ablation。")
    if "visual_evidence_verifier_misses_best_query" in codes:
        recs.append("查看 visual_evidence_top_matches_best 和 best_query_visual_evidence_rank，判断 visual evidence cache/preview 是否真的提供了 query selection 信号。")
    if "modality_gate_collapse" in codes:
        recs.append("检查 gate 是否与 condition 合理对应；必要时加入 gate entropy warmup 或降低 modality dropout。")
    if "missing_query_modality_summary" in codes:
        recs.append("确认 USE_QUERY_MODALITY_ATTENTION=1 并重新 eval，导出 query_modality_gates.csv 后再比较多源组合。")
    if "query_modality_attention_collapse" in codes:
        recs.append("查看 query_modality_gates.csv；若 query gate 总是集中到单模态，可降低 QUERY_MODALITY_FEATURE_WEIGHT 或加入 query gate entropy warmup。")
    if "missing_target_area_strata" in codes or "limited_target_area_strata" in codes:
        recs.append("重新用新版 eval/summary 生成 target_area_px_bin 与 target_area_fraction_bin 分组，单独检查 tiny/small 滑坡斑块的 Dice/Recall。")
    if "missing_gsd_strata" in codes or "gsd_unknown_or_missing" in codes or "ground_area_unknown" in codes:
        recs.append("优先完善 benchmark 中的 gsd_m/resize_transform 元数据，再比较 gsd_token 与 ground_area_m2_bin 下的分割性能。")
    if "missing_best_checkpoint" in codes:
        recs.append("重新训练以生成 checkpoint_best.pt，并用 best checkpoint 做最终 eval。")
    return recs


def diagnose(summary: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    block_name, block = metric_block(summary)
    config = summary.get("config") if isinstance(summary.get("config"), dict) else {}
    issues: list[dict[str, Any]] = []
    recommendations: list[str] = []
    issues.extend(check_artifacts(summary))
    issues.extend(check_validation_coverage(block_name, block, config))
    issues.extend(check_metric_shape(block_name, block, args))
    sweep_issues, sweep_recs = check_threshold_sweep(block, args)
    issues.extend(sweep_issues)
    recommendations.extend(sweep_recs)
    issues.extend(check_proposals(block, args))
    issues.extend(check_evidence_verifier(block, config, args))
    issues.extend(check_visual_evidence_verifier(block, config, args))
    issues.extend(check_empty_false_positives(block, args))
    issues.extend(check_gates(block, args))
    issues.extend(check_query_modality_gates(block, args))
    recommendations.extend(build_recommendations(issues))
    severity_order = {"error": 0, "warning": 1, "info": 2}
    issues = sorted(issues, key=lambda item: (severity_order.get(str(item.get("severity")), 9), str(item.get("code"))))
    overall = block.get("overall") if isinstance(block.get("overall"), dict) else {}
    return {
        "run_dir": summary.get("run_dir"),
        "summary_path": summary.get("_summary_path"),
        "metric_block": block_name,
        "overall": overall,
        "threshold_sweep": block.get("threshold_sweep") or {},
        "issues": issues,
        "recommendations": list(dict.fromkeys(recommendations)),
    }


def main() -> None:
    args = parse_args()
    summary_path = resolve_summary_path(args.run)
    summary = read_json(summary_path)
    summary["_summary_path"] = str(summary_path)
    report = diagnose(summary, args)
    if args.output:
        output = resolve_repo_path(args.output) or Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        report["output"] = str(output)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
