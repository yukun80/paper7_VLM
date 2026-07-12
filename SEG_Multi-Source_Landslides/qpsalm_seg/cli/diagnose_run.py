#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""诊断 QPSALM 运行结果中的低精度原因。

用途：检查验证覆盖、threshold sweep、PMRD proposal、QMEF attention 和误差结构。
推荐运行命令：PYTHONPATH=SEG_Multi-Source_Landslides python -m
qpsalm_seg.cli.diagnose_run --run outputs/RUN/run_summary.json
--output outputs/RUN/diagnose_report.json
主要输入：一个 run 目录或 run_summary.json。
主要输出：诊断 JSON，可选写入 diagnose_report.json。
写入行为：只在指定 --output 时写报告，不修改 checkpoint 或 benchmark。
所属流程：训练/评估完成后的精度与结构诊断。
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from qpsalm_seg.paths import resolve_repo_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose a QPSALM run summary.")
    parser.add_argument("--run", required=True, help="Path to run directory or run_summary.json.")
    parser.add_argument("--output", default=None, help="Optional diagnose_report.json path.")
    parser.add_argument("--low-dice", type=float, default=0.45)
    parser.add_argument("--low-iou", type=float, default=0.30)
    parser.add_argument("--low-recall", type=float, default=0.40)
    parser.add_argument("--threshold-delta", type=float, default=0.03)
    parser.add_argument("--query-collapse-frac", type=float, default=0.80)
    parser.add_argument("--reliability-collapse-weight", type=float, default=0.85)
    parser.add_argument("--query-attention-collapse-peak", type=float, default=0.85)
    parser.add_argument("--empty-fp-area", type=float, default=128.0)
    parser.add_argument("--print-full-report", action="store_true")
    return parser.parse_args()


def default_diagnose_args(**overrides: Any) -> argparse.Namespace:
    """返回与 CLI 默认值一致的诊断参数，供训练后自动报告复用。"""
    values: dict[str, Any] = {
        "low_dice": 0.45,
        "low_iou": 0.30,
        "low_recall": 0.40,
        "threshold_delta": 0.03,
        "query_collapse_frac": 0.80,
        "reliability_collapse_weight": 0.85,
        "query_attention_collapse_peak": 0.85,
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
    if not acceptance.get("research_pipeline_ready"):
        issues.append(issue("error", "pipeline_not_ready", "research_pipeline_ready 不是 true。", acceptance))
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
    families = block.get("family_combos") if isinstance(block.get("family_combos"), dict) else {}
    if len(families) <= 1:
        issues.append(
            issue(
                "warning",
                "limited_modality_coverage",
                "当前指标只覆盖很少的 modality family combo，不能代表完整多源能力。",
                {"num_family_combos": len(families), "groups": sorted(families.keys())},
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
    positive = block.get("positive_only") if isinstance(block.get("positive_only"), dict) else overall
    dice = numeric(positive.get("dice"))
    iou = numeric(positive.get("iou"))
    precision = numeric(positive.get("precision"))
    recall = numeric(positive.get("recall"))
    if dice < args.low_dice:
        issues.append(issue("warning", "low_dice", "Positive-only Dice 偏低，需要继续改进模型或阈值。", {"block": block_name, "dice": dice}))
    if iou < args.low_iou:
        issues.append(issue("warning", "low_iou", "Positive-only IoU 偏低，需要检查召回、边界和假阳性。", {"block": block_name, "iou": iou}))
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
    per_group = sweep.get("best_by_dice_per_group") if isinstance(sweep.get("best_by_dice_per_group"), dict) else {}
    best = per_group.get("positive_only") if isinstance(per_group.get("positive_only"), dict) else {}
    if not best:
        best = sweep.get("best_by_dice") if isinstance(sweep.get("best_by_dice"), dict) else {}
    current_threshold = numeric(block.get("threshold"), 0.5)
    positive = block.get("positive_only") if isinstance(block.get("positive_only"), dict) else overall
    current_dice = numeric(positive.get("dice"))
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
    family_thresholds: dict[str, float] = {}
    for group, values in per_group.items():
        if not isinstance(group, str) or not group.startswith("family_combo=") or "/" in group or not isinstance(values, dict):
            continue
        threshold = values.get("threshold")
        if isinstance(threshold, (int, float)):
            family_thresholds[group] = float(threshold)
    if len(set(family_thresholds.values())) >= 2:
        issues.append(
            issue(
                "info",
                "combo_specific_thresholds",
                "不同 modality family combo 的最佳阈值不同，可能需要按模态组合校准。",
                family_thresholds,
            )
        )
        recommendations.append("查看 threshold_sweep.csv 的 family_combo 行，判断是否需要按证据族报告 calibrated metrics。")
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
    num_queries = max((int(row.get("num_queries", 1)) for row in records), default=1)
    if selected and num_queries > 1:
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
    mean_match = mean_numeric_field(records, "selected_is_matched")
    if mean_match is not None:
        if mean_match < 0.70:
            issues.append(
                issue(
                    "warning",
                    "verifier_selects_unmatched_query",
                    "统一 verifier 经常选择未匹配任何组件的 proposal。",
                    {"mean_selected_is_matched": mean_match},
                )
            )
    return issues


def check_semantic_verifier(block: dict[str, Any], config: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    """检查统一 semantic-evidence verifier 是否记录完整。"""
    issues: list[dict[str, Any]] = []
    verifier_weight = numeric(config.get("semantic_verifier_loss_weight"))
    if verifier_weight <= 0:
        return issues
    records = proposal_records(block)
    if not records:
        return issues
    has_relevance = any(isinstance(row.get("selected_relevance_logit"), (int, float)) for row in records)
    if not has_relevance:
        issues.append(
            issue(
                "warning",
                "missing_semantic_verifier_scores",
                "当前配置启用了统一 verifier，但 proposal diagnostics 缺少 relevance score。",
                {"semantic_verifier_loss_weight": verifier_weight},
            )
        )
        return issues
    relevance_ap = mean_numeric_field(records, "relevance_ap")
    rejection = mean_numeric_field(records, "unmatched_rejection")
    if relevance_ap is not None and relevance_ap < 0.7:
        issues.append(issue("warning", "weak_relevance_ap", "统一 verifier 的 relevance AP 偏低。", {"relevance_ap": relevance_ap}))
    if rejection is not None and rejection < 0.7:
        issues.append(issue("warning", "weak_unmatched_rejection", "未匹配 proposal 抑制不足。", {"unmatched_rejection": rejection}))
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


def check_modality_reliability(block: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    summaries = block.get("modality_reliability_summary") if isinstance(block.get("modality_reliability_summary"), dict) else {}
    for group, payload in summaries.items():
        if not isinstance(payload, dict):
            continue
        weights = payload.get("mean_weights") if isinstance(payload.get("mean_weights"), dict) else {}
        active = payload.get("mean_active") if isinstance(payload.get("mean_active"), dict) else {}
        names = sorted(set(weights) | set(active))
        active_count = sum(numeric(active.get(name)) for name in names)
        null_weight = numeric(payload.get("mean_null_evidence_weight"))
        if group == "overall" and null_weight >= 0.90:
            issues.append(issue(
                "warning", "null_evidence_collapse",
                "QMEF 几乎总是选择 null evidence，多源特征未被有效使用。",
                {"group": group, "mean_null_evidence_weight": null_weight},
            ))
        if active_count <= 1.2:
            continue
        top_name = None
        top_weight = -1.0
        for name in names:
            value = numeric(weights.get(name))
            if value > top_weight:
                top_name = name
                top_weight = value
        if top_weight >= float(args.reliability_collapse_weight):
            issues.append(
                issue(
                    "info",
                    "modality_reliability_collapse",
                    "多模态样本的可靠性先验过度偏向单一模态，需要确认质量元数据与证据是否一致。",
                    {"group": group, "top_modality": top_name, "top_weight": top_weight, "active_count": active_count},
                )
            )
    return issues


def check_query_modality_attention(block: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    """检查 query-level modality attention 是否存在，以及多源组合中是否塌缩。"""
    issues: list[dict[str, Any]] = []
    query_summary = block.get("query_modality_attention_summary") if isinstance(block.get("query_modality_attention_summary"), dict) else {}
    if not query_summary:
        return [issue("info", "missing_query_modality_attention_summary", "缺少 query-level modality attention 汇总，无法判断 proposal 是否按区域选择证据。")]
    for group, payload in query_summary.items():
        if not isinstance(group, str) or not group.startswith("family_combo=") or not isinstance(payload, dict):
            continue
        combo = group.split("=", 1)[1]
        if "+" not in combo:
            continue
        peak = numeric(payload.get("mean_peak"))
        weights = payload.get("mean_query_weights") if isinstance(payload.get("mean_query_weights"), dict) else {}
        top_name = None
        top_weight = -1.0
        for name in sorted(weights):
            value = numeric(weights.get(name))
            if value > top_weight:
                top_name = name
                top_weight = value
        if peak >= float(args.query_attention_collapse_peak):
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
        recs.append("在 YAML 中设置 max_val_batches: 0，并使用固定完整 split 重新生成多源验证报告。")
    if "low_recall" in codes or "precision_recall_imbalance" in codes:
        recs.append("优先检查 positive-only threshold sweep；若低阈值仍无效，再调整 final BCE/Dice 比例与 proposal coverage。")
    if "query_selection_collapse" in codes:
        recs.append("检查 proposal set matching 的组件覆盖与 relevance target，确认 mask tokens 是否学到不同连通区域。")
    if "empty_mask_false_positives" in codes:
        recs.append("检查 no-target 样本的 verifier rejection 与 final BCE，再用 threshold sweep 定位空 mask 误报来源。")
    if "verifier_selects_unmatched_query" in codes or "weak_relevance_ap" in codes or "weak_unmatched_rejection" in codes:
        recs.append("检查 ProposalAssignment 的 relevance target、AP 和 unmatched rejection，校准统一 semantic verifier。")
    if "missing_semantic_verifier_scores" in codes:
        recs.append("重新运行 eval，确认 proposal diagnostics 已包含统一 relevance logit。")
    if "modality_reliability_collapse" in codes:
        recs.append("核对 SANE quality/GSD/sensor 元数据，并做单模态移除实验，判断可靠性集中是否符合数据质量。")
    if "null_evidence_collapse" in codes:
        recs.append("检查 valid coverage、quality prior 与 renderer/cache 版本，确认 QMEF 未把有效模态系统性判为无证据。")
    if "missing_query_modality_attention_summary" in codes:
        recs.append("使用 raw_sane_qmef 或更高 preset 重新 eval，导出 query-level modality attention 后再比较多源组合。")
    if "query_modality_attention_collapse" in codes:
        recs.append("查看 query_modality_attention.csv，并结合 view-removal 对照判断局部证据集中是否合理。")
    if "missing_target_area_strata" in codes or "limited_target_area_strata" in codes:
        recs.append("重新用新版 eval/summary 生成 target_area_px_bin 与 target_area_fraction_bin 分组，单独检查 tiny/small 滑坡斑块的 Dice/Recall。")
    if "ground_area_unknown" in codes:
        recs.append("优先完善 benchmark 中的连续 gsd_m/resize_transform，再比较 ground_area_m2_bin 下的性能。")
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
    issues.extend(check_semantic_verifier(block, config, args))
    issues.extend(check_empty_false_positives(block, args))
    issues.extend(check_modality_reliability(block, args))
    issues.extend(check_query_modality_attention(block, args))
    recommendations.extend(build_recommendations(issues))
    severity_order = {"error": 0, "warning": 1, "info": 2}
    issues = sorted(issues, key=lambda item: (severity_order.get(str(item.get("severity")), 9), str(item.get("code"))))
    overall = block.get("overall") if isinstance(block.get("overall"), dict) else {}
    positive_only = block.get("positive_only") if isinstance(block.get("positive_only"), dict) else {}
    return {
        "run_dir": summary.get("run_dir"),
        "summary_path": summary.get("_summary_path"),
        "metric_block": block_name,
        "overall": overall,
        "positive_only": positive_only,
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
    if args.print_full_report:
        payload = report
    else:
        payload = {
            "output": report.get("output"),
            "metric_block": report.get("metric_block"),
            "overall": report.get("overall"),
            "positive_only": report.get("positive_only"),
            "issue_count": len(report.get("issues") or []),
            "issues": [
                {"severity": item.get("severity"), "code": item.get("code"), "message": item.get("message")}
                for item in report.get("issues") or []
            ],
            "recommendations": report.get("recommendations"),
        }
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
