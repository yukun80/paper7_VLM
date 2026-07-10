#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""QPSALM 评估 JSON 到 CSV 表格的导出工具。"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from qpsalm_seg.data import CANONICAL_MODALITIES, resolve_repo_path


METRIC_FIELDS = ["dice", "iou", "precision", "recall", "n"]
THRESHOLD_FIELDS = ["source", "group", "group_type", "group_value", "threshold", *METRIC_FIELDS]
GATE_WEIGHT_FIELDS = [f"weight_{name}" for name in CANONICAL_MODALITIES]
GATE_ACTIVE_FIELDS = [f"active_{name}" for name in CANONICAL_MODALITIES]
QUERY_GATE_MEAN_FIELDS = [f"query_mean_{name}" for name in CANONICAL_MODALITIES]
QUERY_GATE_SELECTED_FIELDS = [f"query_selected_{name}" for name in CANONICAL_MODALITIES]
QUERY_GATE_BEST_FIELDS = [f"query_best_{name}" for name in CANONICAL_MODALITIES]
PROPOSAL_FIELDS = [
    "sample_id",
    "dataset_name",
    "template_id",
    "task_family",
    "raw_combo",
    "canonical_combo",
    "sensor_combo",
    "normalization_combo",
    "condition_prompt",
    "gsd_token",
    "target_area_px_bin",
    "target_area_fraction_bin",
    "ground_area_m2",
    "ground_area_m2_bin",
    "best_query",
    "selected_query",
    "selected_matches_best",
    "best_query_dice",
    "selected_query_dice",
    "dice_gap_selected_minus_best",
    "selected_selection_logit",
    "best_selection_logit",
    "selection_logit_gap_selected_minus_best",
    "selected_proposal_fg_prob",
    "best_proposal_fg_prob",
    "selected_condition_score",
    "best_condition_score",
    "condition_score_gap_selected_minus_best",
    "selected_condition_cosine",
    "best_condition_cosine",
    "selected_condition_pair_logit",
    "best_condition_pair_logit",
    "selected_evidence_score",
    "best_evidence_score",
    "evidence_score_gap_selected_minus_best",
    "selected_evidence_cosine",
    "best_evidence_cosine",
    "selected_evidence_pair_logit",
    "best_evidence_pair_logit",
    "final_dice",
    "final_iou",
    "final_precision",
    "final_recall",
    "target_area",
    "final_mask_area",
    "selected_mask_area",
    "best_mask_area",
]


def read_json(path_ref: str | Path) -> dict[str, Any]:
    """读取 JSON 文件并确认顶层为 dict。"""
    path = resolve_repo_path(path_ref) or Path(path_ref)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON 顶层必须是 object: {path}")
    return payload


def group_type_and_value(group: str) -> tuple[str, str]:
    """把 canonical_combo=... 这类 group 拆成 type/value。"""
    if "=" not in group:
        return group, group
    group_type, value = group.split("=", 1)
    return group_type, value


def metric_rows_from_metrics(metrics: dict[str, Any], source: str) -> list[dict[str, Any]]:
    """从 report['metrics'] 导出指标行。"""
    rows: list[dict[str, Any]] = []
    for group, values in sorted(metrics.items()):
        if not isinstance(values, dict):
            continue
        group_type, group_value = group_type_and_value(str(group))
        row: dict[str, Any] = {
            "source": source,
            "group": group,
            "group_type": group_type,
            "group_value": group_value,
        }
        for field in METRIC_FIELDS:
            row[field] = values.get(field)
        rows.append(row)
    return rows


def gate_rows_from_summary(gates: dict[str, Any], source: str) -> list[dict[str, Any]]:
    """从 modality_gate_summary 导出模态门控行。"""
    rows: list[dict[str, Any]] = []
    for group, values in sorted(gates.items()):
        if not isinstance(values, dict):
            continue
        group_type, group_value = group_type_and_value(str(group))
        weights = values.get("mean_weights") or {}
        active = values.get("mean_active") or {}
        row: dict[str, Any] = {
            "source": source,
            "group": group,
            "group_type": group_type,
            "group_value": group_value,
            "n": values.get("n"),
        }
        for name in CANONICAL_MODALITIES:
            row[f"weight_{name}"] = weights.get(name)
        for name in CANONICAL_MODALITIES:
            row[f"active_{name}"] = active.get(name)
        rows.append(row)
    return rows


def query_gate_rows_from_summary(query_gates: dict[str, Any], source: str) -> list[dict[str, Any]]:
    """从 query_modality_summary 导出 query-level 模态注意力行。"""
    rows: list[dict[str, Any]] = []
    for group, values in sorted(query_gates.items()):
        if not isinstance(values, dict):
            continue
        group_type, group_value = group_type_and_value(str(group))
        mean_weights = values.get("mean_query_weights") or {}
        selected_weights = values.get("mean_selected_query_weights") or {}
        best_weights = values.get("mean_best_query_weights") or {}
        row: dict[str, Any] = {
            "source": source,
            "group": group,
            "group_type": group_type,
            "group_value": group_value,
            "n": values.get("n"),
            "mean_entropy": values.get("mean_entropy"),
            "mean_peak": values.get("mean_peak"),
        }
        for name in CANONICAL_MODALITIES:
            row[f"query_mean_{name}"] = mean_weights.get(name)
        for name in CANONICAL_MODALITIES:
            row[f"query_selected_{name}"] = selected_weights.get(name)
        for name in CANONICAL_MODALITIES:
            row[f"query_best_{name}"] = best_weights.get(name)
        rows.append(row)
    return rows


def proposal_rows_from_diagnostics(diagnostics: dict[str, Any], source: str) -> list[dict[str, Any]]:
    """从 proposal_diagnostics.records 导出样本级 proposal 选择行。"""
    rows: list[dict[str, Any]] = []
    records = diagnostics.get("records") if isinstance(diagnostics, dict) else None
    if not isinstance(records, list):
        return rows
    for record in records:
        if not isinstance(record, dict):
            continue
        row: dict[str, Any] = {"source": source}
        for field in PROPOSAL_FIELDS:
            row[field] = record.get(field)
        rows.append(row)
    return rows


def threshold_rows_from_sweep(sweep: dict[str, Any], source: str) -> list[dict[str, Any]]:
    """从 threshold_sweep 导出不同阈值下的 overall 与分组指标。"""
    rows: list[dict[str, Any]] = []
    if not isinstance(sweep, dict):
        return rows
    groups_by_threshold = sweep.get("groups_by_threshold")
    if isinstance(groups_by_threshold, dict):
        for threshold_text, groups in sorted(groups_by_threshold.items(), key=lambda item: float(item[0])):
            if not isinstance(groups, dict):
                continue
            for group, values in sorted(groups.items()):
                if not isinstance(values, dict):
                    continue
                group_type, group_value = group_type_and_value(str(group))
                row: dict[str, Any] = {
                    "source": source,
                    "group": group,
                    "group_type": group_type,
                    "group_value": group_value,
                    "threshold": float(threshold_text),
                }
                for field in METRIC_FIELDS:
                    row[field] = values.get(field)
                rows.append(row)
        return rows

    by_threshold = sweep.get("overall_by_threshold")
    if isinstance(by_threshold, dict):
        for threshold_text, values in sorted(by_threshold.items(), key=lambda item: float(item[0])):
            if not isinstance(values, dict):
                continue
            row = {
                "source": source,
                "group": "overall",
                "group_type": "overall",
                "group_value": "overall",
                "threshold": float(threshold_text),
            }
            for field in METRIC_FIELDS:
                row[field] = values.get(field)
            rows.append(row)
    return rows


def rows_from_eval_report(payload: dict[str, Any], source: str) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """读取 eval/validation report 顶层结构。"""
    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
    gates = payload.get("modality_gate_summary") if isinstance(payload.get("modality_gate_summary"), dict) else {}
    query_gates = payload.get("query_modality_summary") if isinstance(payload.get("query_modality_summary"), dict) else {}
    proposal = payload.get("proposal_diagnostics") if isinstance(payload.get("proposal_diagnostics"), dict) else {}
    sweep = payload.get("threshold_sweep") if isinstance(payload.get("threshold_sweep"), dict) else {}
    return (
        metric_rows_from_metrics(metrics, source),
        gate_rows_from_summary(gates, source),
        query_gate_rows_from_summary(query_gates, source),
        proposal_rows_from_diagnostics(proposal, source),
        threshold_rows_from_sweep(sweep, source),
    )


def rows_from_compact_summary(payload: dict[str, Any], source: str) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """读取 run_summary.json 结构，导出 validation/eval/manifest 子表。"""
    metric_rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    query_gate_rows: list[dict[str, Any]] = []
    proposal_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    for split_name in ["validation", "eval"]:
        block = payload.get(split_name)
        if not isinstance(block, dict):
            continue
        metrics: dict[str, Any] = {}
        if isinstance(block.get("overall"), dict):
            metrics["overall"] = block["overall"]
        metrics.update(block.get("canonical_combos") if isinstance(block.get("canonical_combos"), dict) else {})
        metrics.update(block.get("raw_combos") if isinstance(block.get("raw_combos"), dict) else {})
        metrics.update(block.get("sensor_combos") if isinstance(block.get("sensor_combos"), dict) else {})
        metrics.update(block.get("normalization_combos") if isinstance(block.get("normalization_combos"), dict) else {})
        metrics.update(block.get("gsd_tokens") if isinstance(block.get("gsd_tokens"), dict) else {})
        metrics.update(block.get("target_area_px_bins") if isinstance(block.get("target_area_px_bins"), dict) else {})
        metrics.update(block.get("target_area_fraction_bins") if isinstance(block.get("target_area_fraction_bins"), dict) else {})
        metrics.update(block.get("ground_area_m2_bins") if isinstance(block.get("ground_area_m2_bins"), dict) else {})
        metric_rows.extend(metric_rows_from_metrics(metrics, f"{source}:{split_name}"))
        gates = block.get("modality_gate_summary") if isinstance(block.get("modality_gate_summary"), dict) else {}
        gate_rows.extend(gate_rows_from_summary(gates, f"{source}:{split_name}"))
        query_gates = block.get("query_modality_summary") if isinstance(block.get("query_modality_summary"), dict) else {}
        query_gate_rows.extend(query_gate_rows_from_summary(query_gates, f"{source}:{split_name}"))
        proposal = block.get("proposal_diagnostics") if isinstance(block.get("proposal_diagnostics"), dict) else {}
        proposal_rows.extend(proposal_rows_from_diagnostics(proposal, f"{source}:{split_name}"))
        sweep = block.get("threshold_sweep") if isinstance(block.get("threshold_sweep"), dict) else {}
        threshold_rows.extend(threshold_rows_from_sweep(sweep, f"{source}:{split_name}"))

    visualizations = payload.get("visualizations")
    if isinstance(visualizations, dict):
        for name in ["train_manifest", "eval_manifest"]:
            block = visualizations.get(name)
            if not isinstance(block, dict):
                continue
            gates = block.get("modality_gate_summary") if isinstance(block.get("modality_gate_summary"), dict) else {}
            gate_rows.extend(gate_rows_from_summary(gates, f"{source}:{name}"))
            query_gates = block.get("query_modality_summary") if isinstance(block.get("query_modality_summary"), dict) else {}
            query_gate_rows.extend(query_gate_rows_from_summary(query_gates, f"{source}:{name}"))
    return metric_rows, gate_rows, query_gate_rows, proposal_rows, threshold_rows


def rows_from_payload(payload: dict[str, Any], source: str) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """自动判断 JSON 类型并导出指标/gate 行。"""
    if "metrics" in payload or "modality_gate_summary" in payload or "query_modality_summary" in payload or "proposal_diagnostics" in payload:
        return rows_from_eval_report(payload, source)
    if "validation" in payload or "eval" in payload:
        return rows_from_compact_summary(payload, source)
    return [], [], [], [], []


def source_name_for_path(path: Path) -> str:
    """生成 CSV source 名；run_summary/eval_report 用父目录消歧。"""
    if path.stem in {"run_summary", "eval_report", "validation_latest"}:
        return f"{path.parent.name}/{path.stem}"
    return path.stem


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    """写 CSV；没有行时仍写表头。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def export_analysis_tables(inputs: list[str | Path], output_dir_ref: str | Path) -> dict[str, Any]:
    """把多个 eval_report/run_summary JSON 导出为统一 metrics/gates CSV。"""
    output_dir = resolve_repo_path(output_dir_ref) or Path(output_dir_ref)
    all_metrics: list[dict[str, Any]] = []
    all_gates: list[dict[str, Any]] = []
    all_query_gates: list[dict[str, Any]] = []
    all_proposals: list[dict[str, Any]] = []
    all_thresholds: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for input_ref in inputs:
        path = resolve_repo_path(input_ref) or Path(input_ref)
        payload = read_json(path)
        source = source_name_for_path(path)
        metrics, gates, query_gates, proposals, thresholds = rows_from_payload(payload, source)
        all_metrics.extend(metrics)
        all_gates.extend(gates)
        all_query_gates.extend(query_gates)
        all_proposals.extend(proposals)
        all_thresholds.extend(thresholds)
        sources.append(
            {
                "path": str(path),
                "source": source,
                "metrics_rows": len(metrics),
                "gate_rows": len(gates),
                "query_gate_rows": len(query_gates),
                "proposal_rows": len(proposals),
                "threshold_rows": len(thresholds),
            }
        )

    metrics_path = output_dir / "metrics.csv"
    gates_path = output_dir / "modality_gates.csv"
    query_gates_path = output_dir / "query_modality_gates.csv"
    proposals_path = output_dir / "proposal_diagnostics.csv"
    thresholds_path = output_dir / "threshold_sweep.csv"
    write_csv(metrics_path, all_metrics, ["source", "group", "group_type", "group_value", *METRIC_FIELDS])
    write_csv(gates_path, all_gates, ["source", "group", "group_type", "group_value", "n", *GATE_WEIGHT_FIELDS, *GATE_ACTIVE_FIELDS])
    write_csv(
        query_gates_path,
        all_query_gates,
        [
            "source",
            "group",
            "group_type",
            "group_value",
            "n",
            "mean_entropy",
            "mean_peak",
            *QUERY_GATE_MEAN_FIELDS,
            *QUERY_GATE_SELECTED_FIELDS,
            *QUERY_GATE_BEST_FIELDS,
        ],
    )
    write_csv(proposals_path, all_proposals, ["source", *PROPOSAL_FIELDS])
    write_csv(thresholds_path, all_thresholds, THRESHOLD_FIELDS)
    manifest = {
        "output_dir": str(output_dir),
        "metrics_csv": str(metrics_path),
        "modality_gates_csv": str(gates_path),
        "query_modality_gates_csv": str(query_gates_path),
        "proposal_diagnostics_csv": str(proposals_path),
        "threshold_sweep_csv": str(thresholds_path),
        "metrics_rows": len(all_metrics),
        "gate_rows": len(all_gates),
        "query_gate_rows": len(all_query_gates),
        "proposal_rows": len(all_proposals),
        "threshold_rows": len(all_thresholds),
        "sources": sources,
    }
    (output_dir / "analysis_tables_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest
