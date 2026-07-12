#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""QPSALM 阈值推荐工具函数。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from qpsalm_seg.paths import resolve_repo_path


def read_json(path_ref: str | Path) -> dict[str, Any]:
    """读取 JSON object。"""
    path = resolve_repo_path(path_ref) or Path(path_ref)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON 顶层必须是 object: {path}")
    return payload


def resolve_report_path(path_ref: str | Path) -> Path:
    """支持 run 目录、eval 目录、run_summary.json 或 eval_report.json。"""
    path = resolve_repo_path(path_ref) or Path(path_ref)
    if path.is_dir():
        candidates = [
            path / "run_summary.json",
            path / "eval_report.json",
            path / "validation_latest.json",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def select_metric_block(payload: dict[str, Any], block_name: str = "auto") -> tuple[str, dict[str, Any]]:
    """从 run_summary 或 eval/validation report 中选择包含 threshold_sweep 的块。"""
    if "threshold_sweep" in payload and isinstance(payload.get("metrics"), dict):
        return "report", payload
    names = ["eval", "validation_best", "validation"] if block_name == "auto" else [block_name]
    for name in names:
        block = payload.get(name)
        if isinstance(block, dict) and isinstance(block.get("threshold_sweep"), dict):
            return name, block
    return "none", {}


def _metric_value(row: dict[str, Any], key: str) -> float:
    value = row.get(key)
    return float(value) if isinstance(value, (int, float)) else 0.0


def sorted_group_recommendations(
    per_group: dict[str, Any],
    group_prefixes: tuple[str, ...],
    metric: str,
    limit: int,
) -> list[dict[str, Any]]:
    """筛选并排序 per-group 推荐。"""
    rows: list[dict[str, Any]] = []
    for group, values in per_group.items():
        if not isinstance(group, str) or not isinstance(values, dict):
            continue
        if group_prefixes and not any(group.startswith(prefix) for prefix in group_prefixes):
            continue
        rows.append({"group": group, **values})
    rows.sort(key=lambda row: (_metric_value(row, metric), _metric_value(row, "n")), reverse=True)
    return rows[:limit] if limit > 0 else rows


def build_eval_command(
    run_dir: Path,
    threshold: float | None,
    device: str = "cuda",
    output_suffix: str = "best_threshold",
) -> str | None:
    """为 run 目录生成一条 best-threshold eval 命令。"""
    if threshold is None:
        return None
    config = run_dir / "resolved_config.yaml"
    checkpoint = run_dir / "checkpoint_best.pt"
    if not config.exists() or not checkpoint.exists():
        return None
    eval_dir = run_dir.parent / f"{run_dir.name}_eval_{output_suffix}"
    return (
        "PYTHONPATH=SEG_Multi-Source_Landslides "
        "/home/yukun80/miniconda3/envs/qwen3vl/bin/python -m qpsalm_seg.cli.eval "
        f"--config {config} "
        f"--checkpoint {checkpoint} "
        f"--device {device} "
        f"--output-dir {eval_dir} "
        f"--eval-threshold {threshold:.4f} "
        "--max-val-batches 0"
    )


def recommend_thresholds(
    report_ref: str | Path,
    block_name: str = "auto",
    group_prefixes: tuple[str, ...] = ("family_combo=",),
    limit: int = 32,
    eval_device: str = "cuda",
) -> dict[str, Any]:
    """读取报告并生成 overall/per-group 阈值推荐。"""
    report_path = resolve_report_path(report_ref)
    payload = read_json(report_path)
    selected_name, block = select_metric_block(payload, block_name=block_name)
    sweep = block.get("threshold_sweep") if isinstance(block.get("threshold_sweep"), dict) else {}
    best_by_dice = sweep.get("best_by_dice") if isinstance(sweep.get("best_by_dice"), dict) else {}
    best_by_iou = sweep.get("best_by_iou") if isinstance(sweep.get("best_by_iou"), dict) else {}
    per_group_dice = (
        sweep.get("best_by_dice_per_group")
        if isinstance(sweep.get("best_by_dice_per_group"), dict)
        else {}
    )
    per_group_iou = (
        sweep.get("best_by_iou_per_group")
        if isinstance(sweep.get("best_by_iou_per_group"), dict)
        else {}
    )
    current_threshold = block.get("threshold")
    overall = block.get("overall") if isinstance(block.get("overall"), dict) else {}
    if not overall and isinstance(block.get("metrics"), dict):
        overall = block["metrics"].get("overall") or {}
    best_threshold = best_by_dice.get("threshold") if isinstance(best_by_dice, dict) else None
    threshold_float = float(best_threshold) if isinstance(best_threshold, (int, float)) else None
    report_parent = report_path.parent
    if (report_parent / "run_summary.json").exists():
        run_dir = report_parent
    elif report_parent.name.endswith("_eval"):
        run_dir = report_parent.parent / report_parent.name[: -len("_eval")]
    else:
        run_dir = report_parent.parent
    return {
        "report_path": str(report_path),
        "metric_block": selected_name,
        "current": {
            "threshold": current_threshold,
            "overall": overall,
        },
        "best_by_dice": best_by_dice,
        "best_by_iou": best_by_iou,
        "selected_groups_by_dice": sorted_group_recommendations(
            per_group_dice,
            group_prefixes=group_prefixes,
            metric="dice",
            limit=limit,
        ),
        "selected_groups_by_iou": sorted_group_recommendations(
            per_group_iou,
            group_prefixes=group_prefixes,
            metric="iou",
            limit=limit,
        ),
        "eval_command_best_dice": build_eval_command(
            run_dir=run_dir,
            threshold=threshold_float,
            device=eval_device,
            output_suffix="best_dice_threshold",
        ),
    }
