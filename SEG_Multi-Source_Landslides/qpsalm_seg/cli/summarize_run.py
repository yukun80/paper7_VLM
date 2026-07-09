#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""汇总 QPSALM 训练/eval 运行产物。

脚本作用：读取 checkpoint、train_history、validation/eval report 和 PNG 可视化，
生成一个便于验收 Phase 1 闭环的 run_summary.json。
主要输入：outputs/qpsalm_* 运行目录，可选 eval 目录。
主要输出：run_summary.json 和终端 JSON。
是否改写原始数据：只写 summary JSON，不改 benchmark 或模型权重。
典型用法：python -m qpsalm_seg.cli.summarize_run --run-dir outputs/qpsalm_small_qwen_cached_core。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import yaml

from qpsalm_seg.analysis_tables import export_analysis_tables
from qpsalm_seg.data import resolve_repo_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize a QPSALM training/eval run directory.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--eval-dir", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--min-visualizations", type=int, default=4)
    parser.add_argument("--no-export-tables", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def read_yaml(path: Path) -> Any | None:
    if not path.exists():
        return None
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def count_pngs(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for item in path.rglob("*.png") if item.is_file())


def count_diagnostic_pngs(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(
        1
        for item in path.rglob("*.png")
        if item.is_file() and not any(part.startswith("mask_exports") for part in item.parts)
    )


def mask_export_counts(path: Path, export_dir_name: str = "mask_exports") -> dict[str, int]:
    names = ["final", "best_proposal", "gt", "bbox_prior"]
    counts = {name: 0 for name in names}
    if not path.exists():
        return counts
    for export_dir in path.rglob(export_dir_name):
        if not export_dir.is_dir():
            continue
        for name in names:
            counts[name] += sum(1 for item in (export_dir / name).glob("*.png") if item.is_file())
    return counts


def iter_visualization_manifest_records(path: Path) -> list[dict[str, Any]]:
    """读取可视化 manifest JSONL，兼容 train step 子目录和 eval 顶层目录。"""
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for manifest_path in sorted(path.rglob("visualization_manifest.jsonl")):
        with manifest_path.open("r", encoding="utf-8") as f:
            for line in f:
                text = line.strip()
                if not text:
                    continue
                try:
                    row = json.loads(text)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict):
                    row["_manifest_path"] = str(manifest_path)
                    records.append(row)
    return records


def manifest_record_summary(path: Path) -> dict[str, Any]:
    records = iter_visualization_manifest_records(path)
    sample_ids = {
        ((row.get("metadata") or {}).get("sample_id") or row.get("stem"))
        for row in records
    }
    combos: dict[str, int] = {}
    for row in records:
        combo = (row.get("metadata") or {}).get("canonical_combo") or "unknown"
        combos[combo] = combos.get(combo, 0) + 1
    return {
        "records": len(records),
        "unique_samples": len(sample_ids),
        "canonical_combos": dict(sorted(combos.items())),
        "modality_gate_summary": summarize_manifest_gates(records),
    }


def average_modality_dict(rows: list[dict[str, float] | None]) -> dict[str, float]:
    """对 manifest 中的 name->float 模态字段求均值。"""
    names = ["hr_optical", "s2", "s1", "dem", "insar"]
    usable = [row for row in rows if isinstance(row, dict)]
    if not usable:
        return {}
    return {
        name: sum(float(row.get(name, 0.0)) for row in usable) / len(usable)
        for name in names
    }


def summarize_manifest_gates(records: list[dict[str, Any]]) -> dict[str, Any]:
    """汇总可视化 manifest 中的样本级模态 gate。"""
    with_gate = [row for row in records if isinstance(row.get("modality_gate_weights"), dict)]
    if not with_gate:
        return {}
    groups: dict[str, list[dict[str, Any]]] = {"overall": with_gate}
    for row in with_gate:
        meta = row.get("metadata") or {}
        groups.setdefault(f"canonical_combo={meta.get('canonical_combo', 'unknown')}", []).append(row)
        groups.setdefault(f"sensor_combo={meta.get('sensor_combo', 'unknown')}", []).append(row)
        groups.setdefault(f"normalization_combo={meta.get('normalization_combo', 'unknown')}", []).append(row)
        groups.setdefault(f"condition={meta.get('condition_prompt', 'unknown')}", []).append(row)
    out: dict[str, Any] = {}
    for name, rows in sorted(groups.items()):
        out[name] = {
            "n": len(rows),
            "mean_weights": average_modality_dict([row.get("modality_gate_weights") for row in rows]),
            "mean_active": average_modality_dict([row.get("modality_active_mask") for row in rows]),
        }
    return out


def file_info(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else 0,
    }


def compact_metrics(report: dict[str, Any] | None) -> dict[str, Any]:
    if not report:
        return {}
    metrics = report.get("metrics") or {}
    out = {
        "loss": report.get("loss"),
        "overall": metrics.get("overall"),
        "threshold": report.get("threshold"),
        "threshold_sweep": report.get("threshold_sweep") or {},
        "loss_components": report.get("loss_components") or {},
        "modality_gate_summary": report.get("modality_gate_summary") or {},
    }
    proposal_diagnostics = report.get("proposal_diagnostics")
    if isinstance(proposal_diagnostics, dict):
        out["proposal_diagnostics"] = {
            "summary": proposal_diagnostics.get("summary") or {},
            "records": proposal_diagnostics.get("records") or [],
        }
    canonical = {
        key: value
        for key, value in metrics.items()
        if isinstance(key, str) and key.startswith("canonical_combo=")
    }
    raw = {
        key: value
        for key, value in metrics.items()
        if isinstance(key, str) and key.startswith("raw_combo=")
    }
    sensor = {
        key: value
        for key, value in metrics.items()
        if isinstance(key, str) and key.startswith("sensor_combo=")
    }
    normalization = {
        key: value
        for key, value in metrics.items()
        if isinstance(key, str) and key.startswith("normalization_combo=")
    }
    out["canonical_combos"] = dict(sorted(canonical.items()))
    out["raw_combos"] = dict(sorted(raw.items()))
    out["sensor_combos"] = dict(sorted(sensor.items()))
    out["normalization_combos"] = dict(sorted(normalization.items()))
    out["coverage"] = {
        "n": (metrics.get("overall") or {}).get("n") if isinstance(metrics.get("overall"), dict) else None,
        "num_canonical_combos": len(canonical),
        "canonical_combo_names": sorted(canonical.keys()),
        "num_raw_combos": len(raw),
        "raw_combo_names": sorted(raw.keys()),
        "num_sensor_combos": len(sensor),
        "sensor_combo_names": sorted(sensor.keys()),
        "num_normalization_combos": len(normalization),
        "normalization_combo_names": sorted(normalization.keys()),
    }
    out["num_visualizations_listed"] = len(report.get("visualizations") or [])
    if "checkpoint_step" in report:
        out["checkpoint_step"] = report["checkpoint_step"]
    if "step" in report:
        out["step"] = report["step"]
    return out


def train_history_summary(history: list[dict[str, Any]] | None) -> dict[str, Any]:
    if not history:
        return {"num_rows": 0}
    losses = [float(row["loss"]) for row in history if "loss" in row and math.isfinite(float(row["loss"]))]
    return {
        "num_rows": len(history),
        "first": history[0],
        "last": history[-1],
        "min_loss": min(losses) if losses else None,
        "max_loss": max(losses) if losses else None,
    }


def key_config(config: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(config, dict):
        return {}
    keys = [
        "controller",
        "qwen_model_path",
        "condition_embedding_cache",
        "target_size",
        "batch_size",
        "grad_accum_steps",
        "max_steps",
        "max_val_batches",
        "decoder_dim",
        "num_mask_tokens",
        "modality_dropout",
        "train_hflip_prob",
        "train_vflip_prob",
        "use_box_prior",
        "use_focal_loss",
        "boundary_loss_weight",
        "condition_ranking_loss_weight",
        "selection_ranking_loss_weight",
        "foreground_bce_pos_weight",
        "mask_bce_weight",
        "mask_dice_weight",
        "mask_tversky_weight",
        "tversky_alpha",
        "tversky_beta",
        "proposal_cls_weight",
        "condition_cls_weight",
        "proposal_mask_weight",
        "empty_mask_suppression_weight",
        "empty_proposal_suppression_weight",
        "proposal_positive_weight",
        "condition_positive_weight",
        "query_diversity_loss_weight",
        "proposal_mask_diversity_loss_weight",
        "gate_entropy_loss_weight",
        "proposal_soft_target_topk",
        "proposal_soft_target_temperature",
        "query_usage_balance_loss_weight",
        "selection_proposal_weight",
        "selection_condition_weight",
        "selection_temperature",
        "final_foreground_gate_weight",
        "final_mask_fusion",
        "final_topk",
        "final_noisy_or_epsilon",
        "canonical_combo_loss_weights",
        "eval_threshold",
        "threshold_sweep",
        "train_index",
        "val_index",
    ]
    return {key: config.get(key) for key in keys if key in config}


def summarize_run(
    run_dir_ref: str | Path,
    eval_dir_ref: str | Path | None = None,
    output_ref: str | Path | None = None,
    min_visualizations: int = 4,
    export_tables: bool = True,
) -> dict[str, Any]:
    """生成 run summary，并写入 JSON。"""
    run_dir = resolve_repo_path(run_dir_ref)
    if run_dir is None:
        raise FileNotFoundError(run_dir_ref)
    eval_dir = resolve_repo_path(eval_dir_ref) if eval_dir_ref else run_dir
    if eval_dir is None:
        raise FileNotFoundError(eval_dir_ref)
    output = resolve_repo_path(output_ref) if output_ref else run_dir / "run_summary.json"
    if output is None:
        raise FileNotFoundError(output_ref)

    train_history = read_json(run_dir / "train_history.json")
    validation = read_json(run_dir / "validation_latest.json")
    eval_report = read_json(eval_dir / "eval_report.json")
    manifest = read_json(run_dir / "run_manifest.json")
    eval_manifest = read_json(eval_dir / "eval_manifest.json")
    config = read_yaml(run_dir / "resolved_config.yaml")
    train_visual_dir = run_dir / "visualizations"
    eval_visual_dir = eval_dir / "eval_visualizations"
    train_png_count = count_diagnostic_pngs(train_visual_dir)
    eval_png_count = count_diagnostic_pngs(eval_visual_dir)
    train_mask_exports = mask_export_counts(train_visual_dir)
    eval_mask_exports = mask_export_counts(eval_visual_dir)
    train_restored_mask_exports = mask_export_counts(train_visual_dir, export_dir_name="mask_exports_original_size")
    eval_restored_mask_exports = mask_export_counts(eval_visual_dir, export_dir_name="mask_exports_original_size")
    train_manifest_records = manifest_record_summary(train_visual_dir)
    eval_manifest_records = manifest_record_summary(eval_visual_dir)
    checkpoint = file_info(run_dir / "checkpoint_last.pt")
    checkpoint_best = file_info(run_dir / "checkpoint_best.pt")
    validation_best = read_json(run_dir / "validation_best.json")

    has_overall = bool((validation or {}).get("metrics", {}).get("overall"))
    has_eval_overall = bool((eval_report or {}).get("metrics", {}).get("overall"))
    history_info = train_history_summary(train_history if isinstance(train_history, list) else None)
    last_loss = (history_info.get("last") or {}).get("loss") if isinstance(history_info.get("last"), dict) else None
    finite_last_loss = isinstance(last_loss, (int, float)) and math.isfinite(float(last_loss))
    min_mask_exports = int(min_visualizations) * 4
    acceptance = {
        "checkpoint_last_exists": bool(checkpoint["exists"]),
        "validation_latest_exists": validation is not None,
        "train_history_exists": isinstance(train_history, list) and len(train_history) > 0,
        "finite_last_loss": finite_last_loss,
        "validation_overall_metrics": has_overall,
        "eval_overall_metrics": has_eval_overall,
        "enough_train_visualizations": train_png_count >= int(min_visualizations),
        "enough_eval_visualizations": eval_png_count >= int(min_visualizations),
        "enough_train_mask_exports": sum(train_mask_exports.values()) >= min_mask_exports,
        "enough_eval_mask_exports": sum(eval_mask_exports.values()) >= min_mask_exports,
        "enough_train_restored_mask_exports": sum(train_restored_mask_exports.values()) >= min_mask_exports,
        "enough_eval_restored_mask_exports": sum(eval_restored_mask_exports.values()) >= min_mask_exports,
        "enough_train_visualization_manifest_records": train_manifest_records["records"] >= int(min_visualizations),
        "enough_eval_visualization_manifest_records": eval_manifest_records["records"] >= int(min_visualizations),
    }
    acceptance["phase1_smoke_ready"] = all(
        [
            acceptance["checkpoint_last_exists"],
            acceptance["validation_latest_exists"],
            acceptance["train_history_exists"],
            acceptance["finite_last_loss"],
            acceptance["validation_overall_metrics"],
            train_png_count + eval_png_count >= int(min_visualizations),
        ]
    )

    summary = {
        "run_dir": str(run_dir),
        "eval_dir": str(eval_dir),
        "manifest": {
            "path": str(run_dir / "run_manifest.json"),
            "exists": manifest is not None,
            "created_at_utc": manifest.get("created_at_utc") if isinstance(manifest, dict) else None,
            "mode": manifest.get("mode") if isinstance(manifest, dict) else None,
        },
        "eval_manifest": {
            "path": str(eval_dir / "eval_manifest.json"),
            "exists": eval_manifest is not None,
            "created_at_utc": eval_manifest.get("created_at_utc") if isinstance(eval_manifest, dict) else None,
            "checkpoint_step": eval_manifest.get("checkpoint_step") if isinstance(eval_manifest, dict) else None,
        },
        "checkpoint": checkpoint,
        "checkpoint_best": checkpoint_best,
        "artifacts": {
            "train_history": file_info(run_dir / "train_history.json"),
            "validation_latest": file_info(run_dir / "validation_latest.json"),
            "validation_best": file_info(run_dir / "validation_best.json"),
            "eval_report": file_info(eval_dir / "eval_report.json"),
            "run_manifest": file_info(run_dir / "run_manifest.json"),
            "eval_manifest": file_info(eval_dir / "eval_manifest.json"),
        },
        "config": key_config(config if isinstance(config, dict) else None),
        "train_history": history_info,
        "validation": compact_metrics(validation if isinstance(validation, dict) else None),
        "validation_best": compact_metrics(validation_best if isinstance(validation_best, dict) else None),
        "eval": compact_metrics(eval_report if isinstance(eval_report, dict) else None),
        "visualizations": {
            "train_png_count": train_png_count,
            "eval_png_count": eval_png_count,
            "train_mask_exports": train_mask_exports,
            "eval_mask_exports": eval_mask_exports,
            "train_restored_mask_exports": train_restored_mask_exports,
            "eval_restored_mask_exports": eval_restored_mask_exports,
            "train_mask_export_total": sum(train_mask_exports.values()),
            "eval_mask_export_total": sum(eval_mask_exports.values()),
            "train_restored_mask_export_total": sum(train_restored_mask_exports.values()),
            "eval_restored_mask_export_total": sum(eval_restored_mask_exports.values()),
            "train_manifest": train_manifest_records,
            "eval_manifest": eval_manifest_records,
        },
        "acceptance": acceptance,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if export_tables:
        summary["analysis_tables"] = export_analysis_tables([output], output.parent / "analysis_tables")
        output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    summary = summarize_run(
        run_dir_ref=args.run_dir,
        eval_dir_ref=args.eval_dir,
        output_ref=args.output,
        min_visualizations=args.min_visualizations,
        export_tables=not args.no_export_tables,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
