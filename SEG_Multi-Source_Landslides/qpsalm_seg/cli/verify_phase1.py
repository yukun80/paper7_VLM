#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验证 QPSALM Phase 1 运行产物是否满足闭环要求。

脚本作用：读取 phase1_summary.json 与各分支 run_summary.json，检查 checkpoint、
validation/eval metrics、PNG/mask exports、comparison 和 CSV 分析表是否存在。
主要输入：qpsalm-run-phase1 输出目录。
主要输出：终端 JSON 验收报告，失败时以非 0 状态退出。
是否改写原始数据：不会。
典型用法：python -m qpsalm_seg.cli.verify_phase1 --run-root outputs/qpsalm_phase1/qwen_cached_balanced --require-mode both。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


MASK_EXPORT_NAMES = ("final", "best_proposal", "gt", "bbox_prior")
MODALITY_NAMES = ("hr_optical", "s2", "s1", "dem", "insar")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify a QPSALM Phase 1 run directory.")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--require-mode", choices=["any", "baseline", "box-prior", "both"], default="any")
    parser.add_argument("--require-embedding-backend", choices=["any", "qwen", "hash-smoke"], default="any")
    parser.add_argument("--require-device", default="any")
    parser.add_argument("--min-visualizations", type=int, default=1)
    parser.add_argument("--require-analysis-tables", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def resolve_path(path_ref: str | Path) -> Path:
    path = Path(path_ref)
    if path.is_absolute():
        return path
    return Path.cwd() / path


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def require_file(path: Path, label: str, errors: list[str]) -> dict[str, Any]:
    exists = path.exists()
    if not exists:
        errors.append(f"missing {label}: {path}")
    return {"path": str(path), "exists": exists, "size_bytes": path.stat().st_size if exists else 0}


def read_jsonl_records(path: Path, label: str, errors: list[str]) -> list[dict[str, Any]]:
    if not path.exists():
        errors.append(f"missing {label}: {path}")
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                errors.append(f"{label}:{line_no} invalid JSONL: {exc}")
                continue
            if not isinstance(row, dict):
                errors.append(f"{label}:{line_no} is not a JSON object")
                continue
            records.append(row)
    return records


def resolve_report_path(path_ref: str | Path | None, fallback: Path) -> Path:
    if path_ref:
        return resolve_path(path_ref)
    return fallback


def dict_block(payload: dict[str, Any] | None, key: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    block = payload.get(key)
    return block if isinstance(block, dict) else {}


def matches_requirement(actual: Any, required: str | None) -> bool:
    if required is None or required == "any":
        return True
    return str(actual) == str(required)


def _path_from_record(path_ref: Any) -> Path | None:
    if not isinstance(path_ref, str) or not path_ref:
        return None
    return resolve_path(path_ref)


def _require_record_path(path_ref: Any, label: str, errors: list[str]) -> dict[str, Any]:
    path = _path_from_record(path_ref)
    if path is None:
        errors.append(f"missing manifest path field: {label}")
        return {"path": None, "exists": False, "size_bytes": 0}
    return require_file(path, label, errors)


def _has_nonempty_string(payload: dict[str, Any], key: str) -> bool:
    return isinstance(payload.get(key), str) and bool(str(payload.get(key)).strip())


def check_visualization_manifest(
    manifest_path: Path,
    label: str,
    min_visualizations: int,
    errors: list[str],
) -> dict[str, Any]:
    records = read_jsonl_records(manifest_path, label, errors)
    report = {
        "path": str(manifest_path),
        "records": len(records),
        "checked_records": 0,
        "complete_records": 0,
    }
    for idx, record in enumerate(records[: int(min_visualizations)]):
        before = len(errors)
        _require_record_path(record.get("diagnostic_path"), f"{label}[{idx}].diagnostic_path", errors)

        mask_paths = record.get("mask_paths")
        if not isinstance(mask_paths, dict):
            errors.append(f"{label}[{idx}].mask_paths missing or not object")
        else:
            for name in MASK_EXPORT_NAMES:
                _require_record_path(mask_paths.get(name), f"{label}[{idx}].mask_paths.{name}", errors)

        restored_paths = record.get("restored_mask_paths")
        if not isinstance(restored_paths, dict):
            errors.append(f"{label}[{idx}].restored_mask_paths missing or not object")
        else:
            for name in MASK_EXPORT_NAMES:
                _require_record_path(restored_paths.get(name), f"{label}[{idx}].restored_mask_paths.{name}", errors)

        metadata = record.get("metadata")
        if not isinstance(metadata, dict):
            errors.append(f"{label}[{idx}].metadata missing or not object")
        else:
            for key in ("sample_id", "canonical_combo", "raw_combo", "condition_prompt"):
                if not _has_nonempty_string(metadata, key):
                    errors.append(f"{label}[{idx}].metadata.{key} missing")
            transform = metadata.get("resize_transform")
            if not isinstance(transform, dict):
                errors.append(f"{label}[{idx}].metadata.resize_transform missing or not object")
            else:
                for key in ("source_hw", "target_hw", "resized_hw"):
                    value = transform.get(key)
                    if not isinstance(value, list) or len(value) != 2:
                        errors.append(f"{label}[{idx}].metadata.resize_transform.{key} invalid")

        gate = record.get("modality_gate_weights")
        active = record.get("modality_active_mask")
        for field_name, payload in (("modality_gate_weights", gate), ("modality_active_mask", active)):
            if not isinstance(payload, dict):
                errors.append(f"{label}[{idx}].{field_name} missing or not object")
                continue
            missing = [name for name in MODALITY_NAMES if name not in payload]
            if missing:
                errors.append(f"{label}[{idx}].{field_name} missing modalities: {missing}")

        report["checked_records"] += 1
        if len(errors) == before:
            report["complete_records"] += 1
    return report


def check_visualization_tree(
    root: Path,
    label: str,
    min_visualizations: int,
    errors: list[str],
    required: bool,
) -> dict[str, Any]:
    manifest_paths = sorted(root.rglob("visualization_manifest.jsonl")) if root.exists() else []
    if required and not manifest_paths:
        errors.append(f"missing {label}.visualization_manifest under {root}")
    reports = [
        check_visualization_manifest(path, f"{label}.manifest[{idx}]", min_visualizations, errors)
        for idx, path in enumerate(manifest_paths)
    ]
    total_records = sum(int(report.get("records") or 0) for report in reports)
    total_complete = sum(int(report.get("complete_records") or 0) for report in reports)
    if required and total_records < int(min_visualizations):
        errors.append(f"{label}.visualization_manifest total_records {total_records} < {min_visualizations}")
    if required and total_complete < int(min_visualizations):
        errors.append(f"{label}.visualization_manifest complete_records {total_complete} < {min_visualizations}")
    return {
        "root": str(root),
        "manifest_count": len(manifest_paths),
        "total_records": total_records,
        "total_complete_records": total_complete,
        "manifests": reports,
    }


def check_analysis_tables(block: dict[str, Any], label: str, errors: list[str], required: bool) -> dict[str, Any]:
    if not block:
        if required:
            errors.append(f"missing {label}.analysis_tables")
        return {}
    metrics_path = resolve_path(block.get("metrics_csv", ""))
    gates_path = resolve_path(block.get("modality_gates_csv", ""))
    proposal_path_ref = block.get("proposal_diagnostics_csv")
    threshold_path_ref = block.get("threshold_sweep_csv")
    out = {
        "metrics_csv": require_file(metrics_path, f"{label}.metrics_csv", errors),
        "modality_gates_csv": require_file(gates_path, f"{label}.modality_gates_csv", errors),
        "metrics_rows": block.get("metrics_rows"),
        "gate_rows": block.get("gate_rows"),
        "proposal_rows": block.get("proposal_rows"),
        "threshold_rows": block.get("threshold_rows"),
    }
    if proposal_path_ref:
        out["proposal_diagnostics_csv"] = require_file(
            resolve_path(proposal_path_ref),
            f"{label}.proposal_diagnostics_csv",
            errors,
        )
    if threshold_path_ref:
        out["threshold_sweep_csv"] = require_file(
            resolve_path(threshold_path_ref),
            f"{label}.threshold_sweep_csv",
            errors,
        )
    if required and int(block.get("metrics_rows") or 0) <= 0:
        errors.append(f"{label}.analysis_tables metrics_rows <= 0")
    if required and int(block.get("gate_rows") or 0) <= 0:
        errors.append(f"{label}.analysis_tables gate_rows <= 0")
    if required and proposal_path_ref and int(block.get("proposal_rows") or 0) <= 0:
        errors.append(f"{label}.analysis_tables proposal_rows <= 0")
    if required and threshold_path_ref and int(block.get("threshold_rows") or 0) <= 0:
        errors.append(f"{label}.analysis_tables threshold_rows <= 0")
    return out


def check_run_branch(
    run: dict[str, Any],
    min_visualizations: int,
    require_tables: bool,
    require_embedding_backend: str,
    require_device: str,
    errors: list[str],
) -> dict[str, Any]:
    mode = str(run.get("mode", "unknown"))
    run_dir = resolve_path(run.get("run_dir") or "")
    eval_dir = resolve_path(run.get("eval_dir") or "")
    summary_path = resolve_report_path((run.get("summary") or {}).get("path"), run_dir / "run_summary.json")
    manifest_path = resolve_report_path((run.get("manifest") or {}).get("path"), run_dir / "run_manifest.json")
    summary = read_json(summary_path)
    manifest = read_json(manifest_path)
    branch_errors_start = len(errors)
    files = {
        "run_summary": require_file(summary_path, f"{mode}.run_summary", errors),
        "run_manifest": require_file(manifest_path, f"{mode}.run_manifest", errors),
        "checkpoint_last": require_file(run_dir / "checkpoint_last.pt", f"{mode}.checkpoint_last", errors),
        "checkpoint_best": require_file(run_dir / "checkpoint_best.pt", f"{mode}.checkpoint_best", errors),
        "train_history": require_file(run_dir / "train_history.json", f"{mode}.train_history", errors),
        "validation_latest": require_file(run_dir / "validation_latest.json", f"{mode}.validation_latest", errors),
        "validation_best": require_file(run_dir / "validation_best.json", f"{mode}.validation_best", errors),
        "threshold_recommendations": require_file(
            run_dir / "threshold_recommendations.json",
            f"{mode}.threshold_recommendations",
            errors,
        ),
        "eval_report": require_file(eval_dir / "eval_report.json", f"{mode}.eval_report", errors),
        "eval_manifest": require_file(eval_dir / "eval_manifest.json", f"{mode}.eval_manifest", errors),
        "eval_visualization_manifest": require_file(
            eval_dir / "eval_visualizations" / "visualization_manifest.jsonl",
            f"{mode}.eval_visualization_manifest",
            errors,
        ),
    }
    if summary is None:
        errors.append(f"{mode}.run_summary is not readable JSON object: {summary_path}")
        return {"mode": mode, "ok": False, "files": files}
    if manifest is None:
        errors.append(f"{mode}.run_manifest is not readable JSON object: {manifest_path}")
        manifest = {}

    manifest_args = dict_block(manifest, "args")
    branch_backend = manifest_args.get("embedding_backend")
    branch_device = manifest_args.get("device")
    if not matches_requirement(branch_backend, require_embedding_backend):
        errors.append(
            f"{mode}.args.embedding_backend {branch_backend!r} != required {require_embedding_backend!r}"
        )
    if not matches_requirement(branch_device, require_device):
        errors.append(f"{mode}.args.device {branch_device!r} != required {require_device!r}")

    acceptance = summary.get("acceptance") if isinstance(summary.get("acceptance"), dict) else {}
    if not acceptance.get("phase1_smoke_ready"):
        errors.append(f"{mode}.acceptance.phase1_smoke_ready is not true")
    validation = summary.get("validation") if isinstance(summary.get("validation"), dict) else {}
    eval_block = summary.get("eval") if isinstance(summary.get("eval"), dict) else {}
    if not validation.get("overall"):
        errors.append(f"{mode}.validation.overall is missing")
    if not eval_block.get("overall"):
        errors.append(f"{mode}.eval.overall is missing")
    threshold_recommendations = read_json(run_dir / "threshold_recommendations.json")
    if threshold_recommendations is None:
        errors.append(f"{mode}.threshold_recommendations is not readable JSON object")
        threshold_recommendations = {}
    elif not isinstance(threshold_recommendations.get("best_by_dice"), dict):
        errors.append(f"{mode}.threshold_recommendations.best_by_dice is missing")
    visualizations = summary.get("visualizations") if isinstance(summary.get("visualizations"), dict) else {}
    eval_png_count = int(visualizations.get("eval_png_count") or 0)
    if eval_png_count < int(min_visualizations):
        errors.append(f"{mode}.eval_png_count {eval_png_count} < {min_visualizations}")
    if int(visualizations.get("eval_mask_export_total") or 0) < int(min_visualizations) * 4:
        errors.append(f"{mode}.eval_mask_export_total is too small")
    if int(visualizations.get("eval_restored_mask_export_total") or 0) < int(min_visualizations) * 4:
        errors.append(f"{mode}.eval_restored_mask_export_total is too small")
    train_manifest_report = check_visualization_tree(
        run_dir / "visualizations",
        f"{mode}.train_visualizations",
        min_visualizations=min_visualizations,
        errors=errors,
        required=False,
    )
    eval_manifest_report = check_visualization_tree(
        eval_dir / "eval_visualizations",
        f"{mode}.eval_visualizations",
        min_visualizations=min_visualizations,
        errors=errors,
        required=True,
    )

    table_report = check_analysis_tables(
        summary.get("analysis_tables") if isinstance(summary.get("analysis_tables"), dict) else {},
        f"{mode}.run_summary",
        errors,
        require_tables,
    )
    calibrated = run.get("calibrated_eval") if isinstance(run.get("calibrated_eval"), dict) else {}
    calibrated_report: dict[str, Any] = {}
    if calibrated:
        calibrated_dir = resolve_path(calibrated.get("eval_dir") or "")
        calibrated_report = {
            "eval_report": require_file(
                calibrated_dir / "eval_report.json",
                f"{mode}.calibrated_eval.eval_report",
                errors,
            ),
            "eval_manifest": require_file(
                calibrated_dir / "eval_manifest.json",
                f"{mode}.calibrated_eval.eval_manifest",
                errors,
            ),
            "eval_visualization_manifest": require_file(
                calibrated_dir / "eval_visualizations" / "visualization_manifest.jsonl",
                f"{mode}.calibrated_eval.eval_visualization_manifest",
                errors,
            ),
            "threshold": calibrated.get("threshold"),
            "overall": calibrated.get("overall"),
        }
    return {
        "mode": mode,
        "ok": len(errors) == branch_errors_start,
        "files": files,
        "phase1_smoke_ready": acceptance.get("phase1_smoke_ready"),
        "validation_overall": validation.get("overall"),
        "eval_overall": eval_block.get("overall"),
        "eval_png_count": eval_png_count,
        "runtime": {
            "embedding_backend": branch_backend,
            "device": branch_device,
        },
        "visualization_manifests": {
            "train": train_manifest_report,
            "eval": eval_manifest_report,
        },
        "analysis_tables": table_report,
        "threshold_recommendations": {
            "best_by_dice": threshold_recommendations.get("best_by_dice"),
            "eval_command_best_dice": threshold_recommendations.get("eval_command_best_dice"),
        },
        "calibrated_eval": calibrated_report,
    }


def expected_modes(require_mode: str) -> set[str]:
    if require_mode == "both":
        return {"baseline", "box-prior"}
    if require_mode in {"baseline", "box-prior"}:
        return {require_mode}
    return set()


def verify_phase1(
    run_root_ref: str | Path,
    require_mode: str,
    min_visualizations: int,
    require_tables: bool,
    require_embedding_backend: str = "any",
    require_device: str = "any",
) -> dict[str, Any]:
    run_root = resolve_path(run_root_ref)
    errors: list[str] = []
    warnings: list[str] = []
    phase_summary_path = run_root / "phase1_summary.json"
    phase_manifest_path = run_root / "phase1_manifest.json"
    phase_summary = read_json(phase_summary_path)
    phase_manifest = read_json(phase_manifest_path)
    files = {
        "phase1_summary": require_file(phase_summary_path, "phase1_summary", errors),
        "phase1_manifest": require_file(phase_manifest_path, "phase1_manifest", errors),
    }
    if phase_summary is None:
        errors.append(f"phase1_summary is not readable JSON object: {phase_summary_path}")
        return {"run_root": str(run_root), "ok": False, "errors": errors, "warnings": warnings, "files": files}
    if phase_manifest is None:
        errors.append(f"phase1_manifest is not readable JSON object: {phase_manifest_path}")
        phase_manifest = {}

    phase_args = dict_block(phase_manifest, "args")
    phase_backend = phase_args.get("embedding_backend")
    phase_device = phase_args.get("device")
    if not matches_requirement(phase_backend, require_embedding_backend):
        errors.append(f"phase1.args.embedding_backend {phase_backend!r} != required {require_embedding_backend!r}")
    if not matches_requirement(phase_device, require_device):
        errors.append(f"phase1.args.device {phase_device!r} != required {require_device!r}")

    runs = phase_summary.get("runs") if isinstance(phase_summary.get("runs"), list) else []
    run_modes = {str(run.get("mode")) for run in runs if isinstance(run, dict)}
    required = expected_modes(require_mode)
    missing_modes = sorted(required - run_modes)
    if missing_modes:
        errors.append(f"missing required modes: {missing_modes}")
    branch_reports = [
        check_run_branch(
            run,
            min_visualizations=min_visualizations,
            require_tables=require_tables,
            require_embedding_backend=require_embedding_backend,
            require_device=require_device,
            errors=errors,
        )
        for run in runs
        if isinstance(run, dict) and (not required or str(run.get("mode")) in required)
    ]
    if require_mode == "both":
        comparison = phase_summary.get("comparison") if isinstance(phase_summary.get("comparison"), dict) else {}
        comparison_path = resolve_path(comparison.get("path") or (run_root / "comparison_baseline_vs_box-prior.json"))
        files["comparison"] = require_file(comparison_path, "comparison_baseline_vs_box-prior", errors)
        acceptance = comparison.get("acceptance") if isinstance(comparison.get("acceptance"), dict) else {}
        if not acceptance.get("baseline_phase1_smoke_ready"):
            errors.append("comparison baseline_phase1_smoke_ready is not true")
        if not acceptance.get("candidate_phase1_smoke_ready"):
            errors.append("comparison candidate_phase1_smoke_ready is not true")
        primary = comparison.get("primary_deltas") if isinstance(comparison.get("primary_deltas"), dict) else {}
        if "dice" not in primary or "iou" not in primary:
            errors.append("comparison primary_deltas missing dice/iou")

    cache = phase_summary.get("condition_cache") if isinstance(phase_summary.get("condition_cache"), dict) else {}
    cache_backend = None
    if cache and not cache.get("skipped"):
        files["condition_cache"] = require_file(resolve_path(cache.get("output", "")), "condition_cache", errors)
        coverage = cache.get("coverage") if isinstance(cache.get("coverage"), dict) else {}
        if coverage and not coverage.get("ok"):
            errors.append("condition_cache.coverage.ok is not true")
        if coverage:
            missing = (coverage.get("missing") or {}).get("num_texts")
            if missing not in {0, 0.0}:
                errors.append(f"condition_cache.coverage missing texts: {missing}")
        else:
            warnings.append("condition_cache.coverage missing")
        if not cache.get("text_types") and not cache.get("reused"):
            warnings.append("condition_cache.text_types missing")
        cache_backend = cache.get("backend") or ((coverage.get("cache") or {}).get("backend") if coverage else None)
        if not matches_requirement(cache_backend, require_embedding_backend):
            errors.append(
                f"condition_cache.backend {cache_backend!r} != required {require_embedding_backend!r}"
            )
    phase_tables = check_analysis_tables(
        phase_summary.get("analysis_tables") if isinstance(phase_summary.get("analysis_tables"), dict) else {},
        "phase1_summary",
        errors,
        require_tables and len(runs) > 0,
    )
    return {
        "run_root": str(run_root),
        "ok": not errors,
        "require_mode": require_mode,
        "strict_requirements": {
            "embedding_backend": require_embedding_backend,
            "device": require_device,
            "phase_embedding_backend": phase_backend,
            "phase_device": phase_device,
            "condition_cache_backend": cache_backend,
        },
        "modes": sorted(run_modes),
        "errors": errors,
        "warnings": warnings,
        "files": files,
        "branches": branch_reports,
        "analysis_tables": phase_tables,
    }


def main() -> None:
    args = parse_args()
    report = verify_phase1(
        run_root_ref=args.run_root,
        require_mode=args.require_mode,
        min_visualizations=args.min_visualizations,
        require_tables=bool(args.require_analysis_tables),
        require_embedding_backend=args.require_embedding_backend,
        require_device=args.require_device,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report.get("ok"):
        sys.exit(1)


if __name__ == "__main__":
    main()
