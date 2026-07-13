#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成汇报用分层精选推理图库。

用途：从已有 eval_report 中选择强、典型、失败及专题样本，并仅重新推理这些样本。
推荐命令：PYTHONPATH=SEG_Multi-Source_Landslides python -m
qpsalm_seg.cli.curate_gallery --config CONFIG --preset qwen_psalm_full
--checkpoint CHECKPOINT --eval-report EVAL_REPORT --vision-feature-cache CACHE
--split val --task-family referring_landslide_segmentation --device cuda
--output-dir OUTPUT --overwrite-output
主要输出：presentation PNG、mask、gallery_manifest.jsonl、gallery_summary.json 和 gallery_index.html。
写入行为：只写入 --output-dir；--overwrite-output 会删除该输出目录。
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil

from tqdm import tqdm

from qpsalm_seg.config import apply_config_overrides, load_config
from qpsalm_seg.gallery import select_gallery_records
from qpsalm_seg.inference import InferenceSession
from qpsalm_seg.paths import resolve_repo_path
from qpsalm_seg.presentation import append_jsonl, save_presentation_result, write_gallery_html
from qpsalm_seg.presets import PRESET_CHOICES, apply_preset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Curate a presentation gallery from an eval report.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--preset", choices=PRESET_CHOICES, default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--eval-report", required=True)
    parser.add_argument("--vision-feature-cache", required=True)
    parser.add_argument("--benchmark-dir", default=None)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument(
        "--task-family",
        action="append",
        default=None,
        help="仅导出指定 task_family；可重复传入以选择多个任务族。",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-items", type=int, default=120)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite-output", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_path = resolve_repo_path(args.eval_report) or Path(args.eval_report)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    records = ((report.get("proposal_diagnostics") or {}).get("records") or [])
    if not records:
        raise RuntimeError(f"eval report 不包含 proposal_diagnostics.records: {report_path}")
    task_families = {str(value).strip() for value in (args.task_family or []) if str(value).strip()}
    if task_families:
        records = [record for record in records if str(record.get("task_family")) in task_families]
        if not records:
            all_records = ((report.get("proposal_diagnostics") or {}).get("records") or [])
            available = sorted({str(record.get("task_family") or "unknown") for record in all_records})
            raise RuntimeError(
                f"eval report 中没有 task_family={sorted(task_families)}；可用值: {available}"
            )
    selected = select_gallery_records(records, max_items=args.max_items, seed=args.seed)
    config = apply_preset(load_config(args.config), args.preset)
    config = apply_config_overrides(config, {
        "benchmark_dir": args.benchmark_dir,
        "vision_feature_cache": args.vision_feature_cache,
        "modality_dropout": 0.0,
        "num_workers": 0,
    })
    out_dir = resolve_repo_path(args.output_dir) or Path(args.output_dir)
    if args.overwrite_output and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "gallery_manifest.jsonl"
    session = InferenceSession(
        config,
        split=args.split,
        checkpoint=args.checkpoint,
        device=args.device,
    )
    threshold = float(args.threshold if args.threshold is not None else report.get("threshold", config.eval_threshold))
    exports, errors = [], []
    for record in tqdm(selected, desc="qpsalm-ppt-gallery", unit="sample"):
        sample_id = str(record.get("sample_id"))
        try:
            result = session.predict(sample_id, threshold=threshold)
            exported = save_presentation_result(
                result,
                out_dir,
                category=str(record.get("gallery_category")),
                stratum=str(record.get("gallery_stratum")),
            )
            exported["tags"] = list(record.get("gallery_tags") or [record.get("gallery_category")])
        except Exception as exc:  # noqa: BLE001 - 保留完整失败清单，继续导出其他候选
            errors.append({"sample_id": sample_id, "error": f"{type(exc).__name__}: {exc}"})
            continue
        append_jsonl(manifest_path, exported)
        exports.append(exported)
    if not exports:
        raise RuntimeError(f"所有图库候选推理失败，首个错误: {errors[:1]}")
    write_gallery_html(exports, out_dir / "gallery_index.html")
    summary = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": session.checkpoint_step,
        "eval_report": str(report_path),
        "split": args.split,
        "task_families": sorted(task_families) if task_families else ["all"],
        "candidate_records": len(records),
        "threshold": threshold,
        "selected": len(selected),
        "exported": len(exports),
        "failed": len(errors),
        "categories": dict(Counter(str(item["category"]) for item in exports)),
        "tags": dict(Counter(str(tag) for item in exports for tag in (item.get("tags") or []))),
        "datasets": dict(Counter(str(item["dataset_name"]) for item in exports)),
        "errors": errors,
        "contains_oracle_output": False,
    }
    (out_dir / "gallery_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(out_dir), **summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
