#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""步骤 3-1：只读审计遥感描述与区域对齐数据源。

用途：核对 RSICap、RSIEval、MMRS Caption、DIOR-RSVG 的本地统计、结构和许可状态。
推荐运行命令：python scripts/3-description/3-1_scan_description_sources.py --mode small --output-dir benchmark/qpsalm_description_v2_small --overwrite
主要输入：datasets/RSGPT/dataset（兼容 datasets/dataset 与 external/RSGPT/dataset）及 datasets/MMRS-1M。
主要输出：reports/source_audit.json。
写入行为：只写审计报告，不修改源数据；--dry-run 不写文件。
所属流程：Description Benchmark M0。
"""

from __future__ import annotations

import argparse
from collections import Counter
from typing import Any

from description_common import (
    BUILDER_VERSION, MMRS_ROOT, RSGPT_ROOT, description_dir_for_mode, ensure_writable,
    iter_turn_pairs, read_json, scene_prefix, to_project_ref, write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="审计 RSGPT/MMRS 描述数据源")
    parser.add_argument("--mode", choices=["small", "full"], default="small")
    parser.add_argument("--output-dir")
    parser.add_argument("--max-samples", type=int, default=0, help="限制 conversation 深入探测量，不改源总数")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def audit_rsicap() -> dict[str, Any]:
    root = RSGPT_ROOT / "RSICap"
    path = root / "captions.json"
    rows = read_json(path)["annotations"]
    referenced = {str(row["filename"]) for row in rows}
    images = {item.name for item in (root / "images").glob("*") if item.is_file()}
    groups = [scene_prefix(name) for name in referenced]
    return {
        "annotation_path": to_project_ref(path),
        "records": len(rows),
        "unique_referenced_images": len(referenced),
        "image_files": len(images),
        "unreferenced_image_files": len(images - referenced),
        "missing_referenced_images": sorted(referenced - images),
        "instruction_counts": dict(sorted(Counter(str(row["text_input"]) for row in rows).items())),
        "scene_prefix_coverage": sum(value is not None for value in groups) / max(len(groups), 1),
        "unique_scene_prefixes": len({value for value in groups if value}),
        "scene_prefix_policy": "conservative_leakage_group_not_verified_scene_identity",
        "license_status": "academic_only",
        "license_source": "external/RSGPT/README.md",
    }


def audit_rsieval() -> dict[str, Any]:
    root = RSGPT_ROOT / "RSIEval"
    path = root / "annotations.json"
    rows = read_json(path)["annotations"]
    qa_count = sum(len(row.get("qa_pairs", [])) for row in rows)
    return {
        "annotation_path": to_project_ref(path),
        "records": len(rows),
        "unique_referenced_images": len({row["filename"] for row in rows}),
        "image_files": len([item for item in (root / "images").glob("*") if item.is_file()]),
        "local_qa_pairs": qa_count,
        "official_readme_qa_pairs": 936,
        "qa_count_difference": qa_count - 936,
        "split_policy": "test_only",
        "license_status": "academic_only",
        "license_source": "external/RSGPT/README.md",
    }


def audit_mmrs_caption(max_samples: int) -> dict[str, Any]:
    sources: dict[str, Any] = {}
    for path in sorted((MMRS_ROOT / "json/caption").glob("caption_*.json")):
        rows = read_json(path)
        probe = rows[:max_samples] if max_samples > 0 else rows
        refs: Counter[int] = Counter()
        invalid = 0
        for row in probe:
            try:
                refs[sum(1 for _ in iter_turn_pairs(row.get("conversations", [])))] += 1
            except ValueError:
                invalid += 1
        sources[path.stem] = {
            "annotation_path": to_project_ref(path),
            "records": len(rows),
            "unique_images": len({row.get("image") for row in rows}),
            "answers": sum(len(row.get("conversations", [])) // 2 for row in rows),
            "probed_reference_counts": dict(sorted(refs.items())),
            "invalid_role_records_in_probe": invalid,
            "license_status": "source_specific_review_required",
        }
    return {
        "records": sum(value["records"] for value in sources.values()),
        "answers": sum(value["answers"] for value in sources.values()),
        "sources": sources,
    }


def audit_dior(max_samples: int) -> dict[str, Any]:
    path = MMRS_ROOT / "json/RSVG/rsvg_trainval.json"
    rows = read_json(path)
    probe = rows[:max_samples] if max_samples > 0 else rows
    directions: Counter[str] = Counter()
    invalid = 0
    for row in probe:
        try:
            for _, prompt, _ in iter_turn_pairs(row.get("conversations", [])):
                if "short description" in prompt:
                    directions["box_to_phrase"] += 1
                elif "bounding box coordinate" in prompt:
                    directions["phrase_to_box"] += 1
                else:
                    directions["unknown"] += 1
        except ValueError:
            invalid += 1
    turn_pairs = sum(len(row.get("conversations", [])) // 2 for row in rows)
    return {
        "annotation_path": to_project_ref(path),
        "source_json_records": len(rows),
        "unique_parent_images": len({row["image"] for row in rows}),
        "task_turn_pairs": turn_pairs,
        "expected_region_pairs": turn_pairs // 2,
        "probed_directions": dict(sorted(directions.items())),
        "invalid_role_records_in_probe": invalid,
        "image_files": len([item for item in (MMRS_ROOT / "RSVG/DIOR_RSVG/images").glob("*") if item.is_file()]),
        "license_status": "source_specific_review_required",
    }


def main() -> None:
    args = parse_args()
    output_dir = description_dir_for_mode(args.mode, args.output_dir)
    report_path = output_dir / "reports/source_audit.json"
    ensure_writable(report_path, args.overwrite, args.dry_run)
    report = {
        "builder_version": BUILDER_VERSION,
        "mode": args.mode,
        "read_policy": {
            "allowed": ["RSGPT/RSICap", "RSGPT/RSIEval", "MMRS/json/caption", "MMRS/json/RSVG"],
            "forbidden": ["MMRS/json/total.json", "classification", "detection", "VQA", "infrared"],
        },
        "sources": {
            "RSICap": audit_rsicap(), "RSIEval": audit_rsieval(),
            "MMRS-Caption": audit_mmrs_caption(args.max_samples), "DIOR-RSVG": audit_dior(args.max_samples),
        },
        "warnings": [
            "RSIEval local QA count is 943 while the official README reports 936; local records are preserved.",
            "RSICap filename prefixes are conservative leakage groups, not verified scene identities.",
            "MMRS component licenses require source-specific review; selected images may be copied into the local research benchmark but must not be publicly redistributed without source-level clearance.",
        ],
        "errors": [],
    }
    print(
        f"[AUDIT] rsicap={report['sources']['RSICap']['records']} "
        f"rsieval={report['sources']['RSIEval']['records']} "
        f"mmrs={report['sources']['MMRS-Caption']['records']} "
        f"dior_parents={report['sources']['DIOR-RSVG']['unique_parent_images']}"
    )
    if not args.dry_run:
        write_json(report_path, report)
        print(f"[AUDIT] report={report_path}")


if __name__ == "__main__":
    main()
