#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""步骤 3-2：构建 RSICap、RSIEval 与 MMRS Caption 全图描述源索引。

用途：统一多参考 caption、原始 instruction、图像 hash、许可和训练质量权重。
推荐运行命令：python scripts/3-description/3-2_build_global_caption_index.py --mode small --output-dir benchmark/qpsalm_description_v2_small --overwrite
主要输入：RSGPT captions/annotations 和 MMRS json/caption/*.json。
主要输出：indexes/global_caption_source.jsonl、manifests/global_caption_parents.jsonl 和构建报告。
写入行为：只写指向原始 datasets 的 source index；正式图片复制由 3-5 执行。
所属流程：Description Benchmark M1，抽样与 split 在 3-4 执行。
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from description_common import (
    BUILDER_VERSION, MMRS_ROOT, RSGPT_ROOT, answer_record, base_record, deduplicate_texts,
    description_dir_for_mode, ensure_writable, full_image_geometry, iter_turn_pairs,
    mmrs_data_path, probe_image, read_json, scene_prefix, single_image_visual_ref,
    stable_id, to_project_ref, write_json, write_jsonl,
)


CANONICAL_INSTRUCTION = "Describe this remote sensing image in detail."
MMRS_SOURCES = {
    "caption_nwpu": ("MMRS-NWPU-Caption", "nwpu_caption"),
    "caption_rsicd": ("MMRS-RSICD", "rsicd"),
    "caption_rsitmd": ("MMRS-RSITMD", "rstimd_caption"),
    "caption_syndney": ("MMRS-Sydney-Caption", "syndney_caption"),
    "caption_ucm": ("MMRS-UCM-Caption", "ucm_caption"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建 global caption source index")
    parser.add_argument("--mode", choices=["small", "full"], default="small")
    parser.add_argument("--output-dir")
    parser.add_argument("--max-samples", type=int, default=0, help="smoke 时每个来源最多解析的 parent 数")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def parent_manifest(row: dict[str, Any]) -> dict[str, Any]:
    provenance = row["provenance"]
    visual_ref = row["visual_ref"]
    return {
        "parent_sample_id": row["parent_sample_id"], "source_dataset": row["source_dataset"],
        "source_image_path": visual_ref["path"], "width": visual_ref["width"], "height": visual_ref["height"],
        "sha256": visual_ref["sha256"], "dhash64": visual_ref["dhash64"],
        "source_scene_group": provenance.get("source_scene_group"),
        "source_scene_group_status": provenance.get("source_scene_group_status"),
        "source_split": provenance.get("source_split"),
        "task_count": 1,
        "stratum": row.get("sampling_metadata", {}),
    }


def build_rsicap(limit: int) -> list[dict[str, Any]]:
    annotation_path = RSGPT_ROOT / "RSICap/captions.json"
    source_rows = read_json(annotation_path)["annotations"]
    if limit > 0:
        source_rows = source_rows[:limit]
    output: list[dict[str, Any]] = []
    for index, source in enumerate(source_rows):
        image_path = RSGPT_ROOT / "RSICap/images" / source["filename"]
        meta = probe_image(image_path)
        parent_id = stable_id("rsicap", source["filename"])
        caption, flags = answer_record(str(source["text_output"]).strip(), "human", "RSICap")
        row = base_record(
            sample_id=f"{parent_id}__global_caption", parent_sample_id=parent_id,
            source_dataset="RSICap", task_family="global_caption",
            visual_ref=single_image_visual_ref(image_path, "RSICap", meta), region_geometry=full_image_geometry(),
            target_status="present", region_source="full_image", instruction=CANONICAL_INSTRUCTION,
            answer_type="natural_caption", answers=[caption], quality_flags=flags,
            provenance={
                "annotation_path": to_project_ref(annotation_path),
                "source_image_path": to_project_ref(image_path),
                "original_record_id": str(source.get("image_id", index)),
                "source_instruction": str(source.get("text_input", "")),
                "source_scene_group": scene_prefix(source["filename"]),
                "source_scene_group_status": "conservative_filename_prefix_fallback",
                "source_split": None, "license_status": "academic_only",
                "license_source": "external/RSGPT/README.md", "annotation_origin": "human",
            },
        )
        row["sampling_metadata"] = {"source_group": "RSICap", "caption_length_bin": min(len(caption["text"].split()) // 16, 8)}
        output.append(row)
    return output


def build_rsieval(limit: int) -> list[dict[str, Any]]:
    annotation_path = RSGPT_ROOT / "RSIEval/annotations.json"
    source_rows = read_json(annotation_path)["annotations"]
    if limit > 0:
        source_rows = source_rows[:limit]
    output: list[dict[str, Any]] = []
    for index, source in enumerate(source_rows):
        image_path = RSGPT_ROOT / "RSIEval/images" / source["filename"]
        meta = probe_image(image_path)
        parent_id = stable_id("rsieval", source["filename"])
        caption, flags = answer_record(str(source["caption"]).strip(), "human", "RSIEval")
        row = base_record(
            sample_id=f"{parent_id}__global_caption", parent_sample_id=parent_id,
            source_dataset="RSIEval", task_family="global_caption",
            visual_ref=single_image_visual_ref(image_path, "RSIEval", meta), region_geometry=full_image_geometry(),
            target_status="present", region_source="full_image", instruction=CANONICAL_INSTRUCTION,
            answer_type="natural_caption", answers=[caption], quality_flags=flags, split="test",
            provenance={
                "annotation_path": to_project_ref(annotation_path),
                "source_image_path": to_project_ref(image_path), "original_record_id": str(index),
                "source_instruction": CANONICAL_INSTRUCTION,
                "source_scene_group": scene_prefix(source["filename"]),
                "source_scene_group_status": "conservative_filename_prefix_fallback",
                "source_split": "test", "local_qa_count": len(source.get("qa_pairs", [])),
                "license_status": "academic_only", "license_source": "external/RSGPT/README.md",
                "annotation_origin": "human",
            },
        )
        row["sampling_metadata"] = {"source_group": "RSIEval", "caption_length_bin": min(len(caption["text"].split()) // 16, 8)}
        output.append(row)
    return output


def test_names(dataset_dir: str) -> set[str]:
    root = MMRS_ROOT / "caption" / dataset_dir / "test"
    return {path.name for path in root.rglob("*") if path.is_file()} if root.exists() else set()


def build_mmrs(limit: int) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for annotation_path in sorted((MMRS_ROOT / "json/caption").glob("caption_*.json")):
        source_dataset, dataset_dir = MMRS_SOURCES[annotation_path.stem]
        source_rows = read_json(annotation_path)
        if limit > 0:
            source_rows = source_rows[:limit]
        official_test = test_names(dataset_dir)
        for index, source in enumerate(source_rows):
            image_path = mmrs_data_path(str(source["image"]))
            meta = probe_image(image_path)
            turns = list(iter_turn_pairs(source.get("conversations", [])))
            texts = deduplicate_texts(response for _, _, response in turns)
            answers: list[dict[str, Any]] = []
            flags: list[str] = []
            for text in texts:
                item, item_flags = answer_record(text, "source_dataset_caption", source_dataset)
                answers.append(item)
                flags.extend(item_flags)
            parent_id = stable_id(source_dataset, source["image"])
            source_split = "test" if image_path.name in official_test else None
            average_length = sum(len(item["text"].split()) for item in answers) / max(len(answers), 1)
            row = base_record(
                sample_id=f"{parent_id}__global_caption", parent_sample_id=parent_id,
                source_dataset=source_dataset, task_family="global_caption",
                visual_ref=single_image_visual_ref(image_path, source_dataset, meta), region_geometry=full_image_geometry(),
                target_status="present", region_source="full_image", instruction=CANONICAL_INSTRUCTION,
                answer_type="multi_reference_caption", answers=answers, quality_flags=flags, split=source_split,
                provenance={
                    "annotation_path": to_project_ref(annotation_path),
                    "source_image_path": to_project_ref(image_path), "original_record_id": str(index),
                    "source_image_ref": str(source["image"]),
                    "source_instructions": deduplicate_texts(prompt for _, prompt, _ in turns),
                    "source_scene_group": None, "source_scene_group_status": "unavailable",
                    "source_split": source_split, "license_status": "source_specific_review_required",
                    "license_source": f"MMRS-1M/{dataset_dir}", "annotation_origin": "source_dataset_caption",
                },
            )
            row["sampling_metadata"] = {
                "source_group": source_dataset, "caption_length_bin": min(int(average_length) // 8, 8),
                "num_references": len(answers),
            }
            output.append(row)
    return output


def main() -> None:
    args = parse_args()
    output_dir = description_dir_for_mode(args.mode, args.output_dir)
    index_path = output_dir / "indexes/global_caption_source.jsonl"
    parent_path = output_dir / "manifests/global_caption_parents.jsonl"
    report_path = output_dir / "reports/global_caption_build.json"
    for path in (index_path, parent_path, report_path):
        ensure_writable(path, args.overwrite, args.dry_run)

    rows = build_rsicap(args.max_samples) + build_rsieval(args.max_samples) + build_mmrs(args.max_samples)
    rows.sort(key=lambda row: (row["source_dataset"], row["parent_sample_id"]))
    parents = [parent_manifest(row) for row in rows]
    counts = Counter(row["source_dataset"] for row in rows)
    report = {
        "builder_version": BUILDER_VERSION, "mode": args.mode, "num_parents": len(parents),
        "num_records": len(rows), "source_counts": dict(sorted(counts.items())),
        "answer_count": sum(len(row["answers"]) for row in rows),
        "quality_flag_counts": dict(sorted(Counter(flag for row in rows for flag in row["quality_flags"]).items())),
        "errors": [],
    }
    print(f"[GLOBAL] source_parents={len(parents)} sources={len(counts)}")
    if not args.dry_run:
        write_jsonl(index_path, rows)
        write_jsonl(parent_path, parents)
        write_json(report_path, report)
        print(f"[GLOBAL] index={index_path}")


if __name__ == "__main__":
    main()
