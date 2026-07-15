#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""步骤 3-7：汇总 Description Benchmark M1 统计与清洗报告。

用途：统计 parent、task、source、split、caption、物化图片、存储字节与重复候选。
推荐运行命令：python scripts/3-description/3-7_summarize_description_benchmark.py --mode small --output-dir benchmark/qpsalm_description_v2_small --overwrite
主要输入：验证通过的 indexes/all.jsonl、parent manifest 和 reports。
主要输出：reports/statistics.json 与 reports/cleaning_report.md。
写入行为：只写汇总报告；--dry-run 不写文件。
所属流程：Description Benchmark M1 最终阶段。
"""

from __future__ import annotations

import argparse
from collections import Counter
from typing import Any

from description_common import (
    BUILDER_VERSION, atomic_write_text, description_dir_for_mode, ensure_writable, read_json, read_jsonl,
    to_project_ref, write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="汇总 Description Benchmark M1")
    parser.add_argument("--mode", choices=["small", "full"], default="small")
    parser.add_argument("--output-dir")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def counter(values) -> dict[str, int]:
    return dict(sorted(Counter(str(value) for value in values).items()))


def summarize(
    rows: list[dict[str, Any]], parents: list[dict[str, Any]], candidate_count: int,
    verified_count: int, materialization: dict[str, Any], merge_report: dict[str, Any],
) -> dict[str, Any]:
    answer_weights = Counter()
    for row in rows:
        for answer in row["answers"]:
            answer_weights[f"{float(answer['caption_quality_weight']):.1f}"] += 1
    parent_task_counts = Counter(row["parent_sample_id"] for row in rows)
    return {
        "builder_version": BUILDER_VERSION,
        "num_records": len(rows), "num_parents": len(parents),
        "by_split_records": counter(row["split"] for row in rows),
        "by_split_parents": counter(row["split"] for row in parents),
        "by_source_records": counter(row["source_dataset"] for row in rows),
        "by_source_parents": counter(row["source_dataset"] for row in parents),
        "by_source_membership_parents": counter(
            source for row in parents for source in row.get("source_datasets", [row["source_dataset"]])
        ),
        "by_component": counter(row["component_benchmark"] for row in rows),
        "by_task_family": counter(row["task_family"] for row in rows),
        "by_answer_type": counter(row["answer_type"] for row in rows),
        "by_geometry": counter(row["region_geometry"]["type"] for row in rows),
        "answer_weight_distribution": dict(sorted(answer_weights.items())),
        "quality_flags": counter(flag for row in rows for flag in row["quality_flags"]),
        "answers": sum(len(row["answers"]) for row in rows),
        "avg_tasks_per_parent": len(rows) / max(len(parents), 1),
        "max_tasks_per_parent": max(parent_task_counts.values(), default=0),
        "perceptual_duplicate_candidate_groups": candidate_count,
        "verified_perceptual_duplicate_clusters": verified_count,
        "source_parents_before_canonical_merge": int(merge_report["source_parent_count"]),
        "parents_removed_by_canonical_merge": int(merge_report["parents_removed_by_merge"]),
        "materialized_files": int(materialization["num_files"]),
        "materialized_bytes": int(materialization["total_bytes"]),
        "materialization_status_counts": materialization["status_counts"],
        "materialized_files_by_split": materialization["files_by_split"],
        "materialized_bytes_by_split": materialization["bytes_by_split"],
        "materialized_files_by_source": materialization["files_by_source"],
        "storage_modes": counter(parent.get("storage_mode", "missing") for parent in parents),
    }


def markdown(stats: dict[str, Any], output_dir) -> str:
    lines = [
        "# QPSALM Description Benchmark 清洗报告", "",
        f"- benchmark：`{to_project_ref(output_dir)}`",
        f"- parent 数：{stats['num_parents']}",
        f"- task record 数：{stats['num_records']}",
        f"- answer 数：{stats['answers']}",
        f"- 物化图片数：{stats['materialized_files']}",
        f"- 物化图片总字节：{stats['materialized_bytes']}",
        f"- 平均每 parent task 数：{stats['avg_tasks_per_parent']:.3f}",
        f"- perceptual duplicate 候选组：{stats['perceptual_duplicate_candidate_groups']}", "",
        f"- verified duplicate canonical 簇：{stats['verified_perceptual_duplicate_clusters']}",
        f"- canonical merge 移除的重复 parent：{stats['parents_removed_by_canonical_merge']}", "",
        "## Parent Split",
    ]
    lines.extend(f"- {key}: {value}" for key, value in stats["by_split_parents"].items())
    lines.extend(["", "## 数据源 Parent"])
    lines.extend(f"- {key}: {value}" for key, value in stats["by_source_parents"].items())
    lines.extend(["", "## 任务族"])
    lines.extend(f"- {key}: {value}" for key, value in stats["by_task_family"].items())
    lines.extend(["", "## Caption 质量权重"])
    lines.extend(f"- {key}: {value}" for key, value in stats["answer_weight_distribution"].items())
    lines.extend(["", "## 质量标记"])
    lines.extend(f"- {key}: {value}" for key, value in stats["quality_flags"].items())
    lines.extend([
        "", "## 协议说明",
        "- 本 benchmark 已复制所有入选 parent 图片，训练、验证和推理不依赖 datasets 图片。",
        "- source indexes 与 provenance 保留 datasets 逻辑路径，只用于数据血缘审计。",
        "- DIOR 只用于区域对齐，不作为详细区域 caption 真值。",
        "- dHash 只生成候选；RGB64 MAE 通过门槛的同图重编码已合并为 canonical parent。",
        "- train_eligible.jsonl 排除了零权重答案，完整审计索引仍保留原始记录。",
        "- benchmark 图片仅用于本地研究；未经逐源许可审核不得公开重新分发。",
        "- source_specific_review_required 表示许可仍需逐源人工核查。", "",
    ])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    output_dir = description_dir_for_mode(args.mode, args.output_dir)
    validation = read_json(output_dir / "reports/validation_report.json")
    if validation.get("errors"):
        raise ValueError("validation_report.json 仍有 errors，禁止生成最终汇总")
    rows = read_jsonl(output_dir / "indexes/all.jsonl")
    parents = read_jsonl(output_dir / "manifests/parent_manifest.jsonl")
    candidates = read_jsonl(output_dir / "manifests/perceptual_duplicate_candidates.jsonl")
    verified = read_jsonl(output_dir / "manifests/verified_perceptual_duplicates.jsonl")
    materialization = read_json(output_dir / "reports/materialization_report.json")
    merge_report = read_json(output_dir / "reports/canonical_merge_report.json")
    stats = summarize(rows, parents, len(candidates), len(verified), materialization, merge_report)
    json_path = output_dir / "reports/statistics.json"
    md_path = output_dir / "reports/cleaning_report.md"
    for path in (json_path, md_path):
        ensure_writable(path, args.overwrite, args.dry_run)
    print(f"[SUMMARY] parents={stats['num_parents']} records={stats['num_records']} tasks={len(stats['by_task_family'])}")
    if not args.dry_run:
        write_json(json_path, stats)
        atomic_write_text(md_path, markdown(stats, output_dir))
        print(f"[SUMMARY] reports={output_dir / 'reports'}")


if __name__ == "__main__":
    main()
