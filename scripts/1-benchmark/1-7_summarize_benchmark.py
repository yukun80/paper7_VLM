#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""步骤 1-7：汇总 benchmark 统计，生成中文清洗报告。

用途：汇总样本数、数据集分布、模态组合、尺寸 bucket、标注状态、
指代目标分布和质量标记，为论文图表和实验记录准备统计结果。
主要输入：benchmark/multisource_landslide_v2_<mode>/indexes/all.jsonl。
主要输出：reports/statistics.json 和 reports/cleaning_report.md。
写入行为：不会改写 datasets/ 或索引，只写 benchmark/ 下的统计报告。
所属流程：benchmark 构建 1-7，作为数据构建的最终汇总阶段。
推荐运行命令：python scripts/1-benchmark/1-7_summarize_benchmark.py --benchmark-dir benchmark/multisource_landslide_v2_small
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from geohazard_benchmark_common import (
    DEFAULT_BENCHMARK_ROOT,
    modality_combo,
    project_path_arg,
    read_jsonl,
    referring_target_index_paths,
    split_index_paths,
    to_repo_rel,
    write_json,
)


def counter_dict(counter: Counter[Any]) -> dict[str, int]:
    return {str(key): value for key, value in sorted(counter.items(), key=lambda item: str(item[0]))}


def summarize(samples: list[dict[str, Any]], benchmark_dir: Path) -> dict[str, Any]:
    by_split = Counter(sample.get("split", "unknown") for sample in samples)
    by_dataset = Counter(sample.get("dataset_name", "unknown") for sample in samples)
    by_combo = Counter(modality_combo(sample) for sample in samples)
    by_task = Counter(sample.get("task_type", "unknown") for sample in samples)
    by_level = Counter(sample.get("source_level", "unknown") for sample in samples)
    by_bucket = Counter((sample.get("spatial") or {}).get("bucket_size", "unknown") for sample in samples)
    by_region: dict[str, Counter[str]] = defaultdict(Counter)
    empty_masks = 0
    bbox_status = Counter()
    quality_flags = Counter()
    for sample in samples:
        by_region[str(sample.get("dataset_name", "unknown"))][str(sample.get("region", "unknown"))] += 1
        mask = sample.get("mask") or {}
        if mask.get("empty_mask") is True:
            empty_masks += 1
        bbox_status[mask.get("bbox_status", "no_mask")] += 1
        for flag in sample.get("quality_flags") or []:
            quality_flags[flag] += 1
    materialized_files = list((benchmark_dir / "data").glob("**/*.npy"))
    preview_files = list((benchmark_dir / "data").glob("**/preview/*.png"))
    referring_preview_files = list((benchmark_dir / "data").glob("**/preview/referring.png"))
    referring_targets = read_jsonl(referring_target_index_paths(benchmark_dir)["all"])
    referring_by_split = Counter(sample.get("split", "unknown") for sample in referring_targets)
    referring_by_category = Counter(sample.get("category", "unknown") for sample in referring_targets)
    referring_by_subtype = Counter(
        f"{sample.get('category', 'unknown')}:{sample.get('subtype', 'unknown')}"
        for sample in referring_targets
    )
    referring_by_parent = Counter(sample.get("parent_sample_id", "unknown") for sample in referring_targets)
    referring_flags = Counter()
    target_area_ratios = []
    for sample in referring_targets:
        for flag in sample.get("quality_flags") or []:
            if str(flag).startswith("referring_"):
                referring_flags[flag] += 1
        target_mask = sample.get("target_mask") or {}
        if target_mask.get("area_ratio") is not None:
            target_area_ratios.append(float(target_mask["area_ratio"]))
    if target_area_ratios:
        target_area_stats = {
            "min": min(target_area_ratios),
            "mean": sum(target_area_ratios) / len(target_area_ratios),
            "max": max(target_area_ratios),
        }
    else:
        target_area_stats = {"min": None, "mean": None, "max": None}
    return {
        "num_samples": len(samples),
        "by_split": counter_dict(by_split),
        "by_dataset": counter_dict(by_dataset),
        "by_modality_combo": counter_dict(by_combo),
        "by_task_type": counter_dict(by_task),
        "by_source_level": counter_dict(by_level),
        "by_bucket_size": counter_dict(by_bucket),
        "by_region": {dataset: counter_dict(counter) for dataset, counter in sorted(by_region.items())},
        "empty_mask_samples": empty_masks,
        "bbox_status": counter_dict(bbox_status),
        "quality_flags": counter_dict(quality_flags),
        "materialized_data": {
            "num_npy_files": len(materialized_files),
            "bytes": sum(path.stat().st_size for path in materialized_files),
            "num_preview_files": len(preview_files),
            "num_referring_preview_files": len(referring_preview_files),
        },
        "referring_targets": {
            "num_samples": len(referring_targets),
            "by_split": counter_dict(referring_by_split),
            "by_category": counter_dict(referring_by_category),
            "by_subtype": counter_dict(referring_by_subtype),
            "avg_targets_per_parent": (len(referring_targets) / len(referring_by_parent)) if referring_by_parent else 0.0,
            "parents_with_targets": len(referring_by_parent),
            "target_area_ratio": target_area_stats,
            "quality_flags": counter_dict(referring_flags),
        },
    }


def report_markdown(stats: dict[str, Any], benchmark_dir: Path) -> str:
    lines: list[str] = []
    lines.append("# 多源滑坡 benchmark 清洗报告")
    lines.append("")
    lines.append(f"- benchmark 目录：`{to_repo_rel(benchmark_dir)}`")
    lines.append(f"- 样本总数：{stats['num_samples']}")
    lines.append(f"- 空 mask/负样本数量：{stats['empty_mask_samples']}")
    lines.append(f"- 物化 `.npy` 文件数：{stats['materialized_data']['num_npy_files']}")
    lines.append(f"- preview PNG 文件数：{stats['materialized_data']['num_preview_files']}")
    lines.append(f"- referring preview PNG 文件数：{stats['materialized_data']['num_referring_preview_files']}")
    lines.append(f"- 物化数据大小：{stats['materialized_data']['bytes']} bytes")
    lines.append(f"- 指代目标样本数量：{stats['referring_targets']['num_samples']}")
    lines.append(f"- 平均每个父样本 target 数：{stats['referring_targets']['avg_targets_per_parent']:.3f}")
    lines.append("")
    lines.append("## Split 分布")
    for key, value in stats["by_split"].items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("## 数据集分布")
    for key, value in stats["by_dataset"].items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("## 模态组合分布")
    for key, value in stats["by_modality_combo"].items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("## 尺寸 bucket 分布")
    for key, value in stats["by_bucket_size"].items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("## bbox 状态")
    for key, value in stats["bbox_status"].items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("## 指代目标类别分布")
    for key, value in stats["referring_targets"]["by_category"].items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("## 指代目标子类分布")
    for key, value in stats["referring_targets"]["by_subtype"].items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("## 指代目标质量标记")
    for key, value in stats["referring_targets"]["quality_flags"].items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("## 主要质量标记")
    for key, value in stats["quality_flags"].items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("## 说明")
    lines.append("- 最终索引应只指向 benchmark 内部 `data/` 下的 `.npy` 文件。")
    lines.append("- source_* 索引保留原始 datasets/ 路径，仅用于复现物化过程。")
    lines.append("- HDF5/NetCDF 通道语义、InSAR 单位与方向应在训练 dataloader 和论文数据说明中继续补全。")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="汇总 benchmark 统计并生成中文清洗报告。")
    parser.add_argument("--benchmark-dir", type=project_path_arg, default=DEFAULT_BENCHMARK_ROOT, help="后缀式 small 或 full benchmark 输出目录。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = read_jsonl(split_index_paths(args.benchmark_dir)["all"])
    stats = summarize(samples, args.benchmark_dir)
    reports_dir = args.benchmark_dir / "reports"
    write_json(reports_dir / "statistics.json", stats)
    (reports_dir / "cleaning_report.md").write_text(report_markdown(stats, args.benchmark_dir), encoding="utf-8")
    print(f"统计报告已生成: {to_repo_rel(reports_dir / 'statistics.json')} 和 {to_repo_rel(reports_dir / 'cleaning_report.md')}")


if __name__ == "__main__":
    main()
