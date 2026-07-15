#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""步骤 5-3：汇总统一 segdesc 引用索引。

用途：输出 split、任务、组件、专家监督和采样权重分布。
推荐运行命令：python scripts/5-segdesc/5-3_summarize_unified_index.py --mode small --overwrite
主要输入：已通过 5-2 的 indexes/all.jsonl。
主要输出：reports/statistics.json 和 reports/summary.md。
写入行为：只写汇总报告；--dry-run 不写文件。
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict

from segdesc_common import read_json, read_jsonl, resolve_path, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="汇总 QPSALM unified segdesc index")
    parser.add_argument("--mode", choices=["small", "full"], default="small")
    parser.add_argument("--output-dir")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = resolve_path(args.output_dir or f"benchmark/multisource_landslide_segdesc_v1_{args.mode}")
    validation = read_json(output / "reports/validation_report.json")
    if validation.get("errors"):
        raise ValueError("统一索引 validation errors 非空，拒绝汇总为可用 benchmark")
    rows = read_jsonl(output / "indexes/all.jsonl")
    if args.max_samples > 0:
        rows = rows[:args.max_samples]
    weighted = defaultdict(float)
    for row in rows:
        weighted[str(row["task_group"])] += float(row["sample_weight"])
    report = {
        "protocol": "qpsalm_segdesc_index_statistics_v1",
        "mode": args.mode,
        "num_records": len(rows),
        "num_parents": len({(row["component"], row["parent_sample_id"]) for row in rows}),
        "by_split": dict(sorted(Counter(row["split"] for row in rows).items())),
        "by_task_group": dict(sorted(Counter(row["task_group"] for row in rows).items())),
        "by_component": dict(sorted(Counter(row["component"] for row in rows).items())),
        "weighted_task_mass": dict(sorted(weighted.items())),
        "expert_records": sum(int(row["expert_supervision"]) for row in rows),
        "storage_mode": "component_references_only",
    }
    print(f"[SEGDESC:SUMMARY] records={len(rows)} parents={report['num_parents']} tasks={len(weighted)}")
    if args.dry_run:
        return
    write_json(output / "reports/statistics.json", report)
    lines = [
        "# Unified Segmentation-Description Index\n",
        f"- Records: {report['num_records']}",
        f"- Scoped parents: {report['num_parents']}",
        f"- Expert records: {report['expert_records']}",
        "- Storage: component references only",
        "\n## Task groups",
    ]
    lines.extend(f"- {name}: {count}" for name, count in report["by_task_group"].items())
    (output / "reports/summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

