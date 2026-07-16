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

from segdesc_common import (
    BUILDER_VERSION,
    INDEX_SCHEMA,
    STATISTICS_PROTOCOL,
    VALIDATION_PROTOCOL,
    read_json,
    read_jsonl,
    resolve_path,
    write_json,
)


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
    if validation.get("protocol") != VALIDATION_PROTOCOL:
        raise ValueError("统一索引 validation protocol 过期，请重新执行 5-2")
    if validation.get("component_contracts_verified") is not True:
        raise ValueError("统一索引 component contracts 未通过独立验证")
    manifest = read_json(output / "manifests/component_manifest.json")
    if manifest.get("builder_version") != BUILDER_VERSION:
        raise ValueError("统一索引 builder version 过期，请重新执行 5-1")
    if manifest.get("schema_version") != INDEX_SCHEMA or manifest.get("mode") != args.mode:
        raise ValueError("统一索引 schema/mode 与汇总命令不一致")
    rows = read_jsonl(output / "indexes/all.jsonl")
    if args.max_samples > 0:
        rows = rows[:args.max_samples]
    weighted = defaultdict(float)
    for row in rows:
        weighted[str(row["task_group"])] += float(row["sample_weight"])
    report = {
        "protocol": STATISTICS_PROTOCOL,
        "builder_version": BUILDER_VERSION,
        "schema_version": INDEX_SCHEMA,
        "mode": args.mode,
        "num_records": len(rows),
        "num_parents": len({(row["component"], row["parent_sample_id"]) for row in rows}),
        "by_split": dict(sorted(Counter(row["split"] for row in rows).items())),
        "by_task_group": dict(sorted(Counter(row["task_group"] for row in rows).items())),
        "by_component": dict(sorted(Counter(row["component"] for row in rows).items())),
        "weighted_task_mass": dict(sorted(weighted.items())),
        "expert_records": sum(int(row["expert_supervision"]) for row in rows),
        "bridge_status": manifest.get("bridge_status"),
        "expert_index_present": bool(manifest.get("expert_index_present")),
        "expert_index_published": bool(manifest.get("expert_index_published")),
        "stale_expert_index_ignored": bool(manifest.get("stale_expert_index_ignored")),
        "stale_bridge_gate_ignored": bool(manifest.get("stale_bridge_gate_ignored")),
        "bridge_gate": manifest.get("bridge_gate"),
        "component_validation_reports": manifest.get("component_validation_reports"),
        "component_contracts_verified": True,
        "storage_mode": manifest.get("storage_mode"),
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
        f"- Bridge status: {report['bridge_status']}",
        f"- Expert index published: {report['expert_index_published']}",
        "- Storage: component references only",
        "\n## Task groups",
    ]
    lines.extend(f"- {name}: {count}" for name, count in report["by_task_group"].items())
    (output / "reports/summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
