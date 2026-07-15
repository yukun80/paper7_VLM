#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""步骤 5-2：验证统一索引引用、split 和 component 指纹。

用途：确保统一索引不复制数据、不产生悬空引用，也不伪造 expert supervision。
推荐运行命令：python scripts/5-segdesc/5-2_validate_unified_index.py --mode small --overwrite
主要输入：5-1 输出及三个 component benchmark。
主要输出：reports/validation_report.json；errors 非空时非零退出。
写入行为：只写验证报告；--dry-run 不写文件。
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from segdesc_common import (
    INDEX_SCHEMA, benchmark_root, project_ref, read_json, read_jsonl,
    resolve_path, sha256_file, write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="验证 QPSALM unified segdesc index")
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
    all_path = output / "indexes/all.jsonl"
    errors: list[str] = []
    warnings: list[str] = []
    if not all_path.is_file():
        errors.append(f"缺少统一索引: {all_path}")
        rows = []
    else:
        rows = read_jsonl(all_path)
    if args.max_samples > 0:
        rows = rows[:args.max_samples]
    ids = [str(row.get("unified_record_id") or "") for row in rows]
    if len(ids) != len(set(ids)):
        errors.append("unified_record_id 不唯一")
    source_cache: dict[Path, tuple[str, list[dict], dict[str, dict]]] = {}
    parent_splits: dict[tuple[str, str], set[str]] = {}
    for row in rows:
        if row.get("schema_version") != INDEX_SCHEMA:
            errors.append(f"schema_version 非法: {row.get('unified_record_id')}")
            continue
        index = resolve_path(str(row.get("component_index") or ""))
        if not index.is_file():
            errors.append(f"component index 缺失: {index}")
            continue
        if index not in source_cache:
            source_rows = read_jsonl(index)
            mapping = {}
            for source in source_rows:
                key = str(source.get("sample_id") or source.get("bridge_record_id") or "")
                if key in mapping:
                    errors.append(f"component record ID 重复: {index}:{key}")
                mapping[key] = source
            source_cache[index] = (sha256_file(index), source_rows, mapping)
        digest, source_rows, mapping = source_cache[index]
        if digest != row.get("component_index_sha256"):
            errors.append(f"component index hash 不一致: {project_ref(index)}")
        component_record_id = str(row.get("component_record_id") or "")
        line_number = row.get("component_line_number")
        if not isinstance(line_number, int) or not (1 <= line_number <= len(source_rows)):
            errors.append(
                f"component line number 非法: {row.get('unified_record_id')}:{line_number}"
            )
            source_at_line = None
        else:
            source_at_line = source_rows[line_number - 1]
            line_record_id = str(
                source_at_line.get("sample_id")
                or source_at_line.get("bridge_record_id")
                or ""
            )
            if line_record_id != component_record_id:
                errors.append(
                    "component line/ID 不一致: "
                    f"{row.get('unified_record_id')} expected={component_record_id} "
                    f"actual={line_record_id}"
                )
        source = mapping.get(component_record_id)
        if source is None:
            errors.append(f"component record 缺失: {component_record_id}")
            continue
        if source_at_line is not None and source_at_line != source:
            errors.append(f"component line 内容不一致: {row.get('unified_record_id')}")
        expected_parent = str(source.get("parent_sample_id") or source.get("sample_id") or source.get("bridge_record_id"))
        if expected_parent != str(row.get("parent_sample_id")):
            errors.append(f"parent 引用不一致: {row.get('unified_record_id')}")
        if str(source.get("split")) != str(row.get("split")):
            errors.append(f"split 引用不一致: {row.get('unified_record_id')}")
        if row.get("expert_supervision") and not isinstance(source.get("expert_target"), dict):
            errors.append(f"expert row 缺少 expert_target: {row.get('unified_record_id')}")
        parent_splits.setdefault(
            (str(row.get("component")), str(row.get("parent_sample_id"))), set()
        ).add(str(row.get("split")))
    crossed = [key for key, values in parent_splits.items() if len(values) > 1]
    if crossed:
        errors.append(f"component parent 跨 split: count={len(crossed)} examples={crossed[:3]}")
    unexpected_files = [
        path for path in output.rglob("*")
        if path.is_file() and path.suffix.casefold() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".npy"}
    ] if output.exists() else []
    if unexpected_files:
        errors.append(f"统一引用 benchmark 不得复制图像/mask: {unexpected_files[:3]}")
    part_files = list(output.rglob("*.part")) if output.exists() else []
    if part_files:
        errors.append(f"存在未完成 .part 文件: {part_files[:3]}")
    report = {
        "protocol": "qpsalm_segdesc_index_validation_v1",
        "schema_version": INDEX_SCHEMA,
        "num_records": len(rows),
        "by_split": dict(sorted(Counter(str(row.get("split")) for row in rows).items())),
        "by_task_group": dict(sorted(Counter(str(row.get("task_group")) for row in rows).items())),
        "referenced_component_indexes": len(source_cache),
        "errors": errors,
        "warnings": warnings,
        "status": "valid" if not errors else "invalid",
    }
    print(f"[SEGDESC:VALIDATE] records={len(rows)} errors={len(errors)} warnings={len(warnings)}")
    if not args.dry_run:
        write_json(output / "reports/validation_report.json", report)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
