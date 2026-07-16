#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""步骤 5-1：发布分割、全图描述、区域对齐和 Bridge 的统一引用索引。

用途：只建立 component record 引用与任务采样元数据，不复制图像、mask 或多源数组。
推荐运行命令：python scripts/5-segdesc/5-1_build_unified_index.py --mode small --overwrite
主要输入：Landslide V2、Description V2 与 Landslide Bridge 已验证索引。
主要输出：indexes/all/train/dev/val/test.jsonl 与 component_manifest.json。
写入行为：只写 multisource_landslide_segdesc_v1_<mode>；--dry-run 不写文件。
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from pathlib import Path
from typing import Any

from segdesc_common import (
    BUILDER_VERSION,
    INDEX_SCHEMA,
    SEGMENTATION_INSTRUCTION_REPORT,
    TASK_WEIGHTS,
    bridge_expert_artifact_errors,
    bridge_publication_policy,
    component_contract_errors,
    component_validation_contract,
    ensure_output,
    project_ref,
    read_json,
    read_jsonl,
    resolve_path,
    segmentation_instruction_contract_errors,
    segmentation_instruction_validation_contract,
    sha256_file,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建 QPSALM segmentation-description unified index")
    parser.add_argument("--mode", choices=["small", "full"], default="small")
    parser.add_argument("--segmentation-benchmark")
    parser.add_argument("--description-benchmark")
    parser.add_argument("--bridge-benchmark")
    parser.add_argument("--output-dir")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _validated(root: Path, *, component: str, mode: str) -> dict[str, Any]:
    path = root / "reports/validation_report.json"
    if not path.is_file():
        raise FileNotFoundError(f"缺少 validation report: {path}")
    report = read_json(path)
    if report.get("errors"):
        raise ValueError(f"component validation errors 非空: {path}")
    contract_errors = component_contract_errors(
        component, report, mode=mode, root=root,
    )
    if contract_errors:
        raise ValueError(
            f"component validation contract 过期: {component}:{path}: "
            + "; ".join(contract_errors)
        )
    if component == "bridge" and report.get("status") == "expert_pilot_frozen":
        artifact_errors = bridge_expert_artifact_errors(root, report)
        if artifact_errors:
            raise ValueError(
                "frozen Bridge expert artifact contract 无效: "
                + "; ".join(artifact_errors)
            )
    return report


def _validation_binding(
    root: Path,
    report: dict[str, Any],
    *,
    component: str,
    mode: str,
) -> dict[str, Any]:
    path = root / "reports/validation_report.json"
    return {
        "path": project_ref(path),
        "sha256": sha256_file(path),
        "status": report.get("status"),
        "errors": len(report.get("errors") or []),
        "contract": component_validation_contract(
            component, mode=mode, root=root,
        ),
    }


def _segmentation_instruction_binding(root: Path) -> dict[str, Any]:
    path = root / SEGMENTATION_INSTRUCTION_REPORT
    if not path.is_file():
        raise FileNotFoundError(f"缺少 segmentation instruction validation report: {path}")
    report = read_json(path)
    if report.get("errors"):
        raise ValueError(f"segmentation instruction validation errors 非空: {path}")
    contract_errors = segmentation_instruction_contract_errors(report, root=root)
    if contract_errors:
        raise ValueError(
            f"segmentation instruction validation contract 过期: {path}: "
            + "; ".join(contract_errors)
        )
    return {
        "path": project_ref(path),
        "sha256": sha256_file(path),
        "errors": len(report.get("errors") or []),
        "contract": segmentation_instruction_validation_contract(root),
    }


def _record_id(row: dict[str, Any]) -> str:
    for key in ("sample_id", "bridge_record_id"):
        if row.get(key):
            return str(row[key])
    raise ValueError(f"component row 缺少 record ID: {sorted(row)[:8]}")


def _task_row(
    row: dict[str, Any], *, component: str, task_group: str, index: Path,
    index_sha256: str, line_number: int,
) -> dict[str, Any]:
    record_id = _record_id(row)
    parent = str(row.get("parent_sample_id") or record_id)
    split = str(row.get("split") or "")
    key = f"{component}:{record_id}:{task_group}"
    return {
        "schema_version": INDEX_SCHEMA,
        "unified_record_id": hashlib.sha256(key.encode()).hexdigest()[:24],
        "component": component,
        "component_index": project_ref(index),
        "component_index_sha256": index_sha256,
        "component_line_number": int(line_number),
        "component_record_id": record_id,
        "parent_sample_id": parent,
        "split": split,
        "task_group": task_group,
        "task_family": str(row.get("task_family") or "unknown"),
        "sample_weight": float(TASK_WEIGHTS[task_group]),
        "expert_supervision": task_group == "region_description_expert",
    }


def _append(rows: list[dict[str, Any]], index: Path, component: str, task_group_fn) -> None:
    source = read_jsonl(index)
    digest = sha256_file(index)
    added_by_task: Counter[str] = Counter()
    for line_number, row in enumerate(source, start=1):
        task_group = task_group_fn(row)
        if task_group is None:
            continue
        value = _task_row(
            row, component=component, task_group=task_group,
            index=index, index_sha256=digest, line_number=line_number,
        )
        rows.append(value)
        added_by_task[task_group] += 1
    print(
        f"[SEGDESC:INDEX] component={component} source={index.name} "
        f"records={sum(added_by_task.values())} tasks={dict(sorted(added_by_task.items()))}"
    )


def main() -> None:
    args = parse_args()
    segmentation = resolve_path(args.segmentation_benchmark or f"benchmark/multisource_landslide_v2_{args.mode}")
    description = resolve_path(args.description_benchmark or f"benchmark/qpsalm_description_v2_{args.mode}")
    bridge = resolve_path(args.bridge_benchmark or f"benchmark/landslide_region_description_v1_{args.mode}")
    output = resolve_path(args.output_dir or f"benchmark/multisource_landslide_segdesc_v1_{args.mode}")
    description_report = _validated(
        description, component="description", mode=args.mode,
    )
    segmentation_report = _validated(
        segmentation, component="segmentation", mode=args.mode,
    )
    bridge_report = _validated(bridge, component="bridge", mode=args.mode)
    bridge_status = str(bridge_report.get("status") or "")
    ensure_output(output, args.overwrite, args.dry_run)

    rows: list[dict[str, Any]] = []
    for split, source_split in (("train", "train"), ("val", "val"), ("test", "test")):
        index = segmentation / f"indexes/instruction_{source_split}.jsonl"
        _append(rows, index, "landslide_segmentation_v2", lambda _row: "segmentation")
    for split in ("train", "dev", "test"):
        name = "train_eligible.jsonl" if split == "train" else f"{split}.jsonl"
        index = description / "indexes" / name
        _append(
            rows, index, "description_v2",
            lambda row: (
                "global_caption" if row.get("task_family") == "global_caption"
                else "region_alignment" if row.get("task_family") in {
                    "region_referring_expression", "region_grounding",
                } else None
            ),
        )
    auto_index = bridge / "indexes/auto_train.jsonl"
    _append(rows, auto_index, "landslide_bridge_v1", lambda _row: "region_description_auto")
    expert_index = bridge / "indexes/expert_all.jsonl"
    gate_path = bridge / "manifests/evaluation_gate_manifest.json"
    publication = bridge_publication_policy(
        bridge_status,
        expert_index_present=expert_index.is_file(),
        gate_present=gate_path.is_file(),
    )
    expert_published = publication["expert_index_published"]
    if expert_published:
        _append(rows, expert_index, "landslide_bridge_v1", lambda _row: "region_description_expert")

    if args.max_samples > 0:
        rows = sorted(
            rows,
            key=lambda row: hashlib.sha256(
                f"{args.seed}:{row['unified_record_id']}".encode()
            ).hexdigest(),
        )[:args.max_samples]
    rows.sort(key=lambda row: (row["split"], row["task_group"], row["unified_record_id"]))
    duplicates = len(rows) - len({row["unified_record_id"] for row in rows})
    if duplicates:
        raise ValueError(f"unified_record_id collision/duplicate: {duplicates}")
    components = {
        "segmentation": project_ref(segmentation),
        "description": project_ref(description),
        "bridge": project_ref(bridge),
    }
    report = {
        "builder_version": BUILDER_VERSION,
        "schema_version": INDEX_SCHEMA,
        "mode": args.mode,
        "components": components,
        "component_validation_reports": {
            "segmentation": {
                **_validation_binding(
                    segmentation, segmentation_report,
                    component="segmentation", mode=args.mode,
                ),
                "instruction_validation": _segmentation_instruction_binding(
                    segmentation,
                ),
            },
            "description": _validation_binding(
                description, description_report,
                component="description", mode=args.mode,
            ),
            "bridge": _validation_binding(
                bridge, bridge_report, component="bridge", mode=args.mode,
            ),
        },
        "bridge_status": bridge_status,
        "bridge_gate": (
            {"path": project_ref(gate_path), "sha256": sha256_file(gate_path)}
            if publication["bridge_gate_published"] else None
        ),
        "expert_index_present": expert_index.is_file(),
        "expert_index_published": expert_published,
        "stale_expert_index_ignored": publication["stale_expert_index_ignored"],
        "stale_bridge_gate_ignored": publication["stale_bridge_gate_ignored"],
        "num_records": len(rows),
        "by_split": dict(sorted(Counter(row["split"] for row in rows).items())),
        "by_task_group": dict(sorted(Counter(row["task_group"] for row in rows).items())),
        "contains_expert_bridge": any(row["expert_supervision"] for row in rows),
        "storage_mode": "component_references_only",
    }
    print(
        f"[SEGDESC:BUILD] mode={args.mode} records={len(rows)} "
        f"tasks={len(report['by_task_group'])} expert={report['contains_expert_bridge']}"
    )
    if args.dry_run:
        return
    write_jsonl(output / "indexes/all.jsonl", rows)
    for split in ("train", "dev", "val", "test"):
        write_jsonl(output / f"indexes/{split}.jsonl", [row for row in rows if row["split"] == split])
    write_json(output / "manifests/component_manifest.json", report)
    write_json(output / "reports/build_report.json", report)


if __name__ == "__main__":
    main()
