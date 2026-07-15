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
import hashlib
from pathlib import Path

from segdesc_common import (
    BRIDGE_AWAITING_STATUS, BRIDGE_FROZEN_STATUS,
    BUILDER_VERSION, INDEX_SCHEMA, TASK_COMPONENTS, TASK_INDEX_NAMES, TASK_WEIGHTS,
    VALIDATION_PROTOCOL, bridge_publication_policy,
    project_ref, read_json, read_jsonl, resolve_path, sha256_file, write_json,
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


def _validate_publication_manifest(
    output: Path,
    manifest: dict,
    errors: list[str],
    warnings: list[str],
    *,
    expected_mode: str,
) -> tuple[dict[str, Path], str]:
    if manifest.get("builder_version") != BUILDER_VERSION:
        errors.append("component manifest builder_version 过期")
    if manifest.get("schema_version") != INDEX_SCHEMA:
        errors.append("component manifest schema_version 非法")
    if manifest.get("mode") != expected_mode:
        errors.append("component manifest mode 与命令不一致")
    if manifest.get("storage_mode") != "component_references_only":
        errors.append("component manifest storage_mode 非法")
    components = dict(manifest.get("components") or {})
    required_components = {"segmentation", "description", "bridge"}
    if set(components) != required_components:
        errors.append("component manifest components 不完整")
    component_roots: dict[str, Path] = {}
    component_names = {
        "segmentation": "landslide_segmentation_v2",
        "description": "description_v2",
        "bridge": "landslide_bridge_v1",
    }
    for name, component in component_names.items():
        raw_root = components.get(name)
        if not isinstance(raw_root, str) or not raw_root.strip():
            errors.append(f"component benchmark root 未记录: {name}")
            continue
        root = resolve_path(raw_root)
        component_roots[component] = root
        if not root.is_dir():
            errors.append(f"component benchmark root 缺失: {component}:{root}")
    bindings = dict(manifest.get("component_validation_reports") or {})
    if set(bindings) != required_components:
        errors.append("component validation report bindings 不完整")
    for name in sorted(required_components):
        binding = dict(bindings.get(name) or {})
        path = resolve_path(str(binding.get("path") or ""))
        component_root = component_roots.get(component_names[name])
        if component_root is not None and path.resolve(strict=False) != (
            component_root / "reports/validation_report.json"
        ).resolve(strict=False):
            errors.append(f"component validation report 路径越出绑定 benchmark: {name}")
        if not path.is_file():
            errors.append(f"component validation report 缺失: {name}:{path}")
            continue
        if sha256_file(path) != binding.get("sha256"):
            errors.append(f"component validation report hash 不一致: {name}")
            continue
        report = read_json(path)
        if report.get("errors"):
            errors.append(f"component validation report errors 非空: {name}")
        if int(binding.get("errors", -1)) != len(report.get("errors") or []):
            errors.append(f"component validation report errors count 不一致: {name}")
        if binding.get("status") != report.get("status"):
            errors.append(f"component validation report status 不一致: {name}")

    bridge_status = str(manifest.get("bridge_status") or "")
    if (bindings.get("bridge") or {}).get("status") != bridge_status:
        errors.append("Bridge manifest status 与 validation binding 不一致")
    bridge_root = component_roots.get("landslide_bridge_v1")
    actual_expert_present = bool(
        bridge_root is not None and (bridge_root / "indexes/expert_all.jsonl").is_file()
    )
    actual_gate_present = bool(
        bridge_root is not None
        and (bridge_root / "manifests/evaluation_gate_manifest.json").is_file()
    )
    if bool(manifest.get("expert_index_present")) != actual_expert_present:
        errors.append("Bridge expert_index_present 与物理文件状态不一致")
    expert_published = bool(manifest.get("expert_index_published"))
    gate_binding = manifest.get("bridge_gate")
    try:
        expected_publication = bridge_publication_policy(
            bridge_status,
            expert_index_present=actual_expert_present,
            gate_present=actual_gate_present,
        )
    except ValueError as exc:
        expected_publication = {}
        errors.append(str(exc))
    for field in (
        "expert_index_published",
        "stale_expert_index_ignored",
        "stale_bridge_gate_ignored",
    ):
        if expected_publication and bool(manifest.get(field)) != expected_publication[field]:
            errors.append(f"Bridge publication policy 字段不一致: {field}")
    if bridge_status == BRIDGE_FROZEN_STATUS:
        if not expert_published:
            errors.append("frozen Bridge 未发布 expert component")
        if not manifest.get("expert_index_present"):
            errors.append("frozen Bridge manifest 未记录 expert index")
        if not isinstance(gate_binding, dict):
            errors.append("frozen Bridge 缺少 gate binding")
        else:
            gate_path = resolve_path(str(gate_binding.get("path") or ""))
            expected_gate_path = (
                bridge_root / "manifests/evaluation_gate_manifest.json"
                if bridge_root is not None else None
            )
            if expected_gate_path is not None and gate_path.resolve(strict=False) != (
                expected_gate_path.resolve(strict=False)
            ):
                errors.append("Bridge gate 路径越出绑定 benchmark")
            if not gate_path.is_file():
                errors.append(f"Bridge gate 缺失: {gate_path}")
            elif sha256_file(gate_path) != gate_binding.get("sha256"):
                errors.append("Bridge gate hash 不一致")
            else:
                gate = read_json(gate_path)
                if (
                    gate.get("protocol") != "landslide_bridge_evaluation_gate_v2"
                    or gate.get("frozen") is not True
                    or gate.get("status") != "frozen_after_pilot"
                ):
                    errors.append("Bridge gate 不是人工冻结的 v2 Pilot gate")
    elif bridge_status == BRIDGE_AWAITING_STATUS:
        if expert_published or gate_binding is not None:
            errors.append("awaiting-review Bridge 不得发布 expert component 或 gate")
        if manifest.get("stale_expert_index_ignored"):
            warnings.append("Bridge 中存在旧 expert index，但本次统一索引已按协议忽略")
        if manifest.get("stale_bridge_gate_ignored"):
            warnings.append("Bridge 中存在旧 evaluation gate，但本次统一索引已按协议忽略")
    else:
        errors.append(f"Bridge status 不允许发布统一索引: {bridge_status!r}")
    return component_roots, bridge_status


def main() -> None:
    args = parse_args()
    output = resolve_path(args.output_dir or f"benchmark/multisource_landslide_segdesc_v1_{args.mode}")
    all_path = output / "indexes/all.jsonl"
    errors: list[str] = []
    warnings: list[str] = []
    manifest_path = output / "manifests/component_manifest.json"
    build_report_path = output / "reports/build_report.json"
    if not manifest_path.is_file():
        errors.append(f"缺少 component manifest: {manifest_path}")
        manifest = {}
    else:
        manifest = read_json(manifest_path)
    if not build_report_path.is_file():
        errors.append(f"缺少 build report: {build_report_path}")
    elif manifest and read_json(build_report_path) != manifest:
        errors.append("build report 与 component manifest 不一致")
    component_roots, bridge_status = _validate_publication_manifest(
        output, manifest, errors, warnings, expected_mode=args.mode
    )
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
        required_fields = {
            "schema_version", "unified_record_id", "component", "component_index",
            "component_index_sha256", "component_line_number", "component_record_id",
            "parent_sample_id", "split", "task_group", "task_family",
            "sample_weight", "expert_supervision",
        }
        missing = sorted(required_fields - set(row))
        if missing:
            errors.append(f"unified row 缺少字段: {row.get('unified_record_id')}:{missing}")
            continue
        if row.get("schema_version") != INDEX_SCHEMA:
            errors.append(f"schema_version 非法: {row.get('unified_record_id')}")
            continue
        task_group = str(row.get("task_group") or "")
        component = str(row.get("component") or "")
        if task_group not in TASK_WEIGHTS:
            errors.append(f"task_group 非法: {row.get('unified_record_id')}:{task_group}")
            continue
        if component != TASK_COMPONENTS[task_group]:
            errors.append(
                f"task/component 不一致: {row.get('unified_record_id')}:{task_group}/{component}"
            )
        try:
            sample_weight = float(row["sample_weight"])
        except (TypeError, ValueError):
            sample_weight = -1.0
        if abs(sample_weight - TASK_WEIGHTS[task_group]) > 1.0e-12:
            errors.append(f"sample_weight 不一致: {row.get('unified_record_id')}")
        if str(row.get("split")) not in {"train", "dev", "val", "test"}:
            errors.append(f"split 非法: {row.get('unified_record_id')}:{row.get('split')}")
        expected_expert = task_group == "region_description_expert"
        if bool(row.get("expert_supervision")) != expected_expert:
            errors.append(f"expert_supervision 标志不一致: {row.get('unified_record_id')}")
        if expected_expert and bridge_status != BRIDGE_FROZEN_STATUS:
            errors.append(f"未冻结 Bridge 发布了 expert row: {row.get('unified_record_id')}")
        record_id = str(row.get("component_record_id") or "")
        expected_unified_id = hashlib.sha256(
            f"{component}:{record_id}:{task_group}".encode()
        ).hexdigest()[:24]
        if row.get("unified_record_id") != expected_unified_id:
            errors.append(f"unified_record_id 指纹不一致: {row.get('unified_record_id')}")
        index = resolve_path(str(row.get("component_index") or ""))
        if not index.is_file():
            errors.append(f"component index 缺失: {index}")
            continue
        component_root = component_roots.get(component)
        if component_root is None:
            errors.append(f"component root 缺失: {component}")
        else:
            try:
                relative_index = index.resolve(strict=False).relative_to(
                    component_root.resolve(strict=False)
                )
                if not relative_index.parts or relative_index.parts[0] != "indexes":
                    errors.append(
                        f"component index 不位于绑定 benchmark/indexes: {project_ref(index)}"
                    )
                if index.name not in TASK_INDEX_NAMES[task_group]:
                    errors.append(
                        "task/component index 文件不一致: "
                        f"{row.get('unified_record_id')}:{task_group}/{index.name}"
                    )
            except ValueError:
                errors.append(f"component index 越出绑定 benchmark: {project_ref(index)}")
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
        component_record_id = record_id
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
    if args.max_samples <= 0 and rows:
        partition_ids: list[str] = []
        for split in ("train", "dev", "val", "test"):
            split_path = output / f"indexes/{split}.jsonl"
            if not split_path.is_file():
                errors.append(f"缺少 split index: {split_path}")
                continue
            split_rows = read_jsonl(split_path)
            if any(str(row.get("split")) != split for row in split_rows):
                errors.append(f"split index 包含错误 split: {split}")
            partition_ids.extend(str(row.get("unified_record_id") or "") for row in split_rows)
        if Counter(partition_ids) != Counter(ids):
            errors.append("train/dev/val/test partition 与 all index 不一致")
        if int(manifest.get("num_records", -1)) != len(rows):
            errors.append("component manifest num_records 与 all index 不一致")
        manifest_split = {
            str(key): int(value) for key, value in (manifest.get("by_split") or {}).items()
        }
        observed_split = dict(sorted(Counter(str(row.get("split")) for row in rows).items()))
        if manifest_split != observed_split:
            errors.append("component manifest split 统计不一致")
        manifest_tasks = {
            str(key): int(value) for key, value in (manifest.get("by_task_group") or {}).items()
        }
        observed_tasks = dict(sorted(Counter(str(row.get("task_group")) for row in rows).items()))
        if manifest_tasks != observed_tasks:
            errors.append("component manifest task 统计不一致")
        if bool(manifest.get("contains_expert_bridge")) != any(
            bool(row.get("expert_supervision")) for row in rows
        ):
            errors.append("component manifest expert 统计不一致")
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
        "protocol": VALIDATION_PROTOCOL,
        "schema_version": INDEX_SCHEMA,
        "builder_version": manifest.get("builder_version"),
        "mode": args.mode,
        "num_records": len(rows),
        "by_split": dict(sorted(Counter(str(row.get("split")) for row in rows).items())),
        "by_task_group": dict(sorted(Counter(str(row.get("task_group")) for row in rows).items())),
        "referenced_component_indexes": len(source_cache),
        "bridge_status": bridge_status,
        "expert_index_published": bool(manifest.get("expert_index_published")),
        "bridge_gate": manifest.get("bridge_gate"),
        "component_validation_reports": manifest.get("component_validation_reports"),
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
