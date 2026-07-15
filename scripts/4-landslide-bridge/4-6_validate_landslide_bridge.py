#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""步骤 4-6：验证 Landslide Bridge 自动产物与可选专家冻结状态。

用途：检查区域、三级证据、候选隔离、Pilot 审核约束和专家合并完整性。
推荐运行命令：python scripts/4-landslide-bridge/4-6_validate_landslide_bridge.py --mode small --overwrite
主要输入：4-1 到 4-5 的 Bridge indexes、manifests、mask 和配置/schema。
主要输出：reports/validation_report.json；errors 非空时返回非零状态。
写入行为：只写验证报告；--dry-run 不写报告。
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from landslide_bridge_common import (
    BUILDER_VERSION,
    REPO_ROOT,
    binary_mask,
    bridge_dir,
    ensure_writable,
    load_config,
    mask_digest,
    read_json,
    read_jsonl,
    resolve_project_path,
    sha256_file,
    validate_bridge_structured_target,
    write_json,
)


REQUIRED_FIELDS = {
    "schema_version", "bridge_record_id", "parent_sample_id", "source_benchmark",
    "source_parent_index", "split", "dataset_name", "region_id", "region_source",
    "target_status", "task_family", "instruction", "condition", "answer_type",
    "region_geometry", "region_mask", "visual_ref", "modality_metadata",
    "modality_evidence", "structured_targets", "candidate", "review", "provenance",
    "quality_flags",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="验证 Landslide Bridge M2")
    parser.add_argument("--mode", choices=["small", "full"], default="small")
    parser.add_argument("--output-dir")
    parser.add_argument("--config", default="configs/landslide_bridge_v1.yaml")
    parser.add_argument("--require-expert-complete", action="store_true")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _record_paths(payload: Any) -> list[str]:
    result: list[str] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key in {"path", "parent_index", "source_parent_index", "source_benchmark"} and isinstance(value, str):
                result.append(value)
            result.extend(_record_paths(value))
    elif isinstance(payload, list):
        for value in payload:
            result.extend(_record_paths(value))
    return result


def _validate_schema_config(config_path: str, errors: list[str]) -> dict[str, Any]:
    try:
        config = load_config(config_path)
    except Exception as exc:
        errors.append(f"Bridge config 无法解析: {exc}")
        return {}
    schema_path = REPO_ROOT / "configs/qpsalm_landslide_region_description_v1.schema.json"
    try:
        schema = read_json(schema_path)
        if schema.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
            errors.append("Bridge JSON Schema 必须使用 draft 2020-12")
    except Exception as exc:
        errors.append(f"Bridge JSON Schema 无法解析: {exc}")
    return config


def validate_record(record: dict[str, Any], errors: list[str], mask_refs: set[str]) -> None:
    record_id = str(record.get("bridge_record_id") or "<missing>")
    missing = sorted(REQUIRED_FIELDS - set(record))
    if missing:
        errors.append(f"{record_id}: 缺少字段 {missing}")
        return
    if record["schema_version"] != "qpsalm_landslide_region_description_v1":
        errors.append(f"{record_id}: schema_version 非法")
    if record["split"] not in {"train", "val", "test"}:
        errors.append(f"{record_id}: split 非法")
    if record["region_source"] not in {
        "gt_global_mask", "pseudo_instance_component", "gt_referring_mask", "no_target"
    }:
        errors.append(f"{record_id}: region_source 非法")
    if record["target_status"] not in {"present", "absent", "uncertain"}:
        errors.append(f"{record_id}: target_status 非法")
    if not str(record["instruction"]).strip() or not str(record["condition"]).strip():
        errors.append(f"{record_id}: instruction/condition 为空")
    if record["answer_type"] != "structured_json_with_summary":
        errors.append(f"{record_id}: answer_type 非法")
    if record["candidate"].get("is_expert_truth") is not False:
        errors.append(f"{record_id}: 自动 candidate 不得标记为专家真值")
    if record["candidate"].get("origin") not in {"deterministic_rules", "optional_teacher"}:
        errors.append(f"{record_id}: candidate origin 非法")
    if not str(record["candidate"].get("summary") or "").strip():
        errors.append(f"{record_id}: candidate summary 为空")

    paths = _record_paths(record)
    if any(path.startswith("datasets/") for path in paths):
        errors.append(f"{record_id}: Bridge final record 不得引用原始 datasets")
    if any(not path.startswith("benchmark/") for path in paths):
        errors.append(f"{record_id}: 路径必须使用 benchmark 逻辑引用")
    modality_paths = record["visual_ref"].get("modality_paths") or {}
    if not modality_paths:
        errors.append(f"{record_id}: visual_ref 缺少活动模态")
    for path_ref in modality_paths.values():
        if not resolve_project_path(path_ref).is_file():
            errors.append(f"{record_id}: 模态文件不存在 {path_ref}")

    geometry = record["region_geometry"]
    mask_spec = record.get("region_mask")
    if record["target_status"] == "absent":
        if mask_spec is not None or int(geometry.get("area_pixels") or 0) != 0:
            errors.append(f"{record_id}: absent target 必须使用 null mask 和零面积")
    elif record["target_status"] == "present":
        if not isinstance(mask_spec, dict) or not mask_spec.get("path"):
            errors.append(f"{record_id}: present target 缺少 region mask")
        else:
            path_ref = str(mask_spec["path"])
            mask_refs.add(path_ref)
            try:
                mask = binary_mask(path_ref)
                if not bool(mask.any()):
                    errors.append(f"{record_id}: present region mask 为空")
                if list(mask.shape) != list(mask_spec.get("shape") or []):
                    errors.append(f"{record_id}: region mask shape 不一致")
                if mask_digest(mask) != mask_spec.get("sha256"):
                    errors.append(f"{record_id}: region mask digest 不一致")
                if int(mask.sum()) != int(mask_spec.get("positive_pixels") or -1):
                    errors.append(f"{record_id}: region mask positive_pixels 不一致")
            except Exception as exc:
                errors.append(f"{record_id}: region mask 无法读取: {exc}")

    for name, evidence in record.get("modality_evidence", {}).items():
        level = evidence.get("evidence_level")
        value_space = evidence.get("value_space")
        coverage = float(evidence.get("coverage") or 0.0)
        if level not in {"A_physical", "B_normalized_relative", "C_unavailable"}:
            errors.append(f"{record_id}/{name}: evidence_level 非法")
        if not 0.0 <= coverage <= 1.0:
            errors.append(f"{record_id}/{name}: coverage 越界")
        if level == "A_physical" and (value_space != "physical" or not evidence.get("units")):
            errors.append(f"{record_id}/{name}: Level A 必须有 physical value space 和单位")
        if level == "B_normalized_relative" and value_space != "normalized":
            errors.append(f"{record_id}/{name}: Level B 只能使用 normalized relative evidence")
        if level == "C_unavailable" and value_space != "unavailable":
            errors.append(f"{record_id}/{name}: Level C value_space 必须 unavailable")


def _validate_cross_stage(
    inventory: list[dict[str, Any]], facts: list[dict[str, Any]], candidates: list[dict[str, Any]],
    auto_train: list[dict[str, Any]], pilot: list[dict[str, Any]],
    selection: list[dict[str, Any]], package: list[dict[str, Any]],
    errors: list[str],
) -> None:
    def ids(rows: list[dict[str, Any]]) -> list[str]:
        return [str(row["bridge_record_id"]) for row in rows]

    for label, rows in (("inventory", inventory), ("facts", facts), ("candidates", candidates)):
        values = ids(rows)
        if len(values) != len(set(values)):
            errors.append(f"{label}: bridge_record_id 不唯一")
    if set(ids(inventory)) != set(ids(facts)) or set(ids(facts)) != set(ids(candidates)):
        errors.append("inventory/facts/candidate record 集合不一致")
    expected_train = {row["bridge_record_id"] for row in candidates if row["split"] == "train"}
    if {row["bridge_record_id"] for row in auto_train} != expected_train:
        errors.append("auto_train 必须恰好包含所有 train candidate")

    parent_splits: dict[str, set[str]] = defaultdict(set)
    for row in candidates:
        parent_splits[str(row["parent_sample_id"])].add(str(row["split"]))
    for parent_id, splits in parent_splits.items():
        if len(splits) != 1:
            errors.append(f"Bridge parent 跨 split: {parent_id} -> {sorted(splits)}")

    review_counts: dict[str, Counter[str]] = defaultdict(Counter)
    pilot_ids = [str(row.get("parent_sample_id") or "") for row in pilot]
    if any(not value for value in pilot_ids) or len(pilot_ids) != len(set(pilot_ids)):
        errors.append("pilot parent ID 缺失或重复")
    pilot_id_set = set(pilot_ids)
    selected_record_ids = {row["bridge_record_id"] for row in selection}
    candidate_by_id = {row["bridge_record_id"]: row for row in candidates}
    for row in selection:
        candidate = candidate_by_id.get(row["bridge_record_id"])
        if candidate is None:
            errors.append(f"review selection 引用未知 record: {row['bridge_record_id']}")
            continue
        if str(candidate["parent_sample_id"]) not in pilot_id_set:
            errors.append(
                f"review selection 引用了 Pilot 外 parent: {candidate['parent_sample_id']}"
            )
        status = str(candidate["target_status"])
        review_counts[str(candidate["parent_sample_id"])][status] += 1
    for parent_id, counts in review_counts.items():
        if counts["present"] > 1 or counts["absent"] > 1:
            errors.append(f"review parent 超过一正一负限制: {parent_id} -> {dict(counts)}")
    if set(review_counts) != pilot_id_set:
        errors.append(
            "每个 Pilot parent 必须至少有一个 review item: "
            f"pilot={len(pilot_id_set)} reviewed={len(review_counts)}"
        )
    if package and {row["bridge_record_id"] for row in package} != selected_record_ids:
        errors.append("review package manifest 与 review selection 不一致")
    if any(row.get("candidate_is_expert_truth") is not False for row in package):
        errors.append("review package 错误地包含专家真值标记")


def _validate_files(output_dir: Path, referenced_masks: set[str], errors: list[str]) -> None:
    part_files = list(output_dir.rglob("*.part")) if output_dir.exists() else []
    if part_files:
        errors.append(f"Bridge benchmark 存在未完成 .part 文件: {part_files[:3]}")
    materialized = {
        path.resolve(strict=False)
        for path in (output_dir / "data/regions").rglob("*.npy")
    } if (output_dir / "data/regions").exists() else set()
    referenced = {resolve_project_path(path).resolve(strict=False) for path in referenced_masks}
    stale = materialized - referenced
    missing = referenced - materialized
    if stale:
        errors.append(f"存在未登记 region mask: {sorted(str(path) for path in stale)[:3]}")
    if missing:
        errors.append(f"存在缺失 region mask: {sorted(str(path) for path in missing)[:3]}")


def _validate_expert(output_dir: Path, errors: list[str]) -> dict[str, Any]:
    required = [
        output_dir / "indexes/expert_all.jsonl",
        output_dir / "indexes/pending_arbitration.jsonl",
        output_dir / "reports/expert_review_report.json",
        output_dir / "manifests/evaluation_gate_manifest.json",
    ]
    missing = [path for path in required if not path.is_file()]
    if missing:
        errors.append(f"专家冻结产物缺失: {[str(path) for path in missing]}")
        return {"expert_records": 0, "pending_arbitration": None, "gate_frozen": False}
    expert = read_jsonl(required[0])
    pending = read_jsonl(required[1])
    review_report = read_json(required[2])
    gate = read_json(required[3])
    if gate.get("protocol") != "landslide_bridge_evaluation_gate_v2":
        errors.append("evaluation gate protocol 不是 landslide_bridge_evaluation_gate_v2")
    if pending:
        errors.append(f"仍有 {len(pending)} 条待仲裁记录")
    if gate.get("frozen") is not True or gate.get("status") != "frozen_after_pilot":
        errors.append("evaluation gate 尚未由用户冻结")
    binding_paths = {
        "pilot_parent_manifest_sha256": output_dir / "manifests/pilot_parent_manifest.jsonl",
        "review_selection_sha256": output_dir / "manifests/review_selection.jsonl",
        "candidate_index_sha256": output_dir / "indexes/candidate_all.jsonl",
    }
    missing_binding_paths = [str(path) for path in binding_paths.values() if not path.is_file()]
    expected_bindings = (
        {name: sha256_file(path) for name, path in binding_paths.items()}
        if not missing_binding_paths else {}
    )
    if missing_binding_paths:
        errors.append(f"evaluation gate 绑定文件缺失: {missing_binding_paths}")
    if gate.get("builder_version") != BUILDER_VERSION:
        errors.append("evaluation gate builder_version 与当前 Bridge 不一致")
    if gate.get("bindings") != expected_bindings:
        errors.append("evaluation gate 与当前 Pilot/candidate hash 不一致")
    thresholds = gate.get("thresholds") or {}
    for key in ("no_target_rejection", "unsupported_claim_rate", "expert_fact_score"):
        value = thresholds.get(key)
        if not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0:
            errors.append(f"evaluation gate 阈值非法: {key}={value!r}")
    if review_report.get("status") != "complete":
        errors.append("expert_review_report 尚未 complete")
    if not isinstance(review_report.get("field_agreement"), dict) or not review_report["field_agreement"]:
        errors.append("expert_review_report 缺少字段级一致性统计")
    if int(review_report.get("pending_arbitration", -1)) != len(pending):
        errors.append("expert_review_report pending 数与索引不一致")
    for split in ("train", "val", "test"):
        split_path = output_dir / f"indexes/expert_{split}.jsonl"
        if not split_path.is_file():
            errors.append(f"缺少 expert split index: {split_path}")
        elif not read_jsonl(split_path):
            errors.append(f"expert_{split}.jsonl 为空，Pilot 不能冻结该 split")
    for row in expert:
        if row.get("review", {}).get("status") not in {"accepted", "revised", "arbitrated"}:
            errors.append(f"expert record 审核状态非法: {row.get('bridge_record_id')}")
        if not row.get("expert_target", {}).get("summary"):
            errors.append(f"expert record 缺少 expert_target: {row.get('bridge_record_id')}")
        target_errors = validate_bridge_structured_target(
            row.get("expert_target", {}).get("structured_output"),
            expected_target_status=str(row.get("target_status")),
        )
        if target_errors:
            errors.append(
                f"expert record structured target 非法: {row.get('bridge_record_id')} -> {target_errors}"
            )
    return {
        "expert_records": len(expert),
        "pending_arbitration": len(pending),
        "gate_frozen": gate.get("frozen") is True,
    }


def main() -> None:
    args = parse_args()
    output_dir = bridge_dir(args.mode, args.output_dir)
    errors: list[str] = []
    warnings: list[str] = []
    config = _validate_schema_config(args.config, errors)

    required_paths = {
        "inventory": output_dir / "indexes/region_inventory.jsonl",
        "facts": output_dir / "indexes/region_facts_all.jsonl",
        "candidates": output_dir / "indexes/candidate_all.jsonl",
        "auto_train": output_dir / "indexes/auto_train.jsonl",
        "pilot": output_dir / "manifests/pilot_parent_manifest.jsonl",
        "selection": output_dir / "manifests/review_selection.jsonl",
        "package": output_dir / "manifests/review_package_manifest.jsonl",
    }
    for label, path in required_paths.items():
        if not path.is_file():
            errors.append(f"缺少 {label}: {path}")
    rows: dict[str, list[dict[str, Any]]] = {
        label: read_jsonl(path) if path.is_file() else [] for label, path in required_paths.items()
    }
    candidates_to_check = rows["candidates"][: args.max_samples or None]
    referenced_masks: set[str] = {
        str(row["region_mask"]["path"])
        for row in rows["candidates"] if row.get("region_mask")
    }
    for record in candidates_to_check:
        validate_record(record, errors, referenced_masks)
    _validate_cross_stage(
        rows["inventory"], rows["facts"], rows["candidates"], rows["auto_train"],
        rows["pilot"], rows["selection"], rows["package"], errors,
    )
    inventory_report_path = output_dir / "reports/region_inventory_report.json"
    inventory_report = (
        read_json(inventory_report_path) if inventory_report_path.is_file() else {}
    )
    if not inventory_report:
        errors.append("缺少 region_inventory_report.json")
    else:
        pilot_count = len(rows["pilot"])
        requested = int(inventory_report.get("pilot_requested_parents") or 0)
        observed_by_split = Counter(str(row.get("split")) for row in rows["pilot"])
        requested_quotas = {
            str(key): int(value)
            for key, value in (
                inventory_report.get("pilot_requested_split_quotas") or {}
            ).items()
        }
        counts_match = (
            pilot_count == requested
            and all(observed_by_split[split] == quota for split, quota in requested_quotas.items())
        )
        smoke_limit = int(inventory_report.get("source_parent_limit") or 0)
        protocol_complete = bool(inventory_report.get("pilot_protocol_complete"))
        if smoke_limit <= 0 and (not protocol_complete or not counts_match):
            errors.append(
                "正式 Pilot 未达到请求总数或 split 配额: "
                f"requested={requested} observed={pilot_count} "
                f"quotas={requested_quotas} observed_split={dict(observed_by_split)}"
            )
        elif smoke_limit > 0 and not protocol_complete:
            warnings.append(
                "source_parent_limit > 0：当前仅为 Bridge smoke Pilot，不可冻结科学评价 gate"
            )
        if args.require_expert_complete:
            configured_total = int(config.get("pilot", {}).get("parents") or 0)
            configured_quotas = {
                str(key): int(value)
                for key, value in (
                    config.get("pilot", {}).get("split_parent_quotas") or {}
                ).items()
            }
            if (
                not protocol_complete
                or requested != configured_total
                or requested_quotas != configured_quotas
            ):
                errors.append(
                    "专家 Pilot 只能在配置冻结的完整 parent/split 配额上完成: "
                    f"expected={configured_total}/{configured_quotas} "
                    f"observed={requested}/{requested_quotas}"
                )
    if args.max_samples <= 0:
        _validate_files(output_dir, referenced_masks, errors)
    else:
        warnings.append("max_samples > 0：跳过全量 stale mask 检查")

    expert_status = {"expert_records": 0, "pending_arbitration": None, "gate_frozen": False}
    if args.require_expert_complete:
        expert_status = _validate_expert(output_dir, errors)
    pilot_complete = bool(
        inventory_report.get("pilot_protocol_complete") if inventory_report else False
    )
    status = (
        "expert_pilot_frozen" if args.require_expert_complete and not errors else
        "awaiting_expert_review" if not args.require_expert_complete and not errors and pilot_complete else
        "smoke_only" if not args.require_expert_complete and not errors else
        "invalid"
    )
    report = {
        "builder_version": BUILDER_VERSION,
        "mode": args.mode,
        "status": status,
        "require_expert_complete": args.require_expert_complete,
        "records": len(rows["candidates"]),
        "parents": len({row["parent_sample_id"] for row in rows["candidates"]}),
        "pilot_parents": len(rows["pilot"]),
        "pilot_protocol_complete": pilot_complete,
        "review_items": len(rows["selection"]),
        "records_by_region_source": dict(sorted(Counter(
            row["region_source"] for row in rows["candidates"]
        ).items())),
        "expert": expert_status,
        "errors": errors,
        "warnings": warnings,
    }
    report_path = output_dir / "reports/validation_report.json"
    ensure_writable(report_path, args.overwrite, args.dry_run)
    print(
        f"[BRIDGE:VALIDATE] status={status} records={len(rows['candidates'])} "
        f"errors={len(errors)} warnings={len(warnings)}"
    )
    if not args.dry_run:
        write_json(report_path, report)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
