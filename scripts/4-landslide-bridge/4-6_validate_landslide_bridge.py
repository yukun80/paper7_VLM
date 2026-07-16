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
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from landslide_bridge_common import (
    BUILDER_VERSION,
    EXPERT_ARTIFACT_BINDING_PROTOCOL,
    EXPERT_REVIEW_REPLAY_PROTOCOL,
    REPO_ROOT,
    binary_mask,
    bridge_dir,
    ensure_writable,
    expert_review_report_statistics,
    file_artifact_binding,
    load_config,
    mask_digest,
    read_json,
    read_jsonl,
    read_review_file,
    replay_expert_review_merge,
    resolve_project_path,
    sha256_file,
    validate_bridge_structured_target,
    validate_file_artifact_binding,
    validate_frozen_evaluation_gate_science,
    unique_review_rows,
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
    structured_errors = validate_bridge_structured_target(
        record.get("structured_targets"),
        expected_target_status=str(record.get("target_status")),
    )
    if structured_errors:
        errors.append(
            f"{record_id}: structured_targets 非法 -> {structured_errors}"
        )
    candidate_errors = validate_bridge_structured_target(
        record.get("candidate", {}).get("structured_output"),
        expected_target_status=str(record.get("target_status")),
    )
    if candidate_errors:
        errors.append(
            f"{record_id}: candidate structured_output 非法 -> {candidate_errors}"
        )
    if (
        record.get("candidate", {}).get("origin") == "deterministic_rules"
        and record.get("candidate", {}).get("structured_output")
        != record.get("structured_targets")
    ):
        errors.append(
            f"{record_id}: deterministic candidate 必须逐字段复制 structured_targets"
        )

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


def _artifact_record_count(binding: Any, label: str, errors: list[str]) -> int | None:
    if not isinstance(binding, dict) or not isinstance(binding.get("path"), str):
        return None
    path = resolve_project_path(binding["path"])
    if not path.is_file():
        return None
    try:
        if path.suffix.casefold() == ".csv":
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                return sum(1 for _ in csv.DictReader(handle))
        return len(read_jsonl(path))
    except Exception as exc:
        errors.append(f"{label} artifact 无法读取记录数: {exc}")
        return None


def _validate_expert_artifact_binding(
    output_dir: Path,
    review_report: dict[str, Any],
    *,
    expert: list[dict[str, Any]],
    split_rows: dict[str, list[dict[str, Any]]],
    pending: list[dict[str, Any]],
    errors: list[str],
) -> dict[str, Any] | None:
    """Replay raw sources, output bytes, and the exact expert merge semantics."""
    binding = review_report.get("expert_artifact_binding")
    if not isinstance(binding, dict):
        errors.append("expert_review_report 缺少 expert_artifact_binding")
        return None
    if binding.get("protocol") != EXPERT_ARTIFACT_BINDING_PROTOCOL:
        errors.append("expert artifact binding protocol 过期或非法")
    if binding.get("builder_version") != BUILDER_VERSION:
        errors.append("expert artifact binding builder_version 与当前 Bridge 不一致")
    sources = binding.get("sources")
    outputs = binding.get("outputs")
    expected_source_keys = {
        "reviewer_1", "reviewer_2", "arbitration", "evaluation_gate_source",
    }
    expected_output_paths = {
        "expert_all": output_dir / "indexes/expert_all.jsonl",
        "expert_train": output_dir / "indexes/expert_train.jsonl",
        "expert_val": output_dir / "indexes/expert_val.jsonl",
        "expert_test": output_dir / "indexes/expert_test.jsonl",
        "pending_arbitration": output_dir / "indexes/pending_arbitration.jsonl",
        "evaluation_gate": output_dir / "manifests/evaluation_gate_manifest.json",
    }
    if not isinstance(sources, dict) or set(sources) != expected_source_keys:
        errors.append("expert artifact binding sources 集合不完整")
        sources = sources if isinstance(sources, dict) else {}
    if not isinstance(outputs, dict) or set(outputs) != set(expected_output_paths):
        errors.append("expert artifact binding outputs 集合不完整")
        outputs = outputs if isinstance(outputs, dict) else {}

    for name in ("reviewer_1", "reviewer_2"):
        source_binding = sources.get(name)
        count = _artifact_record_count(source_binding, name, errors)
        errors.extend(validate_file_artifact_binding(
            source_binding,
            label=name,
            expected_records=count,
        ))
        if count != int(review_report.get("review_items", -1)):
            errors.append(
                f"{name} 记录数必须等于 review_items: "
                f"expected={review_report.get('review_items')} observed={count}"
            )
    arbitration_binding = sources.get("arbitration")
    if arbitration_binding is not None:
        count = _artifact_record_count(arbitration_binding, "arbitration", errors)
        errors.extend(validate_file_artifact_binding(
            arbitration_binding,
            label="arbitration",
            expected_records=count,
        ))
    gate_source_binding = sources.get("evaluation_gate_source")
    errors.extend(validate_file_artifact_binding(
        gate_source_binding,
        label="evaluation_gate_source",
    ))
    if isinstance(gate_source_binding, dict):
        gate_source_path = resolve_project_path(
            str(gate_source_binding.get("path") or "")
        )
        gate_output_path = expected_output_paths["evaluation_gate"]
        if gate_source_path.is_file() and gate_output_path.is_file():
            source_gate = read_json(gate_source_path)
            expected_gate = dict(source_gate)
            expected_gate["source_file"] = gate_source_binding.get("path")
            if read_json(gate_output_path) != expected_gate:
                errors.append(
                    "published evaluation gate 不是 frozen gate source 的精确带来源副本"
                )

    expected_counts = {
        "expert_all": len(expert),
        "expert_train": len(split_rows.get("train", [])),
        "expert_val": len(split_rows.get("val", [])),
        "expert_test": len(split_rows.get("test", [])),
        "pending_arbitration": len(pending),
    }
    for name, path in expected_output_paths.items():
        errors.extend(validate_file_artifact_binding(
            outputs.get(name),
            label=name,
            expected_path=path,
            expected_records=expected_counts.get(name),
        ))

    semantic_replay: dict[str, Any] | None = None
    try:
        reviewer_paths = {
            name: resolve_project_path(str(sources[name]["path"]))
            for name in ("reviewer_1", "reviewer_2")
            if isinstance(sources.get(name), dict)
        }
        if set(reviewer_paths) != {"reviewer_1", "reviewer_2"}:
            raise ValueError("双审 source binding 不完整")
        if reviewer_paths["reviewer_1"] == reviewer_paths["reviewer_2"]:
            raise ValueError("两名独立 reviewer 不得绑定同一个物理文件")
        source_paths = set(reviewer_paths.values())
        for name in ("arbitration", "evaluation_gate_source"):
            source_binding = sources.get(name)
            if isinstance(source_binding, dict):
                source_paths.add(
                    resolve_project_path(str(source_binding.get("path") or ""))
                )
        output_paths = {
            resolve_project_path(str(value.get("path") or ""))
            for value in outputs.values()
            if isinstance(value, dict)
        }
        collisions = source_paths & output_paths
        if collisions:
            raise ValueError(
                "人工 review/gate source 与派生输出路径冲突: "
                f"{sorted(str(path) for path in collisions)}"
            )

        left = unique_review_rows(
            read_review_file(reviewer_paths["reviewer_1"], "reviewer_1"),
            "reviewer_1",
        )
        right = unique_review_rows(
            read_review_file(reviewer_paths["reviewer_2"], "reviewer_2"),
            "reviewer_2",
        )
        arbitration_binding = sources.get("arbitration")
        arbitration = (
            unique_review_rows(
                read_review_file(str(arbitration_binding["path"])),
                "arbitration",
            )
            if isinstance(arbitration_binding, dict)
            else {}
        )
        candidate_path = output_dir / "indexes/candidate_all.jsonl"
        selection_path = output_dir / "manifests/review_selection.jsonl"
        replay = replay_expert_review_merge(
            candidates=read_jsonl(candidate_path),
            selection=read_jsonl(selection_path),
            reviewer_1=left,
            reviewer_2=right,
            arbitration=arbitration,
        )
        if replay["expert"] != expert:
            errors.append(
                "expert_all 不是 candidate/selection/双审/仲裁源的精确语义重放结果"
            )
        if replay["pending"] != pending:
            errors.append(
                "pending_arbitration 不是双审分歧与仲裁源的精确语义重放结果"
            )
        if review_report.get("final_decisions") != replay["final_decisions"]:
            errors.append(
                "expert_review_report final_decisions 与语义重放结果不一致"
            )
        replay_statistics = expert_review_report_statistics(
            candidates=read_jsonl(candidate_path),
            selection=read_jsonl(selection_path),
            reviewer_1=left,
            reviewer_2=right,
            replay=replay,
        )
        statistics_mismatches = [
            field
            for field, expected in replay_statistics.items()
            if review_report.get(field) != expected
        ]
        if statistics_mismatches:
            errors.append(
                "expert_review_report 审核统计不是 raw review/expert 的精确重算结果: "
                f"{statistics_mismatches}"
            )
        semantic_replay = {
            "protocol": EXPERT_REVIEW_REPLAY_PROTOCOL,
            "candidate_index": file_artifact_binding(
                candidate_path,
                records=len(read_jsonl(candidate_path)),
            ),
            "review_selection": file_artifact_binding(
                selection_path,
                records=len(read_jsonl(selection_path)),
            ),
            "review_items": len(left),
            "disputed_review_items": len(replay["disputed_review_item_ids"]),
            "expert_records": len(replay["expert"]),
            "pending_arbitration": len(replay["pending"]),
            "final_decisions": replay["final_decisions"],
            "review_report_statistics_verified": not statistics_mismatches,
        }
    except Exception as exc:
        errors.append(f"expert review semantic replay 失败: {exc}")
    return {
        "protocol": EXPERT_ARTIFACT_BINDING_PROTOCOL,
        "builder_version": BUILDER_VERSION,
        "review_report": file_artifact_binding(
            output_dir / "reports/expert_review_report.json"
        ),
        "merge_artifacts": binding,
        "semantic_replay": semantic_replay,
    }


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
    errors.extend(validate_frozen_evaluation_gate_science(gate))
    if pending:
        errors.append(f"仍有 {len(pending)} 条待仲裁记录")
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
    if review_report.get("status") != "complete":
        errors.append("expert_review_report 尚未 complete")
    if review_report.get("builder_version") != BUILDER_VERSION:
        errors.append("expert_review_report builder_version 与当前 Bridge 不一致")
    if review_report.get("frozen_evaluation_gate") is not True:
        errors.append("expert_review_report 未绑定 frozen evaluation gate")
    if not isinstance(review_report.get("field_agreement"), dict) or not review_report["field_agreement"]:
        errors.append("expert_review_report 缺少字段级一致性统计")
    if not isinstance(review_report.get("disputed_field_counts"), dict):
        errors.append("expert_review_report 缺少 disputed field 统计")
    disagreement = review_report.get("disagreement_distribution")
    if not isinstance(disagreement, dict) or set(disagreement) != {
        "region_source", "modality_family_combo", "evidence_level",
    }:
        errors.append("expert_review_report 缺少争议证据/模态分布")
    for name in ("acceptance_rate", "modification_rate", "rejection_rate"):
        value = review_report.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0.0 <= float(value) <= 1.0
        ):
            errors.append(f"expert_review_report {name} 非法")
    modifications = review_report.get("expert_modification_statistics")
    required_modification_fields = {
        "summary_mean_edit_distance_characters",
        "summary_mean_normalized_edit_distance",
        "factual_claim_modification_rate",
        "structured_claim_fields_changed",
        "structured_claim_fields_total",
    }
    if not isinstance(modifications, dict) or not required_modification_fields.issubset(
        modifications
    ):
        errors.append("expert_review_report 缺少 summary/claim 修改统计")
    if int(review_report.get("pending_arbitration", -1)) != len(pending):
        errors.append("expert_review_report pending 数与索引不一致")
    review_selection_path = output_dir / "manifests/review_selection.jsonl"
    selection_count = (
        len(read_jsonl(review_selection_path))
        if review_selection_path.is_file() else -1
    )
    if int(review_report.get("review_items", -1)) != selection_count:
        errors.append(
            "expert_review_report 必须精确覆盖完整 review_selection: "
            f"expected={selection_count} observed={review_report.get('review_items')!r}"
        )
    final_decisions = review_report.get("final_decisions")
    try:
        final_decision_count = (
            sum(int(value) for value in final_decisions.values())
            if isinstance(final_decisions, dict) else -1
        )
    except (TypeError, ValueError):
        final_decision_count = -1
    if final_decision_count != selection_count:
        errors.append("expert_review_report final_decisions 未精确覆盖 review_selection")
    split_rows: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "val", "test"):
        split_path = output_dir / f"indexes/expert_{split}.jsonl"
        if not split_path.is_file():
            errors.append(f"缺少 expert split index: {split_path}")
            split_rows[split] = []
        else:
            split_rows[split] = read_jsonl(split_path)
            if not split_rows[split]:
                errors.append(f"expert_{split}.jsonl 为空，Pilot 不能冻结该 split")
            expected_split_rows = [row for row in expert if row.get("split") == split]
            if split_rows[split] != expected_split_rows:
                errors.append(
                    f"expert_{split}.jsonl 与 expert_all 的精确 split 投影不一致"
                )
    expert_ids = [str(row.get("bridge_record_id") or "") for row in expert]
    if not all(expert_ids) or len(expert_ids) != len(set(expert_ids)):
        errors.append("expert_all bridge_record_id 缺失或重复")
    if int(review_report.get("expert_records", -1)) != len(expert):
        errors.append("expert_review_report expert_records 与 expert_all 不一致")
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
    artifact_binding = _validate_expert_artifact_binding(
        output_dir,
        review_report,
        expert=expert,
        split_rows=split_rows,
        pending=pending,
        errors=errors,
    )
    return {
        "expert_records": len(expert),
        "pending_arbitration": len(pending),
        "gate_frozen": gate.get("frozen") is True,
        "artifact_binding": artifact_binding,
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
    expert_artifact_binding = expert_status.pop("artifact_binding", None)
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
        "expert_artifact_binding": expert_artifact_binding,
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
