#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""步骤 4-5：合并双人专家审核与显式仲裁结果。

用途：只消费人工完成的 reviewer 文件；分歧项没有仲裁时保持 pending，不生成专家真值。
推荐运行命令：python scripts/4-landslide-bridge/4-5_merge_expert_reviews.py --mode small --reviewer-1 <完成文件> --reviewer-2 <完成文件> --overwrite
主要输入：两份 completed review JSONL/CSV，可选 arbitration 和人工冻结 gate JSON。
主要输出：expert_all/split indexes、pending_arbitration、agreement report。
写入行为：只写 Bridge 派生结果，不覆盖 review 原文件；--dry-run 不写文件。
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from landslide_bridge_common import (
    BUILDER_VERSION,
    bridge_dir,
    cohen_kappa,
    ensure_writable,
    flatten_bridge_structured_target,
    krippendorff_alpha_nominal,
    read_json,
    read_jsonl,
    resolve_project_path,
    sha256_file,
    to_project_ref,
    validate_bridge_structured_target,
    write_json,
    write_jsonl,
)


DECISIONS = {"accept", "revise", "reject"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="合并 Landslide Bridge 专家审核")
    parser.add_argument("--mode", choices=["small", "full"], default="small")
    parser.add_argument("--output-dir")
    parser.add_argument("--config", default="configs/landslide_bridge_v1.yaml")
    parser.add_argument("--reviewer-1", required=True)
    parser.add_argument("--reviewer-2", required=True)
    parser.add_argument("--arbitration-file")
    parser.add_argument("--evaluation-gate")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _parse_json_field(value: Any, field: str, path: Path, row_number: int) -> Any:
    if value in (None, ""):
        return None
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}:{row_number}: {field} 不是合法 JSON") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{path}:{row_number}: {field} 必须是 JSON object")
    return parsed


def read_review_file(path_ref: str, expected_reviewer: str | None = None) -> list[dict[str, Any]]:
    path = resolve_project_path(path_ref)
    if not path.is_file():
        raise FileNotFoundError(f"审核文件不存在: {path_ref} -> {path}")
    if path.suffix.casefold() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    else:
        rows = read_jsonl(path)
    normalized: list[dict[str, Any]] = []
    for row_number, source in enumerate(rows, start=2 if path.suffix.casefold() == ".csv" else 1):
        row = dict(source)
        decision = str(row.get("decision") or "").strip().casefold()
        if decision not in DECISIONS:
            raise ValueError(f"{path}:{row_number}: decision 必须为 accept/revise/reject，当前={decision!r}")
        reviewer_id = str(row.get("reviewer_id") or expected_reviewer or "").strip()
        if expected_reviewer and reviewer_id != expected_reviewer:
            raise ValueError(f"{path}:{row_number}: reviewer_id 应为 {expected_reviewer}")
        row["reviewer_id"] = reviewer_id
        row["decision"] = decision
        row["corrected_structured_targets"] = _parse_json_field(
            row.get("corrected_structured_targets"), "corrected_structured_targets", path, row_number
        )
        row["revised_summary"] = str(row.get("revised_summary") or "").strip()
        if decision == "revise" and (
            row["corrected_structured_targets"] is None or not row["revised_summary"]
        ):
            raise ValueError(f"{path}:{row_number}: revise 必须填写 corrected_structured_targets 和 revised_summary")
        normalized.append(row)
    return normalized


def _unique_by_item(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        item_id = str(row.get("review_item_id") or "")
        if not item_id:
            raise ValueError(f"{label}: review_item_id 不能为空")
        if item_id in result:
            raise ValueError(f"{label}: review_item_id 重复: {item_id}")
        result[item_id] = row
    return result


def _same_revision(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return (
        left["corrected_structured_targets"] == right["corrected_structured_targets"]
        and left["revised_summary"] == right["revised_summary"]
    )


def _resolved_target(
    record: dict[str, Any], decision: str, response: dict[str, Any], status: str,
    reviewer_responses: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if decision == "reject":
        return None
    if decision == "accept":
        structured = copy.deepcopy(record["candidate"]["structured_output"])
        summary = str(record["candidate"]["summary"])
    else:
        structured = copy.deepcopy(response["corrected_structured_targets"])
        summary = str(response["revised_summary"])
    target_errors = validate_bridge_structured_target(
        structured, expected_target_status=str(record["target_status"])
    )
    if target_errors:
        raise ValueError(
            f"{record['bridge_record_id']}: expert structured target 非法: {target_errors}"
        )
    if not summary.strip():
        raise ValueError(f"{record['bridge_record_id']}: expert summary 不能为空")
    result = copy.deepcopy(record)
    result["expert_target"] = {
        "structured_output": structured,
        "summary": summary,
        "language": "en",
        "source": "double_reviewed_pilot",
    }
    result["review"] = {
        "status": status,
        "final_decision": decision,
        "reviewer_responses": copy.deepcopy(reviewer_responses),
    }
    result["provenance"]["expert_review_merger"] = BUILDER_VERSION
    return result


def _load_frozen_gate(
    path_ref: str | None,
    output_dir: Path,
) -> dict[str, Any] | None:
    if not path_ref:
        return None
    path = resolve_project_path(path_ref)
    gate = read_json(path)
    if gate.get("protocol") != "landslide_bridge_evaluation_gate_v2":
        raise ValueError(f"evaluation gate protocol 不正确: {path}")
    if gate.get("frozen") is not True or gate.get("status") != "frozen_after_pilot":
        raise ValueError("evaluation gate 必须由用户显式设为 frozen_after_pilot 且 frozen=true")
    thresholds = gate.get("thresholds") or {}
    required = {"no_target_rejection", "unsupported_claim_rate", "expert_fact_score"}
    if any(thresholds.get(key) is None for key in required):
        raise ValueError(f"evaluation gate 缺少冻结阈值: {sorted(required)}")
    for key in required:
        value = thresholds[key]
        if not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"evaluation gate 阈值必须位于 [0,1]: {key}={value!r}")
    if gate.get("builder_version") != BUILDER_VERSION:
        raise ValueError(
            "evaluation gate builder_version 与当前 Bridge 不一致；"
            "必须从本轮 review package 模板冻结"
        )
    expected_bindings = {
        "pilot_parent_manifest_sha256": sha256_file(
            output_dir / "manifests/pilot_parent_manifest.jsonl"
        ),
        "review_selection_sha256": sha256_file(
            output_dir / "manifests/review_selection.jsonl"
        ),
        "candidate_index_sha256": sha256_file(
            output_dir / "indexes/candidate_all.jsonl"
        ),
    }
    if gate.get("bindings") != expected_bindings:
        raise ValueError(
            "evaluation gate 与当前 Pilot/candidate 不匹配；"
            f"expected={expected_bindings} observed={gate.get('bindings')}"
        )
    result = copy.deepcopy(gate)
    result["source_file"] = to_project_ref(path)
    return result


def main() -> None:
    args = parse_args()
    output_dir = bridge_dir(args.mode, args.output_dir)
    candidates = {row["bridge_record_id"]: row for row in read_jsonl(output_dir / "indexes/candidate_all.jsonl")}
    selection = read_jsonl(output_dir / "manifests/review_selection.jsonl")
    if args.max_samples > 0:
        selection = selection[:args.max_samples]
    selection_by_item = {row["review_item_id"]: row for row in selection}
    left = _unique_by_item(read_review_file(args.reviewer_1, "reviewer_1"), "reviewer_1")
    right = _unique_by_item(read_review_file(args.reviewer_2, "reviewer_2"), "reviewer_2")
    selected_ids = set(selection_by_item)
    if set(left) != selected_ids or set(right) != selected_ids:
        raise ValueError(
            "两份 reviewer 文件必须恰好覆盖 review_selection；"
            f"selection={len(selected_ids)} reviewer_1={len(left)} reviewer_2={len(right)}"
        )
    arbitration = (
        _unique_by_item(read_review_file(args.arbitration_file), "arbitration")
        if args.arbitration_file else {}
    )
    unexpected_arbitration = set(arbitration) - selected_ids
    if unexpected_arbitration:
        raise ValueError(f"arbitration 包含未知 review item: {sorted(unexpected_arbitration)[:3]}")
    frozen_gate = _load_frozen_gate(args.evaluation_gate, output_dir)

    expert: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    final_decisions: Counter[str] = Counter()
    for item_id in sorted(selected_ids):
        item = selection_by_item[item_id]
        record = candidates[item["bridge_record_id"]]
        first, second = left[item_id], right[item_id]
        responses = [first, second]
        same = first["decision"] == second["decision"]
        if same and first["decision"] == "revise":
            same = _same_revision(first, second)
        if same:
            decision, response = first["decision"], first
            status = {"accept": "accepted", "revise": "revised", "reject": "rejected"}[decision]
        elif item_id in arbitration:
            response = arbitration[item_id]
            decision = response["decision"]
            responses.append(response)
            status = "arbitrated"
        else:
            pending.append({
                **item,
                "status": "needs_arbitration",
                "reviewer_responses": responses,
            })
            continue
        final_decisions[decision] += 1
        resolved = _resolved_target(record, decision, response, status, responses)
        if resolved is not None:
            expert.append(resolved)

    expert.sort(key=lambda row: (row["split"], row["parent_sample_id"], row["bridge_record_id"]))
    pending.sort(key=lambda row: (row["split"], row["parent_sample_id"], row["review_item_id"]))
    output_paths = [
        output_dir / "indexes/expert_all.jsonl",
        output_dir / "indexes/pending_arbitration.jsonl",
        output_dir / "reports/expert_review_report.json",
    ] + [output_dir / f"indexes/expert_{split}.jsonl" for split in ("train", "val", "test")]
    if frozen_gate is not None:
        output_paths.append(output_dir / "manifests/evaluation_gate_manifest.json")
    for path in output_paths:
        ensure_writable(path, args.overwrite, args.dry_run)

    left_decisions = [left[item_id]["decision"] for item_id in sorted(selected_ids)]
    right_decisions = [right[item_id]["decision"] for item_id in sorted(selected_ids)]
    field_ratings: dict[str, list[list[str]]] = {}
    for item_id in sorted(selected_ids):
        record = candidates[selection_by_item[item_id]["bridge_record_id"]]
        reviewer_fields = []
        for response in (left[item_id], right[item_id]):
            if response["decision"] == "reject":
                reviewer_fields.append({"review_decision": "reject"})
                continue
            structured = (
                record["candidate"]["structured_output"]
                if response["decision"] == "accept"
                else response["corrected_structured_targets"]
            )
            target_errors = validate_bridge_structured_target(
                structured, expected_target_status=str(record["target_status"])
            )
            if target_errors:
                raise ValueError(
                    f"{item_id}/{response['reviewer_id']}: structured target 非法: {target_errors}"
                )
            reviewer_fields.append(flatten_bridge_structured_target(structured))
        field_names = sorted(set(reviewer_fields[0]) | set(reviewer_fields[1]))
        for field in field_names:
            field_ratings.setdefault(field, []).append([
                reviewer_fields[0].get(field, "<rejected>"),
                reviewer_fields[1].get(field, "<rejected>"),
            ])
    field_agreement = {
        field: {
            "cohen_kappa": cohen_kappa(
                [pair[0] for pair in ratings], [pair[1] for pair in ratings]
            ),
            "krippendorff_alpha_nominal": krippendorff_alpha_nominal(ratings),
            "exact_agreement": sum(pair[0] == pair[1] for pair in ratings) / max(len(ratings), 1),
            "num_items": len(ratings),
        }
        for field, ratings in sorted(field_ratings.items())
    }
    resolved_count = sum(final_decisions.values())
    report = {
        "builder_version": BUILDER_VERSION,
        "review_items": len(selected_ids),
        "expert_records": len(expert),
        "pending_arbitration": len(pending),
        "final_decisions": dict(sorted(final_decisions.items())),
        "cohen_kappa_decision": cohen_kappa(left_decisions, right_decisions),
        "krippendorff_alpha_decision": krippendorff_alpha_nominal([
            [left[item_id]["decision"], right[item_id]["decision"]]
            for item_id in sorted(selected_ids)
        ]),
        "initial_decision_agreement_rate": sum(
            left[item_id]["decision"] == right[item_id]["decision"]
            for item_id in selected_ids
        ) / max(len(selected_ids), 1),
        "field_agreement": field_agreement,
        "reviewer_decision_counts": {
            "reviewer_1": dict(sorted(Counter(left_decisions).items())),
            "reviewer_2": dict(sorted(Counter(right_decisions).items())),
        },
        "modification_rate": final_decisions["revise"] / max(resolved_count, 1),
        "rejection_rate": final_decisions["reject"] / max(resolved_count, 1),
        "accepted_or_revised_rate": len(expert) / max(len(selected_ids), 1),
        "frozen_evaluation_gate": frozen_gate is not None,
        "status": "complete" if not pending and frozen_gate is not None else "awaiting_arbitration_or_gate",
        "errors": [],
    }
    print(
        f"[BRIDGE:MERGE] review_items={len(selected_ids)} expert={len(expert)} "
        f"pending={len(pending)} gate={'frozen' if frozen_gate else 'pending'}"
    )
    if not args.dry_run:
        write_jsonl(output_dir / "indexes/expert_all.jsonl", expert)
        write_jsonl(output_dir / "indexes/pending_arbitration.jsonl", pending)
        for split in ("train", "val", "test"):
            write_jsonl(output_dir / f"indexes/expert_{split}.jsonl", [row for row in expert if row["split"] == split])
        if frozen_gate is not None:
            write_json(output_dir / "manifests/evaluation_gate_manifest.json", frozen_gate)
        write_json(output_dir / "reports/expert_review_report.json", report)


if __name__ == "__main__":
    main()
