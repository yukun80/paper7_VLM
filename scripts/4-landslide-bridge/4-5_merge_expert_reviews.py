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
from pathlib import Path
from typing import Any

from landslide_bridge_common import (
    BUILDER_VERSION,
    EXPERT_ARTIFACT_BINDING_PROTOCOL,
    bridge_dir,
    ensure_writable,
    expert_review_report_statistics,
    file_artifact_binding,
    read_json,
    read_jsonl,
    read_review_file,
    replay_expert_review_merge,
    resolve_project_path,
    sha256_file,
    to_project_ref,
    validate_frozen_evaluation_gate_science,
    unique_review_rows as _unique_by_item,
    write_json,
    write_jsonl,
)


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

def _load_frozen_gate(
    path_ref: str | None,
    output_dir: Path,
) -> dict[str, Any] | None:
    if not path_ref:
        return None
    path = resolve_project_path(path_ref)
    gate = read_json(path)
    scientific_errors = validate_frozen_evaluation_gate_science(gate)
    if scientific_errors:
        raise ValueError("evaluation gate 科学协议非法: " + "; ".join(scientific_errors))
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
    if args.max_samples > 0 and args.evaluation_gate:
        raise ValueError(
            "人工 frozen gate 不允许 --max-samples；必须合并完整 review_selection"
        )
    output_dir = bridge_dir(args.mode, args.output_dir)
    output_paths = {
        "expert_all": output_dir / "indexes/expert_all.jsonl",
        "expert_train": output_dir / "indexes/expert_train.jsonl",
        "expert_val": output_dir / "indexes/expert_val.jsonl",
        "expert_test": output_dir / "indexes/expert_test.jsonl",
        "pending_arbitration": output_dir / "indexes/pending_arbitration.jsonl",
        "evaluation_gate": output_dir / "manifests/evaluation_gate_manifest.json",
        "review_report": output_dir / "reports/expert_review_report.json",
    }
    reviewer_paths = {
        "reviewer_1": resolve_project_path(args.reviewer_1),
        "reviewer_2": resolve_project_path(args.reviewer_2),
    }
    if reviewer_paths["reviewer_1"] == reviewer_paths["reviewer_2"]:
        raise ValueError("两名独立 reviewer 不得使用同一个物理文件")
    source_paths = dict(reviewer_paths)
    if args.arbitration_file:
        source_paths["arbitration"] = resolve_project_path(args.arbitration_file)
    if args.evaluation_gate:
        source_paths["evaluation_gate"] = resolve_project_path(args.evaluation_gate)
    output_physical_paths = {path.resolve(strict=False) for path in output_paths.values()}
    colliding_sources = {
        name: str(path) for name, path in source_paths.items()
        if path.resolve(strict=False) in output_physical_paths
    }
    if colliding_sources:
        raise ValueError(
            "人工输入文件不得与 Bridge 派生输出共用路径: "
            f"{colliding_sources}"
        )
    candidate_rows = read_jsonl(output_dir / "indexes/candidate_all.jsonl")
    candidates: dict[str, dict[str, Any]] = {}
    for row in candidate_rows:
        record_id = str(row.get("bridge_record_id") or "").strip()
        if not record_id or record_id in candidates:
            raise ValueError(
                f"candidate bridge_record_id 缺失或重复: {record_id!r}"
            )
        candidates[record_id] = row
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
    replay = replay_expert_review_merge(
        candidates=candidate_rows,
        selection=selection,
        reviewer_1=left,
        reviewer_2=right,
        arbitration=arbitration,
    )
    frozen_gate = _load_frozen_gate(args.evaluation_gate, output_dir)
    source_artifacts = {
        "reviewer_1": file_artifact_binding(
            reviewer_paths["reviewer_1"], records=len(left)
        ),
        "reviewer_2": file_artifact_binding(
            reviewer_paths["reviewer_2"], records=len(right)
        ),
        "arbitration": (
            file_artifact_binding(source_paths["arbitration"], records=len(arbitration))
            if "arbitration" in source_paths else None
        ),
        "evaluation_gate_source": (
            file_artifact_binding(source_paths["evaluation_gate"])
            if "evaluation_gate" in source_paths else None
        ),
    }

    expert = replay["expert"]
    pending = replay["pending"]
    report_statistics = expert_review_report_statistics(
        candidates=candidate_rows,
        selection=selection,
        reviewer_1=left,
        reviewer_2=right,
        replay=replay,
    )
    paths_to_write = [
        output_paths["expert_all"],
        output_paths["pending_arbitration"],
        output_paths["review_report"],
        output_paths["expert_train"],
        output_paths["expert_val"],
        output_paths["expert_test"],
    ]
    if frozen_gate is not None:
        paths_to_write.append(output_paths["evaluation_gate"])
    for path in paths_to_write:
        ensure_writable(path, args.overwrite, args.dry_run)

    report = {
        "builder_version": BUILDER_VERSION,
        **report_statistics,
        "frozen_evaluation_gate": frozen_gate is not None,
        "status": "complete" if not pending and frozen_gate is not None else "awaiting_arbitration_or_gate",
        "errors": [],
        "expert_artifact_binding": None,
    }
    print(
        f"[BRIDGE:MERGE] review_items={len(selected_ids)} expert={len(expert)} "
        f"pending={len(pending)} gate={'frozen' if frozen_gate else 'pending'}"
    )
    if not args.dry_run:
        write_jsonl(output_paths["expert_all"], expert)
        write_jsonl(output_paths["pending_arbitration"], pending)
        split_rows: dict[str, list[dict[str, Any]]] = {}
        for split in ("train", "val", "test"):
            split_rows[split] = [row for row in expert if row["split"] == split]
            write_jsonl(output_paths[f"expert_{split}"], split_rows[split])
        if frozen_gate is not None:
            write_json(output_paths["evaluation_gate"], frozen_gate)
        report["expert_artifact_binding"] = {
            "protocol": EXPERT_ARTIFACT_BINDING_PROTOCOL,
            "builder_version": BUILDER_VERSION,
            "sources": source_artifacts,
            "outputs": {
                "expert_all": file_artifact_binding(
                    output_paths["expert_all"], records=len(expert)
                ),
                "expert_train": file_artifact_binding(
                    output_paths["expert_train"], records=len(split_rows["train"])
                ),
                "expert_val": file_artifact_binding(
                    output_paths["expert_val"], records=len(split_rows["val"])
                ),
                "expert_test": file_artifact_binding(
                    output_paths["expert_test"], records=len(split_rows["test"])
                ),
                "pending_arbitration": file_artifact_binding(
                    output_paths["pending_arbitration"], records=len(pending)
                ),
                "evaluation_gate": (
                    file_artifact_binding(output_paths["evaluation_gate"])
                    if frozen_gate is not None else None
                ),
            },
        }
        # 报告自身由后续 validation report 绑定，因此必须最后原子写入。
        write_json(output_paths["review_report"], report)


if __name__ == "__main__":
    main()
