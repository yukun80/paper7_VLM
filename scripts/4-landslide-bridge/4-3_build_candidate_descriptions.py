#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""步骤 4-3：从协议约束事实生成可审核的英文候选描述。

用途：将确定性 geometry/evidence 组织为 review candidate；候选文本绝不标为专家真值。
推荐运行命令：python scripts/4-landslide-bridge/4-3_build_candidate_descriptions.py --mode small --overwrite
主要输入：4-2 的 indexes/region_facts_all.jsonl。
主要输出：indexes/candidate_all.jsonl、indexes/auto_train.jsonl 和候选报告。
写入行为：只写 Bridge 派生索引；--dry-run 不写文件。
"""

from __future__ import annotations

import argparse
import copy
from collections import Counter
from typing import Any

from landslide_bridge_common import (
    BUILDER_VERSION,
    bridge_dir,
    ensure_writable,
    read_jsonl,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建 Landslide Bridge 候选描述")
    parser.add_argument("--mode", choices=["small", "full"], default="small")
    parser.add_argument("--output-dir")
    parser.add_argument("--config", default="configs/landslide_bridge_v1.yaml")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _words(value: str) -> str:
    return str(value).replace("_", " ")


def _present_summary(record: dict[str, Any]) -> str:
    region = record["structured_targets"]["region"]
    geometry = record["region_geometry"]
    sentences = [
        (
            f"The specified landslide region is {_words(region['size_class'])}, "
            f"{_words(region['shape'])}, and located in the {_words(region['location'])} "
            "part of the valid image area."
        )
    ]
    if region.get("fragmentation") not in {None, "unavailable"}:
        sentences.append(f"Its mask geometry is {_words(region['fragmentation'])}.")
    observations: list[str] = []
    for name, evidence in sorted(record.get("modality_evidence", {}).items()):
        if evidence.get("evidence_level") == "C_unavailable":
            continue
        observation = str(evidence.get("observation") or "").strip()
        if observation:
            family = _words(str(evidence.get("family") or name))
            observations.append(f"{family.capitalize()} evidence: {observation}")
    if observations:
        sentences.extend(observations)
        sentences.append(
            "These observations describe available regional evidence but do not by themselves "
            "establish a causal or activity interpretation."
        )
    else:
        sentences.append("The available modalities do not provide sufficient regional evidence for further interpretation.")
    area_ratio = float(geometry.get("valid_area_ratio") or 0.0)
    sentences.append(f"The region covers {area_ratio:.4%} of the valid image area.")
    return " ".join(sentences)


def build_candidate(record: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(record)
    if record["target_status"] == "absent":
        summary = (
            "The specified landslide target is absent. No region mask is available for regional "
            "geometry or multisource evidence extraction."
        )
    elif record["target_status"] == "present":
        summary = _present_summary(record)
    else:
        summary = "The target status is uncertain, so no definitive regional description is provided."
    result["candidate"] = {
        "origin": "deterministic_rules",
        "summary": summary,
        "structured_output": copy.deepcopy(record["structured_targets"]),
        "is_expert_truth": False,
        "language": "en",
        "protocol": "landslide_bridge_rule_candidate_v1",
    }
    result["provenance"]["candidate_builder"] = BUILDER_VERSION
    return result


def main() -> None:
    args = parse_args()
    output_dir = bridge_dir(args.mode, args.output_dir)
    facts = read_jsonl(output_dir / "indexes/region_facts_all.jsonl")
    if args.max_samples > 0:
        facts = facts[:args.max_samples]
    candidates = [build_candidate(record) for record in facts]
    candidates.sort(key=lambda row: (row["split"], row["parent_sample_id"], row["region_id"]))
    auto_train = [row for row in candidates if row["split"] == "train"]

    output_path = output_dir / "indexes/candidate_all.jsonl"
    train_path = output_dir / "indexes/auto_train.jsonl"
    report_path = output_dir / "reports/candidate_description_report.json"
    for path in (output_path, train_path, report_path):
        ensure_writable(path, args.overwrite, args.dry_run)

    report = {
        "builder_version": BUILDER_VERSION,
        "candidate_protocol": "landslide_bridge_rule_candidate_v1",
        "records": len(candidates),
        "auto_train_records": len(auto_train),
        "records_by_target_status": dict(sorted(Counter(row["target_status"] for row in candidates).items())),
        "expert_truth_records": sum(bool(row["candidate"]["is_expert_truth"]) for row in candidates),
        "errors": [],
    }
    print(f"[BRIDGE:CANDIDATE] records={len(candidates)} auto_train={len(auto_train)} expert_truth=0")
    if not args.dry_run:
        write_jsonl(output_path, candidates)
        write_jsonl(train_path, auto_train)
        write_json(report_path, report)


if __name__ == "__main__":
    main()
