#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""步骤 3-4：验证近重复图像、合并 canonical parent 并冻结 split。

用途：在 task view 展开后的 source index 上构建视觉 parent 簇。SHA-256 exact
duplicate 和经过 RGB-MAE 验证的 dHash 候选会合并为一个 canonical parent；
RSGPT scene group 只约束 split，不会被错误地合并成同一图像。
推荐运行命令：python scripts/3-description/3-4_deduplicate_and_split.py --mode small --output-dir benchmark/qpsalm_description_v2_small --overwrite
主要输入：3-2/3-3 的 source indexes 与 parent manifests。
主要输出：canonical selected source index、split/duplicate manifests 和合并报告。
写入行为：只写待物化源记录，不复制图片；--dry-run 不写文件。
所属流程：Description Benchmark M1.1 科学收尾。
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from description_common import (
    BUILDER_VERSION,
    description_dir_for_mode,
    deterministic_split,
    ensure_writable,
    perceptual_rgb_mae,
    read_jsonl,
    select_parent_ids,
    sha256_jsonl_rows,
    stable_id,
    write_json,
    write_jsonl,
)


SPLIT_PROTOCOL = "qpsalm_description_parent_split_v3"
PERCEPTUAL_PROTOCOL = "dhash_exact_rgb64_mae_v1"
DEFAULT_MAE_THRESHOLD = 3.0
LOSSLESS_SUFFIXES = {".png", ".tif", ".tiff", ".bmp"}


class UnionFind:
    def __init__(self, values: list[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            next_value = self.parent[value]
            self.parent[value] = root
            value = next_value
        return root

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Description canonical parent dedup/split freeze")
    parser.add_argument("--mode", choices=["small", "full"], default="small")
    parser.add_argument("--output-dir")
    parser.add_argument("--small-mmrs-parents", type=int, default=12000)
    parser.add_argument("--small-dior-parents", type=int, default=5000)
    parser.add_argument("--perceptual-mae-threshold", type=float, default=DEFAULT_MAE_THRESHOLD)
    parser.add_argument("--max-samples", type=int, default=0, help="smoke 时限制最终 canonical parent 总数")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_sources(output_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    index_dir = output_dir / "indexes"
    manifest_dir = output_dir / "manifests"
    rows = read_jsonl(index_dir / "global_caption_source.jsonl") + read_jsonl(
        index_dir / "region_alignment_source.jsonl"
    )
    parents = read_jsonl(manifest_dir / "global_caption_parents.jsonl") + read_jsonl(
        manifest_dir / "region_alignment_parents.jsonl"
    )
    return rows, parents


def _is_caption_parent(row: dict[str, Any]) -> bool:
    return str(row.get("source_dataset")) != "DIOR-RSVG"


def _held_out(row: dict[str, Any]) -> bool:
    return row.get("source_split") == "test" or row.get("source_dataset") == "RSIEval"


def _canonical_sort_key(row: dict[str, Any]) -> tuple[int, int, int, str]:
    suffix = Path(str(row["source_image_path"])).suffix.casefold()
    return (
        -int(_held_out(row)),
        -int(row.get("width", 0)) * int(row.get("height", 0)),
        -int(suffix in LOSSLESS_SUFFIXES),
        str(row["parent_sample_id"]),
    )


def analyze_perceptual_candidates(
    parents: list[dict[str, Any]],
    mae_threshold: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[tuple[str, str]]]:
    """验证 dHash 候选并返回候选组、verified edges 和可合并边。"""
    if not 0 < mae_threshold <= 255:
        raise ValueError("--perceptual-mae-threshold 必须位于 (0, 255]")
    by_dhash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for parent in parents:
        by_dhash[str(parent["dhash64"])].append(parent)

    candidates: list[dict[str, Any]] = []
    verified_pairs: list[dict[str, Any]] = []
    merge_edges: list[tuple[str, str]] = []
    for dhash, group in sorted(by_dhash.items()):
        group.sort(key=lambda row: str(row["parent_sample_id"]))
        if len(group) <= 1 or len({str(row["sha256"]) for row in group}) <= 1:
            continue
        pair_records: list[dict[str, Any]] = []
        for left, right in itertools.combinations(group, 2):
            left_id = str(left["parent_sample_id"])
            right_id = str(right["parent_sample_id"])
            if not (_is_caption_parent(left) and _is_caption_parent(right)):
                pair_records.append({
                    "left_parent_id": left_id,
                    "right_parent_id": right_id,
                    "status": "candidate_not_mergeable_across_task_components",
                    "rgb64_mae": None,
                })
                continue
            mae = perceptual_rgb_mae(left["source_image_path"], right["source_image_path"], size=64)
            verified = mae <= mae_threshold
            pair = {
                "left_parent_id": left_id,
                "right_parent_id": right_id,
                "left_source_dataset": str(left["source_dataset"]),
                "right_source_dataset": str(right["source_dataset"]),
                "rgb64_mae": round(mae, 8),
                "threshold": mae_threshold,
                "status": "verified_near_duplicate" if verified else "possible_near_duplicate",
            }
            pair_records.append(pair)
            if verified:
                verified_pairs.append({"dhash64": dhash, **pair})
                merge_edges.append((left_id, right_id))
        candidates.append({
            "candidate_type": "dhash_exact_near_duplicate_candidate",
            "verification_protocol": PERCEPTUAL_PROTOCOL,
            "dhash64": dhash,
            "parent_sample_ids": [str(row["parent_sample_id"]) for row in group],
            "source_datasets": sorted({str(row["source_dataset"]) for row in group}),
            "pairs": pair_records,
        })
    return candidates, verified_pairs, merge_edges


def build_canonical_parents(
    parents: list[dict[str, Any]],
    merge_edges: list[tuple[str, str]],
) -> tuple[list[dict[str, Any]], dict[str, str], list[dict[str, Any]]]:
    ids = [str(row["parent_sample_id"]) for row in parents]
    if len(ids) != len(set(ids)):
        raise ValueError("parent_sample_id 在 global/region manifests 中不唯一")
    parent_by_id = {str(row["parent_sample_id"]): row for row in parents}
    union = UnionFind(ids)

    by_sha: dict[str, list[str]] = defaultdict(list)
    for row in parents:
        if _is_caption_parent(row):
            by_sha[str(row["sha256"])].append(str(row["parent_sample_id"]))
    for group in by_sha.values():
        for value in group[1:]:
            union.union(group[0], value)
    for left, right in merge_edges:
        union.union(left, right)

    components: dict[str, list[str]] = defaultdict(list)
    for parent_id in ids:
        components[union.find(parent_id)].append(parent_id)

    canonical_parents: list[dict[str, Any]] = []
    original_to_canonical: dict[str, str] = {}
    cluster_records: list[dict[str, Any]] = []
    edge_set = {tuple(sorted(edge)) for edge in merge_edges}
    for members in components.values():
        members.sort()
        member_rows = [parent_by_id[parent_id] for parent_id in members]
        primary = sorted(member_rows, key=_canonical_sort_key)[0]
        canonical_id = str(primary["parent_sample_id"])
        for parent_id in members:
            original_to_canonical[parent_id] = canonical_id

        has_verified_edge = any(
            tuple(sorted((left, right))) in edge_set
            for left, right in itertools.combinations(members, 2)
        )
        cluster_kind = (
            "verified_near_duplicate"
            if has_verified_edge
            else "exact_duplicate" if len(members) > 1 else "singleton"
        )
        cluster_id = stable_id("visual_cluster", *members)
        canonical = {
            **copy.deepcopy(primary),
            "parent_sample_id": canonical_id,
            "canonical_parent_id": canonical_id,
            "merged_parent_ids": members,
            "source_datasets": sorted({str(row["source_dataset"]) for row in member_rows}),
            "source_split": "test" if any(_held_out(row) for row in member_rows) else None,
            "perceptual_cluster_id": cluster_id,
            "canonical_merge_kind": cluster_kind,
            "source_parent_count": len(members),
        }
        canonical_parents.append(canonical)
        if len(members) > 1:
            cluster_records.append({
                "perceptual_cluster_id": cluster_id,
                "canonical_parent_id": canonical_id,
                "merged_parent_ids": members,
                "source_datasets": canonical["source_datasets"],
                "merge_kind": cluster_kind,
                "contains_held_out_source": any(_held_out(row) for row in member_rows),
                "canonical_selection_protocol": "held_out_then_pixels_then_lossless_then_parent_id",
            })
    canonical_parents.sort(key=lambda row: str(row["parent_sample_id"]))
    cluster_records.sort(key=lambda row: str(row["perceptual_cluster_id"]))
    return canonical_parents, original_to_canonical, cluster_records


def _answer_key(answer: dict[str, Any]) -> str:
    return " ".join(str(answer.get("text", "")).split()).casefold()


def _answer_source_provenance(
    row: dict[str, Any], answer: dict[str, Any], answer_index: int,
) -> dict[str, Any]:
    provenance = row["provenance"]
    text = str(answer.get("text", ""))
    return {
        "source_dataset": str(row["source_dataset"]),
        "source_parent_sample_id": str(row["parent_sample_id"]),
        "source_sample_id": str(row["sample_id"]),
        "source_answer_index": int(answer_index),
        "source_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "annotation_origin": str(answer.get("annotation_origin") or "unknown"),
        "annotation_path": provenance.get("annotation_path"),
        "original_record_id": provenance.get("original_record_id"),
    }


def merge_caption_records(
    rows: list[dict[str, Any]],
    parents: list[dict[str, Any]],
    original_to_canonical: dict[str, str],
) -> list[dict[str, Any]]:
    parent_by_id = {str(row["parent_sample_id"]): row for row in parents}
    by_canonical: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["component_benchmark"] == "rs_global_caption_v1":
            by_canonical[original_to_canonical[str(row["parent_sample_id"])]].append(row)

    merged: list[dict[str, Any]] = []
    for canonical_id, source_rows in sorted(by_canonical.items()):
        source_rows.sort(key=lambda row: str(row["parent_sample_id"]))
        primary = next(row for row in source_rows if str(row["parent_sample_id"]) == canonical_id)
        output = copy.deepcopy(primary)
        source_ids = sorted(str(row["parent_sample_id"]) for row in source_rows)
        source_datasets = sorted({str(row["source_dataset"]) for row in source_rows})
        answer_by_key: dict[str, dict[str, Any]] = {}
        for row in source_rows:
            for answer_index, answer in enumerate(row["answers"]):
                answer_source = _answer_source_provenance(row, answer, answer_index)
                key = _answer_key(answer)
                if key not in answer_by_key:
                    item = copy.deepcopy(answer)
                    item["source_provenance"] = [answer_source]
                    answer_by_key[key] = item
                else:
                    item = answer_by_key[key]
                    item["quality"] = max(float(item["quality"]), float(answer["quality"]))
                    item["caption_quality_weight"] = max(
                        float(item["caption_quality_weight"]),
                        float(answer["caption_quality_weight"]),
                    )
                    item["source_provenance"].append(answer_source)

        output["sample_id"] = f"{canonical_id}__global_caption"
        output["parent_sample_id"] = canonical_id
        output["canonical_parent_id"] = canonical_id
        output["merged_parent_ids"] = source_ids
        output["source_datasets"] = source_datasets
        output["perceptual_cluster_id"] = next(
            row["perceptual_cluster_id"]
            for row in parents
            if str(row["parent_sample_id"]) == canonical_id
        )
        output["answers"] = sorted(answer_by_key.values(), key=lambda item: _answer_key(item))
        output["answer_type"] = (
            "multi_reference_caption" if len(output["answers"]) > 1 else output["answer_type"]
        )
        output["quality_flags"] = sorted({
            flag for row in source_rows for flag in row.get("quality_flags", [])
        } | ({"canonical_visual_parent_merged"} if len(source_rows) > 1 else set()))
        provenance_sources = []
        for row in source_rows:
            provenance_sources.append({
                "source_dataset": str(row["source_dataset"]),
                "source_parent_sample_id": str(row["parent_sample_id"]),
                **copy.deepcopy(row["provenance"]),
            })
        output["provenance"]["sources"] = provenance_sources
        output["provenance"]["merged_source_sample_ids"] = sorted(
            str(row["sample_id"]) for row in source_rows
        )
        output["sampling_metadata"] = {
            "source_group": "+".join(source_datasets),
            "caption_length_bin": min(
                int(sum(len(item["text"].split()) for item in output["answers"]) / max(len(output["answers"]), 1)) // 8,
                8,
            ),
            "num_references": len(output["answers"]),
        }
        merged.append(output)
    return merged


def connected_assignments(parents: list[dict[str, Any]], seed: int) -> dict[str, dict[str, Any]]:
    """对 canonical parent 施加 exact/scene split 约束，但不再合并图像。"""
    ids = [str(row["parent_sample_id"]) for row in parents]
    union = UnionFind(ids)
    by_sha: dict[str, list[str]] = defaultdict(list)
    by_scene: dict[str, list[str]] = defaultdict(list)
    parent_by_id = {str(row["parent_sample_id"]): row for row in parents}
    for row in parents:
        parent_id = str(row["parent_sample_id"])
        by_sha[str(row["sha256"])].append(parent_id)
        scene = row.get("source_scene_group")
        source_datasets = set(row.get("source_datasets") or [row["source_dataset"]])
        if scene and source_datasets & {"RSICap", "RSIEval"}:
            by_scene[f"rsgpt_dota:{scene}"].append(parent_id)
    for group in list(by_sha.values()) + list(by_scene.values()):
        for value in group[1:]:
            union.union(group[0], value)

    components: dict[str, list[str]] = defaultdict(list)
    for parent_id in ids:
        components[union.find(parent_id)].append(parent_id)
    result: dict[str, dict[str, Any]] = {}
    for component in components.values():
        component.sort()
        explicit_test = any(_held_out(parent_by_id[parent_id]) for parent_id in component)
        split_key = stable_id("split_group", seed, *component)
        split = "test" if explicit_test else deterministic_split(split_key, train_ratio=0.9)
        reason = "source_or_rsieval_test_priority" if explicit_test else "deterministic_connected_parent_group"
        for parent_id in component:
            result[parent_id] = {
                "split": split,
                "split_reason": reason,
                "connected_group_id": split_key,
                "exact_cluster_id": stable_id("exact", parent_by_id[parent_id]["sha256"]),
                "perceptual_cluster_id": parent_by_id[parent_id].get("perceptual_cluster_id"),
            }
    return result


def select_small(parents: list[dict[str, Any]], args: argparse.Namespace) -> set[str]:
    if args.mode == "full":
        selected = {str(row["parent_sample_id"]) for row in parents}
    else:
        def sources(row: dict[str, Any]) -> set[str]:
            return set(row.get("source_datasets") or [str(row["source_dataset"])])

        rsicap_eval = [row for row in parents if sources(row) & {"RSICap", "RSIEval"}]
        mmrs = [row for row in parents if any(value.startswith("MMRS-") for value in sources(row))]
        dior = [row for row in parents if row["source_dataset"] == "DIOR-RSVG"]
        for row in mmrs + dior:
            row["sampling_stratum"] = json.dumps(row.get("stratum", {}), sort_keys=True)
        selected = {str(row["parent_sample_id"]) for row in rsicap_eval}
        selected |= select_parent_ids(mmrs, args.small_mmrs_parents, ["source_dataset", "sampling_stratum"])
        selected |= select_parent_ids(dior, args.small_dior_parents, ["source_dataset", "sampling_stratum"])
    if args.max_samples > 0 and len(selected) > args.max_samples:
        candidates = [row for row in parents if row["parent_sample_id"] in selected]
        selected = select_parent_ids(candidates, args.max_samples, ["source_dataset"])
    return selected


def main() -> None:
    args = parse_args()
    output_dir = description_dir_for_mode(args.mode, args.output_dir)
    source_rows, source_parents = load_sources(output_dir)
    candidates, verified_pairs, merge_edges = analyze_perceptual_candidates(
        source_parents, args.perceptual_mae_threshold
    )
    canonical_parents, original_to_canonical, merge_clusters = build_canonical_parents(
        source_parents, merge_edges
    )
    canonical_by_id = {str(row["parent_sample_id"]): row for row in canonical_parents}
    caption_rows = merge_caption_records(
        source_rows, canonical_parents, original_to_canonical
    )
    region_rows = [
        copy.deepcopy(row)
        for row in source_rows
        if row["component_benchmark"] == "rs_region_alignment_v1"
    ]
    for row in region_rows:
        parent = canonical_by_id[str(row["parent_sample_id"])]
        row["canonical_parent_id"] = str(row["parent_sample_id"])
        row["merged_parent_ids"] = [str(row["parent_sample_id"])]
        row["source_datasets"] = [str(row["source_dataset"])]
        row["perceptual_cluster_id"] = parent["perceptual_cluster_id"]
        for answer_index, answer in enumerate(row.get("answers", [])):
            answer["source_provenance"] = [
                _answer_source_provenance(row, answer, answer_index)
            ]
        row["provenance"]["sources"] = [{
            "source_dataset": str(row["source_dataset"]),
            "source_parent_sample_id": str(row["parent_sample_id"]),
            **copy.deepcopy(row["provenance"]),
        }]

    canonical_rows = caption_rows + region_rows
    assignments = connected_assignments(canonical_parents, args.seed)
    selected = select_small(canonical_parents, args)

    selected_parents: list[dict[str, Any]] = []
    for parent in canonical_parents:
        parent_id = str(parent["parent_sample_id"])
        if parent_id not in selected:
            continue
        if not parent.get("source_image_path"):
            raise ValueError(f"canonical parent 缺少 source_image_path: {parent_id}")
        selected_parents.append({**parent, **assignments[parent_id], "selected_for_mode": args.mode})
    selected_parents.sort(key=lambda row: str(row["parent_sample_id"]))

    selected_rows: list[dict[str, Any]] = []
    for row in canonical_rows:
        parent_id = str(row["parent_sample_id"])
        if parent_id not in selected:
            continue
        source_path = str(row.get("provenance", {}).get("source_image_path") or row["visual_ref"]["path"])
        if not source_path.startswith("datasets/"):
            raise ValueError(f"canonical source record 必须指向 datasets: {row['sample_id']} -> {source_path}")
        selected_rows.append({
            **row,
            "split": assignments[parent_id]["split"],
            "split_metadata": assignments[parent_id],
        })
    selected_rows.sort(
        key=lambda row: (row["split"], row["source_dataset"], row["parent_sample_id"], row["sample_id"])
    )
    split_rows = {split: [row for row in selected_rows if row["split"] == split] for split in ("train", "dev", "test")}

    selected_original_ids = {
        original_id
        for original_id, canonical_id in original_to_canonical.items()
        if canonical_id in selected
    }
    verified_manifest = []
    source_parent_by_id = {
        str(parent["parent_sample_id"]): parent for parent in source_parents
    }
    for cluster in merge_clusters:
        if cluster["merge_kind"] != "verified_near_duplicate":
            continue
        members = set(cluster["merged_parent_ids"])
        edges = [
            pair for pair in verified_pairs
            if pair["left_parent_id"] in members and pair["right_parent_id"] in members
        ]
        canonical_id = str(cluster["canonical_parent_id"])
        final_split = str(assignments[canonical_id]["split"])
        member_details = []
        for member_id in cluster["merged_parent_ids"]:
            member = source_parent_by_id[str(member_id)]
            declared_split = member.get("source_split")
            if declared_split == "test":
                split_action = "preserved_official_test"
            elif final_split == "test":
                split_action = "promoted_to_test_by_cluster_priority"
            else:
                split_action = "assigned_by_deterministic_split"
            member_details.append({
                "parent_sample_id": str(member_id),
                "source_dataset": str(member["source_dataset"]),
                "source_image_path": str(member["source_image_path"]),
                "source_declared_split": declared_split,
                "assigned_final_split": final_split,
                "split_action": split_action,
                "is_canonical": str(member_id) == canonical_id,
                "width": int(member["width"]),
                "height": int(member["height"]),
                "suffix": Path(str(member["source_image_path"])).suffix.casefold(),
                "sha256": str(member["sha256"]),
            })
        verified_manifest.append({
            **cluster,
            "verification_protocol": PERCEPTUAL_PROTOCOL,
            "mae_threshold": args.perceptual_mae_threshold,
            "verified_edges": edges,
            "assigned_final_split": final_split,
            "member_details": member_details,
            "selected_for_mode": cluster["canonical_parent_id"] in selected,
        })
    verified_manifest.sort(key=lambda row: str(row["perceptual_cluster_id"]))

    split_manifest = [{
        "parent_sample_id": row["parent_sample_id"],
        "canonical_parent_id": row["canonical_parent_id"],
        "merged_parent_ids": row["merged_parent_ids"],
        "source_dataset": row["source_dataset"],
        "source_datasets": row["source_datasets"],
        "sha256": row["sha256"],
        "dhash64": row["dhash64"],
        "source_scene_group": row.get("source_scene_group"),
        "split": row["split"],
        "split_reason": row["split_reason"],
        "connected_group_id": row["connected_group_id"],
        "exact_cluster_id": row["exact_cluster_id"],
        "perceptual_cluster_id": row["perceptual_cluster_id"],
    } for row in selected_parents]
    split_manifest.sort(key=lambda row: str(row["parent_sample_id"]))

    outputs = {
        output_dir / "indexes/selected_source_all.jsonl": selected_rows,
        output_dir / "manifests/selected_parent_source_manifest.jsonl": selected_parents,
        output_dir / "manifests/split_manifest.jsonl": split_manifest,
        output_dir / "manifests/perceptual_duplicate_candidates.jsonl": candidates,
        output_dir / "manifests/verified_perceptual_duplicates.jsonl": verified_manifest,
    }
    report_path = output_dir / "reports/dedup_split_report.json"
    merge_report_path = output_dir / "reports/canonical_merge_report.json"
    protocol_path = output_dir / "manifests/split_protocol.json"
    state_path = output_dir / "manifests/materialization_state.json"
    for path in [*outputs, report_path, merge_report_path, protocol_path, state_path]:
        ensure_writable(path, args.overwrite, args.dry_run)

    exact_groups = Counter(row["exact_cluster_id"] for row in split_manifest)
    report = {
        "builder_version": BUILDER_VERSION,
        "mode": args.mode,
        "seed": args.seed,
        "source_parent_count": len(source_parents),
        "canonical_parent_count": len(canonical_parents),
        "selected_parent_count": len(selected_parents),
        "record_count": len(selected_rows),
        "parent_by_source": dict(sorted(Counter(row["source_dataset"] for row in selected_parents).items())),
        "parent_by_split": dict(sorted(Counter(row["split"] for row in selected_parents).items())),
        "records_by_split": {key: len(value) for key, value in split_rows.items()},
        "records_by_task": dict(sorted(Counter(row["task_family"] for row in selected_rows).items())),
        "exact_duplicate_groups": sum(count > 1 for count in exact_groups.values()),
        "perceptual_candidate_groups": len(candidates),
        "verified_perceptual_pairs": len(verified_pairs),
        "verified_perceptual_clusters": len(verified_manifest),
        "selected_original_parent_count": len(selected_original_ids),
        "perceptual_policy": "verified_clusters_merged_before_split_and_sampling",
        "errors": [],
    }
    merge_report = {
        "builder_version": BUILDER_VERSION,
        "protocol": PERCEPTUAL_PROTOCOL,
        "mae_threshold": args.perceptual_mae_threshold,
        "source_parent_count": len(source_parents),
        "canonical_parent_count": len(canonical_parents),
        "parents_removed_by_merge": len(source_parents) - len(canonical_parents),
        "source_caption_record_count": sum(
            row["component_benchmark"] == "rs_global_caption_v1" for row in source_rows
        ),
        "canonical_caption_record_count": len(caption_rows),
        "caption_records_removed_by_merge": sum(
            row["component_benchmark"] == "rs_global_caption_v1" for row in source_rows
        ) - len(caption_rows),
        "source_caption_answer_count": sum(
            len(row.get("answers", []))
            for row in source_rows
            if row["component_benchmark"] == "rs_global_caption_v1"
        ),
        "canonical_caption_answer_count": sum(len(row.get("answers", [])) for row in caption_rows),
        "merge_clusters": len(merge_clusters),
        "verified_near_duplicate_clusters": sum(
            row["merge_kind"] == "verified_near_duplicate" for row in merge_clusters
        ),
        "exact_duplicate_clusters": sum(row["merge_kind"] == "exact_duplicate" for row in merge_clusters),
        "selected_verified_clusters": sum(row["selected_for_mode"] for row in verified_manifest),
        "errors": [],
    }
    protocol = {
        "protocol": SPLIT_PROTOCOL,
        "immutable": True,
        "seed": args.seed,
        "priority": ["RSIEval test", "source official test", "verified/exact canonical cluster", "RSGPT scene group", "deterministic train/dev"],
        "perceptual_verification": {
            "protocol": PERCEPTUAL_PROTOCOL,
            "dhash_hamming_distance": 0,
            "rgb_preview_size": 64,
            "mae_threshold": args.perceptual_mae_threshold,
        },
        "small_quotas": {"mmrs_canonical_parents": args.small_mmrs_parents, "dior_parents": args.small_dior_parents},
        "source_index_hash": sha256_jsonl_rows(source_rows),
        "selected_source_index_hash": sha256_jsonl_rows(selected_rows),
        "selected_parent_manifest_hash": sha256_jsonl_rows(selected_parents),
        "verified_duplicate_manifest_hash": sha256_jsonl_rows(verified_manifest),
        "split_manifest_hash": sha256_jsonl_rows(split_manifest),
    }
    state = {
        "protocol": "qpsalm_description_materialization_v2",
        "builder_version": BUILDER_VERSION,
        "status": "pending",
        "selected_source_index_hash": protocol["selected_source_index_hash"],
        "selected_parent_manifest_hash": protocol["selected_parent_manifest_hash"],
    }
    print(
        f"[SPLIT] mode={args.mode} source_parents={len(source_parents)} "
        f"canonical={len(canonical_parents)} selected={len(selected_parents)} records={len(selected_rows)}"
    )
    print(
        f"[DEDUP] candidates={len(candidates)} verified_pairs={len(verified_pairs)} "
        f"verified_clusters={len(verified_manifest)} threshold={args.perceptual_mae_threshold}"
    )
    print(f"[SPLIT] manifest_sha256={protocol['split_manifest_hash']}")
    if not args.dry_run:
        for path, payload in outputs.items():
            write_jsonl(path, payload)
        write_json(report_path, report)
        write_json(merge_report_path, merge_report)
        write_json(protocol_path, protocol)
        write_json(state_path, state)
        print(f"[SPLIT] manifest={output_dir / 'manifests/split_manifest.jsonl'}")


if __name__ == "__main__":
    main()
