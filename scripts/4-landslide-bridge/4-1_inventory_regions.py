#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""步骤 4-1：清点 Landslide V2 区域并物化 Bridge region mask。

用途：从父语义 mask、8 邻域连通域和 referring target 构建可追溯区域清单，
同时确定 300-parent 专家 Pilot。pseudo component 不声明为真实滑坡实例。
推荐运行命令：python scripts/4-landslide-bridge/4-1_inventory_regions.py --mode small --overwrite
主要输入：benchmark/multisource_landslide_v2_<mode> final/referring indexes。
主要输出：region_inventory、pilot/review manifests 和 data/regions/*.npy。
写入行为：只写 Bridge benchmark，不修改 Landslide V2；--dry-run 不写文件。
"""

from __future__ import annotations

import argparse
import copy
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from landslide_bridge_common import (
    BUILDER_VERSION,
    SCHEMA_VERSION,
    area_bin,
    atomic_save_npy,
    binary_mask,
    bridge_parent_from_landslide_v2,
    bridge_dir,
    connected_components,
    ensure_writable,
    geometry_from_mask,
    load_config,
    mask_digest,
    modality_family_combo,
    parent_index_ref,
    read_jsonl,
    resolve_project_path,
    safe_slug,
    source_benchmark_dir,
    stable_hash,
    stable_id,
    stratified_select,
    to_project_ref,
    valid_canvas,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建 Landslide Bridge region inventory")
    parser.add_argument("--mode", choices=["small", "full"], default="small")
    parser.add_argument("--source-benchmark")
    parser.add_argument("--output-dir")
    parser.add_argument("--config", default="configs/landslide_bridge_v1.yaml")
    parser.add_argument("--pilot-parents", type=int, default=300)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def region_mask_path(output_dir: Path, parent: dict[str, Any], region_id: str) -> Path:
    return (
        output_dir / "data/regions" / str(parent["split"]) / safe_slug(str(parent["dataset_name"]))
        / str(parent["parent_sample_id"]) / region_id / "mask.npy"
    )


def base_inventory_record(
    parent: dict[str, Any], source_dir: Path, region_id: str, region_source: str,
    target_status: str, mask: Any | None, valid: Any, mask_path: Path | None,
) -> dict[str, Any]:
    geometry = geometry_from_mask(mask, valid)
    modality_paths = {
        name: str(item["path"])
        for name, item in parent.get("modalities", {}).items()
        if item.get("available", True) and item.get("path")
    }
    if any(not path.startswith("benchmark/") for path in modality_paths.values()):
        raise ValueError(f"Bridge 只能引用已物化 benchmark 模态: {parent['parent_sample_id']}")
    mask_ref = None
    if mask is not None and mask_path is not None:
        mask_ref = {
            "path": to_project_ref(mask_path),
            "sha256": mask_digest(mask),
            "shape": list(mask.shape),
            "positive_pixels": int((mask > 0).sum()),
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "bridge_record_id": stable_id("bridge", parent["parent_sample_id"], region_id),
        "parent_sample_id": str(parent["parent_sample_id"]),
        "source_benchmark": to_project_ref(source_dir),
        "source_parent_index": parent_index_ref(source_dir, str(parent["split"])),
        "split": str(parent["split"]),
        "dataset_name": str(parent["dataset_name"]),
        "region_id": region_id,
        "region_source": region_source,
        "target_status": target_status,
        "task_family": "landslide_region_caption" if target_status == "present" else "no_target_response",
        "instruction": (
            "Describe the specified landslide region using the available remote-sensing evidence."
            if target_status == "present" else
            "Determine whether the specified landslide target is present and explain the available evidence."
        ),
        "condition": "specified landslide region",
        "answer_type": "structured_json_with_summary",
        "region_geometry": geometry,
        "region_mask": mask_ref,
        "visual_ref": {
            "type": "multisource_parent",
            "parent_index": parent_index_ref(source_dir, str(parent["split"])),
            "modality_paths": modality_paths,
            "preview_paths": copy.deepcopy(parent.get("preview", {}).get("paths", {})),
            "original_size": list(parent.get("spatial", {}).get("original_size", [])),
        },
        "modality_metadata": copy.deepcopy(parent.get("modalities", {})),
        "modality_family_combo": modality_family_combo(parent),
        "source_region_aliases": [],
        "modality_evidence": {},
        "structured_targets": {},
        "candidate": {"origin": "deterministic_rules", "summary": "", "is_expert_truth": False},
        "review": {"status": "not_selected"},
        "provenance": {
            "builder_version": BUILDER_VERSION,
            "source_parent_sample_id": str(parent["parent_sample_id"]),
            "source_mask_path": parent.get("mask", {}).get("path"),
            "region_generation": region_source,
        },
        "quality_flags": [],
    }


def _materialize_mask(path: Path, mask: Any, overwrite: bool, dry_run: bool) -> None:
    ensure_writable(path, overwrite, dry_run)
    if not dry_run:
        atomic_save_npy(path, mask.astype("uint8"))


def _remove_stale_region_masks(
    output_dir: Path,
    inventory: list[dict[str, Any]],
    *,
    overwrite: bool,
    dry_run: bool,
) -> int:
    """Remove only obsolete files below this Bridge benchmark's region root."""
    if not overwrite or dry_run:
        return 0
    root = (output_dir / "data/regions").resolve(strict=False)
    if not root.exists():
        return 0
    expected = {
        resolve_project_path(row["region_mask"]["path"]).resolve(strict=False)
        for row in inventory
        if row.get("region_mask")
    }
    removed = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file() and not path.is_symlink():
            continue
        resolved = path.resolve(strict=False)
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"拒绝清理 region root 外路径: {path}") from exc
        if resolved not in expected:
            path.unlink()
            removed += 1
    for directory in sorted((value for value in root.rglob("*") if value.is_dir()), reverse=True):
        try:
            directory.rmdir()
        except OSError:
            pass
    return removed


def build_parent_regions(
    parent: dict[str, Any], referring_rows: list[dict[str, Any]], source_dir: Path,
    output_dir: Path, config: dict[str, Any], overwrite: bool, dry_run: bool,
) -> list[dict[str, Any]]:
    mask_spec = parent.get("mask") or {}
    if not mask_spec.get("path"):
        return []
    global_mask = binary_mask(str(mask_spec["path"]))
    valid = valid_canvas(parent, global_mask.shape)
    global_mask = ((global_mask > 0) & (valid > 0)).astype("uint8")
    settings = config["regions"]
    records: list[dict[str, Any]] = []
    records_by_mask_hash: dict[str, dict[str, Any]] = {}

    if bool(global_mask.any()):
        region_id = "global_" + mask_digest(global_mask)[:12]
        path = region_mask_path(output_dir, parent, region_id)
        _materialize_mask(path, global_mask, overwrite, dry_run)
        global_record = base_inventory_record(
            parent, source_dir, region_id, "gt_global_mask", "present", global_mask, valid, path
        )
        records.append(global_record)
        records_by_mask_hash[mask_digest(global_mask)] = global_record
        for component_index, component in enumerate(connected_components(
            global_mask, valid,
            int(settings["min_component_area_pixels"]),
            float(settings["min_component_area_fraction"]),
        ), start=1):
            region_id = f"component_{component_index:03d}_{mask_digest(component)[:12]}"
            path = region_mask_path(output_dir, parent, region_id)
            _materialize_mask(path, component, overwrite, dry_run)
            record = base_inventory_record(
                parent, source_dir, region_id, "pseudo_instance_component", "present", component, valid, path
            )
            record["quality_flags"].append("pseudo_instance_not_human_instance_annotation")
            records.append(record)
            records_by_mask_hash.setdefault(mask_digest(component), record)

    referring_by_hash: dict[str, dict[str, Any]] = {}
    no_target_aliases: list[dict[str, Any]] = []
    for referring in referring_rows:
        target = referring.get("target_mask") or {}
        alias = {
            "sample_id": referring.get("sample_id"),
            "category": referring.get("category"),
            "subtype": referring.get("subtype"),
            "target_key": target.get("target_key"),
        }
        if target.get("empty_mask") or int(target.get("positive_pixels") or 0) == 0:
            no_target_aliases.append(alias)
            continue
        target_mask = binary_mask(str(target["path"]))
        if target_mask.shape != global_mask.shape:
            raise ValueError(f"referring mask shape 不一致: {referring.get('sample_id')}")
        target_mask = ((target_mask > 0) & (valid > 0)).astype("uint8")
        if not bool(target_mask.any()):
            alias["valid_canvas_status"] = "empty_after_valid_mask"
            no_target_aliases.append(alias)
            continue
        digest = mask_digest(target_mask)
        if digest in records_by_mask_hash:
            existing = records_by_mask_hash[digest]
            existing["source_region_aliases"].append(alias)
            if "also_gt_referring_mask" not in existing["quality_flags"]:
                existing["quality_flags"].append("also_gt_referring_mask")
            continue
        if digest not in referring_by_hash:
            region_id = "referring_" + digest[:12]
            path = region_mask_path(output_dir, parent, region_id)
            _materialize_mask(path, target_mask, overwrite, dry_run)
            referring_by_hash[digest] = base_inventory_record(
                parent, source_dir, region_id, "gt_referring_mask", "present", target_mask, valid, path
            )
            records_by_mask_hash[digest] = referring_by_hash[digest]
        referring_by_hash[digest]["source_region_aliases"].append(alias)
    records.extend(referring_by_hash.values())

    if not bool(global_mask.any()) or no_target_aliases:
        region_id = "no_target"
        record = base_inventory_record(
            parent, source_dir, region_id, "no_target", "absent", None, valid, None
        )
        record["source_region_aliases"] = no_target_aliases
        record["quality_flags"].append("counterfactual_or_empty_target")
        records.append(record)
    return records


def _split_quotas(total: int, config: dict[str, Any]) -> dict[str, int]:
    configured = config["pilot"]["split_parent_quotas"]
    base_total = sum(int(value) for value in configured.values())
    if total == base_total:
        return {key: int(value) for key, value in configured.items()}
    ratios = {key: int(value) / base_total for key, value in configured.items()}
    quotas = {key: int(total * ratio) for key, ratio in ratios.items()}
    for split in sorted(quotas, key=lambda key: (-ratios[key], key)):
        if sum(quotas.values()) >= total:
            break
        quotas[split] += 1
    return quotas


def build_review_selection(
    inventory: list[dict[str, Any]], parent_rows: dict[str, dict[str, Any]],
    pilot_parents: int, config: dict[str, Any], seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in inventory:
        by_parent[str(record["parent_sample_id"])].append(record)
    summaries: list[dict[str, Any]] = []
    for parent_id, regions in by_parent.items():
        parent = parent_rows[parent_id]
        present = [row for row in regions if row["target_status"] == "present"]
        summaries.append({
            "parent_sample_id": parent_id,
            "split": str(parent["split"]),
            "dataset_name": str(parent["dataset_name"]),
            "modality_family_combo": modality_family_combo(parent),
            "present_area_bin": area_bin(max(
                (float(row["region_geometry"]["valid_area_ratio"]) for row in present), default=0.0
            )),
            "has_no_target": any(row["target_status"] == "absent" for row in regions),
            "available_region_sources": sorted({row["region_source"] for row in regions}),
        })
    quotas = _split_quotas(pilot_parents, config)
    selected_parents: list[dict[str, Any]] = []
    fields = config["pilot"]["stratify_by"]
    for split, quota in quotas.items():
        candidates = [row for row in summaries if row["split"] == split]
        selected_parents.extend(stratified_select(candidates, quota, fields, seed))

    review: list[dict[str, Any]] = []
    for parent in selected_parents:
        regions = by_parent[parent["parent_sample_id"]]
        present = [row for row in regions if row["target_status"] == "present"]
        absent = [row for row in regions if row["target_status"] == "absent"]
        if present:
            sources = sorted({row["region_source"] for row in present})
            desired = sources[int(stable_hash(seed, parent["parent_sample_id"], "source")[:8], 16) % len(sources)]
            options = [row for row in present if row["region_source"] == desired]
            selected = sorted(options, key=lambda row: stable_hash(seed, row["bridge_record_id"]))[0]
            review.append({
                "review_item_id": stable_id("review", selected["bridge_record_id"]),
                "bridge_record_id": selected["bridge_record_id"],
                "parent_sample_id": selected["parent_sample_id"],
                "split": selected["split"],
                "target_status": "present",
                "region_source": selected["region_source"],
            })
        if absent:
            selected = sorted(absent, key=lambda row: stable_hash(seed, row["bridge_record_id"]))[0]
            review.append({
                "review_item_id": stable_id("review", selected["bridge_record_id"]),
                "bridge_record_id": selected["bridge_record_id"],
                "parent_sample_id": selected["parent_sample_id"],
                "split": selected["split"],
                "target_status": "absent",
                "region_source": selected["region_source"],
            })
    review.sort(key=lambda row: (row["split"], row["parent_sample_id"], row["review_item_id"]))
    selected_parents.sort(key=lambda row: (row["split"], row["parent_sample_id"]))
    return selected_parents, review


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    source_dir = source_benchmark_dir(args.mode, args.source_benchmark)
    output_dir = bridge_dir(args.mode, args.output_dir)
    source_rows = read_jsonl(source_dir / "indexes/all.jsonl")
    parent_source_rows = [
        row for row in source_rows
        if row.get("source_level") == "patch" and row.get("supervision") == "mask"
    ]
    parents = [bridge_parent_from_landslide_v2(row) for row in parent_source_rows]
    parents.sort(key=lambda row: str(row["parent_sample_id"]))
    if args.max_samples > 0:
        parents = parents[:args.max_samples]
    parent_by_id = {str(row["parent_sample_id"]): row for row in parents}
    referring = read_jsonl(source_dir / "indexes/referring_target_all.jsonl")
    referring_by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in referring:
        parent_id = str(row["parent_sample_id"])
        if parent_id in parent_by_id:
            referring_by_parent[parent_id].append(row)

    inventory_path = output_dir / "indexes/region_inventory.jsonl"
    pilot_path = output_dir / "manifests/pilot_parent_manifest.jsonl"
    review_path = output_dir / "manifests/review_selection.jsonl"
    report_path = output_dir / "reports/region_inventory_report.json"
    for path in (inventory_path, pilot_path, review_path, report_path):
        ensure_writable(path, args.overwrite, args.dry_run)

    inventory: list[dict[str, Any]] = []
    for parent in parents:
        inventory.extend(build_parent_regions(
            parent, referring_by_parent.get(str(parent["parent_sample_id"]), []),
            source_dir, output_dir, config, args.overwrite, args.dry_run,
        ))
    inventory.sort(key=lambda row: (row["split"], row["parent_sample_id"], row["region_id"]))
    stale_masks_removed = _remove_stale_region_masks(
        output_dir, inventory, overwrite=args.overwrite, dry_run=args.dry_run
    )
    pilot_parents, review_selection = build_review_selection(
        inventory, parent_by_id, args.pilot_parents, config, args.seed
    )
    pilot_quotas = _split_quotas(args.pilot_parents, config)
    pilot_split_counts = Counter(row["split"] for row in pilot_parents)
    pilot_ids = [str(row["parent_sample_id"]) for row in pilot_parents]
    pilot_protocol_complete = bool(
        args.max_samples <= 0
        and len(pilot_parents) == args.pilot_parents
        and len(pilot_ids) == len(set(pilot_ids))
        and all(pilot_split_counts[split] == quota for split, quota in pilot_quotas.items())
    )
    selected_review_ids = {row["bridge_record_id"] for row in review_selection}
    for record in inventory:
        if record["bridge_record_id"] in selected_review_ids:
            record["review"] = {"status": "pending"}

    report = {
        "builder_version": BUILDER_VERSION,
        "mode": args.mode,
        "source_benchmark": to_project_ref(source_dir),
        "source_parent_count": len(parents),
        "inventory_records": len(inventory),
        "inventory_by_source": dict(sorted(Counter(row["region_source"] for row in inventory).items())),
        "inventory_by_split": dict(sorted(Counter(row["split"] for row in inventory).items())),
        "source_parent_limit": int(args.max_samples),
        "pilot_requested_parents": int(args.pilot_parents),
        "pilot_requested_split_quotas": pilot_quotas,
        "pilot_parents": len(pilot_parents),
        "pilot_parent_by_split": dict(sorted(pilot_split_counts.items())),
        "pilot_protocol_complete": pilot_protocol_complete,
        "review_items": len(review_selection),
        "stale_region_masks_removed": stale_masks_removed,
        "errors": [],
    }
    print(
        f"[BRIDGE:INVENTORY] parents={len(parents)} regions={len(inventory)} "
        f"pilot_parents={len(pilot_parents)}/{args.pilot_parents} "
        f"pilot_complete={pilot_protocol_complete} review_items={len(review_selection)}"
    )
    if not args.dry_run:
        write_jsonl(inventory_path, inventory)
        write_jsonl(pilot_path, pilot_parents)
        write_jsonl(review_path, review_selection)
        write_json(report_path, report)


if __name__ == "__main__":
    main()
