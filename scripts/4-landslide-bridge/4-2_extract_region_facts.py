#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""步骤 4-2：按三级证据协议提取滑坡区域确定性事实。

用途：计算 mask 几何、区域/背景环有效覆盖及逐模态物理或归一化相对统计。
推荐运行命令：python scripts/4-landslide-bridge/4-2_extract_region_facts.py --mode small --overwrite
主要输入：4-1 region inventory 与 Landslide V2 已物化模态。
主要输出：indexes/region_facts_all.jsonl 和 evidence report。
写入行为：只写 Bridge 派生索引；不读取原始 datasets；--dry-run 不写文件。
"""

from __future__ import annotations

import argparse
import copy
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from landslide_bridge_common import (
    BUILDER_VERSION,
    binary_mask,
    bridge_dir,
    context_ring,
    ensure_writable,
    load_array,
    load_config,
    read_jsonl,
    source_benchmark_dir,
    to_project_ref,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="提取 Landslide Bridge 区域事实")
    parser.add_argument("--mode", choices=["small", "full"], default="small")
    parser.add_argument("--source-benchmark")
    parser.add_argument("--output-dir")
    parser.add_argument("--config", default="configs/landslide_bridge_v1.yaml")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _resize_binary(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if mask.shape == shape:
        return mask.astype(np.uint8)
    image = Image.fromarray((mask > 0).astype(np.uint8) * 255)
    resized = image.resize((shape[1], shape[0]), Image.Resampling.NEAREST)
    return (np.asarray(resized) > 0).astype(np.uint8)


def _valid_mask(item: dict[str, Any], shape: tuple[int, int]) -> np.ndarray:
    spec = item.get("valid_mask") or {}
    if not spec.get("path"):
        return np.zeros(shape, dtype=np.uint8)
    valid = binary_mask(str(spec["path"]))
    return _resize_binary(valid, shape)


def _physical_allowed(item: dict[str, Any], config: dict[str, Any]) -> bool:
    method = str((item.get("normalization") or {}).get("method") or "")
    allowed = set(config["evidence"]["physical_normalization_methods"])
    unknown_units = {str(value).casefold() for value in config["evidence"]["unknown_units"]}
    units = str(item.get("units") or "unknown").casefold()
    if method not in allowed or units in unknown_units:
        return False
    if item.get("family") == "deformation":
        metadata = item.get("band_metadata") or []
        sign = metadata[0].get("sign_convention") if metadata else None
        if not sign or str(sign).casefold() in {"unknown", "source_defined"}:
            return False
    return True


def _channel_statistics(
    array: np.ndarray, region: np.ndarray, ring: np.ndarray, valid: np.ndarray,
    band_names: list[str], tolerance: float,
) -> list[dict[str, Any]]:
    statistics: list[dict[str, Any]] = []
    region_valid = (region > 0) & (valid > 0)
    ring_valid = (ring > 0) & (valid > 0)
    for index in range(array.shape[0]):
        values = np.asarray(array[index], dtype=np.float64)
        finite = np.isfinite(values)
        region_values = values[region_valid & finite]
        ring_values = values[ring_valid & finite]
        observed = values[(valid > 0) & finite]
        if region_values.size == 0 or ring_values.size == 0 or observed.size == 0:
            continue
        region_median = float(np.median(region_values))
        ring_median = float(np.median(ring_values))
        low, high = np.percentile(observed, [10, 90])
        scale = max(float(high - low), 1.0e-6)
        normalized_delta = float((region_median - ring_median) / scale)
        direction = (
            "higher" if normalized_delta > tolerance else
            "lower" if normalized_delta < -tolerance else
            "similar"
        )
        statistics.append({
            "band_name": str(band_names[index] if index < len(band_names) else f"band_{index}"),
            "region_mean": float(region_values.mean()),
            "region_median": region_median,
            "context_mean": float(ring_values.mean()),
            "context_median": ring_median,
            "normalized_delta": normalized_delta,
            "relative_direction": direction,
        })
    return statistics


def modality_evidence(
    item: dict[str, Any], region_mask: np.ndarray | None, config: dict[str, Any],
) -> dict[str, Any]:
    family = str(item.get("family") or "unknown")
    if region_mask is None or not bool(region_mask.any()):
        return {
            "evidence_level": "C_unavailable", "coverage": 0.0,
            "value_space": "unavailable", "observation": "No target region is available for evidence extraction.",
            "family": family, "sensor": item.get("sensor"), "product_type": item.get("product_type"),
            "channel_statistics": [], "support_assessment": "unavailable",
        }
    array = load_array(str(item["path"]))
    if array.ndim == 2:
        array = array[None]
    if array.ndim != 3:
        raise ValueError(f"模态数组必须是 CxHxW: {item['path']}")
    shape = (int(array.shape[-2]), int(array.shape[-1]))
    region = _resize_binary(region_mask, shape)
    valid = _valid_mask(item, shape)
    target_area = max(int(region.sum()), 1)
    coverage = float(((region > 0) & (valid > 0)).sum() / target_area)
    minimum_coverage = float(config["evidence"]["minimum_region_coverage"])
    if coverage < minimum_coverage:
        return {
            "evidence_level": "C_unavailable", "coverage": coverage,
            "value_space": "unavailable",
            "observation": f"Valid coverage {coverage:.3f} is below the required threshold.",
            "family": family, "sensor": item.get("sensor"), "product_type": item.get("product_type"),
            "channel_statistics": [], "support_assessment": "insufficient_evidence",
        }
    ring = context_ring(region, valid, config)
    if int(ring.sum()) == 0:
        return {
            "evidence_level": "C_unavailable", "coverage": coverage,
            "value_space": "unavailable", "observation": "No valid surrounding context ring is available.",
            "family": family, "sensor": item.get("sensor"), "product_type": item.get("product_type"),
            "channel_statistics": [], "support_assessment": "insufficient_evidence",
        }
    statistics = _channel_statistics(
        array, region, ring, valid, list(item.get("band_names") or []),
        float(config["evidence"]["normalized_relative_tolerance"]),
    )
    if not statistics:
        return {
            "evidence_level": "C_unavailable", "coverage": coverage,
            "value_space": "unavailable", "observation": "Finite region and context statistics are unavailable.",
            "family": family, "sensor": item.get("sensor"), "product_type": item.get("product_type"),
            "channel_statistics": [], "support_assessment": "insufficient_evidence",
        }
    physical = _physical_allowed(item, config)
    lead = max(statistics, key=lambda row: abs(float(row["normalized_delta"])))
    if physical:
        observation = (
            f"{lead['band_name']} region median is {lead['region_median']:.4g} {item['units']}; "
            f"the context median is {lead['context_median']:.4g} {item['units']}."
        )
        level, value_space = "A_physical", "physical"
    else:
        observation = (
            f"In normalized value space, regional {lead['band_name']} is "
            f"{lead['relative_direction']} than the surrounding context."
        )
        level, value_space = "B_normalized_relative", "normalized"
    return {
        "evidence_level": level,
        "coverage": coverage,
        "context_coverage_pixels": int(ring.sum()),
        "value_space": value_space,
        "observation": observation,
        "family": family,
        "sensor": item.get("sensor"),
        "product_type": item.get("product_type"),
        "units": item.get("units") if physical else None,
        "normalization": copy.deepcopy(item.get("normalization")),
        "channel_statistics": statistics,
        "support_assessment": "insufficient_evidence",
    }


def structured_targets(record: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    geometry = record["region_geometry"]
    by_family: dict[str, list[dict[str, Any]]] = {}
    for item in evidence.values():
        by_family.setdefault(str(item["family"]), []).append(item)

    def support(family: str) -> str:
        values = by_family.get(family, [])
        if not values:
            return "unavailable"
        if all(item["evidence_level"] == "C_unavailable" for item in values):
            return "insufficient_evidence"
        return "insufficient_evidence"

    available = [item for item in evidence.values() if item["evidence_level"] != "C_unavailable"]
    sufficiency = "unavailable" if not evidence else "insufficient" if not available else "partial"
    surface = "unavailable"
    optical = by_family.get("optical", []) + by_family.get("multispectral", [])
    if optical:
        surface = next((item["observation"] for item in optical if item["evidence_level"] != "C_unavailable"), "insufficient evidence")
    return {
        "target_status": record["target_status"],
        "region": {
            key: geometry[key]
            for key in ("location", "size_class", "shape", "elongation", "compactness", "fragmentation")
        },
        "evidence": {
            "surface_observation": surface,
            "terrain_support": support("terrain"),
            "sar_support": support("sar"),
            "deformation_support": support("deformation"),
            "surrounding_context": "available" if available else "unavailable",
            "evidence_sufficiency": sufficiency,
        },
        "field_provenance": {
            "region": "deterministic_mask_geometry",
            "evidence": "protocol_constrained_region_context_statistics",
        },
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    source_dir = source_benchmark_dir(args.mode, args.source_benchmark)
    output_dir = bridge_dir(args.mode, args.output_dir)
    inventory = read_jsonl(output_dir / "indexes/region_inventory.jsonl")
    if args.max_samples > 0:
        inventory = inventory[:args.max_samples]
    output_path = output_dir / "indexes/region_facts_all.jsonl"
    report_path = output_dir / "reports/region_facts_report.json"
    for path in (output_path, report_path):
        ensure_writable(path, args.overwrite, args.dry_run)

    facts: list[dict[str, Any]] = []
    for source_record in inventory:
        record = copy.deepcopy(source_record)
        region_mask = (
            binary_mask(str(record["region_mask"]["path"]))
            if record.get("region_mask") else None
        )
        evidence = {
            name: modality_evidence(item, region_mask, config)
            for name, item in record.get("modality_metadata", {}).items()
            if item.get("available", True) and item.get("path")
        }
        record["modality_evidence"] = evidence
        record["structured_targets"] = structured_targets(record, evidence)
        record["provenance"]["fact_extractor"] = BUILDER_VERSION
        facts.append(record)

    levels = Counter(
        item["evidence_level"] for row in facts for item in row["modality_evidence"].values()
    )
    report = {
        "builder_version": BUILDER_VERSION,
        "source_benchmark": to_project_ref(source_dir),
        "records": len(facts),
        "evidence_levels": dict(sorted(levels.items())),
        "records_by_target_status": dict(sorted(Counter(row["target_status"] for row in facts).items())),
        "errors": [],
    }
    print(f"[BRIDGE:FACTS] records={len(facts)} evidence_levels={dict(levels)}")
    if not args.dry_run:
        write_jsonl(output_path, facts)
        write_json(report_path, report)


if __name__ == "__main__":
    main()
