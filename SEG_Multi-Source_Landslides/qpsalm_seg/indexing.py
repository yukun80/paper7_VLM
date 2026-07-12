#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""不依赖 torch 的 instruction 索引统计工具。"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .paths import resolve_repo_path

@dataclass
class DatasetStats:
    num_rows: int
    num_usable: int
    skipped_by_reason: dict[str, int]
    by_template: dict[str, int]
    by_raw_combo: dict[str, int]
    by_family_combo: dict[str, int]
    by_sensor_combo: dict[str, int]
    by_product_combo: dict[str, int]
    by_normalization_methods: dict[str, int]
    by_shape: dict[str, dict[str, int]]
    gsd_ranges: dict[str, int]
    quality_flags: dict[str, int]


def iter_jsonl(path: Path, max_rows: int | None = None) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if max_rows is not None and line_no > max_rows:
                break
            text = line.strip()
            if text:
                yield json.loads(text)


def read_jsonl(path: Path, max_rows: int | None = None) -> list[dict[str, Any]]:
    return list(iter_jsonl(path, max_rows=max_rows))


def available_modality_names(row: dict[str, Any]) -> list[str]:
    modalities = row.get("modalities") or {}
    return sorted(name for name, item in modalities.items() if isinstance(item, dict) and item.get("available", True))


def raw_modality_combo(row: dict[str, Any]) -> str:
    names = available_modality_names(row)
    return "+".join(names) if names else "none"


def family_combo(row: dict[str, Any]) -> str:
    modalities = row.get("modalities") or {}
    names = sorted(
        {
            str(item.get("family"))
            for item in modalities.values()
            if isinstance(item, dict)
            and item.get("available", True)
            and item.get("family")
        }
    )
    return "+".join(names) if names else "none"


def metadata_combo(row: dict[str, Any], field: str, fallback: str = "unknown") -> str:
    values = []
    for name in available_modality_names(row):
        item = (row.get("modalities") or {}).get(name) or {}
        value = item.get(field)
        values.append(str(value if value not in (None, "") else fallback))
    return "+".join(sorted(values)) if values else "none"


def sensor_combo(row: dict[str, Any]) -> str:
    return metadata_combo(row, "sensor")


def product_combo(row: dict[str, Any]) -> str:
    return metadata_combo(row, "product_type")


def normalization_methods(row: dict[str, Any]) -> str:
    methods = []
    for name in available_modality_names(row):
        normalization = ((row.get("modalities") or {}).get(name) or {}).get("normalization") or {}
        methods.append(str(normalization.get("method") or "unknown"))
    return "+".join(sorted(methods)) if methods else "none"


def gsd_range(gsd: Any) -> str:
    if gsd is None or gsd == "" or str(gsd).lower() == "none":
        return "unknown"
    try:
        value = float(gsd)
    except (TypeError, ValueError):
        return "unknown"
    if not math.isfinite(value) or value <= 0:
        return "unknown"
    if value <= 1.0:
        return "sub_meter"
    if value <= 5.0:
        return "meter_1_5"
    if value <= 10.0:
        return "meter_5_10"
    return "meter_gt_10"


def row_template_id(row: dict[str, Any]) -> str:
    return str(row.get("template_id") or row.get("task_template_id") or "")


def should_skip_row(row: dict[str, Any], task_families: Iterable[str]) -> str | None:
    if row.get("task_family") not in set(task_families):
        return "unsupported_task_family"
    if row.get("source_level") != "patch":
        return "scene_level_deferred"
    flags = set(row.get("quality_flags") or [])
    if "requires_tiling_for_patch_training" in flags or "scene_level_large_image" in flags:
        return "requires_tiling_deferred"
    if not isinstance(row.get("mask"), dict):
        return "missing_mask"
    if not available_modality_names(row):
        return "missing_modalities"
    return None


def summarize_rows(rows: Iterable[dict[str, Any]], task_families: Iterable[str]) -> DatasetStats:
    skipped = Counter()
    by_template = Counter()
    by_raw_combo = Counter()
    by_family_combo = Counter()
    by_sensor_combo = Counter()
    by_product_combo = Counter()
    by_normalization_methods = Counter()
    by_shape: dict[str, Counter[str]] = defaultdict(Counter)
    gsd_counter = Counter()
    quality_flags = Counter()
    num_rows = 0
    num_usable = 0
    for row in rows:
        num_rows += 1
        reason = should_skip_row(row, task_families)
        if reason is not None:
            skipped[reason] += 1
            continue
        num_usable += 1
        by_template[row_template_id(row) or "unknown"] += 1
        by_raw_combo[raw_modality_combo(row)] += 1
        by_family_combo[family_combo(row)] += 1
        by_sensor_combo[sensor_combo(row)] += 1
        by_product_combo[product_combo(row)] += 1
        by_normalization_methods[normalization_methods(row)] += 1
        spatial = row.get("spatial") or {}
        gsd_counter[gsd_range(spatial.get("gsd_m"))] += 1
        quality_flags.update(str(flag) for flag in row.get("quality_flags") or [])
        for name, item in (row.get("modalities") or {}).items():
            if not isinstance(item, dict) or not item.get("available", True):
                continue
            by_shape[name][str(item.get("shape"))] += 1
    return DatasetStats(
        num_rows=num_rows,
        num_usable=num_usable,
        skipped_by_reason=dict(sorted(skipped.items())),
        by_template=dict(sorted(by_template.items())),
        by_raw_combo=dict(sorted(by_raw_combo.items())),
        by_family_combo=dict(sorted(by_family_combo.items())),
        by_sensor_combo=dict(sorted(by_sensor_combo.items())),
        by_product_combo=dict(sorted(by_product_combo.items())),
        by_normalization_methods=dict(sorted(by_normalization_methods.items())),
        by_shape={key: dict(sorted(value.items())) for key, value in sorted(by_shape.items())},
        gsd_ranges=dict(sorted(gsd_counter.items())),
        quality_flags=dict(sorted(quality_flags.items())),
    )


def stats_to_text(stats: DatasetStats, limit: int | None = None) -> str:
    lines = [
        f"rows={stats.num_rows}",
        f"usable_core_rows={stats.num_usable}",
        f"skipped={stats.skipped_by_reason}",
        "",
        "templates:",
    ]
    for key, value in Counter(stats.by_template).most_common(limit):
        lines.append(f"  {key}: {value}")
    lines.append("raw modality combos:")
    for key, value in Counter(stats.by_raw_combo).most_common(limit):
        lines.append(f"  {key}: {value}")
    lines.append("modality family combos:")
    for key, value in Counter(stats.by_family_combo).most_common(limit):
        lines.append(f"  {key}: {value}")
    lines.append("sensor combos:")
    for key, value in Counter(stats.by_sensor_combo).most_common(limit):
        lines.append(f"  {key}: {value}")
    lines.append("product combos:")
    for key, value in Counter(stats.by_product_combo).most_common(limit):
        lines.append(f"  {key}: {value}")
    lines.append("normalization methods:")
    for key, value in Counter(stats.by_normalization_methods).most_common(limit):
        lines.append(f"  {key}: {value}")
    lines.append(f"gsd_ranges: {stats.gsd_ranges}")
    lines.append(f"quality_flags: {dict(Counter(stats.quality_flags).most_common(limit))}")
    lines.append("shapes:")
    for name, shape_counts in stats.by_shape.items():
        top = Counter(shape_counts).most_common(limit)
        joined = ", ".join(f"{shape}:{count}" for shape, count in top)
        lines.append(f"  {name}: {joined}")
    return "\n".join(lines)
