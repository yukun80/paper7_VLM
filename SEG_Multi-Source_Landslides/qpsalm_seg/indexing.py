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

MODALITY_TO_CANONICAL = {
    "optical_rgb": "hr_optical",
    "optical_multiband": "hr_optical",
    "multispectral": "s2",
    "sar_asc": "s1",
    "sar_dsc": "s1",
    "dem": "dem",
    "slope": "dem",
    "insar_vel": "insar",
}


@dataclass
class DatasetStats:
    num_rows: int
    num_usable: int
    skipped_by_reason: dict[str, int]
    by_template: dict[str, int]
    by_raw_combo: dict[str, int]
    by_canonical_combo: dict[str, int]
    by_sensor_combo: dict[str, int]
    by_normalization_combo: dict[str, int]
    by_shape: dict[str, dict[str, int]]
    gsd_tokens: dict[str, int]
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


def canonical_modality_name(raw_name: str, item: dict[str, Any] | None = None) -> str | None:
    item = item or {}
    sensor = str(item.get("sensor") or "").lower()
    role = str(item.get("role") or "").lower()
    source = str(item.get("source") or "").lower()
    value_encoding = str(item.get("value_encoding") or "").lower()
    if raw_name == "optical_rgb" and (
        sensor == "sentinel2"
        or "sentinel-2" in source
        or "sentinel2" in source
        or "sentinel2" in role
        or "sentinel2" in value_encoding
    ):
        return "s2"
    return MODALITY_TO_CANONICAL.get(raw_name)


def canonical_modality_combo(row: dict[str, Any]) -> str:
    modalities = row.get("modalities") or {}
    names = sorted(
        {
            canonical
            for name, item in modalities.items()
            if isinstance(item, dict)
            and item.get("available", True)
            and (canonical := canonical_modality_name(name, item)) is not None
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


def normalization_combo(row: dict[str, Any]) -> str:
    return metadata_combo(row, "normalization")


def gsd_to_token(gsd: Any) -> str:
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


def should_skip_row(row: dict[str, Any], core_templates: Iterable[str]) -> str | None:
    tid = row_template_id(row)
    if tid not in set(core_templates):
        return "non_core_template"
    if row.get("task_family") == "referring_landslide_segmentation":
        return "referring_deferred"
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


def summarize_rows(rows: Iterable[dict[str, Any]], core_templates: Iterable[str]) -> DatasetStats:
    skipped = Counter()
    by_template = Counter()
    by_raw_combo = Counter()
    by_canonical_combo = Counter()
    by_sensor_combo = Counter()
    by_normalization_combo = Counter()
    by_shape: dict[str, Counter[str]] = defaultdict(Counter)
    gsd_counter = Counter()
    quality_flags = Counter()
    num_rows = 0
    num_usable = 0
    for row in rows:
        num_rows += 1
        reason = should_skip_row(row, core_templates)
        if reason is not None:
            skipped[reason] += 1
            continue
        num_usable += 1
        by_template[row_template_id(row) or "unknown"] += 1
        by_raw_combo[raw_modality_combo(row)] += 1
        by_canonical_combo[canonical_modality_combo(row)] += 1
        by_sensor_combo[sensor_combo(row)] += 1
        by_normalization_combo[normalization_combo(row)] += 1
        spatial = row.get("spatial") or {}
        gsd_counter[gsd_to_token(spatial.get("gsd_m"))] += 1
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
        by_canonical_combo=dict(sorted(by_canonical_combo.items())),
        by_sensor_combo=dict(sorted(by_sensor_combo.items())),
        by_normalization_combo=dict(sorted(by_normalization_combo.items())),
        by_shape={key: dict(sorted(value.items())) for key, value in sorted(by_shape.items())},
        gsd_tokens=dict(sorted(gsd_counter.items())),
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
    lines.append("canonical modality combos:")
    for key, value in Counter(stats.by_canonical_combo).most_common(limit):
        lines.append(f"  {key}: {value}")
    lines.append("sensor combos:")
    for key, value in Counter(stats.by_sensor_combo).most_common(limit):
        lines.append(f"  {key}: {value}")
    lines.append("normalization combos:")
    for key, value in Counter(stats.by_normalization_combo).most_common(limit):
        lines.append(f"  {key}: {value}")
    lines.append(f"gsd_tokens: {stats.gsd_tokens}")
    lines.append(f"quality_flags: {dict(Counter(stats.quality_flags).most_common(limit))}")
    lines.append("shapes:")
    for name, shape_counts in stats.by_shape.items():
        top = Counter(shape_counts).most_common(limit)
        joined = ", ".join(f"{shape}:{count}" for shape, count in top)
        lines.append(f"  {name}: {joined}")
    return "\n".join(lines)
