#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Landslide Bridge M2 公共协议、路径、几何、证据和审核统计工具。

运行方式：内部公共模块，不作为独立入口运行。
写入行为：仅由 4-1 到 4-6 显式调用时写入派生 benchmark。
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image
from scipy import ndimage
import yaml


SCHEMA_VERSION = "qpsalm_landslide_region_description_v1"
BUILDER_VERSION = "landslide_bridge_m2_v2_expert_schema"
REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT.parent
DATASETS_ROOT = Path(os.environ.get("PAPER7_DATASETS_ROOT") or WORKSPACE_ROOT / "datasets").resolve(strict=False)
BENCHMARK_ROOT = Path(os.environ.get("PAPER7_BENCHMARK_ROOT") or WORKSPACE_ROOT / "benchmark").resolve(strict=False)

REGION_FIELD_VALUES = {
    "location": {
        "upper_left", "upper_center", "upper_right", "center_left", "center",
        "center_right", "lower_left", "lower_center", "lower_right", "distributed",
        "unknown", "unavailable",
    },
    "size_class": {"tiny", "small", "medium", "large", "extensive", "unknown", "unavailable"},
    "shape": {"compact", "elongated", "branching", "fragmented", "irregular", "unknown", "unavailable"},
    "elongation": {"low", "moderate", "high", "unknown", "unavailable"},
    "compactness": {"compact", "moderate", "dispersed", "unknown", "unavailable"},
    "fragmentation": {
        "single", "few_components", "many_components", "highly_fragmented",
        "unknown", "unavailable",
    },
}
EVIDENCE_SUPPORT_FIELDS = {"terrain_support", "sar_support", "deformation_support"}
EVIDENCE_SUPPORT_VALUES = {
    "supports", "does_not_support", "insufficient_evidence", "unknown", "unavailable",
}
EVIDENCE_SUFFICIENCY_VALUES = {"sufficient", "partial", "insufficient", "unavailable"}


def validate_bridge_structured_target(
    target: Any,
    *,
    expected_target_status: str | None = None,
) -> list[str]:
    """Validate the reviewable subset of qpsalm_description_output_v1."""
    errors: list[str] = []
    if not isinstance(target, dict):
        return ["structured target 必须是 JSON object"]
    status = target.get("target_status")
    if status not in {"present", "absent", "uncertain"}:
        errors.append("target_status 非法或缺失")
    if expected_target_status is not None and status != expected_target_status:
        errors.append(
            f"target_status 不得改变 GT 状态: expected={expected_target_status} actual={status}"
        )
    region = target.get("region")
    if not isinstance(region, dict):
        errors.append("region 必须是 object")
    else:
        for field, allowed in REGION_FIELD_VALUES.items():
            if region.get(field) not in allowed:
                errors.append(f"region.{field} 非法或缺失")
    evidence = target.get("evidence")
    if not isinstance(evidence, dict):
        errors.append("evidence 必须是 object")
    else:
        for field in ("surface_observation", "surrounding_context"):
            if not isinstance(evidence.get(field), str) or not evidence[field].strip():
                errors.append(f"evidence.{field} 必须是非空字符串")
        for field in EVIDENCE_SUPPORT_FIELDS:
            if evidence.get(field) not in EVIDENCE_SUPPORT_VALUES:
                errors.append(f"evidence.{field} 非法或缺失")
        if evidence.get("evidence_sufficiency") not in EVIDENCE_SUFFICIENCY_VALUES:
            errors.append("evidence.evidence_sufficiency 非法或缺失")
    return errors


def flatten_bridge_structured_target(target: dict[str, Any]) -> dict[str, str]:
    """Return ontology fields used for per-field reviewer agreement."""
    region = target.get("region") or {}
    evidence = target.get("evidence") or {}
    result = {"target_status": str(target.get("target_status") or "<missing>")}
    result.update({f"region.{field}": str(region.get(field) or "<missing>") for field in REGION_FIELD_VALUES})
    result.update({
        f"evidence.{field}": str(evidence.get(field) or "<missing>")
        for field in (*sorted(EVIDENCE_SUPPORT_FIELDS), "evidence_sufficiency")
    })
    return result


def resolve_project_path(ref: str | Path) -> Path:
    path = Path(ref).expanduser()
    if path.is_absolute():
        return path.resolve(strict=False)
    if path.parts and path.parts[0] == "benchmark":
        return BENCHMARK_ROOT.joinpath(*path.parts[1:]).resolve(strict=False)
    if path.parts and path.parts[0] == "datasets":
        return DATASETS_ROOT.joinpath(*path.parts[1:]).resolve(strict=False)
    return (REPO_ROOT / path).resolve(strict=False)


def to_project_ref(path: str | Path) -> str:
    source = Path(path)
    if not source.is_absolute():
        return source.as_posix()
    source = source.resolve(strict=False)
    for logical, root in (("benchmark", BENCHMARK_ROOT), ("datasets", DATASETS_ROOT)):
        try:
            return (Path(logical) / source.relative_to(root)).as_posix()
        except ValueError:
            pass
    return source.relative_to(REPO_ROOT).as_posix()


def source_benchmark_dir(mode: str, value: str | Path | None = None) -> Path:
    return resolve_project_path(value) if value else BENCHMARK_ROOT / f"multisource_landslide_v2_{mode}"


def bridge_dir(mode: str, value: str | Path | None = None) -> Path:
    return resolve_project_path(value) if value else BENCHMARK_ROOT / f"landslide_region_description_v1_{mode}"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = resolve_project_path(path or "configs/landslide_bridge_v1.yaml")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if payload.get("version") != "landslide_bridge_v1":
        raise ValueError(f"Bridge config 版本不正确: {config_path}")
    return payload


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: 非法 JSONL: {exc}") from exc
    return rows


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    temporary.replace(path)


def write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    values = list(rows)
    atomic_write_text(
        path,
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in values),
    )
    return len(values)


def ensure_writable(path: Path, overwrite: bool, dry_run: bool) -> None:
    if path.exists() and not overwrite and not dry_run:
        raise FileExistsError(f"输出已存在，请使用 --overwrite: {path}")


def stable_hash(*parts: object) -> str:
    return hashlib.sha256("\x1f".join(str(part) for part in parts).encode("utf-8")).hexdigest()


def stable_id(prefix: str, *parts: object, length: int = 16) -> str:
    safe = re.sub(r"[^a-z0-9]+", "_", prefix.casefold()).strip("_")
    return f"{safe}_{stable_hash(*parts)[:length]}"


def safe_slug(value: str) -> str:
    result = re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
    if not result:
        raise ValueError(f"无法生成 slug: {value!r}")
    return result


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_array(path_ref: str) -> np.ndarray:
    path = resolve_project_path(path_ref)
    if not path.is_file():
        raise FileNotFoundError(f"数组不存在: {path_ref} -> {path}")
    array = np.load(path)
    return np.asarray(array)


def binary_mask(path_ref: str) -> np.ndarray:
    array = load_array(path_ref)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2:
        raise ValueError(f"mask 必须是 HxW 或 1xHxW: {path_ref} shape={array.shape}")
    return (np.nan_to_num(array) > 0).astype(np.uint8)


def atomic_save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".part", delete=False) as handle:
        temporary = Path(handle.name)
        np.save(handle, array)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def modality_family_combo(parent: dict[str, Any]) -> str:
    families = sorted({
        str(item.get("family"))
        for item in parent.get("modalities", {}).values()
        if item.get("available", True)
    })
    return "+".join(families) if families else "none"


def valid_canvas(parent: dict[str, Any], shape: tuple[int, int]) -> np.ndarray:
    spec = parent.get("spatial", {}).get("valid_pixel_mask", {})
    path = spec.get("path") if isinstance(spec, dict) else None
    if not path:
        return np.ones(shape, dtype=np.uint8)
    valid = binary_mask(str(path))
    if valid.shape != shape:
        raise ValueError(f"parent valid canvas shape 不一致: {parent['parent_sample_id']}")
    return valid


def connected_components(mask: np.ndarray, valid: np.ndarray, min_pixels: int, min_fraction: float) -> list[np.ndarray]:
    structure = np.ones((3, 3), dtype=np.uint8)
    labels, count = ndimage.label((mask > 0) & (valid > 0), structure=structure)
    threshold = max(int(min_pixels), int(round(float(valid.sum()) * float(min_fraction))))
    components: list[tuple[int, np.ndarray]] = []
    for label_id in range(1, int(count) + 1):
        component = labels == label_id
        area = int(component.sum())
        if area >= threshold:
            components.append((area, component.astype(np.uint8)))
    components.sort(key=lambda item: (-item[0], sha256_bytes(item[1].tobytes())))
    return [item[1] for item in components]


def size_class(area_ratio: float) -> str:
    if area_ratio < 0.001:
        return "tiny"
    if area_ratio < 0.01:
        return "small"
    if area_ratio < 0.05:
        return "medium"
    if area_ratio < 0.2:
        return "large"
    return "extensive"


def area_bin(area_ratio: float) -> str:
    if area_ratio <= 0:
        return "absent"
    return size_class(area_ratio)


def geometry_from_mask(mask: np.ndarray | None, valid: np.ndarray) -> dict[str, Any]:
    valid_area = int((valid > 0).sum())
    if mask is None or not bool((mask > 0).any()):
        return {
            "area_pixels": 0,
            "valid_area_pixels": valid_area,
            "valid_area_ratio": 0.0,
            "bbox_xyxy_pixel_half_open": None,
            "centroid_xy_normalized": None,
            "location": "unavailable",
            "size_class": "unavailable",
            "shape": "unavailable",
            "elongation": "unavailable",
            "elongation_ratio": None,
            "compactness": "unavailable",
            "compactness_value": None,
            "fragmentation": "unavailable",
            "component_count": 0,
            "orientation_degrees": None,
        }
    binary = (mask > 0) & (valid > 0)
    ys, xs = np.where(binary)
    area = int(xs.size)
    height, width = binary.shape
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    cx, cy = float(xs.mean()), float(ys.mean())
    col = min(2, int(3 * cx / max(width, 1)))
    row = min(2, int(3 * cy / max(height, 1)))
    positions = (
        ("upper_left", "upper_center", "upper_right"),
        ("center_left", "center", "center_right"),
        ("lower_left", "lower_center", "lower_right"),
    )
    coordinates = np.column_stack((xs - cx, ys - cy)).astype(np.float64)
    covariance = np.cov(coordinates, rowvar=False) if area > 2 else np.eye(2)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(eigenvalues, 1.0e-6)
    major_index = int(np.argmax(eigenvalues))
    major, minor = float(eigenvalues[major_index]), float(eigenvalues[1 - major_index])
    elongation_ratio = float(math.sqrt(major / minor))
    vector = eigenvectors[:, major_index]
    orientation = float(math.degrees(math.atan2(float(vector[1]), float(vector[0]))))
    boundary = binary & ~ndimage.binary_erosion(binary, structure=np.ones((3, 3), dtype=bool))
    perimeter = max(float(boundary.sum()), 1.0)
    compactness_value = float(4.0 * math.pi * area / (perimeter * perimeter))
    component_count = int(ndimage.label(binary, structure=np.ones((3, 3), dtype=np.uint8))[1])
    elongation = "low" if elongation_ratio < 1.5 else "moderate" if elongation_ratio < 3.0 else "high"
    compactness = "compact" if compactness_value >= 0.5 else "moderate" if compactness_value >= 0.2 else "dispersed"
    fragmentation = (
        "single" if component_count == 1 else
        "few_components" if component_count <= 3 else
        "many_components" if component_count <= 8 else
        "highly_fragmented"
    )
    shape = (
        "fragmented" if component_count >= 3 else
        "elongated" if elongation_ratio >= 3.0 else
        "compact" if compactness_value >= 0.5 else
        "irregular"
    )
    return {
        "area_pixels": area,
        "valid_area_pixels": valid_area,
        "valid_area_ratio": float(area / max(valid_area, 1)),
        "bbox_xyxy_pixel_half_open": [x1, y1, x2, y2],
        "centroid_xy_normalized": [cx / max(width - 1, 1), cy / max(height - 1, 1)],
        "location": positions[row][col],
        "size_class": size_class(area / max(valid_area, 1)),
        "shape": shape,
        "elongation": elongation,
        "elongation_ratio": elongation_ratio,
        "compactness": compactness,
        "compactness_value": compactness_value,
        "fragmentation": fragmentation,
        "component_count": component_count,
        "orientation_degrees": orientation,
    }


def context_ring(mask: np.ndarray, valid: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    area = max(int((mask > 0).sum()), 1)
    equivalent_radius = math.sqrt(area / math.pi)
    evidence = config["evidence"]
    radius = int(round(equivalent_radius * float(evidence["context_ring_fraction_of_equivalent_radius"])))
    radius = max(int(evidence["context_ring_min_pixels"]), min(int(evidence["context_ring_max_pixels"]), radius))
    dilated = ndimage.binary_dilation(mask > 0, iterations=radius)
    return (dilated & ~(mask > 0) & (valid > 0)).astype(np.uint8)


def mask_digest(mask: np.ndarray) -> str:
    return sha256_bytes(np.ascontiguousarray(mask.astype(np.uint8)).tobytes())


def parent_index_ref(source_dir: Path, split: str) -> str:
    return to_project_ref(source_dir / f"indexes/{split}.jsonl")


def stratified_select(rows: Sequence[dict[str, Any]], limit: int, fields: Sequence[str], seed: int) -> list[dict[str, Any]]:
    if limit <= 0 or len(rows) <= limit:
        return sorted(rows, key=lambda row: stable_hash(seed, row["parent_sample_id"]))
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = tuple(str(row.get(field, "unknown")) for field in fields)
        groups[key].append(row)
    for values in groups.values():
        values.sort(key=lambda row: stable_hash(seed, row["parent_sample_id"]))
    selected: list[dict[str, Any]] = []
    keys = sorted(groups)
    while len(selected) < limit:
        progressed = False
        for key in keys:
            if groups[key] and len(selected) < limit:
                selected.append(groups[key].pop())
                progressed = True
        if not progressed:
            break
    return selected


def cohen_kappa(left: Sequence[str], right: Sequence[str]) -> float | None:
    if len(left) != len(right) or not left:
        return None
    labels = sorted(set(left) | set(right))
    observed = sum(a == b for a, b in zip(left, right)) / len(left)
    expected = sum(
        (left.count(label) / len(left)) * (right.count(label) / len(right))
        for label in labels
    )
    return float((observed - expected) / (1.0 - expected)) if expected < 1.0 else 1.0


def krippendorff_alpha_nominal(ratings: Sequence[Sequence[str | None]]) -> float | None:
    pairs_total = 0
    disagreements = 0
    counts: defaultdict[str, int] = defaultdict(int)
    total_ratings = 0
    for item in ratings:
        observed = [value for value in item if value is not None]
        for value in observed:
            counts[value] += 1
            total_ratings += 1
        for left_index in range(len(observed)):
            for right_index in range(left_index + 1, len(observed)):
                pairs_total += 1
                disagreements += observed[left_index] != observed[right_index]
    if pairs_total == 0 or total_ratings < 2:
        return None
    observed_disagreement = disagreements / pairs_total
    expected_disagreement = 1.0 - sum((count / total_ratings) ** 2 for count in counts.values())
    return float(1.0 - observed_disagreement / expected_disagreement) if expected_disagreement > 0 else 1.0


def preview_image(path_ref: str, size: int = 512) -> Image.Image:
    path = resolve_project_path(path_ref)
    with Image.open(path) as image:
        image.load()
        result = image.convert("RGB")
    result.thumbnail((size, size), Image.Resampling.LANCZOS)
    return result
