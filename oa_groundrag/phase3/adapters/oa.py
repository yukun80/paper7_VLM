"""现有 OA-AuxSeg Benchmark 到 OA train/gold canonical 的只读 adapter。"""

from __future__ import annotations

import io
import json
import math
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from PIL import Image
from scipy import ndimage

from scripts.phase1_benchmark_build.benchmark_common import BenchmarkDataset

from ..common import (
    canonical_json,
    read_json,
    read_jsonl,
    sha256_bytes,
    sha256_file,
    stable_hash,
)
from ..contracts import (
    AdapterResult,
    AnnotationLayer,
    InputLayout,
    LogicalRole,
    MediaType,
    OutputModality,
    PendingAsset,
    ReviewStatus,
    SourceExample,
    SupervisionKind,
    TaskFamily,
)
from ..errors import ReasonCode, SchemaError, SourceAuditError
from .base import summarize_result


GOLD_SCHEMA_VERSION = "oa_landslidedesc.gold.v1"
SILVER_SCHEMA_VERSION = "oa_landslidedesc.silver.v1"


def _validate_benchmark_root(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = read_json(root / "manifest.json")
    rows = read_jsonl(root / "index.jsonl")
    if manifest.get("schema_version") != "oa_auxseg_hdf5_v1":
        raise SourceAuditError(
            ReasonCode.SCHEMA_MISMATCH,
            f"{root}: OA Benchmark schema 不受支持",
        )
    index_hash = sha256_file(root / "index.jsonl")
    if manifest.get("index_sha256") != index_hash:
        raise SourceAuditError(
            ReasonCode.HASH_MISMATCH,
            f"{root}: manifest/index SHA-256 不一致",
        )
    if int(manifest.get("sample_count", -1)) != len(rows):
        raise SourceAuditError(
            ReasonCode.SCHEMA_MISMATCH,
            f"{root}: manifest sample_count 与 index 不一致",
        )
    return manifest, rows


def _parent_key(row: Mapping[str, Any]) -> tuple[str, tuple[str, ...]]:
    group = row.get("source_group_id")
    status = row.get("group_status")
    if isinstance(group, str) and group and status == "known":
        return f"group:{row['source']}:{group}", ()
    return f"sample:{row['sample_id']}", ("parent_identity_unknown",)


def _deterministic_npz(arrays: Mapping[str, np.ndarray]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(
        output,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as archive:
        for name in sorted(arrays):
            array_buffer = io.BytesIO()
            np.lib.format.write_array(
                array_buffer,
                np.ascontiguousarray(arrays[name]),
                allow_pickle=False,
            )
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, array_buffer.getvalue())
    return output.getvalue()


def _png_bytes(array: np.ndarray, *, mode: str) -> bytes:
    output = io.BytesIO()
    image = Image.fromarray(array)
    if image.mode != mode:
        image = image.convert(mode)
    image.save(
        output,
        format="PNG",
        optimize=False,
        compress_level=9,
    )
    return output.getvalue()


def _rgb_indices(names: list[str]) -> tuple[int, int, int, str]:
    name_to_index = {name: index for index, name in enumerate(names)}
    if all(name in name_to_index for name in ("Red", "Green", "Blue")):
        return (
            name_to_index["Red"],
            name_to_index["Green"],
            name_to_index["Blue"],
            "named_rgb",
        )
    if all(name in name_to_index for name in ("B04", "B03", "B02")):
        return (
            name_to_index["B04"],
            name_to_index["B03"],
            name_to_index["B02"],
            "sentinel2_b04_b03_b02",
        )
    if len(names) >= 3:
        return 0, 1, 2, "first_three_channels"
    raise SourceAuditError(
        ReasonCode.SCHEMA_MISMATCH,
        f"OA optical 通道不足 3：{names}",
    )


def _preview(
    sample: Mapping[str, Any],
    row: Mapping[str, Any],
    statistics: Mapping[str, Any],
    clip: tuple[float, float],
) -> tuple[np.ndarray, str]:
    optical = np.asarray(sample["optical"], dtype=np.float32)
    valid = np.asarray(sample["optical_pixel_valid"], dtype=np.uint8)
    names = [str(value) for value in row["optical"]["channel_names"]]
    r, g, b, renderer = _rgb_indices(names)
    indices = (r, g, b)
    stats = statistics.get("sources", {}).get(row["source"], {}).get("optical", [])
    by_index = {
        int(item["index"]): item
        for item in stats
        if isinstance(item, dict) and "index" in item
    }
    channels: list[np.ndarray] = []
    low, high = clip
    for index in indices:
        values = optical[index].copy()
        channel_valid = valid[index].astype(bool) & np.isfinite(values)
        item = by_index.get(index)
        if item is not None and float(item.get("std", 0.0)) > 0:
            values = (values - float(item["mean"])) / float(item["std"])
        else:
            selected = values[channel_valid]
            if selected.size:
                mean = float(selected.mean())
                std = float(selected.std())
                values = (values - mean) / std if std > 0 else np.zeros_like(values)
            else:
                values = np.zeros_like(values)
        values = np.clip((values - low) / (high - low), 0.0, 1.0)
        values[~channel_valid] = 0.0
        channels.append(np.rint(values * 255.0).astype(np.uint8))
    return np.stack(channels, axis=-1), renderer


def _round(value: float) -> float:
    return float(round(value, 10))


def mask_facts(mask: np.ndarray) -> dict[str, Any]:
    value = np.asarray(mask)
    if value.ndim == 3 and value.shape[0] == 1:
        value = value[0]
    if value.ndim != 2:
        raise SchemaError(ReasonCode.TYPE_MISMATCH, f"OA mask shape 非法：{value.shape}")
    unique = set(np.unique(value).tolist())
    if not unique.issubset({0, 1}):
        raise SchemaError(
            ReasonCode.MASK_INVALID_VALUES,
            f"OA mask 值非法：{sorted(unique)[:20]}",
        )
    binary = value.astype(bool)
    height, width = binary.shape
    foreground = int(binary.sum())
    if foreground == 0:
        return {
            "has_target": False,
            "component_count": 0,
            "area_pixels": 0,
            "area_fraction": 0.0,
            "union_bbox_xyxy_norm": None,
            "centroid_xy_norm": None,
            "coarse_position": None,
            "components": [],
            "connectivity": 8,
            "min_component_pixels": 1,
        }
    structure = np.ones((3, 3), dtype=np.uint8)
    labels, count = ndimage.label(binary, structure=structure)
    component_rows: list[dict[str, Any]] = []
    union_x0 = width
    union_y0 = height
    union_x1 = 0
    union_y1 = 0
    for label_index in range(1, count + 1):
        component = labels == label_index
        coordinates = np.argwhere(component)
        area = int(coordinates.shape[0])
        y0, x0 = coordinates.min(axis=0)
        y1, x1 = coordinates.max(axis=0)
        union_x0 = min(union_x0, int(x0))
        union_y0 = min(union_y0, int(y0))
        union_x1 = max(union_x1, int(x1) + 1)
        union_y1 = max(union_y1, int(y1) + 1)
        centroid_y, centroid_x = coordinates.mean(axis=0)
        boundary = component & ~ndimage.binary_erosion(component, structure=structure)
        perimeter = int(boundary.sum())
        compactness = (
            _round(4.0 * math.pi * area / (perimeter * perimeter))
            if perimeter > 0
            else None
        )
        centered = coordinates.astype(np.float64) - coordinates.mean(axis=0)
        if area >= 2:
            covariance = centered.T @ centered / area
            eigenvalues = np.linalg.eigvalsh(covariance)
            elongation = (
                _round(math.sqrt(float(eigenvalues[-1] / eigenvalues[0])))
                if eigenvalues[0] > 0
                else None
            )
        else:
            elongation = None
        component_rows.append(
            {
                "component_index": label_index - 1,
                "area_pixels": area,
                "area_fraction": _round(area / (width * height)),
                "bbox_xyxy_norm": [
                    _round(x0 / width),
                    _round(y0 / height),
                    _round((x1 + 1) / width),
                    _round((y1 + 1) / height),
                ],
                "centroid_xy_norm": [
                    _round((centroid_x + 0.5) / width),
                    _round((centroid_y + 0.5) / height),
                ],
                "perimeter_pixels": perimeter,
                "compactness": compactness,
                "elongation": elongation,
            }
        )
    total_coordinates = np.argwhere(binary)
    centroid_y, centroid_x = total_coordinates.mean(axis=0)
    x_norm = (float(centroid_x) + 0.5) / width
    y_norm = (float(centroid_y) + 0.5) / height
    horizontal = "left" if x_norm < 1 / 3 else "right" if x_norm >= 2 / 3 else "center"
    vertical = "top" if y_norm < 1 / 3 else "bottom" if y_norm >= 2 / 3 else "middle"
    return {
        "has_target": True,
        "component_count": int(count),
        "area_pixels": foreground,
        "area_fraction": _round(foreground / (width * height)),
        "union_bbox_xyxy_norm": [
            _round(union_x0 / width),
            _round(union_y0 / height),
            _round(union_x1 / width),
            _round(union_y1 / height),
        ],
        "centroid_xy_norm": [_round(x_norm), _round(y_norm)],
        "coarse_position": f"{vertical}_{horizontal}",
        "components": component_rows,
        "connectivity": 8,
        "min_component_pixels": 1,
    }


def _auto_response(facts: Mapping[str, Any]) -> str:
    if not facts["has_target"]:
        return "No landslide target is present in the provided mask."
    percentage = float(facts["area_fraction"]) * 100.0
    return (
        f"The mask contains {facts['component_count']} connected landslide region(s), "
        f"covering {percentage:.4f}% of the image, with its combined centroid in the "
        f"{str(facts['coarse_position']).replace('_', ' ')} area."
    )


def _sample_assets(
    sample: Mapping[str, Any],
    row: Mapping[str, Any],
    dataset: BenchmarkDataset,
    clip: tuple[float, float],
) -> tuple[tuple[PendingAsset, ...], dict[str, Any], tuple[str, ...]]:
    optical = np.asarray(sample["optical"], dtype=np.float32)
    optical_valid = np.asarray(sample["optical_pixel_valid"], dtype=np.uint8)
    optical_channel_valid = np.asarray(
        sample["optical_channel_valid"], dtype=np.uint8
    )
    mask = np.asarray(sample["mask"], dtype=np.uint8)
    arrays: dict[str, np.ndarray] = {
        "optical": optical,
        "optical_pixel_valid": optical_valid,
        "optical_channel_valid": optical_channel_valid,
        "mask": mask,
    }
    auxiliaries = sample["auxiliaries"]
    for name in sorted(auxiliaries):
        value = auxiliaries[name]
        arrays[f"aux__{name}__values"] = np.asarray(value["values"], dtype=np.float32)
        arrays[f"aux__{name}__pixel_valid"] = np.asarray(
            value["pixel_valid"], dtype=np.uint8
        )
        arrays[f"aux__{name}__channel_valid"] = np.asarray(
            value["channel_valid"], dtype=np.uint8
        )
    preview, renderer = _preview(sample, row, dataset.statistics, clip)
    mask_2d = mask[0] if mask.ndim == 3 else mask
    assets = (
        PendingAsset(
            role="image",
            media_type=MediaType.IMAGE,
            extension="png",
            source_ref=f"oa/{row['sample_id']}/preview.png",
            payload=_png_bytes(preview, mode="RGB"),
        ),
        PendingAsset(
            role="mask",
            media_type=MediaType.MASK,
            extension="png",
            source_ref=f"oa/{row['sample_id']}/mask.png",
            payload=_png_bytes((mask_2d * 255).astype(np.uint8), mode="L"),
            align_to_role="image",
            allow_empty=True,
        ),
        PendingAsset(
            role="oa_tensor",
            media_type=MediaType.ARRAY,
            extension="npz",
            source_ref=f"oa/{row['sample_id']}/sample.npz",
            payload=_deterministic_npz(arrays),
        ),
    )
    return assets, mask_facts(mask), (f"oa_renderer:{renderer}",)


def _selection_id(index_hash: str, split: str, sample_id: str) -> str:
    return f"selection_{sha256_bytes(canonical_json([index_hash, split, sample_id]).encode())}"


def _provenance_optical_identity(row: Mapping[str, Any]) -> str | None:
    provenance = row.get("provenance")
    if not isinstance(provenance, dict):
        return None
    for key in ("source_image_slice_sha256", "source_image_sha256"):
        value = provenance.get(key)
        if (
            isinstance(value, str)
            and len(value) == 64
            and all(character in "0123456789abcdef" for character in value)
        ):
            return f"{key}:{value}"
    return None


def _canonical_optical_identity(
    sample: Mapping[str, Any],
    row: Mapping[str, Any],
) -> str:
    optical = np.ascontiguousarray(
        np.asarray(sample["optical"], dtype=np.float32)
    )
    pixel_valid = np.ascontiguousarray(
        np.asarray(sample["optical_pixel_valid"], dtype=np.uint8)
    )
    channel_valid = np.ascontiguousarray(
        np.asarray(sample["optical_channel_valid"], dtype=np.uint8)
    )
    header = canonical_json(
        {
            "shape": list(optical.shape),
            "channel_names": row["optical"]["channel_names"],
            "pixel_valid_shape": list(pixel_valid.shape),
            "channel_valid_shape": list(channel_valid.shape),
        }
    ).encode("utf-8")
    return "canonical_optical_sha256:" + sha256_bytes(
        header
        + optical.tobytes()
        + pixel_valid.tobytes()
        + channel_valid.tobytes()
    )


def select_gold_candidates(
    full_root: Path,
    selection_root: Path,
    *,
    seed: int,
    audit: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """锁定 158 train + 100 val + 100 test 的身份，不生成或伪造真值。"""

    _, full_rows = _validate_benchmark_root(full_root)
    selection_manifest, selection_rows = _validate_benchmark_root(selection_root)
    full_group_splits: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in full_rows:
        group = row.get("source_group_id")
        if row.get("group_status") == "known" and isinstance(group, str) and group:
            full_group_splits[(str(row["source"]), group)].add(str(row["split"]))

    selection_dataset = BenchmarkDataset(
        selection_root,
        split=None,
        auxiliary_policy="none",
        normalization="none",
    )
    dataset_position = {
        str(row["sample_id"]): index
        for index, row in enumerate(selection_dataset.rows)
    }
    derived: dict[str, dict[str, Any]] = {}
    for row in selection_rows:
        sample_id = str(row["sample_id"])
        sample = selection_dataset[dataset_position[sample_id]]
        mask = np.asarray(sample["mask"], dtype=np.uint8)
        content_identity = (
            _provenance_optical_identity(row)
            or _canonical_optical_identity(sample, row)
        )
        binary = mask[0] if mask.ndim == 3 else mask
        _, region_count = ndimage.label(
            binary.astype(bool),
            structure=np.ones((3, 3), dtype=np.uint8),
        )
        derived[sample_id] = {
            "content_identity": content_identity,
            "region_count": int(region_count),
            "aux_signature": ",".join(sorted(row.get("auxiliaries", {}))) or "none",
        }

    full_content_splits: dict[str, set[str]] = defaultdict(set)
    fallback_full_rows = [
        row
        for row in full_rows
        if row.get("group_status") != "known"
        and _provenance_optical_identity(row) is None
    ]
    fallback_identities: dict[str, str] = {}
    if fallback_full_rows:
        full_dataset = BenchmarkDataset(
            full_root,
            split=None,
            auxiliary_policy="none",
            normalization="none",
        )
        full_positions = {
            str(row["sample_id"]): index
            for index, row in enumerate(full_dataset.rows)
        }
        for row in fallback_full_rows:
            sample_id = str(row["sample_id"])
            fallback_identities[sample_id] = _canonical_optical_identity(
                full_dataset[full_positions[sample_id]],
                row,
            )
    for row in full_rows:
        if row.get("group_status") == "known":
            continue
        sample_id = str(row["sample_id"])
        identity = _provenance_optical_identity(row)
        if identity is None:
            identity = fallback_identities[sample_id]
        full_content_splits[identity].add(str(row["split"]))
    if audit is not None:
        audit.update(
            {
                "full_unknown_parent_row_count": sum(
                    row.get("group_status") != "known" for row in full_rows
                ),
                "full_content_fallback_decoded_count": len(fallback_full_rows),
            }
        )

    foreground_bins: dict[str, int] = {}
    for split in ("train", "val", "test"):
        positive = sorted(
            float(row["foreground_ratio"])
            for row in selection_rows
            if row["split"] == split and float(row["foreground_ratio"]) > 0
        )
        if positive:
            q1, q2, q3 = (
                float(value)
                for value in np.quantile(
                    np.asarray(positive, dtype=np.float64),
                    [0.25, 0.5, 0.75],
                )
            )
        else:
            q1 = q2 = q3 = 0.0
        for row in selection_rows:
            if row["split"] != split:
                continue
            ratio = float(row["foreground_ratio"])
            if ratio == 0:
                foreground_bins[str(row["sample_id"])] = 0
            elif ratio <= q1:
                foreground_bins[str(row["sample_id"])] = 1
            elif ratio <= q2:
                foreground_bins[str(row["sample_id"])] = 2
            elif ratio <= q3:
                foreground_bins[str(row["sample_id"])] = 3
            else:
                foreground_bins[str(row["sample_id"])] = 4

    def strata(row: Mapping[str, Any]) -> dict[str, Any]:
        sample_id = str(row["sample_id"])
        region_count = int(derived[sample_id]["region_count"])
        return {
            "mask_status": (
                "positive" if float(row["foreground_ratio"]) > 0 else "empty"
            ),
            "region_count_bin": (
                "0"
                if region_count == 0
                else "1"
                if region_count == 1
                else "2"
                if region_count == 2
                else "3_plus"
            ),
            "foreground_ratio_quantile_bin": foreground_bins[sample_id],
            "aux_signature": derived[sample_id]["aux_signature"],
        }

    def selection_row(row: Mapping[str, Any], split: str) -> dict[str, Any]:
        sample_id = str(row["sample_id"])
        return {
            "selection_id": _selection_id(
                selection_manifest["index_sha256"], split, sample_id
            ),
            "sample_id": sample_id,
            "split": split,
            "source": row["source"],
            "upstream_index_sha256": selection_manifest["index_sha256"],
            "strata": strata(row),
            "stable_rank": stable_hash(seed, split, row["source"], sample_id),
            "parent_identity_status": (
                "known_group"
                if row.get("group_status") == "known"
                and isinstance(row.get("source_group_id"), str)
                and row.get("source_group_id")
                else "unknown_group_exact_content_checked"
            ),
        }

    def coverage_pick(
        rows: list[dict[str, Any]],
        *,
        split: str,
        source: str,
        count: int,
    ) -> list[dict[str, Any]]:
        buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            item = strata(row)
            key = (
                item["mask_status"],
                item["region_count_bin"],
                item["foreground_ratio_quantile_bin"],
                item["aux_signature"],
            )
            buckets[key].append(row)
        for key in buckets:
            buckets[key].sort(
                key=lambda row: (
                    stable_hash(seed, split, source, *key, row["sample_id"]),
                    row["sample_id"],
                )
            )
        positions = {key: 0 for key in buckets}
        chosen: list[dict[str, Any]] = []
        keys = sorted(buckets, key=lambda key: canonical_json(key))
        while len(chosen) < count:
            progressed = False
            for key in keys:
                position = positions[key]
                if position >= len(buckets[key]):
                    continue
                chosen.append(buckets[key][position])
                positions[key] += 1
                progressed = True
                if len(chosen) >= count:
                    break
            if not progressed:
                break
        return chosen

    selected: list[dict[str, Any]] = []
    train_rows = [row for row in selection_rows if row["split"] == "train"]
    if len(train_rows) != 158:
        raise SourceAuditError(
            ReasonCode.OA_REVIEW_INCOMPLETE,
            f"锁定 small train 应为 158，实际 {len(train_rows)}",
        )
    for row in sorted(train_rows, key=lambda item: item["sample_id"]):
        selected.append(selection_row(row, "train"))

    already_parents: set[str] = set()
    rejected_known_cross_split = 0
    rejected_content_cross_split = 0
    rejected_repeated_parent = 0
    safe_candidate_counts: dict[str, int] = {}
    for split in ("val", "test"):
        safe: list[dict[str, Any]] = []
        for row in selection_rows:
            if row["split"] != split:
                continue
            group = row.get("source_group_id")
            if row.get("group_status") == "known" and isinstance(group, str) and group:
                if len(full_group_splits[(str(row["source"]), group)]) > 1:
                    rejected_known_cross_split += 1
                    continue
                identity = f"group:{row['source']}:{group}"
            else:
                content_identity = derived[str(row["sample_id"])][
                    "content_identity"
                ]
                if len(full_content_splits[content_identity]) > 1:
                    rejected_content_cross_split += 1
                    continue
                identity = f"sample:{row['sample_id']}"
            if identity in already_parents:
                rejected_repeated_parent += 1
                continue
            copy = dict(row)
            copy["_selection_parent"] = identity
            safe.append(copy)
        safe_candidate_counts[split] = len(safe)
        by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in safe:
            by_source[str(row["source"])].append(row)
        total = len(safe)
        if total < 100:
            raise SourceAuditError(
                ReasonCode.OA_REVIEW_INCOMPLETE,
                f"{split}: 安全候选不足 100，实际 {total}",
            )
        raw_quotas = {
            source: 100 * len(rows) / total for source, rows in by_source.items()
        }
        quotas = {source: int(math.floor(value)) for source, value in raw_quotas.items()}
        remainder = 100 - sum(quotas.values())
        order = sorted(
            raw_quotas,
            key=lambda source: (
                -(raw_quotas[source] - quotas[source]),
                source,
            ),
        )
        for source in order[:remainder]:
            quotas[source] += 1
        chosen: list[dict[str, Any]] = []
        for source in sorted(by_source):
            chosen.extend(
                coverage_pick(
                    by_source[source],
                    split=split,
                    source=source,
                    count=quotas[source],
                )
            )
        if len(chosen) < 100:
            chosen_ids = {row["sample_id"] for row in chosen}
            remaining = sorted(
                (row for row in safe if row["sample_id"] not in chosen_ids),
                key=lambda row: (stable_hash(seed, split, row["sample_id"]), row["sample_id"]),
            )
            chosen.extend(remaining[: 100 - len(chosen)])
        chosen = chosen[:100]
        for row in chosen:
            already_parents.add(str(row["_selection_parent"]))
            selected.append(selection_row(row, split))
    if len(selected) != 358:
        raise SourceAuditError(
            ReasonCode.OA_REVIEW_INCOMPLETE,
            f"gold selection 应为 358，实际 {len(selected)}",
        )
    if audit is not None:
        audit.update(
            {
                "safe_candidate_counts": safe_candidate_counts,
                "rejected_known_parent_cross_split": rejected_known_cross_split,
                "rejected_exact_content_cross_split": (
                    rejected_content_cross_split
                ),
                "rejected_repeated_parent": rejected_repeated_parent,
            }
        )
    return selected


def _strict_annotation_rows(
    path: Path,
    *,
    schema_version: str,
    required: set[str],
) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    for line_number, row in enumerate(rows, 1):
        if set(row) != required:
            raise SchemaError(
                ReasonCode.SCHEMA_MISMATCH,
                f"{path}:{line_number}: 字段错误",
            )
        if row["schema_version"] != schema_version:
            raise SchemaError(
                ReasonCode.SCHEMA_MISMATCH,
                f"{path}:{line_number}: schema_version 错误",
            )
        location = f"{path}:{line_number}"
        if schema_version == GOLD_SCHEMA_VERSION:
            string_fields = (
                "selection_id",
                "sample_id",
                "upstream_index_sha256",
                "description",
                "review_status",
                "reviewer_id",
                "reviewed_at",
            )
            if any(
                not isinstance(row[field], str) or not row[field].strip()
                for field in string_fields
            ):
                raise SchemaError(
                    ReasonCode.TYPE_MISMATCH,
                    f"{location}: gold 字符串字段非法",
                )
            if (
                not str(row["selection_id"]).startswith("selection_")
                or len(str(row["selection_id"])) != len("selection_") + 64
                or any(
                    character not in "0123456789abcdef"
                    for character in str(row["selection_id"])[len("selection_") :]
                )
                or len(str(row["upstream_index_sha256"])) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in str(row["upstream_index_sha256"])
                )
                or row["review_status"] not in {"approved", "rejected"}
                or not isinstance(row["facts"], dict)
            ):
                raise SchemaError(
                    ReasonCode.SCHEMA_MISMATCH,
                    f"{location}: gold identity/facts/review 合同非法",
                )
        elif schema_version == SILVER_SCHEMA_VERSION:
            if (
                any(
                    not isinstance(row[field], str) or not row[field].strip()
                    for field in ("sample_id", "description", "filter_version")
                )
                or not isinstance(row["evidence"], dict)
                or not row["evidence"]
            ):
                raise SchemaError(
                    ReasonCode.TYPE_MISMATCH,
                    f"{location}: silver 字段合同非法",
                )
        else:
            raise SchemaError(
                ReasonCode.SCHEMA_MISMATCH,
                f"{location}: annotation schema 不受支持",
            )
    return rows


class OABenchmarkAdapter:
    source_name = "oa"

    def __init__(self, config: Any) -> None:
        self.config = config

    def scan(
        self,
        *,
        deep: bool = False,
        for_build: bool = True,
    ) -> AdapterResult:
        root = self.config.oa.benchmark_root
        selection_root = self.config.oa.selection_root
        if root is None or selection_root is None:
            raise SourceAuditError(
                ReasonCode.SCHEMA_MISMATCH,
                "OA adapter 启用时 benchmark_root/selection_root 必须存在",
            )
        manifest, rows = _validate_benchmark_root(root)
        result = AdapterResult(source=self.source_name)
        train_rows = [row for row in rows if row["split"] == "train"]
        selected_train = train_rows
        if self.config.profile == "smoke":
            chosen: list[dict[str, Any]] = []
            for status, positive in (("empty", False), ("positive", True)):
                candidates = [
                    row
                    for row in train_rows
                    if (float(row["foreground_ratio"]) > 0) is positive
                ]
                candidates.sort(
                    key=lambda row: (
                        stable_hash(
                            self.config.seed,
                            "oa",
                            status,
                            row["sample_id"],
                        ),
                        row["sample_id"],
                    )
                )
                cap = self.config.limits.per_task.get(
                    f"oa:{TaskFamily.OA_MASK_FACTS.value}:{status}",
                    1,
                )
                chosen.extend(candidates[:cap])
            selected_train = chosen

        gold_selection: list[dict[str, Any]] = []
        selection_by_id: dict[str, dict[str, Any]] = {}
        if self.config.profile == "full":
            selection_audit: dict[str, Any] = {}
            gold_selection = select_gold_candidates(
                root,
                selection_root,
                seed=self.config.seed,
                audit=selection_audit,
            )
            result.audit.update(selection_audit)
            selection_by_id = {
                row["sample_id"]: row for row in gold_selection
            }
            result.audit["gold_selection_count"] = len(gold_selection)
            result.audit["gold_selection_sha256"] = sha256_bytes(
                canonical_json(gold_selection).encode("utf-8")
            )
            result.audit["gold_selection_split_counts"] = dict(
                sorted(Counter(row["split"] for row in gold_selection).items())
            )
            result.audit["gold_selection_source_counts"] = {
                split: dict(
                    sorted(
                        Counter(
                            row["source"]
                            for row in gold_selection
                            if row["split"] == split
                        ).items()
                    )
                )
                for split in ("train", "val", "test")
            }
            result.audit["gold_selection"] = gold_selection
        else:
            result.audit["gold_selection_evaluated"] = False
        gold_path = self.config.oa.gold_annotations
        if not for_build and self.config.profile == "full":
            if gold_path is None or not gold_path.is_file():
                result.audit["formal_blocker"] = (
                    ReasonCode.OA_REVIEW_INCOMPLETE.value
                )
                result.audit["reviewed_gold_count"] = 0
            else:
                gold_rows = _strict_annotation_rows(
                    gold_path,
                    schema_version=GOLD_SCHEMA_VERSION,
                    required={
                        "schema_version",
                        "selection_id",
                        "sample_id",
                        "upstream_index_sha256",
                        "facts",
                        "description",
                        "review_status",
                        "reviewer_id",
                        "reviewed_at",
                    },
                )
                valid_identities = 0
                seen: set[str] = set()
                for annotation in gold_rows:
                    sample_id = str(annotation["sample_id"])
                    expected = selection_by_id.get(sample_id)
                    if (
                        expected is not None
                        and sample_id not in seen
                        and annotation["review_status"] == "approved"
                        and annotation["selection_id"] == expected["selection_id"]
                        and annotation["upstream_index_sha256"]
                        == expected["upstream_index_sha256"]
                    ):
                        valid_identities += 1
                        seen.add(sample_id)
                result.audit["reviewed_gold_count"] = valid_identities
                if valid_identities != self.config.oa.expected_gold_count:
                    result.audit["formal_blocker"] = (
                        ReasonCode.OA_REVIEW_INCOMPLETE.value
                    )
            result.audit.update(
                {
                    "upstream_sample_count": len(rows),
                    "upstream_train_count": len(train_rows),
                    "upstream_index_sha256": manifest["index_sha256"],
                    "selected_oa_auto_count": len(selected_train),
                    "audit_materialized_oa_assets": False,
                }
            )
            summarize_result(result)
            result.audit["full_candidate_examples"] = len(train_rows)
            result.audit["selected_examples"] = len(selected_train)
            return result

        dataset = BenchmarkDataset(
            root,
            split=None,
            auxiliary_policy="all",
            normalization="none",
        )
        row_position = {
            str(row["sample_id"]): index for index, row in enumerate(dataset.rows)
        }
        cached: dict[str, tuple[tuple[PendingAsset, ...], dict[str, Any], tuple[str, ...]]] = {}

        def material(sample_row: Mapping[str, Any]) -> tuple[
            tuple[PendingAsset, ...], dict[str, Any], tuple[str, ...]
        ]:
            sample_id = str(sample_row["sample_id"])
            if sample_id not in cached:
                sample = dataset[row_position[sample_id]]
                cached[sample_id] = _sample_assets(
                    sample,
                    sample_row,
                    dataset,
                    self.config.asset_policy.oa_preview_clip,
                )
            return cached[sample_id]

        for row in selected_train:
            assets, facts, render_flags = material(row)
            key, parent_flags = _parent_key(row)
            response = _auto_response(facts)
            source_id = f"{row['sample_id']}:oa_auto"
            result.examples.append(
                SourceExample(
                    source=self.source_name,
                    source_record_id=source_id,
                    source_split="train",
                    parent_key=key,
                    logical_role=LogicalRole.OA_TRAIN,
                    task_family=TaskFamily.OA_MASK_FACTS,
                    supervision_kind=SupervisionKind.STRUCTURED_REPORT,
                    input_layout=InputLayout.MASK_GROUNDED,
                    output_modality=OutputModality.TEXT,
                    assets=assets,
                    target={
                        "type": "mask",
                        "image_role": "image",
                        "mask_role": "mask",
                        "source_convention": "binary_raster_top_left",
                        "empty_allowed": True,
                    },
                    instruction=(
                        "Describe the landslide regions indicated by the mask using "
                        "only deterministic mask-grounded facts."
                    ),
                    training_responses=(response,),
                    reference_responses=(response,),
                    deterministic_facts={
                        **facts,
                        "upstream_source": row["source"],
                        "upstream_sample_id": row["sample_id"],
                        "mask_status": "positive" if facts["has_target"] else "empty",
                        "fact_extractor_version": "oa_mask_facts.v1",
                    },
                    annotation_layer=AnnotationLayer.OA_AUTO,
                    review_status=ReviewStatus.NOT_REQUIRED,
                    provenance=(
                        {
                            "source_item_id": source_id,
                            "upstream_benchmark_schema": manifest["schema_version"],
                            "upstream_index_sha256": manifest["index_sha256"],
                            "upstream_record_sha256": row["record_sha256"],
                            "upstream_sample_id": row["sample_id"],
                        },
                    ),
                    quality_flags=tuple(sorted((*parent_flags, *render_flags))),
                )
            )

        if gold_path is not None and gold_path.is_file():
            gold_rows = _strict_annotation_rows(
                gold_path,
                schema_version=GOLD_SCHEMA_VERSION,
                required={
                    "schema_version",
                    "selection_id",
                    "sample_id",
                    "upstream_index_sha256",
                    "facts",
                    "description",
                    "review_status",
                    "reviewer_id",
                    "reviewed_at",
                },
            )
            if len(gold_rows) != self.config.oa.expected_gold_count:
                raise SourceAuditError(
                    ReasonCode.OA_REVIEW_INCOMPLETE,
                    f"gold 应为 {self.config.oa.expected_gold_count} 条，实际 {len(gold_rows)}",
                )
            selection_rows = {
                row["sample_id"]: row
                for row in read_jsonl(
                    selection_root / "index.jsonl"
                )
            }
            seen_gold: set[str] = set()
            for annotation in gold_rows:
                sample_id = str(annotation["sample_id"])
                expected = selection_by_id.get(sample_id)
                if expected is None or sample_id in seen_gold:
                    raise SourceAuditError(
                        ReasonCode.OA_REVIEW_INCOMPLETE,
                        f"gold sample_id 不在锁定选择或重复：{sample_id}",
                    )
                if (
                    annotation["review_status"] != "approved"
                    or annotation["selection_id"] != expected["selection_id"]
                    or annotation["upstream_index_sha256"]
                    != expected["upstream_index_sha256"]
                ):
                    raise SourceAuditError(
                        ReasonCode.OA_REVIEW_INCOMPLETE,
                        f"gold 身份或审核状态不符：{sample_id}",
                    )
                seen_gold.add(sample_id)
                row = selection_rows[sample_id]
                assets, auto_facts, render_flags = material(row)
                key, parent_flags = _parent_key(row)
                split = str(expected["split"])
                role = LogicalRole(f"oa_{split}")
                description = str(annotation["description"]).strip()
                if not description:
                    raise SchemaError(
                        ReasonCode.TYPE_MISMATCH,
                        f"gold description 不能为空：{sample_id}",
                    )
                result.examples.append(
                    SourceExample(
                        source=self.source_name,
                        source_record_id=f"{sample_id}:oa_gold",
                        source_split=split,
                        parent_key=key,
                        logical_role=role,
                        task_family=TaskFamily.OA_MASK_DESCRIPTION,
                        supervision_kind=SupervisionKind.LONG_DESCRIPTION,
                        input_layout=InputLayout.MASK_GROUNDED,
                        output_modality=OutputModality.TEXT,
                        assets=assets,
                        target={
                            "type": "mask",
                            "image_role": "image",
                            "mask_role": "mask",
                            "source_convention": "binary_raster_top_left",
                            "empty_allowed": True,
                        },
                        instruction=(
                            "Describe the reviewed landslide target indicated by the mask."
                        ),
                        training_responses=(description,) if split == "train" else (),
                        reference_responses=(description,),
                        deterministic_facts={
                            **auto_facts,
                            "reviewed_facts": annotation["facts"],
                            "upstream_source": row["source"],
                        },
                        annotation_layer=AnnotationLayer.OA_GOLD,
                        review_status=ReviewStatus.APPROVED,
                        provenance=(
                            {
                                "source_item_id": f"{sample_id}:oa_gold",
                                "selection_id": annotation["selection_id"],
                                "upstream_index_sha256": annotation[
                                    "upstream_index_sha256"
                                ],
                                "upstream_record_sha256": row["record_sha256"],
                                "reviewer_id": annotation["reviewer_id"],
                                "reviewed_at": annotation["reviewed_at"],
                            },
                        ),
                        quality_flags=tuple(sorted((*parent_flags, *render_flags))),
                    )
                )
            if seen_gold != set(selection_by_id):
                raise SourceAuditError(
                    ReasonCode.OA_REVIEW_INCOMPLETE,
                    "gold 未覆盖完整 358 条锁定 identity",
                )
        else:
            result.audit["formal_blocker"] = ReasonCode.OA_REVIEW_INCOMPLETE.value

        if self.config.oa.silver_annotations is not None:
            silver_rows = _strict_annotation_rows(
                self.config.oa.silver_annotations,
                schema_version=SILVER_SCHEMA_VERSION,
                required={
                    "schema_version",
                    "sample_id",
                    "description",
                    "evidence",
                    "filter_version",
                },
            )
            full_by_id = {str(row["sample_id"]): row for row in rows}
            seen_silver: set[str] = set()
            for annotation in silver_rows:
                sample_id = str(annotation["sample_id"])
                row = full_by_id.get(sample_id)
                if sample_id in seen_silver:
                    raise SourceAuditError(
                        ReasonCode.DUPLICATE_RECORD,
                        f"oa_silver sample_id 重复：{sample_id}",
                    )
                seen_silver.add(sample_id)
                if row is None or row["split"] != "train":
                    raise SourceAuditError(
                        ReasonCode.OA_ROLE_CONTAMINATION,
                        f"oa_silver 只能引用 OA train：{sample_id}",
                    )
                if not isinstance(annotation["evidence"], dict) or not annotation["evidence"]:
                    raise SchemaError(
                        ReasonCode.TYPE_MISMATCH,
                        f"oa_silver evidence 不能为空：{sample_id}",
                    )
                assets, facts, render_flags = material(row)
                key, parent_flags = _parent_key(row)
                description = str(annotation["description"]).strip()
                result.examples.append(
                    SourceExample(
                        source=self.source_name,
                        source_record_id=f"{sample_id}:oa_silver",
                        source_split="train",
                        parent_key=key,
                        logical_role=LogicalRole.OA_TRAIN,
                        task_family=TaskFamily.OA_MASK_DESCRIPTION,
                        supervision_kind=SupervisionKind.LONG_DESCRIPTION,
                        input_layout=InputLayout.MASK_GROUNDED,
                        output_modality=OutputModality.TEXT,
                        assets=assets,
                        target={
                            "type": "mask",
                            "image_role": "image",
                            "mask_role": "mask",
                            "source_convention": "binary_raster_top_left",
                            "empty_allowed": True,
                        },
                        instruction="Describe the mask-grounded landslide target.",
                        training_responses=(description,),
                        reference_responses=(description,),
                        deterministic_facts=facts,
                        annotation_layer=AnnotationLayer.OA_SILVER,
                        review_status=ReviewStatus.NOT_REQUIRED,
                        provenance=(
                            {
                                "source_item_id": f"{sample_id}:oa_silver",
                                "evidence": annotation["evidence"],
                                "filter_version": annotation["filter_version"],
                                "upstream_index_sha256": manifest["index_sha256"],
                            },
                        ),
                        quality_flags=tuple(sorted((*parent_flags, *render_flags))),
                    )
                )

        result.audit.update(
            {
                "upstream_sample_count": len(rows),
                "upstream_train_count": len(train_rows),
                "upstream_index_sha256": manifest["index_sha256"],
                "selected_oa_auto_count": len(selected_train),
            }
        )
        summarize_result(result)
        result.audit["selected_examples"] = len(result.examples)
        return result
