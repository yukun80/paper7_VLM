#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""步骤 1-2：把异构数据集整理成统一 JSONL 索引。

脚本作用：对应 Task_Introduction.md 的“步骤 2”，把不同目录结构、
文件格式和模态组合整理为 source JSONL；该索引允许记录 datasets/ 原始路径，
仅供 1-4 物化阶段读取，不作为最终训练索引。
主要输入：datasets/ 原始数据目录，以及 --mode small|full、--small-limit 等参数。
主要输出：indexes/source_all.jsonl、source_train.jsonl、source_val.jsonl、
source_test.jsonl、source_unlabeled.jsonl。
是否改写原始数据：不会改写 datasets/；.npy/HDF5/NetCDF 使用虚拟引用，不拆大文件。
典型用法：python scripts/1-benchmark/1-2_build_index.py --mode small --small-limit 1000 --datasets-root datasets --out-dir benchmark/multisource_landslide_v1_small
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any
from collections import defaultdict

from geohazard_benchmark_common import (
    benchmark_dir_for_mode,
    DEFAULT_BENCHMARK_ROOT,
    DEFAULT_DATASETS_ROOT,
    ensure_dir,
    hash_split,
    landslide4sense_source_modalities,
    make_sample,
    mask_entry,
    modality_entry,
    parse_npy_header,
    probe_image,
    read_jsonl,
    read_lines,
    sen12_collect,
    sen12_has_mask_variable,
    sen12_parse_key,
    sen12_read_annotated,
    stable_sample_by_split,
    to_repo_rel,
    try_bbox_from_mask,
    write_json,
    write_source_split_indexes,
)


BUILD_DIAGNOSTICS: dict[str, Any] = {}


def path_shape_from_probe(path: Path) -> list[int] | None:
    info = probe_image(path)
    if info.get("height") and info.get("width"):
        channels = info.get("channels")
        if channels:
            return [int(channels), int(info["height"]), int(info["width"])]
        return [int(info["height"]), int(info["width"])]
    return None


def binary_mask_from_path(path: Path, empty_hint: bool | None = None) -> dict[str, Any]:
    bbox, status = try_bbox_from_mask(path)
    shape = path_shape_from_probe(path)
    empty_mask = empty_hint if empty_hint is not None else (True if status == "empty_mask" else None)
    return mask_entry(path, fmt=path.suffix.lower().lstrip(".") or "image", shape=shape, empty_mask=empty_mask, bbox_xyxy=bbox, bbox_status=status)


def apply_mode_limit(dataset_name: str, rows: list[dict[str, Any]], mode: str, small_limit: int, seed: int) -> list[dict[str, Any]]:
    if mode != "small":
        return rows
    sampled = stable_sample_by_split(rows, small_limit, seed)
    for item in sampled:
        item.setdefault("quality_flags", []).append(f"small_mode_sampled_from_{dataset_name}")
        item["quality_flags"] = sorted(set(item["quality_flags"]))
    return sampled


def enforce_small_limit_by_dataset_split(samples: list[dict[str, Any]], mode: str, small_limit: int, seed: int) -> list[dict[str, Any]]:
    """最终写出前按 dataset_name + split 收紧 small 上限。

    这样可以避免同一数据集的多个 subset 分别抽样后叠加超过 small_limit，
    例如 LMHLD 的 different_patch_sizes 与 baseline_same_patch_size。
    """
    if mode != "small" or small_limit <= 0:
        return samples
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        key = (str(sample.get("dataset_name", "unknown")), str(sample.get("split", "unknown")))
        groups[key].append(sample)

    limited: list[dict[str, Any]] = []
    for idx, key in enumerate(sorted(groups)):
        group = groups[key]
        selected = limit_candidates(group, "small", small_limit, seed + 1000 + idx, key_func=lambda row: row.get("sample_id", ""))
        for item in selected:
            flags = set(item.get("quality_flags") or [])
            flags.add(f"small_limit_per_dataset_split_{small_limit}")
            item["quality_flags"] = sorted(flags)
        limited.extend(selected)
    return sorted(limited, key=lambda row: str(row.get("sample_id", "")))


def to_source_sample(sample: dict[str, Any]) -> dict[str, Any]:
    """把构建期样本转换为 source schema，避免被误当作最终训练索引使用。"""
    source = dict(sample)
    source["source_modalities"] = source.pop("modalities", {})
    source["source_mask"] = source.pop("mask", None)
    source["provenance"] = {
        "source_index": True,
        "source_path_root": "datasets",
        "materialized_by": "scripts/1-benchmark/1-4_preprocess_samples.py",
    }
    return source


def limit_candidates(items: list[Any], mode: str, small_limit: int, seed: int, key_func=str) -> list[Any]:
    """small 模式先限制候选，避免为了抽样遍历完整大数据集。"""
    if mode != "small" or small_limit <= 0 or len(items) <= small_limit:
        return items
    ordered = sorted(items, key=lambda item: str(key_func(item)))
    import random

    rng = random.Random(seed)
    rng.shuffle(ordered)
    return sorted(ordered[:small_limit], key=lambda item: str(key_func(item)))


def build_gdcld(root: Path, mode: str, small_limit: int, seed: int) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for split in ["train", "val"]:
        data_dir = root / f"{split}_data"
        label_dir = root / f"{split}_label"
        img_paths = limit_candidates(sorted(data_dir.glob("*")), mode, small_limit, seed + (1 if split == "train" else 2), key_func=lambda p: p.name)
        for img_path in img_paths:
            mask_path = label_dir / img_path.name
            if not mask_path.exists():
                continue
            shape = path_shape_from_probe(img_path)
            samples.append(make_sample(
                dataset_name="GDCLD",
                split=split,
                task_type="landslide_segmentation",
                source_key=f"{split}/{img_path.name}",
                source_level="patch",
                subset="patch",
                modalities={
                    "optical_rgb": modality_entry(img_path, fmt="image", band_names=["R", "G", "B"], shape=shape, role="vlm_visual"),
                },
                mask=binary_mask_from_path(mask_path),
                region="mixed",
                quality_flags=["tif_extension_may_be_png_encoded"],
            ))

    test_data = root / "test_data"
    test_label = root / "test_label"
    for region_dir in sorted([p for p in test_data.iterdir() if p.is_dir()] if test_data.exists() else []):
        for img_path in sorted(region_dir.glob("*")):
            mask_path = test_label / region_dir.name / img_path.name
            if not mask_path.exists():
                continue
            shape = path_shape_from_probe(img_path)
            samples.append(make_sample(
                dataset_name="GDCLD",
                split="test",
                task_type="scene_level_landslide_segmentation",
                source_key=f"test/{region_dir.name}/{img_path.name}",
                source_level="scene",
                subset="scene",
                modalities={
                    "optical_rgb": modality_entry(img_path, fmt="geotiff_or_image", band_names=["R", "G", "B"], shape=shape, role="vlm_visual"),
                },
                mask=binary_mask_from_path(mask_path),
                region=region_dir.name,
                quality_flags=["scene_level_large_image", "requires_tiling_for_patch_training"],
            ))
    return apply_mode_limit("GDCLD", samples, mode, small_limit, seed)


def extract_landslidebench_image(line: dict[str, Any]) -> str | None:
    """从 Qwen3VL 消息格式中提取 image 路径。"""
    messages = line.get("messages", [])
    if not messages:
        return None
    content = messages[0].get("content", [])
    for item in content:
        if isinstance(item, dict) and item.get("type") == "image":
            return str(item.get("image"))
    return None


def subtype_from_landslidebench_name(name: str) -> tuple[str, str | None]:
    if name.startswith("non"):
        return "non_landslide", None
    zoom = None
    match = re.search(r"_Level_(\d+)", name)
    if match:
        zoom = match.group(1)
    prefix = re.sub(r"\d.*$", "", name).strip("_ ").lower()
    return prefix.replace(" ", "_") or "landslide", zoom


def build_landslidebench(root: Path, mode: str, small_limit: int, seed: int) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for split in ["train", "val", "test"]:
        jsonl_path = root / f"qwen3vl_landslide_{split}.jsonl"
        rows_with_image = []
        for row in read_jsonl(jsonl_path):
            image_rel = extract_landslidebench_image(row)
            if not image_rel:
                continue
            rows_with_image.append((row, image_rel))
        rows_with_image = limit_candidates(rows_with_image, mode, small_limit, seed + len(samples), key_func=lambda item: item[1])
        for _row, image_rel in rows_with_image:
            img_path = root / image_rel
            mask_path = root / "mask" / Path(image_rel).name
            stem = Path(image_rel).stem
            subtype, zoom = subtype_from_landslidebench_name(stem)
            is_negative = stem.startswith("non")
            shape = [3, 512, 512]
            flags = ["negative_empty_mask"] if is_negative else []
            if zoom:
                flags.append(f"web_zoom_level_{zoom}")
            samples.append(make_sample(
                dataset_name="LandslideBench_agent",
                split=split,
                task_type="negative_landslide_segmentation" if is_negative else "landslide_segmentation",
                source_key=f"{split}/{Path(image_rel).name}",
                subset="qwen3vl_jsonl",
                source_level="patch",
                modalities={
                    "optical_rgb": modality_entry(img_path, fmt="png", band_names=["R", "G", "B"], shape=shape, role="vlm_visual"),
                },
                mask=mask_entry(mask_path, fmt="png", shape=[1, 512, 512], empty_mask=True if is_negative else None, bbox_status="pending_pixel_read"),
                region="mixed",
                event_id=subtype,
                quality_flags=flags,
            ))
    return apply_mode_limit("LandslideBench_agent", samples, mode, small_limit, seed + 10)


def build_lmhld_block(root: Path, subset_name: str, subset_root: Path, mode: str, small_limit: int, seed: int) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    region_dirs = sorted([p for p in subset_root.iterdir() if p.is_dir()]) if subset_root.exists() else []
    if subset_name == "baseline_same_patch_size" and subset_root.exists():
        region_dirs = [subset_root]
    for region_dir in region_dirs:
        region = region_dir.name if region_dir != subset_root else "mixed"
        for split in ["train", "val", "test"]:
            img_npy = region_dir / f"{split}_images.npy"
            lab_npy = region_dir / f"{split}_labels.npy"
            if not img_npy.exists() or not lab_npy.exists():
                continue
            img_meta = parse_npy_header(img_npy)
            lab_meta = parse_npy_header(lab_npy)
            count = min(img_meta["shape"][0], lab_meta["shape"][0])
            channels = img_meta["shape"][1] if len(img_meta["shape"]) >= 4 else None
            shape = list(img_meta["shape"][1:]) if len(img_meta["shape"]) >= 4 else None
            mask_shape = list(lab_meta["shape"][1:]) if len(lab_meta["shape"]) >= 4 else None
            indices = list(range(count))
            indices = limit_candidates(indices, mode, small_limit, seed + len(samples), key_func=lambda x: x)
            for idx in indices:
                flags = ["npy_virtual_sample", f"channels_{channels}"]
                if subset_name == "baseline_same_patch_size":
                    flags.append("baseline_subset_not_primary")
                samples.append(make_sample(
                    dataset_name="LMHLD",
                    split=split,
                    task_type="baseline_landslide_segmentation" if subset_name == "baseline_same_patch_size" else "landslide_segmentation",
                    source_key=f"{subset_name}/{region}/{split}/{idx}",
                    subset=subset_name,
                    source_level="patch",
                    modalities={
                        "optical_multiband": modality_entry(
                            f"{to_repo_rel(img_npy)}::{idx}",
                            fmt="npy",
                            band_names=[f"B{i + 1}" for i in range(int(channels or 0))],
                            shape=shape,
                            internal_key=idx,
                            role="multiband_visual",
                        ),
                    },
                    mask=mask_entry(f"{to_repo_rel(lab_npy)}::{idx}", fmt="npy", shape=mask_shape, internal_key=idx, bbox_status="pending_array_read"),
                    region=region,
                    quality_flags=flags,
                ))
    return apply_mode_limit(f"LMHLD_{subset_name}", samples, mode, small_limit, seed)


def build_lmhld(root: Path, mode: str, small_limit: int, seed: int) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    samples.extend(build_lmhld_block(root, "different_patch_sizes", root / "LMHLD_dataset_different_patch_sizes", mode, small_limit, seed + 20))
    samples.extend(build_lmhld_block(root, "baseline_same_patch_size", root / "Comparison_dataset_same_patch_size", mode, small_limit, seed + 21))
    return samples


def image_number(path: Path) -> str:
    return path.stem.split("_", 1)[1]


def build_landslide4sense(root: Path, mode: str, small_limit: int, seed: int) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    split_defs = [
        ("train", "TrainData", "official_train", seed + 31),
        ("val", "ValidData", "official_val", seed + 32),
        ("test", "TestData", "official_test", seed + 33),
    ]
    diagnostics: dict[str, dict[str, int]] = {}
    for split, folder, subset, split_seed in split_defs:
        img = {image_number(p): p for p in (root / folder / "img").glob("image_*.h5")}
        mask = {image_number(p): p for p in (root / folder / "mask").glob("mask_*.h5")}
        keys = sorted(set(img) & set(mask), key=lambda item: int(item))
        diagnostics[split] = {
            "num_images": len(img),
            "num_masks": len(mask),
            "num_paired": len(keys),
            "num_images_without_mask": len(set(img) - set(mask)),
            "num_masks_without_image": len(set(mask) - set(img)),
        }
        keys = limit_candidates(keys, mode, small_limit, split_seed, key_func=lambda item: int(item))
        for key in keys:
            samples.append(make_sample(
                dataset_name="landslide4sense",
                split=split,
                split_source=f"{subset}_paired_mask",
                task_type="landslide_segmentation",
                source_key=f"{split}/{key}",
                subset=subset,
                source_level="patch",
                modalities=landslide4sense_source_modalities(img[key]),
                mask=mask_entry(mask[key], fmt="hdf5", shape=[1, 128, 128], internal_key="mask", bbox_status="pending_hdf5_read"),
                region="mixed",
                supervision="mask",
                quality_flags=["landslide4sense_official_band_mapping_applied", "needs_optional_derived_split"],
            ))
    sampled = apply_mode_limit("landslide4sense", samples, mode, small_limit, seed + 30)
    BUILD_DIAGNOSTICS["landslide4sense"] = {
        "split_pairing": diagnostics,
        "num_samples_after_sampling": len(sampled),
        "unlabeled_policy": "disabled_all_landslide4sense_samples_require_mask",
    }
    return sampled


def sen12_key_allowed(
    key: tuple[str, str],
    data: dict[str, dict[tuple[str, str], Path]],
    modal_policy: str,
) -> bool:
    """按研究策略筛选 Sen12 模态组合，避免并集样本污染特定实验。"""
    has_s2 = key in data["s2"]
    has_asc = key in data["s1asc"]
    has_dsc = key in data["s1dsc"]
    has_sar = has_asc or has_dsc
    if modal_policy == "union":
        return True
    if modal_policy == "require_s2":
        return has_s2
    if modal_policy == "require_s2_sar":
        return has_s2 and has_sar
    if modal_policy == "strict_all":
        return has_s2 and has_asc and has_dsc
    raise ValueError(f"未知 Sen12 modal policy: {modal_policy}")


def build_sen12(root: Path, mode: str, small_limit: int, seed: int, modal_policy: str) -> list[dict[str, Any]]:
    data = sen12_collect_limited(root, small_limit) if mode == "small" else sen12_collect(root)
    all_keys = sorted(set(data["s2"]) | set(data["s1asc"]) | set(data["s1dsc"]))
    keys = sorted(key for key in all_keys if sen12_key_allowed(key, data, modal_policy))
    BUILD_DIAGNOSTICS["Sen12Landslides"] = {
        "modal_policy": modal_policy,
        "num_raw_union_keys": len(all_keys),
        "num_selected_keys_before_sampling": len(keys),
        "num_skipped_without_s2": sum(1 for key in all_keys if key not in data["s2"]),
        "num_skipped_by_policy": len(all_keys) - len(keys),
        "num_skipped_annotated_false": 0,
        "num_skipped_missing_mask_variable": 0,
        "num_kept_annotated_missing_or_unreadable_with_mask": 0,
        "num_skipped_mask_status_unreadable": 0,
    }
    if mode == "small":
        split_groups: dict[str, list[tuple[str, str]]] = {"train": [], "val": [], "test": []}
        for key in keys:
            split_groups[hash_split(f"Sen12Landslides/{key[0]}/{key[1]}")].append(key)
        keys = []
        for idx, split in enumerate(["train", "val", "test"]):
            keys.extend(limit_candidates(split_groups[split], mode, small_limit, seed + 40 + idx, key_func=lambda item: f"{item[0]}/{item[1]}"))
        keys = sorted(keys)
    samples: list[dict[str, Any]] = []
    s2_bands = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
    for region, sample_no in keys:
        modalities: dict[str, dict[str, Any]] = {}
        mask_path = None
        flags = ["netcdf_virtual_multitemporal_sample"]
        if modal_policy != "union":
            flags.append(f"sen12_modal_policy_{modal_policy}")
        if (region, sample_no) in data["s2"]:
            p = data["s2"][(region, sample_no)]
            modalities["multispectral"] = modality_entry(
                p,
                fmt="netcdf",
                band_names=s2_bands,
                shape=[15, len(s2_bands), 128, 128],
                gsd_m=10,
                internal_key=s2_bands,
                role="sentinel2_multispectral",
                source="Sentinel-2_B02_B12",
                sensor="sentinel2",
                value_encoding="sen12_netcdf_reflectance",
            )
            modalities["dem"] = modality_entry(
                p,
                fmt="netcdf",
                band_names=["DEM"],
                shape=[15, 1, 128, 128],
                gsd_m=10,
                internal_key="DEM",
                role="terrain_dem",
                source="Sen12Landslides_S2_file_DEM",
                sensor="dem",
                value_encoding="sen12_netcdf_dem",
            )
            mask_path = p
        if (region, sample_no) in data["s1asc"]:
            p = data["s1asc"][(region, sample_no)]
            modalities["sar_asc"] = modality_entry(
                p,
                fmt="netcdf",
                band_names=["VV", "VH"],
                shape=[15, 2, 128, 128],
                gsd_m=10,
                internal_key=["VV", "VH"],
                role="sentinel1_ascending",
                source="Sentinel-1_ASC_VV_VH",
                sensor="sentinel1",
                value_encoding="sen12_netcdf_sar",
            )
            mask_path = mask_path or p
        if (region, sample_no) in data["s1dsc"]:
            p = data["s1dsc"][(region, sample_no)]
            modalities["sar_dsc"] = modality_entry(
                p,
                fmt="netcdf",
                band_names=["VV", "VH"],
                shape=[15, 2, 128, 128],
                gsd_m=10,
                internal_key=["VV", "VH"],
                role="sentinel1_descending",
                source="Sentinel-1_DSC_VV_VH",
                sensor="sentinel1",
                value_encoding="sen12_netcdf_sar",
            )
            mask_path = mask_path or p
        if "multispectral" not in modalities:
            flags.append("sen12_without_s2_dem")
        if "sar_asc" not in modalities or "sar_dsc" not in modalities:
            flags.append("sen12_missing_one_or_more_sar_tracks")

        annotated = sen12_read_annotated(mask_path) if mask_path else None
        has_mask_variable = sen12_has_mask_variable(mask_path) if mask_path else None
        if annotated is False:
            BUILD_DIAGNOSTICS["Sen12Landslides"]["num_skipped_annotated_false"] += 1
            continue
        if has_mask_variable is False:
            BUILD_DIAGNOSTICS["Sen12Landslides"]["num_skipped_missing_mask_variable"] += 1
            continue
        if has_mask_variable is None:
            BUILD_DIAGNOSTICS["Sen12Landslides"]["num_skipped_mask_status_unreadable"] += 1
            continue
        if annotated is None:
            flags.append("annotated_flag_missing_or_unreadable")
            BUILD_DIAGNOSTICS["Sen12Landslides"]["num_kept_annotated_missing_or_unreadable_with_mask"] += 1
        split = hash_split(f"Sen12Landslides/{region}/{sample_no}")
        samples.append(make_sample(
            dataset_name="Sen12Landslides",
            split=split,
            split_source="derived_hash_from_region_id",
            task_type="multisource_temporal_landslide_segmentation",
            source_key=f"{region}/{sample_no}",
            subset="aligned_union",
            source_level="patch",
            modalities=modalities,
            mask=mask_entry(mask_path, fmt="netcdf", shape=[15, 1, 128, 128], internal_key="MASK", bbox_status="pending_netcdf_read"),
            region=region,
            supervision="mask",
            quality_flags=flags,
        ))
    sampled = apply_mode_limit("Sen12Landslides", samples, mode, small_limit, seed + 40)
    BUILD_DIAGNOSTICS["Sen12Landslides"]["num_samples_after_sampling"] = len(sampled)
    return sampled


def sen12_collect_limited(root: Path, small_limit: int) -> dict[str, dict[tuple[str, str], Path]]:
    """small 模式专用：只扫描每个 sensor 的有限候选，避免遍历 3.9 万个 NetCDF。"""
    out: dict[str, dict[tuple[str, str], Path]] = {"s1asc": {}, "s1dsc": {}, "s2": {}}
    limit = max(small_limit * 8, 300)
    for sensor in out:
        sensor_dir = root / sensor
        if not sensor_dir.exists():
            continue
        count = 0
        with os.scandir(sensor_dir) as entries:
            for entry in entries:
                if not entry.is_file() or not entry.name.endswith(".nc"):
                    continue
                parsed = sen12_parse_key(Path(entry.path))
                if not parsed:
                    continue
                region, parsed_sensor, sample_id = parsed
                out[parsed_sensor][(region, sample_id)] = Path(entry.path)
                count += 1
                if count >= limit:
                    break
    return out


def build_multimodal(root: Path, mode: str, small_limit: int, seed: int, use_extended_pool: bool) -> list[dict[str, Any]]:
    base = root / "multimodal-landslide-dataset"
    split_files = [("train", base / "train.txt", "official"), ("val", base / "val.txt", "official")]
    if use_extended_pool:
        split_files.extend([
            ("extended_pool", base / "完整list" / "train.txt", "extended_pool"),
            ("extended_pool", base / "完整list" / "val.txt", "extended_pool"),
        ])
    samples: list[dict[str, Any]] = []
    seen_source_keys: set[str] = set()
    for split, list_path, subset in split_files:
        names = limit_candidates(read_lines(list_path), mode, small_limit, seed + len(samples), key_func=lambda x: x)
        for name in names:
            source_key = f"{subset}/{name}"
            if source_key in seen_source_keys:
                continue
            seen_source_keys.add(source_key)
            rgb = base / "rgb" / f"{name}.tif"
            dem = base / "dem" / f"{name}.tif"
            insar = base / "insar_vel" / f"{name}.tif"
            label = base / "label" / f"{name}.tif"
            shape_rgb = [3, 128, 128]
            shape_aux = [1, 128, 128]
            region = name.split("_", 1)[0]
            samples.append(make_sample(
                dataset_name="multimodal-landslide-dataset",
                split=split,
                split_source=subset,
                task_type="evidence_conditioned_landslide_segmentation",
                source_key=source_key,
                subset=subset,
                source_level="patch",
                modalities={
                    "optical_rgb": modality_entry(
                        rgb,
                        fmt="geotiff",
                        band_names=["R", "G", "B"],
                        shape=shape_rgb,
                        gsd_m=10,
                        role="sentinel2_rgb",
                        source="Sentinel-2_RGB",
                        sensor="sentinel2",
                        value_encoding="multimodal_rgb_int16",
                    ),
                    "dem": modality_entry(dem, fmt="geotiff", band_names=["DEM"], shape=shape_aux, role="terrain"),
                    "insar_vel": modality_entry(insar, fmt="geotiff", band_names=["insar_velocity"], shape=shape_aux, role="deformation"),
                },
                mask=mask_entry(label, fmt="geotiff", shape=[1, 128, 128], bbox_status="pending_pixel_read"),
                region=region,
                quality_flags=["insar_unit_and_sign_need_documentation"] + (["extended_pool_not_default"] if subset == "extended_pool" else []),
            ))
    return apply_mode_limit("multimodal-landslide-dataset", samples, mode, small_limit, seed + 50)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建多源滑坡分割 benchmark 的统一 JSONL 索引。")
    parser.add_argument("--datasets-root", type=Path, default=DEFAULT_DATASETS_ROOT, help="原始 datasets 根目录。")
    parser.add_argument("--out-dir", type=Path, default=None, help="当前模式 benchmark 输出目录，默认使用后缀式 multisource_landslide_v1_<mode>。")
    parser.add_argument("--mode", choices=["small", "full"], default="small", help="small 抽样模式或 full 完整模式。")
    parser.add_argument("--small-limit", type=int, default=1000, help="small 模式下每个 dataset_name + split 的最大样本数，默认 1000。")
    parser.add_argument("--seed", type=int, default=42, help="确定性抽样随机种子。")
    parser.add_argument("--use-extended-pool", action="store_true", help="full 模式是否纳入 multimodal-landslide-dataset/完整list。")
    parser.add_argument(
        "--sen12-modal-policy",
        choices=["union", "require_s2", "require_s2_sar", "strict_all"],
        default="require_s2",
        help="Sen12 样本配对策略：union 保留任意可用组合；require_s2_sar/strict_all 用于更干净的多源训练。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or benchmark_dir_for_mode(args.mode)
    ensure_dir(out_dir / "indexes")
    builders = [
        ("GDCLD", lambda: build_gdcld(args.datasets_root / "GDCLD", args.mode, args.small_limit, args.seed)),
        ("LandslideBench_agent", lambda: build_landslidebench(args.datasets_root / "LandslideBench_agent", args.mode, args.small_limit, args.seed)),
        ("LMHLD", lambda: build_lmhld(args.datasets_root / "LMHLD", args.mode, args.small_limit, args.seed)),
        ("landslide4sense", lambda: build_landslide4sense(args.datasets_root / "landslide4sense", args.mode, args.small_limit, args.seed)),
        ("Sen12Landslides", lambda: build_sen12(args.datasets_root / "Sen12Landslides", args.mode, args.small_limit, args.seed, args.sen12_modal_policy)),
        ("multimodal-landslide-dataset", lambda: build_multimodal(args.datasets_root / "multimodal-landslide-dataset", args.mode, args.small_limit, args.seed, args.use_extended_pool)),
    ]
    samples: list[dict[str, Any]] = []
    for name, builder in builders:
        print(f"  - 构建 {name} 索引...", flush=True)
        group = builder()
        print(f"    完成 {name}: {len(group)} 条", flush=True)
        samples.extend(group)
    samples = enforce_small_limit_by_dataset_split(samples, args.mode, args.small_limit, args.seed)
    source_samples = [to_source_sample(sample) for sample in samples]
    write_source_split_indexes(out_dir, source_samples)
    summary = {
        "说明": "源索引由 1-2_build_index.py 生成；其中 source_* 路径允许指向 datasets/，仅供物化阶段读取。",
        "mode": args.mode,
        "small_limit": args.small_limit if args.mode == "small" else None,
        "sen12_modal_policy": args.sen12_modal_policy,
        "diagnostics": BUILD_DIAGNOSTICS,
        "num_samples": len(source_samples),
        "num_by_split": {split: sum(1 for row in source_samples if row.get("split") == split) for split in ["train", "val", "test", "unlabeled", "extended_pool"]},
    }
    write_json(out_dir / "reports" / "index_build_summary.json", summary)
    print(f"已生成源索引: {to_repo_rel(out_dir / 'indexes' / 'source_all.jsonl')}，样本数 {len(source_samples)}")


if __name__ == "__main__":
    main()
