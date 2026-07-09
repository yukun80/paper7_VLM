#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""步骤 1-1：扫描多源滑坡数据集，生成原始数据清单。

脚本作用：对应 Task_Introduction.md 的“步骤 1”，统计每个原始数据集
的模态、格式、split、样本量、尺寸、标签状态和已知 warning。
主要输入：datasets/ 下的 GDCLD、LandslideBench_agent、LMHLD、
landslide4sense、Sen12Landslides、multimodal-landslide-dataset。
主要输出：source_manifest.csv 和 dataset_inventory.json。
是否改写原始数据：不会改写 datasets/，只写 benchmark/ 下的清单文件。
典型用法：python scripts/1-benchmark/1-1_scan_sources.py --datasets-root datasets --out-dir benchmark/multisource_landslide_v1_small
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from geohazard_benchmark_common import (
    DEFAULT_BENCHMARK_ROOT,
    DEFAULT_DATASETS_ROOT,
    MANIFEST_FIELDS,
    count_lines,
    ensure_dir,
    hdf5_dataset_meta,
    image_size_text,
    parse_npy_header,
    probe_image,
    read_lines,
    sen12_collect,
    to_repo_rel,
    write_csv,
    write_json,
)


def row(**kwargs: Any) -> dict[str, Any]:
    """按统一字段补齐清单行。"""
    return {field: kwargs.get(field, "") for field in MANIFEST_FIELDS}


def image_number(path: Path) -> str:
    """解析 Landslide4Sense image_123.h5 / mask_123.h5 的数字 ID。"""
    return path.stem.split("_", 1)[1]


def scan_gdcld(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not root.exists():
        return [row(dataset_name="GDCLD", warning=f"目录不存在: {to_repo_rel(root)}")]

    for split in ["train", "val"]:
        data_dir = root / f"{split}_data"
        label_dir = root / f"{split}_label"
        images = sorted(data_dir.glob("*"))
        labels = sorted(label_dir.glob("*"))
        probe = probe_image(images[0]) if images else {}
        rows.append(row(
            dataset_name="GDCLD",
            split=split,
            subset="patch",
            modalities="optical_rgb",
            file_format="tif_extension_png_or_tiff",
            num_samples=min(len(images), len(labels)),
            image_size=image_size_text(probe),
            gsd_m="unknown",
            label_status="supervised_mask",
            region="mixed",
            task_type="landslide_segmentation",
            warning="部分 .tif 扩展名文件实际为 PNG 编码，读取器需要按文件头识别",
        ))

    test_data = root / "test_data"
    test_label = root / "test_label"
    for region_dir in sorted([p for p in test_data.iterdir() if p.is_dir()] if test_data.exists() else []):
        labels = test_label / region_dir.name
        image_files = sorted(region_dir.glob("*"))
        label_files = sorted(labels.glob("*")) if labels.exists() else []
        probe = probe_image(image_files[0]) if image_files else {}
        rows.append(row(
            dataset_name="GDCLD",
            split="test",
            subset="scene",
            modalities="optical_rgb",
            file_format="geotiff_or_png_encoded_tif",
            num_samples=min(len(image_files), len(label_files)),
            image_size=image_size_text(probe),
            gsd_m="unknown",
            label_status="supervised_mask",
            region=region_dir.name,
            task_type="scene_level_landslide_segmentation",
            warning="test 为区域级大图，训练前应切片或按大图评估流程处理",
        ))
    return rows


def scan_landslidebench(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not root.exists():
        return [row(dataset_name="LandslideBench_agent", warning=f"目录不存在: {to_repo_rel(root)}")]
    first_image = next(iter(sorted((root / "images").glob("*.png"))), None)
    probe = probe_image(first_image) if first_image else {}
    for split in ["train", "val", "test"]:
        jsonl = root / f"qwen3vl_landslide_{split}.jsonl"
        rows.append(row(
            dataset_name="LandslideBench_agent",
            split=split,
            subset="qwen3vl_jsonl",
            modalities="optical_rgb",
            file_format="png+jsonl",
            num_samples=count_lines(jsonl),
            image_size=image_size_text(probe),
            gsd_m="unknown",
            label_status="supervised_mask_with_negative",
            region="mixed",
            task_type="landslide_segmentation",
            warning="non* 文件作为无滑坡负样本，mask 应为空",
        ))
    return rows


def scan_lmhld(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not root.exists():
        return [row(dataset_name="LMHLD", warning=f"目录不存在: {to_repo_rel(root)}")]

    diff_root = root / "LMHLD_dataset_different_patch_sizes"
    for region_dir in sorted([p for p in diff_root.iterdir() if p.is_dir()] if diff_root.exists() else []):
        for split in ["train", "val", "test"]:
            img = region_dir / f"{split}_images.npy"
            lab = region_dir / f"{split}_labels.npy"
            if not img.exists() or not lab.exists():
                continue
            meta = parse_npy_header(img)
            shape = meta["shape"]
            count = shape[0] if shape else 0
            channels = shape[1] if len(shape) >= 4 else "unknown"
            size = f"{shape[-2]}x{shape[-1]}" if len(shape) >= 4 else "unknown"
            rows.append(row(
                dataset_name="LMHLD",
                split=split,
                subset="different_patch_sizes",
                modalities=f"optical_{channels}band",
                file_format="npy_virtual_samples",
                num_samples=count,
                image_size=size,
                gsd_m="unknown",
                label_status="supervised_mask",
                region=region_dir.name,
                task_type="landslide_segmentation",
                warning="主线保留不同 patch size，用于尺度鲁棒性实验",
            ))

    comp_root = root / "Comparison_dataset_same_patch_size"
    for split in ["train", "val", "test"]:
        img = comp_root / f"{split}_images.npy"
        lab = comp_root / f"{split}_labels.npy"
        if not img.exists() or not lab.exists():
            continue
        meta = parse_npy_header(img)
        shape = meta["shape"]
        size = f"{shape[-2]}x{shape[-1]}" if len(shape) >= 4 else "unknown"
        rows.append(row(
            dataset_name="LMHLD",
            split=split,
            subset="baseline_same_patch_size",
            modalities="optical_4band",
            file_format="npy_virtual_samples",
            num_samples=shape[0] if shape else 0,
            image_size=size,
            gsd_m="unknown",
            label_status="supervised_mask",
            region="mixed",
            task_type="baseline_landslide_segmentation",
            warning="baseline subset 可能与不同尺寸版本存在语义重复，训练采样时需显式区分",
        ))
    return rows


def scan_landslide4sense(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not root.exists():
        return [row(dataset_name="landslide4sense", warning=f"目录不存在: {to_repo_rel(root)}")]
    image_size = "128x128"
    warning = "官方通道语义：B1-B12=Sentinel-2, B13=slope, B14=DEM"
    first_img = next((root / "TrainData" / "img").glob("image_*.h5"), None)
    if first_img:
        try:
            meta = hdf5_dataset_meta(first_img, "img")
            shape = meta["shape"]
            if len(shape) >= 3:
                image_size = f"{shape[0]}x{shape[1]}"
                warning = f"{warning}; img shape={shape}, dtype={meta['dtype']}"
        except Exception as exc:
            warning = f"{warning}; HDF5 元数据读取失败: {exc}"

    for split, folder, subset in [
        ("train", "TrainData", "official_train"),
        ("val", "ValidData", "official_val"),
        ("test", "TestData", "official_test"),
    ]:
        img = {image_number(p): p for p in (root / folder / "img").glob("image_*.h5")}
        mask = {image_number(p): p for p in (root / folder / "mask").glob("mask_*.h5")}
        paired = set(img) & set(mask)
        if not img and not mask:
            continue
        cur_warning = warning
        missing_mask = len(set(img) - set(mask))
        missing_image = len(set(mask) - set(img))
        if missing_mask or missing_image:
            cur_warning = f"{cur_warning}; 跳过未配对样本 images_without_mask={missing_mask}, masks_without_image={missing_image}"
        rows.append(row(
            dataset_name="landslide4sense",
            split=split,
            subset=subset,
            modalities="multispectral+slope+dem",
            file_format="hdf5",
            num_samples=len(paired),
            image_size=image_size,
            gsd_m=10,
            label_status="supervised_mask",
            region="mixed",
            task_type="landslide_segmentation",
            warning=cur_warning,
        ))
    return rows


def scan_sen12(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not root.exists():
        return [row(dataset_name="Sen12Landslides", warning=f"目录不存在: {to_repo_rel(root)}")]
    data = sen12_collect(root)
    for sensor, mapping in data.items():
        modalities = "sentinel2+dem" if sensor == "s2" else f"sentinel1_{sensor}"
        rows.append(row(
            dataset_name="Sen12Landslides",
            split="all",
            subset=sensor,
            modalities=modalities,
            file_format="netcdf_hdf5",
            num_samples=len(mapping),
            image_size="128x128",
            gsd_m=10,
            label_status="mixed_annotated_flag",
            region="15_events",
            task_type="temporal_landslide_segmentation",
            warning="需要按 annotated=True 过滤监督样本",
        ))
    keys_all = set(data["s2"]) | set(data["s1asc"]) | set(data["s1dsc"])
    keys_asc = set(data["s2"]) & set(data["s1asc"])
    keys_dsc = set(data["s2"]) & set(data["s1dsc"])
    keys_triple = set(data["s2"]) & set(data["s1asc"]) & set(data["s1dsc"])
    rows.append(row(
        dataset_name="Sen12Landslides",
        split="all",
        subset="aligned_union",
        modalities="sentinel2+dem+optional_sentinel1",
        file_format="netcdf_hdf5",
        num_samples=len(keys_all),
        image_size="128x128",
        gsd_m=10,
        label_status="mixed_annotated_flag",
        region="15_events",
        task_type="multisource_temporal_landslide_segmentation",
        warning=f"S2∩S1asc={len(keys_asc)}, S2∩S1dsc={len(keys_dsc)}, S2∩S1asc∩S1dsc={len(keys_triple)}",
    ))
    return rows


def scan_multimodal(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    base = root / "multimodal-landslide-dataset"
    if not base.exists():
        return [row(dataset_name="multimodal-landslide-dataset", warning=f"目录不存在: {to_repo_rel(base)}")]
    rgb = {p.stem for p in (base / "rgb").glob("*.tif")}
    dem = {p.stem for p in (base / "dem").glob("*.tif")}
    insar = {p.stem for p in (base / "insar_vel").glob("*.tif")}
    label = {p.stem for p in (base / "label").glob("*.tif")}
    paired = rgb & dem & insar & label
    first = next(iter(sorted((base / "rgb").glob("*.tif"))), None)
    probe = probe_image(first) if first else {}
    split_counts = {
        "train": len(read_lines(base / "train.txt")),
        "val": len(read_lines(base / "val.txt")),
        "extended_train": len(read_lines(base / "完整list" / "train.txt")),
        "extended_val": len(read_lines(base / "完整list" / "val.txt")),
    }
    for split, count in split_counts.items():
        rows.append(row(
            dataset_name="multimodal-landslide-dataset",
            split=split,
            subset="official" if not split.startswith("extended") else "extended_pool",
            modalities="optical_rgb+dem+insar_vel",
            file_format="geotiff",
            num_samples=count,
            image_size=image_size_text(probe),
            gsd_m="unknown",
            label_status="supervised_mask",
            region="Loess/SiC/XiZ",
            task_type="evidence_conditioned_landslide_segmentation",
            warning=f"完整四模态配对文件数={len(paired)}",
        ))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="扫描多源滑坡数据集，生成 benchmark 原始清单。")
    parser.add_argument("--datasets-root", type=Path, default=DEFAULT_DATASETS_ROOT, help="原始 datasets 根目录。")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_BENCHMARK_ROOT, help="当前模式 benchmark 输出目录。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.out_dir / "reports")
    scanners = [
        scan_gdcld(args.datasets_root / "GDCLD"),
        scan_landslidebench(args.datasets_root / "LandslideBench_agent"),
        scan_lmhld(args.datasets_root / "LMHLD"),
        scan_landslide4sense(args.datasets_root / "landslide4sense"),
        scan_sen12(args.datasets_root / "Sen12Landslides"),
        scan_multimodal(args.datasets_root / "multimodal-landslide-dataset"),
    ]
    rows = [item for group in scanners for item in group]
    write_csv(args.out_dir / "source_manifest.csv", rows, MANIFEST_FIELDS)
    inventory = {
        "说明": "本文件由 1-1_scan_sources.py 生成，只包含轻量数据清单。",
        "datasets_root": to_repo_rel(args.datasets_root),
        "out_dir": to_repo_rel(args.out_dir),
        "num_rows": len(rows),
        "rows": rows,
    }
    write_json(args.out_dir / "dataset_inventory.json", inventory)
    print(f"已生成数据清单: {to_repo_rel(args.out_dir / 'source_manifest.csv')}，共 {len(rows)} 行")


if __name__ == "__main__":
    main()
