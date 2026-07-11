#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""步骤 1-3：验证统一 JSONL 索引。

用途：检查 source、final
或 referring_target 索引中的路径、split、mask、bbox 字段、模态缺失标记和重复样本。
主要输入：source 阶段读取 indexes/source_all.jsonl；final 阶段读取 indexes/all.jsonl；
referring_target 阶段读取 indexes/referring_target_all.jsonl。
主要输出：reports/validation_report_source.json、validation_report.json 或
validation_report_referring_target.json。
写入行为：不会改写 datasets/ 或索引，只写 benchmark/ 下的验证报告。
所属流程：benchmark 构建的 source/final/referring_target 阶段门禁。
推荐运行命令：python scripts/1-benchmark/1-3_validate_index.py --stage final --benchmark-dir benchmark/multisource_landslide_v1_small
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from geohazard_benchmark_common import (
    DEFAULT_BENCHMARK_ROOT,
    LANDSLIDE4SENSE_MODALITY_SHAPES,
    final_index_paths,
    hdf5_has_dataset,
    path_is_inside_benchmark,
    project_path_arg,
    read_jsonl,
    referring_target_index_paths,
    resolve_repo_path,
    source_index_paths,
    to_repo_rel,
    write_json,
)


def check_path(path_ref: str | None) -> tuple[bool, str]:
    if not path_ref:
        return False, "路径为空"
    path = resolve_repo_path(path_ref)
    if path is None:
        return False, "路径无法解析"
    if not path.exists():
        return False, f"路径不存在: {to_repo_rel(path)}"
    return True, ""


def sample_modality_combo(sample: dict[str, Any], stage: str) -> str:
    key = "source_modalities" if stage == "source" else "modalities"
    names = [name for name, data in sample.get(key, {}).items() if data.get("available", True)]
    return "+".join(sorted(names)) if names else "none"


def validate_referring_target_sample(sample: dict[str, Any], benchmark_dir: Path) -> tuple[list[str], list[str]]:
    """检查 expression-level 指代目标样本。"""
    errors: list[str] = []
    warnings: list[str] = []
    sid = sample.get("sample_id", "<missing_sample_id>")
    if sample.get("task_type") != "referring_landslide_target":
        errors.append(f"{sid}: referring target 样本 task_type 必须是 referring_landslide_target")
    if not sample.get("parent_sample_id"):
        errors.append(f"{sid}: referring target 样本缺少 parent_sample_id")

    for forbidden in ["instruction", "text", "text_zh", "template_id"]:
        if forbidden in sample:
            errors.append(f"{sid}: 1-benchmark referring target 不应包含文本/模板字段: {forbidden}")

    category = sample.get("category")
    if category not in {"position", "scale", "morphology", "count"}:
        errors.append(f"{sid}: referring target category 非法: {category}")
    if not sample.get("subtype"):
        errors.append(f"{sid}: referring target 缺少 subtype")
    if not isinstance(sample.get("grounding"), dict):
        errors.append(f"{sid}: referring target 缺少 grounding")

    target = sample.get("target_mask") or {}
    if target.get("format") != "npy":
        errors.append(f"{sid}: target_mask 必须是 benchmark 内 npy，当前 format={target.get('format')}")
    if target.get("dtype") != "uint8":
        errors.append(f"{sid}: target_mask dtype 应为 uint8，当前 {target.get('dtype')}")
    if target.get("empty_mask") is True or int(target.get("positive_pixels") or 0) <= 0:
        errors.append(f"{sid}: target_mask 必须是非空 mask")
    if "bbox_xyxy" not in target:
        errors.append(f"{sid}: target_mask 缺少 bbox_xyxy")
    if not path_is_inside_benchmark(target.get("path"), benchmark_dir):
        errors.append(f"{sid}: target_mask 路径不在 benchmark 目录内: {target.get('path')}")
    return errors, warnings


def validate_landslide4sense_sample(sample: dict[str, Any], *, stage: str, modalities_key: str, mask_key: str) -> tuple[list[str], list[str]]:
    """检查 Landslide4Sense 官方 B1-B12/slope/DEM 模态约定。"""
    errors: list[str] = []
    warnings: list[str] = []
    sid = sample.get("sample_id", "<missing_sample_id>")
    modalities = sample.get(modalities_key) or {}
    supervision = sample.get("supervision", "mask")
    flags = set(sample.get("quality_flags") or [])
    split_source = str(sample.get("split_source") or "")
    subset = str(sample.get("subset") or "")
    task_type = str(sample.get("task_type") or "")

    if supervision != "mask":
        errors.append(f"{sid}: Landslide4Sense 不应作为非监督样本进入 benchmark，当前 supervision={supervision}")
    if "unlabeled_or_hidden_label" in flags:
        errors.append(f"{sid}: Landslide4Sense 不应包含 unlabeled_or_hidden_label 标记")
    if "images_only" in split_source or "images_only" in subset:
        errors.append(f"{sid}: Landslide4Sense 不应使用 images_only split/subset: split_source={split_source}, subset={subset}")
    if task_type == "unlabeled_landslide_segmentation":
        errors.append(f"{sid}: Landslide4Sense 不应使用 unlabeled_landslide_segmentation")

    required = {"multispectral", "slope", "dem"}
    missing = sorted(required - set(modalities))
    if missing:
        errors.append(f"{sid}: Landslide4Sense 缺少模态 {missing}")

    for name in sorted(required & set(modalities)):
        item = modalities[name]
        expected_shape = LANDSLIDE4SENSE_MODALITY_SHAPES[name]
        if item.get("shape") != expected_shape:
            errors.append(f"{sid}: Landslide4Sense {name} shape 应为 {expected_shape}，当前 {item.get('shape')}")
        if stage == "source":
            if item.get("format") != "hdf5" or item.get("internal_key") != "img":
                errors.append(f"{sid}: Landslide4Sense source 模态 {name} 必须使用 hdf5::img")
            path = resolve_repo_path(item.get("path"))
            if path and path.exists() and not hdf5_has_dataset(path, "img"):
                errors.append(f"{sid}: Landslide4Sense HDF5 缺少 img dataset: {to_repo_rel(path)}")
        else:
            if item.get("format") != "npy":
                errors.append(f"{sid}: Landslide4Sense final 模态 {name} 必须物化为 npy")

    if "hdf5_channel_semantics_need_verification" in flags:
        errors.append(f"{sid}: Landslide4Sense 不应再包含 hdf5_channel_semantics_need_verification")
    if "landslide4sense_official_band_mapping_applied" not in flags:
        warnings.append(f"{sid}: 建议记录 landslide4sense_official_band_mapping_applied")

    mask = sample.get(mask_key)
    if not isinstance(mask, dict):
        errors.append(f"{sid}: Landslide4Sense 样本必须包含 mask")
    elif stage == "source":
        if mask.get("format") != "hdf5" or mask.get("internal_key") != "mask":
            errors.append(f"{sid}: Landslide4Sense source mask 必须使用 hdf5::mask")
        path = resolve_repo_path(mask.get("path"))
        if path and path.exists() and not hdf5_has_dataset(path, "mask"):
            errors.append(f"{sid}: Landslide4Sense HDF5 缺少 mask dataset: {to_repo_rel(path)}")
    elif mask.get("shape") != [1, 128, 128]:
        errors.append(f"{sid}: Landslide4Sense final mask shape 应为 [1, 128, 128]，当前 {mask.get('shape')}")
    return errors, warnings


def preview_quality_warnings(sid: str, preview_name: str, path_ref: str | None) -> list[str]:
    """检查 preview 是否明显全白/全黑；只作为 warning，避免误杀低对比遥感图。"""
    if not path_ref:
        return []
    path = resolve_repo_path(path_ref)
    if path is None or not path.exists():
        return []
    try:
        with Image.open(path) as img:
            arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    except Exception as exc:
        return [f"{sid}: preview {preview_name} 无法读取用于质量检查: {exc}"]
    if arr.size == 0:
        return [f"{sid}: preview {preview_name} 为空图像"]
    mean = float(arr.mean())
    std = float(arr.std())
    near_white = float((arr >= 250).mean())
    near_black = float((arr <= 5).mean())
    warnings: list[str] = []
    if near_white > 0.98 or (mean > 248.0 and std < 3.0):
        warnings.append(f"{sid}: preview {preview_name} 疑似全白或严重饱和 mean={mean:.2f} std={std:.2f}")
    if near_black > 0.98 or (mean < 7.0 and std < 3.0):
        warnings.append(f"{sid}: preview {preview_name} 疑似全黑或严重低对比 mean={mean:.2f} std={std:.2f}")
    return warnings


def modality_array_quality_warnings(sid: str, name: str, modality: dict[str, Any]) -> list[str]:
    """检查已物化 Sentinel-2 相关数组是否被归一化成近乎常量。"""
    if modality.get("format") != "npy":
        return []
    sensor = str(modality.get("sensor") or "").lower()
    source = str(modality.get("source") or "").lower()
    role = str(modality.get("role") or "").lower()
    is_s2_like = sensor == "sentinel2" or "sentinel-2" in source or "sentinel2" in role
    if not is_s2_like:
        return []
    path = resolve_repo_path(modality.get("path"))
    if path is None or not path.exists():
        return []
    try:
        arr = np.asarray(np.load(path, mmap_mode="r"), dtype=np.float32)
    except Exception as exc:
        return [f"{sid}: 模态 {name} 无法读取用于质量检查: {exc}"]
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return [f"{sid}: 模态 {name} 全部为非有限值"]
    if finite.size > 500000:
        finite = finite[:: max(1, finite.size // 500000)]
    p01, p99 = np.percentile(finite, [1, 99])
    std = float(np.std(finite))
    warnings: list[str] = []
    if float(p99 - p01) < 1e-4 or std < 1e-4:
        warnings.append(f"{sid}: 模态 {name} 疑似被归一化为近常量 p01={float(p01):.6f} p99={float(p99):.6f}")
    if float((finite >= 0.999).mean()) > 0.98:
        warnings.append(f"{sid}: 模态 {name} 归一化后超过 98% 像素接近 1，可能仍然饱和")
    return warnings


def validate_sen12_sample(sample: dict[str, Any], *, stage: str, modalities_key: str) -> tuple[list[str], list[str]]:
    """Sen12 以 Sentinel-2 为主模态；缺少 S2 的样本不应进入 benchmark。"""
    errors: list[str] = []
    warnings: list[str] = []
    sid = sample.get("sample_id", "<missing_sample_id>")
    modalities = sample.get(modalities_key) or {}
    supervision = sample.get("supervision", "mask")
    split_source = str(sample.get("split_source") or "")
    if supervision != "mask":
        errors.append(f"{sid}: Sen12Landslides annotated=False/无监督样本应在 1-2 阶段跳过，当前 supervision={supervision}")
    if split_source == "annotated_false":
        errors.append(f"{sid}: Sen12Landslides 不应再写入 split_source=annotated_false")
    if "multispectral" not in modalities:
        errors.append(f"{sid}: Sen12Landslides 样本缺少 S2 multispectral 主模态，应在 1-2 阶段过滤")
    flags = set(sample.get("quality_flags") or [])
    if "sen12_without_s2_dem" in flags:
        errors.append(f"{sid}: Sen12Landslides 样本仍带有 sen12_without_s2_dem 标记，应从 benchmark 中排除")
    if stage == "final" and "multispectral" in modalities:
        preview_paths = (sample.get("preview") or {}).get("paths") or {}
        for required in [
            "multispectral_true_color",
            "multispectral_false_color_nir",
            "multispectral_swir",
            "multispectral_band_grid",
        ]:
            if required not in preview_paths:
                errors.append(f"{sid}: Sen12Landslides final 样本缺少 preview {required}")
    return errors, warnings


def validate_sample(sample: dict[str, Any], *, stage: str, benchmark_dir: Path) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    sid = sample.get("sample_id", "<missing_sample_id>")
    modalities_key = "source_modalities" if stage == "source" else "modalities"
    mask_key = "source_mask" if stage == "source" else "mask"

    if not sample.get("dataset_name"):
        errors.append(f"{sid}: 缺少 dataset_name")
    if not sample.get("split"):
        errors.append(f"{sid}: 缺少 split")
    if stage == "final" and any(field in sample for field in ["instruction", "template_id", "task_family"]):
        errors.append(f"{sid}: final 索引不应包含正式 instruction/template 字段；请在 2-instruction 阶段生成")
    if not isinstance(sample.get(modalities_key), dict) or not sample[modalities_key]:
        errors.append(f"{sid}: 缺少 {modalities_key}")

    for name, modality in sample.get(modalities_key, {}).items():
        if modality.get("available", True) is False:
            continue
        ok, message = check_path(modality.get("path"))
        if not ok:
            errors.append(f"{sid}: 模态 {name} {message}")
        if stage in {"final", "referring_target"} and not path_is_inside_benchmark(modality.get("path"), benchmark_dir):
            errors.append(f"{sid}: 最终模态 {name} 路径不在 benchmark 目录内: {modality.get('path')}")
        if modality.get("format") in {"hdf5", "netcdf"} and not modality.get("internal_key"):
            warnings.append(f"{sid}: 模态 {name} 是 {modality.get('format')}，但 internal_key 为空")
        if stage == "final":
            warnings.extend(modality_array_quality_warnings(sid, name, modality))

    supervision = sample.get("supervision", "mask")
    mask = sample.get(mask_key)
    if supervision == "referring_target":
        if stage != "referring_target":
            warnings.append(f"{sid}: supervision=referring_target 只应出现在 referring_target 索引")
        if "mask" in sample:
            errors.append(f"{sid}: referring_target 索引不应使用 mask 字段，应使用 target_mask")
    elif supervision == "mask":
        if not isinstance(mask, dict):
            errors.append(f"{sid}: 监督样本缺少 {mask_key} 字段")
        else:
            ok, message = check_path(mask.get("path"))
            if not ok:
                errors.append(f"{sid}: mask {message}")
            if stage in {"final", "referring_target"} and not path_is_inside_benchmark(mask.get("path"), benchmark_dir):
                errors.append(f"{sid}: 最终 mask 路径不在 benchmark 目录内: {mask.get('path')}")
            if "bbox_xyxy" not in mask:
                errors.append(f"{sid}: mask 缺少 bbox_xyxy 字段")
            elif mask.get("bbox_xyxy") is None and mask.get("bbox_status") in (None, ""):
                warnings.append(f"{sid}: bbox 为空且缺少 bbox_status")
    elif mask is not None:
        warnings.append(f"{sid}: 非监督样本仍包含 mask 字段，请确认是否应进入 supervised split")

    if stage in {"final", "referring_target"} and isinstance(sample.get("preview"), dict):
        preview_paths = sample["preview"].get("paths") or {}
        if not isinstance(preview_paths, dict) or not preview_paths:
            warnings.append(f"{sid}: final 样本缺少 preview paths")
        for name, path_ref in preview_paths.items():
            ok, message = check_path(path_ref)
            if not ok:
                errors.append(f"{sid}: preview {name} {message}")
            if not path_is_inside_benchmark(path_ref, benchmark_dir):
                errors.append(f"{sid}: preview {name} 路径不在 benchmark 目录内: {path_ref}")
            if name in {"visual", "multispectral_true_color", "multispectral_false_color_nir", "multispectral_swir"}:
                warnings.extend(preview_quality_warnings(sid, name, path_ref))

    flags = sample.get("quality_flags") or []
    for flag in flags:
        if flag in {"annotated_flag_missing_or_unreadable", "annotated_flag_not_detected_by_lightweight_reader"}:
            warnings.append(f"{sid}: {flag}")

    if sample.get("dataset_name") == "landslide4sense":
        cur_errors, cur_warnings = validate_landslide4sense_sample(sample, stage=stage, modalities_key=modalities_key, mask_key=mask_key)
        errors.extend(cur_errors)
        warnings.extend(cur_warnings)

    if sample.get("dataset_name") == "Sen12Landslides":
        cur_errors, cur_warnings = validate_sen12_sample(sample, stage=stage, modalities_key=modalities_key)
        errors.extend(cur_errors)
        warnings.extend(cur_warnings)

    if stage == "referring_target":
        cur_errors, cur_warnings = validate_referring_target_sample(sample, benchmark_dir)
        errors.extend(cur_errors)
        warnings.extend(cur_warnings)

    spatial = sample.get("spatial") or {}
    if spatial.get("original_size") is None:
        warnings.append(f"{sid}: original_size 未知，后续 dataloader 需要在线读取")
    if spatial.get("bucket_size") is None:
        warnings.append(f"{sid}: bucket_size 未知，后续 padding 需要在线确定")

    return errors, warnings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="验证 benchmark 统一索引，生成 validation_report.json。")
    parser.add_argument("--benchmark-dir", type=project_path_arg, default=DEFAULT_BENCHMARK_ROOT, help="后缀式 small 或 full benchmark 输出目录。")
    parser.add_argument("--stage", choices=["source", "final", "referring_target"], default="final", help="验证源索引、最终自包含训练索引或指代目标索引。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage == "source":
        paths = source_index_paths(args.benchmark_dir)
    elif args.stage == "referring_target":
        paths = referring_target_index_paths(args.benchmark_dir)
    else:
        paths = final_index_paths(args.benchmark_dir)
    samples = read_jsonl(paths["all"])
    errors: list[str] = []
    warnings: list[str] = []

    seen_ids: dict[str, int] = {}
    seen_source_split: dict[tuple[str, str], set[str]] = defaultdict(set)
    split_counts: Counter[str] = Counter()
    dataset_counts: Counter[str] = Counter()
    combo_counts: Counter[str] = Counter()

    for idx, sample in enumerate(samples):
        sid = str(sample.get("sample_id", ""))
        if not sid:
            errors.append(f"第 {idx} 行缺少 sample_id")
        elif sid in seen_ids:
            errors.append(f"重复 sample_id: {sid}，首次行 {seen_ids[sid]}，重复行 {idx}")
        else:
            seen_ids[sid] = idx

        dataset = str(sample.get("dataset_name", "unknown"))
        source_key = str(sample.get("source_key", sid))
        split = str(sample.get("split", "unknown"))
        if sample.get("supervision", "mask") == "mask":
            seen_source_split[(dataset, source_key)].add(split)
        split_counts[split] += 1
        dataset_counts[dataset] += 1
        combo_counts[sample_modality_combo(sample, args.stage)] += 1

        cur_errors, cur_warnings = validate_sample(sample, stage=args.stage, benchmark_dir=args.benchmark_dir)
        errors.extend(cur_errors)
        warnings.extend(cur_warnings)

    for (dataset, source_key), splits in sorted(seen_source_split.items()):
        real_splits = {split for split in splits if split in {"train", "val", "test"}}
        if len(real_splits) > 1:
            errors.append(f"split 泄漏: {dataset}/{source_key} 同时出现在 {sorted(real_splits)}")

    report = {
        "说明": "source 阶段允许 datasets/ 原始路径；final/referring_target 阶段要求读取路径全部位于 benchmark 目录内。",
        "stage": args.stage,
        "benchmark_dir": to_repo_rel(args.benchmark_dir),
        "num_samples": len(samples),
        "num_errors": len(errors),
        "num_warnings": len(warnings),
        "errors": errors,
        "warnings": warnings[:2000],
        "warning_truncated": len(warnings) > 2000,
        "counts": {
            "by_split": dict(sorted(split_counts.items())),
            "by_dataset": dict(sorted(dataset_counts.items())),
            "by_modality_combo": dict(sorted(combo_counts.items())),
        },
    }
    if args.stage == "source":
        report_name = "validation_report_source.json"
    elif args.stage == "referring_target":
        report_name = "validation_report_referring_target.json"
    else:
        report_name = "validation_report.json"
    write_json(args.benchmark_dir / "reports" / report_name, report)
    print(f"验证完成({args.stage}): errors={len(errors)}, warnings={len(warnings)} -> {to_repo_rel(args.benchmark_dir / 'reports' / report_name)}")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
