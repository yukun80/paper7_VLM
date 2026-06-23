#!/usr/bin/env python3
"""1-2 生成 Sen12 VLM 视图。

用途：
    从 Sen12Landslides 的 NetCDF patch 生成 VLM 可读取的 RGB 图像和二值滑坡掩膜。

输入：
    - `benchmark/<run>/intermediate/source_manifest.jsonl`
    - `datasets/Sen12Landslides/{s2,s1asc,s1dsc}/*.nc`

输出：
    - `benchmark/<run>/intermediate/sen12_samples.jsonl`
    - `benchmark/<run>/vlm_views/sen12/`
    - `benchmark/<run>/segmentation_masks/sen12/`
    - `benchmark/<run>/segmentation_masks_redblack/sen12/`

关键处理：
    - Sen12 原始数据已经是 128×128 patch，因此按 patch-level 直接渲染，不再切片。
    - V0 只生成 Sentinel-2 事件后真彩色视图。
    - V1 增加 Sentinel-2 假彩色、事前/事后多图样本和 Sentinel-1 SAR 视图。
    - 灾前/灾后样本会分别保存 pre、post 两张主输入图，pair_preview 仅用于人工查看。
    - V2 在 V1 基础上增加 DEM hillshade 辅助视图。
    - bbox 从语义 mask 自动派生，只表示证据范围，不能视为实例级滑坡标注。

示例命令：
python scripts/1-2_prepare_sen12_views.py \
    --out-dir benchmark/geohazard_halluground_v0 \
    --version v2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from geohazard_common import (
    bbox_from_mask,
    ensure_dir,
    evidence_sufficiency,
    hillshade,
    indexed_date,
    make_rgb,
    mask_area,
    norm_bbox_1000,
    quality_from_s2,
    read_jsonl,
    read_sen12_var,
    robust_stretch,
    save_mask,
    save_mask_visual,
    save_rgb,
    write_jsonl,
)


def sen12_base_from_manifest(row: dict[str, Any]) -> dict[str, Any]:
    """把 1-1 的清单行转换为统一 metadata 需要的基础字段。"""
    return {
        "source_dataset": "Sen12Landslides",
        "source_file": row["source_file"],
        "region_id": row["region_id"],
        "event_id": row["event_id"],
        "native_patch_id": row["native_patch_id"],
        "trigger_type": "mixed_or_inventory_defined",
        "event_date": row.get("event_date"),
        "pre_index": int(row.get("pre_index", 0)),
        "post_index": int(row.get("post_index", 14)),
        "ann_id": row.get("ann_id", ""),
        "ann_bbox_source_crs": row.get("ann_bbox_source_crs", ""),
        "date_confidence": row.get("date_confidence", ""),
        "crs": row.get("crs", ""),
        "center_lat": row.get("center_lat", ""),
        "center_lon": row.get("center_lon", ""),
        "split_group": row["split_group"],
        "split": row["split"],
        "license_note": "Sen12Landslides 公开数据集；使用时需保留原始数据来源引用。",
        "modality_dir": row["modality_dir"],
        "spatial_resolution_m": row.get("spatial_resolution_m", 10),
    }


def make_sen12_sample(
    path: Path,
    base: dict[str, Any],
    rgb: np.ndarray,
    mask: np.ndarray,
    out_dir: Path,
    sample_suffix: str,
    sensor_type: str,
    modality: str,
    native_bands: list[str],
    date: str | None,
    pre_date: str | None,
    post_date: str | None,
    quality_label: str,
) -> dict[str, Any]:
    sample_id = f"sen12_{base['region_id']}_{base['native_patch_id']}_{sample_suffix}"
    image_path = out_dir / "vlm_views" / "sen12" / f"{sample_id}.png"
    mask_path = out_dir / "segmentation_masks" / "sen12" / f"{sample_id}.png"
    mask_visual_path = out_dir / "segmentation_masks_redblack" / "sen12" / f"{sample_id}.png"
    save_rgb(image_path, rgb)
    save_mask(mask_path, mask)
    save_mask_visual(mask_visual_path, mask)

    # bbox 由语义二值掩膜派生，只用于 VLM grounding 训练和检测格式评估，
    # 不能解释为一个个独立滑坡实例的人工标注。
    bbox = bbox_from_mask(mask)
    height, width = mask.shape
    hazard_present = bool(mask_area(mask) > 0)
    return {
        "sample_id": sample_id,
        "source_dataset": base["source_dataset"],
        "source_file": base["source_file"],
        "rendered_image": image_path.as_posix(),
        "sensor_type": sensor_type,
        "modality": modality,
        "region_id": base["region_id"],
        "event_id": base["event_id"],
        "trigger_type": base["trigger_type"],
        "date": date,
        "pre_date": pre_date,
        "post_date": post_date,
        "spatial_resolution_m": base["spatial_resolution_m"],
        "native_bands": native_bands,
        "image_width": int(width),
        "image_height": int(height),
        "mask_path": mask_path.as_posix(),
        "mask_visual_path": mask_visual_path.as_posix(),
        "bbox_xyxy": bbox,
        "bbox_norm_1000": norm_bbox_1000(bbox, width, height),
        "hazard_present": hazard_present,
        "hazard_type": "landslide" if hazard_present else "none",
        "quality_label": quality_label,
        "evidence_sufficiency": evidence_sufficiency(hazard_present, modality, quality_label),
        "split_group": base["split_group"],
        "split": base["split"],
        "license_note": base["license_note"],
        "ann_id": base["ann_id"],
        "ann_bbox_source_crs": base["ann_bbox_source_crs"],
        "date_confidence": base["date_confidence"],
        "crs": base["crs"],
        "center_lat": base["center_lat"],
        "center_lon": base["center_lon"],
        "hard_negative_type": "unknown_background" if not hazard_present else "",
        "source_scene_file": "",
        "source_label_file": "",
        "source_window_xywh": [],
        "source_scene_width": None,
        "source_scene_height": None,
        "tile_index": None,
        "is_future_work": False,
        "mask_positive_pixels": mask_area(mask),
    }


def make_sen12_pre_post_sample(
    path: Path,
    base: dict[str, Any],
    pre_rgb: np.ndarray,
    post_rgb: np.ndarray,
    mask: np.ndarray,
    out_dir: Path,
    pre_date: str | None,
    post_date: str | None,
    quality_label: str,
) -> dict[str, Any]:
    sample_suffix = "s2_pre_post"
    sample_id = f"sen12_{base['region_id']}_{base['native_patch_id']}_{sample_suffix}"
    pre_image_path = out_dir / "vlm_views" / "sen12" / "pre" / f"{sample_id}_pre.png"
    post_image_path = out_dir / "vlm_views" / "sen12" / "post" / f"{sample_id}_post.png"
    pair_preview_path = out_dir / "vlm_views" / "sen12" / "pair_preview" / f"{sample_id}_preview.png"
    mask_path = out_dir / "segmentation_masks" / "sen12" / f"{sample_id}.png"
    mask_visual_path = out_dir / "segmentation_masks_redblack" / "sen12" / f"{sample_id}.png"

    # VLM 训练主输入是两张独立图；pair_preview 只用于 audit 和人工查看。
    save_rgb(pre_image_path, pre_rgb)
    save_rgb(post_image_path, post_rgb)
    save_rgb(pair_preview_path, np.concatenate([pre_rgb, post_rgb], axis=1))
    save_mask(mask_path, mask)
    save_mask_visual(mask_visual_path, mask)

    bbox = bbox_from_mask(mask)
    height, width = mask.shape
    hazard_present = bool(mask_area(mask) > 0)
    return {
        "sample_id": sample_id,
        "source_dataset": base["source_dataset"],
        "source_file": base["source_file"],
        "rendered_image": post_image_path.as_posix(),
        "pre_image": pre_image_path.as_posix(),
        "post_image": post_image_path.as_posix(),
        "pair_preview_image": pair_preview_path.as_posix(),
        "image_sequence": [
            {"role": "pre_event", "path": pre_image_path.as_posix(), "date": pre_date, "bands": ["B04", "B03", "B02"]},
            {"role": "post_event", "path": post_image_path.as_posix(), "date": post_date, "bands": ["B04", "B03", "B02"]},
        ],
        "primary_image_role": "post_event",
        "sensor_type": "Sentinel-2",
        "modality": "optical_pre_post_pair",
        "region_id": base["region_id"],
        "event_id": base["event_id"],
        "trigger_type": base["trigger_type"],
        "date": post_date,
        "pre_date": pre_date,
        "post_date": post_date,
        "spatial_resolution_m": base["spatial_resolution_m"],
        "native_bands": ["B04", "B03", "B02"],
        "image_width": int(width),
        "image_height": int(height),
        "mask_path": mask_path.as_posix(),
        "mask_visual_path": mask_visual_path.as_posix(),
        "bbox_xyxy": bbox,
        "bbox_norm_1000": norm_bbox_1000(bbox, width, height),
        "hazard_present": hazard_present,
        "hazard_type": "landslide" if hazard_present else "none",
        "quality_label": quality_label,
        "evidence_sufficiency": evidence_sufficiency(hazard_present, "optical_pre_post_pair", quality_label),
        "split_group": base["split_group"],
        "split": base["split"],
        "license_note": base["license_note"],
        "ann_id": base["ann_id"],
        "ann_bbox_source_crs": base["ann_bbox_source_crs"],
        "date_confidence": base["date_confidence"],
        "crs": base["crs"],
        "center_lat": base["center_lat"],
        "center_lon": base["center_lon"],
        "hard_negative_type": "unknown_background" if not hazard_present else "",
        "source_scene_file": "",
        "source_label_file": "",
        "source_window_xywh": [],
        "source_scene_width": None,
        "source_scene_height": None,
        "tile_index": None,
        "is_future_work": False,
        "mask_positive_pixels": mask_area(mask),
    }


def render_s2(row: dict[str, Any], out_dir: Path, version: str) -> list[dict[str, Any]]:
    path = Path(row["source_file"])
    base = sen12_base_from_manifest(row)
    post_idx = min(max(base["post_index"], 0), 14)
    pre_idx = min(max(base["pre_index"], 0), 14)

    b04_post, post_tags = read_sen12_var(path, "B04", post_idx)
    b03_post, _ = read_sen12_var(path, "B03", post_idx)
    b02_post, _ = read_sen12_var(path, "B02", post_idx)
    mask, _ = read_sen12_var(path, "MASK", post_idx)
    mask = (mask > 0).astype(np.uint8)

    samples: list[dict[str, Any]] = []
    true_rgb = make_rgb([b04_post, b03_post, b02_post])
    samples.append(
        make_sen12_sample(
            path=path,
            base=base,
            rgb=true_rgb,
            mask=mask,
            out_dir=out_dir,
            sample_suffix="s2_true_post",
            sensor_type="Sentinel-2",
            modality="optical_true_color_post_event",
            native_bands=["B04", "B03", "B02"],
            date=indexed_date(post_tags, post_idx),
            pre_date=indexed_date(post_tags, pre_idx),
            post_date=indexed_date(post_tags, post_idx),
            quality_label=quality_from_s2(path, post_idx, true_rgb),
        )
    )

    if version in {"v1", "v2"}:
        b08_post, _ = read_sen12_var(path, "B08", post_idx)
        false_rgb = make_rgb([b08_post, b04_post, b03_post])
        samples.append(
            make_sen12_sample(
                path=path,
                base=base,
                rgb=false_rgb,
                mask=mask,
                out_dir=out_dir,
                sample_suffix="s2_false_post",
                sensor_type="Sentinel-2",
                modality="optical_false_color_post_event",
                native_bands=["B08", "B04", "B03"],
                date=indexed_date(post_tags, post_idx),
                pre_date=indexed_date(post_tags, pre_idx),
                post_date=indexed_date(post_tags, post_idx),
                quality_label=quality_from_s2(path, post_idx, false_rgb),
            )
        )

        b04_pre, pre_tags = read_sen12_var(path, "B04", pre_idx)
        b03_pre, _ = read_sen12_var(path, "B03", pre_idx)
        b02_pre, _ = read_sen12_var(path, "B02", pre_idx)
        pre_rgb = make_rgb([b04_pre, b03_pre, b02_pre])
        samples.append(
            make_sen12_pre_post_sample(
                path=path,
                base=base,
                out_dir=out_dir,
                pre_rgb=pre_rgb,
                post_rgb=true_rgb,
                mask=mask,
                pre_date=indexed_date(pre_tags, pre_idx),
                post_date=indexed_date(post_tags, post_idx),
                quality_label=quality_from_s2(path, post_idx, true_rgb),
            )
        )
    return samples


def render_s1(row: dict[str, Any], out_dir: Path, version: str) -> list[dict[str, Any]]:
    if version == "v0":
        return []
    path = Path(row["source_file"])
    base = sen12_base_from_manifest(row)
    post_idx = min(max(base["post_index"], 0), 14)
    pre_idx = min(max(base["pre_index"], 0), 14)

    vv, tags = read_sen12_var(path, "VV", post_idx)
    vh, _ = read_sen12_var(path, "VH", post_idx)
    mask, _ = read_sen12_var(path, "MASK", post_idx)
    mask = (mask > 0).astype(np.uint8)
    diff = vv.astype("float32") - vh.astype("float32")
    rgb = make_rgb([vv, vh, diff])
    orbit = "ascending" if base["modality_dir"] == "s1asc" else "descending"
    orbit_zh = "升轨" if base["modality_dir"] == "s1asc" else "降轨"
    return [
        make_sen12_sample(
            path=path,
            base=base,
            rgb=rgb,
            mask=mask,
            out_dir=out_dir,
            sample_suffix=f"{base['modality_dir']}_sar_post",
            sensor_type=f"Sentinel-1 {orbit_zh}",
            modality=f"sar_backscatter_{orbit}_post_event",
            native_bands=["VV", "VH", "VV_minus_VH"],
            date=indexed_date(tags, post_idx),
            pre_date=indexed_date(tags, pre_idx),
            post_date=indexed_date(tags, post_idx),
            quality_label=f"可用 SAR 后向散射影像（{orbit_zh}，斑点噪声属预期现象）",
        )
    ]


def render_dem(row: dict[str, Any], out_dir: Path, version: str) -> list[dict[str, Any]]:
    if version != "v2":
        return []
    path = Path(row["source_file"])
    base = sen12_base_from_manifest(row)
    post_idx = min(max(base["post_index"], 0), 14)
    pre_idx = min(max(base["pre_index"], 0), 14)

    dem, tags = read_sen12_var(path, "DEM", post_idx)
    mask, _ = read_sen12_var(path, "MASK", post_idx)
    mask = (mask > 0).astype(np.uint8)
    hs = hillshade(dem)
    rgb = np.dstack([hs, robust_stretch(dem), hs])
    return [
        make_sen12_sample(
            path=path,
            base=base,
            rgb=rgb,
            mask=mask,
            out_dir=out_dir,
            sample_suffix=f"{base['modality_dir']}_dem_hillshade",
            sensor_type="Copernicus DEM",
            modality="dem_hillshade_auxiliary",
            native_bands=["DEM", "hillshade"],
            date=indexed_date(tags, post_idx),
            pre_date=indexed_date(tags, pre_idx),
            post_date=indexed_date(tags, post_idx),
            quality_label="辅助地形信息，不作为主要视觉证据",
        )
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从 Sen12 NetCDF patch 生成 VLM RGB 视图和二值掩膜。")
    parser.add_argument("--out-dir", default="benchmark/geohazard_halluground_v0", help="流水线输出目录。")
    parser.add_argument("--version", choices=["v0", "v1", "v2"], default="v0", help="要生成的 benchmark 版本。")
    parser.add_argument("--max-sen12-s2", type=int, default=None, help="额外限制 S2 patch 数量；默认使用清单中的全部。")
    parser.add_argument("--max-sen12-s1asc", type=int, default=None, help="额外限制 S1 升轨 patch 数量；默认使用清单中的全部。")
    parser.add_argument("--max-sen12-s1dsc", type=int, default=None, help="额外限制 S1 降轨 patch 数量；默认使用清单中的全部。")
    return parser.parse_args()


def apply_limits(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    limits = {"s2": args.max_sen12_s2, "s1asc": args.max_sen12_s1asc, "s1dsc": args.max_sen12_s1dsc}
    counts = {"s2": 0, "s1asc": 0, "s1dsc": 0}
    kept: list[dict[str, Any]] = []
    for row in rows:
        modality_dir = row.get("modality_dir")
        limit = limits.get(modality_dir)
        if limit is not None and counts[modality_dir] >= limit:
            continue
        counts[modality_dir] += 1
        kept.append(row)
    return kept


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    manifest_path = out_dir / "intermediate" / "source_manifest.jsonl"
    if not manifest_path.exists():
        raise SystemExit(f"缺少数据源清单：{manifest_path}，请先运行 1-1_scan_sources.py。")

    ensure_dir(out_dir / "intermediate")
    rows = [row for row in read_jsonl(manifest_path) if row.get("entry_type") == "sen12_patch"]
    rows = apply_limits(rows, args)
    samples: list[dict[str, Any]] = []

    for row in tqdm(rows, desc="生成 Sen12 VLM 视图", unit="patch"):
        try:
            # Sen12 是固定大小 patch 数据，保留 patch-level 单元能避免相邻切片泄漏和重复采样。
            if row["modality_dir"] == "s2":
                samples.extend(render_s2(row, out_dir, args.version))
                samples.extend(render_dem(row, out_dir, args.version))
            elif row["modality_dir"] in {"s1asc", "s1dsc"}:
                samples.extend(render_s1(row, out_dir, args.version))
                samples.extend(render_dem(row, out_dir, args.version))
        except Exception as exc:
            print(f"[警告] Sen12 patch 处理失败：{row.get('source_file')}；原因：{exc}", file=sys.stderr)

    out_path = out_dir / "intermediate" / "sen12_samples.jsonl"
    write_jsonl(out_path, samples)
    print(f"已写入 Sen12 样本清单：{out_path}，共 {len(samples)} 条。")


if __name__ == "__main__":
    main()
