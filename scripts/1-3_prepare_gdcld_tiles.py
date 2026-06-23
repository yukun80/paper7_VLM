#!/usr/bin/env python3
"""1-3 生成 GDCLD tile 视图。

用途：
    将 GDCLD 高分辨率整景图像切成 VLM 训练 tile，并生成与 tile 对齐的二值掩膜。

输入：
    - `benchmark/<run>/intermediate/source_manifest.jsonl`
    - GDCLD image-label pair，通常位于 `datasets/GDCLD/extracted`

输出：
    - `benchmark/<run>/intermediate/gdcld_samples.jsonl`
    - `benchmark/<run>/vlm_views/gdcld/`
    - `benchmark/<run>/segmentation_masks/gdcld/`
    - `benchmark/<run>/segmentation_masks_redblack/gdcld/`

关键处理：
    - GDCLD 是高分辨率整景图，默认按 512×512 tile-level 处理，不整图缩放。
    - 使用 rasterio windowed read，避免超大图像一次性读入内存。
    - 标签中所有 `>0` 的像元统一视为滑坡，解决 1、85、255 等不同正类编码。
    - image/label 如果存在 1 像素尺寸差，以二者共同覆盖范围为准。
    - 自动过滤标签边缘整行/整列伪阳性，避免 1 像素边框进入训练集。
    - Future work 默认不进入训练集；只有显式开启时才作为 `test_candidate` 候选数据导出。
    - bbox 从语义 mask 自动派生，只表示证据范围，不能视为实例级滑坡标注。

示例命令：
python scripts/1-3_prepare_gdcld_tiles.py \
    --out-dir benchmark/geohazard_halluground_v0 \
    --gdcld-tile-size 512 \
    --max-gdcld-tiles 20
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.windows import Window
from tqdm import tqdm

from geohazard_common import (
    bbox_from_mask,
    ensure_dir,
    evidence_sufficiency,
    gdcld_region_from_path,
    infer_gdcld_sensor,
    mask_area,
    norm_bbox_1000,
    read_gdcld_mask_window,
    read_gdcld_rgb_window,
    read_jsonl,
    save_mask,
    save_mask_visual,
    save_rgb,
    write_jsonl,
)


def safe_token(text: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return token.strip("_") or "unknown"


def window_grid(width: int, height: int, tile_size: int, stride: int) -> list[tuple[int, int]]:
    xs = list(range(0, max(0, width - tile_size + 1), stride))
    ys = list(range(0, max(0, height - tile_size + 1), stride))
    if width >= tile_size and (width - tile_size) not in xs:
        xs.append(width - tile_size)
    if height >= tile_size and (height - tile_size) not in ys:
        ys.append(height - tile_size)
    return [(x, y) for y in ys for x in xs]


def detect_label_frame_artifacts(
    label_ds: rasterio.io.DatasetReader,
    common_width: int,
    common_height: int,
    threshold: float = 0.95,
) -> dict[str, bool]:
    """检测标签图边缘整行/整列正类伪影。

    GDCLD 的部分标签比影像大 1 像素，且边缘行/列几乎全为正类。
    这种边框更像导出伪影，不应作为滑坡正样本进入训练集。
    """
    top = label_ds.read(1, window=Window(0, 0, common_width, 1)) > 0
    bottom = label_ds.read(1, window=Window(0, common_height - 1, common_width, 1)) > 0
    left = label_ds.read(1, window=Window(0, 0, 1, common_height)) > 0
    right = label_ds.read(1, window=Window(common_width - 1, 0, 1, common_height)) > 0
    return {
        "top": float(np.mean(top)) >= threshold,
        "bottom": float(np.mean(bottom)) >= threshold,
        "left": float(np.mean(left)) >= threshold,
        "right": float(np.mean(right)) >= threshold,
    }


def clear_label_frame_artifacts(
    mask: np.ndarray,
    window_xywh: tuple[int, int, int, int],
    common_width: int,
    common_height: int,
    frame_artifacts: dict[str, bool],
) -> np.ndarray:
    """清除当前 tile 与源标签边框重叠的伪阳性行/列。"""
    x, y, w, h = window_xywh
    cleaned = mask.copy()
    if frame_artifacts.get("top") and y == 0:
        cleaned[0, :] = 0
    if frame_artifacts.get("bottom") and y + h >= common_height:
        cleaned[-1, :] = 0
    if frame_artifacts.get("left") and x == 0:
        cleaned[:, 0] = 0
    if frame_artifacts.get("right") and x + w >= common_width:
        cleaned[:, -1] = 0
    return cleaned


def is_valid_positive_mask(mask: np.ndarray, min_positive_pixels: int, min_bbox_size: int, min_area_ratio: float) -> bool:
    area = mask_area(mask)
    if area < min_positive_pixels:
        return False
    bbox = bbox_from_mask(mask)
    if not bbox:
        return False
    x1, y1, x2, y2 = bbox
    if x2 - x1 < min_bbox_size or y2 - y1 < min_bbox_size:
        return False
    if area / float(mask.size) < min_area_ratio:
        return False
    return True


def select_windows(
    label_ds: rasterio.io.DatasetReader,
    width: int,
    height: int,
    tile_size: int,
    stride: int,
    min_positive_pixels: int,
    min_bbox_size: int,
    min_mask_area_ratio: float,
    negative_ratio: float,
    frame_artifacts: dict[str, bool],
) -> list[tuple[int, int, int, int]]:
    positive: list[tuple[int, int, int, int]] = []
    negative: list[tuple[int, int, int, int]] = []
    for x, y in window_grid(width, height, tile_size, stride):
        window_xywh = (x, y, tile_size, tile_size)
        window = Window(x, y, tile_size, tile_size)
        # 先只读取标签窗口，避免为了筛选样本而反复读取高分辨率 RGB 整景。
        mask = read_gdcld_mask_window(label_ds, window)
        mask = clear_label_frame_artifacts(mask, window_xywh, width, height, frame_artifacts)
        pos_pixels = mask_area(mask)
        if is_valid_positive_mask(mask, min_positive_pixels, min_bbox_size, min_mask_area_ratio):
            positive.append(window_xywh)
        elif pos_pixels == 0:
            negative.append(window_xywh)

    if negative_ratio <= 0:
        return positive
    max_negative = math.ceil(max(1, len(positive)) * negative_ratio) if positive else min(32, len(negative))
    return positive + negative[:max_negative]


def make_gdcld_sample(
    row: dict[str, Any],
    out_dir: Path,
    image_ds: rasterio.io.DatasetReader,
    label_ds: rasterio.io.DatasetReader,
    window_xywh: tuple[int, int, int, int],
    tile_index: int,
    common_width: int,
    common_height: int,
    frame_artifacts: dict[str, bool],
) -> dict[str, Any]:
    x, y, w, h = window_xywh
    window = Window(x, y, w, h)
    rgb = read_gdcld_rgb_window(image_ds, window)
    mask = read_gdcld_mask_window(label_ds, window)
    mask = clear_label_frame_artifacts(mask, window_xywh, common_width, common_height, frame_artifacts)

    sample_stem = safe_token(f"{row['pair_id']}_{Path(row['image_path']).stem}")
    sample_id = f"gdcld_{sample_stem}_tile_{tile_index:06d}"
    image_path = out_dir / "vlm_views" / "gdcld" / f"{sample_id}.png"
    mask_path = out_dir / "segmentation_masks" / "gdcld" / f"{sample_id}.png"
    mask_visual_path = out_dir / "segmentation_masks_redblack" / "gdcld" / f"{sample_id}.png"
    save_rgb(image_path, rgb)
    save_mask(mask_path, mask)
    save_mask_visual(mask_visual_path, mask)

    # GDCLD 是语义分割标签，不是实例分割标签；bbox 只作为证据区域外接框。
    bbox = bbox_from_mask(mask)
    height, width = mask.shape
    hazard_present = bool(mask_area(mask) > 0)
    image_file = Path(row["image_path"])
    split = "test_candidate" if row.get("is_future_work") else row.get("split", "train")
    region_id = row.get("region_id") or gdcld_region_from_path(image_file)
    modality = "rgb_high_resolution_tile"
    quality_label = "RGB tile 质量未额外标注"
    return {
        "sample_id": sample_id,
        "source_dataset": "GDCLD",
        "source_file": image_file.as_posix(),
        "rendered_image": image_path.as_posix(),
        "sensor_type": infer_gdcld_sensor(image_file),
        "modality": modality,
        "region_id": region_id,
        "event_id": row.get("event_id") or region_id,
        "trigger_type": "coseismic_or_inventory_defined",
        "date": None,
        "pre_date": None,
        "post_date": None,
        "spatial_resolution_m": None,
        "native_bands": ["R", "G", "B"],
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
        "split_group": f"gdcld:{split}:{region_id}:{row['pair_id']}",
        "split": split,
        "license_note": "GDCLD 公开 Zenodo 数据包；原始 UAV、Map World、Gaofen-6 数据可能存在额外使用限制。",
        "ann_id": "",
        "ann_bbox_source_crs": "",
        "date_confidence": "",
        "crs": "",
        "center_lat": "",
        "center_lon": "",
        "hard_negative_type": "unknown_background" if not hazard_present else "",
        "label_source_file": row["label_path"],
        "source_scene_file": row["image_path"],
        "source_label_file": row["label_path"],
        "source_window_xywh": [int(x), int(y), int(w), int(h)],
        "source_scene_width": int(image_ds.width),
        "source_scene_height": int(image_ds.height),
        "tile_index": int(tile_index),
        "is_future_work": bool(row.get("is_future_work")),
        "mask_positive_pixels": mask_area(mask),
    }


def process_pair(row: dict[str, Any], out_dir: Path, args: argparse.Namespace, remaining: int | None) -> list[dict[str, Any]]:
    if row.get("is_future_work") and not args.include_gdcld_future_work:
        # Future work 数据来源和标注完备性更适合作为候选测试或后续扩展，默认不混入训练/验证/正式测试。
        return []

    image_path = Path(row["image_path"])
    label_path = Path(row["label_path"])
    with rasterio.open(image_path) as image_ds, rasterio.open(label_path) as label_ds:
        # 部分 GDCLD 图像和标签存在 1 像素级尺寸差，这里只使用共同覆盖范围生成 tile。
        common_width = min(image_ds.width, label_ds.width)
        common_height = min(image_ds.height, label_ds.height)
        frame_artifacts = detect_label_frame_artifacts(label_ds, common_width, common_height)
        selected = select_windows(
            label_ds=label_ds,
            width=common_width,
            height=common_height,
            tile_size=args.gdcld_tile_size,
            stride=args.gdcld_stride,
            min_positive_pixels=args.gdcld_min_positive_pixels,
            min_bbox_size=args.gdcld_min_bbox_size,
            min_mask_area_ratio=args.gdcld_min_mask_area_ratio,
            negative_ratio=args.gdcld_negative_ratio,
            frame_artifacts=frame_artifacts,
        )
        if remaining is not None:
            selected = selected[:remaining]

        samples = []
        for local_idx, window_xywh in enumerate(selected):
            samples.append(make_gdcld_sample(row, out_dir, image_ds, label_ds, window_xywh, local_idx, common_width, common_height, frame_artifacts))
        return samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将 GDCLD 高分辨率整景图切成 512×512 VLM tile。")
    parser.add_argument("--out-dir", default="benchmark/geohazard_halluground_v0", help="流水线输出目录。")
    parser.add_argument("--gdcld-tile-size", type=int, default=512, help="GDCLD tile 边长，默认 512。")
    parser.add_argument("--gdcld-stride", type=int, default=512, help="GDCLD tile 滑窗步长，默认 512。")
    parser.add_argument("--gdcld-min-positive-pixels", type=int, default=20, help="正样本 tile 至少需要的滑坡像元数。")
    parser.add_argument("--gdcld-min-bbox-size", type=int, default=8, help="正样本 bbox 的最小宽度和高度，用于过滤细线伪标签。")
    parser.add_argument("--gdcld-min-mask-area-ratio", type=float, default=0.0005, help="正样本 mask 面积占 tile 面积的最小比例。")
    parser.add_argument("--gdcld-negative-ratio", type=float, default=1.0, help="每个图像对保留的负样本比例。")
    parser.add_argument("--max-gdcld-tiles", type=int, default=None, help="最多导出的 GDCLD tile 数量，用于 smoke test。")
    parser.add_argument("--include-gdcld-future-work", action="store_true", help="显式导出 Future work 数据，split 记为 test_candidate。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.gdcld_tile_size <= 0 or args.gdcld_stride <= 0:
        raise SystemExit("`--gdcld-tile-size` 和 `--gdcld-stride` 必须为正整数。")

    out_dir = Path(args.out_dir)
    manifest_path = out_dir / "intermediate" / "source_manifest.jsonl"
    if not manifest_path.exists():
        raise SystemExit(f"缺少数据源清单：{manifest_path}，请先运行 1-1_scan_sources.py。")
    ensure_dir(out_dir / "intermediate")

    rows = [row for row in read_jsonl(manifest_path) if row.get("entry_type") == "gdcld_pair"]
    samples: list[dict[str, Any]] = []
    for row in tqdm(rows, desc="生成 GDCLD tile", unit="图像对"):
        if args.max_gdcld_tiles is not None and len(samples) >= args.max_gdcld_tiles:
            break
        remaining = None if args.max_gdcld_tiles is None else args.max_gdcld_tiles - len(samples)
        try:
            samples.extend(process_pair(row, out_dir, args, remaining))
        except Exception as exc:
            print(f"[警告] GDCLD 图像/标签对处理失败：{row.get('image_path')} / {row.get('label_path')}；原因：{exc}", file=sys.stderr)

    out_path = out_dir / "intermediate" / "gdcld_samples.jsonl"
    write_jsonl(out_path, samples)
    print(f"已写入 GDCLD tile 样本清单：{out_path}，共 {len(samples)} 条。")


if __name__ == "__main__":
    main()
