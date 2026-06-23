#!/usr/bin/env python3
"""1-1 扫描原始数据源。

用途：
    扫描 Sen12Landslides 和 GDCLD 原始数据，生成第一阶段数据源清单。

输入：
    - `datasets/Sen12Landslides`
    - `datasets/GDCLD/extracted` 或完整 GDCLD 解压目录

输出：
    - `benchmark/<run>/intermediate/source_manifest.jsonl`

关键处理：
    - Sen12 记录每个 NetCDF patch 的数据模态和稳定 split。
    - GDCLD 只记录 image-label pair，不读取整幅大图到内存。
    - Future work 默认标记为候选数据，后续不直接进入正式训练集。

示例命令：
python scripts/1-1_scan_sources.py \
    --sen12-root datasets/Sen12Landslides \
    --gdcld-root datasets/GDCLD/extracted \
    --out-dir benchmark/geohazard_halluground_v0 \
    --clean
"""

from __future__ import annotations

import argparse
from pathlib import Path

import rasterio
from tqdm import tqdm

from geohazard_common import (
    clean_dir,
    discover_gdcld_pairs,
    ensure_dir,
    gdcld_region_from_path,
    parse_literal,
    parse_sen12_name,
    raster_basic_info,
    sampled_label_values,
    split_from_group,
    write_jsonl,
)


def sen12_manifest_rows(root: Path, max_s2: int | None, max_s1asc: int | None, max_s1dsc: int | None) -> list[dict]:
    rows: list[dict] = []
    limits = {"s2": max_s2, "s1asc": max_s1asc, "s1dsc": max_s1dsc}
    for modality_dir, limit in limits.items():
        files = sorted((root / modality_dir).glob("*.nc"))
        if limit is not None:
            files = files[:limit]
        for path in tqdm(files, desc=f"扫描 Sen12 {modality_dir}", unit="文件"):
            with rasterio.open(path) as ds:
                tags = {k.replace("NC_GLOBAL#", ""): v for k, v in ds.tags().items()}
            pre_post = parse_literal(tags.get("pre_post_dates"), {})
            region_id, native_patch_id = parse_sen12_name(path)
            event_date = tags.get("event_date") or "unknown_event_date"
            split_group = f"sen12:{region_id}:{event_date}"
            rows.append(
                {
                    "entry_type": "sen12_patch",
                    "source_dataset": "Sen12Landslides",
                    "source_file": path.as_posix(),
                    "modality_dir": modality_dir,
                    "region_id": region_id,
                    "event_id": f"{region_id}_{event_date}",
                    "native_patch_id": native_patch_id,
                    "event_date": None if event_date == "unknown_event_date" else event_date,
                    "pre_index": int(pre_post.get("pre", 0)) if isinstance(pre_post, dict) else 0,
                    "post_index": int(pre_post.get("post", 14)) if isinstance(pre_post, dict) else 14,
                    "ann_id": tags.get("ann_id") or "",
                    "ann_bbox_source_crs": tags.get("ann_bbox") or "",
                    "date_confidence": tags.get("date_confidence") or "",
                    "crs": tags.get("crs") or "",
                    "center_lat": tags.get("center_lat") or "",
                    "center_lon": tags.get("center_lon") or "",
                    "split_group": split_group,
                    "split": split_from_group(split_group),
                    "spatial_resolution_m": 10,
                }
            )
    return rows


def gdcld_manifest_rows(root: Path) -> list[dict]:
    rows: list[dict] = []
    pairs = discover_gdcld_pairs(root)
    for idx, pair in enumerate(tqdm(pairs, desc="扫描 GDCLD 图像/标签对", unit="对")):
        image_path = Path(pair["image_path"])
        label_path = Path(pair["label_path"])
        image_info = raster_basic_info(image_path)
        label_info = raster_basic_info(label_path)
        rows.append(
            {
                "entry_type": "gdcld_pair",
                "source_dataset": "GDCLD",
                "pair_id": f"gdcld_pair_{idx:05d}",
                "image_path": image_path.as_posix(),
                "label_path": label_path.as_posix(),
                "split": pair["split"],
                "is_future_work": bool(pair["is_future_work"]),
                "region_id": gdcld_region_from_path(image_path),
                "event_id": gdcld_region_from_path(image_path),
                "image_width": image_info["width"],
                "image_height": image_info["height"],
                "image_bands": image_info["count"],
                "image_dtype": image_info["dtype"],
                "label_width": label_info["width"],
                "label_height": label_info["height"],
                "label_bands": label_info["count"],
                "label_dtype": label_info["dtype"],
                "label_values_sampled": sampled_label_values(label_path),
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="扫描 Sen12 和 GDCLD 原始数据并生成 source_manifest.jsonl。")
    parser.add_argument("--sen12-root", default="datasets/Sen12Landslides", help="Sen12Landslides 根目录。")
    parser.add_argument("--gdcld-root", default="datasets/GDCLD/extracted", help="GDCLD 解压根目录。")
    parser.add_argument("--out-dir", default="benchmark/geohazard_halluground_v0", help="输出目录。")
    parser.add_argument("--max-sen12-s2", type=int, default=None, help="限制 Sen12 S2 文件数量。")
    parser.add_argument("--max-sen12-s1asc", type=int, default=None, help="限制 Sen12 S1 升轨文件数量。")
    parser.add_argument("--max-sen12-s1dsc", type=int, default=None, help="限制 Sen12 S1 降轨文件数量。")
    parser.add_argument("--clean", action="store_true", help="构建前清理输出目录。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    if args.clean:
        clean_dir(out_dir)
    ensure_dir(out_dir / "intermediate")

    rows: list[dict] = []
    sen12_root = Path(args.sen12_root)
    if sen12_root.exists():
        rows.extend(sen12_manifest_rows(sen12_root, args.max_sen12_s2, args.max_sen12_s1asc, args.max_sen12_s1dsc))
    else:
        print(f"[警告] 未找到 Sen12Landslides 目录：{sen12_root}")

    gdcld_root = Path(args.gdcld_root)
    if gdcld_root.exists():
        rows.extend(gdcld_manifest_rows(gdcld_root))
    else:
        print(f"[警告] 未找到 GDCLD 目录：{gdcld_root}")

    manifest_path = out_dir / "intermediate" / "source_manifest.jsonl"
    write_jsonl(manifest_path, rows)
    print(f"已写入数据源清单：{manifest_path}，共 {len(rows)} 条。")


if __name__ == "__main__":
    main()
