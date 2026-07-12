#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""步骤 1-4：真实预处理并物化自包含 benchmark 数据。

用途：读取 source 索引中的
datasets/ 原始路径，按模态规则生成 benchmark 内部 .npy 数据，并写出最终
训练索引，保证训练阶段不再读取 datasets/。
主要输入：indexes/source_all.jsonl 与 datasets/ 原始数据文件。
主要输出：data/{split}/{dataset_name}/{sample_id}/ 下的 .npy 数据、
indexes/all.jsonl、train.jsonl、val.jsonl、test.jsonl、unlabeled.jsonl。
写入行为：不会改写 datasets/；只在 benchmark 目标目录内写物化数据与最终索引。
所属流程：benchmark 构建 1-4；必须先通过 source 索引验证。
推荐运行命令：python scripts/1-benchmark/1-4_preprocess_samples.py --benchmark-dir benchmark/multisource_landslide_v2_small --strategy materialize
"""

from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import rasterio
import xarray as xr
import yaml
from PIL import Image
from PIL import ImageDraw
from tqdm import tqdm

from geohazard_benchmark_common import (
    DEFAULT_BENCHMARK_ROOT,
    choose_bucket,
    ensure_dir,
    final_index_paths,
    is_tif_path,
    project_path_arg,
    resolve_repo_path,
    source_index_paths,
    to_repo_rel,
    write_json,
    write_jsonl,
    write_split_indexes,
)


PREVIEW_MAX_SIDE = 1024
PREVIEW_TILE_SIDE = 320


def build_preprocess_config() -> dict[str, Any]:
    """生成结构化预处理配置，再由 PyYAML 写出。"""
    return {
        "version": "multisource_landslide_v2",
        "storage": {
            "data_format": "npy",
            "layout": "data/{split}/{dataset_name}/{sample_id}/",
            "modalities_dir": "modalities",
            "mask_dir": "mask",
        },
        "mask": {
            "binarize_rule": "mask > 0",
            "dtype": "uint8",
            "shape_policy": "2D mask 保存为 [1,H,W]",
        },
        "size_policy": {
            "strategy": "keep_original_then_bucket_metadata",
            "buckets": [32, 64, 128, 224, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768],
            "write_valid_pixel_mask": False,
        },
        "preview": {
            "enabled": True,
            "max_side": PREVIEW_MAX_SIDE,
            "tile_side": PREVIEW_TILE_SIDE,
            "files": [
                "visual.png",
                "mask.png",
                "overlay.png",
                "modalities.png",
                "multispectral_true_color.png",
                "multispectral_false_color_nir.png",
                "multispectral_swir.png",
                "multispectral_band_grid.png",
            ],
            "note": "preview 仅用于人工质检，不作为训练输入。",
        },
        "normalization": {
            "product_type:rgb": {"dtype": "uint8_or_float32", "method": "preserve_uint8_else_robust_percentile"},
            "product_type:multiband_optical": {"dtype": "float32", "method": "reflectance_aware"},
            "product_type:surface_reflectance": {"dtype": "float32", "method": "reflectance_aware"},
            "product_type:sar_backscatter": {"dtype": "float32", "method": "clip_-50_10_then_scale"},
            "product_type:elevation/slope/aspect/curvature": {"dtype": "float32", "method": "robust_percentile_scale"},
            "product_type:los_velocity": {"dtype": "float32", "method": "dataset_fixed_signed_symmetric_clip"},
        },
    }


def source_path(path_ref: str | None) -> tuple[Path, int | None]:
    """解析 path 或 path::index 引用。"""
    if not path_ref:
        raise ValueError("源路径为空")
    path_part, _, idx_part = path_ref.partition("::")
    path = resolve_repo_path(path_part)
    if path is None:
        raise ValueError(f"无法解析源路径: {path_ref}")
    return path, int(idx_part) if idx_part else None


def read_raster_array(path: Path, indexes: int | list[int] | None = None) -> Any:
    """使用 rasterio 读取 tif/tiff/geotiff 栅格。"""
    with rasterio.open(path) as src:
        if indexes is None:
            return src.read()
        return src.read(indexes)


def to_chw(array: Any) -> Any:
    """把常见 HWC/HW 数组整理为 CHW。"""
    arr = np.asarray(array)
    if arr.ndim == 2:
        return arr[np.newaxis, :, :]
    if arr.ndim == 3:
        # 如果最后一维看起来是通道，则转成 CHW；否则保持原状。
        if arr.shape[-1] <= 32 and arr.shape[0] > 32 and arr.shape[1] > 32:
            return np.transpose(arr, (2, 0, 1))
        return arr
    return arr


def read_hdf5_array(path: Path, internal_key: Any) -> Any:
    with h5py.File(path, "r") as f:
        if not isinstance(internal_key, str) or internal_key not in f:
            raise KeyError(f"{path} 缺少 HDF5 dataset: {internal_key}")
        data = f[internal_key][()]
    return data


def sen12_event_time_index(ds: Any) -> int:
    """选择与滑坡标注对应的单一事件时刻；失败时取中间时间片。"""
    time_len = int(ds.sizes.get("time", 1))
    raw = ds.attrs.get("pre_post_dates")
    if raw:
        try:
            parsed = ast.literal_eval(str(raw))
            post = int(parsed.get("post"))
            if 0 <= post < time_len:
                return post
        except Exception:
            pass
    return max(0, min(time_len - 1, time_len // 2))


def read_netcdf_vars(path: Path, keys: list[str]) -> Any:
    arrays = []
    with xr.open_dataset(path) as ds:
        t = sen12_event_time_index(ds)
        for key in keys:
            if key not in ds:
                continue
            var = ds[key]
            if "time" in var.dims:
                var = var.isel(time=t)
            arrays.append(np.asarray(var.values))
    if not arrays:
        raise ValueError(f"NetCDF 中未找到变量: {keys}")
    return np.stack(arrays, axis=0)


def split_landslide4sense_array(path: Path, raw: Any, name: str) -> Any:
    """严格按官方 B1-B12/slope/DEM 通道顺序拆分 Landslide4Sense。"""
    arr = to_chw(raw)
    if arr.ndim != 3 or arr.shape[0] != 14:
        raise ValueError(f"Landslide4Sense img 必须是 14 通道，当前 {path} -> shape={arr.shape}")
    if name == "multispectral":
        return arr[:12, :, :]
    if name == "slope":
        return arr[12:13, :, :]
    if name == "dem":
        return arr[13:14, :, :]
    raise ValueError(f"Landslide4Sense 不支持的模态名: {name}")


def read_modality_array(sample: dict[str, Any], name: str, info: dict[str, Any]) -> Any:
    fmt = str(info.get("format"))
    path, idx = source_path(info.get("path"))
    if is_tif_path(path) or fmt in {"geotiff", "geotiff_or_image"}:
        arr = read_raster_array(path)
        if fmt == "geotiff_or_image":
            if arr.shape[0] >= 3:
                return arr[:3]
            if arr.shape[0] == 1:
                return np.repeat(arr, 3, axis=0)
        return arr
    if fmt in {"image", "png"}:
        with Image.open(path) as img:
            return to_chw(img.convert("RGB"))
    if fmt == "npy":
        arr = np.load(path, mmap_mode="r")
        return np.asarray(arr[idx]) if idx is not None else np.asarray(arr)
    if fmt == "hdf5":
        raw = read_hdf5_array(path, info.get("internal_key"))
        if sample.get("dataset_name") == "landslide4sense":
            return split_landslide4sense_array(path, raw, name)
        return to_chw(raw)
    if fmt == "netcdf":
        keys = info.get("internal_key")
        if isinstance(keys, str):
            keys = [keys]
        return read_netcdf_vars(path, list(keys or []))
    raise ValueError(f"暂不支持的模态格式: {fmt}")


def read_mask_array(info: dict[str, Any]) -> Any:
    fmt = str(info.get("format"))
    path, idx = source_path(info.get("path"))
    if is_tif_path(path) or fmt == "geotiff":
        arr = read_raster_array(path, 1)
    elif fmt in {"png", "image"}:
        with Image.open(path) as img:
            arr = np.asarray(img.convert("L"))
    elif fmt == "npy":
        src = np.load(path, mmap_mode="r")
        arr = np.asarray(src[idx]) if idx is not None else np.asarray(src)
    elif fmt == "hdf5":
        arr = read_hdf5_array(path, info.get("internal_key"))
    elif fmt == "netcdf":
        arr = read_netcdf_vars(path, [str(info.get("internal_key") or "MASK")])
    else:
        raise ValueError(f"暂不支持的 mask 格式: {fmt}")
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    elif arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[:, :, 0]
    return (arr > 0).astype(np.uint8)[np.newaxis, :, :]


def robust_scale(arr: Any, low: float = 1.0, high: float = 99.0) -> Any:
    arr = arr.astype(np.float32, copy=False)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    lo, hi = np.percentile(finite, [low, high])
    if math.isclose(float(lo), float(hi)):
        return np.zeros_like(arr, dtype=np.float32)
    arr = np.clip(arr, lo, hi)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


def normalize_reflectance_like(arr: Any) -> tuple[Any, str]:
    """处理已知 Sentinel-2 反射率：兼容 0..1、0..10000 和非标准小范围。"""
    data = np.asarray(arr, dtype=np.float32)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return np.zeros_like(data, dtype=np.float32), "reflectance_all_nonfinite_to_zero"
    amin = float(np.nanmin(finite))
    p99 = float(np.percentile(finite, 99))
    if amin >= 0.0 and p99 <= 1.5:
        return np.clip(data, 0.0, 1.0).astype(np.float32), "preserve_0_1_reflectance"
    if p99 >= 1000.0:
        return (np.clip(data, 0.0, 10000.0) / 10000.0).astype(np.float32), "clip_0_10000_then_scale"
    return robust_scale_channels(data), "reflectance_robust_percentile_nonstandard_range"


def _normalization(method: str, *, scope: str = "sample", **parameters: Any) -> dict[str, Any]:
    return {"method": method, "scope": scope, "parameters": parameters}


def normalize_modality(
    name: str,
    arr: Any,
    sample: dict[str, Any] | None = None,
    info: dict[str, Any] | None = None,
    normalization_stats: dict[str, Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    arr = np.asarray(arr)
    dataset_name = str((sample or {}).get("dataset_name") or "")
    value_encoding = str((info or {}).get("value_encoding") or "")
    family = str((info or {}).get("family") or "")
    product_type = str((info or {}).get("product_type") or "")

    if dataset_name == "multimodal-landslide-dataset" and name == "optical_rgb":
        return robust_scale_channels(arr), _normalization("robust_percentile_channels", source="multimodal_sentinel2_rgb_int16")
    if dataset_name == "landslide4sense" and name == "multispectral":
        data, method = normalize_reflectance_like(arr)
        if method == "reflectance_robust_percentile_nonstandard_range":
            method = "landslide4sense_s2_robust_percentile_nonstandard_range"
        else:
            method = f"landslide4sense_s2_{method}"
        return data, _normalization(method)
    if dataset_name == "Sen12Landslides" and name == "multispectral":
        data, method = normalize_reflectance_like(arr)
        return data, _normalization(f"sen12_s2_{method}")

    if product_type == "rgb":
        if arr.dtype != np.uint8:
            if value_encoding:
                return robust_scale_channels(arr), _normalization("robust_percentile_channels", source=value_encoding)
            return robust_scale_channels(arr), _normalization("robust_percentile_channels", source="non_uint8_rgb")
        return arr, _normalization("preserve_rgb_values", scope="none")
    if product_type in {"surface_reflectance", "multiband_optical"}:
        data, method = normalize_reflectance_like(arr)
        return data, _normalization(f"generic_multispectral_{method}")
    if product_type in {"slope", "aspect", "curvature"}:
        return robust_scale(arr), _normalization(f"terrain_{product_type}_robust_percentile")
    if product_type == "sar_backscatter":
        arr = np.clip(arr.astype(np.float32), -50, 10)
        return ((arr + 50) / 60.0).astype(np.float32), _normalization(
            "linear_clip_scale", scope="product", clip_min=-50.0, clip_max=10.0, source_units="dB"
        )
    if product_type == "elevation":
        return robust_scale(arr), _normalization("terrain_robust_percentile")
    if product_type == "los_velocity" or family == "deformation":
        arr = arr.astype(np.float32, copy=False)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return np.zeros_like(arr, dtype=np.float32), _normalization("signed_symmetric_fixed_clip", scope="dataset", clip_abs=0.0)
        dataset_name = str((sample or {}).get("dataset_name") or "unknown")
        stats_key = f"{dataset_name}:{name}"
        bound = float(((normalization_stats or {}).get(stats_key) or {}).get("clip_abs") or 0.0)
        if bound <= 0:
            raise ValueError(f"缺少 InSAR 数据集级固定裁剪范围: {stats_key}")
        return (np.clip(arr, -bound, bound) / bound).astype(np.float32), _normalization(
            "signed_symmetric_fixed_clip", scope="dataset", clip_abs=bound, source_units=(info or {}).get("units")
        )
    return arr.astype(np.float32), _normalization("float32_no_extra_normalization", scope="none")


def source_valid_mask(arr: Any, info: dict[str, Any]) -> Any:
    """在任何归一化前锁定 finite/nodata 区域，避免无效值被零填充后泄漏。"""
    values = np.asarray(arr)
    if values.ndim == 2:
        values = values[None]
    if values.ndim != 3:
        raise ValueError(f"valid mask 需要 CHW，当前 shape={values.shape}")
    valid = np.isfinite(values).all(axis=0)
    nodata = (info.get("valid_mask") or {}).get("nodata_value")
    if isinstance(nodata, (int, float)) and np.isfinite(float(nodata)):
        valid &= ~np.isclose(values.astype(np.float64), float(nodata), rtol=0.0, atol=1.0e-8).all(axis=0)
    return valid.astype(np.uint8)[None]


def compute_dataset_normalization_stats(source_samples: list[dict[str, Any]]) -> dict[str, Any]:
    """用每个样本绝对值 98 分位的中位数建立固定 InSAR 显示/训练尺度。"""
    candidates: dict[str, list[float]] = {}
    for sample in tqdm(source_samples, desc="统计固定归一化范围", unit="sample"):
        for name, info in (sample.get("source_modalities") or {}).items():
            if info.get("product_type") != "los_velocity" and info.get("family") != "deformation":
                continue
            arr = np.asarray(read_modality_array(sample, name, info), dtype=np.float32)
            finite = np.abs(arr[np.isfinite(arr)])
            if finite.size:
                value = float(np.percentile(finite, 98))
                if value > 0:
                    key = f"{sample.get('dataset_name', 'unknown')}:{name}"
                    candidates.setdefault(key, []).append(value)
    return {
        key: {
            "method": "median_of_sample_abs_p98",
            "clip_abs": float(np.median(values)),
            "num_samples": len(values),
        }
        for key, values in sorted(candidates.items())
    }


def bbox_from_binary_mask(mask: Any) -> tuple[list[int] | None, str, int, bool]:
    arr = np.asarray(mask)
    if arr.ndim == 3:
        arr2 = arr[0]
    else:
        arr2 = arr
    ys, xs = np.where(arr2 > 0)
    positive = int(xs.size)
    if positive == 0:
        return None, "empty_mask", 0, True
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())], "derived", positive, False


def stretch_uint8(arr: Any, low: float = 2.0, high: float = 98.0) -> Any:
    """把单通道或多通道数组按分位数拉伸到 uint8。"""
    data = np.asarray(arr, dtype=np.float32)
    out = np.zeros_like(data, dtype=np.float32)
    if data.ndim == 2:
        finite = data[np.isfinite(data)]
        if finite.size == 0:
            return np.zeros(data.shape, dtype=np.uint8)
        lo, hi = np.percentile(finite, [low, high])
        if math.isclose(float(lo), float(hi)):
            return np.zeros(data.shape, dtype=np.uint8)
        return np.clip((data - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    for idx in range(data.shape[0]):
        out[idx] = stretch_uint8(data[idx], low, high)
    return out.astype(np.uint8)


def robust_scale_channels(arr: Any, low: float = 2.0, high: float = 98.0) -> Any:
    """按通道稳健拉伸到 0..1，用于非 uint8 RGB 和非标准多光谱范围。"""
    data = np.asarray(arr, dtype=np.float32)
    out = np.zeros_like(data, dtype=np.float32)
    if data.ndim == 2:
        finite = data[np.isfinite(data)]
        if finite.size == 0:
            return np.zeros(data.shape, dtype=np.float32)
        lo, hi = np.percentile(finite, [low, high])
        if math.isclose(float(lo), float(hi)):
            return np.zeros(data.shape, dtype=np.float32)
        return np.clip((data - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
    for idx in range(data.shape[0]):
        out[idx] = robust_scale_channels(data[idx], low, high)
    return out.astype(np.float32)


def chw_to_hwc_rgb(arr: Any) -> Any:
    """把 CHW/HW 数组转为 HWC RGB uint8。"""
    data = np.asarray(arr)
    if data.ndim == 2:
        gray = stretch_uint8(data)
        return np.stack([gray, gray, gray], axis=-1)
    if data.ndim == 3:
        if data.shape[0] == 1:
            gray = stretch_uint8(data[0])
            return np.stack([gray, gray, gray], axis=-1)
        rgb = data[:3]
        if rgb.dtype == np.uint8 and rgb.max(initial=0) > 1:
            rgb8 = rgb
        else:
            rgb8 = stretch_uint8(rgb)
        return np.transpose(rgb8[:3], (1, 2, 0))
    raise ValueError(f"无法转为 RGB preview: shape={data.shape}")


def preview_multispectral_rgb(arr: Any, band_names: list[str]) -> Any:
    """按 Sentinel-2 真彩组合生成多光谱 RGB preview。"""
    return preview_multispectral_composite(arr, band_names, ("B04", "B03", "B02"))


def multispectral_band_index(band_names: list[str], band: str) -> int | None:
    """兼容 B04/B4 等 Sentinel-2 波段命名。"""
    names = [name.lower() for name in band_names]
    candidates = {band.lower()}
    if band.upper().startswith("B0"):
        candidates.add("b" + band.upper()[2:].lstrip("0").lower())
    elif band.upper().startswith("B"):
        number = band.upper()[1:]
        if number.isdigit():
            candidates.add(f"b{int(number):02d}")
    for candidate in candidates:
        if candidate in names:
            return names.index(candidate)
    return None


def preview_multispectral_composite(arr: Any, band_names: list[str], bands: tuple[str, str, str]) -> Any:
    """按指定三波段组合生成多光谱 RGB preview。"""
    data = np.asarray(arr)
    idxs: list[int] = []
    for band in bands:
        idx = multispectral_band_index(band_names, band)
        if idx is not None and idx < data.shape[0]:
            idxs.append(idx)
    if not idxs:
        idxs = list(range(min(3, data.shape[0])))
    while len(idxs) < 3:
        idxs.append(idxs[-1] if idxs else 0)
    return chw_to_hwc_rgb(data[idxs[:3]])


def preview_multispectral_composites(arr: Any, band_names: list[str]) -> dict[str, Any]:
    """生成 Sentinel-2 常用真彩、近红外假彩和 SWIR 组合。"""
    return {
        "true_color": preview_multispectral_composite(arr, band_names, ("B04", "B03", "B02")),
        "false_color_nir": preview_multispectral_composite(arr, band_names, ("B08", "B04", "B03")),
        "swir": preview_multispectral_composite(arr, band_names, ("B12", "B08", "B04")),
    }


def preview_for_modality(name: str, arr: Any, meta: dict[str, Any]) -> Any:
    """生成单个模态的 RGB preview。"""
    if name == "multispectral":
        return preview_multispectral_rgb(arr, meta.get("band_names") or [])
    if name in {"optical_rgb", "optical_multiband"}:
        return chw_to_hwc_rgb(arr)
    data = np.asarray(arr)
    if data.ndim == 3:
        data = np.nanmean(data, axis=0)
    gray = stretch_uint8(data)
    return np.stack([gray, gray, gray], axis=-1)


def resize_preview(image: Any, max_side: int = PREVIEW_MAX_SIDE) -> Image.Image:
    """按最大边缩小 preview，避免超大 PNG。"""
    pil = Image.fromarray(np.asarray(image, dtype=np.uint8))
    width, height = pil.size
    side = max(width, height)
    if side <= max_side:
        return pil
    scale = max_side / float(side)
    size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return pil.resize(size, Image.Resampling.BILINEAR)


def save_preview_png(path: Path, image: Any, max_side: int = PREVIEW_MAX_SIDE) -> str:
    ensure_dir(path.parent)
    resize_preview(image, max_side=max_side).save(path)
    return to_repo_rel(path) or path.as_posix()


def mask_preview_rgb(mask: Any) -> Any:
    arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = arr[0]
    binary = (arr > 0).astype(np.uint8) * 255
    return np.stack([binary, binary, binary], axis=-1)


def overlay_preview(visual: Any, mask: Any) -> Any:
    base = np.asarray(visual, dtype=np.uint8)
    mask_arr = np.asarray(mask)
    if mask_arr.ndim == 3:
        mask_arr = mask_arr[0]
    height = min(base.shape[0], mask_arr.shape[0])
    width = min(base.shape[1], mask_arr.shape[1])
    base = base[:height, :width].copy()
    binary = mask_arr[:height, :width] > 0
    red = np.zeros_like(base)
    red[..., 0] = 255
    base[binary] = (0.55 * base[binary] + 0.45 * red[binary]).astype(np.uint8)
    return base


def labeled_tile(name: str, image: Any) -> Image.Image:
    tile = resize_preview(image, max_side=PREVIEW_TILE_SIDE).convert("RGB")
    canvas = Image.new("RGB", (tile.width, tile.height + 24), color=(0, 0, 0))
    canvas.paste(tile, (0, 24))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 5), name, fill=(255, 255, 255))
    return canvas


def save_modalities_grid(path: Path, tiles: list[tuple[str, Any]]) -> str | None:
    if not tiles:
        return None
    pil_tiles = [labeled_tile(name, image) for name, image in tiles]
    cols = min(3, len(pil_tiles))
    rows = int(math.ceil(len(pil_tiles) / cols))
    cell_w = max(tile.width for tile in pil_tiles)
    cell_h = max(tile.height for tile in pil_tiles)
    grid = Image.new("RGB", (cell_w * cols, cell_h * rows), color=(20, 20, 20))
    for idx, tile in enumerate(pil_tiles):
        x = (idx % cols) * cell_w
        y = (idx // cols) * cell_h
        grid.paste(tile, (x, y))
    ensure_dir(path.parent)
    grid.save(path)
    return to_repo_rel(path) or path.as_posix()


def multispectral_band_grid(arr: Any, band_names: list[str], max_bands: int = 12) -> Any:
    """把多光谱各波段按灰度小图排成一张质检图。"""
    data = np.asarray(arr)
    if data.ndim != 3:
        raise ValueError(f"多光谱 band grid 需要 CHW，当前 shape={data.shape}")
    tiles: list[tuple[str, Any]] = []
    for idx in range(min(data.shape[0], max_bands)):
        label = band_names[idx] if idx < len(band_names) else f"band_{idx + 1}"
        gray = stretch_uint8(data[idx])
        tiles.append((label, np.stack([gray, gray, gray], axis=-1)))
    if not tiles:
        raise ValueError("多光谱 band grid 没有可用波段")
    pil_tiles = [labeled_tile(name, image) for name, image in tiles]
    cols = min(4, len(pil_tiles))
    rows = int(math.ceil(len(pil_tiles) / cols))
    cell_w = max(tile.width for tile in pil_tiles)
    cell_h = max(tile.height for tile in pil_tiles)
    grid = Image.new("RGB", (cell_w * cols, cell_h * rows), color=(20, 20, 20))
    for idx, tile in enumerate(pil_tiles):
        x = (idx % cols) * cell_w
        y = (idx // cols) * cell_h
        grid.paste(tile, (x, y))
    return np.asarray(grid)


def choose_visual_for_preview(arrays: dict[str, Any], metas: dict[str, dict[str, Any]]) -> Any | None:
    """选择一个主视觉底图，优先 RGB，其次多光谱真彩。"""
    if "optical_rgb" in arrays:
        return preview_for_modality("optical_rgb", arrays["optical_rgb"], metas["optical_rgb"])
    if "multispectral" in arrays:
        return preview_for_modality("multispectral", arrays["multispectral"], metas["multispectral"])
    if arrays:
        first_name = next(iter(arrays))
        return preview_for_modality(first_name, arrays[first_name], metas[first_name])
    return None


def build_previews(sample: dict[str, Any], sample_dir: Path, arrays: dict[str, Any], metas: dict[str, dict[str, Any]], mask: Any | None) -> tuple[dict[str, str], list[str]]:
    """写出 visual/mask/overlay/modalities preview；失败只记录，不影响物化样本。"""
    preview_dir = ensure_dir(sample_dir / "preview")
    paths: dict[str, str] = {}
    errors: list[str] = []
    visual = None
    try:
        visual = choose_visual_for_preview(arrays, metas)
        if visual is not None:
            paths["visual"] = save_preview_png(preview_dir / "visual.png", visual)
    except Exception as exc:
        errors.append(f"visual preview 失败: {exc}")

    try:
        tiles = []
        for name, arr in arrays.items():
            if name == "multispectral":
                band_names = metas[name].get("band_names") or []
                composites = preview_multispectral_composites(arr, band_names)
                for comp_name, image in composites.items():
                    paths[f"multispectral_{comp_name}"] = save_preview_png(preview_dir / f"multispectral_{comp_name}.png", image)
                    tiles.append((f"multispectral_{comp_name}", image))
                band_grid = multispectral_band_grid(arr, band_names)
                paths["multispectral_band_grid"] = save_preview_png(preview_dir / "multispectral_band_grid.png", band_grid, max_side=max(PREVIEW_MAX_SIDE, PREVIEW_TILE_SIDE * 4))
                tiles.append(("multispectral_band_grid", band_grid))
            else:
                tiles.append((name, preview_for_modality(name, arr, metas[name])))
        grid_path = save_modalities_grid(preview_dir / "modalities.png", tiles)
        if grid_path:
            paths["modalities"] = grid_path
    except Exception as exc:
        errors.append(f"modalities preview 失败: {exc}")

    if mask is not None:
        try:
            paths["mask"] = save_preview_png(preview_dir / "mask.png", mask_preview_rgb(mask))
        except Exception as exc:
            errors.append(f"mask preview 失败: {exc}")
        try:
            if visual is not None:
                paths["overlay"] = save_preview_png(preview_dir / "overlay.png", overlay_preview(visual, mask))
        except Exception as exc:
            errors.append(f"overlay preview 失败: {exc}")
    return paths, errors


def materialize_sample(
    sample: dict[str, Any],
    benchmark_dir: Path,
    normalization_stats: dict[str, Any],
) -> dict[str, Any]:
    sample_dir = benchmark_dir / "data" / sample["split"] / sample["dataset_name"] / sample["sample_id"]
    modality_dir = ensure_dir(sample_dir / "modalities")
    mask_dir = ensure_dir(sample_dir / "mask")
    ensure_dir(sample_dir / "preview")

    final_modalities: dict[str, dict[str, Any]] = {}
    preview_arrays: dict[str, Any] = {}
    for name, info in sample.get("source_modalities", {}).items():
        source_arr = read_modality_array(sample, name, info)
        valid_arr = source_valid_mask(source_arr, info)
        arr, norm = normalize_modality(name, source_arr, sample, info, normalization_stats)
        out_path = modality_dir / f"{name}.npy"
        valid_path = modality_dir / f"{name}_valid.npy"
        np.save(out_path, arr)
        np.save(valid_path, valid_arr)
        final_modalities[name] = {
            "path": to_repo_rel(out_path),
            "format": "npy",
            "internal_key": None,
            "band_names": info.get("band_names") or [],
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "family": info["family"],
            "sensor": info["sensor"],
            "product_type": info["product_type"],
            "band_metadata": info["band_metadata"],
            "native_gsd_m": info.get("native_gsd_m"),
            "units": info["units"],
            "signed": bool(info["signed"]),
            "orbit": info["orbit"],
            "quality": float(info["quality"]),
            "available": True,
            "role": info.get("role"),
            "source": info.get("source"),
            "value_encoding": info.get("value_encoding"),
            "normalization": norm,
            "valid_mask": {
                "path": to_repo_rel(valid_path),
                "format": "npy",
                "shape": list(valid_arr.shape),
                "dtype": "uint8",
                "status": "materialized_before_normalization",
                "nodata_value": (info.get("valid_mask") or {}).get("nodata_value"),
            },
        }
        preview_arrays[name] = arr

    source_mask = sample.get("source_mask")
    final_mask = None
    materialized_mask = None
    if sample.get("supervision", "mask") == "mask" and source_mask:
        mask = read_mask_array(source_mask)
        bbox, bbox_status, positive, empty_mask = bbox_from_binary_mask(mask)
        mask_path = mask_dir / "mask.npy"
        np.save(mask_path, mask.astype(np.uint8))
        materialized_mask = mask
        final_mask = {
            "path": to_repo_rel(mask_path),
            "format": "npy",
            "internal_key": None,
            "label_type": "binary_landslide",
            "shape": list(mask.shape),
            "dtype": "uint8",
            "positive_pixels": positive,
            "empty_mask": empty_mask,
            "bbox_xyxy": bbox,
            "bbox_status": bbox_status,
            "binarize_rule": "mask > 0",
        }

    sizes = [m["shape"][-2:] for m in final_modalities.values() if len(m.get("shape") or []) >= 2]
    if final_mask and len(final_mask.get("shape") or []) >= 2:
        sizes.append(final_mask["shape"][-2:])
    original_size = sizes[0] if sizes else None
    shape_mismatch = len({tuple(size) for size in sizes if size}) > 1

    final = dict(sample)
    final.pop("source_modalities", None)
    final.pop("source_mask", None)
    final["modalities"] = final_modalities
    final["mask"] = final_mask
    final["spatial"] = {
        **(sample.get("spatial") or {}),
        "original_size": original_size,
        "bucket_size": choose_bucket(original_size),
        "valid_pixel_mask_required": False,
        "valid_pixel_mask": {"path": None, "status": "not_materialized_no_padding_applied"},
    }
    if shape_mismatch:
        final["spatial"]["shape_mismatch_warning"] = {
            "message": "模态或 mask 尺寸不完全一致，训练 dataloader 需要按任务策略裁剪或 padding。",
            "shapes": sizes,
        }
    flags = set(final.get("quality_flags") or [])
    flags.add("materialized_to_benchmark_npy")
    if shape_mismatch:
        flags.add("shape_mismatch_warning")
    preview_paths, preview_errors = build_previews(final, sample_dir, preview_arrays, final_modalities, materialized_mask)
    if preview_errors:
        flags.add("preview_failed")
    final["quality_flags"] = sorted(flags)
    final["preview"] = {
        "max_side": PREVIEW_MAX_SIDE,
        "paths": preview_paths,
        "errors": preview_errors,
    }
    final["provenance"] = {
        "source_modalities": sample.get("source_modalities", {}),
        "source_mask": sample.get("source_mask"),
        "source_key": sample.get("source_key"),
        "materialized_dir": to_repo_rel(sample_dir),
    }
    write_json(sample_dir / "sample_meta.json", final)
    return final


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="物化自包含 benchmark 数据，并生成最终训练索引。")
    parser.add_argument("--benchmark-dir", type=project_path_arg, default=DEFAULT_BENCHMARK_ROOT, help="后缀式 small 或 full benchmark 输出目录。")
    parser.add_argument("--strategy", choices=["materialize"], default="materialize", help="预处理输出策略；当前默认并只支持 materialize。")
    parser.add_argument("--max-failures", type=int, default=20, help="允许跳过的单样本失败数量；超过后停止。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_all_path = source_index_paths(args.benchmark_dir)["all"]
    source_samples = [json.loads(line) for line in source_all_path.read_text(encoding="utf-8").splitlines() if line.strip()] if source_all_path.exists() else []
    if not source_samples:
        raise SystemExit(f"未找到源索引: {source_all_path}")

    normalization_stats = compute_dataset_normalization_stats(source_samples)
    write_json(args.benchmark_dir / "reports" / "normalization_stats.json", normalization_stats)
    (args.benchmark_dir / "preprocess_config.yaml").write_text(
        yaml.safe_dump(build_preprocess_config(), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    final_samples: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    progress = tqdm(source_samples, desc="物化样本", unit="sample")
    for sample in progress:
        try:
            final_samples.append(materialize_sample(sample, args.benchmark_dir, normalization_stats))
        except Exception as exc:
            failures.append({"sample_id": sample.get("sample_id"), "dataset_name": sample.get("dataset_name"), "error": str(exc)})
            if len(failures) > args.max_failures:
                break
        progress.set_postfix({"成功": len(final_samples), "失败": len(failures)})

    if failures:
        write_jsonl(args.benchmark_dir / "reports" / "materialize_failures.jsonl", failures)
    if len(failures) > args.max_failures:
        raise SystemExit(f"物化失败数量超过 --max-failures={args.max_failures}，请查看 reports/materialize_failures.jsonl")

    write_split_indexes(args.benchmark_dir, final_samples)
    materialized_files = list((args.benchmark_dir / "data").glob("**/*.npy"))
    preview_files = list((args.benchmark_dir / "data").glob("**/preview/*.png"))
    preview_failures = [sample for sample in final_samples if "preview_failed" in (sample.get("quality_flags") or [])]
    total_bytes = sum(path.stat().st_size for path in materialized_files)
    report = {
        "说明": "最终索引已指向 benchmark 内部 data/ 下的 .npy 文件，训练阶段不需要读取 datasets/。",
        "strategy": args.strategy,
        "num_source_samples": len(source_samples),
        "num_final_samples": len(final_samples),
        "num_failures": len(failures),
        "num_npy_files": len(materialized_files),
        "num_preview_files": len(preview_files),
        "num_preview_failures": len(preview_failures),
        "materialized_bytes": total_bytes,
        "preprocess_config": to_repo_rel(args.benchmark_dir / "preprocess_config.yaml"),
        "normalization_stats": to_repo_rel(args.benchmark_dir / "reports" / "normalization_stats.json"),
        "final_index": to_repo_rel(final_index_paths(args.benchmark_dir)["all"]),
    }
    write_json(args.benchmark_dir / "reports" / "preprocess_report.json", report)
    print(f"物化完成: 成功 {len(final_samples)} 条，失败 {len(failures)} 条 -> {to_repo_rel(final_index_paths(args.benchmark_dir)['all'])}")


if __name__ == "__main__":
    main()
