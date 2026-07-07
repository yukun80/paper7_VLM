#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""多源滑坡 benchmark 构建公共工具库。

脚本作用：为 scripts/1-benchmark/ 下的阶段脚本提供路径、JSONL、CSV、
抽样、样本 ID、专业遥感/数组读取和 Sen12 文件名配对等公共函数。
主要输入：各阶段脚本传入的 datasets/ 路径、benchmark/ 输出路径和样本记录。
主要输出：公共函数返回值；本文件本身不作为流程入口，也不单独生成文件。
是否改写原始数据：不会改写 datasets/ 原始数据。
典型用法：由 1-1_scan_sources.py、1-2_build_index.py 等脚本 import 后复用。
"""

from __future__ import annotations

import csv
import copy
import hashlib
import json
import random
import re
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import rasterio
from netCDF4 import Dataset
from PIL import Image
from rasterio.errors import NotGeoreferencedWarning


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASETS_ROOT = REPO_ROOT / "datasets"
DEFAULT_BENCHMARK_PREFIX = REPO_ROOT / "benchmark" / "multisource_landslide_v1"
DEFAULT_BENCHMARK_ROOT = DEFAULT_BENCHMARK_PREFIX.with_name(f"{DEFAULT_BENCHMARK_PREFIX.name}_small")
BUCKETS = [32, 64, 128, 224, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
LANDSLIDE4SENSE_S2_BANDS = [f"B{i}" for i in range(1, 13)]
LANDSLIDE4SENSE_MODALITY_SHAPES = {
    "multispectral": [12, 128, 128],
    "slope": [1, 128, 128],
    "dem": [1, 128, 128],
}


def is_tif_path(path: Path | str | None) -> bool:
    """判断是否为 tif/tiff 遥感栅格路径；这类文件统一用 rasterio 读取。"""
    if path is None:
        return False
    return Path(path).suffix.lower() in {".tif", ".tiff"}


MANIFEST_FIELDS = [
    "dataset_name",
    "split",
    "subset",
    "modalities",
    "file_format",
    "num_samples",
    "image_size",
    "gsd_m",
    "label_status",
    "region",
    "task_type",
    "warning",
]


def ensure_dir(path: Path) -> Path:
    """确保目录存在，并返回 Path，便于链式调用。"""
    path.mkdir(parents=True, exist_ok=True)
    return path


def benchmark_dir_for_mode(mode: str, prefix: Path | None = None) -> Path:
    """按后缀式命名返回 benchmark 目录，例如 multisource_landslide_v1_small。"""
    base = prefix or DEFAULT_BENCHMARK_PREFIX
    if mode not in {"small", "full"}:
        raise ValueError(f"未知 benchmark 模式: {mode}")
    return base.with_name(f"{base.name}_{mode}")


def to_repo_rel(path: Path | str | None) -> str | None:
    """把路径写成相对仓库根目录的形式，避免大量 resolve 带来的 I/O 开销。"""
    if path is None:
        return None
    p = Path(path)
    if not p.is_absolute():
        return p.as_posix()
    try:
        return p.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def resolve_repo_path(path_ref: str | None) -> Path | None:
    """解析索引中的路径引用；支持 path::index 这种虚拟样本写法。"""
    if not path_ref:
        return None
    base = path_ref.split("::", 1)[0]
    p = Path(base)
    if p.is_absolute():
        return p
    return REPO_ROOT / p


def is_relative_to(path: Path, parent: Path) -> bool:
    """兼容 Python 3.8+ 的 Path.is_relative_to。"""
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def path_is_inside_benchmark(path_ref: str | None, benchmark_dir: Path) -> bool:
    """判断索引里的训练读取路径是否位于当前 benchmark 目录内。"""
    path = resolve_repo_path(path_ref)
    return bool(path and is_relative_to(path, benchmark_dir))


def write_json(path: Path, data: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} 不是合法 JSONL: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def read_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def natural_key(value: str | Path) -> list[Any]:
    """自然排序键：image_2 会排在 image_10 前面。"""
    text = str(value)
    parts = re.split(r"(\d+)", text)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def stable_sample(rows: list[dict[str, Any]], limit: int | None, seed: int, key: str = "sample_id") -> list[dict[str, Any]]:
    """确定性抽样，保证 small 模式可复现。"""
    if limit is None or limit <= 0 or len(rows) <= limit:
        return rows
    ordered = sorted(rows, key=lambda row: str(row.get(key, "")))
    rng = random.Random(seed)
    rng.shuffle(ordered)
    return sorted(ordered[:limit], key=lambda row: str(row.get(key, "")))


def stable_sample_by_split(rows: list[dict[str, Any]], limit: int | None, seed: int) -> list[dict[str, Any]]:
    """small 模式按 split 抽样，避免某个 split 被抽空。"""
    if limit is None or limit <= 0:
        return rows
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("split", "unknown"))].append(row)
    sampled: list[dict[str, Any]] = []
    for idx, split in enumerate(sorted(groups)):
        sampled.extend(stable_sample(groups[split], limit, seed + idx))
    return sorted(sampled, key=lambda row: str(row.get("sample_id", "")))


def make_sample_id(dataset_name: str, *parts: Any) -> str:
    """生成短而稳定的样本 ID。"""
    raw = "::".join([dataset_name] + [str(part) for part in parts])
    clean = re.sub(r"[^0-9A-Za-z]+", "_", raw).strip("_").lower()
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]
    return f"{clean[:70]}_{digest}"


def hash_split(key: str, train_ratio: float = 0.8, val_ratio: float = 0.1) -> str:
    """对没有官方划分的数据做确定性 train/val/test 划分。"""
    value = int(hashlib.sha1(key.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    if value < train_ratio:
        return "train"
    if value < train_ratio + val_ratio:
        return "val"
    return "test"


def choose_bucket(size: list[int] | tuple[int, int] | None) -> int | None:
    """根据原始 H/W 选择 padding bucket，不在此处做真实 resize。"""
    if not size or len(size) != 2 or not all(isinstance(v, int) and v > 0 for v in size):
        return None
    side = max(size)
    for bucket in BUCKETS:
        if side <= bucket:
            return bucket
    return BUCKETS[-1]


def parse_npy_header(path: Path) -> dict[str, Any]:
    """用 numpy mmap 读取 .npy 元数据，不加载完整数组。"""
    arr = np.load(path, mmap_mode="r")
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "fortran_order": bool(arr.flags["F_CONTIGUOUS"]),
    }


def probe_image(path: Path) -> dict[str, Any]:
    """用 rasterio/Pillow 探测 PNG/TIFF/GeoTIFF 尺寸和通道数。"""
    info: dict[str, Any] = {"format": "unknown", "width": None, "height": None, "channels": None, "warning": ""}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NotGeoreferencedWarning)
            with rasterio.open(path) as src:
                info.update({
                    "format": src.driver,
                    "width": int(src.width),
                    "height": int(src.height),
                    "channels": int(src.count),
                    "dtype": ",".join(src.dtypes),
                })
                return info
    except Exception as rio_exc:
        if is_tif_path(path):
            info["warning"] = f"TIF 必须使用 rasterio 读取，但探测失败: {rio_exc}"
            return info
        try:
            with Image.open(path) as img:
                info.update({
                    "format": img.format or "image",
                    "width": int(img.width),
                    "height": int(img.height),
                    "channels": len(img.getbands()),
                    "dtype": str(img.mode),
                    "warning": f"rasterio 读取失败，已使用 Pillow: {rio_exc}",
                })
                return info
        except Exception as pil_exc:
            info["warning"] = f"图像探测失败: rasterio={rio_exc}; pillow={pil_exc}"
            return info


def hdf5_dataset_meta(path: Path, key: str) -> dict[str, Any]:
    """读取 HDF5 指定 dataset 的 shape/dtype，key 缺失时直接报错。"""
    with h5py.File(path, "r") as f:
        if key not in f:
            raise KeyError(f"{path} 缺少 HDF5 dataset: {key}")
        data = f[key]
        return {"shape": list(data.shape), "dtype": str(data.dtype)}


def hdf5_has_dataset(path: Path, key: str) -> bool:
    """检查 HDF5 指定 key 是否存在。"""
    try:
        with h5py.File(path, "r") as f:
            return key in f
    except Exception:
        return False


def sen12_read_annotated(path: Path) -> bool | None:
    """用 netCDF4 读取 Sen12 annotated 属性。"""
    try:
        with Dataset(path) as ds:
            value = getattr(ds, "annotated", None)
    except Exception:
        return None
    if isinstance(value, str):
        text = value.strip().lower()
        if text == "true":
            return True
        if text == "false":
            return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return None


def landslide4sense_source_modalities(path: Path) -> dict[str, dict[str, Any]]:
    """按官方 14 通道语义生成 Landslide4Sense 三个源模态。"""
    return {
        "multispectral": modality_entry(
            path,
            fmt="hdf5",
            band_names=LANDSLIDE4SENSE_S2_BANDS,
            shape=LANDSLIDE4SENSE_MODALITY_SHAPES["multispectral"],
            gsd_m=10,
            internal_key="img",
            role="sentinel2_multispectral",
            source="Sentinel-2_B1_B12",
        ),
        "slope": modality_entry(
            path,
            fmt="hdf5",
            band_names=["slope"],
            shape=LANDSLIDE4SENSE_MODALITY_SHAPES["slope"],
            gsd_m=10,
            internal_key="img",
            role="terrain_slope",
            source="ALOS_PALSAR_B13",
        ),
        "dem": modality_entry(
            path,
            fmt="hdf5",
            band_names=["DEM"],
            shape=LANDSLIDE4SENSE_MODALITY_SHAPES["dem"],
            gsd_m=10,
            internal_key="img",
            role="terrain_dem",
            source="ALOS_PALSAR_B14",
        ),
    }


def image_size_text(info: dict[str, Any] | None) -> str:
    if not info:
        return "unknown"
    width, height = info.get("width"), info.get("height")
    if width and height:
        return f"{height}x{width}"
    return "unknown"


def modality_entry(
    path: Path | str | None,
    *,
    fmt: str,
    band_names: list[str] | None = None,
    shape: list[int] | None = None,
    gsd_m: float | None = None,
    internal_key: str | list[str] | int | None = None,
    available: bool = True,
    role: str | None = None,
    source: str | None = None,
) -> dict[str, Any]:
    entry = {
        "path": to_repo_rel(path) if path else None,
        "format": fmt,
        "internal_key": internal_key,
        "band_names": band_names or [],
        "shape": shape,
        "gsd_m": gsd_m,
        "available": available,
        "role": role,
    }
    if source:
        entry["source"] = source
    return entry


def mask_entry(
    path: Path | str | None,
    *,
    fmt: str,
    shape: list[int] | None = None,
    internal_key: str | int | None = None,
    empty_mask: bool | None = None,
    bbox_xyxy: list[int] | None = None,
    bbox_status: str = "pending_pixel_read",
) -> dict[str, Any]:
    return {
        "path": to_repo_rel(path) if path else None,
        "format": fmt,
        "internal_key": internal_key,
        "label_type": "binary_landslide",
        "shape": shape,
        "positive_pixels": None,
        "empty_mask": empty_mask,
        "bbox_xyxy": bbox_xyxy,
        "bbox_status": bbox_status,
        "binarize_rule": "mask > 0",
    }


def instruction_for(task_type: str, modality_names: Iterable[str]) -> dict[str, str]:
    """第一阶段使用稳定短指令，避免自然语言模板过度发散。"""
    mods = sorted(set(modality_names))
    if task_type == "negative_landslide_segmentation":
        text = "Segment all landslide regions. If there is no landslide, output an empty mask."
        template = "negative_aware_landslide_v1"
    elif "insar_vel" in mods:
        text = "Segment landslide regions using optical, DEM, SAR, or InSAR deformation evidence."
        template = "multisource_evidence_landslide_v1"
    elif "sar_asc" in mods or "sar_dsc" in mods or "dem" in mods:
        text = "Segment landslide regions using the available multisource remote sensing evidence."
        template = "multisource_landslide_v1"
    else:
        text = "Segment all landslide regions."
        template = "generic_landslide_v1"
    return {"language": "en", "template_id": template, "text": text}


def make_sample(
    *,
    dataset_name: str,
    split: str,
    task_type: str,
    source_key: str,
    modalities: dict[str, dict[str, Any]],
    mask: dict[str, Any] | None,
    region: str | None = None,
    event_id: str | None = None,
    source_level: str = "patch",
    quality_flags: list[str] | None = None,
    subset: str | None = None,
    split_source: str = "official",
    supervision: str = "mask",
) -> dict[str, Any]:
    sizes = []
    for modality in modalities.values():
        shape = modality.get("shape")
        if isinstance(shape, list) and len(shape) >= 2:
            sizes.append(shape[-2:])
    if mask and isinstance(mask.get("shape"), list) and len(mask["shape"]) >= 2:
        sizes.append(mask["shape"][-2:])
    original_size = sizes[0] if sizes else None
    sample_id = make_sample_id(dataset_name, subset or "main", source_key)
    instruction = instruction_for(task_type, modalities.keys())
    return {
        "sample_id": sample_id,
        "dataset_name": dataset_name,
        "subset": subset or "main",
        "split": split,
        "split_source": split_source,
        "task_type": task_type,
        "source_key": source_key,
        "source_level": source_level,
        "region": region,
        "event_id": event_id,
        "modalities": modalities,
        "mask": mask,
        "instruction": instruction,
        "spatial": {
            "original_size": original_size,
            "bucket_size": choose_bucket(original_size),
            "gsd_m": infer_sample_gsd(modalities),
            "crs": None,
            "transform": None,
            "valid_pixel_mask_required": True,
        },
        "quality_flags": sorted(set(quality_flags or [])),
        "supervision": supervision,
    }


def infer_sample_gsd(modalities: dict[str, dict[str, Any]]) -> float | None:
    gsds = [m.get("gsd_m") for m in modalities.values() if m.get("gsd_m") is not None]
    return gsds[0] if gsds else None


def modality_combo(sample: dict[str, Any]) -> str:
    names = [name for name, data in sample.get("modalities", {}).items() if data.get("available", True)]
    return "+".join(sorted(names)) if names else "none"


def source_index_paths(benchmark_dir: Path) -> dict[str, Path]:
    """源索引路径：允许包含 datasets/ 原始路径，仅供物化阶段读取。"""
    index_dir = benchmark_dir / "indexes"
    return {
        "all": index_dir / "source_all.jsonl",
        "train": index_dir / "source_train.jsonl",
        "val": index_dir / "source_val.jsonl",
        "test": index_dir / "source_test.jsonl",
        "unlabeled": index_dir / "source_unlabeled.jsonl",
        "extended_pool": index_dir / "source_extended_pool.jsonl",
    }


def final_index_paths(benchmark_dir: Path) -> dict[str, Path]:
    """最终训练索引路径：所有训练读取 path 都必须指向 benchmark/ 内物化数据。"""
    index_dir = benchmark_dir / "indexes"
    return {
        "all": index_dir / "all.jsonl",
        "train": index_dir / "train.jsonl",
        "val": index_dir / "val.jsonl",
        "test": index_dir / "test.jsonl",
        "unlabeled": index_dir / "unlabeled.jsonl",
        "extended_pool": index_dir / "extended_pool.jsonl",
    }


def referring_index_paths(benchmark_dir: Path) -> dict[str, Path]:
    """指代表达训练索引路径：每一行对应一条 expression-level 监督样本。"""
    index_dir = benchmark_dir / "indexes"
    return {
        "all": index_dir / "referring_all.jsonl",
        "train": index_dir / "referring_train.jsonl",
        "val": index_dir / "referring_val.jsonl",
        "test": index_dir / "referring_test.jsonl",
        "unlabeled": index_dir / "referring_unlabeled.jsonl",
        "extended_pool": index_dir / "referring_extended_pool.jsonl",
    }


def split_index_paths(benchmark_dir: Path) -> dict[str, Path]:
    """兼容旧调用：默认返回最终训练索引路径。"""
    return final_index_paths(benchmark_dir)


def write_source_split_indexes(benchmark_dir: Path, samples: list[dict[str, Any]]) -> None:
    paths = source_index_paths(benchmark_dir)
    write_jsonl(paths["all"], sorted(samples, key=lambda row: row["sample_id"]))
    for split in ["train", "val", "test", "unlabeled", "extended_pool"]:
        rows = [row for row in samples if row.get("split") == split]
        write_jsonl(paths[split], sorted(rows, key=lambda row: row["sample_id"]))


def write_split_indexes(benchmark_dir: Path, samples: list[dict[str, Any]]) -> None:
    paths = final_index_paths(benchmark_dir)
    write_jsonl(paths["all"], sorted(samples, key=lambda row: row["sample_id"]))
    for split in ["train", "val", "test", "unlabeled", "extended_pool"]:
        rows = [row for row in samples if row.get("split") == split]
        write_jsonl(paths[split], sorted(rows, key=lambda row: row["sample_id"]))


def make_referring_training_sample(parent: dict[str, Any], expression: dict[str, Any]) -> dict[str, Any]:
    """把父样本中的一条指代表达展开成可直接训练的 segmentation 样本。"""
    parent_id = str(parent.get("sample_id"))
    expression_id = str(expression.get("expression_id"))
    item = copy.deepcopy(parent)
    item["sample_id"] = f"{parent_id}__{expression_id}"
    item["parent_sample_id"] = parent_id
    item["parent_task_type"] = parent.get("task_type")
    item["task_type"] = "referring_landslide_segmentation"
    item["source_key"] = f"{parent.get('source_key')}/{expression_id}"
    item["instruction"] = {
        "language": "en",
        "template_id": f"referring_{expression.get('category')}_{expression.get('subtype')}_v1",
        "text": expression.get("text"),
        "text_zh": expression.get("text_zh"),
    }
    item["mask"] = copy.deepcopy(expression.get("target_mask"))
    item["referring_expression"] = copy.deepcopy(expression)
    item.pop("referring_expressions", None)

    flags = set(item.get("quality_flags") or [])
    flags.update(expression.get("quality_flags") or [])
    flags.add("referring_expression_rule_generated")
    item["quality_flags"] = sorted(flags)
    item["supervision"] = "mask"
    return item


def flatten_referring_samples(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """从最终样本索引展开所有指代表达样本。"""
    rows: list[dict[str, Any]] = []
    for sample in samples:
        for expression in sample.get("referring_expressions") or []:
            rows.append(make_referring_training_sample(sample, expression))
    return sorted(rows, key=lambda row: str(row.get("sample_id", "")))


def write_referring_split_indexes(benchmark_dir: Path, samples: list[dict[str, Any]]) -> None:
    """写出指代表达 all/train/val/test JSONL。"""
    paths = referring_index_paths(benchmark_dir)
    write_jsonl(paths["all"], sorted(samples, key=lambda row: row["sample_id"]))
    for split in ["train", "val", "test", "unlabeled", "extended_pool"]:
        rows = [row for row in samples if row.get("split") == split]
        write_jsonl(paths[split], sorted(rows, key=lambda row: row["sample_id"]))


def sen12_parse_key(path: Path) -> tuple[str, str, str] | None:
    """解析 Sen12 文件名：<region>_<sensor>_<id>.nc。"""
    name = path.stem
    for sensor in ("s1asc", "s1dsc", "s2"):
        marker = f"_{sensor}_"
        if marker in name:
            region, sample_id = name.split(marker, 1)
            return region, sensor, sample_id
    return None


def sen12_collect(root: Path) -> dict[str, dict[tuple[str, str], Path]]:
    out: dict[str, dict[tuple[str, str], Path]] = {"s1asc": {}, "s1dsc": {}, "s2": {}}
    for sensor in out:
        sensor_dir = root / sensor
        for path in sensor_dir.glob("*.nc"):
            parsed = sen12_parse_key(path)
            if parsed:
                region, parsed_sensor, sample_id = parsed
                out[parsed_sensor][(region, sample_id)] = path
    return out


def try_bbox_from_mask(path: Path | None) -> tuple[list[int] | None, str]:
    """从图像 mask 像素派生 bbox，mask>0 视为滑坡区域。"""
    if path is None or not path.exists():
        return None, "mask_path_missing"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NotGeoreferencedWarning)
            with rasterio.open(path) as src:
                arr = src.read(1)
    except Exception as rio_exc:
        if is_tif_path(path):
            return None, f"bbox_derive_failed_rasterio_tif: {rio_exc}"
        try:
            with Image.open(path) as img:
                arr = np.asarray(img.convert("L"))
        except Exception as exc:
            return None, f"bbox_derive_failed: {exc}"
    ys, xs = np.where(np.asarray(arr) > 0)
    if xs.size == 0:
        return None, "empty_mask"
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())], "derived"
