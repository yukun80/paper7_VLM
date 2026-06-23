"""地灾 VLM 基准数据集构建的共享工具函数。

用途：
    为 `1-1` 到 `1-6` 的阶段脚本提供通用读写、渲染、切片、标注和校验函数。

说明：
    本模块只放可复用逻辑，不直接启动完整流水线。这样每个阶段脚本都能保持短小，
    后续扩展新数据集或新视图时也不需要改动一个庞大的单体脚本。
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import random
import re
import shutil
import warnings
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from PIL import Image, ImageDraw, ImageFont
from rasterio.enums import Resampling
from rasterio.errors import NotGeoreferencedWarning
from rasterio.windows import Window
from tqdm import tqdm


warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

IMAGE_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
LABEL_HINTS = ("label", "labels", "mask", "masks", "gt", "groundtruth", "ground_truth", "binary")


def find_cjk_font() -> str | None:
    """优先选择系统中可用的中文字体，避免统计图和抽查图中文字变成方块。"""
    preferred = [
        "Noto Sans CJK SC",
        "Noto Serif CJK SC",
        "Microsoft YaHei",
        "Microsoft YaHei UI",
        "WenQuanYi Micro Hei",
        "Source Han Sans SC",
        "SimHei",
    ]
    for font_name in preferred:
        try:
            path = font_manager.findfont(font_name, fallback_to_default=False)
        except ValueError:
            continue
        if path:
            return path
    return None


CJK_FONT_PATH = find_cjk_font()
if CJK_FONT_PATH:
    font_manager.fontManager.addfont(CJK_FONT_PATH)
    plt.rcParams["font.sans-serif"] = [font_manager.FontProperties(fname=CJK_FONT_PATH).get_name()]
plt.rcParams["axes.unicode_minus"] = False


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def stable_hash(text: str) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16)


def split_from_group(group: str) -> str:
    """Sen12 没有官方 train/val/test 时，用区域和事件生成稳定划分。"""
    bucket = stable_hash(group) % 100
    if bucket < 70:
        return "train"
    if bucket < 80:
        return "val"
    return "test"


def parse_literal(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return default


def robust_stretch(arr: np.ndarray, low: float = 2, high: float = 98) -> np.ndarray:
    """把任意数值影像稳健拉伸到 8-bit，避免少数极值主导显示效果。"""
    data = arr.astype("float32", copy=False)
    valid = np.isfinite(data)
    valid &= data > -30000
    if not np.any(valid):
        return np.zeros(data.shape, dtype=np.uint8)
    lo, hi = np.percentile(data[valid], [low, high])
    if math.isclose(float(lo), float(hi)):
        lo, hi = float(np.min(data[valid])), float(np.max(data[valid]))
    if math.isclose(float(lo), float(hi)):
        return np.zeros(data.shape, dtype=np.uint8)
    out = (np.clip(data, lo, hi) - lo) / (hi - lo)
    return (out * 255).astype(np.uint8)


def make_rgb(channels: list[np.ndarray]) -> np.ndarray:
    return np.dstack([robust_stretch(ch) for ch in channels]).astype(np.uint8)


def save_rgb(path: Path, arr: np.ndarray) -> None:
    ensure_dir(path.parent)
    Image.fromarray(arr.astype(np.uint8)).save(path)


def save_mask(path: Path, mask: np.ndarray) -> None:
    ensure_dir(path.parent)
    Image.fromarray((mask > 0).astype(np.uint8)).save(path)


def save_mask_visual(path: Path, mask: np.ndarray) -> None:
    """保存红黑可视化标签：黑色为背景，红色为滑坡，仅用于人工检查。"""
    ensure_dir(path.parent)
    binary = mask > 0
    visual = np.zeros((*binary.shape, 3), dtype=np.uint8)
    visual[binary] = [255, 0, 0]
    Image.fromarray(visual).save(path)


def bbox_from_mask(mask: np.ndarray) -> list[int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return []
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def norm_bbox_1000(bbox: list[int], width: int, height: int) -> list[int]:
    if not bbox:
        return []
    x1, y1, x2, y2 = bbox
    return [
        int(round(x1 * 1000 / width)),
        int(round(y1 * 1000 / height)),
        int(round(x2 * 1000 / width)),
        int(round(y2 * 1000 / height)),
    ]


def mask_area(mask: np.ndarray) -> int:
    return int(np.sum(mask > 0))


def evidence_sufficiency(hazard_present: bool, modality: str, quality_label: str) -> str:
    if "低质量" in quality_label:
        return "证据受图像质量限制"
    if modality.startswith("sar"):
        return "仅有 SAR 后向散射视觉证据，需谨慎解释" if hazard_present else "仅有 SAR 后向散射，未见明确滑坡证据"
    if hazard_present:
        return "足以支持可见滑坡存在判断"
    return "足以支持该切片内未见可见滑坡"


def parse_time_values(tags: dict[str, str]) -> list[int]:
    nums = re.findall(r"-?\d+", tags.get("NETCDF_DIM_time_VALUES", ""))
    return [int(n) for n in nums]


def parse_days_since(tags: dict[str, str]) -> date | None:
    match = re.search(r"days since (\d{4}-\d{2}-\d{2})", tags.get("time#units", ""))
    if not match:
        return None
    return date.fromisoformat(match.group(1))


def indexed_date(tags: dict[str, str], index_zero_based: int) -> str | None:
    base = parse_days_since(tags)
    offsets = parse_time_values(tags)
    if base is None or not offsets or index_zero_based >= len(offsets):
        return None
    return (base + timedelta(days=offsets[index_zero_based])).isoformat()


def sen12_subdataset(path: Path, var_name: str) -> str:
    return f'NETCDF:"{path.as_posix()}":{var_name}'


def read_sen12_var(path: Path, var_name: str, index_zero_based: int) -> tuple[np.ndarray, dict[str, str]]:
    with rasterio.open(sen12_subdataset(path, var_name)) as ds:
        band = min(max(index_zero_based + 1, 1), ds.count)
        return ds.read(band), ds.tags()


def parse_sen12_name(path: Path) -> tuple[str, str]:
    match = re.match(r"(.+)_(s2|s1asc|s1dsc)_(\d+)\.nc$", path.name)
    if not match:
        return path.stem, "unknown"
    return match.group(1), match.group(3)


def quality_from_s2(path: Path, post_idx: int, rgb: np.ndarray) -> str:
    try:
        scl, _ = read_sen12_var(path, "SCL", post_idx)
        bad_fraction = float(np.mean(np.isin(scl, [0, 1, 3, 8, 9, 10, 11])))
        if bad_fraction > 0.5:
            return "低质量：云、阴影、雪或无效像元占比较高"
        if bad_fraction > 0.15:
            return "部分退化：存在云、阴影、雪或无效像元"
    except Exception:
        pass
    gray = np.mean(rgb, axis=2)
    if float(np.mean(gray < 5)) > 0.25 or float(np.mean(gray > 250)) > 0.25:
        return "低质量：曝光异常或无效像元占比较高"
    return "可用光学影像"


def hillshade(dem: np.ndarray) -> np.ndarray:
    dem = dem.astype("float32")
    gy, gx = np.gradient(dem)
    slope = np.pi / 2 - np.arctan(np.sqrt(gx * gx + gy * gy))
    aspect = np.arctan2(-gx, gy)
    azimuth = np.deg2rad(315)
    altitude = np.deg2rad(45)
    shaded = np.sin(altitude) * np.sin(slope) + np.cos(altitude) * np.cos(slope) * np.cos(azimuth - aspect)
    return robust_stretch(shaded)


def split_from_gdcld_path(path: Path) -> str:
    lowered = [p.lower() for p in path.parts]
    if any(p in {"test", "test_data", "test_dataset"} for p in lowered):
        return "test"
    if any(p in {"val", "valid", "validation", "val_dataset", "val_data"} for p in lowered):
        return "val"
    if any(p in {"train", "train_dataset", "train_data"} for p in lowered):
        return "train"
    if "future work" in "/".join(lowered):
        return "test_candidate"
    return "train"


def is_future_work(path: Path) -> bool:
    return "future work" in path.as_posix().lower()


def is_label_file(path: Path) -> bool:
    text = "/".join(p.lower() for p in path.parts)
    return any(hint in text for hint in LABEL_HINTS)


def normalized_pair_key(path: Path) -> str:
    stem = path.stem.lower()
    for token in ["_label", "-label", "label", "_mask", "-mask", "mask", "_gt", "-gt", "gt", "_binary", "-binary", "binary"]:
        stem = stem.replace(token, "")
    return re.sub(r"[^a-z0-9]+", "", stem)


def discover_gdcld_pairs(root: Path) -> list[dict[str, Any]]:
    """根据路径和文件名配对 GDCLD 图像与标签。"""
    if not root.exists():
        return []
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    labels = [p for p in files if is_label_file(p)]
    images = [p for p in files if not is_label_file(p)]
    image_by_key: dict[tuple[str, str], list[Path]] = defaultdict(list)
    for image in images:
        image_by_key[(split_from_gdcld_path(image), normalized_pair_key(image))].append(image)

    pairs: list[dict[str, Any]] = []
    for label in labels:
        split = split_from_gdcld_path(label)
        key = normalized_pair_key(label)
        candidates = image_by_key.get((split, key), [])
        if not candidates:
            for fallback in ["train", "val", "test", "test_candidate"]:
                candidates.extend(image_by_key.get((fallback, key), []))
        if candidates:
            image = candidates[0]
            pairs.append(
                {
                    "image_path": image.as_posix(),
                    "label_path": label.as_posix(),
                    "split": split_from_gdcld_path(image if split == "train" else label),
                    "is_future_work": is_future_work(image) or is_future_work(label),
                    "pair_key": key,
                }
            )
    pairs.sort(key=lambda item: item["image_path"])
    return pairs


def raster_basic_info(path: Path) -> dict[str, Any]:
    with rasterio.open(path) as ds:
        return {"width": ds.width, "height": ds.height, "count": ds.count, "dtype": str(ds.dtypes[0])}


def sampled_label_values(path: Path, max_side: int = 1024) -> list[int]:
    """用最近邻下采样读取标签值，避免扫描超大标签整图。"""
    with rasterio.open(path) as ds:
        scale = min(1.0, max_side / max(ds.width, ds.height))
        out_h = max(1, int(round(ds.height * scale)))
        out_w = max(1, int(round(ds.width * scale)))
        arr = ds.read(1, out_shape=(out_h, out_w), resampling=Resampling.nearest)
    return [int(v) for v in np.unique(arr)[:32].tolist()]


def infer_gdcld_sensor(path: Path) -> str:
    text = path.as_posix().lower()
    if "uav" in text:
        return "UAV"
    if "planet" in text:
        return "PlanetScope"
    if "gaofen" in text or "gf-6" in text or "gf6" in text:
        return "Gaofen-6"
    if "map" in text and "world" in text:
        return "Map World"
    return "RGB 高分辨率遥感影像"


def gdcld_region_from_path(path: Path) -> str:
    parts = list(path.parts)
    for marker in ["test_data", "data", "label", "test_label"]:
        if marker in parts:
            idx = parts.index(marker)
            if idx + 1 < len(parts):
                return parts[idx + 1]
    return path.parent.name or "unknown_region"


def read_gdcld_rgb_window(ds: rasterio.io.DatasetReader, window: Window) -> np.ndarray:
    """GDCLD 可能是 3 或 4 波段；VLM 输入只取前三个 RGB 波段。"""
    indexes = list(range(1, min(ds.count, 3) + 1))
    arr = ds.read(indexes, window=window)
    if arr.shape[0] == 1:
        arr = np.repeat(arr, 3, axis=0)
    rgb = np.dstack([robust_stretch(arr[i]) for i in range(arr.shape[0])])
    if rgb.shape[2] < 3:
        rgb = np.dstack([rgb[:, :, 0], rgb[:, :, 0], rgb[:, :, 0]])
    return rgb[:, :, :3].astype(np.uint8)


def read_gdcld_mask_window(ds: rasterio.io.DatasetReader, window: Window) -> np.ndarray:
    # GDCLD 标签存在 0/1/85/255 等编码。语义上只需区分滑坡与背景，因此统一用 >0 二值化。
    arr = ds.read(1, window=window)
    return (arr > 0).astype(np.uint8)


def qwen_messages(sample: dict[str, Any], task: str, version: str) -> dict[str, Any]:
    image_sequence = sample.get("image_sequence") or []
    has_pre_post = len(image_sequence) >= 2
    sensor_context = (
        f"传感器：{sample['sensor_type']}；模态：{sample['modality']}；"
        f"空间分辨率：{sample.get('spatial_resolution_m') or '未知'} m；"
        f"事件或观测日期：{sample.get('date') or sample.get('post_date') or '未知'}。"
    )
    hazard = bool(sample["hazard_present"])
    bbox = sample.get("bbox_norm_1000", [])
    if task == "classification":
        question = sensor_context + "请判断图像中是否存在可见滑坡。请用 JSON 回答，字段包括 hazard_present、hazard_type、evidence、quality 和 recommendation。"
        answer = {
            "hazard_present": hazard,
            "hazard_type": "landslide" if hazard else "none",
            "evidence": "参考掩膜中存在滑坡像元，可作为可见滑坡证据" if hazard else "该切片的参考掩膜中没有滑坡像元",
            "quality": sample["quality_label"],
            "recommendation": "下游使用前应结合 bbox 或 mask 核对滑坡证据位置" if hazard else "缺少额外证据时不要过度声称存在滑坡",
        }
    elif task == "grounding":
        question = sensor_context + "如果图像中存在可见滑坡，请返回滑坡证据区域 bbox。bbox 使用 0-1000 归一化图像坐标，格式为 [x1,y1,x2,y2]；如果没有可见滑坡，则返回空 bbox。"
        answer = {"hazard_present": hazard, "hazard_type": "landslide" if hazard else "none", "bbox_0_1000": bbox if hazard else [], "evidence_sufficiency": sample["evidence_sufficiency"]}
    elif task == "quality":
        question = sensor_context + "请判断当前图像质量和可用模态是否足以支持较有把握的滑坡判读。请回答 quality_label、evidence_sufficiency，并给出保守回答规则。"
        answer = {"quality_label": sample["quality_label"], "evidence_sufficiency": sample["evidence_sufficiency"], "conservative_response_rule": "当视觉证据、传感器模态或元数据不足时应明确表达不确定性；不得编造 InSAR 形变证据"}
    else:
        raise ValueError(task)
    if has_pre_post:
        content = [
            {
                "type": "text",
                "text": f"图像1为灾前 Sentinel-2 真彩色影像，日期：{image_sequence[0].get('date') or '未知'}。",
            },
            {"type": "image", "image": image_sequence[0]["path"]},
            {
                "type": "text",
                "text": f"图像2为灾后 Sentinel-2 真彩色影像，日期：{image_sequence[1].get('date') or '未知'}。{question} bbox 坐标如需输出，均以图像2灾后影像为基准。",
            },
            {"type": "image", "image": image_sequence[1]["path"]},
        ]
    else:
        content = [{"type": "image", "image": sample["rendered_image"]}, {"type": "text", "text": question}]
    return {
        "id": f"{sample['sample_id']}::{task}",
        "image": sample["rendered_image"],
        "task": task,
        "messages": [{"role": "user", "content": content}, {"role": "assistant", "content": json.dumps(answer, ensure_ascii=False)}],
    }


def build_coco(metadata_rows: list[dict[str, Any]]) -> dict[str, Any]:
    images = []
    annotations = []
    ann_id = 1
    for image_id, row in enumerate(metadata_rows, start=1):
        images.append({"id": image_id, "file_name": row["rendered_image"], "width": row["image_width"], "height": row["image_height"], "sample_id": row["sample_id"], "source_dataset": row["source_dataset"], "split": row["split"]})
        bbox = row.get("bbox_xyxy") or []
        if row.get("hazard_present") and bbox:
            x1, y1, x2, y2 = bbox
            # bbox 由语义 mask 自动派生，只表示证据范围，不代表实例级滑坡对象。
            annotations.append({"id": ann_id, "image_id": image_id, "category_id": 1, "bbox": [x1, y1, x2 - x1, y2 - y1], "area": int((x2 - x1) * (y2 - y1)), "iscrowd": 0, "sample_id": row["sample_id"], "note": "bbox 由语义二值掩膜自动提取，不代表实例级标注"})
            ann_id += 1
    return {"info": {"description": "GeoHazard-HalluGround 滑坡基准数据集派生 bbox 标注"}, "categories": [{"id": 1, "name": "landslide", "supercategory": "geohazard"}], "images": images, "annotations": annotations}


def validate_outputs(
    metadata_rows: list[dict[str, Any]],
    show_progress: bool = False,
    max_samples: int | None = None,
    skip_pixel_check: bool = False,
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings_out: list[str] = []
    split_by_group: dict[str, set[str]] = defaultdict(set)
    rows_to_check = metadata_rows[:max_samples] if max_samples is not None else metadata_rows
    iterator = tqdm(rows_to_check, desc="校验样本文件", unit="样本") if show_progress else rows_to_check
    for row in iterator:
        sample_id = row["sample_id"]
        image_path = Path(row["rendered_image"])
        mask_path = Path(row["mask_path"])
        if not image_path.exists():
            errors.append(f"{sample_id}: 缺少渲染图像 {image_path}")
            continue
        if not mask_path.exists():
            errors.append(f"{sample_id}: 缺少掩膜 {mask_path}")
            continue
        with Image.open(image_path) as im:
            width, height = im.size
        if width != row["image_width"] or height != row["image_height"]:
            errors.append(f"{sample_id}: metadata 记录的图像尺寸与实际文件不一致")
        with Image.open(mask_path) as mask_im:
            mask = np.asarray(mask_im.convert("L"))
        if mask.shape != (row["image_height"], row["image_width"]):
            errors.append(f"{sample_id}: 掩膜尺寸与 metadata 不一致")
        if not skip_pixel_check:
            unique = set(np.unique(mask).tolist())
            if not unique.issubset({0, 1}):
                errors.append(f"{sample_id}: 掩膜存在非二值像元 {sorted(unique)[:10]}")
        visual_path_value = row.get("mask_visual_path")
        if visual_path_value:
            visual_path = Path(visual_path_value)
            if not visual_path.exists():
                errors.append(f"{sample_id}: 缺少红黑可视化标签 {visual_path}")
            else:
                with Image.open(visual_path) as visual_im:
                    if visual_im.size != (width, height):
                        errors.append(f"{sample_id}: 红黑可视化标签尺寸与图像不一致")
        sequence = row.get("image_sequence") or []
        for item in sequence:
            seq_path = Path(item.get("path", ""))
            role = item.get("role", "unknown_role")
            if not seq_path.exists():
                errors.append(f"{sample_id}: 缺少 {role} 图像 {seq_path}")
                continue
            with Image.open(seq_path) as seq_im:
                if seq_im.size != (width, height):
                    errors.append(f"{sample_id}: {role} 图像尺寸与主图像不一致")
        for field in ["pre_image", "post_image"]:
            field_value = row.get(field)
            if field_value and not Path(field_value).exists():
                errors.append(f"{sample_id}: 缺少 {field} 图像 {field_value}")
        preview_value = row.get("pair_preview_image")
        if preview_value:
            preview_path = Path(preview_value)
            if not preview_path.exists():
                errors.append(f"{sample_id}: 缺少灾前灾后预览图 {preview_path}")
            else:
                with Image.open(preview_path) as preview_im:
                    if preview_im.size != (width * 2, height):
                        errors.append(f"{sample_id}: 灾前灾后预览图尺寸应为 {width * 2}x{height}")
        bbox = row.get("bbox_xyxy") or []
        if bbox:
            x1, y1, x2, y2 = bbox
            if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                errors.append(f"{sample_id}: bbox 越界，bbox={bbox}，图像尺寸={width}x{height}")
        if not skip_pixel_check and row.get("hazard_present") and int(np.sum(mask > 0)) <= 0:
            errors.append(f"{sample_id}: 正样本的掩膜面积为 0")
        leak_key = f"{row['source_dataset']}:{row.get('region_id')}:{row.get('event_id')}:{row.get('ann_id') or row.get('native_patch_id') or row.get('source_window_xywh') or row['sample_id']}"
        split_by_group[leak_key].add(row["split"])
    if max_samples is not None and max_samples < len(metadata_rows):
        warnings_out.append(f"快速校验模式：仅检查 {len(rows_to_check)} / {len(metadata_rows)} 个样本文件，split 泄漏检查也仅覆盖该子集")
    for group, splits in split_by_group.items():
        if len(splits) > 1:
            errors.append(f"数据划分泄漏：{group} 同时出现在 {sorted(splits)}")
    return errors, warnings_out


def save_splits(out_dir: Path, metadata_rows: list[dict[str, Any]]) -> None:
    split_dir = ensure_dir(out_dir / "splits")
    for split in ["train", "val", "test", "test_candidate"]:
        write_jsonl(split_dir / f"{split}.jsonl", [row for row in metadata_rows if row["split"] == split])


def plot_counter(counter: Counter[str], title: str, path: Path) -> None:
    if not counter:
        return
    labels, values = zip(*counter.most_common())
    plt.figure(figsize=(max(6, min(16, len(labels) * 0.8)), 4))
    plt.bar(range(len(labels)), values)
    plt.xticks(range(len(labels)), labels, rotation=35, ha="right")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_stats(out_dir: Path, metadata_rows: list[dict[str, Any]]) -> None:
    stats_dir = ensure_dir(out_dir / "figures")
    summary = {
        "total_samples": len(metadata_rows),
        "by_source_dataset": Counter(row["source_dataset"] for row in metadata_rows),
        "by_sensor_type": Counter(row["sensor_type"] for row in metadata_rows),
        "by_modality": Counter(row["modality"] for row in metadata_rows),
        "by_split": Counter(row["split"] for row in metadata_rows),
        "by_hazard_present": Counter(str(row["hazard_present"]) for row in metadata_rows),
        "by_quality_label": Counter(row["quality_label"] for row in metadata_rows),
    }
    (out_dir / "summary.json").write_text(json.dumps({k: dict(v) if isinstance(v, Counter) else v for k, v in summary.items()}, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_counter(summary["by_source_dataset"], "按数据来源统计样本数量", stats_dir / "samples_by_source.png")
    plot_counter(summary["by_sensor_type"], "按传感器类型统计样本数量", stats_dir / "samples_by_sensor.png")
    plot_counter(summary["by_split"], "按数据划分统计样本数量", stats_dir / "samples_by_split.png")
    plot_counter(summary["by_quality_label"], "按质量标签统计样本数量", stats_dir / "samples_by_quality.png")


def draw_audit_image(row: dict[str, Any], out_path: Path) -> None:
    use_pair_preview = bool(row.get("pair_preview_image"))
    if use_pair_preview:
        with Image.open(row["pair_preview_image"]) as im:
            image = im.convert("RGB")
    else:
        with Image.open(row["rendered_image"]) as im:
            image = im.convert("RGB")
    with Image.open(row["mask_path"]) as mask_im:
        mask = np.asarray(mask_im.convert("L")) > 0
    if use_pair_preview:
        if mask.shape != (image.height, image.width // 2):
            raise ValueError(f"{row['sample_id']}: 灾后掩膜和 pair preview 尺寸不一致，跳过 audit 叠加图")
        preview_mask = np.zeros((image.height, image.width), dtype=bool)
        preview_mask[:, image.width // 2 :] = mask
        mask = preview_mask
    elif mask.shape != (image.height, image.width):
        raise ValueError(f"{row['sample_id']}: 图像和掩膜尺寸不一致，跳过 audit 叠加图")
    overlay_arr = np.asarray(image).copy()
    overlay_arr[mask, 0] = 255
    overlay_arr[mask, 1] = (overlay_arr[mask, 1] * 0.35).astype(np.uint8)
    overlay_arr[mask, 2] = (overlay_arr[mask, 2] * 0.35).astype(np.uint8)
    overlay = Image.fromarray(overlay_arr)
    draw = ImageDraw.Draw(overlay)
    bbox = row.get("bbox_xyxy") or []
    if bbox:
        if use_pair_preview:
            x1, y1, x2, y2 = bbox
            bbox = [x1 + image.width // 2, y1, x2 + image.width // 2, y2]
        draw.rectangle(bbox, outline=(255, 255, 0), width=max(1, round(max(image.size) / 180)))
    window_text = row.get("source_window_xywh") or "无窗口"
    text_lines = [
        f"{row['sample_id']} | 划分={row['split']} | 传感器={row['sensor_type']} | 是否有滑坡={row['hazard_present']}",
        f"mask={Path(row['mask_path']).name} | window={window_text} | 主图={row.get('primary_image_role') or 'single_image'}",
    ]
    try:
        font = ImageFont.truetype(CJK_FONT_PATH, 16) if CJK_FONT_PATH else ImageFont.load_default()
    except Exception:
        font = None
    pad = 4
    line_boxes = [draw.textbbox((0, 0), line, font=font) for line in text_lines]
    line_heights = [box[3] - box[1] for box in line_boxes]
    label_h = sum(line_heights) + pad * (len(text_lines) + 1)
    canvas = Image.new("RGB", (overlay.width, overlay.height + label_h), "white")
    canvas.paste(overlay, (0, label_h))
    canvas_draw = ImageDraw.Draw(canvas)
    y = pad
    for line, line_h in zip(text_lines, line_heights):
        canvas_draw.text((pad, y), line, fill=(0, 0, 0), font=font)
        y += line_h + pad
    ensure_dir(out_path.parent)
    canvas.save(out_path)


def save_audit(out_dir: Path, metadata_rows: list[dict[str, Any]], sample_count: int, seed: int) -> None:
    if sample_count <= 0 or not metadata_rows:
        return
    audit_dir = out_dir / "audit"
    clean_dir(audit_dir)
    rows = metadata_rows[:]
    random.Random(seed).shuffle(rows)
    for row in rows[:sample_count]:
        try:
            draw_audit_image(row, audit_dir / f"{row['sample_id']}.png")
        except ValueError as exc:
            print(f"[警告] {exc}")
