#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""多源滑坡 instruction segmentation 数据读取。

脚本作用：读取 benchmark/multisource_landslide_v1_small 的 instruction JSONL，
完成模态别名映射、.npy 加载、归一化、target size 对齐、GSD fallback 和 bbox fallback。
主要输入：indexes/instruction_train.jsonl、indexes/instruction_val.jsonl。
主要输出：PyTorch Dataset/DataLoader batch。
是否改写原始数据：不会。
典型用法：MultiSourceLandslideDataset(config, split="train")。
"""

from __future__ import annotations

import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .config import QPSalmConfig


REPO_ROOT = Path(__file__).resolve().parents[2]

CANONICAL_MODALITIES = ["hr_optical", "s2", "s1", "dem", "insar"]
CANONICAL_CHANNELS = {
    "hr_optical": 5,
    "s2": 12,
    "s1": 4,
    "dem": 2,
    "insar": 1,
}
MODALITY_TO_CANONICAL = {
    "optical_rgb": "hr_optical",
    "optical_multiband": "hr_optical",
    "multispectral": "s2",
    "sar_asc": "s1",
    "sar_dsc": "s1",
    "dem": "dem",
    "slope": "dem",
    "insar_vel": "insar",
}
GSD_TOKENS = ["unknown", "sub_meter", "meter_1_5", "meter_5_10", "meter_gt_10"]


@dataclass
class DatasetStats:
    """轻量数据统计，用于 inspect CLI 和训练日志。"""

    num_rows: int
    num_usable: int
    skipped_by_reason: dict[str, int]
    by_template: dict[str, int]
    by_raw_combo: dict[str, int]
    by_canonical_combo: dict[str, int]
    by_sensor_combo: dict[str, int]
    by_normalization_combo: dict[str, int]
    by_shape: dict[str, dict[str, int]]
    gsd_tokens: dict[str, int]
    quality_flags: dict[str, int]


def resolve_repo_path(path_ref: str | Path | None) -> Path | None:
    """解析 repo 相对路径。"""
    if path_ref is None:
        return None
    p = Path(path_ref)
    if p.is_absolute():
        return p
    return REPO_ROOT / p


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    """流式读取 JSONL 文件。"""
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                yield json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} 不是合法 JSONL: {exc}") from exc


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取完整 JSONL 文件；仅用于 inspect/统计，不用于小样本 smoke。"""
    rows: list[dict[str, Any]] = []
    for row in iter_jsonl(path):
        rows.append(row)
    return rows


def available_modality_names(row: dict[str, Any]) -> list[str]:
    """返回 available=True 的原始模态名。"""
    modalities = row.get("modalities") or {}
    return sorted(name for name, item in modalities.items() if isinstance(item, dict) and item.get("available", True))


def raw_modality_combo(row: dict[str, Any]) -> str:
    names = available_modality_names(row)
    return "+".join(names) if names else "none"


def canonical_modality_name(raw_name: str, item: dict[str, Any] | None = None) -> str | None:
    """把 benchmark 原始模态名映射到模型 canonical 槽位，并参考 sensor 元数据。"""
    item = item or {}
    sensor = str(item.get("sensor") or "").lower()
    role = str(item.get("role") or "").lower()
    source = str(item.get("source") or "").lower()
    value_encoding = str(item.get("value_encoding") or "").lower()
    if raw_name == "optical_rgb" and (
        sensor == "sentinel2"
        or "sentinel-2" in source
        or "sentinel2" in source
        or "sentinel2" in role
        or "sentinel2" in value_encoding
    ):
        return "s2"
    return MODALITY_TO_CANONICAL.get(raw_name)


def canonical_modality_combo(row: dict[str, Any]) -> str:
    modalities = row.get("modalities") or {}
    names = sorted(
        {
            canonical
            for name, item in modalities.items()
            if isinstance(item, dict)
            and item.get("available", True)
            and (canonical := canonical_modality_name(name, item)) is not None
        }
    )
    return "+".join(names) if names else "none"


def metadata_combo(row: dict[str, Any], field: str, fallback: str = "unknown") -> str:
    """按模态元数据字段聚合 combo，用于质检和 prompt。"""
    values = []
    for name in available_modality_names(row):
        item = (row.get("modalities") or {}).get(name) or {}
        value = item.get(field)
        values.append(str(value if value not in (None, "") else fallback))
    return "+".join(sorted(values)) if values else "none"


def sensor_combo(row: dict[str, Any]) -> str:
    return metadata_combo(row, "sensor")


def normalization_combo(row: dict[str, Any]) -> str:
    return metadata_combo(row, "normalization")


def gsd_to_token(gsd: Any) -> str:
    """把空间分辨率转换成稳定 token。"""
    if gsd is None or gsd == "" or str(gsd).lower() == "none":
        return "unknown"
    try:
        value = float(gsd)
    except (TypeError, ValueError):
        return "unknown"
    if not math.isfinite(value) or value <= 0:
        return "unknown"
    if value <= 1.0:
        return "sub_meter"
    if value <= 5.0:
        return "meter_1_5"
    if value <= 10.0:
        return "meter_5_10"
    return "meter_gt_10"


def gsd_token_id(token: str) -> int:
    return GSD_TOKENS.index(token) if token in GSD_TOKENS else 0


def row_template_id(row: dict[str, Any]) -> str:
    """兼容不同 instruction 索引里的模板字段名。"""
    return str(row.get("template_id") or row.get("task_template_id") or "")


def availability_prompt_tokens(row: dict[str, Any]) -> list[str]:
    """构造显式可用/缺失模态 token。"""
    available = set(canonical_modality_combo(row).split("+"))
    tokens = []
    names = {
        "hr_optical": "HR_OPTICAL",
        "s2": "S2",
        "s1": "S1",
        "dem": "DEM",
        "insar": "INSAR",
    }
    for canonical, label in names.items():
        state = "AVAILABLE" if canonical in available else "MISSING"
        tokens.append(f"<{label}_{state}>")
    return tokens


def should_skip_row(row: dict[str, Any], core_templates: Iterable[str]) -> str | None:
    """返回跳过原因；None 表示可用。"""
    tid = row_template_id(row)
    if tid not in set(core_templates):
        return "non_core_template"
    if row.get("task_family") == "referring_landslide_segmentation":
        return "referring_deferred"
    if row.get("source_level") != "patch":
        return "scene_level_deferred"
    flags = set(row.get("quality_flags") or [])
    if "requires_tiling_for_patch_training" in flags or "scene_level_large_image" in flags:
        return "requires_tiling_deferred"
    if not isinstance(row.get("mask"), dict):
        return "missing_mask"
    if not available_modality_names(row):
        return "missing_modalities"
    return None


def load_npy_array(path_ref: str) -> np.ndarray:
    """稳健读取 .npy 数组。"""
    path = resolve_repo_path(path_ref)
    if path is None or not path.exists():
        raise FileNotFoundError(f"数组路径不存在: {path_ref}")
    arr = np.load(path)
    if arr.ndim == 2:
        arr = arr[None, :, :]
    elif arr.ndim == 3:
        if arr.shape[0] <= 32:
            pass
        elif arr.shape[-1] <= 32:
            arr = np.moveaxis(arr, -1, 0)
        else:
            raise ValueError(f"无法判断通道维: {path_ref}, shape={arr.shape}")
    else:
        raise ValueError(f"仅支持 2D/3D npy: {path_ref}, shape={arr.shape}")
    return arr


def normalize_modality(
    arr: np.ndarray,
    item: dict[str, Any] | None = None,
    raw_name: str = "",
    canonical: str | None = None,
) -> torch.Tensor:
    """按 benchmark 元数据归一化；避免对已物化数组重复拉伸。"""
    item = item or {}
    arr = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if arr.size == 0:
        raise ValueError("空数组不能作为模态输入")
    amin = float(arr.min())
    amax = float(arr.max())

    normalization = str(item.get("normalization") or "").lower()
    value_encoding = str(item.get("value_encoding") or "").lower()
    role = str(item.get("role") or "").lower()
    is_signed_evidence = (
        raw_name == "insar_vel"
        or canonical == "insar"
        or "insar" in role
        or "deformation" in role
        or "signed_symmetric" in normalization
    )
    if is_signed_evidence:
        if amin >= -1.5 and amax <= 1.5:
            return torch.from_numpy(np.clip(arr, -1.0, 1.0).astype(np.float32))
        bound = float(np.percentile(np.abs(arr[np.isfinite(arr)]), 98))
        if bound > 0:
            arr = np.clip(arr, -bound, bound) / bound
        else:
            arr = np.zeros_like(arr, dtype=np.float32)
        return torch.from_numpy(arr.astype(np.float32))

    if arr.dtype == np.uint8 or "preserve_rgb_values" in normalization:
        if amax > 1.5:
            arr = arr / 255.0
        return torch.from_numpy(np.clip(arr, 0.0, 1.0).astype(np.float32))

    metadata_says_normalized = bool(normalization) and (
        "robust" in normalization
        or "preserve_0_1" in normalization
        or "clip_0_10000" in normalization
        or "clip_-50_10" in normalization
        or "scale" in normalization
        or "reflectance" in normalization
        or "float32_no_extra" in normalization
    )
    if metadata_says_normalized and amin >= -0.05 and amax <= 1.5:
        return torch.from_numpy(np.clip(arr, 0.0, 1.0).astype(np.float32))

    if amin >= 0.0 and amax <= 1.5 and ("reflectance" in value_encoding or canonical in {"s2", "dem", "s1"}):
        return torch.from_numpy(np.clip(arr, 0.0, 1.0).astype(np.float32))
    if amin >= 0.0 and amax <= 255.0 and amax > 1.5:
        arr = arr / 255.0
    elif amax > 1.0 or amin < 0.0:
        out = np.zeros_like(arr, dtype=np.float32)
        for idx in range(arr.shape[0]):
            channel = arr[idx]
            lo = float(np.percentile(channel, 2))
            hi = float(np.percentile(channel, 98))
            if hi <= lo:
                out[idx] = 0.0
            else:
                out[idx] = np.clip((channel - lo) / (hi - lo), 0.0, 1.0)
        arr = out
    return torch.from_numpy(arr.astype(np.float32))


def normalize_mask(arr: np.ndarray) -> torch.Tensor:
    mask = (np.nan_to_num(arr.astype(np.float32)) > 0).astype(np.float32)
    if mask.ndim == 2:
        mask = mask[None, :, :]
    if mask.shape[0] != 1:
        mask = mask[:1]
    return torch.from_numpy(mask)


def resize_pad_tensor(tensor: torch.Tensor, target_size: int, mode: str) -> tuple[torch.Tensor, dict[str, Any]]:
    """等比例 resize 并 padding 到固定训练尺寸，避免非方形 patch 被拉伸。"""
    src_h, src_w = int(tensor.shape[-2]), int(tensor.shape[-1])
    if src_h <= 0 or src_w <= 0:
        raise ValueError(f"输入空间尺寸非法: shape={tuple(tensor.shape)}")
    scale = min(float(target_size) / float(src_h), float(target_size) / float(src_w))
    new_h = max(1, min(target_size, int(round(src_h * scale))))
    new_w = max(1, min(target_size, int(round(src_w * scale))))
    tensor4d = tensor.unsqueeze(0)
    if mode == "nearest":
        resized = F.interpolate(tensor4d, size=(new_h, new_w), mode=mode)
    else:
        resized = F.interpolate(tensor4d, size=(new_h, new_w), mode=mode, align_corners=False)
    pad_top = (target_size - new_h) // 2
    pad_bottom = target_size - new_h - pad_top
    pad_left = (target_size - new_w) // 2
    pad_right = target_size - new_w - pad_left
    out = F.pad(resized.squeeze(0), (pad_left, pad_right, pad_top, pad_bottom), value=0.0)
    transform = {
        "source_hw": [src_h, src_w],
        "target_hw": [target_size, target_size],
        "resized_hw": [new_h, new_w],
        "scale": scale,
        "pad_top": pad_top,
        "pad_bottom": pad_bottom,
        "pad_left": pad_left,
        "pad_right": pad_right,
    }
    return out, transform


def resize_tensor(tensor: torch.Tensor, target_size: int, mode: str) -> torch.Tensor:
    """兼容旧调用：等比例 resize+pad 到固定训练尺寸。"""
    out, _ = resize_pad_tensor(tensor, target_size, mode)
    return out


def pad_or_trim_channels(tensor: torch.Tensor, channels: int) -> torch.Tensor:
    """把任意输入通道数整理到 canonical 固定通道数。"""
    if tensor.shape[0] == channels:
        return tensor
    if tensor.shape[0] > channels:
        return tensor[:channels]
    pad = torch.zeros((channels - tensor.shape[0], tensor.shape[1], tensor.shape[2]), dtype=tensor.dtype)
    return torch.cat([tensor, pad], dim=0)


def compute_bbox_from_mask(mask: torch.Tensor) -> list[int] | None:
    """从 CHW mask 计算 xyxy bbox。"""
    pos = torch.nonzero(mask[0] > 0.5, as_tuple=False)
    if pos.numel() == 0:
        return None
    y0 = int(pos[:, 0].min().item())
    y1 = int(pos[:, 0].max().item())
    x0 = int(pos[:, 1].min().item())
    x1 = int(pos[:, 1].max().item())
    return [x0, y0, x1, y1]


def parse_bbox_xyxy(value: Any) -> list[float] | None:
    """兼容 list/tuple/dict 格式的 bbox。"""
    if value is None:
        return None
    if isinstance(value, dict):
        if {"x0", "y0", "x1", "y1"}.issubset(value):
            value = [value["x0"], value["y0"], value["x1"], value["y1"]]
        elif {"xmin", "ymin", "xmax", "ymax"}.issubset(value):
            value = [value["xmin"], value["ymin"], value["xmax"], value["ymax"]]
        else:
            return None
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x0, y0, x1, y1 = [float(v) for v in value]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in [x0, y0, x1, y1]):
        return None
    return [x0, y0, x1, y1]


def scale_bbox_to_target(bbox: list[float], source_hw: list[int], target_size: int) -> list[int] | None:
    """把原始 mask 坐标系的 bbox 映射到 target_size；保留旧接口。"""
    if len(source_hw) != 2:
        return None
    src_h, src_w = max(1, int(source_hw[0])), max(1, int(source_hw[1]))
    scale = min(float(target_size) / float(src_h), float(target_size) / float(src_w))
    new_h = max(1, min(target_size, int(round(src_h * scale))))
    new_w = max(1, min(target_size, int(round(src_w * scale))))
    transform = {
        "source_hw": [src_h, src_w],
        "target_hw": [target_size, target_size],
        "resized_hw": [new_h, new_w],
        "scale": scale,
        "pad_top": (target_size - new_h) // 2,
        "pad_left": (target_size - new_w) // 2,
    }
    return transform_bbox_to_target(bbox, transform)


def transform_bbox_to_target(bbox: list[float], transform: dict[str, Any]) -> list[int] | None:
    """用 resize+pad transform 把原始 xyxy bbox 映射到 target canvas。"""
    x0, y0, x1, y1 = bbox
    scale = float(transform.get("scale", 1.0))
    pad_left = int(transform.get("pad_left", 0))
    pad_top = int(transform.get("pad_top", 0))
    target_hw = transform.get("target_hw") or [1, 1]
    target_h, target_w = max(1, int(target_hw[0])), max(1, int(target_hw[1]))
    out = [
        int(math.floor(x0 * scale)) + pad_left,
        int(math.floor(y0 * scale)) + pad_top,
        int(math.ceil((x1 + 1.0) * scale)) - 1 + pad_left,
        int(math.ceil((y1 + 1.0) * scale)) - 1 + pad_top,
    ]
    out[0] = max(0, min(target_w - 1, out[0]))
    out[1] = max(0, min(target_h - 1, out[1]))
    out[2] = max(0, min(target_w - 1, out[2]))
    out[3] = max(0, min(target_h - 1, out[3]))
    if out[2] < out[0] or out[3] < out[1]:
        return None
    return out


def bbox_to_prior(bbox: list[int] | None, target_size: int) -> torch.Tensor:
    """把 xyxy bbox 转为 [1,H,W] prior mask。"""
    prior = torch.zeros((1, target_size, target_size), dtype=torch.float32)
    if bbox is None:
        return prior
    x0, y0, x1, y1 = bbox
    prior[:, y0 : y1 + 1, x0 : x1 + 1] = 1.0
    return prior


def infer_condition_prompt(row: dict[str, Any]) -> str:
    """按模板和指令文本派生 condition prompt。"""
    template_id = row_template_id(row)
    instruction = row.get("instruction") or {}
    text = str(instruction.get("text") or "").lower()
    if template_id == "insar_evidence_landslide_v1" or "insar" in text or "deformation" in text:
        return "deformation-supported landslide"
    if template_id == "sar_terrain_landslide_v1" or "sar" in text:
        return "SAR/terrain-supported landslide"
    if template_id == "terrain_evidence_landslide_v1" or "terrain" in text:
        return "terrain-supported landslide"
    if template_id == "multisource_landslide_v1" or "multi-source" in text or "multisource" in text:
        return "multi-source evidence landslide"
    if template_id == "negative_aware_landslide_v1":
        return "landslide with background awareness"
    if "newly" in text or "new landslide" in text:
        return "new landslide"
    if "active" in text:
        return "active landslide"
    return "landslide"


def build_proposal_context_text(row: dict[str, Any]) -> str:
    """构造 mask proposal 生成所需的任务/模态/尺度上下文。"""
    instruction = row.get("instruction") or {}
    text = str(instruction.get("text") or "Segment all landslide regions.")
    canonical = canonical_modality_combo(row)
    raw = raw_modality_combo(row)
    modality_details = []
    for name in available_modality_names(row):
        item = (row.get("modalities") or {}).get(name) or {}
        detail = (
            f"{name}"
            f"[role={item.get('role') or 'unknown'},"
            f"sensor={item.get('sensor') or 'unknown'},"
            f"encoding={item.get('value_encoding') or 'unknown'},"
            f"norm={item.get('normalization') or 'unknown'}]"
        )
        modality_details.append(detail)
    spatial = row.get("spatial") or {}
    gsd_token = gsd_to_token(spatial.get("gsd_m"))
    parts = [
        f"Task instruction: {text}",
        f"Task family: {row.get('task_family', 'unknown')}.",
        f"Dataset: {row.get('dataset_name', 'unknown')}.",
        f"Available modalities: {canonical}.",
        f"Raw modalities: {raw}.",
        f"Sensor combo: {sensor_combo(row)}.",
        f"Normalization combo: {normalization_combo(row)}.",
        f"Modality metadata: {'; '.join(modality_details) if modality_details else 'none'}.",
        f"Modality tokens: {' '.join(availability_prompt_tokens(row))}.",
        f"GSD token: <GSD_{gsd_token}>.",
    ]
    return " ".join(parts)


def build_condition_prompt_text(row: dict[str, Any]) -> str:
    """构造 proposal 分类/打分所需的语义条件文本。"""
    condition_prompt = infer_condition_prompt(row)
    return f"Condition prompt: {condition_prompt}."


def build_condition_text(row: dict[str, Any]) -> str:
    """兼容完整 prompt：proposal context + condition prompt。"""
    return f"{build_proposal_context_text(row)} {build_condition_prompt_text(row)}"


def canonical_combo_loss_weight(config: QPSalmConfig, combo: str) -> float:
    """按 canonical modality combo 取样本 loss 权重，默认 1。"""
    weights = getattr(config, "canonical_combo_loss_weights", {}) or {}
    if not isinstance(weights, dict):
        return 1.0
    value = weights.get(combo, weights.get(f"canonical_combo={combo}", 1.0))
    try:
        weight = float(value)
    except (TypeError, ValueError):
        return 1.0
    if not math.isfinite(weight) or weight <= 0.0:
        return 1.0
    return weight


class MultiSourceLandslideDataset(Dataset):
    """核心模板 instruction segmentation 数据集。"""

    def __init__(
        self,
        config: QPSalmConfig,
        split: str,
        max_samples: int | None = None,
        shuffle_seed: int | None = None,
    ) -> None:
        self.config = config
        self.split = split
        self.target_size = int(config.target_size)
        index_path = resolve_repo_path(config.index_path(split))
        if index_path is None or not index_path.exists():
            raise FileNotFoundError(f"索引不存在: {config.index_path(split)}")
        self.skipped = Counter()
        usable: list[dict[str, Any]] = []
        self.index_scan_complete = True
        for row in iter_jsonl(index_path):
            reason = should_skip_row(row, config.core_templates)
            if reason is None:
                usable.append(row)
                if max_samples is not None and max_samples > 0 and len(usable) >= max_samples:
                    self.index_scan_complete = False
                    break
            else:
                self.skipped[reason] += 1
        if shuffle_seed is not None:
            rng = random.Random(shuffle_seed)
            rng.shuffle(usable)
        self.rows = usable

    def __len__(self) -> int:
        return len(self.rows)

    def _load_modalities(self, row: dict[str, Any]) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        canonical_tensors = {
            name: torch.zeros((CANONICAL_CHANNELS[name], self.target_size, self.target_size), dtype=torch.float32)
            for name in CANONICAL_MODALITIES
        }
        availability = torch.zeros((len(CANONICAL_MODALITIES),), dtype=torch.float32)
        cursors = {name: 0 for name in CANONICAL_MODALITIES}
        for raw_name, item in (row.get("modalities") or {}).items():
            if not isinstance(item, dict) or not item.get("available", True):
                continue
            canonical = canonical_modality_name(raw_name, item)
            if canonical is None:
                continue
            arr = load_npy_array(str(item.get("path")))
            tensor = normalize_modality(arr, item=item, raw_name=raw_name, canonical=canonical)
            tensor, _ = resize_pad_tensor(tensor, self.target_size, mode="bilinear")
            max_channels = CANONICAL_CHANNELS[canonical]
            start = cursors[canonical]
            if start >= max_channels:
                continue
            take = min(tensor.shape[0], max_channels - start)
            canonical_tensors[canonical][start : start + take] = tensor[:take]
            cursors[canonical] += take
            availability[CANONICAL_MODALITIES.index(canonical)] = 1.0
        return canonical_tensors, availability

    def _apply_train_augment(
        self,
        modalities: dict[str, torch.Tensor],
        mask: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict[str, bool]]:
        """训练期轻量空间增强；所有模态和 mask 保持同一几何变换。"""
        if self.split != "train":
            return modalities, mask, {"hflip": False, "vflip": False}
        hflip_prob = float(getattr(self.config, "train_hflip_prob", 0.0))
        vflip_prob = float(getattr(self.config, "train_vflip_prob", 0.0))
        hflip = bool(hflip_prob > 0.0 and torch.rand(()) < hflip_prob)
        vflip = bool(vflip_prob > 0.0 and torch.rand(()) < vflip_prob)
        if not hflip and not vflip:
            return modalities, mask, {"hflip": False, "vflip": False}
        dims = []
        if vflip:
            dims.append(-2)
        if hflip:
            dims.append(-1)
        augmented_modalities = {
            name: torch.flip(tensor, dims=dims)
            for name, tensor in modalities.items()
        }
        return augmented_modalities, torch.flip(mask, dims=dims), {"hflip": hflip, "vflip": vflip}

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        modalities, availability = self._load_modalities(row)
        mask_info = row.get("mask") or {}
        mask = normalize_mask(load_npy_array(str(mask_info.get("path"))))
        original_mask_size = list(mask.shape[-2:])
        mask, resize_transform = resize_pad_tensor(mask, self.target_size, mode="nearest")
        bbox_raw = parse_bbox_xyxy(mask_info.get("bbox_xyxy") or row.get("bbox_xyxy") or row.get("bbox"))
        bbox = transform_bbox_to_target(bbox_raw, resize_transform) if bbox_raw is not None else None
        if bbox is None:
            bbox = compute_bbox_from_mask(mask)
        is_empty = torch.tensor(float(mask.max().item() <= 0.5), dtype=torch.float32)
        if float(is_empty.item()) > 0.5:
            bbox = None
        modalities, mask, augment = self._apply_train_augment(modalities, mask)
        if float(is_empty.item()) <= 0.5:
            bbox = compute_bbox_from_mask(mask)
        bbox_prior = bbox_to_prior(bbox, self.target_size)
        spatial = row.get("spatial") or {}
        gsd_token = gsd_to_token(spatial.get("gsd_m"))
        condition_prompt = infer_condition_prompt(row)
        proposal_context_text = build_proposal_context_text(row)
        condition_prompt_text = build_condition_prompt_text(row)
        condition_text = f"{proposal_context_text} {condition_prompt_text}"
        canonical_combo = canonical_modality_combo(row)
        raw_combo = raw_modality_combo(row)
        sensors = sensor_combo(row)
        normalizations = normalization_combo(row)
        sample_weight = canonical_combo_loss_weight(self.config, canonical_combo)
        metadata = {
            "sample_id": row.get("sample_id", ""),
            "parent_sample_id": row.get("parent_sample_id", row.get("sample_id", "")),
            "dataset_name": row.get("dataset_name", "unknown"),
            "template_id": row_template_id(row) or "unknown",
            "task_family": row.get("task_family", "unknown"),
            "raw_combo": raw_combo,
            "canonical_combo": canonical_combo,
            "sensor_combo": sensors,
            "normalization_combo": normalizations,
            "quality_flags": list(row.get("quality_flags") or []),
            "gsd_token": gsd_token,
            "gsd_m": spatial.get("gsd_m"),
            "original_size": spatial.get("original_size") or original_mask_size,
            "mask_original_size": original_mask_size,
            "resize_transform": resize_transform,
            "bbox_xyxy": bbox,
            "train_augment": augment,
            "instruction": (row.get("instruction") or {}).get("text", "Segment all landslide regions."),
            "condition_prompt": condition_prompt,
            "proposal_context_text": proposal_context_text,
            "condition_prompt_text": condition_prompt_text,
            "condition_text": condition_text,
            "sample_weight": sample_weight,
        }
        return {
            "modalities": modalities,
            "availability": availability,
            "mask": mask,
            "bbox_prior": bbox_prior,
            "is_empty": is_empty,
            "gsd_id": torch.tensor(gsd_token_id(gsd_token), dtype=torch.long),
            "sample_weight": torch.tensor(sample_weight, dtype=torch.float32),
            "metadata": metadata,
        }


def qpsalm_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """DataLoader collate：保留 metadata list，张量堆叠。"""
    modalities = {
        name: torch.stack([item["modalities"][name] for item in batch], dim=0)
        for name in CANONICAL_MODALITIES
    }
    return {
        "modalities": modalities,
        "availability": torch.stack([item["availability"] for item in batch], dim=0),
        "mask": torch.stack([item["mask"] for item in batch], dim=0),
        "bbox_prior": torch.stack([item["bbox_prior"] for item in batch], dim=0),
        "is_empty": torch.stack([item["is_empty"] for item in batch], dim=0),
        "gsd_id": torch.stack([item["gsd_id"] for item in batch], dim=0),
        "sample_weight": torch.stack([item["sample_weight"] for item in batch], dim=0),
        "metadata": [item["metadata"] for item in batch],
        "proposal_context_text": [item["metadata"]["proposal_context_text"] for item in batch],
        "condition_prompt_text": [item["metadata"]["condition_prompt_text"] for item in batch],
        "condition_text": [item["metadata"]["condition_text"] for item in batch],
    }


def summarize_rows(rows: list[dict[str, Any]], core_templates: Iterable[str]) -> DatasetStats:
    """不加载数组，只统计索引字段。"""
    skipped = Counter()
    usable: list[dict[str, Any]] = []
    for row in rows:
        reason = should_skip_row(row, core_templates)
        if reason is None:
            usable.append(row)
        else:
            skipped[reason] += 1
    by_template = Counter(row_template_id(row) or "unknown" for row in usable)
    by_raw_combo = Counter(raw_modality_combo(row) for row in usable)
    by_canonical_combo = Counter(canonical_modality_combo(row) for row in usable)
    by_sensor_combo = Counter(sensor_combo(row) for row in usable)
    by_normalization_combo = Counter(normalization_combo(row) for row in usable)
    by_shape: dict[str, Counter[str]] = defaultdict(Counter)
    gsd_counter = Counter()
    quality_flags = Counter()
    for row in usable:
        spatial = row.get("spatial") or {}
        gsd_counter[gsd_to_token(spatial.get("gsd_m"))] += 1
        quality_flags.update(str(flag) for flag in row.get("quality_flags") or [])
        for name, item in (row.get("modalities") or {}).items():
            if not isinstance(item, dict) or not item.get("available", True):
                continue
            shape = item.get("shape")
            by_shape[name][str(shape)] += 1
    return DatasetStats(
        num_rows=len(rows),
        num_usable=len(usable),
        skipped_by_reason=dict(sorted(skipped.items())),
        by_template=dict(sorted(by_template.items())),
        by_raw_combo=dict(sorted(by_raw_combo.items())),
        by_canonical_combo=dict(sorted(by_canonical_combo.items())),
        by_sensor_combo=dict(sorted(by_sensor_combo.items())),
        by_normalization_combo=dict(sorted(by_normalization_combo.items())),
        by_shape={key: dict(sorted(value.items())) for key, value in sorted(by_shape.items())},
        gsd_tokens=dict(sorted(gsd_counter.items())),
        quality_flags=dict(sorted(quality_flags.items())),
    )


def stats_to_text(stats: DatasetStats, limit: int | None = None) -> str:
    """把统计结果格式化为便于终端查看的文本。"""
    lines = [
        f"rows={stats.num_rows}",
        f"usable_core_rows={stats.num_usable}",
        f"skipped={stats.skipped_by_reason}",
        "",
        "templates:",
    ]
    for key, value in Counter(stats.by_template).most_common(limit):
        lines.append(f"  {key}: {value}")
    lines.append("raw modality combos:")
    for key, value in Counter(stats.by_raw_combo).most_common(limit):
        lines.append(f"  {key}: {value}")
    lines.append("canonical modality combos:")
    for key, value in Counter(stats.by_canonical_combo).most_common(limit):
        lines.append(f"  {key}: {value}")
    lines.append("sensor combos:")
    for key, value in Counter(stats.by_sensor_combo).most_common(limit):
        lines.append(f"  {key}: {value}")
    lines.append("normalization combos:")
    for key, value in Counter(stats.by_normalization_combo).most_common(limit):
        lines.append(f"  {key}: {value}")
    lines.append(f"gsd_tokens: {stats.gsd_tokens}")
    lines.append(f"quality_flags: {dict(Counter(stats.quality_flags).most_common(limit))}")
    lines.append("shapes:")
    for name, shape_counts in stats.by_shape.items():
        top = Counter(shape_counts).most_common(limit)
        joined = ", ".join(f"{shape}:{count}" for shape, count in top)
        lines.append(f"  {name}: {joined}")
    return "\n".join(lines)


def safe_slug(text: str) -> str:
    """生成文件名安全 slug。"""
    slug = re.sub(r"[^0-9A-Za-z._-]+", "_", text).strip("_")
    return slug[:120] or "sample"
