#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""多源滑坡 instruction segmentation 数据读取。

脚本作用：读取 small/full benchmark 的 instruction JSONL，构造变长原生尺度
模态实例、有效区域、尺寸桶、地学元数据和统一 semantic-evidence prompts。
主要输入：indexes/instruction_train.jsonl、indexes/instruction_val.jsonl。
主要输出：PyTorch Dataset/DataLoader batch。
是否改写原始数据：不会。
典型用法：MultiSourceLandslideDataset(config, split="train")。
"""

from __future__ import annotations

import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler

from .config import QPSalmConfig
from .indexing import (
    available_modality_names,
    canonical_modality_combo,
    canonical_modality_name,
    gsd_to_token,
    iter_jsonl,
    normalization_combo,
    raw_modality_combo,
    row_template_id,
    sensor_combo,
    should_skip_row,
)
from .paths import resolve_repo_path
from .schema import ModalityBatch, ModalityInstance


GSD_TOKENS = ["unknown", "sub_meter", "meter_1_5", "meter_5_10", "meter_gt_10"]


def _safe_positive_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) and parsed > 0.0 else None


def gsd_token_id(token: str) -> int:
    return GSD_TOKENS.index(token) if token in GSD_TOKENS else 0


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
    source_dtype = arr.dtype
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

    if source_dtype == np.uint8 or "preserve_rgb_values" in normalization:
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


def modality_valid_mask(arr: np.ndarray, item: dict[str, Any] | None = None) -> torch.Tensor:
    """由 finite 值和显式 nodata 元数据生成逐模态空间有效区。"""
    item = item or {}
    valid = np.isfinite(arr).all(axis=0)
    nodata = item.get("nodata_value", item.get("nodata", item.get("no_data")))
    if isinstance(nodata, (int, float)) and np.isfinite(float(nodata)):
        nodata_pixels = np.isclose(arr.astype(np.float64), float(nodata), rtol=0.0, atol=1.0e-8).all(axis=0)
        valid &= ~nodata_pixels
    return torch.from_numpy(valid.astype(np.float32))[None]


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


def valid_mask_from_transform(transform: dict[str, Any]) -> torch.Tensor:
    """根据 resize+pad 变换生成有效像素区域，形状为 [1,H,W]。"""
    target_hw = transform.get("target_hw") or [1, 1]
    resized_hw = transform.get("resized_hw") or [1, 1]
    target_h, target_w = int(target_hw[0]), int(target_hw[1])
    resized_h, resized_w = int(resized_hw[0]), int(resized_hw[1])
    pad_top = int(transform.get("pad_top", 0))
    pad_left = int(transform.get("pad_left", 0))
    valid = torch.zeros((1, target_h, target_w), dtype=torch.float32)
    valid[:, pad_top : pad_top + resized_h, pad_left : pad_left + resized_w] = 1.0
    return valid


def _preview_three_channels(tensor: torch.Tensor) -> torch.Tensor:
    """把任意模态张量压成 3 通道，仅供导出诊断图。"""
    if tensor.shape[0] >= 3:
        return tensor[:3].float().clamp(-1.0, 1.0)
    if tensor.shape[0] == 2:
        mean = tensor[:2].mean(dim=0, keepdim=True)
        return torch.cat([tensor[:2], mean], dim=0).float().clamp(-1.0, 1.0)
    if tensor.shape[0] == 1:
        return tensor[:1].expand(3, -1, -1).float().clamp(-1.0, 1.0)
    raise ValueError(f"无法从空模态构造 preview: shape={tuple(tensor.shape)}")


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
        f"GSD meters: {spatial.get('gsd_m') if spatial.get('gsd_m') not in (None, '') else 'unknown'}.",
    ]
    return " ".join(parts)


def build_condition_prompt_text(row: dict[str, Any]) -> str:
    """构造 proposal 分类/打分所需的语义条件文本。"""
    condition_prompt = infer_condition_prompt(row)
    return f"Condition prompt: {condition_prompt}."


def build_evidence_reasoning_text(row: dict[str, Any]) -> str:
    """构造 Qwen semantic/evidence controller 的地学证据推理文本。

    这条 prompt 让 controller 明确当前任务应该如何调度光学、S2、SAR、DEM
    和 InSAR 证据，并在 proposal 选择时避免把
    道路、河谷、裸地或阴影当成滑坡。
    """
    condition_prompt = infer_condition_prompt(row)
    canonical = set(canonical_modality_combo(row).split("+"))
    role_hints = []
    if "hr_optical" in canonical:
        role_hints.append("HR optical: inspect scar texture, exposed soil, vegetation disruption, and sharp local contrast.")
    if "s2" in canonical:
        role_hints.append("Sentinel-2: use multispectral or RGB reflectance cues for vegetation-soil contrast at coarser scale.")
    if "s1" in canonical:
        role_hints.append("Sentinel-1 SAR: use roughness, moisture, and structural backscatter as support under cloud or shadow.")
    if "dem" in canonical:
        role_hints.append("DEM/slope: check terrain plausibility, steep slope context, source area, and runout consistency.")
    if "insar" in canonical:
        role_hints.append("InSAR deformation: treat signed deformation as activity evidence, not as ordinary texture.")
    if not role_hints:
        role_hints.append("No strong auxiliary modality is available; rely on visible landslide morphology.")

    spatial = row.get("spatial") or {}
    gsd_value = spatial.get("gsd_m") if spatial.get("gsd_m") not in (None, "") else "unknown"
    instruction = row.get("instruction") or {}
    task_text = str(instruction.get("text") or "Segment all landslide regions.")
    parts = [
        f"Evidence reasoning target: {condition_prompt}.",
        f"Task instruction: {task_text}",
        f"Dataset/source: {row.get('dataset_name', 'unknown')}.",
        f"Available evidence: {canonical_modality_combo(row)}.",
        f"Raw evidence roles: {raw_modality_combo(row)}.",
        f"Sensor combo: {sensor_combo(row)}.",
        f"Scale: GSD {gsd_value} meters, token <GSD_{gsd_to_token(spatial.get('gsd_m'))}>.",
        "Use Qwen as a semantic controller, evidence scheduler, and proposal verifier.",
        "Landslides may be irregular patches with fuzzy boundaries and multiple instances.",
        "Prefer mask proposals supported by morphology plus terrain or deformation evidence when available.",
        "Reject background look-alikes such as roads, riverbeds, shadows, bare soil, terraces, or continuous slope texture without landslide evidence.",
        "Evidence roles: " + " ".join(role_hints),
    ]
    return " ".join(parts)


def build_visual_evidence_key(row: dict[str, Any]) -> str:
    """Visual cache is shared by all instructions derived from one physical patch."""
    parent = str(row.get("parent_sample_id") or row.get("sample_id") or "")
    return f"qmv-parent:{parent}"


def raw_modality_metadata(row: dict[str, Any]) -> list[dict[str, Any]]:
    """收集原始模态元数据，供完整多模态推理可视化使用。"""
    records: list[dict[str, Any]] = []
    for raw_name, item in sorted((row.get("modalities") or {}).items()):
        if not isinstance(item, dict) or not item.get("available", True):
            continue
        records.append(
            {
                "name": raw_name,
                "canonical": canonical_modality_name(raw_name, item),
                "path": item.get("path"),
                "shape": item.get("shape"),
                "sensor": item.get("sensor"),
                "normalization": item.get("normalization"),
                "value_encoding": item.get("value_encoding"),
                "role": item.get("role"),
                "band_names": item.get("band_names") or item.get("bands"),
                "gsd_m": item.get("gsd_m"),
                "units": item.get("units"),
            }
        )
    return records


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

    def bucket_size(self, index: int) -> int:
        buckets = sorted({int(value) for value in getattr(self.config, "size_buckets", []) if int(value) > 0})
        if not buckets:
            return self.target_size
        row = self.rows[index]
        spatial = row.get("spatial") or {}
        shape = spatial.get("original_size") or (row.get("mask") or {}).get("shape") or [self.target_size, self.target_size]
        longest = max(int(shape[-2]), int(shape[-1])) if isinstance(shape, (list, tuple)) and len(shape) >= 2 else self.target_size
        requested = min(longest, buckets[-1])
        return next((bucket for bucket in buckets if bucket >= requested), buckets[-1])

    @staticmethod
    def _modality_family(raw_name: str, item: dict[str, Any]) -> str | None:
        canonical = canonical_modality_name(raw_name, item)
        return {
            "hr_optical": "optical",
            "s2": "multispectral",
            "s1": "sar",
            "dem": "terrain",
            "insar": "deformation",
        }.get(str(canonical))

    @staticmethod
    def _quality_score(row: dict[str, Any], item: dict[str, Any]) -> float:
        flags = list(row.get("quality_flags") or []) + list(item.get("quality_flags") or [])
        severe = sum(any(token in str(flag).lower() for token in ("missing", "invalid", "misalign", "corrupt")) for flag in flags)
        uncertain = sum(any(token in str(flag).lower() for token in ("unknown", "fallback", "nonstandard")) for flag in flags)
        return max(0.2, 1.0 - 0.2 * severe - 0.08 * uncertain)

    @staticmethod
    def _downscale_native(tensor: torch.Tensor, max_side: int, mode: str = "bilinear") -> torch.Tensor:
        height, width = int(tensor.shape[-2]), int(tensor.shape[-1])
        if max(height, width) <= max_side:
            return tensor
        scale = float(max_side) / float(max(height, width))
        target = (max(1, int(round(height * scale))), max(1, int(round(width * scale))))
        if mode == "nearest":
            return F.interpolate(tensor.unsqueeze(0), size=target, mode=mode).squeeze(0)
        return F.interpolate(tensor.unsqueeze(0), size=target, mode=mode, align_corners=False).squeeze(0)

    def _load_modalities(self, row: dict[str, Any], target_size: int) -> list[ModalityInstance]:
        instances: list[ModalityInstance] = []
        spatial = row.get("spatial") or {}
        aligned_gsd = _safe_positive_float(spatial.get("gsd_m"))
        max_native_size = min(int(getattr(self.config, "max_native_size", 384)), int(target_size))
        for raw_name, item in (row.get("modalities") or {}).items():
            if not isinstance(item, dict) or not item.get("available", True):
                continue
            family = self._modality_family(raw_name, item)
            if family is None:
                continue
            arr = load_npy_array(str(item.get("path")))
            valid_mask = modality_valid_mask(arr, item)
            canonical = canonical_modality_name(raw_name, item)
            tensor = normalize_modality(arr, item=item, raw_name=raw_name, canonical=canonical)
            tensor = self._downscale_native(tensor, max_native_size)
            valid_mask = self._downscale_native(valid_mask, max_native_size, mode="nearest")
            valid_mask = (valid_mask >= 0.5).float()
            band_names = item.get("band_names") or item.get("bands") or []
            names = tuple(str(name) for name in band_names)
            if len(names) != int(tensor.shape[0]):
                names = tuple(f"band_{index}" for index in range(int(tensor.shape[0])))
            native_gsd = _safe_positive_float(item.get("gsd_m")) or aligned_gsd
            orbit = "ascending" if "asc" in raw_name else ("descending" if "dsc" in raw_name else "unknown")
            instances.append(
                ModalityInstance(
                    name=str(raw_name),
                    family=family,
                    sensor=str(item.get("sensor") or "unknown"),
                    band_names=names,
                    orbit=orbit,
                    image=tensor,
                    valid_mask=valid_mask,
                    native_gsd_m=native_gsd,
                    aligned_gsd_m=aligned_gsd or native_gsd,
                    quality=self._quality_score(row, item),
                    metadata=dict(item),
                )
            )
        if not instances:
            raise ValueError(f"样本没有可编码模态: {row.get('sample_id', '<unknown>')}")
        return instances

    def _apply_train_augment(
        self,
        instances: list[ModalityInstance],
        mask: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[list[ModalityInstance], torch.Tensor, torch.Tensor, dict[str, bool]]:
        """训练期轻量空间增强；所有模态和 mask 保持同一几何变换。"""
        if self.split != "train":
            return instances, mask, valid_mask, {"hflip": False, "vflip": False}
        hflip_prob = float(getattr(self.config, "train_hflip_prob", 0.0))
        vflip_prob = float(getattr(self.config, "train_vflip_prob", 0.0))
        hflip = bool(hflip_prob > 0.0 and torch.rand(()) < hflip_prob)
        vflip = bool(vflip_prob > 0.0 and torch.rand(()) < vflip_prob)
        if not hflip and not vflip:
            return instances, mask, valid_mask, {"hflip": False, "vflip": False}
        dims = []
        if vflip:
            dims.append(-2)
        if hflip:
            dims.append(-1)
        augmented_instances = [
            ModalityInstance(
                name=item.name,
                family=item.family,
                sensor=item.sensor,
                band_names=item.band_names,
                orbit=item.orbit,
                image=torch.flip(item.image, dims=dims),
                valid_mask=torch.flip(item.valid_mask, dims=dims),
                native_gsd_m=item.native_gsd_m,
                aligned_gsd_m=item.aligned_gsd_m,
                quality=item.quality,
                metadata=item.metadata,
            )
            for item in instances
        ]
        return (
            augmented_instances,
            torch.flip(mask, dims=dims),
            torch.flip(valid_mask, dims=dims),
            {"hflip": hflip, "vflip": vflip},
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        target_size = self.bucket_size(index)
        instances = self._load_modalities(row, target_size)
        mask_info = row.get("mask") or {}
        mask = normalize_mask(load_npy_array(str(mask_info.get("path"))))
        original_mask_size = list(mask.shape[-2:])
        mask, resize_transform = resize_pad_tensor(mask, target_size, mode="nearest")
        valid_mask = valid_mask_from_transform(resize_transform)
        instances, mask, valid_mask, augment = self._apply_train_augment(instances, mask, valid_mask)
        priority = {"optical": 0, "multispectral": 1, "sar": 2, "terrain": 3, "deformation": 4}
        preview_instance = min(instances, key=lambda item: priority.get(item.family, 99))
        preview = _preview_three_channels(preview_instance.image)
        visual_preview, _ = resize_pad_tensor(preview, target_size, mode="bilinear")
        if float(visual_preview.min().item()) < 0.0:
            visual_preview = (visual_preview + 1.0) * 0.5
        visual_preview = visual_preview.clamp(0.0, 1.0)
        visual_preview_source = preview_instance.name
        spatial = row.get("spatial") or {}
        gsd_token = gsd_to_token(spatial.get("gsd_m"))
        condition_prompt = infer_condition_prompt(row)
        proposal_context_text = build_proposal_context_text(row)
        condition_prompt_text = build_condition_prompt_text(row)
        evidence_reasoning_text = build_evidence_reasoning_text(row)
        visual_evidence_key = build_visual_evidence_key(row)
        canonical_combo = canonical_modality_combo(row)
        raw_combo = raw_modality_combo(row)
        sensors = sensor_combo(row)
        normalizations = normalization_combo(row)
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
            "target_size": target_size,
            "train_augment": augment,
            "instruction": (row.get("instruction") or {}).get("text", "Segment all landslide regions."),
            "condition_prompt": condition_prompt,
            "proposal_context_text": proposal_context_text,
            "condition_prompt_text": condition_prompt_text,
            "evidence_reasoning_text": evidence_reasoning_text,
            "visual_evidence_key": visual_evidence_key,
            "visual_preview_source": visual_preview_source,
            "raw_modalities": raw_modality_metadata(row),
        }
        return {
            "instances": instances,
            "visual_preview": visual_preview,
            "mask": mask,
            "valid_mask": valid_mask,
            "gsd_id": torch.tensor(gsd_token_id(gsd_token), dtype=torch.long),
            "metadata": metadata,
        }


def qpsalm_collate(batch: list[dict[str, Any]]) -> ModalityBatch:
    """构造允许不同模态数量和通道数的 typed batch。"""
    shapes = {tuple(item["mask"].shape[-2:]) for item in batch}
    if len(shapes) != 1:
        raise ValueError(f"同一 batch 必须来自同一尺寸桶，收到 {sorted(shapes)}")
    return ModalityBatch(
        instances=[item["instances"] for item in batch],
        visual_preview=torch.stack([item["visual_preview"] for item in batch], dim=0),
        mask=torch.stack([item["mask"] for item in batch], dim=0),
        valid_mask=torch.stack([item["valid_mask"] for item in batch], dim=0),
        metadata=[item["metadata"] for item in batch],
        proposal_context_text=[item["metadata"]["proposal_context_text"] for item in batch],
        condition_prompt_text=[item["metadata"]["condition_prompt_text"] for item in batch],
        evidence_reasoning_text=[item["metadata"]["evidence_reasoning_text"] for item in batch],
        visual_evidence_key=[item["metadata"]["visual_evidence_key"] for item in batch],
    )


class SizeBucketBatchSampler(Sampler[list[int]]):
    """按 reference canvas 分桶，避免同一 batch 内重复 padding 到最大样本。"""

    def __init__(
        self,
        dataset: MultiSourceLandslideDataset,
        batch_size: int,
        *,
        shuffle: bool,
        seed: int,
        drop_last: bool = False,
    ) -> None:
        self.dataset = dataset
        self.batch_size = max(1, int(batch_size))
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0

    def __iter__(self):
        groups: dict[int, list[int]] = defaultdict(list)
        for index in range(len(self.dataset)):
            groups[self.dataset.bucket_size(index)].append(index)
        rng = random.Random(self.seed + self.epoch)
        batches: list[list[int]] = []
        for indices in groups.values():
            if self.shuffle:
                rng.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start : start + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)
        if self.shuffle:
            rng.shuffle(batches)
        self.epoch += 1
        yield from batches

    def __len__(self) -> int:
        counts = Counter(self.dataset.bucket_size(index) for index in range(len(self.dataset)))
        if self.drop_last:
            return sum(count // self.batch_size for count in counts.values())
        return sum((count + self.batch_size - 1) // self.batch_size for count in counts.values())


def safe_slug(text: str) -> str:
    """生成文件名安全 slug。"""
    slug = re.sub(r"[^0-9A-Za-z._-]+", "_", text).strip("_")
    return slug[:120] or "sample"
