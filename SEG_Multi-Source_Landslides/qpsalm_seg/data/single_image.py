#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""描述 benchmark 单图到 ModalityInstance 的唯一适配入口。

用途：把 RSICap、RSIEval、MMRS Caption 和 DIOR-RSVG 的 RGB 图像转换为
现有 SANE 可消费的 optical ModalityInstance。
主要输入：qpsalm_description_v2 记录中的 single_image visual_ref。
主要输出：ModalityInstance；本模块不写文件，也不伪造 GSD 或传感器物理量。
运行方式：内部公共模块，不作为独立程序运行。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import Image

from qpsalm_seg.paths import resolve_project_path
from qpsalm_seg.schema import ModalityInstance


def _band_metadata(name: str) -> dict[str, Any]:
    return {
        "name": name,
        "native_gsd_m": None,
        "center_wavelength_nm": None,
        "bandwidth_nm": None,
        "polarization": None,
        "units": "display_rgb",
        "signed": False,
        "measurement_geometry": "unknown",
        "sign_convention": "not_applicable",
    }


def build_single_image_modality_instance(visual_ref: dict[str, Any]) -> ModalityInstance:
    """严格解码一个 description single_image 引用并构造光学模态。"""
    if visual_ref.get("type") != "single_image":
        raise ValueError(f"仅支持 single_image visual_ref，当前为 {visual_ref.get('type')!r}")
    path = resolve_project_path(visual_ref.get("path"))
    if path is None or not path.exists():
        raise FileNotFoundError(f"单图路径不存在: {visual_ref.get('path')}")

    with Image.open(path) as source:
        source.load()
        if source.mode not in {"RGB", "RGBA"}:
            raise ValueError(f"描述 benchmark 正式输入必须是 RGB/RGBA，当前 mode={source.mode} path={path}")
        alpha = None
        if source.mode == "RGBA":
            alpha = np.asarray(source.getchannel("A"), dtype=np.uint8)
        rgb = np.asarray(source.convert("RGB"), dtype=np.float32)

    height, width = rgb.shape[:2]
    if visual_ref.get("width") not in {None, width} or visual_ref.get("height") not in {None, height}:
        raise ValueError(
            f"visual_ref 尺寸与图像不一致: index=({visual_ref.get('width')},{visual_ref.get('height')}) "
            f"decoded=({width},{height})"
        )
    valid = np.ones((height, width), dtype=np.float32) if alpha is None else (alpha > 0).astype(np.float32)
    if not bool(valid.any()):
        raise ValueError(f"图像没有有效像素: {path}")

    metadata = dict(visual_ref.get("modality_instance") or {})
    sensor = str(metadata.get("sensor") or "generic_aerial_rgb")
    quality_value = metadata.get("quality", 1.0)
    quality = float(quality_value) if isinstance(quality_value, (int, float)) else 1.0
    return ModalityInstance(
        name="single_image_rgb",
        family="optical",
        sensor=sensor,
        product_type="rgb",
        band_names=("R", "G", "B"),
        band_metadata=tuple(_band_metadata(name) for name in ("R", "G", "B")),
        orbit="not_applicable",
        units="display_rgb",
        signed=False,
        image=torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1) / 255.0)),
        valid_mask=torch.from_numpy(valid[None]),
        native_gsd_m=None,
        aligned_gsd_m=None,
        quality=quality,
        metadata={**metadata, "source_type": "single_image", "source_path": str(visual_ref["path"])},
    )

