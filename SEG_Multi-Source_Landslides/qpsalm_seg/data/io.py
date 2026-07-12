#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Strict benchmark-v2 array loading and modality construction."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from qpsalm_seg.paths import resolve_project_path
from qpsalm_seg.schema import MODALITY_FAMILIES, ModalityInstance


SCHEMA_VERSION = "multisource_landslide_schema_v2"
FAMILIES = set(MODALITY_FAMILIES)


def positive_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) and parsed > 0.0 else None


def load_npy_array(path_ref: str) -> np.ndarray:
    path = resolve_project_path(path_ref)
    if path is None or not path.exists():
        raise FileNotFoundError(f"数组路径不存在: {path_ref}")
    arr = np.load(path)
    if arr.ndim == 2:
        arr = arr[None]
    if arr.ndim != 3:
        raise ValueError(f"benchmark v2 模态必须是 [C,H,W]: path={path_ref} shape={arr.shape}")
    return np.asarray(arr)


def normalize_materialized(arr: np.ndarray, modality: dict[str, Any]) -> torch.Tensor:
    """Trust v2 materialization; never infer a second robust stretch at training time."""
    normalization = modality.get("normalization")
    if not isinstance(normalization, dict) or not normalization.get("method"):
        raise ValueError("benchmark v2 modality 缺少结构化 normalization")
    values = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if str(normalization["method"]) == "preserve_rgb_values":
        values = values / 255.0
    if bool(modality["signed"]):
        values = np.clip(values, -1.0, 1.0)
    else:
        values = np.clip(values, 0.0, 1.0)
    return torch.from_numpy(values.astype(np.float32, copy=False))


def modality_valid_mask(arr: np.ndarray, modality: dict[str, Any]) -> torch.Tensor:
    spec = modality.get("valid_mask")
    if not isinstance(spec, dict):
        raise ValueError("benchmark v2 modality 缺少 valid_mask 规范")
    if spec.get("path"):
        stored = load_npy_array(str(spec["path"]))
        if stored.shape[0] != 1 or stored.shape[-2:] != arr.shape[-2:]:
            raise ValueError(f"物化 valid mask shape 不匹配: valid={stored.shape} modality={arr.shape}")
        return torch.from_numpy((stored > 0).astype(np.float32, copy=False))
    valid = np.isfinite(arr).all(axis=0)
    nodata = spec.get("nodata_value")
    if isinstance(nodata, (int, float)) and np.isfinite(float(nodata)):
        invalid = np.isclose(arr.astype(np.float64), float(nodata), rtol=0.0, atol=1.0e-8).all(axis=0)
        valid &= ~invalid
    return torch.from_numpy(valid.astype(np.float32))[None]


def validate_modality_schema(name: str, item: dict[str, Any]) -> None:
    required = {
        "path", "format", "family", "sensor", "product_type", "band_names",
        "band_metadata", "native_gsd_m", "units", "signed", "orbit", "quality",
        "normalization", "valid_mask",
    }
    missing = sorted(required - set(item))
    if missing:
        raise ValueError(f"模态 {name} 缺少 benchmark v2 字段: {missing}")
    if item["family"] not in FAMILIES:
        raise ValueError(f"模态 {name} family 非法: {item['family']!r}")
    if not item["sensor"] or not item["product_type"]:
        raise ValueError(f"模态 {name} sensor/product_type 不能为空")
    if len(item["band_names"]) != len(item["band_metadata"]):
        raise ValueError(f"模态 {name} band_names/band_metadata 数量不一致")
    required_band = {
        "name", "native_gsd_m", "center_wavelength_nm", "bandwidth_nm",
        "polarization", "units", "signed", "measurement_geometry", "sign_convention",
    }
    for band_name, metadata in zip(item["band_names"], item["band_metadata"]):
        if not isinstance(metadata, dict):
            raise ValueError(f"模态 {name} band={band_name} metadata 必须是 dict")
        missing_band = sorted(required_band - set(metadata))
        if missing_band:
            raise ValueError(f"模态 {name} band={band_name} 缺少物理字段: {missing_band}")
        if str(metadata["name"]) != str(band_name):
            raise ValueError(f"模态 {name} band metadata 顺序或名称不一致: {band_name}")
        if not metadata["units"]:
            raise ValueError(f"模态 {name} band={band_name} units 不能为空")
    if not isinstance(item["valid_mask"], dict) or not item["valid_mask"].get("path"):
        raise ValueError(f"模态 {name} 必须使用归一化前物化的 valid mask")


def build_modality_instance(name: str, item: dict[str, Any], aligned_gsd_m: float | None) -> ModalityInstance:
    validate_modality_schema(name, item)
    arr = load_npy_array(str(item["path"]))
    if item.get("shape") and list(arr.shape) != list(item["shape"]):
        raise ValueError(f"模态 {name} shape 与索引不一致: array={arr.shape} index={item['shape']}")
    names = tuple(str(value) for value in item["band_names"])
    if len(names) != int(arr.shape[0]):
        raise ValueError(f"模态 {name} channel/band 数量不一致: C={arr.shape[0]} bands={len(names)}")
    return ModalityInstance(
        name=name,
        family=str(item["family"]),
        sensor=str(item["sensor"]),
        product_type=str(item["product_type"]),
        band_names=names,
        band_metadata=tuple(dict(value) for value in item["band_metadata"]),
        orbit=str(item["orbit"]),
        units=str(item["units"]),
        signed=bool(item["signed"]),
        image=normalize_materialized(arr, item),
        valid_mask=modality_valid_mask(arr, item),
        native_gsd_m=positive_float(item.get("native_gsd_m")),
        aligned_gsd_m=aligned_gsd_m,
        quality=float(item["quality"]),
        metadata=dict(item),
    )


def normalize_mask(arr: np.ndarray) -> torch.Tensor:
    mask = (np.nan_to_num(arr.astype(np.float32)) > 0).astype(np.float32)
    if mask.ndim == 2:
        mask = mask[None]
    if mask.ndim != 3 or mask.shape[0] != 1:
        raise ValueError(f"mask 必须是 [1,H,W]，当前 {mask.shape}")
    return torch.from_numpy(mask)
