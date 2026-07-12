#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SANE: sensor-aware raw residual and optional pretrained Qwen-ViT features."""

from __future__ import annotations

import math
import hashlib

import torch
from torch import nn
import torch.nn.functional as F

from qpsalm_seg.schema import MODALITY_FAMILIES, ModalityBatch, ModalityInstance, ModalityPyramid, MultiScaleFeatures

from .common import ConvBlock


FAMILIES = MODALITY_FAMILIES
SENSORS = (
    "generic_rgb", "generic_multiband_optical", "sentinel2", "sentinel1",
    "generic_dem", "alos_palsar_dem", "generic_insar",
)
PRODUCTS = (
    "rgb", "multiband_optical", "surface_reflectance", "sar_backscatter",
    "elevation", "slope", "aspect", "curvature", "los_velocity",
)
ORBITS = ("unknown", "ascending", "descending")
UNITS = (
    "unknown", "digital_number", "reflectance", "db", "meter", "degree",
    "source_native", "source_native_velocity", "meter_per_year", "millimeter_per_year",
)
MEASUREMENT_GEOMETRIES = ("unknown", "line_of_sight", "ground_range", "map_plane")
SIGN_CONVENTIONS = (
    "unknown", "unsigned", "source_defined", "toward_sensor_positive",
    "away_from_sensor_positive", "elevation_positive_up",
)
BANDS = tuple(
    ["R", "G", "B", "VV", "VH", "HH", "HV", "DEM", "SLOPE", "ASPECT", "CURVATURE", "INSAR_VELOCITY"]
    + [f"B{i:02d}" for i in range(1, 33)] + ["B8A"]
)
UNKNOWN_ID_BUCKETS = 32


def _id(value: str, registry: tuple[str, ...], field: str) -> int:
    normalized = value.upper() if field == "band" else value.lower()
    lookup = [item.upper() if field == "band" else item.lower() for item in registry]
    if normalized in lookup:
        return lookup.index(normalized)
    digest = int(hashlib.sha256(f"{field}:{normalized}".encode()).hexdigest()[:8], 16)
    return len(registry) + digest % UNKNOWN_ID_BUCKETS


def _log_gsd(value: float | None) -> tuple[float, float]:
    if value is None or not math.isfinite(value) or value <= 0:
        return 0.0, 0.0
    return max(-2.0, min(3.0, math.log10(value))) / 3.0, 1.0


class SharedPhysicalBandEncoder(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        hidden = max(16, dim // 2)
        self.detail_stem = nn.Sequential(nn.Conv2d(1, hidden, 5, stride=2, padding=2, bias=False), nn.GroupNorm(1, hidden), nn.GELU())
        self.high_stem = nn.Sequential(nn.Conv2d(hidden, dim, 3, stride=2, padding=1, bias=False), nn.GroupNorm(1, dim), nn.GELU())
        self.band_gate = nn.Sequential(nn.LayerNorm(dim * 2), nn.Linear(dim * 2, dim), nn.GELU(), nn.Linear(dim, 1))
        self.detail_project = nn.Conv2d(hidden, dim, 1)

    def forward(
        self,
        bands: torch.Tensor,
        band_tokens: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        detail_per_band = self.detail_stem(bands[:, None])
        high_per_band = self.high_stem(detail_per_band) + band_tokens[:, :, None, None]
        high_valid = F.interpolate(
            valid_mask[None].float(), size=high_per_band.shape[-2:], mode="nearest"
        )
        pooled = (
            (high_per_band * high_valid).sum((2, 3))
            / high_valid.sum((2, 3)).clamp_min(1.0)
        )
        weights = torch.softmax(self.band_gate(torch.cat([pooled, band_tokens], -1)).squeeze(-1).float(), 0).to(bands.dtype)
        detail = (self.detail_project(detail_per_band) * weights[:, None, None, None]).sum(0, keepdim=True)
        high = (high_per_band * weights[:, None, None, None]).sum(0, keepdim=True)
        return detail, high


class SensorAwareNativeScaleEncoder(nn.Module):
    def __init__(self, dim: int, pretrained_bank: nn.Module | None = None) -> None:
        super().__init__()
        self.dim = int(dim)
        self.pretrained_bank = pretrained_bank
        self.band_encoder = SharedPhysicalBandEncoder(dim)
        self.family_embedding = nn.Embedding(len(FAMILIES) + UNKNOWN_ID_BUCKETS, dim)
        self.sensor_embedding = nn.Embedding(len(SENSORS) + UNKNOWN_ID_BUCKETS, dim)
        self.product_embedding = nn.Embedding(len(PRODUCTS) + UNKNOWN_ID_BUCKETS, dim)
        self.band_embedding = nn.Embedding(len(BANDS) + UNKNOWN_ID_BUCKETS, dim)
        self.orbit_embedding = nn.Embedding(len(ORBITS) + UNKNOWN_ID_BUCKETS, dim)
        self.unit_embedding = nn.Embedding(len(UNITS) + UNKNOWN_ID_BUCKETS, dim)
        self.measurement_embedding = nn.Embedding(len(MEASUREMENT_GEOMETRIES) + UNKNOWN_ID_BUCKETS, dim)
        self.sign_embedding = nn.Embedding(len(SIGN_CONVENTIONS) + UNKNOWN_ID_BUCKETS, dim)
        self.modality_physical = nn.Sequential(nn.Linear(5, dim), nn.GELU(), nn.Linear(dim, dim))
        self.band_physical = nn.Sequential(nn.Linear(7, dim), nn.GELU(), nn.Linear(dim, dim))
        self.family_high = nn.ModuleDict({family: ConvBlock(dim, dim) for family in FAMILIES})
        self.family_mid = nn.ModuleDict({family: nn.Sequential(nn.Conv2d(dim, dim, 3, 2, 1), nn.GELU(), ConvBlock(dim, dim)) for family in FAMILIES})
        self.family_low = nn.ModuleDict({family: nn.Sequential(nn.Conv2d(dim, dim, 3, 2, 1), nn.GELU(), ConvBlock(dim, dim)) for family in FAMILIES})
        pretrained_channels = int(getattr(pretrained_bank, "spatial_channels", 1024))
        self.pretrained_adapters = nn.ModuleList([
            nn.Sequential(nn.Conv2d(pretrained_channels, dim, 1), nn.GroupNorm(1, dim), nn.GELU())
            for _ in range(4)
        ])
        self.raw_residual_scale = nn.Parameter(torch.full((4,), -4.0))

    def _metadata(self, item: ModalityInstance, device, dtype):
        native, native_known = _log_gsd(item.native_gsd_m)
        aligned, aligned_known = _log_gsd(item.aligned_gsd_m)
        values = torch.tensor([native, native_known, aligned, aligned_known, item.quality], device=device, dtype=dtype)
        return (
            self.family_embedding.weight[_id(item.family, FAMILIES, "family")]
            + self.sensor_embedding.weight[_id(item.sensor, SENSORS, "sensor")]
            + self.product_embedding.weight[_id(item.product_type, PRODUCTS, "product")]
            + self.orbit_embedding.weight[_id(item.orbit, ORBITS, "orbit")]
            + self.unit_embedding.weight[_id(item.units, UNITS, "unit")]
            + self.modality_physical(values)
        )

    def _band_token(self, name: str, meta: dict, item: ModalityInstance, device, dtype):
        center = float(meta.get("center_wavelength_nm") or 0.0) / 2500.0
        width = float(meta.get("bandwidth_nm") or 0.0) / 500.0
        gsd, known = _log_gsd(meta.get("native_gsd_m"))
        polarization = {None: 0.0, "VV": 0.25, "VH": 0.5, "HH": 0.75, "HV": 1.0}.get(meta.get("polarization"), 0.0)
        values = torch.tensor([center, width, gsd, known, polarization, float(meta.get("signed", item.signed)), item.quality], device=device, dtype=dtype)
        geometry = str(meta.get("measurement_geometry") or "unknown")
        sign = str(meta.get("sign_convention") or ("source_defined" if item.signed else "unsigned"))
        units = str(meta.get("units") or item.units)
        return (
            self.band_embedding.weight[_id(name, BANDS, "band")]
            + self.unit_embedding.weight[_id(units, UNITS, "unit")]
            + self.measurement_embedding.weight[_id(geometry, MEASUREMENT_GEOMETRIES, "measurement")]
            + self.sign_embedding.weight[_id(sign, SIGN_CONVENTIONS, "sign")]
            + self.band_physical(values)
        )

    def _encode_instance(self, item: ModalityInstance, device) -> ModalityPyramid:
        image = item.image.to(device=device, dtype=torch.float32, non_blocking=True)
        valid_native = item.valid_mask.to(
            device=device, dtype=image.dtype, non_blocking=True
        )
        image = image * valid_native
        metadata = self._metadata(item, device, image.dtype)
        band_tokens = torch.stack([
            self._band_token(name, meta, item, device, image.dtype)
            for name, meta in zip(item.band_names, item.band_metadata)
        ]) + metadata
        raw_detail, raw_high = self.band_encoder(image, band_tokens, valid_native)
        raw_high = self.family_high[item.family](raw_high)
        raw_mid = self.family_mid[item.family](raw_high)
        raw_low = self.family_low[item.family](raw_mid)
        detail, high, mid, low = raw_detail, raw_high, raw_mid, raw_low
        if self.pretrained_bank is not None:
            cached = self.pretrained_bank.features_for(item, device=device)
            if cached:
                adapted = [
                    adapter(feature[None].to(next(adapter.parameters()).dtype))
                    for adapter, feature in zip(self.pretrained_adapters, cached)
                ]
                pretrained_sources = adapted
                target_sizes = [detail.shape[-2:], high.shape[-2:], mid.shape[-2:], low.shape[-2:]]
                pretrained = [
                    F.interpolate(value, size=size, mode="bilinear", align_corners=False)
                    for value, size in zip(pretrained_sources, target_sizes)
                ]
                raw = [detail, high, mid, low]
                scales = torch.sigmoid(self.raw_residual_scale)
                detail, high, mid, low = [
                    pre + scales[i] * value
                    for i, (pre, value) in enumerate(zip(pretrained, raw))
                ]
        valid = valid_native.to(dtype=detail.dtype)[None]
        masks = [self._valid_at_scale(valid, value.shape[-2:])[0] for value in (detail, high, mid, low)]
        return ModalityPyramid(
            instance=item, detail=detail[0], high=high[0], mid=mid[0], low=low[0],
            detail_valid=masks[0], high_valid=masks[1], mid_valid=masks[2], low_valid=masks[3],
            metadata_token=metadata, active=True,
        )

    @staticmethod
    def _valid_at_scale(valid: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
        if target_hw[0] <= valid.shape[-2] and target_hw[1] <= valid.shape[-1]:
            return F.adaptive_max_pool2d(valid, target_hw)
        return F.interpolate(valid, size=target_hw, mode="nearest")

    def forward(self, batch: ModalityBatch, use_full: bool = False) -> MultiScaleFeatures:
        device = next(self.parameters()).device
        source = batch.full_instances if use_full else batch.instances
        samples = [[self._encode_instance(item, device) for item in instances] for instances in source]
        return MultiScaleFeatures(samples=samples, reference_hw=batch.reference_hw)
