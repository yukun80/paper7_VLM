#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sensor-Aware Native-Scale Encoder (SANE)."""

from __future__ import annotations

import hashlib
import math

import torch
from torch import nn
import torch.nn.functional as F

from qpsalm_seg.schema import ModalityBatch, ModalityInstance, ModalityPyramid, MultiScaleFeatures

from .common import ConvBlock


FAMILIES = ("optical", "multispectral", "sar", "terrain", "deformation")
ORBITS = ("unknown", "ascending", "descending")


def _hash_bucket(value: str, buckets: int) -> int:
    digest = hashlib.sha1(value.lower().encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little") % int(buckets)


def _log_gsd(value: float | None) -> tuple[float, float]:
    if value is None or not math.isfinite(value) or value <= 0.0:
        return 0.0, 0.0
    return max(-2.0, min(3.0, math.log10(value))) / 3.0, 1.0


class SharedBandPyramid(nn.Module):
    """Encode each physical band independently before learned band aggregation."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        hidden = max(16, dim // 2)
        self.stem = nn.Sequential(
            nn.Conv2d(1, hidden, kernel_size=5, stride=2, padding=2, bias=False),
            nn.GroupNorm(1, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(1, dim),
            nn.GELU(),
        )
        self.band_gate = nn.Sequential(nn.LayerNorm(dim * 2), nn.Linear(dim * 2, dim), nn.GELU(), nn.Linear(dim, 1))

    def forward(self, bands: torch.Tensor, band_tokens: torch.Tensor) -> torch.Tensor:
        features = self.stem(bands[:, None])
        features = features + band_tokens[:, :, None, None]
        pooled = features.mean(dim=(2, 3))
        logits = self.band_gate(torch.cat([pooled, band_tokens], dim=-1)).squeeze(-1)
        weights = torch.softmax(logits.float(), dim=0).to(features.dtype)
        return (features * weights[:, None, None, None]).sum(dim=0, keepdim=True)


class SensorAwareNativeScaleEncoder(nn.Module):
    """Preserve sensor, band, orbit, GSD and quality semantics without channel truncation."""

    def __init__(
        self,
        dim: int,
        modality_dropout: float = 0.2,
        sensor_buckets: int = 64,
        band_buckets: int = 128,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.modality_dropout = float(modality_dropout)
        self.band_encoder = SharedBandPyramid(dim)
        self.family_embedding = nn.Embedding(len(FAMILIES), dim)
        self.sensor_embedding = nn.Embedding(sensor_buckets, dim)
        self.band_embedding = nn.Embedding(band_buckets, dim)
        self.orbit_embedding = nn.Embedding(len(ORBITS), dim)
        self.scale_quality = nn.Sequential(nn.Linear(5, dim), nn.GELU(), nn.Linear(dim, dim))
        self.family_high = nn.ModuleDict({family: ConvBlock(dim, dim) for family in FAMILIES})
        self.family_mid = nn.ModuleDict(
            {family: nn.Sequential(nn.Conv2d(dim, dim, 3, stride=2, padding=1), nn.GELU(), ConvBlock(dim, dim)) for family in FAMILIES}
        )
        self.family_low = nn.ModuleDict(
            {family: nn.Sequential(nn.Conv2d(dim, dim, 3, stride=2, padding=1), nn.GELU(), ConvBlock(dim, dim)) for family in FAMILIES}
        )

    @staticmethod
    def _gradient_bands(image: torch.Tensor) -> torch.Tensor:
        dx = image[:, :, 1:] - image[:, :, :-1]
        dy = image[:, 1:, :] - image[:, :-1, :]
        dx = F.pad(dx, (0, 1, 0, 0))
        dy = F.pad(dy, (0, 0, 0, 1))
        return torch.sqrt(dx.square() + dy.square() + 1.0e-6)

    def _metadata_token(self, item: ModalityInstance, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        family_id = FAMILIES.index(item.family)
        orbit_id = ORBITS.index(item.orbit) if item.orbit in ORBITS else 0
        native_log, native_known = _log_gsd(item.native_gsd_m)
        aligned_log, aligned_known = _log_gsd(item.aligned_gsd_m)
        continuous = torch.tensor(
            [native_log, native_known, aligned_log, aligned_known, float(item.quality)],
            dtype=dtype,
            device=device,
        )
        return (
            self.family_embedding.weight[family_id]
            + self.sensor_embedding.weight[_hash_bucket(item.sensor, self.sensor_embedding.num_embeddings)]
            + self.orbit_embedding.weight[orbit_id]
            + self.scale_quality(continuous)
        )

    def _encode_instance(self, item: ModalityInstance, device: torch.device) -> ModalityPyramid:
        image = item.image.to(device=device, dtype=torch.float32)
        band_names = list(item.band_names)
        if item.family in {"sar", "terrain", "deformation"}:
            image = torch.cat([image, self._gradient_bands(image)], dim=0)
            band_names.extend(f"gradient:{name}" for name in item.band_names)
        metadata_token = self._metadata_token(item, device, image.dtype)
        band_ids = torch.tensor(
            [_hash_bucket(name, self.band_embedding.num_embeddings) for name in band_names],
            dtype=torch.long,
            device=device,
        )
        band_tokens = self.band_embedding(band_ids) + metadata_token[None]
        high = self.family_high[item.family](self.band_encoder(image, band_tokens))
        mid = self.family_mid[item.family](high)
        low = self.family_low[item.family](mid)
        valid = item.valid_mask.to(device=device, dtype=high.dtype).unsqueeze(0)
        high_valid = F.interpolate(valid, size=high.shape[-2:], mode="nearest").squeeze(0)
        mid_valid = F.interpolate(valid, size=mid.shape[-2:], mode="nearest").squeeze(0)
        low_valid = F.interpolate(valid, size=low.shape[-2:], mode="nearest").squeeze(0)
        return ModalityPyramid(
            instance=item,
            high=high.squeeze(0),
            mid=mid.squeeze(0),
            low=low.squeeze(0),
            high_valid=high_valid,
            mid_valid=mid_valid,
            low_valid=low_valid,
            metadata_token=metadata_token,
        )

    def _active_mask(self, count: int, device: torch.device, apply_dropout: bool) -> torch.Tensor:
        active = torch.ones((count,), dtype=torch.bool, device=device)
        if self.training and apply_dropout and count > 1 and self.modality_dropout > 0.0:
            active = torch.rand((count,), device=device) >= self.modality_dropout
            if not active.any():
                active[torch.randint(0, count, (1,), device=device)] = True
        return active

    def forward(self, batch: ModalityBatch, apply_dropout: bool = True) -> MultiScaleFeatures:
        device = next(self.parameters()).device
        samples: list[list[ModalityPyramid]] = []
        for instances in batch.instances:
            active = self._active_mask(len(instances), device, apply_dropout)
            pyramids = [
                self._encode_instance(item, device)
                for index, item in enumerate(instances)
                if bool(active[index].item())
            ]
            samples.append(pyramids)
        return MultiScaleFeatures(samples=samples, reference_hw=batch.reference_hw)
