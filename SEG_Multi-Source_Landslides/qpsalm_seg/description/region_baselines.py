#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Controlled single-vector baselines for MGRR ablations."""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn
import torch.nn.functional as F

from qpsalm_seg.schema import MultisourceBackboneState, RegionEvidenceState

from .mgrr import _bbox_pool, _geometry_values, _masked_pool


BaselineMode = Literal["crop_only", "masked_pooling", "full_image_box"]


class SingleVectorRegionPooling(nn.Module):
    """Reduce each region to one vector without component/context replay."""

    def __init__(self, dim: int, mode: BaselineMode) -> None:
        super().__init__()
        if mode not in {"crop_only", "masked_pooling", "full_image_box"}:
            raise ValueError(f"未知 region pooling baseline={mode!r}")
        self.dim = int(dim)
        self.mode = mode
        self.project = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.LayerNorm(dim))
        self.geometry_project = nn.Sequential(
            nn.Linear(10, dim), nn.GELU(), nn.LayerNorm(dim)
        )
        self.null_region = nn.Parameter(torch.randn(dim) * 0.02)

    def forward(
        self,
        backbone: MultisourceBackboneState,
        region_masks: torch.Tensor,
        *,
        region_valid_mask: torch.Tensor | None = None,
    ) -> RegionEvidenceState:
        if region_masks.ndim == 3:
            region_masks = region_masks[:, None]
        if region_masks.ndim != 4:
            raise ValueError("region_masks 必须是 [B,R,H,W]")
        batch_size, region_count, height, width = region_masks.shape
        if (height, width) != backbone.reference_hw or batch_size != len(backbone.features.samples):
            raise ValueError("region mask 与 backbone state 不一致")
        device = backbone.valid_mask.device
        masks = region_masks.to(device=device, dtype=torch.float32)
        valid = (
            region_valid_mask.to(device=device, dtype=torch.float32)
            if region_valid_mask is not None else backbone.valid_mask.to(device=device, dtype=torch.float32)
        )
        if valid.ndim == 3:
            valid = valid[:, None]
        rows = []
        max_modalities = max((len(sample) for sample in backbone.features.samples), default=1)
        modality_rows = []
        for batch_index, pyramids in enumerate(backbone.features.samples):
            sample = []
            sample_modalities = []
            for region_index in range(region_count):
                mask = ((masks[batch_index, region_index:region_index + 1] > 0.5) & (valid[batch_index] > 0.5)).float()
                tokens = []
                for pyramid in pyramids:
                    feature = F.interpolate(
                        pyramid.detail[None], size=(height, width), mode="bilinear", align_corners=False
                    )[0]
                    modality_valid = F.interpolate(
                        pyramid.detail_valid.float()[None], size=(height, width), mode="nearest"
                    )[0]
                    effective = mask * modality_valid
                    if self.mode == "crop_only":
                        pooled = _bbox_pool(feature, effective)
                    elif self.mode == "masked_pooling":
                        pooled = _masked_pool(feature, effective)
                    else:
                        pooled = _masked_pool(
                            feature,
                            modality_valid * valid[batch_index],
                        )
                    tokens.append(self.project(pooled))
                if tokens and bool(mask.any()):
                    stacked = torch.stack(tokens)
                    region_token = stacked.mean(0)
                    if self.mode == "full_image_box":
                        region_token = region_token + self.geometry_project(
                            _geometry_values(mask, valid[batch_index])
                        )
                    padded = F.pad(stacked, (0, 0, 0, max_modalities - stacked.shape[0]))
                else:
                    region_token = self.null_region
                    padded = self.null_region.new_zeros((max_modalities, self.dim))
                sample.append(region_token)
                sample_modalities.append(padded)
            rows.append(torch.stack(sample))
            modality_rows.append(torch.stack(sample_modalities))
        region_tokens = torch.stack(rows)
        sequence_mask = torch.ones(
            region_tokens.shape[:2] + (1,), dtype=torch.bool, device=region_tokens.device
        )
        return RegionEvidenceState(
            backbone=backbone,
            region_masks=masks,
            region_valid_mask=valid,
            region_tokens=region_tokens,
            region_sequence_tokens=region_tokens[:, :, None],
            region_sequence_mask=sequence_mask,
            context_tokens=torch.zeros_like(region_tokens),
            geometry_tokens=torch.zeros_like(region_tokens),
            modality_tokens=torch.stack(modality_rows),
            diagnostics={
                "baseline_crop_only": masks.new_tensor(float(self.mode == "crop_only")),
                "baseline_masked_pooling": masks.new_tensor(float(self.mode == "masked_pooling")),
                "baseline_full_image_box": masks.new_tensor(float(self.mode == "full_image_box")),
            },
        )
