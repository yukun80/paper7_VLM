#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Controlled single-vector baselines for MGRR ablations."""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn
import torch.nn.functional as F

from qpsalm_seg.schema import MultisourceBackboneState, RegionEvidenceState

from .backbone import region_mask_for_modality_view
from .mgrr import (
    RegionProtocol,
    _bbox_pool,
    _box_coordinate_values,
    _geometry_values,
    _masked_pool,
)


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
        protocol: RegionProtocol = "vision_only",
    ) -> RegionEvidenceState:
        if protocol not in {"assisted", "vision_only"}:
            raise ValueError(f"未知 region baseline protocol={protocol!r}")
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
        if valid.shape != (batch_size, 1, height, width):
            raise ValueError(
                f"region_valid_mask shape 非法: {tuple(valid.shape)}"
            )
        rows = []
        geometry_rows = []
        geometry_input_rows = []
        max_modalities = max((len(sample) for sample in backbone.features.samples), default=1)
        modality_rows = []
        transform_rows = []
        visual_activity_rows = []
        region_present_rows = []
        for batch_index, pyramids in enumerate(backbone.features.samples):
            sample = []
            sample_geometries = []
            sample_geometry_inputs = []
            sample_modalities = []
            sample_transforms = []
            sample_visual_activity = []
            sample_region_present = []
            for region_index in range(region_count):
                mask = ((masks[batch_index, region_index:region_index + 1] > 0.5) & (valid[batch_index] > 0.5)).float()
                is_present = bool(mask.any())
                if is_present and protocol == "assisted":
                    geometry_input = _geometry_values(mask, valid[batch_index])
                    geometry = self.geometry_project(geometry_input)
                elif self.mode == "full_image_box":
                    geometry_input = (
                        _box_coordinate_values(mask, valid[batch_index])
                        if is_present else mask.new_zeros(10)
                    )
                    # 零坐标经过同一投影形成显式 no-box 状态；不能把 null
                    # 样本退化为与 crop-only 相同、完全无全图视觉的 token。
                    geometry = self.geometry_project(geometry_input)
                else:
                    geometry_input = mask.new_zeros(10)
                    geometry = mask.new_zeros(self.dim)
                tokens: list[torch.Tensor | None] = []
                transform_flags = []
                visual_activity = []
                for pyramid in pyramids:
                    view_mask, transform_applied = region_mask_for_modality_view(
                        backbone, batch_index, pyramid, mask
                    )
                    feature = F.interpolate(
                        pyramid.detail[None], size=(height, width), mode="bilinear", align_corners=False
                    )[0]
                    modality_valid = F.interpolate(
                        pyramid.detail_valid.float()[None], size=(height, width), mode="nearest"
                    )[0]
                    effective = (
                        modality_valid
                        if self.mode == "full_image_box"
                        else view_mask * modality_valid
                    )
                    transform_flags.append(float(transform_applied))
                    if not bool(effective.any()):
                        tokens.append(None)
                        visual_activity.append(False)
                        continue
                    if self.mode == "crop_only":
                        pooled = _bbox_pool(feature, effective, modality_valid)
                    elif self.mode == "masked_pooling":
                        pooled = _masked_pool(feature, effective)
                    else:
                        pooled = _masked_pool(
                            feature,
                            modality_valid,
                        )
                    tokens.append(self.project(pooled))
                    visual_activity.append(True)
                active_tokens = [token for token in tokens if token is not None]
                if self.mode == "full_image_box":
                    region_token = (
                        torch.stack(active_tokens).mean(0)
                        if active_tokens else self.null_region
                    ) + geometry
                    zero = region_token.new_zeros(self.dim)
                    padded = torch.stack([
                        tokens[index]
                        if index < len(tokens) and tokens[index] is not None else zero
                        for index in range(max_modalities)
                    ])
                elif is_present:
                    # 无有效视觉覆盖时保留显式 null region；Assisted 或 box
                    # baseline 的合法几何仍可独立进入单向量对照。
                    region_token = (
                        torch.stack(active_tokens).mean(0)
                        if active_tokens else self.null_region
                    )
                    region_token = region_token + geometry
                    zero = region_token.new_zeros(self.dim)
                    padded = torch.stack([
                        tokens[index]
                        if index < len(tokens) and tokens[index] is not None else zero
                        for index in range(max_modalities)
                    ])
                else:
                    region_token = self.null_region
                    padded = self.null_region.new_zeros((max_modalities, self.dim))
                padded_transforms = (
                    F.pad(
                        masks.new_tensor(transform_flags),
                        (0, max_modalities - len(transform_flags)),
                    )
                    if transform_flags else masks.new_zeros(max_modalities)
                )
                sample.append(region_token)
                sample_geometries.append(geometry)
                sample_geometry_inputs.append(geometry_input)
                sample_modalities.append(padded)
                sample_transforms.append(padded_transforms)
                sample_visual_activity.append(F.pad(
                    torch.tensor(
                        visual_activity, dtype=torch.bool, device=device
                    ),
                    (0, max_modalities - len(visual_activity)),
                ))
                sample_region_present.append(is_present)
            rows.append(torch.stack(sample))
            geometry_rows.append(torch.stack(sample_geometries))
            geometry_input_rows.append(torch.stack(sample_geometry_inputs))
            modality_rows.append(torch.stack(sample_modalities))
            transform_rows.append(torch.stack(sample_transforms))
            visual_activity_rows.append(torch.stack(sample_visual_activity))
            region_present_rows.append(torch.tensor(
                sample_region_present, dtype=torch.bool, device=device
            ))
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
            geometry_tokens=torch.stack(geometry_rows),
            modality_tokens=torch.stack(modality_rows),
            diagnostics={
                "baseline_crop_only": masks.new_tensor(float(self.mode == "crop_only")),
                "baseline_masked_pooling": masks.new_tensor(float(self.mode == "masked_pooling")),
                "baseline_full_image_box": masks.new_tensor(float(self.mode == "full_image_box")),
                "protocol_assisted": masks.new_tensor(float(protocol == "assisted")),
                "geometry_input_values": torch.stack(geometry_input_rows),
                "view_transform_retargeted": torch.stack(transform_rows) > 0,
                "visual_evidence_active": torch.stack(visual_activity_rows),
                "region_present": torch.stack(region_present_rows),
                "full_image_visual_for_null_region": masks.new_tensor(
                    float(self.mode == "full_image_box")
                ),
            },
        )
