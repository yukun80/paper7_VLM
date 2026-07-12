#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Spatial transforms shared by modalities, masks and cached feature maps."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch
import torch.nn.functional as F

from qpsalm_seg.schema import ModalityInstance


def resize_pad_tensor(tensor: torch.Tensor, target_size: int, mode: str) -> tuple[torch.Tensor, dict[str, Any]]:
    src_h, src_w = int(tensor.shape[-2]), int(tensor.shape[-1])
    scale = min(float(target_size) / max(src_h, 1), float(target_size) / max(src_w, 1))
    new_h = max(1, min(target_size, int(round(src_h * scale))))
    new_w = max(1, min(target_size, int(round(src_w * scale))))
    kwargs = {"size": (new_h, new_w), "mode": mode}
    if mode != "nearest":
        kwargs["align_corners"] = False
    resized = F.interpolate(tensor[None], **kwargs)[0]
    top = (target_size - new_h) // 2
    left = (target_size - new_w) // 2
    out = F.pad(resized, (left, target_size - new_w - left, top, target_size - new_h - top))
    return out, {
        "source_hw": [src_h, src_w], "target_hw": [target_size, target_size],
        "resized_hw": [new_h, new_w], "scale": scale,
        "pad_top": top, "pad_bottom": target_size - new_h - top,
        "pad_left": left, "pad_right": target_size - new_w - left,
    }


def valid_mask_from_transform(transform: dict[str, Any]) -> torch.Tensor:
    target_h, target_w = map(int, transform["target_hw"])
    resized_h, resized_w = map(int, transform["resized_hw"])
    top, left = int(transform["pad_top"]), int(transform["pad_left"])
    valid = torch.zeros((1, target_h, target_w), dtype=torch.float32)
    valid[:, top:top + resized_h, left:left + resized_w] = 1.0
    return valid


def swap_padding_after_flip(
    tensor: torch.Tensor,
    transform: dict[str, Any],
    *,
    hflip: bool,
    vflip: bool,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Move an already flipped native image into the canvas padding that a canvas flip produces."""
    if not hflip and not vflip:
        return tensor, transform
    updated = dict(transform)
    top, bottom = int(transform["pad_top"]), int(transform["pad_bottom"])
    left, right = int(transform["pad_left"]), int(transform["pad_right"])
    resized_h, resized_w = map(int, transform["resized_hw"])
    content = tensor[..., top:top + resized_h, left:left + resized_w]
    new_top, new_bottom = (bottom, top) if vflip else (top, bottom)
    new_left, new_right = (right, left) if hflip else (left, right)
    updated.update({
        "pad_top": new_top, "pad_bottom": new_bottom,
        "pad_left": new_left, "pad_right": new_right,
    })
    return F.pad(content, (new_left, new_right, new_top, new_bottom)), updated


def downscale_native(tensor: torch.Tensor, max_side: int, mode: str = "bilinear") -> torch.Tensor:
    h, w = tensor.shape[-2:]
    if max(h, w) <= max_side:
        return tensor
    scale = float(max_side) / max(h, w)
    target = (max(1, round(h * scale)), max(1, round(w * scale)))
    kwargs = {"size": target, "mode": mode}
    if mode != "nearest":
        kwargs["align_corners"] = False
    return F.interpolate(tensor[None], **kwargs)[0]


def apply_flips(
    instances: list[ModalityInstance],
    mask: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    hflip: bool,
    vflip: bool,
) -> tuple[list[ModalityInstance], torch.Tensor, torch.Tensor]:
    dims = ([-2] if vflip else []) + ([-1] if hflip else [])
    if not dims:
        return instances, mask, valid_mask
    transformed = [
        replace(item, image=torch.flip(item.image, dims), valid_mask=torch.flip(item.valid_mask, dims))
        for item in instances
    ]
    return transformed, torch.flip(mask, dims), torch.flip(valid_mask, dims)
