#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sensor-aware multi-view rendering for frozen Qwen visual evidence."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import torch
import torch.nn.functional as F

from .schema import ModalityInstance


RENDERER_VERSION = "sensor_multiview_v2"


@dataclass
class RenderedView:
    name: str
    description: str
    image: torch.Tensor
    source_modalities: tuple[str, ...]
    quality_flags: tuple[str, ...]
    content_hash: str


def _unit(channel: torch.Tensor) -> torch.Tensor:
    values = channel.detach().float()
    finite = values[torch.isfinite(values)]
    if finite.numel() == 0:
        return torch.zeros_like(values)
    if float(finite.min()) >= 0.0 and float(finite.max()) <= 1.0:
        return values.nan_to_num().clamp(0.0, 1.0)
    low = torch.quantile(finite, 0.02)
    high = torch.quantile(finite, 0.98)
    return ((values.nan_to_num() - low) / (high - low).clamp_min(1.0e-6)).clamp(0.0, 1.0)


def _resize_square(image: torch.Tensor, size: int) -> torch.Tensor:
    height, width = image.shape[-2:]
    scale = min(float(size) / max(height, 1), float(size) / max(width, 1))
    resized_h = max(1, int(round(height * scale)))
    resized_w = max(1, int(round(width * scale)))
    resized = F.interpolate(image[None], size=(resized_h, resized_w), mode="bilinear", align_corners=False)[0]
    top = (size - resized_h) // 2
    left = (size - resized_w) // 2
    return F.pad(resized, (left, size - resized_w - left, top, size - resized_h - top))


def _hash_image(image: torch.Tensor) -> str:
    array = (image.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).cpu().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def _view(
    name: str,
    description: str,
    image: torch.Tensor,
    sources: tuple[str, ...],
    size: int,
    flags: tuple[str, ...] = (),
) -> RenderedView:
    rendered = _resize_square(image, size).clamp(0.0, 1.0)
    return RenderedView(name, description, rendered, sources, flags, _hash_image(rendered))


def _band_index(item: ModalityInstance, aliases: tuple[str, ...]) -> int | None:
    names = [name.upper().replace("_", "") for name in item.band_names]
    wanted = {name.upper().replace("_", "") for name in aliases}
    return next((index for index, name in enumerate(names) if name in wanted), None)


def _rgb(item: ModalityInstance, aliases: tuple[tuple[str, ...], ...]) -> torch.Tensor | None:
    indices = [_band_index(item, choices) for choices in aliases]
    if any(index is None for index in indices):
        return None
    return torch.stack([_unit(item.image[int(index)]) for index in indices], dim=0)


def _optical_views(item: ModalityInstance, size: int) -> list[RenderedView]:
    sources = (item.name,)
    true_rgb = _rgb(item, (("R", "RED", "B04", "B4"), ("G", "GREEN", "B03", "B3"), ("B", "BLUE", "B02", "B2")))
    views: list[RenderedView] = []
    if true_rgb is not None:
        views.append(_view(f"{item.name}_true_color", f"{item.sensor} optical true-color RGB", true_rgb, sources, size))
    if item.family == "multispectral":
        false_rgb = _rgb(item, (("B12", "SWIR2"), ("B08", "B8", "NIR"), ("B04", "B4", "R")))
        if false_rgb is None:
            false_rgb = _rgb(item, (("B11", "SWIR1"), ("B08", "B8", "NIR"), ("B04", "B4", "R")))
        if false_rgb is not None:
            views.append(_view(f"{item.name}_false_color", "Sentinel-2 SWIR/NIR/red false-color view", false_rgb, sources, size))
    if not views:
        image = item.image
        if image.shape[0] == 1:
            fallback = _unit(image[0])[None].expand(3, -1, -1)
        elif image.shape[0] == 2:
            fallback = torch.stack([_unit(image[0]), _unit(image[1]), _unit(image.mean(dim=0))])
        else:
            fallback = torch.stack([_unit(image[index]) for index in range(3)])
        views.append(
            _view(
                f"{item.name}_fallback",
                f"{item.sensor} multiband fallback; physical RGB order is uncertain",
                fallback,
                sources,
                size,
                ("unknown_band_order",),
            )
        )
    return views


def _sar_view(item: ModalityInstance, size: int) -> RenderedView:
    vv_index = _band_index(item, ("VV",)) or 0
    vh_index = _band_index(item, ("VH",))
    vh_index = vh_index if vh_index is not None else min(1, item.image.shape[0] - 1)
    vv = _unit(item.image[vv_index])
    vh = _unit(item.image[vh_index])
    ratio = _unit(vv - vh)
    image = torch.stack([vv, vh, ratio], dim=0)
    return _view(
        f"{item.name}_vv_vh_ratio",
        f"Sentinel-1 {item.orbit} SAR: VV, VH and normalized VV-VH ratio",
        image,
        (item.name,),
        size,
    )


def _terrain_view(item: ModalityInstance, size: int) -> RenderedView:
    elevation = _unit(item.image[0])
    dx = F.pad(elevation[:, 1:] - elevation[:, :-1], (0, 1, 0, 0))
    dy = F.pad(elevation[1:, :] - elevation[:-1, :], (0, 0, 0, 1))
    slope = _unit(torch.sqrt(dx.square() + dy.square() + 1.0e-6))
    hillshade = (0.65 - 0.45 * dx - 0.55 * dy).clamp(0.0, 1.0)
    return _view(
        f"{item.name}_terrain",
        f"Terrain evidence from {item.name}: normalized value, hillshade and local slope",
        torch.stack([elevation, hillshade, slope], dim=0),
        (item.name,),
        size,
    )


def _deformation_view(item: ModalityInstance, size: int) -> RenderedView:
    values = item.image[0].detach().float().nan_to_num()
    scale = torch.quantile(values.abs().flatten(), 0.98).clamp_min(1.0e-6)
    signed = (values / scale).clamp(-1.0, 1.0)
    positive = signed.clamp_min(0.0)
    negative = (-signed).clamp_min(0.0)
    neutral = 1.0 - signed.abs()
    image = torch.stack([neutral + positive, neutral, neutral + negative], dim=0).clamp(0.0, 1.0)
    return _view(
        f"{item.name}_signed",
        "Signed InSAR deformation: red positive, blue negative, white near zero",
        image,
        (item.name,),
        size,
    )


def render_sensor_views(instances: list[ModalityInstance], size: int = 224) -> list[RenderedView]:
    """Render every available modality into physically interpretable Qwen views."""
    views: list[RenderedView] = []
    for item in instances:
        if item.family in {"optical", "multispectral"}:
            views.extend(_optical_views(item, size))
        elif item.family == "sar":
            views.append(_sar_view(item, size))
        elif item.family == "terrain":
            views.append(_terrain_view(item, size))
        elif item.family == "deformation":
            views.append(_deformation_view(item, size))
    return views
