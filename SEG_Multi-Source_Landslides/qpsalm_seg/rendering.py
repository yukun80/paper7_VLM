#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Physically explicit benchmark-v2 sensor views for frozen Qwen vision."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import torch
import torch.nn.functional as F

from .schema import ModalityInstance


RENDERER_VERSION = "sensor_multiview_v5"


@dataclass
class RenderedView:
    name: str
    description: str
    image: torch.Tensor
    valid_mask: torch.Tensor
    source_modalities: tuple[str, ...]
    quality_flags: tuple[str, ...]
    content_hash: str
    render_transform: dict[str, int | float]


def _unit(channel: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
    values = channel.detach().float().nan_to_num()
    valid = (
        valid_mask.detach().bool() & torch.isfinite(channel.detach())
        if valid_mask is not None else torch.isfinite(channel.detach())
    )
    observed = values[valid]
    if not observed.numel():
        return torch.zeros_like(values)
    minimum, maximum = observed.min(), observed.max()
    if float(minimum) >= 0 and float(maximum) <= 1:
        return values.clamp(0, 1)
    return ((values - minimum) / (maximum - minimum).clamp_min(1e-6)).clamp(0, 1)


def _resize_square(tensor: torch.Tensor, size: int, mode: str) -> tuple[torch.Tensor, dict[str, int | float]]:
    h, w = tensor.shape[-2:]
    scale = min(size / max(h, 1), size / max(w, 1))
    rh, rw = max(1, round(h * scale)), max(1, round(w * scale))
    kwargs = {"size": (rh, rw), "mode": mode}
    if mode != "nearest":
        kwargs["align_corners"] = False
    resized = F.interpolate(tensor[None], **kwargs)[0]
    top, left = (size - rh) // 2, (size - rw) // 2
    return F.pad(resized, (left, size - rw - left, top, size - rh - top)), {
        "source_h": h, "source_w": w, "resized_h": rh, "resized_w": rw,
        "pad_top": top, "pad_left": left, "size": size, "scale": scale,
    }


def _hash(image, valid):
    rgb = (image.clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy().tobytes()
    mask = valid.to(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(rgb + mask).hexdigest()


def _view(name, description, image, item, size, flags=()):
    rendered, transform = _resize_square(image, size, "bilinear")
    valid, _ = _resize_square(item.valid_mask.float(), size, "nearest")
    valid = valid >= 0.5
    rendered = torch.where(valid.expand_as(rendered), rendered.clamp(0, 1), rendered.new_full((), 0.5))
    return RenderedView(name, description + " Neutral gray denotes nodata.", rendered, valid, (item.name,), tuple(flags), _hash(rendered, valid), transform)


def _index(item, aliases):
    names = [name.upper().replace("_", "") for name in item.band_names]
    wanted = {name.upper().replace("_", "") for name in aliases}
    return next((i for i, name in enumerate(names) if name in wanted), None)


def _rgb(item, aliases):
    indices = [_index(item, values) for values in aliases]
    if any(value is None for value in indices):
        return None
    valid = item.valid_mask[0] >= 0.5
    return torch.stack([_unit(item.image[int(value)], valid) for value in indices])


def _optical(item, size, strict):
    true = _rgb(item, (("R", "B04"), ("G", "B03"), ("B", "B02")))
    views = []
    if true is not None:
        views.append(_view(f"{item.name}_true_color", f"{item.sensor} true-color RGB", true, item, size))
    if item.product_type == "surface_reflectance":
        false = _rgb(item, (("B12",), ("B08", "B8"), ("B04", "R")))
        if false is not None:
            views.append(_view(f"{item.name}_false_color", "Sentinel-2 SWIR2/NIR/red false color", false, item, size))
        elif strict:
            raise ValueError(f"正式 S2 evidence 缺少 B12/B08/B04: {item.name}")
    if not views:
        if strict and item.family == "multispectral":
            raise ValueError(f"正式 multispectral evidence 不允许未知 band order: {item.name}")
        image = item.image[:3] if item.image.shape[0] >= 3 else item.image[:1].expand(3, -1, -1)
        valid = item.valid_mask[0] >= 0.5
        views.append(_view(
            f"{item.name}_fallback", "Uncertain optical band order",
            torch.stack([_unit(v, valid) for v in image]), item, size, ("uncertain_band_order",),
        ))
    return views


def _sar(item, size):
    vv_i, vh_i = _index(item, ("VV",)), _index(item, ("VH",))
    if vv_i is None or vh_i is None:
        raise ValueError(f"SAR view requires VV and VH: {item.name}")
    valid = item.valid_mask[0] >= 0.5
    vv, vh = _unit(item.image[vv_i], valid), _unit(item.image[vh_i], valid)
    difference = ((item.image[vv_i].float() - item.image[vh_i].float()).clamp(-1, 1) + 1.0) * 0.5
    description = (
        f"Sentinel-1 {item.orbit} SAR: VV, VH, and zero-centered normalized VV-VH difference; "
        f"source units={item.units}"
    )
    return _view(f"{item.name}_vv_vh_difference", description, torch.stack([vv, vh, difference]), item, size)


def _terrain(item, size):
    value = _unit(item.image[0], item.valid_mask[0] >= 0.5)
    if item.product_type == "elevation":
        dx = F.pad(value[:, 1:] - value[:, :-1], (0, 1, 0, 0))
        dy = F.pad(value[1:] - value[:-1], (0, 0, 0, 1))
        slope = _unit(torch.sqrt(dx.square() + dy.square() + 1e-6))
        hillshade = (0.65 - 0.45 * dx - 0.55 * dy).clamp(0, 1)
        image = torch.stack([value, hillshade, slope])
        description = f"DEM elevation, derived hillshade, and derived slope in {item.units}"
    else:
        image = value[None].expand(3, -1, -1)
        description = f"Terrain product {item.product_type} in {item.units}; no elevation derivatives applied"
    return _view(f"{item.name}_{item.product_type}", description, image, item, size)


def _deformation(item, size):
    signed = item.image[0].float().nan_to_num().clamp(-1, 1)
    positive, negative, neutral = signed.clamp_min(0), (-signed).clamp_min(0), 1 - signed.abs()
    image = torch.stack([neutral + positive, neutral, neutral + negative]).clamp(0, 1)
    normalization = item.metadata.get("normalization") or {}
    clip_abs = (normalization.get("parameters") or {}).get("clip_abs")
    fixed_range = (
        f"[-{clip_abs:g}, +{clip_abs:g}] {item.units}"
        if isinstance(clip_abs, (int, float)) and clip_abs > 0
        else f"a dataset-fixed symmetric range in {item.units}"
    )
    sign = (item.band_metadata[0].get("sign_convention") if item.band_metadata else None) or "source_defined"
    description = (
        f"Signed LOS deformation rendered over {fixed_range}: red positive, blue negative, "
        f"white near zero; sign convention={sign}"
    )
    return _view(f"{item.name}_signed_fixed_scale", description, image, item, size)


def render_sensor_views(instances: list[ModalityInstance], size: int = 224, strict: bool = True) -> list[RenderedView]:
    views = []
    for item in instances:
        if item.family in {"optical", "multispectral"}:
            views.extend(_optical(item, size, strict))
        elif item.family == "sar":
            views.append(_sar(item, size))
        elif item.family == "terrain":
            views.append(_terrain(item, size))
        elif item.family == "deformation":
            views.append(_deformation(item, size))
    return views
