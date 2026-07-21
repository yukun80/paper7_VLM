"""Deterministic, metadata-driven P2 rendering outside model forward."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image
from torch import Tensor

from sami_gsd.contracts.canonical import ModalityRecord


RENDERER_REVISION = "sami_p2_renderer_v1_declared_channels_valid_minmax"


def array_to_hwc(array: np.ndarray, *, band_count: int, modality_id: str) -> np.ndarray:
    """Return an explicit HWC array using the declared band count.

    Raises instead of guessing when both the first and last dimensions could be
    the channel axis.
    """

    if array.ndim == 2:
        if band_count != 1:
            raise ValueError(f"{modality_id} is 2D but declares {band_count} bands")
        return array[:, :, None]
    if array.ndim != 3:
        raise ValueError(f"{modality_id} asset must be 2D or 3D")
    first_matches = array.shape[0] == band_count
    last_matches = array.shape[-1] == band_count
    if first_matches == last_matches:
        raise ValueError(f"{modality_id} channel axis cannot be uniquely resolved from declared bands")
    return np.moveaxis(array, 0, -1) if first_matches else array


def render_modality_rgb(
    array_hwc: np.ndarray,
    valid_mask: np.ndarray,
    modality: ModalityRecord,
) -> tuple[Image.Image, Tensor]:
    """Render one declared modality to RGB while excluding invalid pixels.

    The operation is deterministic and uses only declared channel names plus
    per-channel valid min/max display scaling. It never changes the canonical
    asset and never exposes normalization metadata to the language prompt.
    """

    if array_hwc.ndim != 3 or array_hwc.shape[:2] != valid_mask.shape:
        raise ValueError(f"{modality.modality_id} array and valid mask grids must match")
    band_indices = {name: index for index, name in enumerate(modality.band_names)}
    declared_channels = modality.render_policy.channels or modality.band_names
    try:
        selected_indices = tuple(band_indices[name] for name in declared_channels)
    except KeyError as error:
        raise ValueError(
            f"{modality.modality_id} render channel {error.args[0]!r} is not a declared band"
        ) from error
    if not selected_indices:
        raise ValueError(f"{modality.modality_id} has no declared render channel")
    selected = np.asarray(array_hwc[:, :, selected_indices], dtype=np.float32)
    if selected.ndim == 2:
        selected = selected[:, :, None]
    finite = np.isfinite(selected).all(axis=2)
    effective_valid = np.asarray(valid_mask, dtype=bool) & finite
    if not bool(effective_valid.any()):
        raise ValueError(f"{modality.modality_id} has no finite valid render pixels")

    channels: list[np.ndarray] = []
    for channel_index in range(selected.shape[2]):
        channel = selected[:, :, channel_index]
        values = channel[effective_valid]
        if modality.render_policy.clip_percentiles is None:
            lower = float(values.min())
            upper = float(values.max())
        else:
            lower, upper = (
                float(value)
                for value in np.percentile(values, modality.render_policy.clip_percentiles)
            )
        if not math.isfinite(lower) or not math.isfinite(upper):
            raise ValueError(f"{modality.modality_id} display range is non-finite")
        if upper <= lower:
            scaled = np.zeros_like(channel, dtype=np.float32)
        else:
            scaled = np.clip((channel - lower) / (upper - lower), 0.0, 1.0)
        scaled[~effective_valid] = 0.0
        channels.append(scaled)

    if len(channels) == 1:
        rgb = np.stack((channels[0], channels[0], channels[0]), axis=2)
    elif len(channels) == 2:
        rgb = np.stack((channels[0], channels[1], (channels[0] + channels[1]) * 0.5), axis=2)
    else:
        rgb = np.stack(channels[:3], axis=2)
    image = Image.fromarray(np.rint(rgb * 255.0).astype(np.uint8))
    return image, torch.from_numpy(effective_valid.copy()).to(dtype=torch.bool)


def image_content_sha256(image: Image.Image) -> str:
    """Hash exact RGB bytes together with dimensions and renderer revision."""

    if image.mode != "RGB":
        raise ValueError("only RGB model inputs can be hashed")
    header = f"{RENDERER_REVISION}|RGB|{image.height}|{image.width}|".encode("utf-8")
    return hashlib.sha256(header + image.tobytes()).hexdigest()


def valid_mask_sha256(valid_mask: Tensor) -> str:
    """Hash one CPU bool valid mask with its exact grid."""

    if valid_mask.ndim != 2 or valid_mask.dtype is not torch.bool or valid_mask.device.type != "cpu":
        raise ValueError("valid mask hashing requires a CPU bool [H,W] tensor")
    height, width = valid_mask.shape
    header = f"bool|{height}|{width}|".encode("utf-8")
    payload = valid_mask.contiguous().to(dtype=torch.uint8).numpy().tobytes()
    return hashlib.sha256(header + payload).hexdigest()


def resize_to_pixel_budget(image: Image.Image, valid_mask: Tensor, max_pixels: int) -> tuple[Image.Image, Tensor]:
    """Downscale, never upscale, to an area budget using bilinear/nearest."""

    if max_pixels <= 0:
        raise ValueError("max_pixels must be positive")
    width, height = image.size
    if tuple(valid_mask.shape) != (height, width):
        raise ValueError("image and valid mask must share a grid before budgeting")
    area = height * width
    if area <= max_pixels:
        return image.copy(), valid_mask.clone()
    scale = math.sqrt(max_pixels / area)
    output_h = max(1, int(math.floor(height * scale)))
    output_w = max(1, int(math.floor(width * scale)))
    while output_h * output_w > max_pixels:
        if output_h >= output_w and output_h > 1:
            output_h -= 1
        elif output_w > 1:
            output_w -= 1
        else:
            break
    resized_image = image.resize((output_w, output_h), resample=Image.Resampling.BILINEAR)
    resized_valid = functional.interpolate(
        valid_mask[None, None].to(dtype=torch.float32),
        size=(output_h, output_w),
        mode="nearest",
    )[0, 0].to(dtype=torch.bool)
    if not bool(resized_valid.any()):
        raise ValueError("pixel budgeting removed every valid pixel")
    return resized_image, resized_valid


__all__: Sequence[str] = (
    "RENDERER_REVISION",
    "array_to_hwc",
    "image_content_sha256",
    "render_modality_rgb",
    "resize_to_pixel_budget",
    "valid_mask_sha256",
)
