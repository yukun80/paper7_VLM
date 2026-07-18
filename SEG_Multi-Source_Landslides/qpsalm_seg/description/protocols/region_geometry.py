"""Renderer-bound region geometry shared by data, modeling and evaluation."""

from __future__ import annotations

import hashlib

import numpy as np
import torch
import torch.nn.functional as F

from qpsalm_seg.schema import ModalityPyramid, MultisourceBackboneState


REGION_SOURCE_CANVAS_MAPPING_PROTOCOL = (
    "qpsalm_region_source_canvas_mapping_v1_nearest_full_extent"
)


def bridge_region_mask_digest(mask: np.ndarray) -> str:
    """Replay the semantic digest stored by Landslide Bridge v7.

    Bridge ``region_mask.sha256`` binds decoded binary pixels rather than the
    serialization bytes of the ``.npy`` container.  Predicted-region indexes
    deliberately use file SHA-256 and must not call this helper.
    """
    array = np.ascontiguousarray(np.asarray(mask).astype(np.uint8, copy=False))
    return hashlib.sha256(array.tobytes()).hexdigest()


def render_transform_spec(transform: dict, *, label: str) -> tuple[int, ...]:
    values = tuple(int(transform.get(key) or 0) for key in (
        "source_h", "source_w", "resized_h", "resized_w",
        "pad_top", "pad_left", "size",
    ))
    source_h, source_w, resized_h, resized_w, top, left, size = values
    if min(source_h, source_w, resized_h, resized_w, size) <= 0:
        raise ValueError(f"{label} render transform 缺少正尺寸: {transform}")
    if min(top, left) < 0 or top + resized_h > size or left + resized_w > size:
        raise ValueError(f"{label} render transform padding 非法: {transform}")
    return values


def map_native_region_mask_to_render_source(
    mask: torch.Tensor,
    transform: dict,
) -> tuple[torch.Tensor, dict]:
    """Map a benchmark-native mask onto the renderer's preprocessed source.

    Segmentation Vision Cache v3 is rendered from size-bucketed modality
    tensors, while Bridge masks stay on the benchmark-native canvas. Both
    cover the complete parent extent, so the mapping is deterministic nearest
    full-extent resampling.
    """
    value = mask.float()
    if value.ndim == 2:
        value = value[None]
    if value.ndim != 3 or value.shape[0] != 1:
        raise ValueError(
            f"native region mask 必须为 [1,H,W]，当前 {tuple(value.shape)}"
        )
    source_h, source_w = int(value.shape[-2]), int(value.shape[-1])
    render_h, render_w = render_transform_spec(
        transform, label="native-to-render-source"
    )[:2]
    resampled = (source_h, source_w) != (render_h, render_w)
    mapped = (
        F.interpolate(value[None], size=(render_h, render_w), mode="nearest")[0]
        if resampled else value
    )
    return mapped, {
        "protocol": REGION_SOURCE_CANVAS_MAPPING_PROTOCOL,
        "source_hw": [source_h, source_w],
        "render_source_hw": [render_h, render_w],
        "interpolation": "nearest",
        "extent_policy": "full_extent",
        "resampled": resampled,
    }


def project_native_region_mask_to_cache(
    mask: torch.Tensor,
    transform: dict,
) -> tuple[torch.Tensor, dict]:
    """Replay native-to-render mapping followed by the bound cache transform."""
    mapped, mapping = map_native_region_mask_to_render_source(mask, transform)
    return transform_region_mask_to_cache(mapped, transform), mapping


def transform_region_mask_to_cache(mask: torch.Tensor, transform: dict) -> torch.Tensor:
    """Apply the exact renderer resize/pad transform to one source-space mask."""
    value = mask.float()
    if value.ndim == 2:
        value = value[None]
    if value.ndim != 3 or value.shape[0] != 1:
        raise ValueError(f"region mask 必须为 [1,H,W]，当前 {tuple(value.shape)}")
    source_h, source_w, resized_h, resized_w, top, left, size = (
        render_transform_spec(transform, label="target")
    )
    if (source_h, source_w) != tuple(value.shape[-2:]):
        raise ValueError(
            "region source shape 与 cache render transform 不一致: "
            f"mask={tuple(value.shape[-2:])} transform={(source_h, source_w)}"
        )
    resized = F.interpolate(
        value[None], size=(resized_h, resized_w), mode="nearest"
    )[0]
    right = size - resized_w - left
    bottom = size - resized_h - top
    if min(top, left, right, bottom) < 0:
        raise ValueError(f"cache render padding 非法: {transform}")
    return F.pad(resized, (left, right, top, bottom))


def restore_region_mask_from_cache(
    mask: torch.Tensor, transform: dict,
) -> torch.Tensor:
    """Undo renderer padding/resize for faithful source-image overlays."""
    value = mask.float()
    if value.ndim == 2:
        value = value[None]
    if value.ndim != 3 or value.shape[0] != 1:
        raise ValueError(
            f"cache region mask 必须为 [1,H,W]，当前 {tuple(value.shape)}"
        )
    source_h, source_w, resized_h, resized_w, top, left, size = (
        render_transform_spec(transform, label="source restore")
    )
    if tuple(value.shape[-2:]) != (size, size):
        raise ValueError(
            "cache region canvas 与 render transform 不一致: "
            f"mask={tuple(value.shape[-2:])} expected={(size, size)}"
        )
    crop = value[:, top:top + resized_h, left:left + resized_w]
    if tuple(crop.shape[-2:]) != (resized_h, resized_w):
        raise ValueError(f"cache region restore crop 非法: {transform}")
    return F.interpolate(crop[None], size=(source_h, source_w), mode="nearest")[0]


def retarget_region_mask_between_cache_views(
    mask: torch.Tensor,
    source_transform: dict,
    target_transform: dict,
) -> torch.Tensor:
    """Move one region from the reference cache view to another view canvas."""
    value = mask.float()
    if value.ndim == 2:
        value = value[None]
    source = render_transform_spec(source_transform, label="source")
    target = render_transform_spec(target_transform, label="target")
    source_h, source_w, resized_h, resized_w, top, left, size = source
    if target[-1] != size:
        raise ValueError(
            "Description cache views 必须共享 render canvas size: "
            f"source={size} target={target[-1]}"
        )
    if tuple(value.shape) != (1, size, size):
        raise ValueError(
            "reference cache region shape 与 source transform 不一致: "
            f"mask={tuple(value.shape)} expected={(1, size, size)}"
        )
    source_crop = value[:, top:top + resized_h, left:left + resized_w]
    source_raster = F.interpolate(
        source_crop[None], size=(source_h, source_w), mode="nearest"
    )[0]
    target_h, target_w = target[:2]
    if (target_h, target_w) != (source_h, source_w):
        source_raster = F.interpolate(
            source_raster[None], size=(target_h, target_w), mode="nearest"
        )[0]
    return (
        transform_region_mask_to_cache(source_raster, target_transform) > 0.5
    ).to(value.dtype)


def region_mask_for_modality_view(
    backbone: MultisourceBackboneState,
    batch_index: int,
    pyramid: ModalityPyramid,
    reference_mask: torch.Tensor,
) -> tuple[torch.Tensor, bool]:
    """Resolve a reference-cache mask for one native-size modality/view."""
    sample_metadata = backbone.metadata[batch_index]
    render_transforms = sample_metadata.get("render_transforms")
    target_transform = pyramid.instance.metadata.get("render_transform")
    if not render_transforms and not target_transform:
        return reference_mask, False
    if (
        not isinstance(render_transforms, (list, tuple))
        or not render_transforms
        or not isinstance(render_transforms[0], dict)
        or not isinstance(target_transform, dict)
        or not target_transform
    ):
        raise ValueError(
            "Description cache backbone 缺少 reference/view render transform"
        )
    source_transform = render_transforms[0]
    source_spec = render_transform_spec(source_transform, label="reference")
    target_spec = render_transform_spec(target_transform, label="view")
    if source_spec == target_spec:
        return (reference_mask > 0.5).to(reference_mask.dtype), False
    return (
        retarget_region_mask_between_cache_views(
            reference_mask, source_transform, target_transform
        ),
        True,
    )
