"""Dependency-free spatial reference primitives for Canonical Benchmark v3.

This module is the CPU reference implementation for P1.2.  Image rasters use
explicit ``(H, W, C)`` nested sequences of finite numbers and are resized with
bilinear half-pixel-center sampling.  Mask and valid rasters use explicit
``(H, W)`` binary sequences and are resized only with nearest sampling.  The
module performs no file I/O and makes no dtype, device or layout guesses.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias

from sami_gsd.contracts.canonical import (
    HalfOpenBox,
    TransformStep,
    validate_half_open_box,
    validate_transform_sequence,
)


INTERPOLATION_POLICY = "image_bilinear_mask_valid_nearest"
ContinuousPoint: TypeAlias = tuple[float, float]
ContinuousBox: TypeAlias = tuple[float, float, float, float]
QwenBox1000: TypeAlias = tuple[int, int, int, int]
BinaryRaster: TypeAlias = tuple[tuple[int, ...], ...]
ImagePixel: TypeAlias = tuple[float, ...]
ImageRaster: TypeAlias = tuple[tuple[ImagePixel, ...], ...]


class SpatialTransformError(ValueError):
    """Raised when a transform or coordinate falls outside its audited domain."""


@dataclass(frozen=True)
class MaskValidResult:
    """Binary transform result with explicit exclusion statistics.

    All rasters have shape ``(H, W)`` and integer values in ``{0, 1}``.
    ``effective_mask`` is the target mask intersected with ``valid`` and is the
    only target surface eligible for later loss, metric or region pooling.
    """

    mask: BinaryRaster
    valid: BinaryRaster
    effective_mask: BinaryRaster
    total_pixel_count: int
    valid_pixel_count: int
    excluded_pixel_count: int
    positive_valid_pixel_count: int


def _validate_hw(hw: tuple[int, int], *, context: str) -> tuple[int, int]:
    """Validate an explicit positive integer ``(height, width)`` pair."""

    if len(hw) != 2 or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in hw):
        raise SpatialTransformError(f"{context} must be a positive integer (height, width) pair")
    return hw


def identity_step(hw: tuple[int, int]) -> TransformStep:
    """Return an audited identity step for one ``(height, width)`` grid."""

    return TransformStep(
        operation="identity",
        input_hw=hw,
        output_hw=hw,
        interpolation="not_applicable",
        invertible=True,
        parameters={},
    )


def crop_step(
    input_hw: tuple[int, int],
    *,
    top: int,
    left: int,
    height: int,
    width: int,
) -> TransformStep:
    """Return a crop step whose inverse is defined on retained coordinates."""

    return TransformStep(
        operation="crop",
        input_hw=input_hw,
        output_hw=(height, width),
        interpolation="not_applicable",
        invertible=True,
        parameters={"top": top, "left": left, "height": height, "width": width},
    )


def resize_step(input_hw: tuple[int, int], output_hw: tuple[int, int]) -> TransformStep:
    """Return a resize step with the frozen image/mask/valid policy."""

    return TransformStep(
        operation="resize",
        input_hw=input_hw,
        output_hw=output_hw,
        interpolation=INTERPOLATION_POLICY,
        invertible=True,
        parameters={
            "coordinate_mapping": "pixel_edges",
            "raster_sampling": "half_pixel_centers",
            "raster_border_mode": "clamp",
        },
    )


def pad_step(
    input_hw: tuple[int, int],
    *,
    top: int,
    bottom: int,
    left: int,
    right: int,
) -> TransformStep:
    """Return a zero-fill pad step; padded pixels are always invalid evidence."""

    output_hw = (input_hw[0] + top + bottom, input_hw[1] + left + right)
    return TransformStep(
        operation="pad",
        input_hw=input_hw,
        output_hw=output_hw,
        interpolation="not_applicable",
        invertible=True,
        parameters={
            "top": top,
            "bottom": bottom,
            "left": left,
            "right": right,
            "image_fill": 0.0,
            "mask_fill": 0,
            "valid_fill": 0,
        },
    )


def build_transform_chain(steps: Sequence[TransformStep]) -> tuple[TransformStep, ...]:
    """Freeze and validate a non-empty, grid-continuous transform chain."""

    try:
        return validate_transform_sequence(tuple(steps))
    except ValueError as error:
        raise SpatialTransformError(str(error)) from error


def coordinate_inverse_available(steps: Sequence[TransformStep]) -> bool:
    """Return whether every step exposes a coordinate inverse on valid content."""

    chain = build_transform_chain(steps)
    return all(step.invertible for step in chain)


def _require_supported(step: TransformStep) -> None:
    """Reject operations deferred beyond the P1.2 crop/resize/pad scope."""

    if step.operation not in {"identity", "crop", "resize", "pad"}:
        raise SpatialTransformError(f"operation {step.operation!r} is not implemented in P1.2")


def _require_point_in_grid(point: ContinuousPoint, hw: tuple[int, int], *, context: str) -> None:
    """Validate a finite pixel-edge coordinate, including the far boundary."""

    x, y = point
    height, width = hw
    if not math.isfinite(x) or not math.isfinite(y):
        raise SpatialTransformError(f"{context} point must be finite")
    if not (0.0 <= x <= float(width) and 0.0 <= y <= float(height)):
        raise SpatialTransformError(f"{context} point lies outside grid {hw}")


def _forward_point_step(point: ContinuousPoint, step: TransformStep) -> ContinuousPoint:
    """Map one pixel-edge point through a supported step."""

    _require_supported(step)
    _require_point_in_grid(point, step.input_hw, context="forward")
    x, y = point
    if step.operation == "identity":
        result = (x, y)
    elif step.operation == "crop":
        left = float(step.parameters["left"])
        top = float(step.parameters["top"])
        width = float(step.parameters["width"])
        height = float(step.parameters["height"])
        if not (left <= x <= left + width and top <= y <= top + height):
            raise SpatialTransformError("coordinate lies outside the retained crop footprint")
        result = (x - left, y - top)
    elif step.operation == "resize":
        input_h, input_w = step.input_hw
        output_h, output_w = step.output_hw
        result = (x * output_w / input_w, y * output_h / input_h)
    else:
        result = (x + float(step.parameters["left"]), y + float(step.parameters["top"]))
    _require_point_in_grid(result, step.output_hw, context="forward output")
    return result


def _inverse_point_step(point: ContinuousPoint, step: TransformStep) -> ContinuousPoint:
    """Map one valid-content pixel-edge point through the inverse step."""

    _require_supported(step)
    if not step.invertible:
        raise SpatialTransformError(f"operation {step.operation!r} has no coordinate inverse")
    _require_point_in_grid(point, step.output_hw, context="inverse")
    x, y = point
    if step.operation == "identity":
        result = (x, y)
    elif step.operation == "crop":
        result = (x + float(step.parameters["left"]), y + float(step.parameters["top"]))
    elif step.operation == "resize":
        input_h, input_w = step.input_hw
        output_h, output_w = step.output_hw
        result = (x * input_w / output_w, y * input_h / output_h)
    else:
        left = float(step.parameters["left"])
        top = float(step.parameters["top"])
        input_h, input_w = step.input_hw
        if not (left <= x <= left + input_w and top <= y <= top + input_h):
            raise SpatialTransformError("padded coordinate has no source-space inverse")
        result = (x - left, y - top)
    _require_point_in_grid(result, step.input_hw, context="inverse output")
    return result


def forward_point(point: ContinuousPoint, steps: Sequence[TransformStep]) -> ContinuousPoint:
    """Map one finite ``(x, y)`` pixel-edge point through a chain."""

    result = point
    for step in build_transform_chain(steps):
        result = _forward_point_step(result, step)
    return result


def inverse_point(point: ContinuousPoint, steps: Sequence[TransformStep]) -> ContinuousPoint:
    """Invert one valid-content ``(x, y)`` point through a chain."""

    chain = build_transform_chain(steps)
    if not all(step.invertible for step in chain):
        raise SpatialTransformError("transform chain does not expose a complete coordinate inverse")
    result = point
    for step in reversed(chain):
        result = _inverse_point_step(result, step)
    return result


def _validate_continuous_box(box: ContinuousBox, hw: tuple[int, int]) -> None:
    """Validate an axis-aligned half-open box against a canvas."""

    x0, y0, x1, y1 = box
    if not all(math.isfinite(value) for value in box):
        raise SpatialTransformError("box coordinates must be finite")
    if x0 < 0.0 or y0 < 0.0 or x1 <= x0 or y1 <= y0:
        raise SpatialTransformError("box must satisfy non-negative x0<x1 and y0<y1")
    height, width = _validate_hw(hw, context="canvas_hw")
    if x1 > width or y1 > height:
        raise SpatialTransformError(f"box lies outside canvas {hw}")


def forward_box(box: ContinuousBox, steps: Sequence[TransformStep]) -> ContinuousBox:
    """Map a half-open pixel-edge box through axis-aligned P1.2 transforms."""

    chain = build_transform_chain(steps)
    _validate_continuous_box(box, chain[0].input_hw)
    x0, y0 = forward_point((box[0], box[1]), chain)
    x1, y1 = forward_point((box[2], box[3]), chain)
    result = (x0, y0, x1, y1)
    _validate_continuous_box(result, chain[-1].output_hw)
    return result


def inverse_box(box: ContinuousBox, steps: Sequence[TransformStep]) -> ContinuousBox:
    """Invert a valid-content half-open box through axis-aligned transforms."""

    chain = build_transform_chain(steps)
    _validate_continuous_box(box, chain[-1].output_hw)
    x0, y0 = inverse_point((box[0], box[1]), chain)
    x1, y1 = inverse_point((box[2], box[3]), chain)
    result = (x0, y0, x1, y1)
    _validate_continuous_box(result, chain[0].input_hw)
    return result


def quantize_covering_box(box: ContinuousBox, canvas_hw: tuple[int, int]) -> HalfOpenBox:
    """Expand a continuous box to the smallest covering integer half-open box."""

    _validate_continuous_box(box, canvas_hw)
    height, width = canvas_hw
    result = (
        max(0, math.floor(box[0])),
        max(0, math.floor(box[1])),
        min(width, math.ceil(box[2])),
        min(height, math.ceil(box[3])),
    )
    return validate_half_open_box(result)


def _validate_box_within_canvas(box: HalfOpenBox, canvas_hw: tuple[int, int]) -> HalfOpenBox:
    """Validate an integer reference box and its far boundaries."""

    if any(isinstance(value, bool) or not isinstance(value, int) for value in box):
        raise SpatialTransformError("reference half-open box coordinates must be integers")
    validated = validate_half_open_box(box)
    height, width = _validate_hw(canvas_hw, context="canvas_hw")
    if validated[2] > width or validated[3] > height:
        raise SpatialTransformError(f"half-open box lies outside canvas {canvas_hw}")
    return validated


def serialize_qwen1000_box(box: HalfOpenBox, canvas_hw: tuple[int, int]) -> QwenBox1000:
    """Serialize a reference-pixel box to coverage-preserving Qwen integers.

    Minima use floor and maxima use ceil, so quantization never discards target
    coverage.  The conversion occurs only at the language grounding boundary.
    """

    x0, y0, x1, y1 = _validate_box_within_canvas(box, canvas_hw)
    height, width = _validate_hw(canvas_hw, context="canvas_hw")
    result = (
        1000 * x0 // width,
        1000 * y0 // height,
        (1000 * x1 + width - 1) // width,
        (1000 * y1 + height - 1) // height,
    )
    if not (0 <= result[0] < result[2] <= 1000 and 0 <= result[1] < result[3] <= 1000):
        raise SpatialTransformError("Qwen-1000 serialization produced an invalid box")
    return result


def _validate_qwen_box(box: QwenBox1000) -> QwenBox1000:
    """Reject non-integer, bool or out-of-range Qwen coordinates."""

    if any(isinstance(value, bool) or not isinstance(value, int) for value in box):
        raise SpatialTransformError("Qwen-1000 coordinates must be integers")
    x0, y0, x1, y1 = box
    if not (0 <= x0 < x1 <= 1000 and 0 <= y0 < y1 <= 1000):
        raise SpatialTransformError("Qwen-1000 box must satisfy 0<=min<max<=1000")
    return box


def deserialize_qwen1000_box(box: QwenBox1000, canvas_hw: tuple[int, int]) -> HalfOpenBox:
    """Deserialize Qwen integers to a covering reference-pixel half-open box."""

    x0, y0, x1, y1 = _validate_qwen_box(box)
    height, width = _validate_hw(canvas_hw, context="canvas_hw")
    result = (
        x0 * width // 1000,
        y0 * height // 1000,
        min(width, (x1 * width + 999) // 1000),
        min(height, (y1 * height + 999) // 1000),
    )
    return _validate_box_within_canvas(result, canvas_hw)


def qwen_round_trip_error(box: HalfOpenBox, canvas_hw: tuple[int, int]) -> tuple[int, int, int, int]:
    """Return non-negative edge expansion after Qwen-1000 round-trip."""

    original = _validate_box_within_canvas(box, canvas_hw)
    recovered = deserialize_qwen1000_box(serialize_qwen1000_box(original, canvas_hw), canvas_hw)
    error = (
        original[0] - recovered[0],
        original[1] - recovered[1],
        recovered[2] - original[2],
        recovered[3] - original[3],
    )
    if min(error) < 0:
        raise SpatialTransformError("Qwen-1000 round-trip failed to preserve box coverage")
    height, width = _validate_hw(canvas_hw, context="canvas_hw")
    tolerance = (
        math.ceil(width / 1000),
        math.ceil(height / 1000),
        math.ceil(width / 1000),
        math.ceil(height / 1000),
    )
    if any(actual > allowed for actual, allowed in zip(error, tolerance, strict=True)):
        raise SpatialTransformError("Qwen-1000 round-trip exceeded its declared per-edge quantization bound")
    return error


def _normalize_binary_raster(raster: Sequence[Sequence[int | bool]], *, name: str) -> BinaryRaster:
    """Validate an explicit non-empty rectangular binary ``(H, W)`` raster."""

    rows = tuple(tuple(row) for row in raster)
    if not rows or not rows[0]:
        raise SpatialTransformError(f"{name} raster must be non-empty")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise SpatialTransformError(f"{name} raster must be rectangular")
    if any(
        not isinstance(value, (int, bool)) or value not in (0, 1, False, True)
        for row in rows
        for value in row
    ):
        raise SpatialTransformError(f"{name} raster values must be binary")
    return tuple(tuple(int(value) for value in row) for row in rows)


def _normalize_image_raster(raster: Sequence[Sequence[Sequence[float | int]]]) -> ImageRaster:
    """Validate an explicit finite, rectangular ``(H, W, C)`` image raster."""

    raw_rows = tuple(tuple(tuple(pixel) for pixel in row) for row in raster)
    if any(
        isinstance(channel, bool) or not isinstance(channel, (int, float))
        for row in raw_rows
        for pixel in row
        for channel in pixel
    ):
        raise SpatialTransformError("image raster channels must be explicit int or float values")
    rows = tuple(tuple(tuple(float(channel) for channel in pixel) for pixel in row) for row in raw_rows)
    if not rows or not rows[0] or not rows[0][0]:
        raise SpatialTransformError("image raster must be non-empty in H, W and C")
    width = len(rows[0])
    channels = len(rows[0][0])
    if any(len(row) != width for row in rows):
        raise SpatialTransformError("image raster must be rectangular")
    if any(len(pixel) != channels for row in rows for pixel in row):
        raise SpatialTransformError("image raster channel count must be constant")
    if any(not math.isfinite(channel) for row in rows for pixel in row for channel in pixel):
        raise SpatialTransformError("image raster values must be finite")
    return rows


def _binary_hw(raster: BinaryRaster) -> tuple[int, int]:
    """Return a validated binary raster's ``(height, width)``."""

    return (len(raster), len(raster[0]))


def _image_hw(raster: ImageRaster) -> tuple[int, int]:
    """Return a validated image raster's ``(height, width)``."""

    return (len(raster), len(raster[0]))


def _nearest_index(output_index: int, *, input_size: int, output_size: int) -> int:
    """Map an output center to its nearest input pixel using integer math."""

    return min(input_size - 1, ((2 * output_index + 1) * input_size) // (2 * output_size))


def _resize_binary_nearest(raster: BinaryRaster, output_hw: tuple[int, int]) -> BinaryRaster:
    """Resize a binary raster with the sole permitted nearest policy."""

    input_h, input_w = _binary_hw(raster)
    output_h, output_w = output_hw
    return tuple(
        tuple(
            raster[
                _nearest_index(output_y, input_size=input_h, output_size=output_h)
            ][_nearest_index(output_x, input_size=input_w, output_size=output_w)]
            for output_x in range(output_w)
        )
        for output_y in range(output_h)
    )


def _bilinear_axis(output_index: int, *, input_size: int, output_size: int) -> tuple[int, int, float]:
    """Return clamped half-pixel-center bilinear neighbors and high weight."""

    source = (output_index + 0.5) * input_size / output_size - 0.5
    source = min(float(input_size - 1), max(0.0, source))
    low = math.floor(source)
    high = min(input_size - 1, low + 1)
    return low, high, source - low


def _resize_image_bilinear(raster: ImageRaster, output_hw: tuple[int, int]) -> ImageRaster:
    """Resize an HWC image with frozen bilinear half-pixel-center sampling."""

    input_h, input_w = _image_hw(raster)
    output_h, output_w = output_hw
    channels = len(raster[0][0])
    rows: list[tuple[ImagePixel, ...]] = []
    for output_y in range(output_h):
        y0, y1, wy = _bilinear_axis(output_y, input_size=input_h, output_size=output_h)
        row: list[ImagePixel] = []
        for output_x in range(output_w):
            x0, x1, wx = _bilinear_axis(output_x, input_size=input_w, output_size=output_w)
            pixel = tuple(
                (1.0 - wy) * ((1.0 - wx) * raster[y0][x0][channel] + wx * raster[y0][x1][channel])
                + wy * ((1.0 - wx) * raster[y1][x0][channel] + wx * raster[y1][x1][channel])
                for channel in range(channels)
            )
            row.append(pixel)
        rows.append(tuple(row))
    return tuple(rows)


def apply_binary_transform(
    raster: Sequence[Sequence[int | bool]],
    steps: Sequence[TransformStep],
    *,
    kind: Literal["mask", "valid"],
) -> BinaryRaster:
    """Apply crop/nearest-resize/zero-pad to an explicit ``(H, W)`` raster.

    Args:
        raster: CPU nested sequence with binary values and no implicit channel.
        steps: Audited transform chain.
        kind: ``mask`` or ``valid``; both are nearest-only by contract.

    Returns:
        Immutable integer binary raster on the final grid.

    Raises:
        SpatialTransformError: Shape, value, operation or policy is invalid.
    """

    result = _normalize_binary_raster(raster, name=kind)
    for step in build_transform_chain(steps):
        _require_supported(step)
        if _binary_hw(result) != step.input_hw:
            raise SpatialTransformError(f"{kind} raster shape does not match transform input_hw")
        if step.operation == "crop":
            top = int(step.parameters["top"])
            left = int(step.parameters["left"])
            height = int(step.parameters["height"])
            width = int(step.parameters["width"])
            result = tuple(tuple(row[left : left + width]) for row in result[top : top + height])
        elif step.operation == "resize":
            if step.interpolation != INTERPOLATION_POLICY:
                raise SpatialTransformError("binary resize is not bound to the frozen nearest policy")
            result = _resize_binary_nearest(result, step.output_hw)
        elif step.operation == "pad":
            top = int(step.parameters["top"])
            bottom = int(step.parameters["bottom"])
            left = int(step.parameters["left"])
            right = int(step.parameters["right"])
            fill_value = int(step.parameters[f"{kind}_fill"])
            output_width = len(result[0]) + left + right
            fill_row = (fill_value,) * output_width
            result = (
                *((fill_row,) * top),
                *(tuple((fill_value,) * left + row + (fill_value,) * right) for row in result),
                *((fill_row,) * bottom),
            )
        if _binary_hw(result) != step.output_hw:
            raise SpatialTransformError(f"{kind} raster output shape does not match transform output_hw")
    return result


def apply_image_transform(
    raster: Sequence[Sequence[Sequence[float | int]]],
    steps: Sequence[TransformStep],
) -> ImageRaster:
    """Apply crop/bilinear-resize/zero-pad to an explicit CPU HWC image."""

    result = _normalize_image_raster(raster)
    for step in build_transform_chain(steps):
        _require_supported(step)
        if _image_hw(result) != step.input_hw:
            raise SpatialTransformError("image raster shape does not match transform input_hw")
        if step.operation == "crop":
            top = int(step.parameters["top"])
            left = int(step.parameters["left"])
            height = int(step.parameters["height"])
            width = int(step.parameters["width"])
            result = tuple(tuple(row[left : left + width]) for row in result[top : top + height])
        elif step.operation == "resize":
            if step.interpolation != INTERPOLATION_POLICY:
                raise SpatialTransformError("image resize is not bound to the frozen bilinear policy")
            result = _resize_image_bilinear(result, step.output_hw)
        elif step.operation == "pad":
            top = int(step.parameters["top"])
            bottom = int(step.parameters["bottom"])
            left = int(step.parameters["left"])
            right = int(step.parameters["right"])
            channels = len(result[0][0])
            fill_pixel = (float(step.parameters["image_fill"]),) * channels
            output_width = len(result[0]) + left + right
            fill_row = (fill_pixel,) * output_width
            result = (
                *((fill_row,) * top),
                *(tuple((fill_pixel,) * left + row + (fill_pixel,) * right) for row in result),
                *((fill_row,) * bottom),
            )
        if _image_hw(result) != step.output_hw:
            raise SpatialTransformError("image raster output shape does not match transform output_hw")
    return result


def transform_mask_and_valid(
    mask: Sequence[Sequence[int | bool]],
    valid: Sequence[Sequence[int | bool]],
    steps: Sequence[TransformStep],
) -> MaskValidResult:
    """Transform target/valid masks and compute the only eligible target area."""

    transformed_mask = apply_binary_transform(mask, steps, kind="mask")
    transformed_valid = apply_binary_transform(valid, steps, kind="valid")
    if _binary_hw(transformed_mask) != _binary_hw(transformed_valid):
        raise SpatialTransformError("mask and valid raster shapes must match")
    effective = tuple(
        tuple(mask_value & valid_value for mask_value, valid_value in zip(mask_row, valid_row, strict=True))
        for mask_row, valid_row in zip(transformed_mask, transformed_valid, strict=True)
    )
    total = len(effective) * len(effective[0])
    valid_count = sum(value for row in transformed_valid for value in row)
    positive_valid = sum(value for row in effective for value in row)
    return MaskValidResult(
        mask=transformed_mask,
        valid=transformed_valid,
        effective_mask=effective,
        total_pixel_count=total,
        valid_pixel_count=valid_count,
        excluded_pixel_count=total - valid_count,
        positive_valid_pixel_count=positive_valid,
    )


__all__ = [
    "BinaryRaster",
    "ContinuousBox",
    "ContinuousPoint",
    "INTERPOLATION_POLICY",
    "ImageRaster",
    "MaskValidResult",
    "QwenBox1000",
    "SpatialTransformError",
    "apply_binary_transform",
    "apply_image_transform",
    "build_transform_chain",
    "coordinate_inverse_available",
    "crop_step",
    "deserialize_qwen1000_box",
    "forward_box",
    "forward_point",
    "identity_step",
    "inverse_box",
    "inverse_point",
    "pad_step",
    "quantize_covering_box",
    "qwen_round_trip_error",
    "resize_step",
    "serialize_qwen1000_box",
    "transform_mask_and_valid",
]
