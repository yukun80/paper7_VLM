"""Small standard-library metadata readers used by bounded source probes."""

from __future__ import annotations

import ast
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Literal

from sami_gsd.data.adapters.base import SourceAdapterError


@dataclass(frozen=True)
class ImageHeader:
    """Container-level image shape with no decoded pixel buffer."""

    container: Literal["png", "jpeg"]
    height: int
    width: int
    channels: int
    dtype: str


@dataclass(frozen=True)
class NpyHeader:
    """Validated NPY header metadata without loading array payload bytes."""

    shape: tuple[int, ...]
    dtype: str
    fortran_order: bool


def _read_png_header(handle: BinaryIO) -> ImageHeader:
    """Read the PNG signature and IHDR only."""

    signature = handle.read(8)
    if signature != b"\x89PNG\r\n\x1a\n":
        raise SourceAdapterError("invalid PNG signature")
    length = struct.unpack(">I", handle.read(4))[0]
    if handle.read(4) != b"IHDR" or length != 13:
        raise SourceAdapterError("PNG does not start with a canonical IHDR")
    width, height, bit_depth, color_type, compression, filtering, interlace = struct.unpack(
        ">IIBBBBB", handle.read(13)
    )
    if min(width, height) <= 0 or compression != 0 or filtering != 0 or interlace not in {0, 1}:
        raise SourceAdapterError("unsupported PNG IHDR")
    channels_by_color = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}
    if color_type not in channels_by_color:
        raise SourceAdapterError("unsupported PNG color type")
    return ImageHeader("png", height, width, channels_by_color[color_type], f"uint{bit_depth}")


def _read_jpeg_header(handle: BinaryIO) -> ImageHeader:
    """Find a baseline/progressive JPEG SOF marker without decoding pixels."""

    if handle.read(2) != b"\xff\xd8":
        raise SourceAdapterError("invalid JPEG signature")
    sof_markers = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
    while True:
        prefix = handle.read(1)
        if not prefix:
            raise SourceAdapterError("JPEG ended before a supported SOF marker")
        if prefix != b"\xff":
            continue
        marker_byte = handle.read(1)
        while marker_byte == b"\xff":
            marker_byte = handle.read(1)
        if not marker_byte:
            raise SourceAdapterError("truncated JPEG marker")
        marker = marker_byte[0]
        if marker in {0x01, *range(0xD0, 0xD9)}:
            continue
        length_bytes = handle.read(2)
        if len(length_bytes) != 2:
            raise SourceAdapterError("truncated JPEG segment length")
        segment_length = struct.unpack(">H", length_bytes)[0]
        if segment_length < 2:
            raise SourceAdapterError("invalid JPEG segment length")
        if marker in sof_markers:
            payload = handle.read(segment_length - 2)
            if len(payload) < 6:
                raise SourceAdapterError("truncated JPEG SOF segment")
            precision, height, width, channels = struct.unpack(">BHHB", payload[:6])
            if min(height, width, channels) <= 0:
                raise SourceAdapterError("invalid JPEG SOF dimensions")
            return ImageHeader("jpeg", height, width, channels, f"uint{precision}")
        handle.seek(segment_length - 2, 1)


def read_image_header(path: Path) -> ImageHeader:
    """Identify PNG/JPEG by bytes, not by an unreliable filename suffix."""

    with path.open("rb") as handle:
        signature = handle.read(8)
        handle.seek(0)
        if signature == b"\x89PNG\r\n\x1a\n":
            return _read_png_header(handle)
        if signature[:2] == b"\xff\xd8":
            return _read_jpeg_header(handle)
    raise SourceAdapterError(f"unsupported image container for bounded adapter: {path.name}")


def read_npy_header(path: Path) -> NpyHeader:
    """Read NPY v1-v3 metadata without importing NumPy or loading the payload."""

    with path.open("rb") as handle:
        if handle.read(6) != b"\x93NUMPY":
            raise SourceAdapterError("invalid NPY signature")
        version = tuple(handle.read(2))
        if version == (1, 0):
            length_bytes = handle.read(2)
            header_length = struct.unpack("<H", length_bytes)[0]
            encoding = "latin1"
        elif version in {(2, 0), (3, 0)}:
            length_bytes = handle.read(4)
            header_length = struct.unpack("<I", length_bytes)[0]
            encoding = "utf-8" if version == (3, 0) else "latin1"
        else:
            raise SourceAdapterError(f"unsupported NPY version: {version}")
        header = handle.read(header_length).decode(encoding)
    try:
        payload = ast.literal_eval(header.strip())
    except (SyntaxError, ValueError) as error:
        raise SourceAdapterError("invalid NPY header mapping") from error
    if not isinstance(payload, dict) or set(payload) != {"descr", "fortran_order", "shape"}:
        raise SourceAdapterError("NPY header must contain exactly descr, fortran_order and shape")
    shape = payload["shape"]
    if not isinstance(shape, tuple) or not shape or any(type(value) is not int or value <= 0 for value in shape):
        raise SourceAdapterError("NPY shape must contain positive integers")
    if type(payload["fortran_order"]) is not bool or not isinstance(payload["descr"], str):
        raise SourceAdapterError("NPY dtype/fortran metadata is invalid")
    return NpyHeader(shape=shape, dtype=payload["descr"], fortran_order=payload["fortran_order"])


def read_first_json_array_item(path: Path, *, array_key: str | None = None, max_bytes: int = 2_000_000) -> Any:
    """Decode only the first item of a top-level or named JSON array."""

    with path.open("r", encoding="utf-8") as handle:
        text = handle.read(max_bytes)
    if array_key is None:
        position = text.find("[")
    else:
        key_position = text.find(json.dumps(array_key))
        position = text.find("[", key_position + len(array_key)) if key_position >= 0 else -1
    if position < 0:
        raise SourceAdapterError(f"JSON array boundary not found in bounded prefix: {path.name}")
    start = position + 1
    while start < len(text) and text[start].isspace():
        start += 1
    decoder = json.JSONDecoder()
    try:
        value, _ = decoder.raw_decode(text, start)
    except json.JSONDecodeError as error:
        raise SourceAdapterError(f"first JSON array item exceeds or violates the bounded parser: {path.name}") from error
    return value


__all__ = ["ImageHeader", "NpyHeader", "read_first_json_array_item", "read_image_header", "read_npy_header"]
