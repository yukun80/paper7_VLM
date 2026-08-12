"""自包含图像规范化、内容寻址与 bbox 同步几何变换。"""

from __future__ import annotations

import io
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from PIL import Image, ImageOps, UnidentifiedImageError

from .common import portable_relative_path, sha256_bytes, sha256_file
from .config import AssetPolicyConfig, LimitsConfig
from .contracts import MediaType, PendingAsset
from .errors import AssetError, ReasonCode


@dataclass(frozen=True)
class ImageGeometry:
    source_size: tuple[int, int]
    oriented_size: tuple[int, int]
    output_size: tuple[int, int]
    exif_orientation: int
    transforms: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class MaterializedAsset:
    media: Mapping[str, Any]
    geometry: ImageGeometry | None


@dataclass(frozen=True)
class NormalizedImage:
    payload: bytes
    source_sha256: str
    extension: str
    width: int
    height: int
    geometry: ImageGeometry


def _has_symlink_component(path: Path) -> bool:
    current = path
    while True:
        if current.is_symlink():
            return True
        if current.parent == current:
            return False
        current = current.parent


def _source_bytes(asset: PendingAsset) -> tuple[bytes, str]:
    asset.validate()
    if asset.payload is not None:
        if not asset.payload:
            raise AssetError(ReasonCode.ASSET_ZERO_BYTES, f"{asset.role}: payload 为空")
        return asset.payload, sha256_bytes(asset.payload)
    if asset.source_path is not None:
        path = asset.source_path
        if _has_symlink_component(path):
            raise AssetError(
                ReasonCode.SOURCE_SYMLINK,
                f"{asset.role}: 不接受 source symlink",
                details={"source_ref": asset.source_ref},
            )
        if not path.is_file():
            raise AssetError(
                ReasonCode.ASSET_MISSING,
                f"{asset.role}: source asset 缺失",
                details={"source_ref": asset.source_ref},
            )
        if path.stat().st_size == 0:
            raise AssetError(
                ReasonCode.ASSET_ZERO_BYTES,
                f"{asset.role}: source asset 为零字节",
                details={"source_ref": asset.source_ref},
            )
        payload = path.read_bytes()
        return payload, sha256_bytes(payload)
    assert asset.archive_path is not None and asset.archive_member is not None
    if _has_symlink_component(asset.archive_path):
        raise AssetError(
            ReasonCode.SOURCE_SYMLINK,
            f"{asset.role}: 不接受 archive symlink",
            details={"source_ref": asset.source_ref},
        )
    if not asset.archive_path.is_file():
        raise AssetError(
            ReasonCode.ASSET_MISSING,
            f"{asset.role}: archive 缺失",
            details={"source_ref": asset.source_ref},
        )
    portable_relative_path(
        asset.archive_member, location=f"asset[{asset.role}].archive_member"
    )
    try:
        with zipfile.ZipFile(asset.archive_path) as archive:
            info = archive.getinfo(asset.archive_member)
            unix_mode = (info.external_attr >> 16) & 0o170000
            if unix_mode == 0o120000:
                raise AssetError(
                    ReasonCode.SOURCE_SYMLINK,
                    f"{asset.role}: archive member 是 symlink",
                    details={"source_ref": asset.source_ref},
                )
            payload = archive.read(info)
    except KeyError as error:
        raise AssetError(
            ReasonCode.ASSET_MISSING,
            f"{asset.role}: archive member 缺失",
            details={"source_ref": asset.source_ref},
        ) from error
    except zipfile.BadZipFile as error:
        raise AssetError(
            ReasonCode.ASSET_CORRUPT,
            f"{asset.role}: archive 损坏",
            details={"source_ref": asset.source_ref},
        ) from error
    if not payload:
        raise AssetError(
            ReasonCode.ASSET_ZERO_BYTES,
            f"{asset.role}: archive member 为零字节",
            details={"source_ref": asset.source_ref},
        )
    return payload, sha256_bytes(payload)


def _orientation_transform(
    x: float,
    y: float,
    *,
    width: int,
    height: int,
    orientation: int,
) -> tuple[float, float]:
    if orientation == 1:
        return x, y
    if orientation == 2:
        return width - x, y
    if orientation == 3:
        return width - x, height - y
    if orientation == 4:
        return x, height - y
    if orientation == 5:
        return y, x
    if orientation == 6:
        return height - y, x
    if orientation == 7:
        return height - y, width - x
    if orientation == 8:
        return y, width - x
    return x, y


def _normalize_image(
    asset: PendingAsset,
    policy: AssetPolicyConfig,
) -> NormalizedImage:
    raw, source_digest = _source_bytes(asset)
    try:
        with Image.open(io.BytesIO(raw)) as opened:
            opened.load()
            source_width, source_height = opened.size
            orientation_value = opened.getexif().get(274, 1)
            orientation = (
                int(orientation_value)
                if isinstance(orientation_value, int)
                and 1 <= orientation_value <= 8
                else 1
            )
            oriented = ImageOps.exif_transpose(opened).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise AssetError(
            ReasonCode.ASSET_CORRUPT,
            f"{asset.role}: 图像无法解码",
            details={"source_ref": asset.source_ref, "error": str(error)},
        ) from error
    oriented_size = oriented.size
    transforms: list[dict[str, Any]] = [
        {
            "type": "exif_orientation",
            "orientation": orientation,
            "input_size": [source_width, source_height],
            "output_size": list(oriented_size),
        },
        {"type": "color_conversion", "output_mode": "RGB"},
    ]
    if policy.target_size is not None and oriented.size != policy.target_size:
        output = oriented.resize(
            policy.target_size,
            resample=Image.Resampling.BILINEAR,
        )
        transforms.append(
            {
                "type": "resize",
                "interpolation": "bilinear",
                "input_size": list(oriented.size),
                "output_size": list(policy.target_size),
            }
        )
    else:
        output = oriented
        transforms.append(
            {
                "type": "resize",
                "interpolation": "none",
                "input_size": list(oriented.size),
                "output_size": list(oriented.size),
            }
        )
    encoded = io.BytesIO()
    if policy.image_format == "jpeg":
        output.save(
            encoded,
            format="JPEG",
            quality=policy.image_quality,
            subsampling=policy.jpeg_subsampling,
            optimize=False,
            progressive=False,
            exif=b"",
        )
        extension = "jpg"
    else:
        output.save(
            encoded,
            format="PNG",
            optimize=False,
            compress_level=9,
        )
        extension = "png"
    geometry = ImageGeometry(
        source_size=(source_width, source_height),
        oriented_size=oriented_size,
        output_size=output.size,
        exif_orientation=orientation,
        transforms=tuple(transforms),
    )
    return NormalizedImage(
        payload=encoded.getvalue(),
        source_sha256=source_digest,
        extension=extension,
        width=output.width,
        height=output.height,
        geometry=geometry,
    )


def normalized_image_sha256(
    asset: PendingAsset,
    *,
    policy: AssetPolicyConfig,
) -> str:
    """返回与最终内容寻址图像完全一致的 SHA-256，不写出资产。"""

    if asset.media_type is not MediaType.IMAGE:
        raise AssetError(
            ReasonCode.TYPE_MISMATCH,
            f"{asset.role}: 只有 image 可以计算规范图像摘要",
        )
    return sha256_bytes(_normalize_image(asset, policy).payload)


class AssetStore:
    def __init__(
        self,
        root: Path,
        *,
        policy: AssetPolicyConfig,
        limits: LimitsConfig,
    ) -> None:
        self.root = root
        self.policy = policy
        self.limits = limits
        self.entries: dict[str, dict[str, Any]] = {}
        self.total_bytes = 0

    def _store(
        self,
        *,
        payload: bytes,
        kind: str,
        extension: str,
        source_sha256: str,
        role: str,
        width: int | None = None,
        height: int | None = None,
        mode: str | None = None,
    ) -> dict[str, Any]:
        digest = sha256_bytes(payload)
        relative = Path("assets") / kind / digest[:2] / f"{digest}.{extension}"
        relative_text = relative.as_posix()
        path = self.root / relative
        if relative_text not in self.entries:
            next_count = len(self.entries) + 1
            next_bytes = self.total_bytes + len(payload)
            if (
                self.limits.max_assets is not None
                and next_count > self.limits.max_assets
            ):
                raise AssetError(
                    ReasonCode.ASSET_LIMIT_EXCEEDED,
                    "唯一资产数超过配置上限",
                    details={"max_assets": self.limits.max_assets},
                )
            if (
                self.limits.max_copied_bytes is not None
                and next_bytes > self.limits.max_copied_bytes
            ):
                raise AssetError(
                    ReasonCode.ASSET_LIMIT_EXCEEDED,
                    "复制字节数超过配置上限",
                    details={
                        "max_copied_bytes": self.limits.max_copied_bytes,
                        "attempted_bytes": next_bytes,
                    },
                )
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                if path.is_symlink() or path.stat().st_nlink != 1:
                    raise AssetError(
                        ReasonCode.OUTPUT_LINK,
                        f"输出资产不能是链接：{relative_text}",
                    )
                if path.read_bytes() != payload:
                    raise AssetError(
                        ReasonCode.HASH_MISMATCH,
                        f"内容寻址冲突：{relative_text}",
                    )
            else:
                descriptor = os.open(
                    path,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o644,
                )
                try:
                    with os.fdopen(descriptor, "wb") as handle:
                        handle.write(payload)
                        handle.flush()
                        os.fsync(handle.fileno())
                except BaseException:
                    path.unlink(missing_ok=True)
                    raise
            self.entries[relative_text] = {
                "asset_id": f"asset_{digest}",
                "path": relative_text,
                "media_type": kind,
                "sha256": digest,
                "size_bytes": len(payload),
                "extension": extension,
            }
            self.total_bytes = next_bytes
        entry = dict(self.entries[relative_text])
        entry.update(
            {
                "role": role,
                "source_sha256": source_sha256,
            }
        )
        if width is not None:
            entry["width"] = width
        if height is not None:
            entry["height"] = height
        if mode is not None:
            entry["mode"] = mode
        return entry

    def add_image(self, asset: PendingAsset) -> MaterializedAsset:
        normalized = _normalize_image(asset, self.policy)
        media = self._store(
            payload=normalized.payload,
            kind=MediaType.IMAGE.value,
            extension=normalized.extension,
            source_sha256=normalized.source_sha256,
            role=asset.role,
            width=normalized.width,
            height=normalized.height,
            mode="RGB",
        )
        return MaterializedAsset(media=media, geometry=normalized.geometry)

    def materialize(
        self,
        assets: tuple[PendingAsset, ...],
    ) -> tuple[list[dict[str, Any]], dict[str, ImageGeometry], list[dict[str, Any]]]:
        by_role: dict[str, PendingAsset] = {}
        for asset in assets:
            if asset.role in by_role:
                raise AssetError(
                    ReasonCode.SCHEMA_MISMATCH,
                    f"重复 asset role：{asset.role}",
                )
            by_role[asset.role] = asset
        output: list[dict[str, Any]] = []
        geometry: dict[str, ImageGeometry] = {}
        transforms: list[dict[str, Any]] = []
        for asset in assets:
            if asset.media_type is not MediaType.IMAGE:
                raise AssetError(
                    ReasonCode.INVALID_ENUM,
                    f"{asset.role}: RS-GeneralDesc 只接受 image media",
                )
            item = self.add_image(asset)
            output.append(dict(item.media))
            assert item.geometry is not None
            geometry[asset.role] = item.geometry
            transforms.extend(
                {"asset_role": asset.role, **dict(row)}
                for row in item.geometry.transforms
            )
        output.sort(key=lambda row: (str(row["role"]), str(row["asset_id"])))
        return output, geometry, transforms

    @property
    def inventory(self) -> list[dict[str, Any]]:
        return [dict(self.entries[path]) for path in sorted(self.entries)]

    def checkpoint(self) -> tuple[frozenset[str], int]:
        return frozenset(self.entries), self.total_bytes

    def rollback(self, checkpoint: tuple[frozenset[str], int]) -> None:
        previous, previous_bytes = checkpoint
        new_paths = sorted(set(self.entries) - set(previous))
        for relative in new_paths:
            path = self.root / relative
            if path.is_file() and not path.is_symlink():
                path.unlink()
            self.entries.pop(relative, None)
        self.total_bytes = previous_bytes


def normalize_bbox_target(
    target: Mapping[str, Any],
    geometry: Mapping[str, ImageGeometry],
) -> dict[str, Any]:
    if target.get("type") != "bbox":
        return dict(target)
    role = target.get("image_role")
    if not isinstance(role, str) or role not in geometry:
        raise AssetError(ReasonCode.BBOX_INVALID, "bbox image_role 缺少对应图像")
    image = geometry[role]
    convention = target.get("source_convention")
    if convention not in {
        "xyxy_normalized_top_left",
        "xywh_pixel_top_left",
    }:
        raise AssetError(
            ReasonCode.BBOX_INVALID,
            f"未知 bbox source_convention：{convention!r}",
        )
    source_width, source_height = image.source_size
    output_width, output_height = image.output_size
    boxes = target.get("boxes")
    if not isinstance(boxes, list) or not boxes:
        raise AssetError(ReasonCode.BBOX_INVALID, "bbox boxes 必须是非空列表")
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(boxes):
        if not isinstance(row, dict) or not isinstance(row.get("source"), list):
            raise AssetError(ReasonCode.BBOX_INVALID, f"bbox[{index}] 合同非法")
        values = row["source"]
        if len(values) != 4 or any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in values
        ):
            raise AssetError(ReasonCode.BBOX_INVALID, f"bbox[{index}] 数值非法")
        if convention == "xyxy_normalized_top_left":
            x0, y0, x1, y1 = (
                float(values[0]) * source_width,
                float(values[1]) * source_height,
                float(values[2]) * source_width,
                float(values[3]) * source_height,
            )
        else:
            x0, y0, width, height = (float(value) for value in values)
            x1 = x0 + width
            y1 = y0 + height
        if (
            x0 < 0
            or y0 < 0
            or x1 > source_width
            or y1 > source_height
            or x1 <= x0
            or y1 <= y0
        ):
            raise AssetError(
                ReasonCode.BBOX_ZERO_AREA
                if x1 <= x0 or y1 <= y0
                else ReasonCode.BBOX_INVALID,
                f"bbox[{index}] 超界或零面积",
                details={
                    "source_xyxy": [x0, y0, x1, y1],
                    "source_size": [source_width, source_height],
                },
            )
        corners = [
            _orientation_transform(
                x,
                y,
                width=source_width,
                height=source_height,
                orientation=image.exif_orientation,
            )
            for x, y in ((x0, y0), (x1, y0), (x0, y1), (x1, y1))
        ]
        oriented_x0 = min(point[0] for point in corners)
        oriented_y0 = min(point[1] for point in corners)
        oriented_x1 = max(point[0] for point in corners)
        oriented_y1 = max(point[1] for point in corners)
        scale_x = output_width / image.oriented_size[0]
        scale_y = output_height / image.oriented_size[1]
        canonical = [
            oriented_x0 * scale_x / output_width,
            oriented_y0 * scale_y / output_height,
            oriented_x1 * scale_x / output_width,
            oriented_y1 * scale_y / output_height,
        ]
        if (
            min(canonical) < -1e-12
            or max(canonical) > 1 + 1e-12
            or canonical[2] <= canonical[0]
            or canonical[3] <= canonical[1]
        ):
            raise AssetError(
                ReasonCode.BBOX_INVALID,
                f"bbox[{index}] 变换后非法",
            )
        normalized.append(
            {
                **{key: value for key, value in row.items() if key != "source"},
                "source": list(values),
                "source_convention": convention,
                "source_image_size": [source_width, source_height],
                "canonical_xyxy_norm": [
                    float(round(max(0.0, min(1.0, value)), 10))
                    for value in canonical
                ],
            }
        )
    return {
        "type": "bbox",
        "image_role": role,
        "source_convention": convention,
        "canonical_convention": "xyxy_normalized_top_left",
        "boxes": normalized,
    }
