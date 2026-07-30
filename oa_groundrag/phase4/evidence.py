"""Mask-Grounded Evidence Builder：资产、确定性事实和限制。"""

from __future__ import annotations

import math
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

from oa_groundrag.phase3.common import first_symlink_component

from .contracts import (
    EVIDENCE_SCHEMA_VERSION,
    AlignmentStatus,
    AuxiliaryView,
    EvidenceAsset,
    EvidenceBundle,
    MaskMode,
    SelectedRegion,
    TargetStatus,
)
from .errors import EvidenceError, ReasonCode
from .regions import mask_bbox, mask_centroid


@dataclass(frozen=True)
class EvidenceBuildResult:
    bundle: EvidenceBundle
    root: Path


def _rgb_image(value: Any, *, location: str) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    array = np.asarray(value)
    if array.ndim == 2:
        if not np.issubdtype(array.dtype, np.number):
            raise EvidenceError(
                ReasonCode.TYPE_MISMATCH,
                f"{location}: 无法渲染非数值二维数组",
            )
        finite = np.isfinite(array)
        if not bool(finite.any()):
            scaled = np.zeros(array.shape, dtype=np.uint8)
        else:
            low = float(np.nanmin(array[finite]))
            high = float(np.nanmax(array[finite]))
            if high <= low:
                scaled = np.zeros(array.shape, dtype=np.uint8)
            else:
                scaled = np.zeros(array.shape, dtype=np.uint8)
                scaled[finite] = np.clip(
                    (array[finite] - low) / (high - low) * 255.0,
                    0,
                    255,
                ).astype(np.uint8)
        return Image.fromarray(scaled).convert("RGB")
    if array.ndim != 3 or array.shape[2] not in {3, 4}:
        raise EvidenceError(
            ReasonCode.TYPE_MISMATCH,
            f"{location}: 必须是 PIL image、HW 或 HWC(3/4) 数组",
        )
    if array.dtype != np.uint8:
        if not np.issubdtype(array.dtype, np.number) or not bool(
            np.isfinite(array).all()
        ):
            raise EvidenceError(
                ReasonCode.NONFINITE_NUMBER,
                f"{location}: 图像数组必须是有限数",
            )
        array = np.clip(array, 0, 255).astype(np.uint8)
    mode = "RGB" if array.shape[2] == 3 else "RGBA"
    return Image.fromarray(array, mode=mode).convert("RGB")


def _component_count(mask: np.ndarray) -> int:
    """8 连通组件数，避免把视觉模型特征误当几何事实。"""

    height, width = mask.shape
    visited = np.zeros(mask.shape, dtype=bool)
    components = 0
    for y, x in zip(*np.nonzero(mask), strict=True):
        if visited[y, x]:
            continue
        components += 1
        stack = [(int(y), int(x))]
        visited[y, x] = True
        while stack:
            current_y, current_x = stack.pop()
            for offset_y in (-1, 0, 1):
                for offset_x in (-1, 0, 1):
                    if offset_x == 0 and offset_y == 0:
                        continue
                    next_y = current_y + offset_y
                    next_x = current_x + offset_x
                    if (
                        0 <= next_y < height
                        and 0 <= next_x < width
                        and mask[next_y, next_x]
                        and not visited[next_y, next_x]
                    ):
                        visited[next_y, next_x] = True
                        stack.append((next_y, next_x))
    return components


def _perimeter_4(mask: np.ndarray) -> int:
    padded = np.pad(mask, 1, constant_values=False)
    center = padded[1:-1, 1:-1]
    perimeter = 0
    for neighbor in (
        padded[:-2, 1:-1],
        padded[2:, 1:-1],
        padded[1:-1, :-2],
        padded[1:-1, 2:],
    ):
        perimeter += int(np.count_nonzero(center & ~neighbor))
    return perimeter


def deterministic_mask_facts(mask: np.ndarray) -> dict[str, Any]:
    """只由 mask 程序计算，输出均为有限 JSON 标量。"""

    height, width = mask.shape
    area = int(mask.sum())
    if area <= 0:
        raise EvidenceError(
            ReasonCode.MASK_INVALID,
            "empty mask 不允许计算几何事实",
        )
    bbox = mask_bbox(mask)
    centroid = mask_centroid(mask)
    row = min(2, int(3 * (centroid[1] + 0.5) / height))
    column = min(2, int(3 * (centroid[0] + 0.5) / width))
    names = (
        ("northwest", "north", "northeast"),
        ("west", "center", "east"),
        ("southwest", "south", "southeast"),
    )
    components = _component_count(mask)
    perimeter = _perimeter_4(mask)
    compactness = float(4.0 * math.pi * area / (perimeter * perimeter))
    ys, xs = np.nonzero(mask)
    elongation: float | None
    if area < 2:
        elongation = None
    else:
        coordinates = np.stack((xs.astype(float), ys.astype(float)), axis=0)
        covariance = np.cov(coordinates, bias=True)
        eigenvalues = np.linalg.eigvalsh(covariance)
        if float(eigenvalues[0]) <= 1e-12:
            elongation = None
        else:
            elongation = float(
                math.sqrt(float(eigenvalues[-1] / eigenvalues[0]))
            )
    return {
        "bbox_xyxy_pixel_half_open": list(bbox),
        "centroid_xy_pixel": [centroid[0], centroid[1]],
        "area_pixels": area,
        "area_ratio": float(area / (height * width)),
        "location_3x3": names[row][column],
        "fragment_connectivity": 8,
        "fragment_count": components,
        "perimeter_connectivity": 4,
        "perimeter_pixels": perimeter,
        "compactness_4pi_area_over_perimeter2": compactness,
        "covariance_elongation": elongation,
    }


class EvidenceBuilder:
    def __init__(
        self,
        *,
        context_margin_ratio: float = 0.15,
        overlay_alpha: float = 0.45,
        max_auxiliary_views: int = 2,
        min_auxiliary_coverage: float = 0.8,
    ) -> None:
        for name, value in (
            ("context_margin_ratio", context_margin_ratio),
            ("overlay_alpha", overlay_alpha),
            ("min_auxiliary_coverage", min_auxiliary_coverage),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0 <= float(value) <= 1
            ):
                raise EvidenceError(
                    ReasonCode.TYPE_MISMATCH,
                    f"{name} 必须是 [0,1] 有限数",
                )
        if (
            isinstance(max_auxiliary_views, bool)
            or not isinstance(max_auxiliary_views, int)
            or max_auxiliary_views < 0
        ):
            raise EvidenceError(
                ReasonCode.TYPE_MISMATCH,
                "max_auxiliary_views 必须是 >= 0 的整数",
            )
        self.context_margin_ratio = float(context_margin_ratio)
        self.overlay_alpha = float(overlay_alpha)
        self.max_auxiliary_views = max_auxiliary_views
        self.min_auxiliary_coverage = float(min_auxiliary_coverage)

    def build(
        self,
        *,
        optical_image: Any,
        selected: SelectedRegion,
        mask_mode: MaskMode,
        output_root: Path,
        auxiliary_views: Sequence[AuxiliaryView] = (),
        rag_context: Sequence[Any] = (),
    ) -> EvidenceBuildResult:
        if mask_mode is MaskMode.EXTERNAL_GENERIC:
            raise EvidenceError(
                ReasonCode.EXTERNAL_MASK_FORBIDDEN,
                "External 记录不能进入 EvidenceBuilder",
            )
        if rag_context:
            raise EvidenceError(
                ReasonCode.RAG_FORBIDDEN,
                "Phase 3 不接受 RAG context",
            )
        if len(auxiliary_views) > self.max_auxiliary_views:
            raise EvidenceError(
                ReasonCode.AUXILIARY_LIMIT_EXCEEDED,
                "辅助视图数量超过配置上限，拒绝静默删除",
                details={
                    "actual": len(auxiliary_views),
                    "limit": self.max_auxiliary_views,
                },
            )
        root = Path(output_root)
        linked = first_symlink_component(root)
        if linked is not None:
            raise EvidenceError(
                ReasonCode.OUTPUT_LINK,
                f"Evidence output_root 含链接组件：{linked}",
            )
        if root.exists() or root.is_symlink():
            raise EvidenceError(
                ReasonCode.OUTPUT_EXISTS,
                f"Evidence output_root 已存在：{root}",
            )
        root.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(prefix=f".{root.name}.staging-", dir=root.parent)
        )
        try:
            optical = _rgb_image(optical_image, location="optical_image")
            width, height = optical.size
            if (
                selected.mask is not None
                and selected.mask.shape != (height, width)
            ):
                raise EvidenceError(
                    ReasonCode.MASK_SHAPE_MISMATCH,
                    "selected mask 与 optical image 尺寸不一致",
                )
            assets: list[EvidenceAsset] = []
            optical.save(staging / "optical_full.png", format="PNG")
            assets.append(
                EvidenceAsset(
                    evidence_id="ev_optical_full",
                    role="optical_full",
                    relative_path="optical_full.png",
                )
            )
            facts: dict[str, Any] = {}
            limitations: list[str] = []
            if selected.target_status is TargetStatus.TARGET_PRESENT:
                assert selected.mask is not None
                facts = deterministic_mask_facts(selected.mask)
                if facts["covariance_elongation"] is None:
                    limitations.append("DEGENERATE_COVARIANCE_ELONGATION")
                overlay = self._overlay(optical, selected.mask)
                overlay.save(staging / "mask_overlay.png", format="PNG")
                crop = self._context_crop(optical, selected.mask)
                crop.save(staging / "context_crop.png", format="PNG")
                assets.extend(
                    (
                        EvidenceAsset(
                            evidence_id="ev_mask_overlay",
                            role="mask_overlay",
                            relative_path="mask_overlay.png",
                        ),
                        EvidenceAsset(
                            evidence_id="ev_context_crop",
                            role="context_crop",
                            relative_path="context_crop.png",
                        ),
                    )
                )
            else:
                limitations.append("NO_TARGET_NO_REGION_GEOMETRY")
            auxiliary_metadata: list[dict[str, Any]] = []
            seen_aux_ids: set[str] = set()
            for index, view in enumerate(auxiliary_views):
                if view.evidence_id in seen_aux_ids or view.evidence_id.startswith(
                    "ev_optical"
                ):
                    raise EvidenceError(
                        ReasonCode.EVIDENCE_REFERENCE_INVALID,
                        f"重复 auxiliary evidence_id：{view.evidence_id}",
                    )
                seen_aux_ids.add(view.evidence_id)
                image = _rgb_image(
                    view.image,
                    location=f"auxiliary_views[{index}].image",
                )
                relative = f"auxiliary_{index:02d}.png"
                image.save(staging / relative, format="PNG")
                assets.append(
                    EvidenceAsset(
                        evidence_id=view.evidence_id,
                        role=f"auxiliary:{view.name}",
                        relative_path=relative,
                    )
                )
                metadata, view_limitations = self._auxiliary_contract(view)
                auxiliary_metadata.append(metadata)
                limitations.extend(view_limitations)
            bundle = EvidenceBundle(
                schema_version=EVIDENCE_SCHEMA_VERSION,
                sample_id=selected.sample_id,
                mask_mode=mask_mode,
                target_status=selected.target_status,
                selection={
                    "mode": selected.selection_mode.value,
                    "region_id": selected.region_id,
                    "source_identity": selected.source_identity,
                    "selector": (
                        "frozen_qwen3vl"
                        if selected.selection_mode.value
                        == "numbered_overlay"
                        else "deterministic_program"
                    ),
                },
                deterministic_facts=facts,
                assets=tuple(assets),
                auxiliary_metadata=tuple(auxiliary_metadata),
                limitations=tuple(sorted(set(limitations))),
                excluded_internal_signals=(
                    "attention",
                    "modality_quality_weight",
                    "region_feature",
                ),
                rag_context=(),
            )
            staging.replace(root)
            return EvidenceBuildResult(bundle=bundle, root=root)
        except BaseException:
            if staging.exists():
                shutil.rmtree(staging)
            raise

    def _overlay(self, optical: Image.Image, mask: np.ndarray) -> Image.Image:
        base = np.asarray(optical, dtype=np.float32).copy()
        red = np.zeros_like(base)
        red[..., 0] = 255
        base[mask] = (
            (1.0 - self.overlay_alpha) * base[mask]
            + self.overlay_alpha * red[mask]
        )
        return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))

    def _context_crop(
        self,
        optical: Image.Image,
        mask: np.ndarray,
    ) -> Image.Image:
        left, top, right, bottom = mask_bbox(mask)
        width = right - left
        height = bottom - top
        margin_x = math.ceil(width * self.context_margin_ratio)
        margin_y = math.ceil(height * self.context_margin_ratio)
        crop = (
            max(0, left - margin_x),
            max(0, top - margin_y),
            min(optical.width, right + margin_x),
            min(optical.height, bottom + margin_y),
        )
        return optical.crop(crop)

    def _auxiliary_contract(
        self,
        view: AuxiliaryView,
    ) -> tuple[dict[str, Any], list[str]]:
        limitations: list[str] = []
        if view.alignment is AlignmentStatus.UNKNOWN:
            limitations.append(f"AUX_ALIGNMENT_UNKNOWN:{view.name}")
        elif view.alignment is AlignmentStatus.APPROXIMATE:
            limitations.append(f"AUX_ALIGNMENT_APPROXIMATE:{view.name}")
        elif view.alignment is AlignmentStatus.UNREGISTERED:
            limitations.append(f"AUX_UNREGISTERED:{view.name}")
        if view.coverage is None:
            limitations.append(f"AUX_COVERAGE_UNKNOWN:{view.name}")
        elif view.coverage < self.min_auxiliary_coverage:
            limitations.append(f"AUX_COVERAGE_INSUFFICIENT:{view.name}")
        if view.unit is None:
            limitations.append(f"AUX_UNIT_UNKNOWN:{view.name}")
        if view.sign_convention is None:
            limitations.append(f"AUX_SIGN_CONVENTION_UNKNOWN:{view.name}")
        quantitative_allowed = (
            view.alignment is AlignmentStatus.REGISTERED
            and view.coverage is not None
            and view.coverage >= self.min_auxiliary_coverage
            and view.unit is not None
            and view.sign_convention is not None
        )
        return (
            {
                "evidence_id": view.evidence_id,
                "name": view.name,
                "alignment": view.alignment.value,
                "coverage": view.coverage,
                "unit": view.unit,
                "sign_convention": view.sign_convention,
                "quantitative_claims_allowed": quantitative_allowed,
                "directional_claims_allowed": (
                    quantitative_allowed and view.sign_convention is not None
                ),
            },
            limitations,
        )
