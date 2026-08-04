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


def render_evidence_image(
    value: Any, *, validity: np.ndarray | None = None, location: str = "evidence_image"
) -> Image.Image:
    """确定性渲染二维数值或 RGB 图；无效像素保持黑色。"""

    if validity is None:
        return _rgb_image(value, location=location)
    array = np.asarray(value)
    valid = np.asarray(validity)
    if array.ndim != 2 or valid.shape != array.shape:
        raise EvidenceError(ReasonCode.TYPE_MISMATCH, f"{location}: validity 必须与二维数值数组同 shape")
    if not np.issubdtype(array.dtype, np.number):
        raise EvidenceError(ReasonCode.TYPE_MISMATCH, f"{location}: 必须是数值数组")
    cleaned = array.astype(np.float64, copy=True)
    cleaned[~valid.astype(bool)] = np.nan
    return _rgb_image(cleaned, location=location)


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


def context_crop_window(
    mask: np.ndarray, *, image_size: tuple[int, int], margin_ratio: float = 0.15
) -> tuple[int, int, int, int]:
    """返回 target mask 的确定性半开 context crop 窗口。"""

    if not 0 <= float(margin_ratio) <= 1:
        raise EvidenceError(ReasonCode.TYPE_MISMATCH, "margin_ratio 必须位于 [0,1]")
    image_width, image_height = image_size
    if mask.shape != (image_height, image_width):
        raise EvidenceError(ReasonCode.MASK_SHAPE_MISMATCH, "mask 与 image_size 不一致")
    left, top, right, bottom = mask_bbox(mask)
    margin_x = math.ceil((right - left) * float(margin_ratio))
    margin_y = math.ceil((bottom - top) * float(margin_ratio))
    return (max(0, left - margin_x), max(0, top - margin_y),
            min(image_width, right + margin_x), min(image_height, bottom + margin_y))


def render_mask_overlay(optical: Image.Image, mask: np.ndarray, *, alpha: float = 0.45) -> Image.Image:
    """在已渲染 optical 上叠加红色 target mask。"""

    if not 0 <= float(alpha) <= 1:
        raise EvidenceError(ReasonCode.TYPE_MISMATCH, "alpha 必须位于 [0,1]")
    optical = optical.convert("RGB")
    if mask.shape != (optical.height, optical.width):
        raise EvidenceError(ReasonCode.MASK_SHAPE_MISMATCH, "mask 与 optical image 尺寸不一致")
    base = np.asarray(optical, dtype=np.float32).copy()
    red = np.zeros_like(base)
    red[..., 0] = 255
    selected = np.asarray(mask, dtype=bool)
    base[selected] = (1.0 - float(alpha)) * base[selected] + float(alpha) * red[selected]
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))


def render_context_crop(
    optical: Image.Image, mask: np.ndarray, *, margin_ratio: float = 0.15
) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """返回 context crop 及其在 full optical 上的半开窗口。"""

    optical = optical.convert("RGB")
    window = context_crop_window(mask, image_size=optical.size, margin_ratio=margin_ratio)
    return optical.crop(window), window


def render_binary_mask(mask: np.ndarray) -> Image.Image:
    """将内部 bool/0-1 mask 发布为严格 PNG-L 语义的 0/255 图像。

    二值 mask 必须独立于 RGB；这里不执行 alpha blending，也不携带类别或置信度。
    """

    array = np.asarray(mask)
    if array.ndim != 2:
        raise EvidenceError(ReasonCode.MASK_INVALID, "binary mask 必须是二维数组")
    if array.dtype == np.bool_:
        binary = array
    elif np.issubdtype(array.dtype, np.integer) and set(np.unique(array).tolist()).issubset({0, 1}):
        binary = array.astype(bool)
    else:
        raise EvidenceError(ReasonCode.MASK_INVALID, "binary mask 只允许 bool 或 0/1")
    return Image.fromarray(binary.astype(np.uint8) * 255)


def binary_mask_array(image: Image.Image) -> np.ndarray:
    """严格读取模型输入 mask，并在进入后续逻辑前转换为 bool。"""

    if image.mode != "L":
        raise EvidenceError(ReasonCode.MASK_INVALID, "发布 binary mask 必须是 PNG L 模式")
    array = np.asarray(image)
    if not set(np.unique(array).tolist()).issubset({0, 255}):
        raise EvidenceError(ReasonCode.MASK_INVALID, "发布 binary mask 像素必须严格为 0/255")
    return array == 255


def render_tight_target_crop(
    optical: Image.Image,
    mask: np.ndarray,
) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """构建 context-removal 专用 bbox-tight 无标记 RGB crop。"""

    optical = optical.convert("RGB")
    binary = np.asarray(mask, dtype=bool)
    if binary.shape != (optical.height, optical.width) or not bool(binary.any()):
        raise EvidenceError(ReasonCode.MASK_SHAPE_MISMATCH, "tight crop 需要同尺寸非空 mask")
    window = mask_bbox(binary)
    return optical.crop(window), window


def deterministic_shift_mask(mask: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """从固定 3x3 anchor 候选中选择低重叠且不越界的平移 mask。

    反事实只改变评价输入，不裁剪、不环绕，也绝不写回 GT mask。排序依次最小化
    mask IoU、最大化质心位移，再按 ``(dy, dx)`` 固定打破并列。
    """

    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2 or not bool(binary.any()):
        raise EvidenceError(ReasonCode.COUNTERFACTUAL_INVALID, "mask shift 需要非空二维 GT mask")
    height, width = binary.shape
    left, top, right, bottom = mask_bbox(binary)
    mask_width, mask_height = right - left, bottom - top
    x_positions = sorted({0, max(0, (width - mask_width) // 2), width - mask_width})
    y_positions = sorted({0, max(0, (height - mask_height) // 2), height - mask_height})
    original_area = int(binary.sum())
    original_centroid = mask_centroid(binary)
    candidates: list[tuple[float, float, int, int, np.ndarray]] = []
    for next_left in x_positions:
        for next_top in y_positions:
            dx, dy = next_left - left, next_top - top
            if dx == 0 and dy == 0:
                continue
            shifted = np.zeros_like(binary)
            ys, xs = np.nonzero(binary)
            next_xs, next_ys = xs + dx, ys + dy
            if (
                int(next_xs.min()) < 0
                or int(next_ys.min()) < 0
                or int(next_xs.max()) >= width
                or int(next_ys.max()) >= height
            ):
                continue
            shifted[next_ys, next_xs] = True
            intersection = int(np.count_nonzero(binary & shifted))
            union = original_area * 2 - intersection
            iou = 0.0 if union == 0 else float(intersection / union)
            next_centroid = mask_centroid(shifted)
            distance = math.hypot(
                next_centroid[0] - original_centroid[0],
                next_centroid[1] - original_centroid[1],
            )
            candidates.append((iou, -distance, dy, dx, shifted))
    if not candidates:
        raise EvidenceError(ReasonCode.COUNTERFACTUAL_INVALID, "GT mask 没有合法的非零 anchor shift")
    iou, negative_distance, dy, dx, shifted = min(candidates, key=lambda item: item[:4])
    if int(shifted.sum()) != original_area:
        raise EvidenceError(ReasonCode.COUNTERFACTUAL_INVALID, "mask shift 意外改变面积")
    return shifted, {
        "algorithm": "mask_shift_3x3_anchor_min_iou.v1",
        "dx_pixels": int(dx),
        "dy_pixels": int(dy),
        "mask_iou_with_gt": float(iou),
        "centroid_distance_pixels": float(-negative_distance),
        "evaluation_only": True,
    }


def boundary_contrast_proxy(optical: Image.Image, mask: np.ndarray) -> float:
    """计算一像素 mask 内外边界环的 RGB 差异，仅作为选样 proxy。

    该数值不是专家“清晰/困难”标签，不能进入地质或风险结论。
    """

    rgb = np.asarray(optical.convert("RGB"), dtype=np.float64) / 255.0
    binary = np.asarray(mask, dtype=bool)
    if binary.shape != rgb.shape[:2] or not bool(binary.any()):
        raise EvidenceError(ReasonCode.MASK_SHAPE_MISMATCH, "boundary proxy 需要同尺寸非空 mask")
    padded = np.pad(binary, 1, constant_values=False)
    neighbors = (
        padded[:-2, 1:-1], padded[2:, 1:-1], padded[1:-1, :-2], padded[1:-1, 2:],
        padded[:-2, :-2], padded[:-2, 2:], padded[2:, :-2], padded[2:, 2:],
    )
    erosion = binary.copy()
    dilation = binary.copy()
    for neighbor in neighbors:
        erosion &= neighbor
        dilation |= neighbor
    inner = binary & ~erosion
    outer = dilation & ~binary
    if not bool(inner.any()) or not bool(outer.any()):
        return 0.0
    inside = rgb[inner].mean(axis=0)
    outside = rgb[outer].mean(axis=0)
    return float(np.abs(inside - outside).mean())


def deterministic_auxiliary_facts(
    values: np.ndarray,
    pixel_valid: np.ndarray,
    channel_valid: np.ndarray,
    *,
    channel_names: Sequence[str],
    mask: np.ndarray | None,
    min_coverage: float = 0.8,
) -> dict[str, Any]:
    """按 OA validity 计算 full/mask/outside 编码值统计。"""

    array = np.asarray(values)
    validity = np.asarray(pixel_valid)
    channels = np.asarray(channel_valid)
    names = tuple(str(name) for name in channel_names)
    if (array.ndim != 3 or validity.shape != array.shape
            or channels.shape != (array.shape[0],) or len(names) != array.shape[0]):
        raise EvidenceError(ReasonCode.TYPE_MISMATCH,
                            "auxiliary values/pixel_valid/channel_valid/channel_names 合同不一致")
    if not 0 <= float(min_coverage) <= 1:
        raise EvidenceError(ReasonCode.TYPE_MISMATCH, "min_coverage 必须位于 [0,1]")
    height, width = array.shape[-2:]
    if mask is not None and np.asarray(mask).shape != (height, width):
        raise EvidenceError(ReasonCode.MASK_SHAPE_MISMATCH, "auxiliary 与 mask shape 不一致")
    effective = validity.astype(bool) & channels.astype(bool)[:, None, None] & np.isfinite(array)

    def summarize(scope: np.ndarray) -> dict[str, Any]:
        scope = np.asarray(scope, dtype=bool)
        denominator = int(scope.sum())
        aggregate = effective.any(axis=0) & scope
        coverage = 0.0 if denominator == 0 else float(aggregate.sum() / denominator)
        status = ("not_applicable" if denominator == 0 else "unavailable"
                  if not bool(aggregate.any()) else "insufficient_coverage"
                  if coverage < float(min_coverage) else "available")
        rows: list[dict[str, Any]] = []
        for index, name in enumerate(names):
            selected = effective[index] & scope
            count = int(selected.sum())
            channel_coverage = 0.0 if denominator == 0 else float(count / denominator)
            statistics: dict[str, Any] | None = None
            if count > 0 and channel_coverage >= float(min_coverage):
                data = array[index][selected].astype(np.float64)
                statistics = {"count": count, "mean": float(data.mean()),
                              "median": float(np.median(data)),
                              "p25": float(np.percentile(data, 25, method="linear")),
                              "p75": float(np.percentile(data, 75, method="linear")),
                              "min": float(data.min()), "max": float(data.max())}
            rows.append({"name": name, "valid_count": count, "coverage": channel_coverage,
                         "statistics": statistics})
        return {
            "status": status, "scope_pixel_count": denominator,
            "valid_pixel_count": int(aggregate.sum()), "coverage": coverage, "channels": rows,
        }

    full = summarize(np.ones((height, width), dtype=bool))
    if mask is None:
        inside = outside = None
    else:
        binary = np.asarray(mask, dtype=bool)
        inside = summarize(binary)
        outside = summarize(~binary)
    differences: dict[str, float | None] | None = None
    if inside is not None and outside is not None:
        differences = {}
        for name, inner, outer in zip(names, inside["channels"], outside["channels"], strict=True):
            inner_stats = inner["statistics"]
            outer_stats = outer["statistics"]
            differences[name] = (None if inner_stats is None or outer_stats is None
                                 else float(inner_stats["mean"] - outer_stats["mean"]))
    return {
        "status": full["status"], "full": full, "mask": inside, "outside": outside,
        "inside_minus_outside_mean": differences,
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
                overlay = render_mask_overlay(
                    optical,
                    selected.mask,
                    alpha=self.overlay_alpha,
                )
                overlay.save(staging / "mask_overlay.png", format="PNG")
                crop, _ = render_context_crop(
                    optical,
                    selected.mask,
                    margin_ratio=self.context_margin_ratio,
                )
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
        return render_mask_overlay(
            optical,
            mask,
            alpha=self.overlay_alpha,
        )

    def _context_crop(
        self,
        optical: Image.Image,
        mask: np.ndarray,
    ) -> Image.Image:
        crop, _ = render_context_crop(
            optical,
            mask,
            margin_ratio=self.context_margin_ratio,
        )
        return crop

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
