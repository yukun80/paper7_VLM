#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""多源滑坡 benchmark 指代目标公共规则库。

脚本作用：集中维护方位、尺度、形态、数量四类 rule-based referring target
生成规则，供 1-6_build_referring_targets.py 调用。
主要输入：已物化的二值 mask、父样本元数据和可选 preview 底图。
主要输出：referring_targets、target-level mask.npy 和 referring preview。
是否改写原始数据：不会读取或改写 datasets/；只由调用方传入 benchmark 输出目录。
典型用法：由 1-6_build_referring_targets.py import 后复用，不建议单独运行。
"""

from __future__ import annotations

import copy
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from geohazard_benchmark_common import ensure_dir, to_repo_rel


REFERRING_MAX_EXPRESSIONS = 8
REFERRING_SCENE_COMPONENT_PIXEL_LIMIT = 2048 * 2048
REFERRING_TILE_SIDE = 320
REFERRING_POSITION_ORDER = [
    "upper-left",
    "upper",
    "upper-right",
    "left",
    "center",
    "right",
    "lower-left",
    "lower",
    "lower-right",
]
REFERRING_POSITION_ZH = {
    "upper-left": "左上部",
    "upper": "上部",
    "upper-right": "右上部",
    "left": "左侧",
    "center": "中部",
    "right": "右侧",
    "lower-left": "左下部",
    "lower": "下部",
    "lower-right": "右下部",
}


def build_referring_config() -> dict[str, Any]:
    """返回可写入 preprocess_config.yaml 的指代目标规则配置。"""
    return {
        "enabled": True,
        "max_targets_per_sample": REFERRING_MAX_EXPRESSIONS,
        "component_connectivity": "8-neighborhood",
        "large_scene_component_pixel_limit": REFERRING_SCENE_COMPONENT_PIXEL_LIMIT,
        "position_rule": "连通域质心落入 3x3 网格后生成方位指代目标，每样本最多保留面积最大的 4 个方位组。",
        "scale_rules": {
            "large_landslide": "largest_area / image_area >= 0.005 或 largest_area / total_landslide_area >= 0.5",
            "small_landslide_patches": "area <= 0.25 * largest_area 且至少 2 个小斑块",
        },
        "morphology_rules": {
            "compact_landslide": "largest_fill_ratio >= 0.35 且 largest_area / total_landslide_area >= 0.75",
            "fragmented_landslides": "component_count >= 3 且 largest_area / total_landslide_area <= 0.85",
            "elongated_landslide": "largest_bbox_aspect_ratio >= 3.0",
        },
        "count_rules": {
            "single_landslide": "component_count == 1",
            "multiple_landslides": "2 <= component_count <= 6",
            "many_landslides": "component_count > 6",
        },
        "note": "指代 target mask 是局部监督目标，不等同于父样本全局 mask；训练文本由 scripts/2-instruction 统一渲染。",
    }


def to_chw(array: Any) -> Any:
    """把 HWC/RGB preview 转为 CHW，便于复用简单视觉处理。"""
    arr = np.asarray(array)
    if arr.ndim == 2:
        return arr[np.newaxis, :, :]
    if arr.ndim == 3 and arr.shape[-1] <= 4:
        return np.transpose(arr[..., :3], (2, 0, 1))
    return arr


def safe_referring_id(value: str) -> str:
    """把 category/subtype 转成稳定的文件夹友好 ID。"""
    return re.sub(r"[^0-9A-Za-z]+", "_", value).strip("_").lower()


def mask_to_2d(mask: Any) -> Any:
    """把 [1,H,W] 或 [H,W] mask 转成二维二值数组。"""
    arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = arr[0]
    return (arr > 0).astype(np.uint8)


def bbox_from_binary_mask(mask: Any) -> tuple[list[int] | None, str, int, bool]:
    """从二值 mask 计算 bbox、正样本像素数和空 mask 标记。"""
    arr = mask_to_2d(mask)
    ys, xs = np.where(arr > 0)
    positive = int(xs.size)
    if positive == 0:
        return None, "empty_mask", 0, True
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())], "derived", positive, False


def position_from_point(x: float, y: float, width: int, height: int) -> str:
    """根据点落入 3x3 网格的位置生成方位标签。"""
    col = 0 if x < width / 3.0 else (1 if x < 2.0 * width / 3.0 else 2)
    row = 0 if y < height / 3.0 else (1 if y < 2.0 * height / 3.0 else 2)
    return REFERRING_POSITION_ORDER[row * 3 + col]


def analyze_mask_components(binary: Any) -> tuple[Any, list[dict[str, Any]]]:
    """使用 scipy.ndimage.label 做 8 邻域连通域分析。"""
    structure = np.ones((3, 3), dtype=np.uint8)
    labels, num_labels = ndimage.label(binary > 0, structure=structure)
    objects = ndimage.find_objects(labels)
    components: list[dict[str, Any]] = []
    for label_id in range(1, int(num_labels) + 1):
        slices = objects[label_id - 1]
        if slices is None:
            continue
        y_slice, x_slice = slices
        sub = labels[y_slice, x_slice] == label_id
        ys, xs = np.where(sub)
        area = int(xs.size)
        if area <= 0:
            continue
        x1 = int(x_slice.start)
        y1 = int(y_slice.start)
        x2 = int(x_slice.stop - 1)
        y2 = int(y_slice.stop - 1)
        width = max(1, x2 - x1 + 1)
        height = max(1, y2 - y1 + 1)
        components.append({
            "label_id": label_id,
            "area": area,
            "bbox_xyxy": [x1, y1, x2, y2],
            "centroid_xy": [float(xs.mean() + x1), float(ys.mean() + y1)],
            "fill_ratio": float(area / float(width * height)),
            "bbox_aspect_ratio": float(max(width / height, height / width)),
        })
    components.sort(key=lambda item: item["area"], reverse=True)
    return labels, components


def component_key(label_ids: list[int]) -> str:
    """生成可复用 target mask 的规范 key。"""
    return "components:" + ",".join(str(idx) for idx in sorted(set(label_ids)))


def target_mask_from_key(binary: Any, labels: Any | None, target_key: str) -> Any:
    """根据 target_key 生成 target-level 目标 mask。"""
    if target_key == "full":
        return binary[np.newaxis, :, :].astype(np.uint8)
    if labels is None or not target_key.startswith("components:"):
        raise ValueError(f"无法解析指代目标: {target_key}")
    ids = [int(part) for part in target_key.split(":", 1)[1].split(",") if part]
    return np.isin(labels, ids).astype(np.uint8)[np.newaxis, :, :]


def make_referring_candidate(
    *,
    category: str,
    subtype: str,
    target_key: str,
    grounding: dict[str, Any],
    confidence: str,
    priority: int,
    quality_flags: list[str] | None = None,
) -> dict[str, Any]:
    """构造待筛选的结构化指代目标候选。"""
    return {
        "category": category,
        "subtype": subtype,
        "target_key": target_key,
        "grounding": grounding,
        "confidence": confidence,
        "priority": priority,
        "quality_flags": sorted(set(quality_flags or [])),
    }


def select_referring_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """先保证类别多样性，再按优先级填满每样本最多 8 条表达。"""
    selected: list[dict[str, Any]] = []
    selected_keys: set[tuple[str, str, str]] = set()

    def add(candidate: dict[str, Any]) -> None:
        key = (str(candidate["category"]), str(candidate["subtype"]), str(candidate["target_key"]))
        if key in selected_keys or len(selected) >= REFERRING_MAX_EXPRESSIONS:
            return
        selected.append(candidate)
        selected_keys.add(key)

    sorted_candidates = sorted(candidates, key=lambda item: (int(item["priority"]), str(item["category"]), str(item["subtype"])))
    for category in ["position", "scale", "morphology", "count"]:
        options = [item for item in sorted_candidates if item["category"] == category]
        if options:
            add(options[0])
    for candidate in sorted_candidates:
        add(candidate)
    return selected


def build_component_referring_candidates(binary: Any, labels: Any, components: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """基于连通域统计生成方位、尺度、形态、数量四类表达候选。"""
    height, width = binary.shape
    image_area = int(height * width)
    total_area = int(binary.sum())
    if total_area <= 0 or not components:
        return []

    candidates: list[dict[str, Any]] = []
    largest = components[0]
    largest_area = int(largest["area"])
    largest_share = float(largest_area / max(total_area, 1))

    position_groups: dict[str, dict[str, Any]] = {}
    for comp in components:
        cx, cy = comp["centroid_xy"]
        pos = position_from_point(cx, cy, width, height)
        group = position_groups.setdefault(pos, {"area": 0, "labels": [], "bboxes": []})
        group["area"] += int(comp["area"])
        group["labels"].append(int(comp["label_id"]))
        group["bboxes"].append(comp["bbox_xyxy"])
    ordered_positions = sorted(position_groups.items(), key=lambda item: item[1]["area"], reverse=True)[:4]
    for idx, (pos, group) in enumerate(ordered_positions):
        candidates.append(make_referring_candidate(
            category="position",
            subtype=pos,
            target_key=component_key(group["labels"]),
            grounding={
                "rule": "component_centroid_3x3_grid",
                "grid": pos,
                "component_labels": group["labels"],
                "component_count": len(group["labels"]),
                "group_area_pixels": int(group["area"]),
                "bboxes_xyxy": group["bboxes"],
            },
            confidence="rule_high",
            priority=10 + idx,
        ))

    largest_key = component_key([int(largest["label_id"])])
    candidates.append(make_referring_candidate(
        category="scale",
        subtype="largest_landslide",
        target_key=largest_key,
        grounding={
            "rule": "largest_connected_component",
            "component_labels": [int(largest["label_id"])],
            "area_pixels": largest_area,
            "area_ratio_image": float(largest_area / max(image_area, 1)),
            "area_ratio_landslide": largest_share,
            "bbox_xyxy": largest["bbox_xyxy"],
        },
        confidence="rule_high",
        priority=20,
    ))
    if largest_area / max(image_area, 1) >= 0.005 or largest_share >= 0.5:
        candidates.append(make_referring_candidate(
            category="scale",
            subtype="large_landslide",
            target_key=largest_key,
            grounding={
                "rule": "large_component_threshold",
                "component_labels": [int(largest["label_id"])],
                "area_pixels": largest_area,
                "area_ratio_image": float(largest_area / max(image_area, 1)),
                "area_ratio_landslide": largest_share,
            },
            confidence="rule_high",
            priority=21,
        ))
    small_components = [comp for comp in components if int(comp["area"]) <= 0.25 * largest_area]
    if len(small_components) >= 2:
        labels_small = [int(comp["label_id"]) for comp in small_components]
        candidates.append(make_referring_candidate(
            category="scale",
            subtype="small_landslide_patches",
            target_key=component_key(labels_small),
            grounding={
                "rule": "small_components_relative_to_largest",
                "component_labels": labels_small,
                "component_count": len(labels_small),
                "largest_area_pixels": largest_area,
                "small_area_threshold_pixels": float(0.25 * largest_area),
            },
            confidence="rule_medium",
            priority=22,
        ))

    if float(largest["fill_ratio"]) >= 0.35 and largest_share >= 0.75:
        candidates.append(make_referring_candidate(
            category="morphology",
            subtype="compact_landslide",
            target_key=largest_key,
            grounding={
                "rule": "largest_component_fill_ratio",
                "component_labels": [int(largest["label_id"])],
                "fill_ratio": float(largest["fill_ratio"]),
                "area_ratio_landslide": largest_share,
            },
            confidence="rule_medium",
            priority=30,
        ))
    if len(components) >= 3 and largest_share <= 0.85:
        candidates.append(make_referring_candidate(
            category="morphology",
            subtype="fragmented_landslides",
            target_key="full",
            grounding={
                "rule": "many_components_with_limited_largest_share",
                "component_count": len(components),
                "largest_area_ratio_landslide": largest_share,
            },
            confidence="rule_medium",
            priority=31,
        ))
    if float(largest["bbox_aspect_ratio"]) >= 3.0:
        candidates.append(make_referring_candidate(
            category="morphology",
            subtype="elongated_landslide",
            target_key=largest_key,
            grounding={
                "rule": "largest_component_bbox_aspect_ratio",
                "component_labels": [int(largest["label_id"])],
                "bbox_aspect_ratio": float(largest["bbox_aspect_ratio"]),
                "bbox_xyxy": largest["bbox_xyxy"],
            },
            confidence="rule_medium",
            priority=32,
        ))

    comp_count = len(components)
    if comp_count == 1:
        subtype = "single_landslide"
    elif comp_count <= 6:
        subtype = "multiple_landslides"
    else:
        subtype = "many_landslides"
    candidates.append(make_referring_candidate(
        category="count",
        subtype=subtype,
        target_key="full",
        grounding={
            "rule": "connected_component_count",
            "component_count": comp_count,
            "total_landslide_area_pixels": total_area,
        },
        confidence="rule_high",
        priority=40,
    ))
    return candidates


def build_large_scene_referring_candidates(binary: Any, final_mask: dict[str, Any]) -> list[dict[str, Any]]:
    """对超大 scene 只用全局 bbox 派生低成本方位表达，不做连通域分析。"""
    bbox = final_mask.get("bbox_xyxy") if final_mask else None
    if not bbox:
        return []
    height, width = binary.shape
    x1, y1, x2, y2 = bbox
    pos = position_from_point((float(x1) + float(x2)) / 2.0, (float(y1) + float(y2)) / 2.0, width, height)
    return [make_referring_candidate(
        category="position",
        subtype=pos,
        target_key="full",
        grounding={"rule": "global_bbox_3x3_grid_large_scene", "grid": pos, "bbox_xyxy": bbox},
        confidence="rule_medium",
        priority=10,
        quality_flags=["referring_component_analysis_skipped_large_scene"],
    )]


def resize_preview(image: Any, max_side: int = REFERRING_TILE_SIDE) -> Image.Image:
    """按最大边缩小 preview，避免超大 PNG。"""
    pil = Image.fromarray(np.asarray(image, dtype=np.uint8))
    width, height = pil.size
    side = max(width, height)
    if side <= max_side:
        return pil
    scale = max_side / float(side)
    size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return pil.resize(size, Image.Resampling.BILINEAR)


def chw_to_hwc_rgb(arr: Any) -> Any:
    """把 CHW/HW 数组转为 HWC RGB uint8。"""
    data = np.asarray(arr)
    if data.ndim == 2:
        gray = data.astype(np.uint8)
        return np.stack([gray, gray, gray], axis=-1)
    if data.ndim == 3:
        if data.shape[0] == 1:
            gray = data[0].astype(np.uint8)
            return np.stack([gray, gray, gray], axis=-1)
        return np.transpose(data[:3].astype(np.uint8), (1, 2, 0))
    raise ValueError(f"无法转为 RGB preview: shape={data.shape}")


def overlay_preview_resized(visual: Any, mask: Any, max_side: int = REFERRING_TILE_SIDE) -> Any:
    """先缩小再叠加 mask，避免 scene-level 大图生成 preview 时占用过高内存。"""
    base_pil = resize_preview(chw_to_hwc_rgb(visual), max_side=max_side).convert("RGB")
    mask_pil = Image.fromarray(mask_to_2d(mask) * 255).resize(base_pil.size, Image.Resampling.NEAREST)
    base = np.asarray(base_pil, dtype=np.uint8).copy()
    binary = np.asarray(mask_pil) > 0
    red = np.zeros_like(base)
    red[..., 0] = 255
    base[binary] = (0.55 * base[binary] + 0.45 * red[binary]).astype(np.uint8)
    return base


def labeled_tile(name: str, image: Any) -> Image.Image:
    """给 referring preview tile 添加短标签。"""
    tile = Image.fromarray(np.asarray(image, dtype=np.uint8)).convert("RGB")
    canvas = Image.new("RGB", (tile.width, tile.height + 24), color=(0, 0, 0))
    canvas.paste(tile, (0, 24))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 5), name, fill=(255, 255, 255))
    return canvas


def save_referring_preview(path: Path, visual: Any | None, referring_targets: list[tuple[str, Any]]) -> str | None:
    """保存指代 target mask 拼图，便于人工核查目标区域是否合理。"""
    if visual is None or not referring_targets:
        return None
    pil_tiles = [labeled_tile(label, overlay_preview_resized(visual, target)) for label, target in referring_targets[:REFERRING_MAX_EXPRESSIONS]]
    cols = min(3, len(pil_tiles))
    rows = int(math.ceil(len(pil_tiles) / cols))
    cell_w = max(tile.width for tile in pil_tiles)
    cell_h = max(tile.height for tile in pil_tiles)
    grid = Image.new("RGB", (cell_w * cols, cell_h * rows), color=(20, 20, 20))
    for idx, tile in enumerate(pil_tiles):
        grid.paste(tile, ((idx % cols) * cell_w, (idx // cols) * cell_h))
    ensure_dir(path.parent)
    grid.save(path)
    return to_repo_rel(path) or path.as_posix()


def build_referring_targets(
    sample: dict[str, Any],
    sample_dir: Path,
    visual: Any | None,
    mask: Any | None,
    final_mask: dict[str, Any] | None,
    *,
    enable_preview: bool = True,
) -> tuple[list[dict[str, Any]], str | None, list[str], list[str]]:
    """生成结构化指代目标、target-level mask 和 referring preview。"""
    if mask is None or not final_mask or final_mask.get("empty_mask") is True:
        return [], None, [], []

    binary = mask_to_2d(mask)
    if int(binary.sum()) <= 0:
        return [], None, [], []

    flags: list[str] = []
    errors: list[str] = []
    labels = None
    if sample.get("source_level") == "scene" and binary.size > REFERRING_SCENE_COMPONENT_PIXEL_LIMIT:
        candidates = build_large_scene_referring_candidates(binary, final_mask)
        flags.append("referring_component_analysis_skipped_large_scene")
    else:
        labels, components = analyze_mask_components(binary)
        candidates = build_component_referring_candidates(binary, labels, components)

    selected = select_referring_candidates(candidates)
    if not selected:
        return [], None, errors, flags

    referring_dir = ensure_dir(sample_dir / "referring")
    target_cache: dict[str, dict[str, Any]] = {}
    target_preview_items: list[tuple[str, Any]] = []
    targets: list[dict[str, Any]] = []

    def target_entry_for(target_key: str, target_id: str) -> tuple[dict[str, Any], Any]:
        if target_key in target_cache:
            target_mask = target_mask_from_key(binary, labels, target_key)
            return copy.deepcopy(target_cache[target_key]), target_mask
        target_mask = target_mask_from_key(binary, labels, target_key)
        bbox, bbox_status, positive, empty_mask = bbox_from_binary_mask(target_mask)
        if target_key == "full":
            entry = copy.deepcopy(final_mask)
            entry["target_source"] = "parent_mask"
            entry["area_ratio"] = float(positive / max(binary.size, 1))
        else:
            out_path = referring_dir / target_id / "mask.npy"
            ensure_dir(out_path.parent)
            np.save(out_path, target_mask.astype(np.uint8))
            entry = {
                "path": to_repo_rel(out_path),
                "format": "npy",
                "internal_key": None,
                "label_type": "binary_landslide_referring_target",
                "shape": list(target_mask.shape),
                "dtype": "uint8",
                "positive_pixels": positive,
                "empty_mask": empty_mask,
                "bbox_xyxy": bbox,
                "bbox_status": bbox_status,
                "binarize_rule": "mask > 0",
                "target_source": "rule_based_connected_components",
                "area_ratio": float(positive / max(binary.size, 1)),
            }
        entry["target_key"] = target_key
        target_cache[target_key] = copy.deepcopy(entry)
        return entry, target_mask

    for idx, candidate in enumerate(selected, start=1):
        target_id = f"ref_{idx:02d}_{safe_referring_id(candidate['category'])}_{safe_referring_id(candidate['subtype'])}"
        target_entry, target_mask = target_entry_for(str(candidate["target_key"]), target_id)
        target_flags = sorted(set((candidate.get("quality_flags") or []) + ["referring_target_rule_generated"]))
        targets.append({
            "target_id": target_id,
            "category": candidate["category"],
            "subtype": candidate["subtype"],
            "target_mask": target_entry,
            "grounding": {**(candidate.get("grounding") or {}), "generator": "rule_based_mask_components_v1"},
            "confidence": candidate["confidence"],
            "quality_flags": target_flags,
        })
        target_preview_items.append((f"{candidate['category']}:{candidate['subtype']}", target_mask))
        flags.extend(target_flags)

    preview_path = None
    if enable_preview:
        try:
            preview_path = save_referring_preview(sample_dir / "preview" / "referring.png", visual, target_preview_items)
        except Exception as exc:
            errors.append(f"referring preview 失败: {exc}")
            flags.append("referring_preview_failed")
    return targets, preview_path, errors, sorted(set(flags))
