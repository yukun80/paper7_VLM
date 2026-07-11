#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""mask 可视化工具。

脚本作用：把验证 batch 输出成 RGB/GT/final/best proposal 四联图，
并额外导出 final/best/GT 二值 PNG，用于观察 PSALM-style proposal
decoder 是否真的在学习候选 mask。推理阶段可额外导出完整多模态 overview。
主要输入：dataloader batch 与 model outputs。
主要输出：PNG overlay 图。
是否改写原始数据：不会。
典型用法：save_visualizations(batch, outputs, out_dir, max_items=4, prefix="val")。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

from .data import (
    load_npy_array,
    normalize_modality,
    resize_pad_tensor,
    safe_slug,
)
from .indexing import canonical_modality_name


def _to_rgb(batch: dict[str, Any], sample_idx: int) -> np.ndarray:
    """读取 dataloader 已构造的稳定 preview。"""
    chosen = batch["visual_preview"][sample_idx].detach().cpu()
    if chosen.shape[0] == 1:
        rgb = chosen.repeat(3, 1, 1)
    elif chosen.shape[0] == 2:
        rgb = torch.cat([chosen, chosen[:1]], dim=0)
    else:
        rgb = chosen[:3]
    rgb = rgb.clamp(0, 1).permute(1, 2, 0).numpy()
    return (rgb * 255).astype(np.uint8)


def _tensor_to_rgb(tensor: torch.Tensor) -> np.ndarray:
    """把任意 CHW 模态张量转为便于人工检查的 RGB。"""
    x = tensor.detach().float().cpu()
    if float(x.min().item()) < 0.0:
        x = (x + 1.0) * 0.5
    x = x.clamp(0, 1)
    if x.shape[0] == 1:
        rgb = x.repeat(3, 1, 1)
    elif x.shape[0] == 2:
        rgb = torch.cat([x[:2], x[:2].mean(dim=0, keepdim=True)], dim=0)
    else:
        rgb = x[:3]
    return (rgb.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def _render_raw_modality(item: dict[str, Any], target_size: int) -> np.ndarray:
    """按 benchmark raw modality 元数据加载 .npy 并渲染为 target canvas RGB。"""
    raw_name = str(item.get("name") or "unknown")
    path = item.get("path")
    if not path:
        return np.zeros((target_size, target_size, 3), dtype=np.uint8)
    arr = load_npy_array(str(path))
    canonical = item.get("canonical") or canonical_modality_name(raw_name, item)
    tensor = normalize_modality(arr, item=item, raw_name=raw_name, canonical=str(canonical) if canonical else None)
    tensor, _ = resize_pad_tensor(tensor, target_size, mode="bilinear")
    return _tensor_to_rgb(tensor)


def _overlay_mask(rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.42) -> np.ndarray:
    """把二值 mask 以指定颜色叠到 RGB 上。"""
    out = rgb.astype(np.float32).copy()
    mask_bool = mask.astype(bool)
    if mask_bool.any():
        color_arr = np.asarray(color, dtype=np.float32).reshape(1, 1, 3)
        out[mask_bool] = (1.0 - alpha) * out[mask_bool] + alpha * color_arr
    return np.clip(out, 0, 255).astype(np.uint8)


def _label_panel(panel: np.ndarray, label: str) -> Image.Image:
    """给单个 panel 加一个轻量标题栏。"""
    image = Image.fromarray(panel)
    width, height = image.size
    canvas = Image.new("RGB", (width, height + 18), (18, 18, 18))
    canvas.paste(image, (0, 18))
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 3), label[:80], fill=(235, 235, 235))
    return canvas


def _compose_grid(panels: list[Image.Image], header_lines: list[str], max_cols: int = 4) -> Image.Image:
    """把多个 panel 组合成带 metadata header 的 overview 网格。"""
    if not panels:
        panels = [Image.new("RGB", (128, 146), (0, 0, 0))]
    cell_w = max(panel.size[0] for panel in panels)
    cell_h = max(panel.size[1] for panel in panels)
    cols = max(1, min(int(max_cols), len(panels)))
    rows = (len(panels) + cols - 1) // cols
    header_h = 20 + 16 * max(1, min(len(header_lines), 4))
    canvas = Image.new("RGB", (cols * cell_w, header_h + rows * cell_h), (12, 12, 12))
    draw = ImageDraw.Draw(canvas)
    y = 6
    for line in header_lines[:4]:
        draw.text((6, y), line[:180], fill=(235, 235, 235))
        y += 16
    for idx, panel in enumerate(panels):
        row = idx // cols
        col = idx % cols
        x0 = col * cell_w
        y0 = header_h + row * cell_h
        canvas.paste(panel, (x0, y0))
    return canvas


def _save_multimodal_overview(
    out_dir: Path,
    stem: str,
    meta: dict[str, Any],
    rgb: np.ndarray,
    gt: np.ndarray,
    final_pred: np.ndarray,
    best_pred: np.ndarray,
    best_query: int,
) -> str | None:
    """导出每个样本一张完整多模态总览图。"""
    target_size = int(final_pred.shape[-1])
    raw_items = list(meta.get("raw_modalities") or [])
    panels: list[Image.Image] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        try:
            raw_rgb = _render_raw_modality(item, target_size=target_size)
        except Exception as exc:  # noqa: BLE001 - 可视化失败不应阻断推理
            raw_rgb = np.zeros((target_size, target_size, 3), dtype=np.uint8)
            raw_rgb[:24, :, :] = 60
            label = f"{item.get('name', 'modality')} load_error={type(exc).__name__}"
        else:
            label = (
                f"{item.get('name', 'modality')} "
                f"[{item.get('sensor') or 'unknown'}|{item.get('normalization') or 'unknown'}]"
            )
        panels.append(_label_panel(raw_rgb, label))
    panels.extend(
        [
            _label_panel(rgb, "visual_preview"),
            _label_panel(_overlay_mask(rgb, gt, (0, 230, 80)), "GT"),
            _label_panel(_overlay_mask(rgb, final_pred, (255, 40, 40)), "Final"),
            _label_panel(_overlay_mask(rgb, best_pred, (255, 190, 40)), f"BestQ {best_query}"),
        ]
    )
    header = [
        f"sample={meta.get('sample_id')} dataset={meta.get('dataset_name')} template={meta.get('template_id')}",
        f"canonical={meta.get('canonical_combo')} raw={meta.get('raw_combo')}",
        f"sensor={meta.get('sensor_combo')} normalization={meta.get('normalization_combo')}",
        f"condition={meta.get('condition_prompt')} gsd={meta.get('gsd_m')} token={meta.get('gsd_token')}",
    ]
    overview = _compose_grid(panels, header_lines=header)
    path = out_dir / "multimodal_overviews" / f"{stem}_overview.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    overview.save(path)
    return path.as_posix()


def _best_query_index(outputs: dict[str, torch.Tensor], sample_idx: int) -> int:
    """优先使用 GT matching 的 best query；无监督时退化为 relevance 最大项。"""
    if "best_query" in outputs:
        return int(outputs["best_query"].detach().cpu()[sample_idx].item())
    return int(torch.argmax(outputs["proposal_relevance_logits"].detach().cpu()[sample_idx]).item())


def _selected_query_index(outputs: dict[str, torch.Tensor], sample_idx: int) -> int:
    """返回统一 semantic verifier 选中的 query。"""
    return int(torch.argmax(outputs["proposal_relevance_logits"].detach().cpu()[sample_idx]).item())


def _compose_diagnostic(
    rgb: np.ndarray,
    gt: np.ndarray,
    final_pred: np.ndarray,
    best_pred: np.ndarray,
    best_query: int,
) -> Image.Image:
    """生成 RGB、GT、final 和 best proposal 四联诊断图。"""
    panels = [
        _label_panel(rgb, "RGB"),
        _label_panel(_overlay_mask(rgb, gt, (0, 230, 80)), "GT"),
        _label_panel(_overlay_mask(rgb, final_pred, (255, 40, 40)), "Final"),
        _label_panel(_overlay_mask(rgb, best_pred, (255, 190, 40)), f"BestQ {best_query}"),
    ]
    width = sum(panel.size[0] for panel in panels)
    height = max(panel.size[1] for panel in panels)
    canvas = Image.new("RGB", (width, height), (0, 0, 0))
    x0 = 0
    for panel in panels:
        canvas.paste(panel, (x0, 0))
        x0 += panel.size[0]
    return canvas


def _save_binary_mask(path: Path, mask: np.ndarray) -> None:
    """保存 0/255 单通道 PNG。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(path)


def _save_mask_exports(
    out_dir: Path,
    stem: str,
    final_pred: np.ndarray,
    best_pred: np.ndarray,
    gt: np.ndarray,
) -> dict[str, str]:
    """导出单独二值 mask，便于下游检查或复用。"""
    export_dir = out_dir / "mask_exports"
    exports = {
        "final": final_pred,
        "best_proposal": best_pred,
        "gt": gt,
    }
    paths: dict[str, str] = {}
    for name, mask in exports.items():
        path = export_dir / name / f"{stem}_{name}.png"
        _save_binary_mask(path, mask)
        paths[name] = path.as_posix()
    return paths


def restore_mask_to_original(mask: np.ndarray, transform: dict[str, Any] | None) -> np.ndarray | None:
    """把 target canvas 上的 mask 反变换回原始 H/W。"""
    if not isinstance(transform, dict):
        return None
    source_hw = transform.get("source_hw")
    resized_hw = transform.get("resized_hw")
    if not isinstance(source_hw, list) or not isinstance(resized_hw, list) or len(source_hw) != 2 or len(resized_hw) != 2:
        return None
    src_h, src_w = int(source_hw[0]), int(source_hw[1])
    resized_h, resized_w = int(resized_hw[0]), int(resized_hw[1])
    if src_h <= 0 or src_w <= 0 or resized_h <= 0 or resized_w <= 0:
        return None
    pad_top = int(transform.get("pad_top", 0))
    pad_left = int(transform.get("pad_left", 0))
    crop = mask[pad_top : pad_top + resized_h, pad_left : pad_left + resized_w]
    if crop.size == 0:
        return None
    image = Image.fromarray(crop.astype(np.uint8) * 255)
    restored = image.resize((src_w, src_h), resample=Image.Resampling.NEAREST)
    return (np.asarray(restored) >= 128).astype(np.uint8)


def _restore_masks(masks: dict[str, np.ndarray], transform: dict[str, Any] | None) -> dict[str, np.ndarray]:
    """批量恢复 mask 到原始 H/W。"""
    restored_masks: dict[str, np.ndarray] = {}
    for name, mask in masks.items():
        restored = restore_mask_to_original(mask, transform)
        if restored is not None:
            restored_masks[name] = restored
    return restored_masks


def _save_restored_mask_exports(out_dir: Path, stem: str, restored_masks: dict[str, np.ndarray]) -> dict[str, str]:
    """导出恢复到原始 H/W 的二值 mask。"""
    export_dir = out_dir / "mask_exports_original_size"
    paths: dict[str, str] = {}
    for name, restored in restored_masks.items():
        path = export_dir / name / f"{stem}_{name}_original.png"
        _save_binary_mask(path, restored)
        paths[name] = path.as_posix()
    return paths


def _append_visualization_manifest(out_dir: Path, record: dict[str, Any]) -> None:
    """追加写出可视化/预测 mask 的样本级索引。"""
    path = out_dir / "visualization_manifest.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def _sample_modality_values(
    tensor: torch.Tensor | None,
    sample_idx: int,
    names: list[str],
) -> dict[str, float] | None:
    """把 [B,M] 模态权重/可用性张量转为可读 dict。"""
    if tensor is None:
        return None
    row = tensor.detach().float().cpu()[sample_idx]
    return {names[idx] if idx < len(names) else f"modality_{idx}": float(row[idx].item()) for idx in range(row.numel())}


def _sample_query_modality_values(
    tensor: torch.Tensor | None,
    sample_idx: int,
    names: list[str],
    query_idx: int | None = None,
) -> dict[str, float] | None:
    """把 [B,Q,M] query modality attention 转为样本级可读 dict。"""
    if tensor is None:
        return None
    values = tensor.detach().float().cpu()[sample_idx]
    if values.ndim != 2:
        return None
    row = values.mean(dim=0) if query_idx is None else values[int(query_idx)]
    return {names[idx] if idx < len(names) else f"modality_{idx}": float(row[idx].item()) for idx in range(row.numel())}


def _sample_query_score_values(
    outputs: dict[str, torch.Tensor],
    sample_idx: int,
    query_idx: int,
) -> dict[str, float | None]:
    """记录统一 semantic-evidence verifier 对某个 query 的 relevance。"""
    tensor = outputs.get("proposal_relevance_logits")
    return {
        "relevance_logit": (
            float(tensor.detach().float().cpu()[sample_idx, query_idx].item())
            if torch.is_tensor(tensor)
            else None
        )
    }


def save_visualizations(
    batch: dict[str, Any],
    outputs: dict[str, torch.Tensor],
    out_dir: Path,
    max_items: int,
    prefix: str,
    threshold: float = 0.5,
    export_multimodal_overview: bool = False,
) -> list[str]:
    """保存四联诊断图，并导出可复用二值 mask PNG。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    probs = torch.sigmoid(outputs["final_mask_logits"]).detach().cpu()
    proposal_probs = torch.sigmoid(outputs["proposal_mask_logits"]).detach().cpu()
    targets = batch["mask"].detach().cpu()
    valid_masks = batch["valid_mask"].detach().cpu()
    reliability_tensor = outputs.get("modality_reliability_weights")
    active_tensor = outputs.get("modality_active")
    query_attention_tensor = outputs.get("query_modality_attention")
    paths: list[str] = []
    n = min(max_items, probs.shape[0])
    for idx in range(n):
        rgb = _to_rgb(batch, idx)
        final_pred = (probs[idx, 0].numpy() >= float(threshold)).astype(np.uint8)
        gt = (targets[idx, 0].numpy() >= 0.5).astype(np.uint8)
        valid = (valid_masks[idx, 0].numpy() >= 0.5).astype(np.uint8)
        final_pred *= valid
        gt *= valid
        best_query = _best_query_index(outputs, idx)
        selected_query = _selected_query_index(outputs, idx)
        best_pred = (proposal_probs[idx, best_query].numpy() >= float(threshold)).astype(np.uint8)
        best_pred *= valid
        diagnostic = _compose_diagnostic(rgb, gt, final_pred, best_pred, best_query)

        meta = batch["metadata"][idx]
        modality_names = [str(item.get("name", f"modality_{j}")) for j, item in enumerate(meta.get("raw_modalities") or [])]
        stem = safe_slug(f"{prefix}_{idx}_{meta.get('sample_id', 'sample')}")
        overview_path = (
            _save_multimodal_overview(out_dir, stem, meta, rgb, gt, final_pred, best_pred, best_query)
            if export_multimodal_overview
            else None
        )
        mask_paths = _save_mask_exports(out_dir, stem, final_pred, best_pred, gt)
        masks_for_restore = {
            "final": final_pred,
            "best_proposal": best_pred,
            "gt": gt,
        }
        restored_masks = _restore_masks(masks_for_restore, meta.get("resize_transform"))
        restored_mask_paths = _save_restored_mask_exports(out_dir, stem, restored_masks)
        path = out_dir / f"{stem}.png"
        diagnostic.save(path)
        _append_visualization_manifest(
            out_dir,
            {
                "prefix": prefix,
                "batch_index": idx,
                "stem": stem,
                "diagnostic_path": path.as_posix(),
                "multimodal_overview_path": overview_path,
                "mask_paths": mask_paths,
                "restored_mask_paths": restored_mask_paths,
                "best_query": best_query,
                "selected_query": selected_query,
                "threshold": float(threshold),
                "mask_area": {
                    "final": int(final_pred.sum()),
                    "best_proposal": int(best_pred.sum()),
                    "gt": int(gt.sum()),
                    "valid_pixels": int(valid.sum()),
                },
                "restored_mask_area": {
                    name: int(mask.sum())
                    for name, mask in restored_masks.items()
                },
                "modality_reliability_weights": _sample_modality_values(reliability_tensor, idx, modality_names),
                "modality_active": _sample_modality_values(active_tensor, idx, modality_names),
                "query_modality_mean_attention": _sample_query_modality_values(query_attention_tensor, idx, modality_names),
                "query_modality_best_query_attention": _sample_query_modality_values(query_attention_tensor, idx, modality_names, best_query),
                "query_modality_selected_query_attention": _sample_query_modality_values(query_attention_tensor, idx, modality_names, selected_query),
                "best_query_scores": _sample_query_score_values(outputs, idx, best_query),
                "selected_query_scores": _sample_query_score_values(outputs, idx, selected_query),
                "metadata": {
                    "sample_id": meta.get("sample_id"),
                    "parent_sample_id": meta.get("parent_sample_id"),
                    "dataset_name": meta.get("dataset_name"),
                    "template_id": meta.get("template_id"),
                    "task_family": meta.get("task_family"),
                    "raw_combo": meta.get("raw_combo"),
                    "canonical_combo": meta.get("canonical_combo"),
                    "sensor_combo": meta.get("sensor_combo"),
                    "normalization_combo": meta.get("normalization_combo"),
                    "quality_flags": meta.get("quality_flags"),
                    "gsd_token": meta.get("gsd_token"),
                    "gsd_m": meta.get("gsd_m"),
                    "visual_preview_source": meta.get("visual_preview_source"),
                    "mask_original_size": meta.get("mask_original_size"),
                    "resize_transform": meta.get("resize_transform"),
                    "condition_prompt": meta.get("condition_prompt"),
                    "instruction": meta.get("instruction"),
                    "proposal_context_text": meta.get("proposal_context_text"),
                    "condition_prompt_text": meta.get("condition_prompt_text"),
                    "evidence_reasoning_text": meta.get("evidence_reasoning_text"),
                    "raw_modalities": meta.get("raw_modalities"),
                },
            },
        )
        paths.append(path.as_posix())
    return paths
