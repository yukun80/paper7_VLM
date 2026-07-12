#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""mask 可视化工具。

脚本作用：把验证 batch 输出成 RGB/GT/final/selected/oracle proposal 诊断图，
并导出对应二值 PNG。oracle proposal 由 GT assignment 产生，仅用于测量
proposal 上限与 verifier selection gap，不属于可部署模型输出。
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
    resize_pad_tensor,
)
from .schema import ModalityInstance


def safe_slug(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in value)[:180]


def _to_rgb(batch: dict[str, Any], sample_idx: int) -> np.ndarray:
    """从活动模态中构造诊断 RGB；该图不进入模型 forward。"""
    instances = batch["instances"][sample_idx]
    chosen_item = next((item for item in instances if item.family in {"optical", "multispectral"}), instances[0])
    chosen = chosen_item.image.detach().cpu()
    chosen, _ = resize_pad_tensor(chosen, batch["mask"].shape[-1], mode="bilinear")
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


def _render_modality(item: ModalityInstance, target_size: int) -> np.ndarray:
    tensor, _ = resize_pad_tensor(item.image.detach().cpu(), target_size, mode="bilinear")
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
    selected_pred: np.ndarray,
    selected_query: int,
    oracle_pred: np.ndarray,
    oracle_query: int,
    instances: list[ModalityInstance],
) -> str | None:
    """导出每个样本一张完整多模态总览图。"""
    target_size = int(final_pred.shape[-1])
    panels: list[Image.Image] = []
    for item in instances:
        try:
            raw_rgb = _render_modality(item, target_size=target_size)
        except Exception as exc:  # noqa: BLE001 - 可视化失败不应阻断推理
            raw_rgb = np.zeros((target_size, target_size, 3), dtype=np.uint8)
            raw_rgb[:24, :, :] = 60
            label = f"{item.name} load_error={type(exc).__name__}"
        else:
            label = f"{item.name} [{item.sensor}|{item.product_type}|{item.units}]"
        panels.append(_label_panel(raw_rgb, label))
    panels.extend(
        [
            _label_panel(rgb, "reference view"),
            _label_panel(_overlay_mask(rgb, gt, (0, 230, 80)), "GT"),
            _label_panel(_overlay_mask(rgb, final_pred, (255, 40, 40)), "Final"),
            _label_panel(_overlay_mask(rgb, selected_pred, (255, 190, 40)), f"SelectedQ {selected_query}"),
            _label_panel(
                _overlay_mask(rgb, oracle_pred, (60, 150, 255)),
                f"OracleMatchedQ {oracle_query}" if oracle_query >= 0 else "OracleMatchedQ none",
            ),
        ]
    )
    header = [
        f"sample={meta.get('sample_id')} dataset={meta.get('dataset_name')} template={meta.get('template_id')}",
        f"families={meta.get('family_combo')} active={meta.get('active_subset')}",
        f"sensor={meta.get('sensor_combo')} gsd={meta.get('gsd_m')}",
        f"instruction={meta.get('instruction')}",
    ]
    overview = _compose_grid(panels, header_lines=header)
    path = out_dir / "multimodal_overviews" / f"{stem}_overview.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    overview.save(path)
    return path.as_posix()


def _selected_query_index(outputs: dict[str, torch.Tensor], sample_idx: int) -> int:
    """返回统一 semantic verifier 选中的 query。"""
    return int(torch.argmax(outputs["proposal_relevance_logits"].detach().cpu()[sample_idx]).item())


def _oracle_matched_query_index(outputs: dict[str, torch.Tensor], sample_idx: int) -> int:
    """返回 GT assignment 中 Dice 最高的 matched query；空目标为 -1。"""
    values = outputs.get("proposal_oracle_matched_query")
    return int(values.detach().long().cpu()[sample_idx].item()) if torch.is_tensor(values) else -1


def _compose_diagnostic(
    rgb: np.ndarray,
    gt: np.ndarray,
    final_pred: np.ndarray,
    selected_pred: np.ndarray,
    selected_query: int,
    oracle_pred: np.ndarray,
    oracle_query: int,
) -> Image.Image:
    """生成 final、模型选中 proposal 与仅供诊断的 GT-oracle proposal。"""
    panels = [
        _label_panel(rgb, "RGB"),
        _label_panel(_overlay_mask(rgb, gt, (0, 230, 80)), "GT"),
        _label_panel(_overlay_mask(rgb, final_pred, (255, 40, 40)), "Final"),
        _label_panel(_overlay_mask(rgb, selected_pred, (255, 190, 40)), f"SelectedQ {selected_query}"),
        _label_panel(
            _overlay_mask(rgb, oracle_pred, (60, 150, 255)),
            f"OracleMatchedQ {oracle_query}" if oracle_query >= 0 else "OracleMatchedQ none",
        ),
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
    Image.fromarray(mask.astype(np.uint8) * 255).save(path)


def _save_mask_exports(
    out_dir: Path,
    stem: str,
    final_pred: np.ndarray,
    selected_pred: np.ndarray,
    oracle_pred: np.ndarray,
    gt: np.ndarray,
) -> dict[str, str]:
    """导出单独二值 mask，便于下游检查或复用。"""
    export_dir = out_dir / "mask_exports"
    exports = {
        "final": final_pred,
        "selected_proposal": selected_pred,
        "oracle_matched_proposal": oracle_pred,
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
    target_hw = transform.get("target_hw")
    if (
        not isinstance(source_hw, list)
        or not isinstance(resized_hw, list)
        or not isinstance(target_hw, list)
        or len(source_hw) != 2
        or len(resized_hw) != 2
        or len(target_hw) != 2
    ):
        return None
    src_h, src_w = int(source_hw[0]), int(source_hw[1])
    resized_h, resized_w = int(resized_hw[0]), int(resized_hw[1])
    target_h, target_w = int(target_hw[0]), int(target_hw[1])
    if (
        src_h <= 0
        or src_w <= 0
        or resized_h <= 0
        or resized_w <= 0
        or target_h <= 0
        or target_w <= 0
        or tuple(mask.shape[-2:]) != (target_h, target_w)
    ):
        return None
    pad_top = int(transform.get("pad_top", 0))
    pad_left = int(transform.get("pad_left", 0))
    if (
        pad_top < 0
        or pad_left < 0
        or pad_top + resized_h > target_h
        or pad_left + resized_w > target_w
    ):
        return None
    crop = mask[pad_top : pad_top + resized_h, pad_left : pad_left + resized_w]
    if crop.shape != (resized_h, resized_w):
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
    gates = outputs.get("proposal_relevance_gates")
    targets = outputs.get("proposal_relevance_targets")
    return {
        "relevance_logit": (
            float(tensor.detach().float().cpu()[sample_idx, query_idx].item())
            if torch.is_tensor(tensor)
            else None
        ),
        "relevance_gate": (
            float(gates.detach().float().cpu()[sample_idx, query_idx].item())
            if torch.is_tensor(gates) else None
        ),
        "assignment_target": (
            float(targets.detach().float().cpu()[sample_idx, query_idx].item())
            if torch.is_tensor(targets) else None
        ),
    }


def _sample_query_geometry(outputs, sample_idx: int, query_idx: int) -> dict[str, Any] | None:
    reference = outputs.get("query_sampling_reference")
    grid = outputs.get("query_sampling_grid")
    if not torch.is_tensor(reference) or not torch.is_tensor(grid):
        return None
    return {
        "reference_xy": reference.detach().float().cpu()[sample_idx, query_idx].tolist(),
        "scale_point_xy": grid.detach().float().cpu()[sample_idx, query_idx].tolist(),
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
    """保存模型输出和显式标注为 GT-only 的 oracle assignment 诊断。"""
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
        selected_query = _selected_query_index(outputs, idx)
        selected_pred = (proposal_probs[idx, selected_query].numpy() >= float(threshold)).astype(np.uint8)
        selected_pred *= valid
        oracle_query = _oracle_matched_query_index(outputs, idx)
        oracle_pred = (
            (proposal_probs[idx, oracle_query].numpy() >= float(threshold)).astype(np.uint8)
            if oracle_query >= 0
            else np.zeros_like(final_pred)
        )
        oracle_pred *= valid
        diagnostic = _compose_diagnostic(
            rgb, gt, final_pred, selected_pred, selected_query, oracle_pred, oracle_query
        )

        meta = batch["metadata"][idx]
        sample_instances = batch["instances"][idx]
        modality_names = [item.name for item in sample_instances]
        stem = safe_slug(f"{prefix}_{idx}_{meta.get('sample_id', 'sample')}")
        overview_path = (
            _save_multimodal_overview(
                out_dir, stem, meta, rgb, gt, final_pred, selected_pred, selected_query,
                oracle_pred, oracle_query, sample_instances
            )
            if export_multimodal_overview
            else None
        )
        mask_paths = _save_mask_exports(out_dir, stem, final_pred, selected_pred, oracle_pred, gt)
        masks_for_restore = {
            "final": final_pred,
            "selected_proposal": selected_pred,
            "oracle_matched_proposal": oracle_pred,
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
                "selected_query": selected_query,
                "oracle_matched_query": oracle_query,
                "oracle_matched_dice": (
                    float(outputs["proposal_oracle_matched_dice"].detach().float().cpu()[idx].item())
                    if torch.is_tensor(outputs.get("proposal_oracle_matched_dice")) else None
                ),
                "oracle_is_gt_diagnostic_only": True,
                "threshold": float(threshold),
                "mask_area": {
                    "final": int(final_pred.sum()),
                    "selected_proposal": int(selected_pred.sum()),
                    "oracle_matched_proposal": int(oracle_pred.sum()),
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
                "query_modality_selected_query_attention": _sample_query_modality_values(query_attention_tensor, idx, modality_names, selected_query),
                "selected_query_scores": _sample_query_score_values(outputs, idx, selected_query),
                "selected_query_sampling": _sample_query_geometry(outputs, idx, selected_query),
                "oracle_matched_query_scores": (
                    _sample_query_score_values(outputs, idx, oracle_query) if oracle_query >= 0 else None
                ),
                "oracle_matched_query_sampling": (
                    _sample_query_geometry(outputs, idx, oracle_query) if oracle_query >= 0 else None
                ),
                "null_evidence_weight": (
                    float(outputs["null_evidence_weight"].detach().float().cpu()[idx].item())
                    if torch.is_tensor(outputs.get("null_evidence_weight")) else None
                ),
                "real_evidence_mass": (
                    float(outputs["real_evidence_mass"].detach().float().cpu()[idx].item())
                    if torch.is_tensor(outputs.get("real_evidence_mass")) else None
                ),
                "visual_evidence_delta_norm": (
                    float(outputs["visual_evidence_delta_norm"].detach().float().cpu()[idx].item())
                    if torch.is_tensor(outputs.get("visual_evidence_delta_norm")) else None
                ),
                "metadata": {
                    "sample_id": meta.get("sample_id"),
                    "parent_sample_id": meta.get("parent_sample_id"),
                    "dataset_name": meta.get("dataset_name"),
                    "template_id": meta.get("template_id"),
                    "task_family": meta.get("task_family"),
                    "family_combo": meta.get("family_combo"),
                    "active_subset": meta.get("active_subset"),
                    "sensor_combo": meta.get("sensor_combo"),
                    "gsd_m": meta.get("gsd_m"),
                    "canvas_gsd_m": meta.get("canvas_gsd_m"),
                    "mask_original_size": meta.get("mask_original_size"),
                    "resize_transform": meta.get("resize_transform"),
                    "instruction": meta.get("instruction"),
                    "condition_prompt": batch["condition_prompt_text"][idx],
                    "evidence_reasoning": batch["evidence_reasoning_text"][idx],
                    "modalities": [
                        {
                            "name": item.name,
                            "family": item.family,
                            "sensor": item.sensor,
                            "product_type": item.product_type,
                            "units": item.units,
                            "source_native_gsd_m": item.metadata.get("source_native_gsd_m"),
                            "encoder_native_gsd_m": item.native_gsd_m,
                            "encoder_aligned_gsd_m": item.aligned_gsd_m,
                            "native_resize_factor": item.metadata.get("native_resize_factor"),
                        }
                        for item in sample_instances
                    ],
                },
            },
        )
        paths.append(path.as_posix())
    return paths
