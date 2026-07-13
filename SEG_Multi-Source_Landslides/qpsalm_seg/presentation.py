#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""面向汇报展示的推理图片与静态图库。

用途：把共享推理结果导出为不含 GT-oracle 的 presentation overview 和 HTML 图库。
运行方式：不作为独立入口；由 ``qpsalm-curate-gallery`` 调用。
"""

from __future__ import annotations

from html import escape
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from qpsalm_seg.inference import PredictionResult
from qpsalm_seg.visualize import (
    compose_image_grid,
    label_image_panel,
    overlay_binary_mask,
    render_modality_view,
    render_reference_view,
    safe_slug,
)


def _probability_rgb(probability: np.ndarray) -> np.ndarray:
    value = np.clip(probability.astype(np.float32), 0.0, 1.0)
    red = np.clip(2.0 * value, 0.0, 1.0)
    blue = np.clip(2.0 * (1.0 - value), 0.0, 1.0)
    green = np.clip(1.0 - np.abs(2.0 * value - 1.0), 0.0, 1.0)
    return (np.stack([red, green, blue], axis=-1) * 255.0).astype(np.uint8)


def _save_mask(path: Path, mask: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255).save(path)
    return path.as_posix()


def presentation_arrays(result: PredictionResult) -> tuple[list[tuple[np.ndarray, str]], list[tuple[np.ndarray, str]]]:
    """返回 Gradio 可直接展示的模态与预测图组。"""
    reference = render_reference_view(result.batch, 0)
    target_size = int(result.final_mask.shape[-1])
    modalities = [
        (
            render_modality_view(instance, target_size=target_size),
            f"{instance.name} [{instance.sensor}|{instance.product_type}]",
        )
        for instance in result.batch.instances[0]
    ]
    predictions = [
        (reference, "Reference"),
        (overlay_binary_mask(reference, result.ground_truth, (0, 230, 80)), "GT reference"),
        (overlay_binary_mask(reference, result.final_mask, (255, 40, 40)), "Final prediction"),
        (
            overlay_binary_mask(reference, result.selected_proposal, (255, 190, 40)),
            f"Selected proposal Q{result.selected_query}",
        ),
        (_probability_rgb(result.probability), "Foreground probability"),
    ]
    return modalities, predictions


def save_presentation_result(
    result: PredictionResult,
    out_dir: Path,
    *,
    category: str,
    stratum: str,
) -> dict[str, Any]:
    """保存单个样本的 PPT overview；不读取或展示 oracle query。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = result.batch.metadata[0]
    modality_arrays, prediction_arrays = presentation_arrays(result)
    panels = [label_image_panel(value, label) for value, label in modality_arrays + prediction_arrays]
    metric_note = "reference-only" if result.metrics_are_reference_only else "benchmark GT"
    header = [
        f"sample={result.sample_id} dataset={metadata.get('dataset_name')} category={category}",
        f"families={metadata.get('family_combo')} modalities={','.join(metadata.get('active_modalities') or [])}",
        f"Dice={result.metrics.get('dice', 0):.4f} IoU={result.metrics.get('iou', 0):.4f} metrics={metric_note}",
        f"instruction={metadata.get('instruction')}",
    ]
    stem = safe_slug(result.sample_id)
    overview_path = out_dir / "overviews" / f"{stem}_presentation.png"
    overview_path.parent.mkdir(parents=True, exist_ok=True)
    compose_image_grid(panels, header_lines=header).save(overview_path)
    mask_paths = {
        "final": _save_mask(out_dir / "masks" / "final" / f"{stem}_final.png", result.final_mask),
        "selected_proposal": _save_mask(
            out_dir / "masks" / "selected_proposal" / f"{stem}_selected.png",
            result.selected_proposal,
        ),
        "gt_reference": _save_mask(
            out_dir / "masks" / "gt_reference" / f"{stem}_gt.png",
            result.ground_truth,
        ),
    }
    if result.restored_final_mask is not None:
        mask_paths["final_original_size"] = _save_mask(
            out_dir / "masks" / "final_original_size" / f"{stem}_final_original.png",
            result.restored_final_mask,
        )
    return {
        "sample_id": result.sample_id,
        "parent_sample_id": metadata.get("parent_sample_id"),
        "dataset_name": metadata.get("dataset_name"),
        "task_family": metadata.get("task_family"),
        "family_combo": metadata.get("family_combo"),
        "category": category,
        "stratum": stratum,
        "instruction": metadata.get("instruction"),
        "condition": metadata.get("condition"),
        "active_modalities": metadata.get("active_modalities"),
        "overview_path": overview_path.as_posix(),
        "mask_paths": mask_paths,
        "metrics": result.metrics,
        "metrics_are_reference_only": result.metrics_are_reference_only,
        "selected_query": result.selected_query,
        "latency_seconds": result.latency_seconds,
        "diagnostics": result.diagnostics,
        "contains_oracle_output": False,
    }


def write_gallery_html(records: list[dict[str, Any]], path: Path) -> None:
    """写出可离线打开的筛选图库。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    cards = []
    for record in records:
        overview = Path(str(record["overview_path"]))
        try:
            source = overview.relative_to(path.parent).as_posix()
        except ValueError:
            source = overview.as_posix()
        metrics = record.get("metrics") or {}
        search = " ".join([
            *(str(record.get(key) or "") for key in (
                "sample_id", "dataset_name", "task_family", "family_combo", "category", "instruction"
            )),
            *(str(tag) for tag in (record.get("tags") or [])),
        ]).lower()
        cards.append(
            f'<article class="card" data-search="{escape(search, quote=True)}">'
            f'<img src="{escape(source, quote=True)}" loading="lazy">'
            f'<h3>{escape(str(record.get("dataset_name")))} | {escape(str(record.get("category")))}</h3>'
            f'<p>{escape(str(record.get("family_combo")))} | {escape(str(record.get("task_family")))}</p>'
            f'<p>Dice {float(metrics.get("dice", 0)):.4f} | IoU {float(metrics.get("iou", 0)):.4f}</p>'
            f'<p class="instruction">{escape(str(record.get("instruction")))}</p>'
            f'<code>{escape(str(record.get("sample_id")))}</code>'
            '</article>'
        )
    html = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>QPSALM PPT Gallery</title><style>
body{{font-family:Arial,sans-serif;margin:20px;background:#f4f5f7;color:#17191c}}
input{{width:min(720px,95%);padding:10px;margin:0 0 18px;border:1px solid #aaa}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:14px}}
.card{{background:white;border:1px solid #d4d7dc;padding:10px;border-radius:6px}}
.card img{{width:100%;height:auto;display:block}}h3{{font-size:16px;margin:10px 0 4px}}
p{{font-size:13px;margin:4px 0}}.instruction{{min-height:34px}}code{{font-size:11px;word-break:break-all}}
</style></head><body><h1>Multi-Source Qwen-PSALM-Seg PPT Gallery</h1>
<input id="filter" placeholder="筛选 dataset、模态、任务、类别或 sample ID">
<section class="grid">{''.join(cards)}</section>
<script>const f=document.getElementById('filter');f.addEventListener('input',()=>{{const q=f.value.toLowerCase();document.querySelectorAll('.card').forEach(c=>c.hidden=!c.dataset.search.includes(q));}});</script>
</body></html>"""
    path.write_text(html, encoding="utf-8")


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
