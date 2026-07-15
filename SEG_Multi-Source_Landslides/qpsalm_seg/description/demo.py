#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gradio session for benchmark-backed grounded region description."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from qpsalm_seg.paths import resolve_project_path

from .checkpoint import load_segdesc_checkpoint
from .common import build_description_dataset, description_device
from .config import SegDescConfig
from .counterfactuals import select_backbone_state
from .output_protocol import parse_description_output
from .runtime import build_segdesc_model


def _display_array(path_ref: str) -> Image.Image:
    path = resolve_project_path(path_ref)
    if path is None:
        raise ValueError(f"无法解析影像: {path_ref}")
    if path.suffix.casefold() != ".npy":
        with Image.open(path) as source:
            return source.convert("RGB")
    value = np.asarray(np.load(path), dtype=np.float32)
    if value.ndim == 2:
        value = value[None]
    if value.ndim == 3 and value.shape[-1] <= 16 and value.shape[0] > 16:
        value = np.moveaxis(value, -1, 0)
    if value.ndim != 3:
        raise ValueError(f"无法显示数组 shape={value.shape}: {path_ref}")
    channels = value[:3] if value.shape[0] >= 3 else np.repeat(value[:1], 3, 0)
    output = []
    for channel in channels:
        finite = channel[np.isfinite(channel)]
        if not finite.size:
            output.append(np.zeros_like(channel))
            continue
        low, high = np.percentile(finite, [2, 98])
        output.append(np.clip((channel - low) / max(high - low, 1.0e-6), 0, 1))
    rgb = np.moveaxis(np.stack(output), 0, -1)
    return Image.fromarray((rgb * 255).astype(np.uint8), "RGB")


def _row_preview(row: dict[str, Any]) -> Image.Image:
    visual = row.get("visual_ref") or {}
    if visual.get("type") == "single_image":
        return _display_array(str(visual["path"]))
    paths = visual.get("modality_paths") or {}
    if not paths:
        raise ValueError("multisource Bridge row 缺少 modality_paths")
    preferred = next((paths[key] for key in sorted(paths) if "opt" in key.casefold() or "s2" in key.casefold()), None)
    return _display_array(str(preferred or paths[sorted(paths)[0]]))


def _overlay(image: Image.Image, mask: torch.Tensor) -> Image.Image:
    resized = F.interpolate(mask[None].float(), size=(image.height, image.width), mode="nearest")[0, 0]
    binary = resized.detach().cpu().numpy() > 0.5
    value = np.asarray(image.convert("RGB"), dtype=np.float32)
    value[binary] = 0.55 * value[binary] + 0.45 * np.array([255, 35, 35], dtype=np.float32)
    return Image.fromarray(value.clip(0, 255).astype(np.uint8), "RGB")


class DescriptionDemoSession:
    def __init__(
        self,
        config: SegDescConfig,
        checkpoint: str,
        split: str,
        device_name: str,
    ) -> None:
        self.config = config
        self.device = description_device(device_name)
        self.model, _migration = build_segdesc_model(config, self.device)
        self.checkpoint_step, _metadata = load_segdesc_checkpoint(checkpoint, self.model)
        self.dataset = build_description_dataset(
            config, self.model.description_backbone.bank, split=split, training=False
        )
        self.by_id = {
            str(row.get("sample_id") or row.get("bridge_record_id")): index
            for index, row in enumerate(self.dataset.rows)
        }
        self.sample_ids = sorted(self.by_id)
        if not self.sample_ids:
            raise RuntimeError(f"description demo split={split} 没有样本")
        self.model.eval()

    def inspect(self, sample_id: str):
        index = self.by_id[sample_id]
        item = self.dataset[index]
        row = self.dataset.rows[index]
        image = _row_preview(row)
        return (
            image,
            _overlay(image, item["region_mask"]),
            item["instruction"],
            {
                "sample_id": item["sample_id"],
                "parent_sample_id": item["parent_sample_id"],
                "task_family": item["task_family"],
                "target_status": item["target_status"],
                "checkpoint_step": self.checkpoint_step,
            },
        )

    @torch.no_grad()
    def infer(self, sample_id: str, instruction: str, mask_mode: str):
        item = self.dataset[self.by_id[sample_id]]
        backbone = self.model.encode_description_requests([item["request"]])
        mask = item["region_mask"][None].to(self.device)
        if mask_mode == "full":
            mask = torch.ones_like(mask)
        elif mask_mode == "zero":
            mask = torch.zeros_like(mask)
        raw = self.model.generate_from_state(
            backbone,
            mask,
            instruction or item["instruction"],
            max_new_tokens=self.config.max_new_tokens,
            protocol=self.config.region_protocol,
            structured_output=bool(item["structured_output"]),
        )
        parsed = parse_description_output(raw) if item["structured_output"] else None
        return raw, ({
            "schema_valid": parsed.schema_valid,
            "parse_errors": list(parsed.parse_errors),
            "parsed_raw": parsed.parsed,
            "deterministic_repair": parsed.repaired,
            "repair_actions": list(parsed.repair_actions),
        } if parsed is not None else {"free_text": True})


def build_demo(session: DescriptionDemoSession) -> gr.Blocks:
    initial = session.sample_ids[0]
    image, overlay, instruction, metadata = session.inspect(initial)
    with gr.Blocks(title="QPSALM Grounded Description") as app:
        gr.Markdown("# Segmentation-Grounded Remote-Sensing Description")
        with gr.Row():
            sample = gr.Dropdown(session.sample_ids, value=initial, label="Benchmark sample")
            mask_mode = gr.Radio(["region", "full", "zero"], value="region", label="Region input")
        instruction_box = gr.Textbox(value=instruction, label="Instruction", lines=3)
        with gr.Row():
            image_view = gr.Image(value=image, label="Reference", type="pil")
            overlay_view = gr.Image(value=overlay, label="Region overlay", type="pil")
        metadata_view = gr.JSON(value=metadata, label="Sample metadata")
        run = gr.Button("Generate", variant="primary")
        raw = gr.Textbox(label="Raw generation", lines=10)
        parsed = gr.JSON(label="Parse and repair diagnostics")

        sample.change(
            session.inspect,
            inputs=[sample],
            outputs=[image_view, overlay_view, instruction_box, metadata_view],
        )
        run.click(session.infer, inputs=[sample, instruction_box, mask_mode], outputs=[raw, parsed])
    return app
