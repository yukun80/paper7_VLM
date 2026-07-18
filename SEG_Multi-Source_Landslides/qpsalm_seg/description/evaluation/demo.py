#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gradio runtime for benchmark-backed grounded region description."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np
from PIL import Image
import torch

from qpsalm_seg.paths import resolve_project_path

from ..training.checkpoint import load_segdesc_checkpoint
from ..protocols.io import sha256_file as _sha256_file
from ..protocols.region_geometry import restore_region_mask_from_cache
from ..data.loaders import build_description_dataset, description_device
from ..protocols.config import SegDescConfig
from ..data.datasets import bridge_region_metadata
from ..evaluation.runner import (
    EndToEndMaskProvider,
)
from ..evaluation.publication import validate_evaluation_checkpoint_binding
from ..protocols.output import parse_description_output
from ..training.runtime import build_segdesc_model


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


def _reference_view_preview(
    row: dict[str, Any], cache_record: dict[str, Any]
) -> tuple[Image.Image, dict[str, Any]]:
    """Display the physical source of cache view 0, whose transform owns the mask."""
    visual = row.get("visual_ref") or {}
    views = cache_record.get("views") or []
    if not views or not isinstance(views[0], dict):
        raise ValueError("description cache record 缺少 reference view")
    view = views[0]
    if visual.get("type") == "single_image":
        path_ref = str(cache_record.get("source_ref") or visual.get("path") or "")
        if path_ref != str(visual.get("path") or ""):
            raise ValueError("single-image cache source_ref 与 benchmark visual_ref 不一致")
        source_modality = "single_image"
    else:
        paths = visual.get("modality_paths") or {}
        source_modality = next((
            str(name) for name in (view.get("source_modalities") or [])
            if str(name) in paths
        ), "")
        if not source_modality:
            raise ValueError(
                "cache reference view 的 source_modalities 无法映射到 Bridge modality_paths"
            )
        path_ref = str(paths[source_modality])
    image = _display_array(path_ref)
    transform = dict(view.get("render_transform") or {})
    if (
        int(transform.get("source_h") or 0) != image.height
        or int(transform.get("source_w") or 0) != image.width
    ):
        raise ValueError(
            "demo reference image 与 cache render transform 原始尺寸不一致: "
            f"image={(image.height, image.width)} transform="
            f"{(transform.get('source_h'), transform.get('source_w'))}"
        )
    return image, {
        "cache_lookup_key": cache_record.get("lookup_key"),
        "cache_fingerprint": cache_record.get("cache_fingerprint"),
        "reference_view_name": view.get("name"),
        "reference_source_modality": source_modality,
        "reference_source_path": path_ref,
        "source_modalities": list(view.get("source_modalities") or []),
        "source_families": list(view.get("source_families") or []),
        "render_transform": transform,
    }


def _overlay(image: Image.Image, mask: torch.Tensor) -> Image.Image:
    value = mask.float()
    if value.ndim == 3 and value.shape[0] == 1:
        value = value[0]
    if value.ndim != 2 or tuple(value.shape) != (image.height, image.width):
        raise ValueError(
            "demo overlay 必须使用已恢复到 reference source 的 mask: "
            f"mask={tuple(value.shape)} image={(image.height, image.width)}"
        )
    binary = value.detach().cpu().numpy() > 0.5
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
        self.model, runtime_migration = build_segdesc_model(config, self.device)
        checkpoint_path = resolve_project_path(checkpoint) or Path(checkpoint)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"description demo checkpoint 不存在: {checkpoint}")
        self.checkpoint_step, checkpoint_metadata = load_segdesc_checkpoint(
            checkpoint_path, self.model
        )
        self.dataset = build_description_dataset(
            config, self.model.description_backbone.bank, split=split, training=False
        )
        self.checkpoint_binding = validate_evaluation_checkpoint_binding(
            config,
            checkpoint_metadata,
            runtime_migration,
            getattr(self.dataset, "predicted_index_audit", None),
            checkpoint=checkpoint_path,
        )
        self.checkpoint_audit = {
            "path": str(checkpoint_path.resolve(strict=False)),
            "sha256": _sha256_file(checkpoint_path),
            "step": int(self.checkpoint_step),
            "binding": self.checkpoint_binding,
        }
        self.end_to_end = (
            EndToEndMaskProvider(
                self.model, split, config.evaluation.segmentation_mask_threshold
            )
            if config.evaluation.evaluation_mode == "end_to_end" else None
        )
        if self.end_to_end is not None:
            self.end_to_end.require_targets(
                bridge_region_metadata(row) for row in self.dataset.rows
            )
        self.by_id: dict[str, int] = {}
        self.sample_choices: list[tuple[str, str]] = []
        for index, row in enumerate(self.dataset.rows):
            sample_id = str(row.get("sample_id") or row.get("bridge_record_id") or "")
            if not sample_id or sample_id in self.by_id:
                raise ValueError("description demo sample identity 必须非空且唯一")
            self.by_id[sample_id] = index
            region = str(row.get("region_id") or row.get("region_pair_id") or "global")
            source = str(row.get("region_source") or row.get("task_family") or "unknown")
            parent = str(row.get("parent_sample_id") or "unknown")
            self.sample_choices.append((
                f"{parent} | {source}:{region} | {sample_id}", sample_id
            ))
        self.sample_choices.sort(key=lambda value: value[0])
        self.sample_ids = sorted(self.by_id)
        if not self.sample_ids:
            raise RuntimeError(f"description demo split={split} 没有样本")
        self.model.eval()

    def _visual_context(
        self, item: dict[str, Any], row: dict[str, Any]
    ) -> tuple[Image.Image, dict[str, Any]]:
        component, parent = item["request"]
        record = self.model.description_backbone.bank.record(component, parent)
        return _reference_view_preview(row, record)

    @staticmethod
    def _display_mask(
        mask: torch.Tensor, view_audit: dict[str, Any]
    ) -> torch.Tensor:
        if mask.ndim != 4 or mask.shape[0] != 1 or mask.shape[1] != 1:
            raise ValueError(f"demo protocol mask 必须为 [1,1,H,W]: {tuple(mask.shape)}")
        return restore_region_mask_from_cache(
            mask[0], dict(view_audit["render_transform"])
        )

    def _protocol_mask(
        self, item: dict[str, Any]
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        base = item["region_mask"][None].to(self.device)
        if self.end_to_end is not None:
            mask, audit = self.end_to_end.predict(
                item, tuple(base.shape[-2:])
            )
            return mask.to(self.device), {
                "mask_source": "end_to_end_segmentation_prediction",
                **audit,
            }
        return base, {
            "mask_source": (
                "fixed_prediction" if self.config.evaluation.evaluation_mode == "fixed_prediction"
                else "gt_mask"
            ),
            "region_mask_path": item.get("region_mask_path"),
        }

    def inspect(self, sample_id: str):
        index = self.by_id[sample_id]
        item = self.dataset[index]
        row = self.dataset.rows[index]
        image, view_audit = self._visual_context(item, row)
        protocol_mask, mask_audit = self._protocol_mask(item)
        display_mask = self._display_mask(protocol_mask, view_audit)
        return (
            image,
            _overlay(image, display_mask),
            item["instruction"],
            {
                "sample_id": item["sample_id"],
                "parent_sample_id": item["parent_sample_id"],
                "task_family": item["task_family"],
                "target_status": item["target_status"],
                "checkpoint_step": self.checkpoint_step,
                "evaluation_mode": self.config.evaluation.evaluation_mode,
                "region_protocol": self.config.model.region_protocol,
                "region_id": item.get("region_id"),
                "region_source": item.get("region_source"),
                "region_mask_path": item.get("region_mask_path"),
                "checkpoint_audit": self.checkpoint_audit,
                "reference_view": view_audit,
                "mask_audit": mask_audit,
            },
        )

    @torch.no_grad()
    def infer(self, sample_id: str, instruction: str, mask_mode: str):
        item = self.dataset[self.by_id[sample_id]]
        row = self.dataset.rows[self.by_id[sample_id]]
        image, view_audit = self._visual_context(item, row)
        backbone = self.model.encode_description_requests(
            [item["request"]], include_spatial=bool(item["use_region_tokens"])
        )
        mask, mask_audit = self._protocol_mask(item)
        if mask_mode == "full":
            mask = torch.ones_like(mask)
            mask_audit = {**mask_audit, "counterfactual_override": "full"}
        elif mask_mode == "zero":
            mask = torch.zeros_like(mask)
            mask_audit = {**mask_audit, "counterfactual_override": "zero"}
        raw = self.model.generate_from_state(
            backbone,
            mask,
            instruction or item["instruction"],
            max_new_tokens=self.config.evaluation.max_new_tokens,
            protocol=self.config.model.region_protocol,
            structured_output=bool(item["structured_output"]),
            use_region_tokens=bool(item["use_region_tokens"]),
        )
        parsed = parse_description_output(raw) if item["structured_output"] else None
        diagnostics = ({
            "schema_valid": parsed.schema_valid,
            "parse_errors": list(parsed.parse_errors),
            "parsed_raw": parsed.parsed,
            "deterministic_repair": parsed.repaired,
            "repair_actions": list(parsed.repair_actions),
        } if parsed is not None else {"free_text": True})
        diagnostics["mask_audit"] = mask_audit
        diagnostics["checkpoint_audit"] = self.checkpoint_audit
        diagnostics["reference_view"] = view_audit
        display_mask = self._display_mask(mask, view_audit)
        return _overlay(image, display_mask), raw, diagnostics


def build_demo(session: DescriptionDemoSession) -> gr.Blocks:
    initial = session.sample_ids[0]
    image, overlay, instruction, metadata = session.inspect(initial)
    with gr.Blocks(title="QPSALM Grounded Description") as app:
        gr.Markdown("# Segmentation-Grounded Remote-Sensing Description")
        with gr.Row():
            sample = gr.Dropdown(
                session.sample_choices, value=initial,
                label="Parent | proposal/region | benchmark sample",
            )
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
        run.click(
            session.infer,
            inputs=[sample, instruction_box, mask_mode],
            outputs=[overlay_view, raw, parsed],
        )
    return app
