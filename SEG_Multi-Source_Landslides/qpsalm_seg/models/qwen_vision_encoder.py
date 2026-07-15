#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared offline Qwen vision-tower encoder for segmentation and description caches.

This is an internal module and is not an independent CLI entry point.
"""

from __future__ import annotations

import hashlib
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from qpsalm_seg.controllers import (
    local_model_revision,
    local_processor_revision,
    select_qwen_model_class,
    validate_qwen_model_dir,
)
from qpsalm_seg.rendering import RenderedView


def rendered_view_to_pil(view: RenderedView):
    from PIL import Image
    import numpy as np

    array = (view.image.permute(1, 2, 0).clamp(0, 1).numpy() * 255).round().astype(np.uint8)
    return Image.fromarray(array)


def restore_qwen_patch_grid(
    hidden: torch.Tensor,
    grid_thw: tuple[int, int, int] | list[int],
    merge_size: int,
) -> torch.Tensor:
    """Undo Qwen3-VL merge-block token permutation and recover [C,H,W]."""
    t, h, w = (int(value) for value in grid_thw)
    merge = int(merge_size)
    if h % merge or w % merge:
        raise ValueError(f"Qwen vision grid {(t, h, w)} cannot be restored with merge_size={merge}")
    expected = t * h * w
    if hidden.ndim != 2 or hidden.shape[0] != expected:
        raise ValueError(f"Qwen hidden shape={tuple(hidden.shape)} expected tokens={expected}")
    channels = int(hidden.shape[-1])
    value = hidden.view(t, h // merge, w // merge, merge, merge, channels)
    value = value.permute(0, 1, 3, 2, 4, 5).reshape(t, h, w, channels)
    return value.mean(0).permute(2, 0, 1).contiguous()


def _hash_features(
    view: RenderedView,
    layers: Sequence[int],
    spatial_sizes: Sequence[int],
    view_tokens: int,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    outputs = []
    for layer, spatial_size in zip(layers, spatial_sizes):
        seed = int(hashlib.sha256(f"{view.content_hash}:{layer}".encode()).hexdigest()[:16], 16) % (2**31)
        outputs.append(torch.randn(
            1024, int(spatial_size), int(spatial_size),
            generator=torch.Generator().manual_seed(seed),
        ))
    seed = int(view.content_hash[:16], 16) % (2**31)
    tokens = torch.randn(int(view_tokens), 2048, generator=torch.Generator().manual_seed(seed))
    return [value.half() for value in outputs], tokens.half()


class HashVisionEncoder:
    revision = "hash-smoke"
    processor_revision = "hash-smoke"
    spatial_channels = 1024
    token_dim = 2048

    def __init__(self, layers: Sequence[int], spatial_sizes: Sequence[int], view_tokens: int) -> None:
        self.layers = tuple(int(value) for value in layers)
        self.spatial_sizes = tuple(int(value) for value in spatial_sizes)
        self.view_tokens = int(view_tokens)

    def encode(self, record: dict[str, Any]) -> list[dict[str, Any]]:
        encoded = []
        for view in record["views"]:
            spatial, tokens = _hash_features(view, self.layers, self.spatial_sizes, self.view_tokens)
            encoded.append({
                "spatial": spatial,
                "tokens": tokens,
                "vision_grid_thw": [1, self.spatial_sizes[0], self.spatial_sizes[0]],
                "merged_grid_hw": [1, int(tokens.shape[0])],
            })
        return encoded

    def close(self) -> None:
        return None


class QwenVisionEncoder:
    """Keep the frozen Qwen vision tower loaded once for offline cache construction."""

    def __init__(
        self,
        model_path: str,
        device: torch.device,
        layers: Sequence[int],
        spatial_sizes: Sequence[int],
    ) -> None:
        from transformers import AutoProcessor

        self.device = device
        self.layers = tuple(int(value) for value in layers)
        self.spatial_sizes = tuple(int(value) for value in spatial_sizes)
        model_dir = validate_qwen_model_dir(model_path)
        model_cls = select_qwen_model_class()
        self.processor = AutoProcessor.from_pretrained(str(model_dir), trust_remote_code=True)
        full_model = model_cls.from_pretrained(
            str(model_dir), torch_dtype=torch.bfloat16, trust_remote_code=True
        ).to(device).eval()
        self.revision = local_model_revision(model_dir)
        self.processor_revision = local_processor_revision(model_dir)
        self.visual = full_model.model.visual
        full_model.model.visual = None
        del full_model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        self.merge_size = int(self.visual.spatial_merge_size)
        self.spatial_channels = int(self.visual.config.hidden_size)
        self.token_dim = int(self.visual.config.out_hidden_size)
        self.captured: dict[int, torch.Tensor] = {}
        self.hooks = []
        for layer in self.layers:
            if layer < 0 or layer >= len(self.visual.blocks):
                self.close()
                raise ValueError(f"vision layer={layer} 超出 [0,{len(self.visual.blocks) - 1}]")
            self.hooks.append(self.visual.blocks[layer].register_forward_hook(
                lambda _module, _inputs, output, layer_index=layer: self.captured.__setitem__(
                    layer_index, output[0] if isinstance(output, tuple) else output
                )
            ))

    @torch.no_grad()
    def encode(self, record: dict[str, Any]) -> list[dict[str, Any]]:
        encoded_views = []
        for view in record["views"]:
            image = rendered_view_to_pil(view)
            prompt = self.processor.apply_chat_template(
                [{"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": view.description},
                ]}],
                tokenize=False,
                add_generation_prompt=False,
            )
            encoded = self.processor(text=[prompt], images=[image], return_tensors="pt")
            pixel_values = encoded["pixel_values"].to(self.device)
            grid = encoded["image_grid_thw"].to(self.device)
            self.captured.clear()
            output = self.visual(pixel_values, grid_thw=grid, return_dict=True)
            maps = []
            t, h, w = [int(value) for value in grid[0].tolist()]
            for layer, spatial_size in zip(self.layers, self.spatial_sizes):
                hidden = restore_qwen_patch_grid(self.captured[layer], (t, h, w), self.merge_size)
                maps.append(F.adaptive_avg_pool2d(hidden[None].float(), int(spatial_size))[0].half().cpu())
            encoded_views.append({
                "spatial": maps,
                "tokens": output.pooler_output.float().half().cpu(),
                "vision_grid_thw": [t, h, w],
                "merged_grid_hw": [h // self.merge_size, w // self.merge_size],
            })
            self.captured.clear()
        return encoded_views

    def close(self) -> None:
        for hook in getattr(self, "hooks", []):
            hook.remove()
        self.hooks = []
        self.captured = {}
        if hasattr(self, "visual"):
            del self.visual
        if hasattr(self, "processor"):
            del self.processor
        if getattr(self, "device", torch.device("cpu")).type == "cuda":
            torch.cuda.empty_cache()
