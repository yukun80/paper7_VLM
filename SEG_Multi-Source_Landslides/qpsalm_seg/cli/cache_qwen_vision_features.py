#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cache Qwen3-VL spatial vision features and view tokens in sharded v3 format.

用途：一次编码 benchmark-v2 父样本视图，供 SANE 和在线 Qwen mask-query controller 复用。
推荐运行命令：PYTHONPATH=SEG_Multi-Source_Landslides python -m
qpsalm_seg.cli.cache_qwen_vision_features --config CONFIG --output-dir outputs/qwen_vision_v3
--preset qwen_psalm_full --backend qwen --device cuda --overwrite
主要输入：benchmark-v2 instruction index 与本地 Qwen3-VL 权重。
主要输出：manifest.json 和 shard_*.pt；构建时只在内存中保留一个 shard。
写入行为：只写 --output-dir；--overwrite 会清理该 cache 目录。
所属流程：pretrained_sane_qmef_pmrd 和 qwen_psalm_full 的离线准备。
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Iterator

import torch
import torch.nn.functional as F
from tqdm import tqdm

from qpsalm_seg.config import load_config
from qpsalm_seg.controllers import (
    local_model_revision,
    local_processor_revision,
    select_qwen_model_class,
    validate_qwen_model_dir,
)
from qpsalm_seg.data import MultiSourceLandslideDataset
from qpsalm_seg.data.prompts import PROMPT_VERSION
from qpsalm_seg.paths import resolve_project_path
from qpsalm_seg.rendering import RENDERER_VERSION, RenderedView, render_sensor_views
from qpsalm_seg.presets import PRESET_CHOICES, apply_preset
from qpsalm_seg.models.vision_cache import view_fingerprint_fragment, vision_input_protocol


CACHE_FORMAT = "qpsalm_qwen_vision_cache_v3"
POOLING_METHOD = "spatial_layers_plus_adaptive_view_tokens"


def parse_args():
    parser = argparse.ArgumentParser(description="Cache sharded Qwen vision features for SANE/QMEF/PMRD.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--preset", choices=PRESET_CHOICES, default=None)
    parser.add_argument("--benchmark-dir", default=None)
    parser.add_argument("--train-index", default=None)
    parser.add_argument("--val-index", default=None)
    parser.add_argument("--eval-index", default=None)
    parser.add_argument("--eval-split", choices=["val", "test"], default="val")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backend", choices=["qwen", "hash-smoke"], default="qwen")
    parser.add_argument("--render-size", type=int, default=256)
    parser.add_argument("--layers", default="5,11,17,23")
    parser.add_argument(
        "--spatial-sizes",
        default="16,8,6,4",
        help="与 --layers 一一对应的缓存空间尺寸，浅层优先保留边界。",
    )
    parser.add_argument("--view-tokens", type=int, default=8)
    parser.add_argument("--shard-size", type=int, default=256)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verify-only", action="store_true", help="只校验已有 cache-v3 manifest/shard 协议。")
    return parser.parse_args()


def _pil(view: RenderedView):
    from PIL import Image
    import numpy as np
    array = (view.image.permute(1, 2, 0).clamp(0, 1).numpy() * 255).round().astype(np.uint8)
    return Image.fromarray(array)


def iter_records(config, render_size, max_samples, eval_split) -> Iterator[dict[str, Any]]:
    """Yield one rendered parent sample at a time without retaining image tensors."""
    stable = replace(config, modality_dropout=0.0, train_hflip_prob=0.0, train_vflip_prob=0.0)
    splits = (eval_split,) if eval_split else ("train", "val")
    seen_keys: set[str] = set()
    for split in splits:
        dataset = MultiSourceLandslideDataset(stable, split)
        for index in tqdm(range(len(dataset)), desc=f"vision-render-{split}"):
            row = dataset.rows[index]
            parent = str(row.get("parent_sample_id") or row.get("sample_id"))
            key = f"qmv3-parent:{parent}"
            if key in seen_keys:
                continue
            item = dataset[index]
            if item["visual_evidence_key"] != key:
                raise RuntimeError(
                    f"vision cache key mismatch: row={key} dataset={item['visual_evidence_key']}"
                )
            seen_keys.add(key)
            yield {
                "lookup_key": key,
                "parent_sample_id": parent,
                "full_subset_signature": item["active_subset"].signature,
                "views": render_sensor_views(item["full_instances"], render_size, strict=True),
                "modality_families": {value.name: value.family for value in item["full_instances"]},
            }
            if max_samples and len(seen_keys) >= max_samples:
                return


def _hash_features(view, layers, spatial_sizes, view_tokens):
    outputs = []
    for layer, spatial_size in zip(layers, spatial_sizes):
        seed = int(hashlib.sha256(f"{view.content_hash}:{layer}".encode()).hexdigest()[:16], 16) % (2**31)
        outputs.append(torch.randn(1024, spatial_size, spatial_size, generator=torch.Generator().manual_seed(seed)))
    seed = int(view.content_hash[:16], 16) % (2**31)
    tokens = torch.randn(view_tokens, 2048, generator=torch.Generator().manual_seed(seed))
    return [value.half() for value in outputs], tokens.half()


def restore_qwen_patch_grid(
    hidden: torch.Tensor,
    grid_thw: tuple[int, int, int] | list[int],
    merge_size: int,
) -> torch.Tensor:
    """Undo Qwen3-VL's merge-block token permutation and recover [C,H,W]."""
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


class HashVisionEncoder:
    revision = "hash-smoke"
    processor_revision = "hash-smoke"
    spatial_channels = 1024
    token_dim = 2048

    def __init__(self, layers, spatial_sizes, view_tokens):
        self.layers = layers
        self.spatial_sizes = spatial_sizes
        self.view_tokens = view_tokens

    def encode(self, record: dict[str, Any]) -> list[dict[str, Any]]:
        encoded = []
        for view in record["views"]:
            spatial, tokens = _hash_features(
                view, self.layers, self.spatial_sizes, self.view_tokens
            )
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
    """Keep Qwen loaded once while encoded records are released shard by shard."""

    def __init__(self, model_path, device, layers, spatial_sizes):
        from transformers import AutoProcessor

        self.device = device
        self.layers = layers
        self.spatial_sizes = spatial_sizes
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
        for layer in layers:
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
            image = _pil(view)
            prompt = self.processor.apply_chat_template(
                [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": view.description}]}],
                tokenize=False, add_generation_prompt=False,
            )
            encoded = self.processor(text=[prompt], images=[image], return_tensors="pt")
            pixel_values = encoded["pixel_values"].to(self.device)
            grid = encoded["image_grid_thw"].to(self.device)
            self.captured.clear()
            output = self.visual(pixel_values, grid_thw=grid, return_dict=True)
            maps = []
            t, h, w = [int(v) for v in grid[0].tolist()]
            for layer, spatial_size in zip(self.layers, self.spatial_sizes):
                hidden = restore_qwen_patch_grid(self.captured[layer], (t, h, w), self.merge_size)
                maps.append(F.adaptive_avg_pool2d(hidden[None].float(), spatial_size)[0].half().cpu())
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


def serialize_record(
    record: dict[str, Any],
    encoded_views: list[dict[str, Any]],
    *,
    spatial_sizes: tuple[int, ...],
    view_tokens: int,
    revision: str,
    processor_revision: str,
) -> dict[str, Any]:
    if len(record["views"]) != len(encoded_views):
        raise ValueError(
            f"view encoding count mismatch: views={len(record['views'])} encoded={len(encoded_views)}"
        )
    views = []
    for view, encoded in zip(record["views"], encoded_views):
        views.append({
            "name": view.name, "description": view.description,
            "source_modalities": list(view.source_modalities), "quality_flags": list(view.quality_flags),
            "source_families": sorted({record["modality_families"][name] for name in view.source_modalities}),
            "content_hash": view.content_hash, "render_transform": view.render_transform,
            "vision_grid_thw": list(encoded["vision_grid_thw"]),
            "merged_grid_hw": list(encoded["merged_grid_hw"]),
            "valid_mask": F.adaptive_avg_pool2d(view.valid_mask.float()[None], spatial_sizes[0])[0].half(),
            "spatial_features": encoded["spatial"],
            "view_tokens": (
                F.adaptive_avg_pool1d(encoded["tokens"].float().T[None], view_tokens)[0].T.half()
                if encoded["tokens"].shape[0] > view_tokens else encoded["tokens"]
            ),
        })
    fingerprint_payload = "|".join(
        [
            RENDERER_VERSION,
            str(revision),
            str(processor_revision),
            PROMPT_VERSION,
            POOLING_METHOD,
            str(record["full_subset_signature"]),
        ]
        + sorted(view_fingerprint_fragment(view) for view in views)
    )
    return {
        "lookup_key": record["lookup_key"],
        "parent_sample_id": record["parent_sample_id"],
        "full_subset_signature": record["full_subset_signature"],
        "cache_fingerprint": hashlib.sha256(fingerprint_payload.encode()).hexdigest(),
        "views": views,
        "_lookup": {
            "source_modalities": sorted(record["modality_families"]),
            "source_families": sorted(set(record["modality_families"].values())),
            "modality_families": dict(sorted(record["modality_families"].items())),
        },
    }


def write_shard(
    output: Path,
    shard_index: int,
    buffered_records: list[dict[str, Any]],
    lookup: dict[str, Any],
) -> str:
    payload_records = []
    for local_index, record in enumerate(buffered_records):
        lookup_metadata = record.pop("_lookup")
        lookup[record["lookup_key"]] = {
            "shard": shard_index,
            "index": local_index,
            **lookup_metadata,
        }
        payload_records.append(record)
    path = output / f"shard_{shard_index:05d}.pt"
    temporary_path = output / f".{path.name}.tmp"
    try:
        torch.save({"format": CACHE_FORMAT, "records": payload_records}, temporary_path)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return path.name


def main():
    args = parse_args()
    overrides = {"benchmark_dir": args.benchmark_dir, "train_index": args.train_index, "val_index": args.val_index}
    if args.eval_index:
        overrides[f"{args.eval_split}_index"] = args.eval_index
    config = apply_preset(load_config(args.config, overrides=overrides), args.preset)
    input_protocol = vision_input_protocol(config)
    missing_indexes = [
        split
        for split, fingerprint in input_protocol["index_fingerprints"].items()
        if fingerprint.get("status") != "present"
    ]
    if missing_indexes:
        raise FileNotFoundError(f"Qwen vision cache 需要完整 train/val/test indexes: missing={missing_indexes}")
    output = resolve_project_path(args.output_dir) or Path(args.output_dir)
    if args.verify_only:
        from qpsalm_seg.models.vision_cache import QwenVisionFeatureBank
        bank = QwenVisionFeatureBank(output, decoder_dim=config.decoder_dim)
        if bank.manifest.get("input_protocol") != input_protocol:
            raise ValueError(
                "cache input protocol 与当前 preset 不一致: "
                f"cache={bank.manifest.get('input_protocol')} current={input_protocol}"
            )
        if bank.manifest.get("backend") != "hash-smoke":
            model_dir = validate_qwen_model_dir(config.qwen_model_path)
            expected = {
                "model_revision": local_model_revision(model_dir),
                "processor_revision": local_processor_revision(model_dir),
            }
            for key, value in expected.items():
                if bank.manifest.get(key) != value:
                    raise ValueError(
                        f"cache {key} 与本地 Qwen 不一致: cache={bank.manifest.get(key)} local={value}"
                    )
        print(json.dumps({
            "output_dir": str(output), "format": bank.manifest["format"],
            "renderer_version": bank.manifest["renderer_version"],
            "num_samples": bank.manifest.get("num_samples"), "status": "valid",
        }, ensure_ascii=False))
        return
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"cache exists: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    layers = tuple(int(v) for v in args.layers.split(","))
    spatial_sizes = tuple(int(value) for value in args.spatial_sizes.split(","))
    if len(spatial_sizes) != len(layers) or any(value <= 0 for value in spatial_sizes):
        raise ValueError("--spatial-sizes 必须与 --layers 数量一致且均为正整数")
    if args.shard_size <= 0 or args.view_tokens <= 0:
        raise ValueError("--shard-size 和 --view-tokens 必须为正整数")
    if any(left < right for left, right in zip(spatial_sizes, spatial_sizes[1:])):
        raise ValueError("--spatial-sizes 必须按浅层到深层非递增")
    if args.backend == "qwen" and spatial_sizes[0] < 12:
        raise ValueError("正式 Qwen cache 的浅层 spatial size 至少为 12")
    if args.backend == "qwen":
        encoder = QwenVisionEncoder(
            config.qwen_model_path, torch.device(args.device), layers, spatial_sizes
        )
    else:
        encoder = HashVisionEncoder(layers, spatial_sizes, args.view_tokens)
    revision = encoder.revision
    processor_revision = encoder.processor_revision
    spatial_channels = encoder.spatial_channels
    token_dim = encoder.token_dim
    lookup: dict[str, Any] = {}
    shard_paths: list[str] = []
    buffered_records: list[dict[str, Any]] = []
    num_samples = 0
    peak_buffer_records = 0
    record_iter = iter_records(
        config, args.render_size, args.max_samples, args.eval_split if args.eval_index else None
    )
    try:
        for record in tqdm(record_iter, desc=f"{args.backend}-vision-v3", unit="sample"):
            buffered_records.append(serialize_record(
                record,
                encoder.encode(record),
                spatial_sizes=spatial_sizes,
                view_tokens=args.view_tokens,
                revision=revision,
                processor_revision=processor_revision,
            ))
            num_samples += 1
            peak_buffer_records = max(peak_buffer_records, len(buffered_records))
            if len(buffered_records) >= args.shard_size:
                shard_paths.append(write_shard(output, len(shard_paths), buffered_records, lookup))
                buffered_records = []
        if buffered_records:
            shard_paths.append(write_shard(output, len(shard_paths), buffered_records, lookup))
            buffered_records = []
    finally:
        encoder.close()
    manifest = {
        "format": CACHE_FORMAT, "renderer_version": RENDERER_VERSION,
        "model_revision": revision, "processor_revision": processor_revision,
        "prompt_version": PROMPT_VERSION, "pooling_method": POOLING_METHOD,
        "layers": list(layers), "spatial_sizes": list(spatial_sizes), "render_size": args.render_size,
        "view_tokens_per_view": args.view_tokens,
        "spatial_channels": spatial_channels, "token_dim": token_dim,
        "backend": args.backend, "subset_policy": "dynamic_by_source_modality",
        "input_protocol": input_protocol,
        "num_samples": num_samples, "shards": shard_paths, "lookup": lookup,
        "shard_size": args.shard_size,
        "peak_buffer_records": peak_buffer_records,
    }
    manifest_path = output / "manifest.json"
    temporary_manifest = output / ".manifest.json.tmp"
    try:
        temporary_manifest.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        temporary_manifest.replace(manifest_path)
    finally:
        temporary_manifest.unlink(missing_ok=True)
    print(json.dumps({
        "output_dir": str(output), "num_samples": num_samples,
        "num_shards": len(shard_paths), "peak_buffer_records": peak_buffer_records,
        "format": CACHE_FORMAT,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
