#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""预计算 QMEF 使用的 Qwen 多视图视觉证据缓存。

用途：将 S2、S1、DEM、InSAR 和光学数据渲染为 sensor-aware views，并缓存 Qwen token。
推荐运行命令：PYTHONPATH=SEG_Multi-Source_Landslides python -m
qpsalm_seg.cli.cache_qwen_visual_evidence --config
SEG_Multi-Source_Landslides/configs/qpsalm_small_qwen_cached_core.yaml
--output outputs/qpsalm_multiview_cache_v2.pt --backend qwen --device cuda --overwrite
主要输入：核心 train/val 索引，或通过 --eval-index 指定的 val/test 索引。
主要输出：按父样本组织的 Qwen visual evidence cache v2 .pt。
写入行为：只写 --output；--overwrite 允许覆盖，不修改 benchmark 模态数组。
所属流程：full_multiview preset 的训练/评估准备，也用于 view shuffle/removal 消融。
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from qpsalm_seg.config import QPSalmConfig, load_config
from qpsalm_seg.controllers import select_qwen_model_class, validate_qwen_model_dir
from qpsalm_seg.data import MultiSourceLandslideDataset
from qpsalm_seg.paths import resolve_repo_path
from qpsalm_seg.rendering import RENDERER_VERSION, RenderedView, render_sensor_views
from qpsalm_seg.train_eval import atomic_torch_save, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache Qwen multi-view evidence tokens.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--benchmark-dir", default=None)
    parser.add_argument("--train-index", default=None)
    parser.add_argument("--val-index", default=None)
    parser.add_argument("--eval-index", default=None, help="Cache only an arbitrary val/test JSONL index.")
    parser.add_argument("--output", default="outputs/qpsalm_qwen_multiview_cache_v2.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--qwen-model-path", default=None)
    parser.add_argument("--batch-size", type=int, default=1, help="Reserved; multi-image samples are processed one at a time.")
    parser.add_argument("--render-size", type=int, default=224)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--backend", choices=["qwen", "hash-smoke"], default="qwen")
    parser.add_argument("--hash-hidden-size", type=int, default=1024)
    parser.add_argument(
        "--pooling-method",
        choices=["vision-token", "image-end", "attention", "image-text-delta"],
        default="vision-token",
    )
    parser.add_argument(
        "--shuffle-views-across-samples",
        action="store_true",
        help="将每个 lookup key 配对到另一个父样本的 views，用于视觉真实性对照。",
    )
    parser.add_argument("--shuffle-seed", type=int, default=17)
    parser.add_argument(
        "--drop-view-pattern",
        action="append",
        default=[],
        help="移除名称/描述中包含该字符串的 view；可重复指定，例如 sar、false_color。",
    )
    return parser.parse_args()


def _to_pil(view: RenderedView) -> Any:
    from PIL import Image

    array = (view.image.clamp(0.0, 1.0).permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(array)


def _cache_key(
    lookup_key: str,
    views: list[RenderedView],
    model_revision: str,
    processor_revision: str,
    pooling_method: str,
    ablation: dict[str, Any],
) -> str:
    parts = [
        lookup_key,
        RENDERER_VERSION,
        model_revision,
        processor_revision,
        pooling_method,
        json.dumps(ablation, sort_keys=True),
    ]
    for view in views:
        parts.extend([view.name, view.description, view.content_hash])
    return "qmv:" + hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def collect_samples(
    config: QPSalmConfig,
    render_size: int,
    max_samples: int | None,
    eval_only: bool,
) -> list[dict[str, Any]]:
    stable = replace(config, train_hflip_prob=0.0, train_vflip_prob=0.0)
    samples: dict[str, dict[str, Any]] = {}
    splits = ("val",) if eval_only else ("train", "val")
    limit = int(max_samples) if max_samples is not None and max_samples > 0 else None
    for split in splits:
        split_collected = 0
        split_limit = stable.max_train_samples if split == "train" else stable.max_val_samples
        dataset = MultiSourceLandslideDataset(stable, split=split, max_samples=split_limit)
        for index in tqdm(range(len(dataset)), desc=f"multiview-render-{split}"):
            item = dataset[index]
            lookup_key = str(item["metadata"]["visual_evidence_key"])
            if lookup_key not in samples:
                views = render_sensor_views(item["instances"], size=render_size)
                samples[lookup_key] = {
                    "lookup_key": lookup_key,
                    "views": views,
                    "metadata": {
                        "sample_id": item["metadata"].get("sample_id"),
                        "parent_sample_id": item["metadata"].get("parent_sample_id"),
                        "dataset_name": item["metadata"].get("dataset_name"),
                        "canonical_combo": item["metadata"].get("canonical_combo"),
                        "sensor_combo": item["metadata"].get("sensor_combo"),
                    },
                }
                split_collected += 1
            if limit is not None and split_collected >= limit:
                break
    return list(samples.values())


def apply_visual_ablation(
    samples: list[dict[str, Any]],
    drop_patterns: list[str],
    shuffle_across_samples: bool,
    shuffle_seed: int,
) -> dict[str, Any]:
    """生成 view removal / cross-sample shuffle 对照，不改变模型 lookup key。"""
    normalized = [pattern.strip().lower() for pattern in drop_patterns if pattern.strip()]
    for sample in samples:
        views = list(sample["views"])
        if normalized:
            views = [
                view
                for view in views
                if not any(pattern in f"{view.name} {view.description}".lower() for pattern in normalized)
            ]
        if not views:
            raise ValueError(
                f"view removal 删除了样本 {sample['lookup_key']} 的全部 views；"
                "请缩小 --drop-view-pattern 或对多模态子集生成消融 cache。"
            )
        sample["views"] = views
        sample["metadata"]["visual_source_lookup_key"] = sample["lookup_key"]
    if shuffle_across_samples:
        if len(samples) < 2:
            raise ValueError("cross-sample view shuffle 至少需要 2 个父样本")
        shift = random.Random(int(shuffle_seed)).randint(1, len(samples) - 1)
        order = list(range(shift, len(samples))) + list(range(shift))
        donor_views = [list(samples[index]["views"]) for index in order]
        donor_keys = [str(samples[index]["lookup_key"]) for index in order]
        for sample, views, donor_key in zip(samples, donor_views, donor_keys):
            sample["views"] = views
            sample["metadata"]["visual_source_lookup_key"] = donor_key
    return {
        "drop_view_patterns": normalized,
        "shuffle_views_across_samples": bool(shuffle_across_samples),
        "shuffle_seed": int(shuffle_seed),
    }


def _multi_image_prompt(processor: Any, views: list[RenderedView]) -> str:
    content: list[dict[str, Any]] = []
    for view in views:
        content.append({"type": "text", "text": f"View {view.name}: {view.description}"})
        content.append({"type": "image", "image": _to_pil(view)})
    content.append(
        {
            "type": "text",
            "text": (
                "Summarize the complementary landslide evidence in these optical, multispectral, SAR, terrain, "
                "and deformation views. Preserve each sensor's physical role and spatial evidence."
            ),
        }
    )
    messages = [{"role": "user", "content": content}]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def _contiguous_groups(positions: torch.Tensor) -> list[torch.Tensor]:
    if positions.numel() == 0:
        return []
    splits = torch.where(positions[1:] != positions[:-1] + 1)[0] + 1
    return list(torch.tensor_split(positions, splits.cpu().tolist()))


@torch.no_grad()
def qwen_view_embeddings(
    samples: list[dict[str, Any]],
    model_path: str,
    device: torch.device,
    allow_cpu: bool,
    pooling_method: str,
) -> tuple[list[torch.Tensor], str, str]:
    if device.type != "cuda" and not allow_cpu:
        raise RuntimeError("backend=qwen multi-view cache 默认需要 CUDA；如确需 CPU，添加 --allow-cpu。")
    from transformers import AutoProcessor

    model_dir = validate_qwen_model_dir(model_path)
    model_cls = select_qwen_model_class()
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(str(model_dir), trust_remote_code=True)
    model = model_cls.from_pretrained(str(model_dir), torch_dtype=dtype, trust_remote_code=True).to(device).eval()
    model_revision = str(getattr(model.config, "_commit_hash", None) or getattr(model.config, "_name_or_path", model_dir))
    processor_revision = str(getattr(processor, "_commit_hash", None) or getattr(processor, "name_or_path", model_dir))
    image_token_id = getattr(model.config, "image_token_id", None)
    if image_token_id is None and hasattr(processor, "tokenizer"):
        image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    if image_token_id is None or int(image_token_id) < 0:
        raise RuntimeError("无法从 Qwen config/processor 确定 image token id。")
    rows: list[torch.Tensor] = []
    for sample in tqdm(samples, desc="qwen-multiview-cache"):
        views: list[RenderedView] = sample["views"]
        prompt = _multi_image_prompt(processor, views)
        images = [_to_pil(view) for view in views]
        encoded = processor(text=[prompt], images=images, padding=True, return_tensors="pt")
        encoded = {key: value.to(device) for key, value in encoded.items()}
        outputs = model(**encoded, output_hidden_states=True, return_dict=True, use_cache=False)
        hidden = outputs.hidden_states[-1][0]
        positions = torch.where(encoded["input_ids"][0] == int(image_token_id))[0]
        groups = _contiguous_groups(positions)
        if len(groups) != len(views):
            raise RuntimeError(
                f"Qwen image token groups 与 view 数量不一致: groups={len(groups)} views={len(views)} "
                f"sample={sample['lookup_key']}"
            )
        attention_mask = encoded.get("attention_mask")
        valid_positions = (
            torch.where(attention_mask[0] > 0)[0]
            if attention_mask is not None
            else torch.arange(hidden.shape[0], device=hidden.device)
        )
        image_position_mask = torch.zeros(hidden.shape[0], dtype=torch.bool, device=hidden.device)
        image_position_mask[positions] = True
        text_positions = valid_positions[~image_position_mask[valid_positions]]
        text_baseline = hidden[text_positions].mean(dim=0) if text_positions.numel() else hidden.new_zeros(hidden.shape[-1])
        attention_query = hidden[valid_positions[-1]]
        pooled_views: list[torch.Tensor] = []
        for group in groups:
            image_hidden = hidden[group]
            if pooling_method == "image-end":
                pooled = image_hidden[-1]
            elif pooling_method == "attention":
                scores = (image_hidden.float() @ attention_query.float()) / float(hidden.shape[-1]) ** 0.5
                pooled = (torch.softmax(scores, dim=0).to(image_hidden.dtype)[:, None] * image_hidden).sum(dim=0)
            else:
                pooled = image_hidden.mean(dim=0)
                if pooling_method == "image-text-delta":
                    pooled = pooled - text_baseline
            pooled_views.append(pooled.float().cpu())
        rows.append(torch.stack(pooled_views, dim=0))
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows, model_revision, processor_revision


def hash_view_embeddings(samples: list[dict[str, Any]], hidden_size: int, pooling_method: str) -> list[torch.Tensor]:
    rows: list[torch.Tensor] = []
    for sample in samples:
        tokens = []
        for view in sample["views"]:
            digest = hashlib.sha256(f"{view.content_hash}:{pooling_method}".encode("utf-8")).hexdigest()
            seed = int(digest[:16], 16) % (2**31)
            tokens.append(torch.randn(hidden_size, generator=torch.Generator().manual_seed(seed)))
        rows.append(torch.stack(tokens, dim=0))
    return rows


def _pad_tokens(rows: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    max_views = max(int(row.shape[0]) for row in rows)
    hidden = int(rows[0].shape[1])
    embeddings = torch.zeros((len(rows), max_views, hidden), dtype=torch.float16)
    mask = torch.zeros((len(rows), max_views), dtype=torch.bool)
    for index, row in enumerate(rows):
        count = int(row.shape[0])
        embeddings[index, :count] = row.to(torch.float16)
        mask[index, :count] = True
    return embeddings, mask


def main() -> None:
    args = parse_args()
    overrides = {
        "benchmark_dir": args.benchmark_dir,
        "train_index": args.train_index,
        "val_index": args.val_index,
        "qwen_model_path": args.qwen_model_path,
    }
    eval_only = bool(args.eval_index)
    if eval_only:
        overrides["val_index"] = args.eval_index
        overrides["max_val_samples"] = None
    config = load_config(args.config, overrides=overrides)
    output = resolve_repo_path(args.output)
    if output is None:
        raise ValueError("--output 不能为空")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"cache 已存在: {output}; 使用 --overwrite 重建")
    samples = collect_samples(config, args.render_size, args.max_samples, eval_only)
    if not samples:
        raise RuntimeError("没有可缓存的 multi-view 样本")
    ablation = apply_visual_ablation(
        samples,
        drop_patterns=list(args.drop_view_pattern),
        shuffle_across_samples=bool(args.shuffle_views_across_samples),
        shuffle_seed=int(args.shuffle_seed),
    )
    if args.backend == "qwen":
        device = resolve_device(args.device)
        rows, model_revision, processor_revision = qwen_view_embeddings(
            samples,
            config.qwen_model_path,
            device,
            args.allow_cpu,
            args.pooling_method,
        )
    else:
        rows = hash_view_embeddings(samples, args.hash_hidden_size, args.pooling_method)
        model_revision = f"hash-smoke-{args.hash_hidden_size}"
        processor_revision = "hash-smoke"
    embeddings, view_mask = _pad_tokens(rows)
    cache_keys = [
        _cache_key(
            sample["lookup_key"],
            sample["views"],
            model_revision,
            processor_revision,
            args.pooling_method,
            ablation,
        )
        for sample in samples
    ]
    payload = {
        "format": "qpsalm_qwen_multiview_cache_v2",
        "backend": args.backend,
        "renderer_version": RENDERER_VERSION,
        "model_revision": model_revision,
        "processor_revision": processor_revision,
        "pooling_method": args.pooling_method,
        "lookup_keys": [sample["lookup_key"] for sample in samples],
        "cache_keys": cache_keys,
        "view_embeddings": embeddings,
        "view_mask": view_mask,
        "view_names": [[view.name for view in sample["views"]] for sample in samples],
        "view_descriptions": [[view.description for view in sample["views"]] for sample in samples],
        "view_content_hashes": [[view.content_hash for view in sample["views"]] for sample in samples],
        "view_quality_flags": [[list(view.quality_flags) for view in sample["views"]] for sample in samples],
        "metadata": [sample["metadata"] for sample in samples],
        "ablation": ablation,
        "config": {
            "render_size": int(args.render_size),
            "num_samples": len(samples),
            "max_views": int(view_mask.shape[1]),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(payload, output)
    summary = {
        "output": str(output),
        "format": payload["format"],
        "backend": args.backend,
        "num_samples": len(samples),
        "max_views": int(view_mask.shape[1]),
        "renderer_version": RENDERER_VERSION,
        "pooling_method": args.pooling_method,
        "ablation": ablation,
    }
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
