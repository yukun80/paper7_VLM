#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""预计算 Qwen visual evidence embedding 缓存。

脚本作用：用冻结 Qwen3-VL 对 visual preview + proposal/condition prompt 做
图文编码，生成训练时可复用的 visual evidence hidden state。
主要输入：核心 train/val instruction JSONL 与物化后的多源模态 .npy。
主要输出：outputs/qpsalm_qwen_visual_evidence_cache.pt。
是否改写原始数据：不会。
典型用法：python -m qpsalm_seg.cli.cache_qwen_visual_evidence --config ... --backend qwen。
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from qpsalm_seg.config import QPSalmConfig, load_config
from qpsalm_seg.controllers import select_qwen_model_class, validate_qwen_model_dir
from qpsalm_seg.data import MultiSourceLandslideDataset, resolve_repo_path
from qpsalm_seg.train_eval import atomic_torch_save, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache frozen Qwen visual evidence hidden states for QPSALM.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-index", default=None)
    parser.add_argument("--val-index", default=None)
    parser.add_argument("--output", default="outputs/qpsalm_qwen_visual_evidence_cache.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--qwen-model-path", default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--backend",
        choices=["qwen", "hash-smoke"],
        default="qwen",
        help="qwen runs real image-text Qwen; hash-smoke only checks cache plumbing.",
    )
    parser.add_argument("--hash-hidden-size", type=int, default=1024)
    return parser.parse_args()


def _preview_to_pil(preview: torch.Tensor) -> Any:
    """把 [3,H,W] preview tensor 转为 PIL.Image，PIL 只在真实 Qwen 路径中需要。"""
    try:
        from PIL import Image
    except Exception as exc:  # pragma: no cover - optional env dependency
        raise RuntimeError("backend=qwen 需要 Pillow 用于把 visual_preview 转为 PIL.Image。") from exc
    arr = preview.detach().cpu().float().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    arr_u8 = (arr * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr_u8)


def collect_visual_evidence_samples(
    config: QPSalmConfig,
    max_samples: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """收集需要缓存的唯一图文 visual evidence 样本。

    训练 split 默认会做随机翻转，这里强制关闭几何增强，保证 cache key 对应的是
    稳定 preview，而不是某次随机增强后的图像。
    """
    cache_config = replace(config, train_hflip_prob=0.0, train_vflip_prob=0.0)
    samples: dict[str, dict[str, Any]] = {}
    split_reports: list[dict[str, Any]] = []
    limit = int(max_samples) if max_samples is not None and max_samples > 0 else None
    for split in ("train", "val"):
        split_max_samples = cache_config.max_train_samples if split == "train" else cache_config.max_val_samples
        dataset = MultiSourceLandslideDataset(cache_config, split=split, max_samples=split_max_samples)
        rows_seen = 0
        rows_added = 0
        for idx in tqdm(range(len(dataset)), desc=f"visual-evidence-scan-{split}"):
            item = dataset[idx]
            rows_seen += 1
            key = str(item["metadata"]["visual_evidence_key"])
            if key not in samples:
                meta = item["metadata"]
                samples[key] = {
                    "key": key,
                    "text": meta["condition_text"],
                    "proposal_context_text": meta["proposal_context_text"],
                    "condition_prompt_text": meta["condition_prompt_text"],
                    "evidence_reasoning_text": meta["evidence_reasoning_text"],
                    "visual_preview": item["visual_preview"],
                    "metadata": {
                        "sample_id": meta.get("sample_id"),
                        "dataset_name": meta.get("dataset_name"),
                        "template_id": meta.get("template_id"),
                        "canonical_combo": meta.get("canonical_combo"),
                        "raw_combo": meta.get("raw_combo"),
                        "sensor_combo": meta.get("sensor_combo"),
                        "normalization_combo": meta.get("normalization_combo"),
                        "visual_preview_source": meta.get("visual_preview_source"),
                        "gsd_m": meta.get("gsd_m"),
                    },
                }
                rows_added += 1
            if limit is not None and len(samples) >= limit:
                break
        split_reports.append(
            {
                "split": split,
                "rows_seen": rows_seen,
                "rows_added_unique": rows_added,
                "max_samples": split_max_samples,
                "skipped": dict(dataset.skipped),
            }
        )
        if limit is not None and len(samples) >= limit:
            break
    return list(samples.values()), {"splits": split_reports, "num_unique": len(samples), "max_samples_cli": limit}


def hash_smoke_embeddings(samples: list[dict[str, Any]], hidden_size: int) -> torch.Tensor:
    """生成确定性伪 visual hidden state，只用于离线验证 cache 管线。"""
    rows = []
    for sample in samples:
        seed_text = f"{sample['key']}\n{sample['text']}"
        digest = hashlib.sha1(seed_text.encode("utf-8")).hexdigest()
        seed = int(digest[:16], 16) % (2**31)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        rows.append(torch.randn(hidden_size, generator=generator, dtype=torch.float32))
    return torch.stack(rows, dim=0)


def _qwen_prompt(processor: Any, image: Any, text: str) -> str:
    """按 Qwen-VL chat template 构造图文 prompt，缺模板时退回纯文本。"""
    if not hasattr(processor, "apply_chat_template"):
        return text
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": text},
            ],
        }
    ]
    try:
        return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    except Exception:
        fallback_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": text},
                ],
            }
        ]
        return processor.apply_chat_template(fallback_messages, tokenize=False, add_generation_prompt=False)


@torch.no_grad()
def qwen_visual_embeddings(
    samples: list[dict[str, Any]],
    model_path: str,
    device: torch.device,
    batch_size: int,
    allow_cpu: bool,
) -> torch.Tensor:
    """运行冻结 Qwen3-VL 图文路径，返回 pooled hidden state。"""
    if device.type != "cuda" and not allow_cpu:
        raise RuntimeError("backend=qwen visual evidence cache 默认需要 CUDA；如确需 CPU，添加 --allow-cpu。")
    try:
        from transformers import AutoProcessor
    except Exception as exc:  # pragma: no cover - depends on optional env
        raise RuntimeError("backend=qwen 需要 transformers.AutoProcessor 支持 Qwen3-VL 图文输入。") from exc
    model_dir = validate_qwen_model_dir(model_path)
    model_cls = select_qwen_model_class()
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(str(model_dir), trust_remote_code=True)
    model = model_cls.from_pretrained(str(model_dir), torch_dtype=dtype, trust_remote_code=True)
    model.eval()
    model.to(device)
    chunks: list[torch.Tensor] = []
    try:
        for start in tqdm(range(0, len(samples), batch_size), desc="qwen-visual-evidence-cache"):
            batch = samples[start : start + batch_size]
            images = [_preview_to_pil(item["visual_preview"]) for item in batch]
            texts = [_qwen_prompt(processor, image, str(item["text"])) for image, item in zip(images, batch)]
            encoded = processor(text=texts, images=images, padding=True, return_tensors="pt")
            encoded = {key: value.to(device) if torch.is_tensor(value) else value for key, value in encoded.items()}
            outputs = model(
                **encoded,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )
            hidden = outputs.hidden_states[-1]
            attention_mask = encoded.get("attention_mask")
            if attention_mask is None:
                pooled = hidden.mean(dim=1)
            else:
                mask = attention_mask.to(hidden.dtype).unsqueeze(-1)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            chunks.append(pooled.detach().float().cpu())
        return torch.cat(chunks, dim=0)
    finally:
        del model
        gc.collect()
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass


def main() -> None:
    args = parse_args()
    output_path = resolve_repo_path(args.output)
    if output_path is None:
        raise FileNotFoundError(args.output)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"输出已存在，使用 --overwrite 覆盖: {output_path}")
    config = load_config(
        args.config,
        overrides={
            "train_index": args.train_index,
            "val_index": args.val_index,
            "qwen_model_path": args.qwen_model_path,
        },
    )
    samples, source_report = collect_visual_evidence_samples(config, max_samples=args.max_samples)
    if not samples:
        raise RuntimeError("没有收集到可缓存的 visual evidence 样本。")

    if args.backend == "hash-smoke":
        embeddings = hash_smoke_embeddings(samples, hidden_size=int(args.hash_hidden_size))
        device_name = "cpu"
    else:
        device = resolve_device(args.device)
        embeddings = qwen_visual_embeddings(
            samples=samples,
            model_path=config.qwen_model_path,
            device=device,
            batch_size=max(1, int(args.batch_size)),
            allow_cpu=bool(args.allow_cpu or config.allow_qwen_cpu),
        )
        device_name = str(device)

    payload = {
        "format": "qpsalm_qwen_visual_evidence_cache_v1",
        "backend": args.backend,
        "model_path": config.qwen_model_path,
        "device": device_name,
        "hidden_size": int(embeddings.shape[1]),
        "keys": [str(item["key"]) for item in samples],
        "texts": [str(item["text"]) for item in samples],
        "metadata": [item["metadata"] for item in samples],
        "embeddings": embeddings.contiguous(),
        "source": source_report,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(payload, output_path)
    print(
        json.dumps(
            {
                "output": str(output_path),
                "backend": args.backend,
                "num_samples": len(samples),
                "hidden_size": int(embeddings.shape[1]),
                "model_path": config.qwen_model_path,
                "source": source_report,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
