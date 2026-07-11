#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""预计算 Qwen semantic-evidence 文本缓存。

用途：用冻结 Qwen3-VL 编码 task、condition 和 evidence reasoning 文本，供训练复用。
推荐运行命令：PYTHONPATH=SEG_Multi-Source_Landslides python -m
qpsalm_seg.cli.cache_qwen_embeddings --config
SEG_Multi-Source_Landslides/configs/qpsalm_small_qwen_cached_core.yaml
--output outputs/qpsalm_condition_cache.pt --backend qwen --device cuda --overwrite
主要输入：核心 train/val 索引，或通过 --eval-index 指定的 val/test 索引。
主要输出：Qwen 文本 hidden-state 缓存 .pt。
写入行为：只写 --output；--overwrite 允许覆盖已有缓存，不修改 benchmark。
所属流程：QPSALM 训练/推理准备；真实 Qwen backend 需要本地模型权重。
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from qpsalm_seg.config import load_config
from qpsalm_seg.controllers import FrozenQwenController
from qpsalm_seg.paths import resolve_repo_path
from qpsalm_seg.qwen_cache import collect_required_qwen_texts
from qpsalm_seg.train_eval import atomic_torch_save, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache frozen Qwen semantic-evidence text states.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--benchmark-dir", default=None)
    parser.add_argument("--train-index", default=None)
    parser.add_argument("--val-index", default=None)
    parser.add_argument("--eval-index", default=None, help="Optional val/test index for standalone inference cache.")
    parser.add_argument("--eval-split", choices=["val", "test"], default="test")
    parser.add_argument("--output", default="outputs/qpsalm_qwen_condition_cache.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--qwen-model-path", default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-texts", type=int, default=None)
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--backend",
        choices=["qwen", "hash-smoke"],
        default="qwen",
        help="qwen runs the real frozen Qwen path; hash-smoke only tests cache plumbing.",
    )
    parser.add_argument("--hash-hidden-size", type=int, default=1024)
    return parser.parse_args()


def collect_controller_texts(
    config_path: str,
    train_index: str | None,
    val_index: str | None,
    eval_index: str | None = None,
    eval_split: str = "test",
    benchmark_dir: str | None = None,
) -> tuple[list[str], dict[str, Any]]:
    overrides: dict[str, Any] = {
        "benchmark_dir": benchmark_dir,
        "train_index": train_index,
        "val_index": val_index,
    }
    if eval_index:
        overrides["test_index" if eval_split == "test" else "val_index"] = eval_index
        overrides["max_val_samples"] = 0
    config = load_config(
        config_path,
        overrides=overrides,
    )
    splits = (eval_split,) if eval_index else ("train", "val")
    texts, report = collect_required_qwen_texts(config, splits=splits)
    report["config"] = config.__dict__
    return texts, report


def hash_smoke_embeddings(texts: list[str], hidden_size: int) -> torch.Tensor:
    """生成确定性伪 hidden state，只用于离线验证 qwen_cache 管线。"""
    rows = []
    for text in texts:
        digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
        seed = int(digest[:16], 16) % (2**31)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        rows.append(torch.randn(hidden_size, generator=generator, dtype=torch.float32))
    return torch.stack(rows, dim=0)


@torch.no_grad()
def qwen_embeddings(
    texts: list[str],
    model_path: str,
    decoder_dim: int,
    device: torch.device,
    batch_size: int,
    allow_cpu: bool,
) -> torch.Tensor:
    controller: FrozenQwenController | None = None
    try:
        controller = FrozenQwenController(
            model_path=model_path,
            decoder_dim=decoder_dim,
            device=device,
            allow_cpu=allow_cpu,
        )
        chunks = []
        for start in tqdm(range(0, len(texts), batch_size), desc="qwen-condition-cache"):
            batch = texts[start : start + batch_size]
            pooled = controller._pool_qwen(batch, device=device)
            chunks.append(pooled.detach().float().cpu())
        return torch.cat(chunks, dim=0)
    finally:
        del controller
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
            "benchmark_dir": args.benchmark_dir,
            "train_index": args.train_index,
            "val_index": args.val_index,
            "test_index": args.eval_index if args.eval_split == "test" else None,
            "qwen_model_path": args.qwen_model_path,
        },
    )
    texts, source_report = collect_controller_texts(
        args.config,
        args.train_index,
        args.val_index,
        eval_index=args.eval_index,
        eval_split=args.eval_split,
        benchmark_dir=args.benchmark_dir,
    )
    if args.max_texts is not None and args.max_texts > 0:
        texts = texts[: args.max_texts]
    if not texts:
        raise RuntimeError("没有收集到可缓存的 semantic-evidence text。")

    if args.backend == "hash-smoke":
        embeddings = hash_smoke_embeddings(texts, hidden_size=int(args.hash_hidden_size))
        device_name = "cpu"
    else:
        device = resolve_device(args.device)
        embeddings = qwen_embeddings(
            texts=texts,
            model_path=config.qwen_model_path,
            decoder_dim=config.decoder_dim,
            device=device,
            batch_size=max(1, int(args.batch_size)),
            allow_cpu=bool(args.allow_cpu or config.allow_qwen_cpu),
        )
        device_name = str(device)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "qpsalm_qwen_condition_cache_v1",
        "backend": args.backend,
        "model_path": config.qwen_model_path,
        "device": device_name,
        "hidden_size": int(embeddings.shape[1]),
        "texts": texts,
        "embeddings": embeddings.contiguous(),
        "source": source_report,
    }
    atomic_torch_save(payload, output_path)
    print(
        json.dumps(
            {
                "output": str(output_path),
                "backend": args.backend,
                "num_texts": len(texts),
                "hidden_size": int(embeddings.shape[1]),
                "model_path": config.qwen_model_path,
                "text_types": source_report.get("text_types"),
                "source": source_report["splits"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
