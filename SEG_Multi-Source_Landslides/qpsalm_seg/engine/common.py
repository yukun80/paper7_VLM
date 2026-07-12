#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared runtime construction for trainer and evaluator."""

from __future__ import annotations

import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from qpsalm_seg.config import QPSalmConfig
from qpsalm_seg.data import MultiSourceLandslideDataset, SizeBucketBatchSampler, qpsalm_collate
from qpsalm_seg.models import MultiSourceQwenPSALMSeg


def amp_dtype(config: QPSalmConfig, device: torch.device) -> torch.dtype:
    if device.type != "cuda" or config.amp_dtype == "fp32":
        return torch.float32
    if config.amp_dtype == "fp16":
        return torch.float16
    if config.amp_dtype == "bf16":
        return torch.bfloat16
    raise ValueError(f"未知 amp_dtype={config.amp_dtype!r}")


def autocast_enabled(config: QPSalmConfig, device: torch.device) -> bool:
    return device.type == "cuda" and config.amp_dtype != "fp32"


def create_grad_scaler(config: QPSalmConfig, device: torch.device):
    return torch.amp.GradScaler(
        device.type,
        enabled=device.type == "cuda" and config.amp_dtype == "fp16",
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def resolve_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested --device {requested}, but CUDA is unavailable in this session.")
    return torch.device(requested)


def build_model(config: QPSalmConfig, device: torch.device) -> MultiSourceQwenPSALMSeg:
    model = MultiSourceQwenPSALMSeg(config, device)
    if config.controller == "qwen_mask_query" and config.qwen_4bit:
        for module in (model.vision_bank, model.sane, model.qmef, model.pmrd):
            if module is not None:
                module.to(device)
    else:
        model.to(device)
    return model


def _loader(dataset, config: QPSalmConfig, *, training: bool) -> DataLoader:
    common = {
        "num_workers": config.num_workers,
        "collate_fn": qpsalm_collate,
        "pin_memory": torch.cuda.is_available(),
    }
    if config.num_workers > 0:
        common.update({
            "prefetch_factor": max(1, int(config.prefetch_factor)),
            "persistent_workers": bool(config.persistent_workers),
        })
    if config.use_size_buckets and config.size_buckets:
        sampler = SizeBucketBatchSampler(
            dataset,
            config.batch_size,
            shuffle=training,
            seed=config.seed,
            task_weights=(
                config.task_sampling_ratios
                if training else {"global": 1.0, "referring": 1.0, "no_target": 1.0}
            ),
            balance_tasks=training,
        )
        return DataLoader(dataset, batch_sampler=sampler, **common)
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=training, **common)


def build_dataloaders(config: QPSalmConfig) -> tuple[DataLoader, DataLoader]:
    train_dataset = MultiSourceLandslideDataset(
        config, "train", max_samples=config.max_train_samples, shuffle_seed=config.seed
    )
    val_dataset = MultiSourceLandslideDataset(
        config,
        "val",
        max_samples=config.monitor_val_samples,
        monitor_seed=config.seed + 1009,
    )
    return _loader(train_dataset, config, training=True), _loader(val_dataset, config, training=False)


def build_eval_loader(config: QPSalmConfig, split: str) -> DataLoader:
    dataset = MultiSourceLandslideDataset(config, split, max_samples=config.max_val_samples)
    return _loader(dataset, config, training=False)


def cosine_lr(step: int, max_steps: int, warmup_steps: int) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return float(step + 1) / float(max(1, warmup_steps))
    progress = (step - warmup_steps) / float(max(1, max_steps - warmup_steps))
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))
