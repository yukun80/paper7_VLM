#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared runtime utilities for description training, evaluation and joint stages."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler

from qpsalm_seg.paths import resolve_project_path

from .config import SegDescConfig
from .data import DescriptionTaskDataset, collate_description
from .vision_cache import DescriptionVisionFeatureBank


class ParentGroupedRegionBatchSampler(Sampler[list[int]]):
    """Keep same-image DIOR regions adjacent so they become hard negatives."""

    def __init__(
        self,
        dataset: DescriptionTaskDataset,
        batch_size: int,
        *,
        seed: int,
        drop_last: bool,
    ) -> None:
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        if self.batch_size < 2:
            raise ValueError("DIOR parent-grouped sampler 要求 batch_size >= 2")

    def __iter__(self):
        grouped: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(self.dataset.rows):
            grouped[str(row["parent_sample_id"])].append(index)
        generator = random.Random(self.seed + 104729 * int(self.dataset.epoch))
        parents = sorted(grouped)
        generator.shuffle(parents)
        ordered = []
        for parent in parents:
            indices = list(grouped[parent])
            generator.shuffle(indices)
            ordered.extend(indices)
        for start in range(0, len(ordered), self.batch_size):
            batch = ordered[start:start + self.batch_size]
            parents_in_batch = [
                str(self.dataset.rows[index]["parent_sample_id"])
                for index in batch
            ]
            has_same_parent_candidates = len(set(parents_in_batch)) < len(parents_in_batch)
            if (
                has_same_parent_candidates
                and (len(batch) == self.batch_size or not self.drop_last)
            ):
                yield batch

    def __len__(self) -> int:
        return sum(1 for _ in self.__iter__())


def set_description_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def description_device(name: str) -> torch.device:
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested --device {name}, but CUDA is unavailable")
    return torch.device(name)


def description_amp_dtype(config: SegDescConfig, device: torch.device) -> torch.dtype:
    if device.type != "cuda" or config.amp_dtype == "fp32":
        return torch.float32
    return torch.float16 if config.amp_dtype == "fp16" else torch.bfloat16


def description_scaler(config: SegDescConfig, device: torch.device) -> torch.amp.GradScaler:
    return torch.amp.GradScaler(
        device.type,
        enabled=device.type == "cuda" and config.amp_dtype == "fp16",
    )


def validation_split(stage: str) -> str | None:
    if stage in {"mmrs_caption", "rsicap_caption", "dior_alignment"}:
        return "dev"
    if stage == "overfit":
        return "train"
    if stage in {"bridge_expert", "predicted_mask"}:
        return "val"
    return None


def build_description_dataset(
    config: SegDescConfig,
    bank: DescriptionVisionFeatureBank,
    *,
    split: str,
    training: bool,
) -> DescriptionTaskDataset:
    limit = config.max_train_samples if training else config.max_val_samples
    stage = "predicted_mask" if config.evaluation_mode == "fixed_prediction" and not training else config.stage
    return DescriptionTaskDataset(
        stage=stage,
        split=split,
        vision_bank=bank,
        description_benchmark=config.description_benchmark,
        bridge_benchmark=config.bridge_benchmark,
        predicted_index=config.predicted_index,
        seed=config.seed,
        max_samples=max(0, int(limit or 0)),
        training=training,
        evaluation_mode=config.evaluation_mode,
        rsicap_mmrs_fraction=config.rsicap_mmrs_fraction,
        predicted_mask_fraction=config.predicted_mask_fraction,
    )


def build_description_loader(
    dataset: DescriptionTaskDataset,
    config: SegDescConfig,
    *,
    training: bool,
    batch_size: int | None = None,
) -> DataLoader:
    effective_batch_size = int(batch_size or config.batch_size)
    if training and config.stage == "dior_alignment":
        sampler = ParentGroupedRegionBatchSampler(
            dataset,
            effective_batch_size,
            seed=config.seed,
            drop_last=True,
        )
        if len(sampler) == 0:
            raise ValueError(
                "DIOR training 没有包含同一 parent 多个候选区域的完整 batch；"
                "请降低 batch size 或检查 region-pair index"
            )
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=int(config.num_workers),
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_description,
        )
    return DataLoader(
        dataset,
        batch_size=effective_batch_size,
        shuffle=bool(training),
        num_workers=int(config.num_workers),
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_description,
        drop_last=False,
    )


def move_description_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        **batch,
        "region_masks": batch["region_masks"].to(device=device, non_blocking=True),
        "weights": batch["weights"].to(device=device, non_blocking=True),
    }


def write_json(path: str | Path, payload: Any) -> None:
    resolved = resolve_project_path(path) or Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved.with_suffix(resolved.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(resolved)


def append_jsonl(path: str | Path, payload: dict[str, Any]) -> None:
    resolved = resolve_project_path(path) or Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
