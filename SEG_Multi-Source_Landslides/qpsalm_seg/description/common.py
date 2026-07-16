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

    protocol = "qpsalm_parent_grouped_region_batch_sampler_v2_epoch_addressable"

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
        self.epoch = 0
        if self.batch_size < 2:
            raise ValueError("DIOR parent-grouped sampler 要求 batch_size >= 2")

    def set_epoch(self, epoch: int) -> None:
        if int(epoch) < 0:
            raise ValueError("sampler epoch 必须为非负整数")
        self.epoch = int(epoch)

    def __iter__(self):
        grouped: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(self.dataset.rows):
            grouped[str(row["parent_sample_id"])].append(index)
        generator = random.Random(self.seed + 104729 * self.epoch)
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


class EpochShuffleBatchSampler(Sampler[list[int]]):
    """Epoch-addressable shuffle used by resumable description streams."""

    protocol = "qpsalm_epoch_shuffle_batch_sampler_v1_cursor_replay"

    def __init__(
        self,
        dataset,
        batch_size: int,
        *,
        seed: int,
        drop_last: bool,
    ) -> None:
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0
        if self.batch_size <= 0:
            raise ValueError("description batch_size 必须为正整数")

    def set_epoch(self, epoch: int) -> None:
        if int(epoch) < 0:
            raise ValueError("sampler epoch 必须为非负整数")
        self.epoch = int(epoch)

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        random.Random(self.seed + 104729 * self.epoch).shuffle(indices)
        for start in range(0, len(indices), self.batch_size):
            batch = indices[start:start + self.batch_size]
            if len(batch) == self.batch_size or not self.drop_last:
                yield batch

    def __len__(self) -> int:
        size = len(self.dataset)
        if self.drop_last:
            return size // self.batch_size
        return (size + self.batch_size - 1) // self.batch_size


def set_loader_epoch(
    loader: DataLoader,
    epoch: int,
    *,
    loader_seed: int | None = None,
) -> None:
    """Address one loader epoch without consuming global training RNG state."""
    epoch = int(epoch)
    if epoch < 0:
        raise ValueError("loader epoch 必须为非负整数")
    dataset = loader.dataset
    if hasattr(dataset, "set_epoch"):
        dataset.set_epoch(epoch)
    batch_sampler = getattr(loader, "batch_sampler", None)
    if hasattr(batch_sampler, "set_epoch"):
        batch_sampler.set_epoch(epoch)
    sampler = getattr(loader, "sampler", None)
    if sampler is not batch_sampler and hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)

    seed = (
        int(loader_seed)
        if loader_seed is not None
        else int(getattr(loader, "_qpsalm_loader_seed", 0))
    )
    loader._qpsalm_loader_seed = seed
    generator = getattr(loader, "generator", None)
    if generator is None:
        generator = torch.Generator()
        loader.generator = generator
    # DataLoader worker base seeds must also be a pure function of epoch.
    generator.manual_seed(seed + 1_000_003 * epoch)


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


def predicted_index_for_dataset(
    config: SegDescConfig,
    *,
    split: str,
    training: bool,
) -> str | None:
    """Keep OOF train masks separate from fixed val/test predictions."""
    if not training and split != "train" and config.predicted_val_index:
        return config.predicted_val_index
    return config.predicted_index


def validate_predicted_training_indexes(
    config: SegDescConfig,
    *,
    stage: str,
) -> dict[str, str] | None:
    """Fail before model loading when a predicted-mask trainer lacks fixed val data."""
    if stage != "predicted_mask":
        return None
    references = {
        "train": config.predicted_index,
        "val": config.predicted_val_index,
    }
    missing = [name for name, value in references.items() if not value]
    if missing:
        raise ValueError(
            "predicted-mask training 必须分别提供 OOF train 与 fixed val index；"
            f"missing={missing}"
        )
    resolved = {
        name: resolve_project_path(value) or Path(str(value))
        for name, value in references.items()
    }
    absent = [name for name, path in resolved.items() if not path.is_file()]
    if absent:
        raise FileNotFoundError(
            "predicted-mask training index 不存在: "
            + ", ".join(f"{name}={resolved[name]}" for name in absent)
        )
    if resolved["train"].resolve(strict=False) == resolved["val"].resolve(strict=False):
        raise ValueError("OOF train index 与 fixed val index 必须是不同产物")
    return {
        name: str(path.resolve(strict=False)) for name, path in resolved.items()
    }


def build_description_dataset(
    config: SegDescConfig,
    bank: DescriptionVisionFeatureBank,
    *,
    split: str,
    training: bool,
) -> DescriptionTaskDataset:
    limit = config.max_train_samples if training else config.max_val_samples
    stage = "predicted_mask" if config.evaluation_mode == "fixed_prediction" and not training else config.stage
    predicted_index = predicted_index_for_dataset(
        config, split=split, training=training
    )
    return DescriptionTaskDataset(
        stage=stage,
        split=split,
        vision_bank=bank,
        description_benchmark=config.description_benchmark,
        bridge_benchmark=config.bridge_benchmark,
        predicted_index=predicted_index,
        seed=config.seed,
        max_samples=max(0, int(limit or 0)),
        training=training,
        evaluation_mode=config.evaluation_mode,
        evaluation_source_dataset=(
            None if training else config.evaluation_source_dataset
        ),
        evaluation_region_source=(
            None if training else config.evaluation_region_source
        ),
        rsicap_mmrs_fraction=config.rsicap_mmrs_fraction,
        predicted_mask_fraction=config.predicted_mask_fraction,
        d4_curriculum_sampling_seed=config.d4_curriculum_sampling_seed,
    )


def build_description_loader(
    dataset: DescriptionTaskDataset,
    config: SegDescConfig,
    *,
    training: bool,
    batch_size: int | None = None,
    sampler_seed: int | None = None,
) -> DataLoader:
    effective_batch_size = int(batch_size or config.batch_size)
    effective_seed = int(config.seed if sampler_seed is None else sampler_seed)
    loader_generator = torch.Generator()
    loader_generator.manual_seed(effective_seed)
    if training and config.stage == "dior_alignment":
        sampler = ParentGroupedRegionBatchSampler(
            dataset,
            effective_batch_size,
            seed=effective_seed,
            drop_last=True,
        )
        if len(sampler) == 0:
            raise ValueError(
                "DIOR training 没有包含同一 parent 多个候选区域的完整 batch；"
                "请降低 batch size 或检查 region-pair index"
            )
        loader = DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=int(config.num_workers),
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_description,
            generator=loader_generator,
        )
        set_loader_epoch(loader, 0, loader_seed=effective_seed)
        return loader
    if training:
        sampler = EpochShuffleBatchSampler(
            dataset,
            effective_batch_size,
            seed=effective_seed,
            drop_last=False,
        )
        loader = DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=int(config.num_workers),
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_description,
            generator=loader_generator,
        )
        set_loader_epoch(loader, 0, loader_seed=effective_seed)
        return loader
    loader = DataLoader(
        dataset,
        batch_size=effective_batch_size,
        shuffle=False,
        num_workers=int(config.num_workers),
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_description,
        drop_last=False,
        generator=loader_generator,
    )
    loader._qpsalm_loader_seed = effective_seed
    return loader


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
    # Python 默认允许 NaN/Infinity；研究产物必须在替换旧文件前拒绝非标准 JSON。
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    ) + "\n"
    try:
        temporary.write_text(encoded, encoding="utf-8")
        temporary.replace(resolved)
    finally:
        temporary.unlink(missing_ok=True)


def append_jsonl(path: str | Path, payload: dict[str, Any]) -> None:
    resolved = resolve_project_path(path) or Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n"
    with resolved.open("a", encoding="utf-8") as handle:
        handle.write(encoded)
