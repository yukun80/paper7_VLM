#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared runtime utilities for description training, evaluation and joint stages."""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler

from qpsalm_seg.paths import resolve_project_path

from ..protocols.config import SegDescConfig
from .datasets import DescriptionTaskDataset, collate_description
from ..protocols.io import append_jsonl as _append_jsonl
from ..protocols.io import atomic_write_json
from ..protocols.versions import DESCRIPTION_COLLATOR_AUDIT_PROTOCOL
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


class DMinusOneTaskPathBatchSampler(Sampler[list[int]]):
    """Keep every optimizer window on one explicit visual-routing path.

    D-1 mixes global captions with box/mask/null examples in one population.
    Grouping ``grad_accum_steps`` consecutive microbatches makes its gradient
    artifact capable of proving both global MGRR isolation and grounded-region
    backpropagation, instead of accepting an ambiguous mixed gradient.
    """

    protocol = "qpsalm_d_minus_one_task_path_batch_sampler_v1_window_homogeneous"

    def __init__(
        self,
        dataset: DescriptionTaskDataset,
        batch_size: int,
        *,
        gradient_window_batches: int,
        seed: int,
    ) -> None:
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.gradient_window_batches = int(gradient_window_batches)
        self.seed = int(seed)
        self.drop_last = False
        self.epoch = 0
        if self.batch_size <= 0 or self.gradient_window_batches <= 0:
            raise ValueError("D-1 batch size/gradient window 必须为正整数")
        self._path_indices()

    def set_epoch(self, epoch: int) -> None:
        if int(epoch) < 0:
            raise ValueError("sampler epoch 必须为非负整数")
        self.epoch = int(epoch)

    def _path_indices(self) -> dict[str, list[int]]:
        grouped = {"global_caption": [], "region_description": []}
        for index, row in enumerate(self.dataset.rows):
            category = str(row.get("_d_minus_one_category") or "")
            if category not in {"global", "box", "mask", "null"}:
                raise ValueError(
                    "D-1 sampler 遇到没有协议类别的 row: "
                    f"index={index} category={category!r}"
                )
            path = (
                "global_caption" if category == "global"
                else "region_description"
            )
            grouped[path].append(index)
        missing = [name for name, indices in grouped.items() if not indices]
        if missing:
            raise ValueError(f"D-1 sampler 缺少 task path: {missing}")
        return grouped

    def _path_windows(
        self,
        indices: list[int],
        generator: random.Random,
    ) -> list[list[list[int]]]:
        by_category: dict[str, list[int]] = defaultdict(list)
        for index in indices:
            by_category[str(
                self.dataset.rows[index]["_d_minus_one_category"]
            )].append(index)
        for values in by_category.values():
            generator.shuffle(values)
        categories = sorted(by_category)
        values = [
            by_category[category][offset]
            for offset in range(max(map(len, by_category.values())))
            for category in categories
            if offset < len(by_category[category])
        ]
        batches = [
            values[start:start + self.batch_size]
            for start in range(0, len(values), self.batch_size)
        ]
        target = (
            (len(batches) + self.gradient_window_batches - 1)
            // self.gradient_window_batches
            * self.gradient_window_batches
        )
        # 过拟合协议允许在 epoch 尾部重放真实行；这样每个优化窗口仍是
        # 单一路径，且下个 epoch 的 cursor 必然从窗口边界开始。
        batches.extend(
            list(batches[index % len(batches)])
            for index in range(target - len(batches))
        )
        return [
            batches[start:start + self.gradient_window_batches]
            for start in range(0, len(batches), self.gradient_window_batches)
        ]

    def __iter__(self):
        generator = random.Random(self.seed + 104729 * self.epoch)
        blocks: list[tuple[str, list[list[int]]]] = []
        for path, indices in self._path_indices().items():
            blocks.extend(
                (path, window)
                for window in self._path_windows(indices, generator)
            )
        generator.shuffle(blocks)
        for _path, window in blocks:
            yield from window

    def __len__(self) -> int:
        return sum(
            len(self._path_windows(indices, random.Random(0)))
            * self.gradient_window_batches
            for indices in self._path_indices().values()
        )


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
    if device.type != "cuda" or config.training.amp_dtype == "fp32":
        return torch.float32
    return torch.float16 if config.training.amp_dtype == "fp16" else torch.bfloat16


def description_scaler(config: SegDescConfig, device: torch.device) -> torch.amp.GradScaler:
    return torch.amp.GradScaler(
        device.type,
        enabled=device.type == "cuda" and config.training.amp_dtype == "fp16",
    )


def validation_split(stage: str) -> str | None:
    from ..protocols.stages import get_stage_spec

    return get_stage_spec(stage).validation_split


def predicted_index_for_dataset(
    config: SegDescConfig,
    *,
    split: str,
    training: bool,
) -> str | None:
    """Keep OOF train masks separate from fixed val/test predictions."""
    if not training and split != "train" and config.data.predicted_val_index:
        return config.data.predicted_val_index
    return config.data.predicted_index


def validate_predicted_training_indexes(
    config: SegDescConfig,
    *,
    stage: str,
) -> dict[str, str] | None:
    """Fail before model loading when a predicted-mask trainer lacks fixed val data."""
    if stage != "predicted_mask":
        return None
    references = {
        "train": config.data.predicted_index,
        "val": config.data.predicted_val_index,
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
    limit = (
        config.data.max_train_samples
        if training else config.data.max_val_samples
    )
    stage = (
        "predicted_mask"
        if config.evaluation.evaluation_mode == "fixed_prediction" and not training
        else config.training.stage
    )
    predicted_index = predicted_index_for_dataset(
        config, split=split, training=training
    )
    return DescriptionTaskDataset(
        stage=stage,
        split=split,
        vision_bank=bank,
        description_benchmark=config.data.description_benchmark,
        bridge_benchmark=config.data.bridge_benchmark,
        predicted_index=predicted_index,
        seed=config.training.seed,
        max_samples=max(0, int(limit or 0)),
        training=training,
        evaluation_mode=config.evaluation.evaluation_mode,
        evaluation_source_dataset=(
            None if training else config.evaluation.evaluation_source_dataset
        ),
        evaluation_region_source=(
            None if training else config.evaluation.evaluation_region_source
        ),
        rsicap_mmrs_fraction=config.data.rsicap_mmrs_fraction,
        predicted_mask_fraction=config.data.predicted_mask_fraction,
        d4_curriculum_sampling_seed=config.data.d4_curriculum_sampling_seed,
    )


def build_description_loader(
    dataset: DescriptionTaskDataset,
    config: SegDescConfig,
    *,
    training: bool,
    batch_size: int | None = None,
    sampler_seed: int | None = None,
) -> DataLoader:
    effective_batch_size = int(batch_size or config.training.batch_size)
    effective_seed = int(config.training.seed if sampler_seed is None else sampler_seed)
    loader_generator = torch.Generator()
    loader_generator.manual_seed(effective_seed)
    if training and config.training.stage == "overfit":
        sampler = DMinusOneTaskPathBatchSampler(
            dataset,
            effective_batch_size,
            gradient_window_batches=int(config.training.grad_accum_steps),
            seed=effective_seed,
        )
        loader = DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=int(config.data.num_workers),
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_description,
            generator=loader_generator,
        )
        set_loader_epoch(loader, 0, loader_seed=effective_seed)
        return loader
    if training and config.training.stage == "dior_alignment":
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
            num_workers=int(config.data.num_workers),
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
            num_workers=int(config.data.num_workers),
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
        num_workers=int(config.data.num_workers),
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


def description_collator_audit(batch: Any) -> dict[str, Any]:
    """Validate the exact tensor contract consumed by description training."""

    if not isinstance(batch, dict):
        raise ValueError("description collator 输出必须为 dict")
    sequence_fields = {
        "requests", "instructions", "target_texts", "reference_texts",
        "structured_outputs", "use_region_tokens", "metadata",
    }
    required = {*sequence_fields, "region_masks", "weights"}
    missing = sorted(required - set(batch))
    if missing:
        raise ValueError(f"description collator 缺少字段: {missing}")
    metadata = batch["metadata"]
    if not isinstance(metadata, list) or not metadata:
        raise ValueError("description collator metadata 为空")
    batch_size = len(metadata)
    for name in sorted(sequence_fields):
        values = batch[name]
        if not isinstance(values, list) or len(values) != batch_size:
            raise ValueError(
                f"description collator {name} 长度与 batch 不一致"
            )
    if any(
        not isinstance(request, (list, tuple))
        or len(request) != 2
        or not all(isinstance(value, str) and value for value in request)
        for request in batch["requests"]
    ):
        raise ValueError("description collator requests 必须为非空二元字符串序列")
    if any(
        not isinstance(value, str) or not value.strip()
        for name in ("instructions", "target_texts")
        for value in batch[name]
    ):
        raise ValueError("description collator instruction/target text 为空")
    if any(
        not isinstance(values, list)
        or not values
        or any(not isinstance(value, str) or not value.strip() for value in values)
        for values in batch["reference_texts"]
    ):
        raise ValueError("description collator reference_texts 非法")
    if any(type(value) is not bool for value in batch["structured_outputs"]):
        raise ValueError("description collator structured_outputs 必须为 bool")
    if any(type(value) is not bool for value in batch["use_region_tokens"]):
        raise ValueError("description collator use_region_tokens 必须为 bool")
    if any(
        not isinstance(row, dict)
        or not str(row.get("task_family") or "").strip()
        for row in metadata
    ):
        raise ValueError("description collator metadata 缺少 task_family")
    masks = batch["region_masks"]
    weights = batch["weights"]
    if (
        not torch.is_tensor(masks)
        or masks.ndim != 4
        or int(masks.shape[0]) != batch_size
        or int(masks.shape[1]) != 1
    ):
        raise ValueError("description collator region_masks batch 维度不一致")
    if not torch.is_tensor(weights) or tuple(weights.shape) != (batch_size,):
        raise ValueError("description collator weights shape 不一致")
    if not bool(torch.isfinite(masks).all()) or not bool(
        torch.isfinite(weights).all()
    ):
        raise ValueError("description collator 产生非 finite tensor")
    if bool((masks < 0).any()) or bool((masks > 1).any()):
        raise ValueError("description collator region_masks 必须位于 [0,1]")
    if bool((weights <= 0).any()):
        raise ValueError("description collator training weights 必须为正")
    return {
        "protocol": DESCRIPTION_COLLATOR_AUDIT_PROTOCOL,
        "batch_size": batch_size,
        "contract_fields": sorted(required),
        "region_masks_shape": list(masks.shape),
        "weights_shape": list(weights.shape),
        "structured_samples": sum(batch["structured_outputs"]),
        "region_grounded_samples": sum(batch["use_region_tokens"]),
        "request_components": sorted({
            request[0] for request in batch["requests"]
        }),
        "row_tasks": sorted({
            str(row["task_family"]) for row in metadata
        }),
    }


def write_json(path: str | Path, payload: Any) -> None:
    atomic_write_json(path, payload)


def append_jsonl(path: str | Path, payload: dict[str, Any]) -> None:
    _append_jsonl(path, payload)
