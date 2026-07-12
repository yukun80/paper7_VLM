#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Benchmark-v2 Dataset with subset-first evidence construction."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import random
from collections import Counter
from typing import Any

import torch
from torch.utils.data import Dataset

from qpsalm_seg.config import QPSalmConfig
from qpsalm_seg.indexing import iter_jsonl, should_skip_row
from qpsalm_seg.paths import resolve_project_path
from qpsalm_seg.schema import ActiveModalitySubset, ModalityBatch, ModalityInstance

from .io import SCHEMA_VERSION, build_modality_instance, load_npy_array, normalize_mask, positive_float
from .prompts import PROMPT_VERSION, build_prompt_triplet, transform_spatial_instruction
from .samplers import task_group
from .transforms import apply_flips, downscale_native, resize_pad_tensor, swap_padding_after_flip, valid_mask_from_transform


def effective_canvas_gsd(row: dict[str, Any], target_size: int) -> float | None:
    spatial = row.get("spatial") or {}
    gsd = positive_float(spatial.get("gsd_m"))
    shape = spatial.get("original_size") or (row.get("mask") or {}).get("shape")
    if gsd is None or not shape or len(shape) < 2:
        return gsd
    height, width = int(shape[-2]), int(shape[-1])
    resize_scale = min(float(target_size) / max(height, 1), float(target_size) / max(width, 1))
    return gsd / max(resize_scale, 1.0e-8)


def scaled_band_metadata(values: tuple[dict[str, Any], ...], factor: float) -> tuple[dict[str, Any], ...]:
    output = []
    for value in values:
        item = dict(value)
        source_gsd = positive_float(item.get("native_gsd_m"))
        item["source_native_gsd_m"] = source_gsd
        item["native_gsd_m"] = source_gsd * factor if source_gsd is not None else None
        output.append(item)
    return tuple(output)


def subset_signature(names: list[str] | tuple[str, ...]) -> str:
    ordered = tuple(sorted(str(name) for name in names))
    digest = hashlib.sha256("\n".join(ordered).encode("utf-8")).hexdigest()[:12]
    return f"subset:{'+'.join(ordered)}:{digest}"


def choose_active_subset(
    instances: list[ModalityInstance],
    *,
    training: bool,
    dropout: float,
    rng: random.Random,
) -> ActiveModalitySubset:
    names = [item.name for item in instances]
    active = list(names)
    if training and len(names) > 1 and dropout > 0:
        active = [name for name in names if rng.random() >= dropout]
        if not active:
            active = [rng.choice(names)]
    active_tuple = tuple(sorted(active))
    dropped = tuple(sorted(set(names) - set(active_tuple)))
    return ActiveModalitySubset(
        active_names=active_tuple,
        dropped_names=dropped,
        signature=subset_signature(active_tuple),
        is_full=not dropped,
    )


class MultiSourceLandslideDataset(Dataset):
    """Strict v2 instruction dataset; legacy modality inference is intentionally unsupported."""

    def __init__(
        self,
        config: QPSalmConfig,
        split: str,
        max_samples: int | None = None,
        shuffle_seed: int | None = None,
    ) -> None:
        self.config = config
        self.split = split
        self.target_size = int(config.target_size)
        index_path = resolve_project_path(config.index_path(split))
        if index_path is None or not index_path.exists():
            raise FileNotFoundError(f"索引不存在: {config.index_path(split)}")
        self.skipped = Counter()
        rows: list[dict[str, Any]] = []
        for row in iter_jsonl(index_path):
            reason = should_skip_row(row, config.task_families)
            if reason is not None:
                self.skipped[reason] += 1
                continue
            if row.get("schema_version") != SCHEMA_VERSION:
                raise ValueError(
                    f"只支持 {SCHEMA_VERSION}，当前 sample={row.get('sample_id')} "
                    f"schema={row.get('schema_version')!r}"
                )
            rows.append(row)
        if max_samples is not None and max_samples > 0 and len(rows) > max_samples:
            grouped: dict[str, list[dict[str, Any]]] = {"global": [], "referring": [], "no_target": []}
            for row in rows:
                grouped[task_group(row)].append(row)
            rng = random.Random(config.seed + {"train": 0, "val": 1, "test": 2}.get(split, 3))
            for values in grouped.values():
                rng.shuffle(values)
            weights = {**{"global": 0.4, "referring": 0.4, "no_target": 0.2}, **config.task_sampling_ratios}
            limited = []
            cursors = {name: 0 for name in grouped}
            names = [name for name, values in grouped.items() if values]
            while len(limited) < max_samples and names:
                name = rng.choices(names, weights=[max(weights.get(value, 0.0), 1.0e-6) for value in names], k=1)[0]
                values = grouped[name]
                limited.append(values[cursors[name]])
                cursors[name] += 1
                if cursors[name] >= len(values):
                    names.remove(name)
            rows = limited
        if shuffle_seed is not None:
            random.Random(shuffle_seed).shuffle(rows)
        self.rows = rows
        self._check_parent_split_isolation()

    def _check_parent_split_isolation(self) -> None:
        for row in self.rows:
            parent = str(row.get("parent_sample_id") or row.get("sample_id"))
            if not parent:
                raise ValueError("instruction row 缺少 parent/sample id")
            if str(row.get("split")) != self.split:
                raise ValueError(f"索引 split 泄漏: expected={self.split} row={row.get('split')} parent={parent}")

    def __len__(self) -> int:
        return len(self.rows)

    def _prompt_row(self, index: int, row: dict[str, Any]) -> dict[str, Any]:
        if self.config.instruction_ablation != "shuffled":
            return row
        if len(self.rows) < 2:
            raise RuntimeError("instruction shuffle 至少需要两个不同 parent 的样本")
        current_text = (row.get("instruction") or {}).get("text")
        for offset in range(1, len(self.rows)):
            candidate = self.rows[(index + offset) % len(self.rows)]
            candidate_text = (candidate.get("instruction") or {}).get("text")
            if candidate_text != current_text and candidate.get("parent_sample_id") != row.get("parent_sample_id"):
                mixed = dict(row)
                mixed["instruction"] = candidate.get("instruction")
                mixed["task_family"] = candidate.get("task_family")
                mixed["template_id"] = candidate.get("template_id")
                mixed["referring_target"] = candidate.get("referring_target")
                return mixed
        raise RuntimeError(
            f"instruction shuffle 未找到不同 parent/text: sample={row.get('sample_id')}"
        )

    def bucket_size(self, index: int) -> int:
        if not self.config.use_size_buckets:
            return self.target_size
        buckets = sorted({int(value) for value in self.config.size_buckets if int(value) > 0})
        if not buckets:
            return self.target_size
        spatial = self.rows[index].get("spatial") or {}
        shape = spatial.get("original_size") or (self.rows[index].get("mask") or {}).get("shape")
        longest = max(int(shape[-2]), int(shape[-1])) if shape and len(shape) >= 2 else self.target_size
        requested = min(longest, buckets[-1])
        return next((value for value in buckets if value >= requested), buckets[-1])

    def _load_instances(self, row: dict[str, Any], target_size: int) -> list[ModalityInstance]:
        aligned_gsd = effective_canvas_gsd(row, target_size)
        max_native = min(int(self.config.max_native_size), int(target_size))
        instances: list[ModalityInstance] = []
        for name, item in sorted((row.get("modalities") or {}).items()):
            if not isinstance(item, dict) or not item.get("available", True):
                continue
            instance = build_modality_instance(str(name), item, aligned_gsd)
            source_h, source_w = instance.image.shape[-2:]
            resized_image = downscale_native(instance.image, max_native)
            resized_valid = downscale_native(instance.valid_mask, max_native, mode="nearest")
            resized_h, resized_w = resized_image.shape[-2:]
            native_resize_factor = max(
                float(source_h) / max(resized_h, 1),
                float(source_w) / max(resized_w, 1),
            )
            source_native_gsd = instance.native_gsd_m
            instance = replace(
                instance,
                image=resized_image,
                valid_mask=(resized_valid >= 0.5).float(),
                native_gsd_m=(
                    source_native_gsd * native_resize_factor
                    if source_native_gsd is not None else None
                ),
                band_metadata=scaled_band_metadata(instance.band_metadata, native_resize_factor),
                metadata={
                    **instance.metadata,
                    "source_native_gsd_m": source_native_gsd,
                    "encoder_native_gsd_m": (
                        source_native_gsd * native_resize_factor
                        if source_native_gsd is not None else None
                    ),
                    "encoder_aligned_gsd_m": aligned_gsd,
                    "native_resize_factor": native_resize_factor,
                },
            )
            if not bool(instance.valid_mask.any()):
                continue
            instances.append(instance)
        if not instances:
            raise ValueError(f"样本没有可编码模态: {row.get('sample_id')}")
        return instances

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        target_size = self.bucket_size(index)
        full_instances = self._load_instances(row, target_size)
        # Training views must change across repeated visits while remaining
        # reproducible under DataLoader worker seeding. Evaluation stays fixed.
        draw_nonce = random.getrandbits(64) if self.split == "train" else 0
        seed_material = f"{self.config.seed}:{self.split}:{row.get('sample_id')}:{draw_nonce}"
        seed = int(hashlib.sha256(seed_material.encode()).hexdigest()[:16], 16)
        subset = choose_active_subset(
            full_instances,
            training=self.split == "train",
            dropout=float(self.config.modality_dropout),
            rng=random.Random(seed),
        )
        active_set = set(subset.active_names)

        mask = normalize_mask(load_npy_array(str((row.get("mask") or {})["path"])))
        original_mask_size = list(mask.shape[-2:])
        mask, resize_transform = resize_pad_tensor(mask, target_size, "nearest")
        valid_mask = valid_mask_from_transform(resize_transform)
        hflip = self.split == "train" and random.Random(seed + 1).random() < float(self.config.train_hflip_prob)
        vflip = self.split == "train" and random.Random(seed + 2).random() < float(self.config.train_vflip_prob)
        full_instances, mask, valid_mask = apply_flips(
            full_instances, mask, valid_mask, hflip=hflip, vflip=vflip
        )
        parent = str(row.get("parent_sample_id") or row.get("sample_id"))
        cache_key = f"qmv3-parent:{parent}"
        augment = {"hflip": hflip, "vflip": vflip}
        full_instances = [
            replace(
                item,
                metadata={
                    **item.metadata,
                    "vision_cache_key": cache_key,
                    "train_augment": augment,
                },
            )
            for item in full_instances
        ]
        evidence_valid = torch.zeros_like(valid_mask)
        aligned_full_instances = []
        for item in full_instances:
            aligned_valid, reference_transform = resize_pad_tensor(item.valid_mask, target_size, "nearest")
            aligned_valid, reference_transform = swap_padding_after_flip(
                aligned_valid, reference_transform, hflip=hflip, vflip=vflip
            )
            aligned_full_instances.append(replace(
                item,
                metadata={**item.metadata, "reference_resize_transform": reference_transform},
            ))
            if item.name in active_set:
                evidence_valid = torch.maximum(evidence_valid, (aligned_valid >= 0.5).to(evidence_valid.dtype))
        full_instances = aligned_full_instances
        active_instances = [item for item in full_instances if item.name in active_set]
        valid_mask = valid_mask * evidence_valid
        if not bool((valid_mask >= 0.5).any()):
            raise ValueError(
                f"样本 active subset 没有任何有效像素: sample={row.get('sample_id')} subset={subset.signature}"
            )
        prompt_row = transform_spatial_instruction(
            self._prompt_row(index, row), hflip=hflip, vflip=vflip
        )
        ablation = self.config.instruction_ablation
        proposal, condition, reasoning = build_prompt_triplet(
            prompt_row, active_instances, subset_signature=subset.signature, ablation=ablation
        )
        full_signature = subset_signature([item.name for item in full_instances])
        full_proposal, full_condition, full_reasoning = build_prompt_triplet(
            prompt_row, full_instances, subset_signature=full_signature, ablation=ablation
        )
        metadata = {
            "sample_id": row.get("sample_id"),
            "parent_sample_id": parent,
            "dataset_name": row.get("dataset_name"),
            "template_id": row.get("template_id"),
            "task_family": row.get("task_family"),
            "instruction": (prompt_row.get("instruction") or {}).get("text"),
            "referring_category": (row.get("referring_target") or {}).get("category"),
            "target_mask_path": (((row.get("referring_target") or {}).get("target_mask") or {}).get("path")),
            "active_subset": subset.signature,
            "active_modalities": list(subset.active_names),
            "full_modalities": [item.name for item in full_instances],
            "raw_combo": "+".join(sorted(item.name for item in active_instances)),
            "family_combo": "+".join(sorted({item.family for item in active_instances})),
            "sensor_combo": "+".join(sorted({item.sensor for item in active_instances})),
            "product_combo": "+".join(sorted({item.product_type for item in active_instances})),
            "normalization_methods": "+".join(sorted({
                str((item.metadata.get("normalization") or {}).get("method") or "unknown")
                for item in active_instances
            })),
            "gsd_m": (row.get("spatial") or {}).get("gsd_m"),
            "canvas_gsd_m": effective_canvas_gsd(row, target_size),
            "original_size": (row.get("spatial") or {}).get("original_size") or original_mask_size,
            "mask_original_size": original_mask_size,
            "resize_transform": resize_transform,
            "target_size": target_size,
            "valid_coverage": float(valid_mask.mean().item()),
            "train_augment": augment,
            "augmented_referring_grid": (
                ((prompt_row.get("referring_target") or {}).get("grounding") or {}).get("grid")
            ),
            "prompt_version": PROMPT_VERSION,
            "instruction_ablation": ablation,
        }
        return {
            "instances": active_instances,
            "full_instances": full_instances,
            "active_subset": subset,
            "mask": mask,
            "valid_mask": valid_mask,
            "metadata": metadata,
            "proposal_context_text": proposal,
            "condition_prompt_text": condition,
            "evidence_reasoning_text": reasoning,
            "full_proposal_context_text": full_proposal,
            "full_condition_prompt_text": full_condition,
            "full_evidence_reasoning_text": full_reasoning,
            "visual_evidence_key": cache_key,
        }


def qpsalm_collate(batch: list[dict[str, Any]]) -> ModalityBatch:
    shapes = {tuple(item["mask"].shape[-2:]) for item in batch}
    if len(shapes) != 1:
        raise ValueError(f"同一 batch 必须来自同一尺寸桶，收到 {sorted(shapes)}")
    return ModalityBatch(
        instances=[item["instances"] for item in batch],
        full_instances=[item["full_instances"] for item in batch],
        active_subsets=[item["active_subset"] for item in batch],
        mask=torch.stack([item["mask"] for item in batch]),
        valid_mask=torch.stack([item["valid_mask"] for item in batch]),
        metadata=[item["metadata"] for item in batch],
        proposal_context_text=[item["proposal_context_text"] for item in batch],
        condition_prompt_text=[item["condition_prompt_text"] for item in batch],
        evidence_reasoning_text=[item["evidence_reasoning_text"] for item in batch],
        full_proposal_context_text=[item["full_proposal_context_text"] for item in batch],
        full_condition_prompt_text=[item["full_condition_prompt_text"] for item in batch],
        full_evidence_reasoning_text=[item["full_evidence_reasoning_text"] for item in batch],
        visual_evidence_key=[item["visual_evidence_key"] for item in batch],
    )
