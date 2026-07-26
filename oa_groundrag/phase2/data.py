"""Phase 2 复用 oa_auxseg_hdf5_v1 的稀疏多模态 batch。"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from scripts.phase1_benchmark_build.benchmark_common import (
    BenchmarkDataset,
    collate_benchmark_samples,
    read_jsonl,
)

from .contracts import (
    ModelRegistry,
    OAAuxSegBatch,
    PackedAuxiliary,
    SUPPORTED_AUXILIARY_ORDER,
    ordered_auxiliary_names,
)


@dataclass
class PreparedBatch:
    model: OAAuxSegBatch
    mask: Tensor
    sample_ids: list[str]
    metadata: list[dict[str, Any]]

    def to(self, device: torch.device, *, non_blocking: bool = False) -> "PreparedBatch":
        return PreparedBatch(
            model=self.model.to(device, non_blocking=non_blocking),
            mask=self.mask.to(device=device, non_blocking=non_blocking),
            sample_ids=self.sample_ids,
            metadata=self.metadata,
        )


def registry_from_benchmark(root: Path | str) -> ModelRegistry:
    rows = read_jsonl(Path(root) / "index.jsonl")
    optical: set[tuple[str, ...]] = set()
    auxiliary_channels: dict[str, tuple[str, ...]] = {}
    availability: dict[tuple[str, ...], set[str]] = defaultdict(set)
    for row in rows:
        signature = tuple(str(name) for name in row["optical"]["channel_names"])
        optical.add(signature)
        for name, contract in row["auxiliaries"].items():
            if name not in SUPPORTED_AUXILIARY_ORDER:
                raise ValueError(f"Benchmark 含未注册辅助模态：{name}")
            channels = tuple(str(item) for item in contract["channel_names"])
            previous = auxiliary_channels.setdefault(name, channels)
            if previous != channels:
                raise ValueError(f"{name}: Benchmark 通道合同不一致")
            availability[signature].add(name)
    ordered_optical = tuple(sorted(optical))
    return ModelRegistry(
        optical_signatures=ordered_optical,
        auxiliary_channels={
            name: auxiliary_channels[name]
            for name in ordered_auxiliary_names(tuple(auxiliary_channels))
        },
        available_auxiliaries={
            signature: ordered_auxiliary_names(
                tuple(availability.get(signature, set()))
            )
            for signature in ordered_optical
        },
    )


def prepare_collated_batch(collated: Mapping[str, Any]) -> PreparedBatch:
    auxiliaries = {
        str(name): PackedAuxiliary(
            sample_indices=value["sample_indices"].to(torch.int64),
            values=value["values"].to(torch.float32),
            pixel_valid=value["pixel_valid"].to(torch.uint8),
            channel_valid=value["channel_valid"].to(torch.uint8),
            channel_names=tuple(str(item) for item in value["channel_names"]),
        )
        for name, value in collated["auxiliaries"].items()
    }
    model_batch = OAAuxSegBatch(
        optical=tuple(value.to(torch.float32) for value in collated["optical"]),
        optical_pixel_valid=tuple(
            value.to(torch.uint8) for value in collated["optical_pixel_valid"]
        ),
        optical_channel_valid=tuple(
            value.to(torch.uint8) for value in collated["optical_channel_valid"]
        ),
        optical_channel_names=tuple(
            tuple(str(item) for item in names)
            for names in collated["optical_channel_names"]
        ),
        auxiliaries=auxiliaries,
    )
    mask = collated["mask"].to(torch.float32)
    if mask.shape != (
        model_batch.batch_size,
        1,
        *model_batch.spatial_size,
    ):
        raise ValueError(f"batch mask shape 非法：{tuple(mask.shape)}")
    return PreparedBatch(
        model=model_batch,
        mask=mask,
        sample_ids=[str(item) for item in collated["sample_id"]],
        metadata=[dict(item) for item in collated["metadata"]],
    )


def make_dataloader(
    root: Path | str,
    *,
    split: str,
    source: str | None = None,
    batch_size: int,
    normalization: str,
    shuffle: bool,
    num_workers: int,
    generator: torch.Generator | None = None,
) -> DataLoader[dict[str, Any]]:
    dataset = BenchmarkDataset(
        root,
        split=split,
        auxiliary_policy="all",
        normalization=normalization,
    )
    if source is not None:
        dataset.rows = [
            row for row in dataset.rows if str(row["source"]) == source
        ]
    if len(dataset) == 0:
        suffix = f" source={source}" if source is not None else ""
        raise ValueError(f"{split}{suffix}: Benchmark split 为空")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        collate_fn=collate_benchmark_samples,
        generator=generator,
        drop_last=False,
    )


def available_auxiliaries_by_sample(batch: OAAuxSegBatch) -> list[tuple[str, ...]]:
    names: list[list[str]] = [[] for _ in range(batch.batch_size)]
    for modality in ordered_auxiliary_names(tuple(batch.auxiliaries)):
        for sample_index in batch.auxiliaries[modality].sample_indices.tolist():
            names[int(sample_index)].append(modality)
    return [tuple(items) for items in names]


def filter_auxiliaries(
    batch: OAAuxSegBatch,
    active_by_sample: Sequence[Sequence[str]],
) -> OAAuxSegBatch:
    if len(active_by_sample) != batch.batch_size:
        raise ValueError("active_by_sample 数量与 batch 不符")
    normalized = [
        set(ordered_auxiliary_names(tuple(names)))
        for names in active_by_sample
    ]
    available = [
        set(names) for names in available_auxiliaries_by_sample(batch)
    ]
    for sample_index, (active, present) in enumerate(
        zip(normalized, available, strict=True)
    ):
        unavailable = active - present
        if unavailable:
            raise ValueError(
                f"sample {sample_index}: 请求了不存在的辅助模态 "
                f"{sorted(unavailable)}"
            )
    auxiliaries: dict[str, PackedAuxiliary] = {}
    for modality in ordered_auxiliary_names(tuple(batch.auxiliaries)):
        packed = batch.auxiliaries[modality]
        selected_positions = [
            position
            for position, sample_index in enumerate(packed.sample_indices.tolist())
            if modality in normalized[int(sample_index)]
        ]
        if not selected_positions:
            continue
        positions = torch.tensor(
            selected_positions,
            dtype=torch.int64,
            device=packed.sample_indices.device,
        )
        auxiliaries[modality] = PackedAuxiliary(
            sample_indices=packed.sample_indices.index_select(0, positions),
            values=packed.values.index_select(0, positions),
            pixel_valid=packed.pixel_valid.index_select(0, positions),
            channel_valid=packed.channel_valid.index_select(0, positions),
            channel_names=packed.channel_names,
        )
    return OAAuxSegBatch(
        optical=batch.optical,
        optical_pixel_valid=batch.optical_pixel_valid,
        optical_channel_valid=batch.optical_channel_valid,
        optical_channel_names=batch.optical_channel_names,
        auxiliaries=auxiliaries,
    )


class AuxiliarySubsetSampler:
    """先均匀采样 cardinality，再执行 modality dropout。"""

    def __init__(self, seed: int, dropout_probability: float) -> None:
        if not 0 <= dropout_probability < 1:
            raise ValueError("dropout_probability 必须位于 [0,1)")
        self.random = random.Random(seed)
        self.dropout_probability = dropout_probability
        self.counts: Counter[str] = Counter()

    def sample(
        self, available: Sequence[Sequence[str]]
    ) -> list[tuple[str, ...]]:
        selected: list[tuple[str, ...]] = []
        for names_value in available:
            names = list(ordered_auxiliary_names(tuple(names_value)))
            if not names:
                active: list[str] = []
            else:
                cardinality = self.random.randint(0, len(names))
                active = self.random.sample(names, cardinality)
                active = [
                    name
                    for name in active
                    if self.random.random() >= self.dropout_probability
                ]
            active_tuple = ordered_auxiliary_names(tuple(active))
            if len(active_tuple) == 0:
                label = "none"
            elif len(active_tuple) == 1:
                label = "single"
            elif len(active_tuple) == len(names):
                label = "all"
            else:
                label = "multi"
            self.counts[label] += 1
            selected.append(active_tuple)
        return selected

    def state_dict(self) -> dict[str, Any]:
        return {
            "random_state": self.random.getstate(),
            "dropout_probability": self.dropout_probability,
            "counts": dict(self.counts),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        probability = float(state["dropout_probability"])
        if probability != self.dropout_probability:
            raise ValueError("modality dropout 配置与 checkpoint 不一致")
        self.random.setstate(state["random_state"])
        self.counts = Counter(
            {str(key): int(value) for key, value in state["counts"].items()}
        )
