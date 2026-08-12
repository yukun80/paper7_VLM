"""OA-AuxSeg 复用 oa_auxseg_hdf5_v1 的稀疏多模态 batch。"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from oa_groundrag.data.oa_auxseg.dataset import (
    SCHEMA_VERSION,
    BenchmarkDataset,
    collate_benchmark_samples,
    read_json,
    read_jsonl,
    sha256_file,
)

from .contracts import (
    BENCHMARK_SCHEMA_VERSION,
    ModelRegistry,
    OAAuxSegBatch,
    PackedAuxiliary,
    PHASE2_EXCLUDED_SOURCES,
    PHASE2_INCLUDED_SOURCES,
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


def benchmark_contract_from_root(root: Path | str) -> dict[str, Any]:
    """读取并严格固化当前 Benchmark 身份与 source selection。"""

    benchmark_root = Path(root)
    manifest_path = benchmark_root / "manifest.json"
    index_path = benchmark_root / "index.jsonl"
    manifest = read_json(manifest_path)
    if SCHEMA_VERSION != BENCHMARK_SCHEMA_VERSION:
        raise RuntimeError("Phase 1B 与 Phase 2 Benchmark schema 常量不一致")
    if manifest.get("schema_version") != BENCHMARK_SCHEMA_VERSION:
        raise ValueError("Benchmark manifest schema_version 不受支持")
    index_sha256 = sha256_file(index_path)
    if manifest.get("index_sha256") != index_sha256:
        raise ValueError("Benchmark manifest 与 index SHA-256 不一致")
    selection = manifest.get("source_selection")
    included = manifest.get("included_sources")
    excluded = manifest.get("excluded_sources")
    if selection not in {"all", "subset"}:
        raise ValueError("Benchmark source_selection 不受支持")
    if not (
        isinstance(included, list)
        and included
        and all(isinstance(source, str) for source in included)
        and isinstance(excluded, list)
        and all(isinstance(source, str) for source in excluded)
    ):
        raise ValueError("Benchmark included/excluded sources 合同非法")
    if set(included) & set(excluded):
        raise ValueError("Benchmark included/excluded sources 不能重叠")
    expected_selection = "subset" if excluded else "all"
    if selection != expected_selection:
        raise ValueError("Benchmark source_selection 与 excluded_sources 不一致")
    if tuple(included) != PHASE2_INCLUDED_SOURCES:
        raise ValueError(
            "Phase 2 v6 只接受固定五源 Benchmark，实际 included_sources="
            f"{included}"
        )
    if tuple(excluded) != PHASE2_EXCLUDED_SOURCES:
        raise ValueError(
            "Phase 2 v6 要求明确排除 Sen12Landslides，实际 "
            f"excluded_sources={excluded}"
        )
    row_sources = {
        str(row["source"]) for row in read_jsonl(index_path)
    }
    if row_sources != set(included):
        raise ValueError("Benchmark index sources 与 included_sources 不一致")
    return {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "index_sha256": index_sha256,
        "manifest_sha256": sha256_file(manifest_path),
        "source_selection": str(selection),
        "included_sources": [str(source) for source in included],
        "excluded_sources": [str(source) for source in excluded],
    }


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


class StatefulTrainingBatcher:
    """严格定长、跨 permutation 边界补齐且可精确恢复的训练 batcher。"""

    def __init__(
        self,
        root: Path | str,
        *,
        batch_size: int,
        normalization: str,
        seed: int,
        policy: str,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size 必须大于 0")
        if policy not in {"uniform", "balanced_target_presence"}:
            raise ValueError(f"未知训练 sampler：{policy}")
        if policy == "balanced_target_presence" and batch_size % 2:
            raise ValueError("balanced_target_presence 要求偶数 batch_size")
        self.dataset = BenchmarkDataset(
            root,
            split="train",
            auxiliary_policy="all",
            normalization=normalization,
        )
        if len(self.dataset) == 0:
            raise ValueError("train: Benchmark split 为空")
        self.batch_size = int(batch_size)
        self.normalization = str(normalization)
        self.policy = str(policy)
        self.random = random.Random(seed)
        all_indices = tuple(range(len(self.dataset)))
        if policy == "uniform":
            base_pools = {"all": all_indices}
        else:
            positive = tuple(
                index
                for index, row in enumerate(self.dataset.rows)
                if float(row["foreground_ratio"]) > 0
            )
            empty = tuple(
                index
                for index, row in enumerate(self.dataset.rows)
                if float(row["foreground_ratio"]) == 0
            )
            if not positive or not empty:
                raise ValueError(
                    "balanced_target_presence 同时需要 positive 和 empty 样本"
                )
            base_pools = {"positive": positive, "empty": empty}
        self._base_pools = base_pools
        self._permutations: dict[str, list[int]] = {}
        self._cursors: dict[str, int] = {}
        for name in self._base_pools:
            self._reset_pool(name)
        self.batches_emitted = 0
        self.samples_emitted = 0

    def _reset_pool(self, name: str) -> None:
        permutation = list(self._base_pools[name])
        self.random.shuffle(permutation)
        self._permutations[name] = permutation
        self._cursors[name] = 0

    def _take(self, name: str, count: int) -> list[int]:
        selected: list[int] = []
        while len(selected) < count:
            permutation = self._permutations[name]
            cursor = self._cursors[name]
            if cursor == len(permutation):
                self._reset_pool(name)
                continue
            take = min(count - len(selected), len(permutation) - cursor)
            selected.extend(permutation[cursor : cursor + take])
            self._cursors[name] = cursor + take
        return selected

    def next_indices(self) -> tuple[int, ...]:
        if self.policy == "uniform":
            indices = self._take("all", self.batch_size)
        else:
            half = self.batch_size // 2
            indices = self._take("positive", half) + self._take("empty", half)
            self.random.shuffle(indices)
        self.batches_emitted += 1
        self.samples_emitted += len(indices)
        return tuple(indices)

    def next(self) -> dict[str, Any]:
        return collate_benchmark_samples(
            [self.dataset[index] for index in self.next_indices()]
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "normalization": self.normalization,
            "policy": self.policy,
            "base_pools": {
                name: list(indices)
                for name, indices in self._base_pools.items()
            },
            "permutations": {
                name: list(indices)
                for name, indices in self._permutations.items()
            },
            "cursors": dict(self._cursors),
            "random_state": self.random.getstate(),
            "batches_emitted": self.batches_emitted,
            "samples_emitted": self.samples_emitted,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if int(state["batch_size"]) != self.batch_size:
            raise ValueError("训练 batcher batch_size 与 checkpoint 不一致")
        if str(state["normalization"]) != self.normalization:
            raise ValueError("训练 batcher normalization 与 checkpoint 不一致")
        if str(state["policy"]) != self.policy:
            raise ValueError("训练 batcher policy 与 checkpoint 不一致")
        recorded_base = {
            str(name): tuple(int(index) for index in indices)
            for name, indices in state["base_pools"].items()
        }
        if recorded_base != self._base_pools:
            raise ValueError("训练 batcher 数据索引与 checkpoint 不一致")
        permutations = {
            str(name): [int(index) for index in indices]
            for name, indices in state["permutations"].items()
        }
        cursors = {
            str(name): int(cursor)
            for name, cursor in state["cursors"].items()
        }
        if set(permutations) != set(self._base_pools):
            raise ValueError("训练 batcher permutation 字段不一致")
        for name, permutation in permutations.items():
            if sorted(permutation) != sorted(self._base_pools[name]):
                raise ValueError(f"训练 batcher {name} permutation 非法")
            if not 0 <= cursors[name] <= len(permutation):
                raise ValueError(f"训练 batcher {name} cursor 越界")
        self._permutations = permutations
        self._cursors = cursors
        self.random.setstate(state["random_state"])
        self.batches_emitted = int(state["batches_emitted"])
        self.samples_emitted = int(state["samples_emitted"])


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
    """在非空 cardinality 上均匀采样，并执行非空 modality dropout。"""

    def __init__(
        self,
        seed: int,
        dropout_probability: float,
    ) -> None:
        if not 0 <= dropout_probability < 1:
            raise ValueError("dropout_probability 必须位于 [0,1)")
        self.random = random.Random(seed)
        self.dropout_probability = dropout_probability
        self.counts: Counter[str] = Counter()
        self.reason_counts: Counter[str] = Counter()
        self.cardinality_counts: Counter[str] = Counter()
        self.dropout_counts: Counter[str] = Counter()
        self.last_reasons: list[str] = []

    def sample(
        self, available: Sequence[Sequence[str]]
    ) -> list[tuple[str, ...]]:
        selected: list[tuple[str, ...]] = []
        reasons: list[str] = []
        for names_value in available:
            names = list(ordered_auxiliary_names(tuple(names_value)))
            if not names:
                active: list[str] = []
                reason = "native_none"
            else:
                cardinality = self.random.randint(1, len(names))
                initially_selected = self.random.sample(names, cardinality)
                self.cardinality_counts[str(cardinality)] += 1
                self.dropout_counts["selected_modalities"] += len(
                    initially_selected
                )
                active = [
                    name
                    for name in initially_selected
                    if (
                        self.random.random()
                        >= self.dropout_probability
                    )
                ]
                self.dropout_counts["dropped_modalities"] += (
                    len(initially_selected) - len(active)
                )
                if not active:
                    active = [self.random.choice(initially_selected)]
                    self.dropout_counts["restored_samples"] += 1
                    reason = "dropout_restored"
                else:
                    reason = "active"
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
            self.reason_counts[reason] += 1
            selected.append(active_tuple)
            reasons.append(reason)
        self.last_reasons = reasons
        return selected

    def state_dict(self) -> dict[str, Any]:
        return {
            "random_state": self.random.getstate(),
            "dropout_probability": self.dropout_probability,
            "counts": dict(self.counts),
            "reason_counts": dict(self.reason_counts),
            "cardinality_counts": dict(self.cardinality_counts),
            "dropout_counts": dict(self.dropout_counts),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        probability = float(state["dropout_probability"])
        if probability != self.dropout_probability:
            raise ValueError("modality dropout 配置与 checkpoint 不一致")
        self.random.setstate(state["random_state"])
        self.counts = Counter(
            {str(key): int(value) for key, value in state["counts"].items()}
        )
        self.reason_counts = Counter(
            {
                str(key): int(value)
                for key, value in state["reason_counts"].items()
            }
        )
        self.cardinality_counts = Counter(
            {
                str(key): int(value)
                for key, value in state["cardinality_counts"].items()
            }
        )
        self.dropout_counts = Counter(
            {
                str(key): int(value)
                for key, value in state["dropout_counts"].items()
            }
        )
        self.last_reasons = []
