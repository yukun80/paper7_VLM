"""Stage 5 parent 隔离、90/10 replay 与 Region monitor 合同。"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, replace
import math
import time
from typing import Any, Iterator, Mapping, Sequence

import torch
from torch import Tensor

from oa_groundrag.data.grounded.supervision.compact_training import (
    CompactTrainingMessageDataset,
    EXPERT_AUTHORITY,
)
from oa_groundrag.artifacts.identity import (
    canonical_json,
    sha256_text,
    stable_hash,
)
from oa_groundrag.vlm.checkpoint import capture_rng_state, restore_rng_state
from oa_groundrag.vlm.data import DescriptionSample
from oa_groundrag.vlm.errors import ModelError, ReasonCode
from oa_groundrag.training.vlm.validation import ValidationItem, ValidationResult


STAGE5_SPLIT_SCHEMA = "rs_vlm.mask_grounded_region_parent_split.v1"
STAGE5_SELECTION_SCHEMA = "rs_vlm.mask_grounded_region_monitor_selection.v1"
STAGE5_SAMPLER_SCHEMA = "rs_vlm.mask_grounded_region_replay_sampler.v1"
REGION_TASK = "mask_grounded_region_description"
REGION_TRAIN_ROLE = "mask_grounded_train"
REGION_MONITOR_ROLE = "mask_grounded_monitor"
REPLAY_ROLE = "external_train"


@dataclass(frozen=True)
class RegionParentSplit:
    seed: int
    train_indices: tuple[int, ...]
    monitor_indices: tuple[int, ...]
    train_parents: tuple[str, ...]
    monitor_parents: tuple[str, ...]
    expert_parents: tuple[str, ...]
    identity_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": STAGE5_SPLIT_SCHEMA,
            "algorithm": "source_stratified_parent_hash_90_10.v1",
            "seed": self.seed,
            "train_ratio": 0.9,
            "train_record_count": len(self.train_indices),
            "monitor_record_count": len(self.monitor_indices),
            "train_parent_count": len(self.train_parents),
            "monitor_parent_count": len(self.monitor_parents),
            "expert_parents": list(self.expert_parents),
            "train_record_indices": list(self.train_indices),
            "monitor_record_indices": list(self.monitor_indices),
            "train_parents": list(self.train_parents),
            "monitor_parents": list(self.monitor_parents),
            "identity_sha256": self.identity_sha256,
            "formal_acceptance": False,
        }


def split_compact_by_parent(
    dataset: CompactTrainingMessageDataset,
    *,
    seed: int = 20260808,
) -> RegionParentSplit:
    """按 parent 做确定性 90/10 split，专家 parent 永久留在 train。"""

    if seed != 20260808:
        raise ModelError(ReasonCode.TYPE_MISMATCH, "Stage 5 split seed 必须为 20260808")
    parent_indices: dict[str, list[int]] = defaultdict(list)
    parent_source: dict[str, str] = {}
    expert_parents: set[str] = set()
    for index, row in enumerate(dataset.records):
        parent = str(row.get("parent_id", ""))
        source = str(row.get("source", ""))
        if not parent or not source:
            raise ModelError(ReasonCode.TYPE_MISMATCH, "compact row 缺少 parent/source")
        if parent in parent_source and parent_source[parent] != source:
            raise ModelError(ReasonCode.PARENT_OVERLAP, "同一 parent 跨 source，无法分层")
        parent_source[parent] = source
        parent_indices[parent].append(index)
        if row.get("supervision_authority") == EXPERT_AUTHORITY:
            expert_parents.add(parent)
    parent_count = len(parent_indices)
    monitor_total = int(round(parent_count * 0.1))
    if monitor_total <= 0 or monitor_total >= parent_count:
        raise ModelError(ReasonCode.TYPE_MISMATCH, "compact parent 数不足以做 90/10 split")
    by_source: dict[str, list[str]] = defaultdict(list)
    for parent, source in parent_source.items():
        if parent not in expert_parents:
            by_source[source].append(parent)
    available = sum(len(values) for values in by_source.values())
    if available < monitor_total:
        raise ModelError(ReasonCode.PARENT_OVERLAP, "专家 parent 强制 train 后 monitor 配额不足")
    raw_quotas = {
        source: monitor_total * len(values) / available
        for source, values in by_source.items()
    }
    quotas = {source: int(math.floor(value)) for source, value in raw_quotas.items()}
    remaining = monitor_total - sum(quotas.values())
    for source in sorted(
        by_source,
        key=lambda value: (
            -(raw_quotas[value] - quotas[value]),
            stable_hash(seed, "monitor_quota", value),
        ),
    )[:remaining]:
        quotas[source] += 1
    monitor_parents: set[str] = set()
    for source, values in sorted(by_source.items()):
        ordered = sorted(
            values,
            key=lambda parent: (
                stable_hash(seed, "region_monitor_parent", source, parent),
                parent,
            ),
        )
        monitor_parents.update(ordered[: quotas[source]])
    train_parents = set(parent_indices) - monitor_parents
    if expert_parents - train_parents or train_parents & monitor_parents:
        raise AssertionError("Stage 5 parent split invariant broken")
    train_indices = tuple(
        index for index, row in enumerate(dataset.records)
        if str(row["parent_id"]) in train_parents
    )
    monitor_indices = tuple(
        index for index, row in enumerate(dataset.records)
        if str(row["parent_id"]) in monitor_parents
    )
    payload = {
        "schema_version": STAGE5_SPLIT_SCHEMA,
        "seed": seed,
        "compact_id": dataset.manifest["compact_id"],
        "train_indices": list(train_indices),
        "monitor_indices": list(monitor_indices),
        "train_parents": sorted(train_parents),
        "monitor_parents": sorted(monitor_parents),
        "expert_parents": sorted(expert_parents),
    }
    return RegionParentSplit(
        seed=seed,
        train_indices=train_indices,
        monitor_indices=monitor_indices,
        train_parents=tuple(sorted(train_parents)),
        monitor_parents=tuple(sorted(monitor_parents)),
        expert_parents=tuple(sorted(expert_parents)),
        identity_sha256=sha256_text(canonical_json(payload)),
    )


class RegionSubsetDataset:
    """compact 的 train 或 monitor parent 子集，不复制 messages。"""

    def __init__(
        self,
        base: CompactTrainingMessageDataset,
        indices: Sequence[int],
        *,
        logical_role: str,
    ) -> None:
        if logical_role not in {REGION_TRAIN_ROLE, REGION_MONITOR_ROLE}:
            raise ModelError(ReasonCode.ROLE_FORBIDDEN, "Region subset role 非法")
        self.base = base
        self.indices = tuple(int(index) for index in indices)
        self.logical_role = logical_role
        self.records = tuple(
            {**dict(base.records[index]), "logical_role": logical_role}
            for index in self.indices
        )
        self.manifest = {
            "payload_root_sha256": sha256_text(canonical_json({
                "compact_id": base.manifest["compact_id"],
                "role": logical_role,
                "indices": list(self.indices),
            })),
        }

    def __len__(self) -> int:
        return len(self.indices)

    def set_epoch(self, epoch: int) -> None:
        self.base.set_epoch(epoch)

    def __getitem__(self, index: int) -> DescriptionSample:
        sample = self.base[self.indices[index]]
        return replace(sample, logical_role=self.logical_role)


class Stage5MixedDataset:
    """把 Region train 和 external_train replay 暴露为一个只读索引空间。"""

    def __init__(self, region: RegionSubsetDataset, replay: Any) -> None:
        self.region = region
        self.replay = replay
        self.records = tuple(region.records) + tuple(replay.records)
        self.manifest = {
            "payload_root_sha256": sha256_text(canonical_json({
                "region": region.manifest["payload_root_sha256"],
                "replay": replay.manifest["payload_root_sha256"],
            })),
        }

    @property
    def region_count(self) -> int:
        return len(self.region)

    def __len__(self) -> int:
        return len(self.records)

    def set_epoch(self, epoch: int) -> None:
        self.region.set_epoch(epoch)
        self.replay.set_epoch(epoch)

    def __getitem__(self, index: int) -> DescriptionSample:
        if index < self.region_count:
            return self.region[index]
        return self.replay[index - self.region_count]


class Stage5MixedSampler:
    """每 epoch 精确输出 7,200 Region + 800 replay，并保持 parent 优先。"""

    samples_per_epoch = 8_000
    region_samples_per_epoch = 7_200
    replay_samples_per_epoch = 800

    def __init__(self, dataset: Stage5MixedDataset, *, seed: int = 20260808) -> None:
        if not dataset.region.records or not dataset.replay.records:
            raise ModelError(ReasonCode.ASSET_MISSING, "Stage 5 mixed dataset 不能为空")
        self.dataset = dataset
        self.seed = seed
        self.epoch = 0
        self._region_by_parent: dict[str, tuple[int, ...]] = {
            parent: tuple(indices)
            for parent, indices in self._group_indices(dataset.region.records).items()
        }
        self._replay_by_task: dict[str, list[int]] = defaultdict(list)
        self._replay_sources: set[str] = set()
        for index, row in enumerate(dataset.replay.records):
            if row.get("logical_role") != REPLAY_ROLE:
                raise ModelError(ReasonCode.ROLE_FORBIDDEN, "replay 必须严格来自 external_train")
            self._replay_by_task[str(row["task_family"])].append(index)
            self._replay_sources.add(str(row["source"]))
        if len(self._replay_by_task) != 7:
            raise ModelError(ReasonCode.ROLE_FORBIDDEN, "replay 必须覆盖冻结七类 RS-General task")

    @staticmethod
    def _group_indices(records: Sequence[Mapping[str, Any]]) -> dict[str, list[int]]:
        output: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(records):
            output[str(row["parent_id"])].append(index)
        return output

    def set_epoch(self, epoch: int) -> None:
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ModelError(ReasonCode.TYPE_MISMATCH, "sampler epoch 必须是非负整数")
        self.epoch = epoch
        self.dataset.set_epoch(epoch)

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": STAGE5_SAMPLER_SCHEMA,
            "dataset_identity_sha256": self.dataset.manifest["payload_root_sha256"],
            "seed": self.seed,
            "epoch": self.epoch,
            "region_ratio": 0.9,
            "replay_ratio": 0.1,
            "samples_per_epoch": self.samples_per_epoch,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        expected = self.state_dict()
        if set(state) != set(expected) or any(
            state[key] != expected[key] for key in expected if key != "epoch"
        ):
            raise ModelError(ReasonCode.CHECKPOINT_INCOMPATIBLE, "Stage 5 sampler state 不兼容")
        self.set_epoch(int(state["epoch"]))

    def _region_order(self) -> list[int]:
        parents = sorted(
            self._region_by_parent,
            key=lambda parent: (stable_hash(self.seed, self.epoch, "region", parent), parent),
        )
        positions: Counter[str] = Counter()
        output: list[int] = []
        cycle = 0
        while len(output) < self.region_samples_per_epoch:
            order = sorted(
                parents,
                key=lambda parent: (
                    stable_hash(self.seed, self.epoch, "region_cycle", cycle, parent),
                    parent,
                ),
            )
            for parent in order:
                indices = self._region_by_parent[parent]
                offset = int(stable_hash(self.seed, self.epoch, parent, "view"), 16) % len(indices)
                output.append(indices[(offset + positions[parent]) % len(indices)])
                positions[parent] += 1
                if len(output) == self.region_samples_per_epoch:
                    break
            cycle += 1
        return output

    def _replay_order_for_epoch(self, epoch: int, forbidden_parents: set[str]) -> list[int]:
        tasks = tuple(sorted(self._replay_by_task))
        candidates: dict[str, list[int]] = {}
        for task, indices in self._replay_by_task.items():
            parent_representative: dict[str, int] = {}
            for index in indices:
                parent = str(self.dataset.replay.records[index]["parent_id"])
                incumbent = parent_representative.get(parent)
                if incumbent is None or stable_hash(self.seed, epoch, task, index) < stable_hash(
                    self.seed, epoch, task, incumbent
                ):
                    parent_representative[parent] = index
            candidates[task] = sorted(
                parent_representative.values(),
                key=lambda index: (
                    stable_hash(
                        self.seed,
                        epoch,
                        "replay",
                        task,
                        self.dataset.replay.records[index]["parent_id"],
                    ),
                    str(self.dataset.replay.records[index]["record_id"]),
                ),
            )
        selected: list[int] = []
        used = set(forbidden_parents)
        task_positions = {task: 0 for task in tasks}
        deficits = {task: 0 for task in tasks}
        while len(selected) < self.replay_samples_per_epoch:
            task = min(tasks, key=lambda value: (deficits[value], value))
            choices = candidates[task]
            found: int | None = None
            while task_positions[task] < len(choices):
                candidate = choices[task_positions[task]]
                task_positions[task] += 1
                parent = str(self.dataset.replay.records[candidate]["parent_id"])
                if parent not in used:
                    found = candidate
                    used.add(parent)
                    break
            if found is None:
                # 不同 epoch 优先不重复；全局 unique parent 不足时仅放宽跨 epoch 约束，
                # 当前 epoch 内仍禁止 parent 重复。
                prior = {
                    str(self.dataset.replay.records[index]["parent_id"])
                    for index in selected
                }
                found = next(
                    (
                        index for index in choices
                        if str(self.dataset.replay.records[index]["parent_id"]) not in prior
                    ),
                    None,
                )
            if found is None:
                raise ModelError(ReasonCode.ASSET_MISSING, "RS-General replay unique parent 配额不足")
            selected.append(found)
            deficits[task] += 1
        if {str(self.dataset.replay.records[index]["source"]) for index in selected} != self._replay_sources:
            raise ModelError(ReasonCode.ASSET_MISSING, "RS-General replay 未覆盖全部已有来源")
        return selected

    def _replay_order(self) -> list[int]:
        prior_parents: set[str] = set()
        selected: list[int] = []
        for epoch in range(self.epoch + 1):
            selected = self._replay_order_for_epoch(epoch, prior_parents)
            prior_parents.update(
                str(self.dataset.replay.records[index]["parent_id"])
                for index in selected
            )
        return [self.dataset.region_count + index for index in selected]

    def __iter__(self) -> Iterator[int]:
        region = self._region_order()
        replay = self._replay_order()
        output: list[int] = []
        for group in range(self.replay_samples_per_epoch):
            output.extend(region[group * 9:(group + 1) * 9])
            output.append(replay[group])
        if len(output) != self.samples_per_epoch:
            raise AssertionError("Stage 5 90/10 sampler length broken")
        return iter(output)

    def __len__(self) -> int:
        return self.samples_per_epoch


@dataclass(frozen=True)
class RegionMonitorSelection:
    benchmark_build_id: str
    benchmark_payload_sha256: str
    seed: int
    max_parents: int
    items: tuple[ValidationItem, ...]
    source_counts: Mapping[str, int]
    task_counts: Mapping[str, int]
    selection_sha256: str

    def identity_dict(self) -> dict[str, Any]:
        return {
            "schema_version": STAGE5_SELECTION_SCHEMA,
            "benchmark_build_id": self.benchmark_build_id,
            "benchmark_payload_sha256": self.benchmark_payload_sha256,
            "seed": self.seed,
            "max_parents": self.max_parents,
            "selected_records": len(self.items),
            "selected_parents": len({item.parent_id for item in self.items}),
            "selection_sha256": self.selection_sha256,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.identity_dict(),
            "role": REGION_MONITOR_ROLE,
            "algorithm": "all_parent_isolated_region_monitor.v1",
            "source_counts": dict(sorted(self.source_counts.items())),
            "task_counts": dict(sorted(self.task_counts.items())),
            "items": [item.to_dict() for item in self.items],
            "formal_acceptance": False,
        }


def build_region_monitor_selection(
    dataset: RegionSubsetDataset,
    *,
    benchmark_build_id: str,
    benchmark_payload_sha256: str,
    seed: int,
) -> RegionMonitorSelection:
    items = tuple(
        ValidationItem(
            dataset_index=index,
            record_id=str(row["record_id"]),
            parent_id=str(row["parent_id"]),
            source=str(row["source"]),
            task_family=REGION_TASK,
        )
        for index, row in enumerate(dataset.records)
    )
    max_parents = len({item.parent_id for item in items})
    if not items or max_parents == 0:
        raise ModelError(ReasonCode.VALIDATION_SELECTION_INVALID, "Region monitor 为空")
    payload = {
        "schema_version": STAGE5_SELECTION_SCHEMA,
        "benchmark_build_id": benchmark_build_id,
        "benchmark_payload_sha256": benchmark_payload_sha256,
        "seed": seed,
        "max_parents": max_parents,
        "items": [item.to_dict() for item in items],
    }
    return RegionMonitorSelection(
        benchmark_build_id=benchmark_build_id,
        benchmark_payload_sha256=benchmark_payload_sha256,
        seed=seed,
        max_parents=max_parents,
        items=items,
        source_counts=dict(Counter(item.source for item in items)),
        task_counts={REGION_TASK: len(items)},
        selection_sha256=sha256_text(canonical_json(payload)),
    )


def evaluate_region_monitor_loss(
    *,
    model: Any,
    collator: Any,
    dataset: RegionSubsetDataset,
    selection: RegionMonitorSelection,
    device: torch.device,
    step: int,
    progress: Any = None,
) -> ValidationResult:
    """对全部 parent-isolated monitor rows 计算 teacher-forced loss。"""

    started = time.perf_counter()
    rng = capture_rng_state()
    was_training = bool(model.model.training)
    losses: list[float] = []
    tokens = 0
    images = 0
    if progress is not None:
        progress.start_validation(step=step, total_samples=len(selection.items))
    try:
        model.eval()
        with torch.inference_mode():
            for completed, item in enumerate(selection.items, 1):
                sample = dataset[item.dataset_index]
                if (
                    sample.record_id != item.record_id
                    or sample.parent_id != item.parent_id
                    or sample.logical_role != REGION_MONITOR_ROLE
                ):
                    raise ModelError(ReasonCode.VALIDATION_SELECTION_INVALID, "Region monitor selection 漂移")
                batch = {
                    key: value.to(device) if isinstance(value, Tensor) else value
                    for key, value in collator([sample]).items()
                }
                loss = getattr(model.forward(batch), "loss", None)
                if not isinstance(loss, Tensor) or loss.ndim != 0:
                    raise ModelError(ReasonCode.LOSS_MASK_INVALID, "Region monitor 必须返回标量 loss")
                value = float(loss.detach().cpu())
                if not math.isfinite(value):
                    raise ModelError(ReasonCode.NONFINITE_NUMBER, "Region monitor loss 非有限")
                losses.append(value)
                tokens += sum(batch["input_token_counts"])
                images += sum(batch["image_counts"])
                if progress is not None:
                    progress.update_validation(
                        completed=completed,
                        total_samples=len(selection.items),
                        running_loss=sum(losses) / len(losses),
                    )
    finally:
        restore_rng_state(rng)
        model.train() if was_training else model.eval()
    overall = sum(losses) / len(losses)
    duration = time.perf_counter() - started
    if progress is not None:
        progress.finish_validation(
            step=step,
            macro_task_loss=overall,
            overall_loss=overall,
            duration_seconds=duration,
        )
    return ValidationResult(
        step=step,
        macro_task_loss=overall,
        overall_loss=overall,
        task_losses={REGION_TASK: overall},
        task_counts={REGION_TASK: len(losses)},
        sample_count=len(losses),
        parent_count=len({item.parent_id for item in selection.items}),
        input_tokens=tokens,
        images=images,
        duration_seconds=duration,
    )


def parse_region_monitor_result(row: Mapping[str, Any]) -> ValidationResult:
    """恢复 Stage 5 validation history；允许同一 parent 的多个监督视图。"""

    expected = {
        "schema_version", "step", "selection_metric", "macro_task_loss",
        "overall_loss", "task_losses", "task_counts", "sample_count",
        "parent_count", "input_tokens", "images", "duration_seconds",
        "formal_acceptance",
    }
    numeric = (
        "macro_task_loss", "overall_loss", "duration_seconds",
    )
    integer = ("step", "sample_count", "parent_count", "input_tokens", "images")
    if (
        set(row) != expected
        or row.get("schema_version") != "rs_vlm.validation_result.v1"
        or row.get("selection_metric") != "macro_task_loss"
        or row.get("formal_acceptance") is not False
        or set(row.get("task_losses", {})) != {REGION_TASK}
        or set(row.get("task_counts", {})) != {REGION_TASK}
        or any(
            isinstance(row.get(key), bool)
            or not isinstance(row.get(key), (int, float))
            or not math.isfinite(float(row[key]))
            for key in numeric
        )
        or any(
            isinstance(row.get(key), bool)
            or not isinstance(row.get(key), int)
            or row[key] <= 0
            for key in integer
        )
    ):
        raise ModelError(ReasonCode.CHECKPOINT_INCOMPATIBLE, "Region validation history 非法")
    task_loss = row["task_losses"][REGION_TASK]
    task_count = row["task_counts"][REGION_TASK]
    if (
        isinstance(task_loss, bool)
        or not isinstance(task_loss, (int, float))
        or not math.isfinite(float(task_loss))
        or isinstance(task_count, bool)
        or not isinstance(task_count, int)
        or task_count != row["sample_count"]
        or not 0 < row["parent_count"] <= row["sample_count"]
        or not math.isclose(float(task_loss), float(row["overall_loss"]), rel_tol=1e-12)
        or not math.isclose(float(task_loss), float(row["macro_task_loss"]), rel_tol=1e-12)
    ):
        raise ModelError(ReasonCode.CHECKPOINT_INCOMPATIBLE, "Region validation 聚合身份非法")
    return ValidationResult(
        step=int(row["step"]),
        macro_task_loss=float(row["macro_task_loss"]),
        overall_loss=float(row["overall_loss"]),
        task_losses={REGION_TASK: float(task_loss)},
        task_counts={REGION_TASK: int(task_count)},
        sample_count=int(row["sample_count"]),
        parent_count=int(row["parent_count"]),
        input_tokens=int(row["input_tokens"]),
        images=int(row["images"]),
        duration_seconds=float(row["duration_seconds"]),
    )
