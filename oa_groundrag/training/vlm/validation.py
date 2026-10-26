"""External 有界验证子集、teacher-forced loss 与 checkpoint 选择合同。"""

from __future__ import annotations

import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

import torch
from torch import Tensor

from oa_groundrag.artifacts.identity import (
    canonical_json,
    sha256_text,
    stable_hash,
)

from oa_groundrag.vlm.checkpoint import capture_rng_state, restore_rng_state
from oa_groundrag.vlm.data import REQUIRED_EXTERNAL_SOURCES, REQUIRED_EXTERNAL_TASKS, DescriptionSample
from oa_groundrag.vlm.errors import ModelError, ReasonCode
from oa_groundrag.vlm.processing import DescriptionCollator


VALIDATION_SELECTION_SCHEMA_VERSION = (
    "rs_vlm.validation_selection.v1"
)
VALIDATION_RESULT_SCHEMA_VERSION = (
    "rs_vlm.validation_result.v1"
)


class ValidationDataset(Protocol):
    records: Sequence[Mapping[str, Any]]

    def __len__(self) -> int:
        ...

    def __getitem__(self, index: int) -> DescriptionSample:
        ...


class ValidationModel(Protocol):
    model: torch.nn.Module

    def train(self) -> None:
        ...

    def eval(self) -> None:
        ...

    def forward(self, batch: Mapping[str, Any]) -> Any:
        ...


class ValidationProgress(Protocol):
    def start_validation(self, *, step: int, total_samples: int) -> None:
        ...

    def update_validation(
        self,
        *,
        completed: int,
        total_samples: int,
        running_loss: float,
    ) -> None:
        ...

    def finish_validation(
        self,
        *,
        step: int,
        macro_task_loss: float,
        overall_loss: float,
        duration_seconds: float,
    ) -> None:
        ...


@dataclass(frozen=True)
class ValidationItem:
    dataset_index: int
    record_id: str
    parent_id: str
    source: str
    task_family: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_index": self.dataset_index,
            "record_id": self.record_id,
            "parent_id": self.parent_id,
            "source": self.source,
            "task_family": self.task_family,
        }


@dataclass(frozen=True)
class ValidationSelection:
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
            "schema_version": VALIDATION_SELECTION_SCHEMA_VERSION,
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
            "role": "external_val",
            "algorithm": "external_val_parent_task_cover.v1",
            "source_counts": dict(sorted(self.source_counts.items())),
            "task_counts": dict(sorted(self.task_counts.items())),
            "items": [item.to_dict() for item in self.items],
            "formal_acceptance": False,
        }


@dataclass(frozen=True)
class ValidationResult:
    step: int
    macro_task_loss: float
    overall_loss: float
    task_losses: Mapping[str, float]
    task_counts: Mapping[str, int]
    sample_count: int
    parent_count: int
    input_tokens: int
    images: int
    duration_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": VALIDATION_RESULT_SCHEMA_VERSION,
            "step": self.step,
            "selection_metric": "macro_task_loss",
            "macro_task_loss": self.macro_task_loss,
            "overall_loss": self.overall_loss,
            "task_losses": dict(sorted(self.task_losses.items())),
            "task_counts": dict(sorted(self.task_counts.items())),
            "sample_count": self.sample_count,
            "parent_count": self.parent_count,
            "input_tokens": self.input_tokens,
            "images": self.images,
            "duration_seconds": self.duration_seconds,
            "formal_acceptance": False,
        }


def _validation_candidates(
    dataset: ValidationDataset,
) -> list[ValidationItem]:
    output: list[ValidationItem] = []
    for index, record in enumerate(dataset.records):
        if record.get("logical_role") != "external_val":
            raise ModelError(
                ReasonCode.VALIDATION_SELECTION_INVALID,
                "有界验证 Dataset 只能包含 external_val",
            )
        values = {
            "record_id": record.get("record_id"),
            "parent_id": record.get("parent_id"),
            "source": record.get("source"),
            "task_family": record.get("task_family"),
        }
        if not all(
            isinstance(value, str) and value
            for value in values.values()
        ):
            raise ModelError(
                ReasonCode.VALIDATION_SELECTION_INVALID,
                "external_val record 缺少稳定身份字段",
                details={"dataset_index": index},
            )
        output.append(ValidationItem(dataset_index=index, **values))
    if not output:
        raise ModelError(
            ReasonCode.VALIDATION_SELECTION_INVALID,
            "external_val Dataset 为空",
        )
    return output


def _task_parent_matching(
    candidates: Sequence[ValidationItem],
    *,
    seed: int,
    required_tasks: Sequence[str],
) -> dict[str, ValidationItem]:
    by_task: dict[str, list[ValidationItem]] = defaultdict(list)
    for item in candidates:
        by_task[item.task_family].append(item)
    missing = sorted(set(required_tasks) - set(by_task))
    if missing:
        raise ModelError(
            ReasonCode.VALIDATION_SELECTION_INVALID,
            f"external_val 缺少 task family：{missing}",
        )
    for task, values in by_task.items():
        values.sort(
            key=lambda item: (
                stable_hash(
                    seed,
                    "external_val_selection.v1",
                    task,
                    item.parent_id,
                    item.record_id,
                ),
                item.record_id,
            )
        )
    ordered_tasks = sorted(
        required_tasks,
        key=lambda task: (
            len({item.parent_id for item in by_task[task]}),
            stable_hash(seed, "validation_task_order", task),
        ),
    )
    parent_owner: dict[str, str] = {}
    choices: dict[str, ValidationItem] = {}

    def assign(task: str, seen_parents: set[str]) -> bool:
        for item in by_task[task]:
            if item.parent_id in seen_parents:
                continue
            seen_parents.add(item.parent_id)
            previous = parent_owner.get(item.parent_id)
            if previous is None or assign(previous, seen_parents):
                parent_owner[item.parent_id] = task
                choices[task] = item
                return True
        return False

    for task in ordered_tasks:
        if not assign(task, set()):
            raise ModelError(
                ReasonCode.VALIDATION_SELECTION_INVALID,
                "无法在 parent 隔离约束下覆盖全部 external_val task",
                details={"task": task},
            )
    return choices


def select_bounded_external_validation(
    dataset: ValidationDataset,
    *,
    benchmark_build_id: str,
    benchmark_payload_sha256: str,
    seed: int,
    max_parents: int,
) -> ValidationSelection:
    """确定性选择至多 128 个不同 parent，并覆盖三源七任务。"""

    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed < 0
        or isinstance(max_parents, bool)
        or not isinstance(max_parents, int)
        or max_parents <= 0
        or max_parents > 128
    ):
        raise ModelError(
            ReasonCode.VALIDATION_SELECTION_INVALID,
            "validation seed/max_parents 非法",
        )
    candidates = _validation_candidates(dataset)
    available_sources = {item.source for item in candidates}
    missing_sources = sorted(REQUIRED_EXTERNAL_SOURCES - available_sources)
    if missing_sources:
        raise ModelError(
            ReasonCode.VALIDATION_SELECTION_INVALID,
            f"external_val 缺少 source：{missing_sources}",
        )
    task_choices = _task_parent_matching(
        candidates,
        seed=seed,
        required_tasks=tuple(sorted(REQUIRED_EXTERNAL_TASKS)),
    )
    selected: list[ValidationItem] = list(task_choices.values())
    selected_parents = {item.parent_id for item in selected}
    selected_sources = {item.source for item in selected}
    for source in sorted(REQUIRED_EXTERNAL_SOURCES - selected_sources):
        choices = [
            item
            for item in candidates
            if item.source == source and item.parent_id not in selected_parents
        ]
        if not choices:
            raise ModelError(
                ReasonCode.VALIDATION_SELECTION_INVALID,
                f"无法用独立 parent 覆盖 external_val source={source}",
            )
        choice = min(
            choices,
            key=lambda item: (
                stable_hash(
                    seed,
                    "external_val_source_cover.v1",
                    source,
                    item.parent_id,
                    item.record_id,
                ),
                item.record_id,
            ),
        )
        selected.append(choice)
        selected_parents.add(choice.parent_id)
    if len(selected) > max_parents:
        raise ModelError(
            ReasonCode.VALIDATION_SELECTION_INVALID,
            "validation_max_parents 不足以覆盖三源七任务",
            details={"required": len(selected), "configured": max_parents},
        )
    by_parent: dict[str, list[ValidationItem]] = defaultdict(list)
    for item in candidates:
        by_parent[item.parent_id].append(item)
    representatives = []
    for parent_id, values in by_parent.items():
        if parent_id in selected_parents:
            continue
        representatives.append(
            min(
                values,
                key=lambda item: (
                    stable_hash(
                        seed,
                        "external_val_parent_record.v1",
                        item.parent_id,
                        item.record_id,
                    ),
                    item.record_id,
                ),
            )
        )
    representatives.sort(
        key=lambda item: (
            stable_hash(
                seed,
                "external_val_parent_fill.v1",
                item.parent_id,
            ),
            item.parent_id,
        )
    )
    selected.extend(representatives[: max_parents - len(selected)])
    selected.sort(
        key=lambda item: (
            stable_hash(
                seed,
                "external_val_final_order.v1",
                item.parent_id,
                item.record_id,
            ),
            item.record_id,
        )
    )
    if len({item.parent_id for item in selected}) != len(selected):
        raise AssertionError("validation selection parent uniqueness broken")
    source_counts = Counter(item.source for item in selected)
    task_counts = Counter(item.task_family for item in selected)
    if set(source_counts) != REQUIRED_EXTERNAL_SOURCES or set(
        task_counts
    ) != REQUIRED_EXTERNAL_TASKS:
        raise AssertionError("validation selection coverage broken")
    hash_payload = {
        "schema_version": VALIDATION_SELECTION_SCHEMA_VERSION,
        "benchmark_build_id": benchmark_build_id,
        "benchmark_payload_sha256": benchmark_payload_sha256,
        "seed": seed,
        "max_parents": max_parents,
        "items": [item.to_dict() for item in selected],
    }
    return ValidationSelection(
        benchmark_build_id=benchmark_build_id,
        benchmark_payload_sha256=benchmark_payload_sha256,
        seed=seed,
        max_parents=max_parents,
        items=tuple(selected),
        source_counts=dict(source_counts),
        task_counts=dict(task_counts),
        selection_sha256=sha256_text(canonical_json(hash_payload)),
    )


def _move_batch(
    batch: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    return {
        key: (
            value.to(device=device, non_blocking=device.type == "cuda")
            if isinstance(value, Tensor)
            else value
        )
        for key, value in batch.items()
    }


def evaluate_teacher_forced_loss(
    *,
    model: ValidationModel,
    collator: DescriptionCollator,
    dataset: ValidationDataset,
    selection: ValidationSelection,
    device: torch.device,
    step: int,
    progress: ValidationProgress | None = None,
) -> ValidationResult:
    """在不改变训练 RNG/sampler 的前提下计算逐任务 validation loss。"""

    started = time.perf_counter()
    rng_state = capture_rng_state()
    was_training = bool(model.model.training)
    losses: list[float] = []
    task_values: dict[str, list[float]] = defaultdict(list)
    parent_ids: set[str] = set()
    input_tokens = 0
    images = 0
    if progress is not None:
        progress.start_validation(
            step=step,
            total_samples=len(selection.items),
        )
    try:
        model.eval()
        with torch.inference_mode():
            for completed, item in enumerate(selection.items, 1):
                sample = dataset[item.dataset_index]
                if (
                    sample.record_id != item.record_id
                    or sample.parent_id != item.parent_id
                    or sample.logical_role != "external_val"
                    or sample.task_family != item.task_family
                ):
                    raise ModelError(
                        ReasonCode.VALIDATION_SELECTION_INVALID,
                        "validation selection 与 Dataset 内容不一致",
                        details={"dataset_index": item.dataset_index},
                    )
                batch = _move_batch(collator([sample]), device)
                result = model.forward(batch)
                loss = getattr(result, "loss", None)
                if not isinstance(loss, Tensor) or loss.ndim != 0:
                    raise ModelError(
                        ReasonCode.LOSS_MASK_INVALID,
                        "validation forward 必须返回标量 loss",
                    )
                value = float(loss.detach().cpu())
                if not math.isfinite(value):
                    raise ModelError(
                        ReasonCode.NONFINITE_NUMBER,
                        "external_val loss 非有限",
                        details={"record_id": item.record_id},
                    )
                losses.append(value)
                task_values[item.task_family].append(value)
                parent_ids.add(item.parent_id)
                input_tokens += sum(batch["input_token_counts"])
                images += sum(batch["image_counts"])
                if progress is not None:
                    progress.update_validation(
                        completed=completed,
                        total_samples=len(selection.items),
                        running_loss=sum(losses) / len(losses),
                    )
    finally:
        restore_rng_state(rng_state)
        if was_training:
            model.train()
        else:
            model.eval()
    task_losses = {
        task: sum(values) / len(values)
        for task, values in task_values.items()
    }
    if set(task_losses) != REQUIRED_EXTERNAL_TASKS:
        raise ModelError(
            ReasonCode.VALIDATION_SELECTION_INVALID,
            "validation result 未覆盖全部 task family",
        )
    overall_loss = sum(losses) / len(losses)
    macro_task_loss = sum(task_losses.values()) / len(task_losses)
    duration = time.perf_counter() - started
    if progress is not None:
        progress.finish_validation(
            step=step,
            macro_task_loss=macro_task_loss,
            overall_loss=overall_loss,
            duration_seconds=duration,
        )
    return ValidationResult(
        step=step,
        macro_task_loss=macro_task_loss,
        overall_loss=overall_loss,
        task_losses=task_losses,
        task_counts={
            task: len(values) for task, values in task_values.items()
        },
        sample_count=len(losses),
        parent_count=len(parent_ids),
        input_tokens=input_tokens,
        images=images,
        duration_seconds=duration,
    )


def validation_is_better(
    candidate: ValidationResult,
    incumbent: ValidationResult | None,
) -> bool:
    """macro task loss → overall loss → earlier step 的稳定选择顺序。"""

    if incumbent is None:
        return True
    return (
        candidate.macro_task_loss,
        candidate.overall_loss,
        candidate.step,
    ) < (
        incumbent.macro_task_loss,
        incumbent.overall_loss,
        incumbent.step,
    )
