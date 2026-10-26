"""确定性训练 batch 规划与有序 CPU processor 预取。"""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Protocol, Sequence

from torch import Tensor

from oa_groundrag.vlm.data import DescriptionSample
from oa_groundrag.vlm.errors import ModelError, ReasonCode
from oa_groundrag.vlm.processing import DescriptionCollator


INPUT_PIPELINE_SYNC = "synchronous.v1"
INPUT_PIPELINE_PREFETCH = "ordered_thread_prefetch.v1"


class PlanningDataset(Protocol):
    records: Sequence[Mapping[str, Any]]
    manifest: Mapping[str, Any]

    def __getitem__(self, index: int) -> DescriptionSample:
        ...

    def set_epoch(self, epoch: int) -> None:
        ...


class EpochSampler(Protocol):
    def set_epoch(self, epoch: int) -> None:
        ...

    def __iter__(self) -> Iterator[int]:
        ...

    def state_dict(self) -> dict[str, Any]:
        ...


@dataclass(frozen=True)
class PlannedBatch:
    """已冻结消息及其消费后游标；可安全交给 processor worker。"""

    samples: tuple[DescriptionSample, ...]
    sample_epochs: tuple[int, ...]
    next_epoch: int
    next_sample_offset: int


@dataclass(frozen=True)
class PreparedBatch:
    plan: PlannedBatch
    batch: Mapping[str, Any]


class DeterministicBatchPlanner:
    """仅在调用线程推进 sampler/Dataset，预取不改变消费顺序。"""

    def __init__(
        self,
        *,
        dataset: PlanningDataset,
        sampler: EpochSampler,
        batch_size: int,
        max_epochs: int,
        start_epoch: int,
        start_sample_offset: int,
        max_batches: int,
    ) -> None:
        for label, value, minimum in (
            ("batch_size", batch_size, 1),
            ("max_epochs", max_epochs, 1),
            ("start_epoch", start_epoch, 0),
            ("start_sample_offset", start_sample_offset, 0),
            ("max_batches", max_batches, 0),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < minimum
            ):
                raise ModelError(
                    ReasonCode.TYPE_MISMATCH,
                    f"{label} 必须是 >= {minimum} 的整数",
                )
        self.dataset = dataset
        self.sampler = sampler
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.epoch = start_epoch
        self.sample_offset = start_sample_offset
        self.max_batches = max_batches
        self.planned_batches = 0
        self._order_epoch: int | None = None
        self._order: list[int] = []

    def next_batch(self) -> PlannedBatch:
        if self.planned_batches >= self.max_batches:
            raise StopIteration
        samples: list[DescriptionSample] = []
        sample_epochs: list[int] = []
        while len(samples) < self.batch_size:
            if self.epoch >= self.max_epochs:
                raise ModelError(
                    ReasonCode.TYPE_MISMATCH,
                    "训练 epochs 已耗尽但尚未达到 max_steps；拒绝隐式循环",
                )
            if self._order_epoch != self.epoch:
                self.sampler.set_epoch(self.epoch)
                self._order = list(iter(self.sampler))
                self._order_epoch = self.epoch
                if self.sample_offset > len(self._order):
                    raise ModelError(
                        ReasonCode.CHECKPOINT_INCOMPATIBLE,
                        "起始 sample_offset 超出当前 sampler order",
                    )
            if self.sample_offset == len(self._order):
                self.epoch += 1
                self.sample_offset = 0
                self._order_epoch = None
                continue
            dataset_index = self._order[self.sample_offset]
            sample_epoch = self.epoch
            samples.append(self.dataset[dataset_index])
            sample_epochs.append(sample_epoch)
            self.sample_offset += 1
            if self.sample_offset == len(self._order):
                self.epoch += 1
                self.sample_offset = 0
                self._order_epoch = None
        self.planned_batches += 1
        return PlannedBatch(
            samples=tuple(samples),
            sample_epochs=tuple(sample_epochs),
            next_epoch=self.epoch,
            next_sample_offset=self.sample_offset,
        )


def sampler_state_at_epoch(
    sampler: EpochSampler,
    *,
    epoch: int,
) -> dict[str, Any]:
    """生成已消费游标的 sampler 状态，不回拨正在预取的 Dataset。"""

    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise ModelError(
            ReasonCode.TYPE_MISMATCH,
            "sampler checkpoint epoch 必须是 >= 0 的整数",
        )
    state = dict(sampler.state_dict())
    if "epoch" not in state:
        raise ModelError(
            ReasonCode.CHECKPOINT_INCOMPATIBLE,
            "sampler state 缺少 epoch",
        )
    state["epoch"] = epoch
    return state


def _prepare(
    collator: DescriptionCollator,
    plan: PlannedBatch,
    *,
    pin_memory: bool,
) -> PreparedBatch:
    batch = dict(collator(plan.samples))
    if pin_memory:
        for key, value in tuple(batch.items()):
            if isinstance(value, Tensor) and value.device.type == "cpu":
                batch[key] = value if value.is_pinned() else value.pin_memory()
    return PreparedBatch(plan=plan, batch=batch)


class OrderedBatchPrefetcher:
    """按规划顺序返回 batch；worker 完成先后不会影响训练轨迹。"""

    def __init__(
        self,
        *,
        planner: DeterministicBatchPlanner,
        collator: DescriptionCollator,
        num_workers: int,
        prefetch_factor: int,
        pin_memory: bool,
        worker_collators: Sequence[DescriptionCollator] | None = None,
    ) -> None:
        if (
            isinstance(num_workers, bool)
            or not isinstance(num_workers, int)
            or num_workers < 0
        ):
            raise ModelError(
                ReasonCode.TYPE_MISMATCH,
                "num_workers 必须是 >= 0 的整数",
            )
        if (
            isinstance(prefetch_factor, bool)
            or not isinstance(prefetch_factor, int)
            or prefetch_factor < 0
        ):
            raise ModelError(
                ReasonCode.TYPE_MISMATCH,
                "prefetch_factor 必须是 >= 0 的整数",
            )
        if num_workers == 0 and prefetch_factor != 0:
            raise ModelError(
                ReasonCode.TYPE_MISMATCH,
                "同步输入要求 prefetch_factor=0",
            )
        if num_workers > 0 and prefetch_factor <= 0:
            raise ModelError(
                ReasonCode.TYPE_MISMATCH,
                "异步输入要求 prefetch_factor>0",
            )
        self.planner = planner
        self.pin_memory = bool(pin_memory)
        self._closed = False
        self._synchronous = num_workers == 0
        self._collator = collator
        self._executors: list[ThreadPoolExecutor] = []
        self._worker_collators: list[DescriptionCollator] = []
        self._pending: deque[
            tuple[PlannedBatch, Future[PreparedBatch]]
        ] = deque()
        self._next_worker = 0
        self._max_pending = 0
        if not self._synchronous:
            if worker_collators is None:
                self._worker_collators = [
                    collator.clone_for_worker() for _ in range(num_workers)
                ]
            else:
                if len(worker_collators) != num_workers:
                    raise ModelError(
                        ReasonCode.TYPE_MISMATCH,
                        "worker_collators 数量必须等于 num_workers",
                    )
                self._worker_collators = list(worker_collators)
            self._executors = [
                ThreadPoolExecutor(
                    max_workers=1,
                    thread_name_prefix=f"vlm-input-{index}",
                )
                for index in range(num_workers)
            ]
            self._max_pending = num_workers * prefetch_factor
            self._fill()

    def __enter__(self) -> "OrderedBatchPrefetcher":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def __iter__(self) -> "OrderedBatchPrefetcher":
        return self

    def __next__(self) -> PreparedBatch:
        if self._closed:
            raise StopIteration
        if self._synchronous:
            plan = self.planner.next_batch()
            return _prepare(
                self._collator,
                plan,
                pin_memory=self.pin_memory,
            )
        if not self._pending:
            raise StopIteration
        _, future = self._pending.popleft()
        prepared = future.result()
        self._fill()
        return prepared

    def _fill(self) -> None:
        while len(self._pending) < self._max_pending:
            try:
                plan = self.planner.next_batch()
            except StopIteration:
                return
            worker = self._next_worker
            self._next_worker = (worker + 1) % len(self._executors)
            future = self._executors[worker].submit(
                _prepare,
                self._worker_collators[worker],
                plan,
                pin_memory=self.pin_memory,
            )
            self._pending.append((plan, future))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        while self._pending:
            _, future = self._pending.popleft()
            future.cancel()
        for executor in self._executors:
            executor.shutdown(wait=True, cancel_futures=True)
        self._executors.clear()
        self._worker_collators.clear()
