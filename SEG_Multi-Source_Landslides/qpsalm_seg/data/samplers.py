#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Size-aware and task-balanced batch samplers."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Iterator

from torch.utils.data import Sampler


TASK_WEIGHTS = {"global": 0.4, "referring": 0.4, "no_target": 0.2}


def task_group(row: dict) -> str:
    family = str(row.get("task_family") or "")
    if family == "no_target_segmentation":
        return "no_target"
    if family == "referring_landslide_segmentation":
        return "referring"
    return "global"


def largest_remainder_quota(total: int, labels: list, weights: list[float]) -> dict:
    """Allocate an exact integer total with deterministic largest remainders."""
    if total <= 0 or not labels:
        return {label: 0 for label in labels}
    nonnegative = [max(0.0, float(value)) for value in weights]
    if not any(nonnegative):
        nonnegative = [1.0] * len(labels)
    weight_sum = sum(nonnegative)
    desired = [total * value / weight_sum for value in nonnegative]
    quotas = [int(math.floor(value)) for value in desired]
    remainder = total - sum(quotas)
    order = sorted(
        range(len(labels)),
        key=lambda index: (desired[index] - quotas[index], nonnegative[index], str(labels[index])),
        reverse=True,
    )
    for index in order[:remainder]:
        quotas[index] += 1
    return dict(zip(labels, quotas))


class TaskBalancedSizeBucketBatchSampler(Sampler[list[int]]):
    """Draw task-balanced batches with homogeneous spatial and Qwen sequence load."""

    protocol = "qpsalm_task_balanced_size_bucket_v2_epoch_addressable"

    def __init__(self, dataset, batch_size: int, *, shuffle: bool, seed: int, drop_last: bool = False, task_weights: dict[str, float] | None = None, balance_tasks: bool = True) -> None:
        self.dataset = dataset
        self.batch_size = max(1, int(batch_size))
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.task_weights = {**TASK_WEIGHTS, **(task_weights or {})}
        self.balance_tasks = bool(balance_tasks)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Expose deterministic epoch addressing without changing legacy iteration."""
        if int(epoch) < 0:
            raise ValueError("sampler epoch must be non-negative")
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        grouped: dict[tuple[int, int], dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
        for index, row in enumerate(self.dataset.rows):
            load_bucket = (
                self.dataset.sequence_load_bucket(index)
                if hasattr(self.dataset, "sequence_load_bucket") else 0
            )
            grouped[(self.dataset.bucket_size(index), load_bucket)][task_group(row)].append(index)
        rng = random.Random(self.seed + self.epoch)
        batches: list[list[int]] = []
        if not self.balance_tasks:
            for _bucket_key, groups in grouped.items():
                values = [index for group_values in groups.values() for index in group_values]
                if self.shuffle:
                    rng.shuffle(values)
                for start in range(0, len(values), self.batch_size):
                    batch = values[start:start + self.batch_size]
                    if len(batch) == self.batch_size or not self.drop_last:
                        batches.append(batch)
        else:
            available = sorted({name for groups in grouped.values() for name, values in groups.items() if values})
            total_batches = (
                len(self.dataset) // self.batch_size
                if self.drop_last else math.ceil(len(self.dataset) / self.batch_size)
            )
            group_quotas = largest_remainder_quota(
                total_batches,
                available,
                [self.task_weights.get(name, 0.0) for name in available],
            )
            schedule: list[tuple[int, str]] = []
            for group in available:
                buckets = sorted(bucket for bucket, groups in grouped.items() if groups.get(group))
                bucket_quotas = largest_remainder_quota(
                    group_quotas[group],
                    buckets,
                    [len(grouped[bucket][group]) for bucket in buckets],
                )
                schedule.extend(
                    (bucket, group)
                    for bucket in buckets
                    for _ in range(bucket_quotas[bucket])
                )
            cursors: dict[tuple[int, str], int] = defaultdict(int)
            for groups in grouped.values():
                for values in groups.values():
                    if self.shuffle:
                        rng.shuffle(values)
            if self.shuffle:
                rng.shuffle(schedule)
            for bucket, group in schedule:
                values = grouped[bucket][group]
                key = (bucket, group)
                batch: list[int] = []
                parents: set[str] = set()
                attempts = 0
                while len(batch) < self.batch_size and attempts < max(1, len(values) * 2):
                    if cursors[key] >= len(values):
                        cursors[key] = 0
                        if self.shuffle:
                            rng.shuffle(values)
                    candidate = values[cursors[key]]
                    cursors[key] += 1
                    attempts += 1
                    row = self.dataset.rows[candidate]
                    parent = str(row.get("parent_sample_id") or row.get("sample_id"))
                    if candidate in batch or parent in parents:
                        continue
                    batch.append(candidate)
                    parents.add(parent)
                if len(batch) < self.batch_size:
                    for candidate in values:
                        if candidate not in batch:
                            batch.append(candidate)
                        if len(batch) == self.batch_size:
                            break
                while len(batch) < self.batch_size:
                    batch.append(values[len(batch) % len(values)])
                batches.append(batch)
        if self.shuffle:
            rng.shuffle(batches)
        self.epoch += 1
        yield from batches

    def __len__(self) -> int:
        if self.balance_tasks:
            return (
                len(self.dataset) // self.batch_size
                if self.drop_last else math.ceil(len(self.dataset) / self.batch_size)
            )
        bucket_counts: dict[tuple[int, int], int] = defaultdict(int)
        for index in range(len(self.dataset)):
            load_bucket = (
                self.dataset.sequence_load_bucket(index)
                if hasattr(self.dataset, "sequence_load_bucket") else 0
            )
            bucket_counts[(self.dataset.bucket_size(index), load_bucket)] += 1
        if self.drop_last:
            return sum(count // self.batch_size for count in bucket_counts.values())
        return sum(math.ceil(count / self.batch_size) for count in bucket_counts.values())


SizeBucketBatchSampler = TaskBalancedSizeBucketBatchSampler
