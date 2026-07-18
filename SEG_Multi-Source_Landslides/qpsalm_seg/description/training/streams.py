"""Deterministic description stream cursors and resume replay."""

from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
from typing import Any

from ..data.loaders import set_loader_epoch
from ..protocols.io import canonical_sha256, strict_json_loads
from .checkpoint import capture_training_rng_state, restore_training_rng_state


DESCRIPTION_TRAINING_PROGRESS_PROTOCOL = (
    "qpsalm_description_training_progress_v1_loader_cursor_bound"
)
DESCRIPTION_STREAM_CURSOR_PROTOCOL = "qpsalm_description_stream_cursor_v1"
DESCRIPTION_STREAM_BINDING_PROTOCOL = "qpsalm_description_stream_binding_v1"


def description_stream_binding(
    name: str,
    stream: dict[str, Any],
    data_audit: dict[str, Any],
) -> dict[str, Any]:
    loader = stream["loader"]
    seed = int(getattr(loader, "_qpsalm_loader_seed", stream["config"].training.seed))
    set_loader_epoch(loader, 0, loader_seed=seed)
    batches = len(loader)
    if batches <= 0:
        raise RuntimeError(f"description stream={name} 没有可训练 batch")
    sampler = loader.batch_sampler
    binding = {
        "protocol": DESCRIPTION_STREAM_BINDING_PROTOCOL,
        "stream": name,
        "stage": str(stream["config"].training.stage),
        "dataset_audit_sha256": canonical_sha256(data_audit),
        "dataset_samples": len(stream["dataset"]),
        "epoch_zero_batches": int(batches),
        "loader_seed": seed,
        "num_workers": int(loader.num_workers),
        "persistent_workers": bool(loader.persistent_workers),
        "batch_sampler": {
            "class": type(sampler).__name__,
            "protocol": getattr(sampler, "protocol", None),
            "batch_size": getattr(sampler, "batch_size", None),
            "seed": getattr(sampler, "seed", None),
            "drop_last": getattr(sampler, "drop_last", None),
            "gradient_window_batches": getattr(
                sampler, "gradient_window_batches", None
            ),
        },
    }
    if binding["persistent_workers"]:
        raise RuntimeError(
            f"description stream={name} 必须关闭 persistent_workers 才能重放 cursor"
        )
    binding["binding_sha256"] = canonical_sha256(binding)
    return binding


def initial_description_stream_states(
    bindings: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "epoch": 0,
            "batch_in_epoch": 0,
            "total_microbatches": 0,
            "batches_per_epoch": int(binding["epoch_zero_batches"]),
            "completed_epoch_batches": [],
        }
        for name, binding in bindings.items()
    }


def description_training_progress_payload(
    *,
    step: int,
    stream_pattern: tuple[str, ...],
    grad_accum_steps: int,
    stream_states: dict[str, dict[str, Any]],
    stream_bindings: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    streams = set(stream_bindings)
    if not stream_pattern or set(stream_pattern) != streams or set(stream_states) != streams:
        raise RuntimeError("description progress stream pattern/binding/state 集合不一致")
    grad_accum_steps = max(1, int(grad_accum_steps))
    optimizer_steps = Counter(
        stream_pattern[index % len(stream_pattern)] for index in range(int(step))
    )
    cursors = {}
    for name in sorted(streams):
        state = stream_states[name]
        completed = [int(value) for value in state["completed_epoch_batches"]]
        total = int(state["total_microbatches"])
        expected_total = int(optimizer_steps[name]) * grad_accum_steps
        if (
            len(completed) != int(state["epoch"])
            or any(value <= 0 for value in completed)
            or sum(completed) + int(state["batch_in_epoch"]) != total
            or total != expected_total
            or int(state["batches_per_epoch"]) <= int(state["batch_in_epoch"])
        ):
            raise RuntimeError(f"description stream={name} cursor 与 step 不一致")
        cursors[name] = {
            "protocol": DESCRIPTION_STREAM_CURSOR_PROTOCOL,
            "epoch": int(state["epoch"]),
            "batch_in_epoch": int(state["batch_in_epoch"]),
            "total_microbatches": total,
            "batches_per_epoch": int(state["batches_per_epoch"]),
            "completed_epoch_batches": completed,
            "stream_binding_sha256": stream_bindings[name]["binding_sha256"],
        }
    return {
        "protocol": DESCRIPTION_TRAINING_PROGRESS_PROTOCOL,
        "step": int(step),
        "stream_pattern": list(stream_pattern),
        "stream_pattern_sha256": canonical_sha256(list(stream_pattern)),
        "grad_accum_steps": grad_accum_steps,
        "optimizer_steps": {
            name: int(optimizer_steps[name]) for name in sorted(streams)
        },
        "stream_cursors": cursors,
    }


def restore_description_training_progress(
    saved: dict[str, Any],
    *,
    checkpoint_step: int,
    required: bool,
    stream_pattern: tuple[str, ...],
    grad_accum_steps: int,
    train_streams: dict[str, dict[str, Any]],
    stream_bindings: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    if not saved:
        if required:
            raise RuntimeError("description resume checkpoint 缺少 training_progress")
        return initial_description_stream_states(stream_bindings)
    streams = set(stream_bindings)
    grad_accum_steps = max(1, int(grad_accum_steps))
    if (
        saved.get("protocol") != DESCRIPTION_TRAINING_PROGRESS_PROTOCOL
        or saved.get("step") != int(checkpoint_step)
    ):
        raise RuntimeError("description resume training_progress protocol/step 不一致")
    if (
        tuple(saved.get("stream_pattern") or ()) != stream_pattern
        or saved.get("stream_pattern_sha256")
        != canonical_sha256(list(stream_pattern))
        or saved.get("grad_accum_steps") != grad_accum_steps
    ):
        raise RuntimeError("description resume stream pattern/grad accumulation 已变化")
    optimizer_steps = saved.get("optimizer_steps")
    cursors = saved.get("stream_cursors")
    if (
        not isinstance(optimizer_steps, dict)
        or not isinstance(cursors, dict)
        or set(optimizer_steps) != streams
        or set(cursors) != streams
        or set(train_streams) != streams
    ):
        raise RuntimeError("description resume stream progress 集合不完整")
    expected_steps = Counter(
        stream_pattern[index % len(stream_pattern)]
        for index in range(int(checkpoint_step))
    )
    restored = {}
    for name in sorted(streams):
        if optimizer_steps[name] != int(expected_steps[name]):
            raise RuntimeError(f"description resume stream={name} optimizer steps 非法")
        cursor = cursors[name]
        if not isinstance(cursor, dict) or cursor.get("protocol") != DESCRIPTION_STREAM_CURSOR_PROTOCOL:
            raise RuntimeError(f"description resume stream={name} cursor protocol 非法")
        if cursor.get("stream_binding_sha256") != stream_bindings[name]["binding_sha256"]:
            raise RuntimeError(f"description resume stream={name} loader/data binding 已变化")
        completed = cursor.get("completed_epoch_batches")
        if not isinstance(completed, list) or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in completed
        ):
            raise RuntimeError(f"description resume stream={name} epoch history 非法")
        integer_fields = (
            "epoch", "batch_in_epoch", "total_microbatches", "batches_per_epoch",
        )
        if any(
            isinstance(cursor.get(field), bool)
            or not isinstance(cursor.get(field), int)
            or int(cursor[field]) < 0
            for field in integer_fields
        ):
            raise RuntimeError(f"description resume stream={name} cursor 字段非法")
        expected_total = int(expected_steps[name]) * grad_accum_steps
        if (
            len(completed) != cursor["epoch"]
            or sum(completed) + cursor["batch_in_epoch"] != cursor["total_microbatches"]
            or cursor["total_microbatches"] != expected_total
            or cursor["batches_per_epoch"] <= cursor["batch_in_epoch"]
        ):
            raise RuntimeError(f"description resume stream={name} cursor 与 step 不一致")
        loader = train_streams[name]["loader"]
        set_loader_epoch(
            loader,
            cursor["epoch"],
            loader_seed=int(stream_bindings[name]["loader_seed"]),
        )
        if len(loader) != cursor["batches_per_epoch"]:
            raise RuntimeError(f"description resume stream={name} 当前 epoch batch 数已变化")
        restored[name] = {
            "epoch": cursor["epoch"],
            "batch_in_epoch": cursor["batch_in_epoch"],
            "total_microbatches": cursor["total_microbatches"],
            "batches_per_epoch": cursor["batches_per_epoch"],
            "completed_epoch_batches": list(completed),
        }
    return restored


def description_iterator_at_cursor(
    stream: dict[str, Any],
    state: dict[str, Any],
    binding: dict[str, Any],
):
    loader = stream["loader"]
    set_loader_epoch(
        loader,
        int(state["epoch"]),
        loader_seed=int(binding["loader_seed"]),
    )
    if len(loader) != int(state["batches_per_epoch"]):
        raise RuntimeError("description stream 当前 epoch batch 数与 cursor 不一致")
    iterator = iter(loader)
    cursor = int(state["batch_in_epoch"])
    if cursor <= 0:
        return iterator
    rng_state = capture_training_rng_state()
    try:
        for _ in range(cursor):
            try:
                next(iterator)
            except StopIteration as exc:
                raise RuntimeError("description stream cursor 超出当前 epoch") from exc
    finally:
        restore_training_rng_state(rng_state)
    return iterator


def next_description_stream_batch(
    stream: dict[str, Any],
    iterator,
    state: dict[str, Any],
    binding: dict[str, Any],
):
    if iterator is None:
        iterator = description_iterator_at_cursor(stream, state, binding)
    try:
        batch = next(iterator)
    except StopIteration as exc:
        raise RuntimeError("description stream 在 batches_per_epoch 前提前耗尽") from exc
    state["total_microbatches"] += 1
    state["batch_in_epoch"] += 1
    if state["batch_in_epoch"] == state["batches_per_epoch"]:
        state["completed_epoch_batches"].append(state["batches_per_epoch"])
        state["epoch"] += 1
        state["batch_in_epoch"] = 0
        loader = stream["loader"]
        set_loader_epoch(
            loader,
            state["epoch"],
            loader_seed=int(binding["loader_seed"]),
        )
        state["batches_per_epoch"] = len(loader)
        if state["batches_per_epoch"] <= 0:
            raise RuntimeError("description stream 新 epoch 没有可训练 batch")
        iterator = None
    return batch, iterator


def load_best(path: Path) -> float:
    if not path.is_file():
        return -math.inf
    try:
        return float(
            strict_json_loads(path.read_text(encoding="utf-8")).get(
                "selection_score", -math.inf
            )
        )
    except (json.JSONDecodeError, TypeError, ValueError):
        return -math.inf
