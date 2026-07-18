#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""绑定并重验证 description 所复用的 segmentation instruction population。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from qpsalm_seg.indexing import iter_jsonl, should_skip_row
from qpsalm_seg.paths import resolve_project_path

from ..protocols.io import (
    canonical_sha256 as _canonical_sha256,
    sha256_file as _sha256_file,
)


SEGMENTATION_INSTRUCTION_SOURCE_PROTOCOL = (
    "qpsalm_segmentation_instruction_source_binding_v1"
)


def _resolved_index(value: Any) -> Path:
    path = resolve_project_path(str(value or ""))
    if path is None or not path.is_file():
        raise FileNotFoundError(f"segmentation instruction index 不存在: {value}")
    return path.resolve(strict=False)


def _filtered_rows(
    index_path: Path, task_families: Iterable[str]
) -> list[dict[str, Any]]:
    families = tuple(str(value) for value in task_families)
    if not families or any(not value for value in families):
        raise ValueError("segmentation instruction source 缺少 task_families")
    if len(families) != len(set(families)):
        raise ValueError("segmentation instruction source task_families 重复")
    return [
        row for row in iter_jsonl(index_path)
        if should_skip_row(row, families) is None
    ]


def build_segmentation_instruction_source_binding(
    config: Any,
    split: str,
    runtime_rows: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Freeze the exact segmentation rows used by online target resolution."""
    index_path = _resolved_index(config.index_path(split))
    task_families = [str(value) for value in config.task_families]
    rows = _filtered_rows(index_path, task_families)
    observed = list(runtime_rows)
    if observed != rows:
        raise ValueError(
            "segmentation runtime rows 与 instruction index/task-family 过滤结果不一致"
        )
    return {
        "protocol": SEGMENTATION_INSTRUCTION_SOURCE_PROTOCOL,
        "split": str(split),
        "index": str(index_path),
        "index_sha256": _sha256_file(index_path),
        "index_bytes": int(index_path.stat().st_size),
        "task_families": task_families,
        "filtered_rows": len(rows),
        "filtered_rows_sha256": _canonical_sha256(rows),
    }


def revalidate_segmentation_instruction_source_binding(
    binding: Any,
) -> list[dict[str, Any]]:
    """Reload a bound index and reject file, filtering or population drift."""
    if not isinstance(binding, dict) or (
        binding.get("protocol") != SEGMENTATION_INSTRUCTION_SOURCE_PROTOCOL
    ):
        raise ValueError("segmentation instruction source binding protocol 不兼容")
    split = str(binding.get("split") or "")
    if split not in {"train", "val", "test"}:
        raise ValueError("segmentation instruction source split 非法")
    index_path = _resolved_index(binding.get("index"))
    if (
        _sha256_file(index_path) != str(binding.get("index_sha256") or "")
        or int(index_path.stat().st_size) != int(binding.get("index_bytes", -1))
    ):
        raise ValueError("segmentation instruction source index 已漂移")
    task_families = binding.get("task_families")
    if not isinstance(task_families, list):
        raise ValueError("segmentation instruction source task_families 非法")
    rows = _filtered_rows(index_path, task_families)
    if (
        len(rows) != int(binding.get("filtered_rows", -1))
        or _canonical_sha256(rows)
        != str(binding.get("filtered_rows_sha256") or "")
    ):
        raise ValueError("segmentation instruction resolver population 已漂移")
    return rows
