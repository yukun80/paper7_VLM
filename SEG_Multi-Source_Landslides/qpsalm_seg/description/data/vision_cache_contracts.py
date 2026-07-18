#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stable Description Vision Cache v1 formats and record validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from qpsalm_seg.paths import resolve_project_path

from ..protocols.io import (
    sha256_file,
    strict_json_loads,
)


DESCRIPTION_CACHE_FORMAT = "qpsalm_description_vision_cache_v1"
DESCRIPTION_CACHE_PROTOCOL = "task_neutral_parent_visual_features_v1"
DESCRIPTION_CACHE_BUILDER_VERSION = (
    "description_vision_cache_m3_v3_shard_content_bound"
)
DESCRIPTION_CACHE_VALIDATION_PROTOCOL = (
    "qpsalm_description_vision_cache_validation_v2_shard_content_bound"
)
DESCRIPTION_CACHE_ARTIFACT_BINDING_PROTOCOL = (
    "qpsalm_description_vision_cache_artifact_binding_v1_validation_bound"
)
DESCRIPTION_CACHE_SHARD_REPLAY_PROTOCOL = (
    "qpsalm_description_vision_cache_shard_replay_v1_sha256_complete"
)
DESCRIPTION_CACHE_ARTIFACT_REVALIDATION_PROTOCOL = (
    "qpsalm_description_vision_cache_artifact_revalidation_v1_checkpoint_bound"
)


def source_cache_snapshot(cache_dir: str | Path) -> dict[str, Any]:
    """Fingerprint cache-v3 metadata without reading or rewriting large shard payloads."""
    root = resolve_project_path(cache_dir) or Path(cache_dir)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"源 vision cache manifest 不存在: {manifest_path}")
    manifest = strict_json_loads(manifest_path.read_text(encoding="utf-8"))
    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError(f"源 vision cache shards 非法: {manifest_path}")
    names = ["manifest.json", *(str(value) for value in shards)]
    entries = []
    for name in names:
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"源 vision cache 文件不存在: {path}")
        stat = path.stat()
        entries.append({
            "path": name,
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        })
    payload = json.dumps(
        entries,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return {
        "manifest_sha256": sha256_file(manifest_path),
        "metadata_fingerprint": hashlib.sha256(payload).hexdigest(),
        "file_count": len(entries),
    }


def validate_source_cache_snapshot(
    expected: dict[str, Any], cache_dir: str | Path,
) -> list[str]:
    """Compare the current read-only source cache with the build-time snapshot."""
    errors = []
    try:
        current = source_cache_snapshot(cache_dir)
    except (FileNotFoundError, ValueError, OSError, json.JSONDecodeError) as exc:
        return [f"源 segmentation cache 无法验证: {exc}"]
    for key in ("manifest_sha256", "metadata_fingerprint", "file_count"):
        if current.get(key) != expected.get(key):
            errors.append(
                f"源 segmentation cache {key} 已变化: "
                f"expected={expected.get(key)!r} current={current.get(key)!r}"
            )
    return errors


def validate_description_cache_record(
    row: dict[str, Any], manifest: dict[str, Any],
) -> None:
    """Validate one task-neutral record independently of a cache reader.

    The migration workflow uses this public contract to validate legacy tensors
    before it publishes a current manifest. It deliberately checks finiteness in
    addition to the stable v1 shapes and content fingerprint.
    """
    if not isinstance(row, dict):
        raise ValueError("description cache record 必须是 object")
    forbidden = {"instruction", "condition", "region_geometry", "segmentation_state"}

    def leaked_paths(value: Any, path: tuple[str, ...] = ()) -> list[str]:
        leaks: list[str] = []
        if isinstance(value, dict):
            for key, nested in value.items():
                nested_path = (*path, str(key))
                if str(key) in forbidden:
                    leaks.append(".".join(nested_path))
                leaks.extend(leaked_paths(nested, nested_path))
        elif isinstance(value, (list, tuple)):
            for index, nested in enumerate(value):
                leaks.extend(leaked_paths(nested, (*path, str(index))))
        return leaks

    task_leaks = leaked_paths(row)
    if task_leaks:
        raise ValueError(
            f"description cache 包含任务相关字段: {sorted(task_leaks)}"
        )
    required = {
        "lookup_key", "component", "parent_sample_id", "source_ref",
        "source_content_hash", "cache_fingerprint", "views",
    }
    if required - set(row):
        raise ValueError(
            f"description cache record 字段不完整: {row.get('lookup_key')}"
        )
    views = row["views"]
    if not isinstance(views, list) or not views:
        raise ValueError(
            f"description cache record 没有 visual views: {row['lookup_key']}"
        )
    expected_sizes = [int(value) for value in manifest["spatial_sizes"]]
    for view in views:
        if not isinstance(view, dict):
            raise ValueError(
                f"description cache view 必须是 object: {row['lookup_key']}"
            )
        spatial = view.get("spatial_features")
        if not isinstance(spatial, (list, tuple)) or len(spatial) != len(expected_sizes):
            raise ValueError(
                f"description cache spatial layers 非法: {row['lookup_key']}"
            )
        for value, size in zip(spatial, expected_sizes):
            if not torch.is_tensor(value) or tuple(value.shape) != (
                int(manifest["spatial_channels"]), size, size
            ):
                raise ValueError(
                    f"description cache spatial feature shape 非法: {row['lookup_key']}"
                )
            if not bool(torch.isfinite(value).all()):
                raise ValueError(
                    f"description cache spatial feature 非 finite: {row['lookup_key']}"
                )
        tokens = view.get("view_tokens")
        if (
            not torch.is_tensor(tokens)
            or tokens.ndim != 2
            or int(tokens.shape[1]) != int(manifest["token_dim"])
            or int(tokens.shape[0]) <= 0
            or int(tokens.shape[0]) > int(manifest["view_tokens_per_view"])
        ):
            raise ValueError(
                f"description cache view token shape 非法: {row['lookup_key']}"
            )
        if not bool(torch.isfinite(tokens).all()):
            raise ValueError(
                f"description cache view token 非 finite: {row['lookup_key']}"
            )
        valid = view.get("valid_mask")
        if (
            not torch.is_tensor(valid)
            or tuple(valid.shape) != (1, expected_sizes[0], expected_sizes[0])
        ):
            raise ValueError(
                f"description cache valid mask shape 非法: {row['lookup_key']}"
            )
        if not bool(torch.isfinite(valid).all()):
            raise ValueError(
                f"description cache valid mask 非 finite: {row['lookup_key']}"
            )
    payload = "|".join([
        DESCRIPTION_CACHE_PROTOCOL,
        str(row["lookup_key"]),
        str(row["source_content_hash"]),
        str(manifest["model_revision"]),
        str(manifest["processor_revision"]),
        *sorted(str(view.get("content_hash") or "") for view in views),
    ])
    if hashlib.sha256(payload.encode()).hexdigest() != row["cache_fingerprint"]:
        raise ValueError(
            f"description cache fingerprint 不一致: {row['lookup_key']}"
        )
    if row.get("source_cache") and row.get("component") != "multisource_parent":
        raise ValueError(
            f"single_image record 禁止复用 segmentation cache: {row['lookup_key']}"
        )
