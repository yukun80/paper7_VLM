#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task-neutral vision cache v1 shared by global and region description."""

from __future__ import annotations

from collections import OrderedDict
import hashlib
from pathlib import Path
from typing import Any
import json

import torch

from qpsalm_seg.paths import resolve_project_path


DESCRIPTION_CACHE_FORMAT = "qpsalm_description_vision_cache_v1"
DESCRIPTION_CACHE_PROTOCOL = "task_neutral_parent_visual_features_v1"


def description_cache_key(component: str, parent_sample_id: str) -> str:
    if component not in {"single_image", "multisource_parent"}:
        raise ValueError(f"未知 description cache component={component!r}")
    return f"qdcv1:{component}:{parent_sample_id}"


class DescriptionVisionFeatureBank:
    """Strict sharded reader that never stores instruction or region state."""

    def __init__(self, cache_dir: str | Path, max_open_shards: int = 8) -> None:
        path = resolve_project_path(cache_dir) or Path(cache_dir)
        manifest_path = path / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"description vision cache manifest 不存在: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self._validate_manifest(manifest, manifest_path)
        self.cache_dir = path
        self.manifest = manifest
        self.lookup = manifest["lookup"]
        self.shards = tuple(str(value) for value in manifest["shards"])
        missing = [name for name in self.shards if not (path / name).is_file()]
        if missing:
            raise FileNotFoundError(f"description vision cache 缺少 shards: {missing[:8]}")
        self.max_open_shards = max(1, int(max_open_shards))
        self._loaded: OrderedDict[int, list[dict[str, Any]]] = OrderedDict()

    @staticmethod
    def _validate_manifest(manifest: dict[str, Any], path: Path) -> None:
        required = {
            "format", "protocol", "builder_version", "model_revision", "processor_revision",
            "layers", "spatial_sizes", "view_tokens_per_view", "spatial_channels", "token_dim",
            "backend", "input_fingerprints", "num_samples", "components", "lookup", "shards",
            "shard_size",
        }
        missing = sorted(required - set(manifest))
        if missing:
            raise ValueError(f"description vision cache manifest 缺少 {missing}: {path}")
        if manifest["format"] != DESCRIPTION_CACHE_FORMAT:
            raise ValueError(f"只支持 {DESCRIPTION_CACHE_FORMAT}: {path}")
        if manifest["protocol"] != DESCRIPTION_CACHE_PROTOCOL:
            raise ValueError(f"description cache protocol 不匹配: {manifest['protocol']!r}")
        if list(manifest["layers"]) != [5, 11, 17, 23]:
            raise ValueError("description cache layers 必须为 [5,11,17,23]")
        if int(manifest["num_samples"]) != len(manifest["lookup"]):
            raise ValueError("description cache lookup/sample 数量不一致")
        if set(manifest["components"]) - {"single_image", "multisource_parent"}:
            raise ValueError(f"description cache component 非法: {manifest['components']}")
        for key, location in manifest["lookup"].items():
            if not str(key).startswith("qdcv1:"):
                raise ValueError(f"description cache key 非法: {key}")
            if not {"shard", "index", "component", "parent_sample_id"} <= set(location):
                raise ValueError(f"description cache lookup 不完整: {key}")

    def _load_shard(self, index: int) -> list[dict[str, Any]]:
        if index in self._loaded:
            rows = self._loaded.pop(index)
            self._loaded[index] = rows
            return rows
        path = self.cache_dir / self.shards[index]
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("format") != DESCRIPTION_CACHE_FORMAT:
            raise ValueError(f"description cache shard 损坏: {path}")
        rows = payload.get("records")
        if not isinstance(rows, list):
            raise ValueError(f"description cache shard records 非法: {path}")
        self._loaded[index] = rows
        while len(self._loaded) > self.max_open_shards:
            self._loaded.popitem(last=False)
        return rows

    def record(self, component: str, parent_sample_id: str) -> dict[str, Any]:
        key = description_cache_key(component, parent_sample_id)
        location = self.lookup.get(key)
        if location is None:
            raise KeyError(f"description vision cache 缺少 parent: {key}")
        row = self._load_shard(int(location["shard"]))[int(location["index"])]
        if row.get("lookup_key") != key:
            raise ValueError(f"description cache lookup/shard key 不一致: {key}")
        self._validate_record(row)
        return row

    def has(self, component: str, parent_sample_id: str) -> bool:
        return description_cache_key(component, parent_sample_id) in self.lookup

    def _validate_record(self, row: dict[str, Any]) -> None:
        forbidden = {"instruction", "condition", "region_geometry", "segmentation_state"}
        if forbidden & set(row):
            raise ValueError(f"description cache 包含任务相关字段: {sorted(forbidden & set(row))}")
        required = {
            "lookup_key", "component", "parent_sample_id", "source_ref", "source_content_hash",
            "cache_fingerprint", "views",
        }
        if required - set(row):
            raise ValueError(f"description cache record 字段不完整: {row.get('lookup_key')}")
        views = row["views"]
        if not isinstance(views, list) or not views:
            raise ValueError(f"description cache record 没有 visual views: {row['lookup_key']}")
        expected_sizes = [int(value) for value in self.manifest["spatial_sizes"]]
        for view in views:
            spatial = view.get("spatial_features")
            if not isinstance(spatial, (list, tuple)) or len(spatial) != len(expected_sizes):
                raise ValueError(f"description cache spatial layers 非法: {row['lookup_key']}")
            for value, size in zip(spatial, expected_sizes):
                if not torch.is_tensor(value) or tuple(value.shape) != (
                    int(self.manifest["spatial_channels"]), size, size
                ):
                    raise ValueError(f"description cache spatial feature shape 非法: {row['lookup_key']}")
            tokens = view.get("view_tokens")
            if (
                not torch.is_tensor(tokens)
                or tokens.ndim != 2
                or int(tokens.shape[1]) != int(self.manifest["token_dim"])
                or int(tokens.shape[0]) > int(self.manifest["view_tokens_per_view"])
            ):
                raise ValueError(f"description cache view token shape 非法: {row['lookup_key']}")
            valid = view.get("valid_mask")
            if not torch.is_tensor(valid) or tuple(valid.shape) != (1, expected_sizes[0], expected_sizes[0]):
                raise ValueError(f"description cache valid mask shape 非法: {row['lookup_key']}")
        payload = "|".join([
            DESCRIPTION_CACHE_PROTOCOL,
            str(row["lookup_key"]),
            str(row["source_content_hash"]),
            str(self.manifest["model_revision"]),
            str(self.manifest["processor_revision"]),
            *sorted(str(view.get("content_hash") or "") for view in views),
        ])
        if hashlib.sha256(payload.encode()).hexdigest() != row["cache_fingerprint"]:
            raise ValueError(f"description cache fingerprint 不一致: {row['lookup_key']}")
