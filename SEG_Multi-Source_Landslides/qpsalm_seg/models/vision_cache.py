#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Strict reader for parent-level Qwen vision feature cache v3."""

from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn
import torch.nn.functional as F

from qpsalm_seg.data.prompts import PROMPT_VERSION
from qpsalm_seg.paths import resolve_project_path
from qpsalm_seg.rendering import RENDERER_VERSION
from qpsalm_seg.schema import MODALITY_FAMILY_IDS, ActiveModalitySubset, ModalityInstance


CACHE_FORMAT = "qpsalm_qwen_vision_cache_v3"
FAMILY_IDS = MODALITY_FAMILY_IDS


def view_fingerprint_fragment(view: dict[str, Any]) -> str:
    metadata = {
        "description": str(view.get("description") or ""),
        "render_transform": view.get("render_transform") or {},
        "vision_grid_thw": list(view.get("vision_grid_thw") or []),
        "merged_grid_hw": list(view.get("merged_grid_hw") or []),
    }
    metadata_hash = hashlib.sha256(
        json.dumps(metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return f"{view.get('content_hash') or ''}:{metadata_hash}"


def vision_input_protocol(config: Any) -> dict[str, Any]:
    use_buckets = bool(config.use_size_buckets)
    index_fingerprints = {}
    for split in ("train", "val", "test"):
        reference = str(config.index_path(split))
        path = resolve_project_path(reference)
        if path is None or not path.is_file():
            index_fingerprints[split] = {
                "reference": reference, "status": "missing", "size": None, "sha256": None,
            }
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        index_fingerprints[split] = {
            "reference": reference,
            "status": "present",
            "size": path.stat().st_size,
            "sha256": digest.hexdigest(),
        }
    return {
        "preset": str(config.preset),
        "use_size_buckets": use_buckets,
        "size_buckets": list(config.size_buckets) if use_buckets else [],
        "target_size": int(config.target_size),
        "max_native_size": int(config.max_native_size),
        "index_fingerprints": index_fingerprints,
    }


class QwenVisionFeatureBank(nn.Module):
    """Load spatial maps and view tokens while enforcing the active subset."""

    def __init__(self, cache_dir: str | Path, decoder_dim: int, max_open_shards: int = 2, visual_ablation: str = "normal") -> None:
        super().__init__()
        path = resolve_project_path(cache_dir) or Path(cache_dir)
        manifest_path = path / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Qwen vision cache v3 manifest 不存在: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self._validate_manifest(manifest, manifest_path)
        self.cache_dir = path
        self.manifest = manifest
        self.lookup = manifest["lookup"]
        self.shards = tuple(str(value) for value in manifest["shards"])
        missing_shards = [name for name in self.shards if not (path / name).is_file()]
        if missing_shards:
            raise FileNotFoundError(f"Qwen vision cache v3 缺少 shards: {missing_shards[:8]}")
        self.max_open_shards = max(1, int(max_open_shards))
        self.set_visual_ablation(visual_ablation)
        self._loaded: OrderedDict[int, list[dict[str, Any]]] = OrderedDict()
        self.spatial_channels = int(manifest.get("spatial_channels", 1024))
        self.token_dim = int(manifest.get("token_dim", 2048))
        del decoder_dim

    def set_visual_ablation(self, visual_ablation: str) -> None:
        """Validate and switch token-only evidence ablation without touching dense SANE maps."""
        supported = {"normal", "shuffled", "text-only", "image-text-delta"}
        if visual_ablation not in supported and not visual_ablation.startswith("remove:"):
            raise ValueError(f"未知 visual_ablation={visual_ablation!r}")
        if visual_ablation.startswith("remove:"):
            removed_family = visual_ablation.split(":", 1)[1]
            if removed_family not in FAMILY_IDS or removed_family == "unknown":
                raise ValueError(f"未知 remove family={removed_family!r}")
        self.visual_ablation = visual_ablation

    @staticmethod
    def _validate_manifest(manifest: dict[str, Any], path: Path) -> None:
        if manifest.get("format") != CACHE_FORMAT:
            raise ValueError(f"只支持 {CACHE_FORMAT}: {path}")
        for key in (
            "renderer_version", "model_revision", "processor_revision", "prompt_version",
            "pooling_method", "layers", "spatial_sizes", "render_size", "view_tokens_per_view",
            "lookup", "shards", "input_protocol", "num_samples", "shard_size",
            "peak_buffer_records",
        ):
            if key not in manifest:
                raise ValueError(f"Qwen vision cache v3 manifest 缺少 {key}: {path}")
        if list(manifest.get("layers") or []) != [5, 11, 17, 23]:
            raise ValueError(f"Qwen vision cache v3 layers 必须为 [5,11,17,23]: {manifest.get('layers')}")
        if manifest.get("subset_policy") != "dynamic_by_source_modality":
            raise ValueError(f"cache subset policy 不安全: {manifest.get('subset_policy')!r}")
        if manifest.get("renderer_version") != RENDERER_VERSION:
            raise ValueError(
                f"renderer version 不匹配: cache={manifest.get('renderer_version')} code={RENDERER_VERSION}"
            )
        if manifest.get("prompt_version") != PROMPT_VERSION:
            raise ValueError(
                f"prompt version 不匹配: cache={manifest.get('prompt_version')} code={PROMPT_VERSION}"
            )
        if manifest.get("pooling_method") != "spatial_layers_plus_adaptive_view_tokens":
            raise ValueError(f"不支持的 vision cache pooling_method={manifest.get('pooling_method')!r}")
        if manifest.get("backend") != "hash-smoke":
            for revision_name in ("model_revision", "processor_revision"):
                revision = manifest.get(revision_name)
                if not isinstance(revision, str) or len(revision) != 64:
                    raise ValueError(f"Qwen vision cache {revision_name} 不是完整 SHA-256")
        input_protocol = manifest.get("input_protocol")
        required_protocol = {
            "preset", "use_size_buckets", "size_buckets", "target_size", "max_native_size",
            "index_fingerprints",
        }
        if not isinstance(input_protocol, dict) or required_protocol - set(input_protocol):
            raise ValueError(f"Qwen vision cache input_protocol 不完整: {input_protocol}")
        index_fingerprints = input_protocol.get("index_fingerprints")
        if not isinstance(index_fingerprints, dict) or set(index_fingerprints) != {"train", "val", "test"}:
            raise ValueError(f"Qwen vision cache index_fingerprints 不完整: {index_fingerprints}")
        for split, fingerprint in index_fingerprints.items():
            required_fingerprint = {"reference", "status", "size", "sha256"}
            if not isinstance(fingerprint, dict) or required_fingerprint - set(fingerprint):
                raise ValueError(f"Qwen vision cache {split} index fingerprint 非法: {fingerprint}")
            digest = fingerprint.get("sha256")
            if (
                fingerprint.get("status") != "present"
                or not isinstance(fingerprint.get("size"), int)
                or int(fingerprint["size"]) <= 0
                or not isinstance(digest, str)
                or len(digest) != 64
            ):
                raise ValueError(f"Qwen vision cache {split} index 不完整或缺失: {fingerprint}")
        spatial_sizes = list(manifest.get("spatial_sizes") or [])
        if len(spatial_sizes) != len(manifest["layers"]) or any(int(value) <= 0 for value in spatial_sizes):
            raise ValueError(f"Qwen vision cache spatial_sizes 非法: {spatial_sizes}")
        if any(int(left) < int(right) for left, right in zip(spatial_sizes, spatial_sizes[1:])):
            raise ValueError(f"Qwen vision cache spatial_sizes 必须非递增: {spatial_sizes}")
        if manifest.get("backend") != "hash-smoke" and int(spatial_sizes[0]) < 12:
            raise ValueError("正式 Qwen vision cache 浅层 spatial size 必须至少为 12")
        if manifest.get("backend") != "hash-smoke" and int(manifest.get("render_size") or 0) < 256:
            raise ValueError("正式 Qwen vision cache render_size 必须至少为 256")
        if int(manifest.get("view_tokens_per_view") or 0) <= 0:
            raise ValueError("Qwen vision cache view_tokens_per_view 必须为正整数")
        num_samples = int(manifest.get("num_samples") or 0)
        shard_size = int(manifest.get("shard_size") or 0)
        peak_buffer = int(manifest.get("peak_buffer_records") or 0)
        lookup = manifest.get("lookup") or {}
        shards = manifest.get("shards") or []
        if num_samples <= 0 or num_samples != len(lookup):
            raise ValueError(
                f"Qwen vision cache sample/lookup 数量不一致: samples={num_samples} lookup={len(lookup)}"
            )
        if shard_size <= 0 or not 0 < peak_buffer <= shard_size:
            raise ValueError(
                f"Qwen vision cache 流式分片协议非法: shard_size={shard_size} peak={peak_buffer}"
            )
        expected_shards = (num_samples + shard_size - 1) // shard_size
        if len(shards) != expected_shards:
            raise ValueError(
                f"Qwen vision cache shard 数量不一致: expected={expected_shards} actual={len(shards)}"
            )
        expected_positions = {
            (shard_index, local_index)
            for shard_index in range(expected_shards)
            for local_index in range(
                shard_size if shard_index < expected_shards - 1 else num_samples - shard_size * shard_index
            )
        }
        actual_positions = {
            (int(location.get("shard", -1)), int(location.get("index", -1)))
            for location in lookup.values()
            if isinstance(location, dict)
        }
        if actual_positions != expected_positions:
            raise ValueError("Qwen vision cache lookup 未完整覆盖流式 shard 位置")
        for key, location in lookup.items():
            required = {"shard", "index", "source_modalities", "source_families", "modality_families"}
            if not isinstance(location, dict) or required - set(location):
                raise ValueError(f"vision cache lookup 元数据不完整: key={key}")

    def _expected_fingerprint(self, record: dict[str, Any]) -> str:
        payload = "|".join(
            [
                str(self.manifest["renderer_version"]),
                str(self.manifest["model_revision"]),
                str(self.manifest["processor_revision"]),
                str(self.manifest["prompt_version"]),
                str(self.manifest["pooling_method"]),
                str(record.get("full_subset_signature") or ""),
            ]
            + sorted(view_fingerprint_fragment(view) for view in record.get("views") or [])
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def _load_shard(self, index: int) -> list[dict[str, Any]]:
        if index in self._loaded:
            records = self._loaded.pop(index)
            self._loaded[index] = records
            return records
        path = self.cache_dir / self.shards[index]
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("format") != CACHE_FORMAT or not isinstance(payload.get("records"), list):
            raise ValueError(f"损坏的 Qwen vision cache shard: {path}")
        records = payload["records"]
        self._loaded[index] = records
        while len(self._loaded) > self.max_open_shards:
            self._loaded.popitem(last=False)
        return records

    def _shuffled_key(self, key: str) -> str:
        keys = sorted(self.lookup)
        current = self.lookup.get(key)
        if current is None:
            raise KeyError(f"Qwen vision cache v3 缺少 parent key: {key}")
        start = keys.index(key)
        ordered = keys[start + 1:] + keys[:start]
        for field in ("source_modalities", "source_families"):
            expected = tuple(current.get(field) or [])
            match = next(
                (
                    candidate
                    for candidate in ordered
                    if tuple(self.lookup[candidate].get(field) or []) == expected
                ),
                None,
            )
            if match is not None:
                return match
        raise RuntimeError(
            f"visual shuffle 缺少同模态语义的其他 parent: key={key}; "
            "请使用覆盖每种组合至少两个 parent 的 cache"
        )

    def _record(self, key: str, *, apply_token_ablation: bool = False) -> dict[str, Any]:
        if apply_token_ablation and self.visual_ablation == "shuffled":
            key = self._shuffled_key(key)
        location = self.lookup.get(key)
        if location is None:
            raise KeyError(f"Qwen vision cache v3 缺少 parent key: {key}")
        record = self._load_shard(int(location["shard"]))[int(location["index"])]
        available = sorted({
            str(name)
            for view in record.get("views") or []
            for name in view.get("source_modalities") or []
        })
        expected_subset = "subset:" + "+".join(available) + ":" + hashlib.sha256(
            "\n".join(available).encode()
        ).hexdigest()[:12]
        if record.get("full_subset_signature") != expected_subset:
            raise ValueError(f"Qwen vision cache v3 full subset signature 不匹配: {key}")
        fingerprint = str(record.get("cache_fingerprint") or "")
        if fingerprint != self._expected_fingerprint(record):
            raise ValueError(f"Qwen vision cache v3 record fingerprint 不匹配: {key}")
        for view in record.get("views") or []:
            if (
                not view.get("content_hash")
                or not view.get("description")
                or not view.get("source_modalities")
                or not view.get("source_families")
            ):
                raise ValueError(f"Qwen vision cache v3 view 元数据不完整: {key}")
            grid = list(view.get("vision_grid_thw") or [])
            merged_grid = list(view.get("merged_grid_hw") or [])
            if len(grid) != 3 or len(merged_grid) != 2 or any(int(value) <= 0 for value in grid + merged_grid):
                raise ValueError(f"Qwen vision cache v3 view 网格非法: key={key} view={view.get('name')}")
            spatial = view.get("spatial_features")
            if not isinstance(spatial, (list, tuple, torch.Tensor)) or len(spatial) != len(self.manifest["layers"]):
                raise ValueError(f"Qwen vision cache v3 spatial feature 层数错误: key={key}")
            for layer_index, feature in enumerate(spatial):
                expected_size = int(self.manifest["spatial_sizes"][layer_index])
                if (
                    not torch.is_tensor(feature)
                    or feature.ndim != 3
                    or int(feature.shape[0]) != self.spatial_channels
                    or tuple(feature.shape[-2:]) != (expected_size, expected_size)
                ):
                    raise ValueError(
                        f"Qwen vision cache v3 spatial feature shape 错误: "
                        f"key={key} layer={layer_index} shape={getattr(feature, 'shape', None)}"
                    )
            tokens = view.get("view_tokens")
            if (
                not torch.is_tensor(tokens)
                or tokens.ndim != 2
                or int(tokens.shape[1]) != self.token_dim
                or int(tokens.shape[0]) > int(self.manifest["view_tokens_per_view"])
            ):
                raise ValueError(f"Qwen vision cache v3 view token shape 错误: key={key}")
            if int(merged_grid[0]) * int(merged_grid[1]) < int(tokens.shape[0]):
                raise ValueError(f"Qwen vision cache v3 merged grid 无法容纳 view tokens: key={key}")
            valid = view.get("valid_mask")
            shallow_size = int(self.manifest["spatial_sizes"][0])
            if not torch.is_tensor(valid) or tuple(valid.shape) != (1, shallow_size, shallow_size):
                raise ValueError(f"Qwen vision cache v3 valid mask shape 错误: key={key}")
        return record

    def _views(self, record: dict[str, Any], *, apply_token_ablation: bool = False) -> list[dict[str, Any]]:
        views = list(record.get("views") or [])
        if apply_token_ablation and self.visual_ablation == "text-only":
            return []
        if apply_token_ablation and self.visual_ablation.startswith("remove:"):
            removed = self.visual_ablation.split(":", 1)[1]
            views = [view for view in views if removed not in set(view.get("source_families") or [])]
        return views

    @staticmethod
    def _view_is_active(view: dict[str, Any], active_names: set[str]) -> bool:
        sources = {str(value) for value in view.get("source_modalities") or []}
        return bool(sources) and sources.issubset(active_names)

    def selected_views_for(
        self,
        key: str,
        subset: ActiveModalitySubset,
        *,
        apply_token_ablation: bool = True,
    ) -> list[dict[str, Any]]:
        """Return the exact cache views visible to one active student subset."""
        active = set(subset.active_names)
        original_meta = self.lookup.get(str(key)) or {}
        active_families = {
            str((original_meta.get("modality_families") or {}).get(name))
            for name in active
            if (original_meta.get("modality_families") or {}).get(name)
        }
        record = self._record(str(key), apply_token_ablation=apply_token_ablation)
        selected = []
        for view in self._views(record, apply_token_ablation=apply_token_ablation):
            visible = (
                set(view.get("source_families") or []).issubset(active_families)
                if apply_token_ablation and self.visual_ablation == "shuffled"
                else self._view_is_active(view, active)
            )
            if visible:
                selected.append(view)
        return selected

    @staticmethod
    def _apply_augment(value: torch.Tensor, augment: dict[str, Any]) -> torch.Tensor:
        if bool(augment.get("hflip")):
            value = value.flip(-1)
        if bool(augment.get("vflip")):
            value = value.flip(-2)
        return value

    @staticmethod
    def _remove_render_padding(value: torch.Tensor, transform: dict[str, Any]) -> torch.Tensor:
        size = float(transform.get("size") or 0)
        if size <= 0:
            raise ValueError("vision cache view 缺少有效 render_transform")
        h, w = value.shape[-2:]
        top = float(transform.get("pad_top") or 0)
        left = float(transform.get("pad_left") or 0)
        resized_h = float(transform.get("resized_h") or size)
        resized_w = float(transform.get("resized_w") or size)
        y0 = max(0, min(h - 1, int(round(top / size * h))))
        x0 = max(0, min(w - 1, int(round(left / size * w))))
        y1 = max(y0 + 1, min(h, int(round((top + resized_h) / size * h))))
        x1 = max(x0 + 1, min(w, int(round((left + resized_w) / size * w))))
        return value[..., y0:y1, x0:x1]

    def features_for(self, item: ModalityInstance, device: torch.device) -> list[torch.Tensor]:
        """Return one map per cached Qwen layer for a physical modality instance."""
        key = str(item.metadata.get("vision_cache_key") or "")
        if not key:
            raise KeyError(f"模态 {item.name} 缺少 vision_cache_key")
        # Evidence ablations must not alter SANE's dense pretrained features.
        # Otherwise a view-shuffle experiment changes both the VLM verifier and
        # the visual backbone, making the causal comparison uninterpretable.
        views = [
            view for view in self._views(self._record(key))
            if item.name in {str(value) for value in view.get("source_modalities") or []}
        ]
        if not views:
            raise KeyError(f"cache parent={key} 没有覆盖 modality={item.name}")
        layer_count = len(self.manifest["layers"])
        outputs: list[torch.Tensor] = []
        augment = item.metadata.get("train_augment") or {}
        for layer in range(layer_count):
            weighted = []
            weights = []
            for view in views:
                feature = view["spatial_features"][layer].float()
                feature = self._remove_render_padding(feature, view.get("render_transform") or {})
                feature = self._apply_augment(feature, augment)
                valid = view.get("valid_mask")
                coverage = float(valid.float().mean().item()) if torch.is_tensor(valid) else 1.0
                quality = 0.5 if view.get("quality_flags") else 1.0
                weighted.append(feature * max(coverage * quality, 1.0e-4))
                weights.append(max(coverage * quality, 1.0e-4))
            outputs.append((sum(weighted) / sum(weights)).to(device=device))
        return outputs

    def tokens_for(
        self,
        keys: Sequence[str],
        subsets: Sequence[ActiveModalitySubset],
        device: torch.device,
        max_tokens_per_view: int,
    ) -> tuple[torch.Tensor, torch.Tensor, list[int], torch.Tensor, list[list[tuple[str, int]]]]:
        """Select only active views, pool tokens, and pad a batch."""
        sequences: list[torch.Tensor] = []
        family_sequences: list[torch.Tensor] = []
        segment_sequences: list[list[tuple[str, int]]] = []
        counts: list[int] = []
        for key, subset in zip(keys, subsets):
            chunks = []
            family_chunks = []
            segments = []
            for view in self.selected_views_for(str(key), subset, apply_token_ablation=True):
                tokens = view["view_tokens"].float()
                if tokens.ndim == 1:
                    tokens = tokens[None]
                limit = max(1, int(max_tokens_per_view))
                if tokens.shape[0] > limit:
                    tokens = F.adaptive_avg_pool1d(tokens.T[None], limit)[0].T
                chunks.append(tokens)
                families = sorted({str(value) for value in view.get("source_families") or []})
                family_id = FAMILY_IDS.get(families[0], 0) if len(families) == 1 else 0
                family_chunks.append(torch.full((tokens.shape[0],), family_id, dtype=torch.long))
                segments.append((str(view["description"]), int(tokens.shape[0])))
            sequence = torch.cat(chunks, dim=0) if chunks else torch.zeros((0, self.token_dim))
            family_sequence = torch.cat(family_chunks) if family_chunks else torch.zeros((0,), dtype=torch.long)
            sequences.append(sequence)
            family_sequences.append(family_sequence)
            segment_sequences.append(segments)
            counts.append(int(sequence.shape[0]))
        max_length = max(max(counts, default=0), 1)
        padded = torch.zeros((len(sequences), max_length, self.token_dim), device=device)
        mask = torch.zeros((len(sequences), max_length), dtype=torch.bool, device=device)
        family_ids = torch.zeros((len(sequences), max_length), dtype=torch.long, device=device)
        for index, (sequence, family_sequence) in enumerate(zip(sequences, family_sequences)):
            if not sequence.numel():
                continue
            length = sequence.shape[0]
            padded[index, :length] = sequence.to(device=device)
            mask[index, :length] = True
            family_ids[index, :length] = family_sequence.to(device=device)
        return padded, mask, counts, family_ids, segment_sequences
