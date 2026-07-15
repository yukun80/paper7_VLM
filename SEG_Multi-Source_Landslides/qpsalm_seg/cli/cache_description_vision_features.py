#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build task-neutral Qwen vision cache v1 for segmentation-grounded description.

用途：按 parent 缓存 single-image 与 multisource Bridge 的视觉特征，不缓存 instruction、region 或分割状态。
推荐运行命令：PYTHONPATH=SEG_Multi-Source_Landslides python -m qpsalm_seg.cli.cache_description_vision_features --config SEG_Multi-Source_Landslides/configs/qpsalm_v2_small.yaml --description-benchmark benchmark/qpsalm_description_v2_small --bridge-benchmark benchmark/landslide_region_description_v1_small --output-dir outputs/qpsalm_description/cache/vision_v1 --device cuda --backend qwen --overwrite
主要输入：Description M1、Bridge M2、可选 segmentation vision cache v3 和本地 Qwen 权重。
主要输出：独立的 qpsalm_description_vision_cache_v1 manifest 与 shards。
写入行为：只写 --output-dir；--overwrite 只清理该 cache，不修改 benchmark 或 segmentation cache。
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Iterator

import torch
import torch.nn.functional as F
from tqdm import tqdm

from qpsalm_seg.config import load_config
from qpsalm_seg.data.io import build_modality_instance, positive_float
from qpsalm_seg.data.single_image import build_single_image_modality_instance
from qpsalm_seg.description.vision_cache import (
    DESCRIPTION_CACHE_FORMAT,
    DESCRIPTION_CACHE_PROTOCOL,
    DescriptionVisionFeatureBank,
    description_cache_key,
)
from qpsalm_seg.models.qwen_vision_encoder import HashVisionEncoder, QwenVisionEncoder
from qpsalm_seg.models.vision_cache import QwenVisionFeatureBank
from qpsalm_seg.paths import resolve_project_path
from qpsalm_seg.rendering import RENDERER_VERSION, render_sensor_views


BUILDER_VERSION = "description_vision_cache_m3_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build task-neutral description vision cache v1")
    parser.add_argument("--config", required=True)
    parser.add_argument("--description-benchmark", required=True)
    parser.add_argument("--bridge-benchmark")
    parser.add_argument("--segmentation-vision-cache")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backend", choices=["qwen", "hash-smoke"], default="qwen")
    parser.add_argument("--components", default="single_image,multisource_parent")
    parser.add_argument("--render-size", type=int, default=256)
    parser.add_argument("--layers", default="5,11,17,23")
    parser.add_argument("--spatial-sizes", default="16,8,6,4")
    parser.add_argument("--view-tokens", type=int, default=8)
    parser.add_argument("--shard-size", type=int, default=256)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _multisource_content_hash(items: dict[str, dict[str, Any]]) -> str:
    """Hash rendered-input content and metadata, never only logical paths."""
    payload = []
    for name, item in sorted(items.items()):
        if not item.get("available", True) or not item.get("path"):
            continue
        value_path = resolve_project_path(item["path"])
        if value_path is None or not value_path.is_file():
            raise FileNotFoundError(f"多源模态不存在: {name} -> {item.get('path')}")
        valid_spec = item.get("valid_mask") or {}
        valid_path = resolve_project_path(valid_spec.get("path")) if valid_spec.get("path") else None
        if valid_path is None or not valid_path.is_file():
            raise FileNotFoundError(f"多源 valid mask 不存在: {name} -> {valid_spec.get('path')}")
        payload.append({
            "name": str(name),
            "value_sha256": _sha256_file(value_path),
            "valid_sha256": _sha256_file(valid_path),
            "family": item.get("family"),
            "sensor": item.get("sensor"),
            "product_type": item.get("product_type"),
            "band_names": item.get("band_names") or [],
            "band_metadata": item.get("band_metadata") or [],
            "native_gsd_m": item.get("native_gsd_m"),
            "units": item.get("units"),
            "signed": bool(item.get("signed")),
            "orbit": item.get("orbit"),
            "quality": item.get("quality"),
            "normalization": item.get("normalization") or {},
        })
    if not payload:
        raise ValueError("多源 parent 没有可哈希的活动模态")
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: 非法 JSONL") from exc
    return rows


def _single_image_records(benchmark: Path, render_size: int) -> Iterator[dict[str, Any]]:
    rows = _read_jsonl(benchmark / "indexes/all.jsonl")
    seen: set[str] = set()
    for row in rows:
        parent_id = str(row["parent_sample_id"])
        if parent_id in seen:
            continue
        seen.add(parent_id)
        visual = row["visual_ref"]
        item = build_single_image_modality_instance(visual)
        yield {
            "lookup_key": description_cache_key("single_image", parent_id),
            "component": "single_image",
            "parent_sample_id": parent_id,
            "source_ref": str(visual["path"]),
            "source_content_hash": str(visual["sha256"]),
            "views": render_sensor_views([item], render_size, strict=True),
            "modality_families": {item.name: item.family},
            "source_cache": None,
        }


def _bridge_records(
    benchmark: Path,
    render_size: int,
    segmentation_bank: QwenVisionFeatureBank | None,
) -> Iterator[dict[str, Any]]:
    rows = _read_jsonl(benchmark / "indexes/candidate_all.jsonl")
    seen: set[str] = set()
    for row in rows:
        parent_id = str(row["parent_sample_id"])
        if parent_id in seen:
            continue
        seen.add(parent_id)
        qmv3_key = f"qmv3-parent:{parent_id}"
        if segmentation_bank is not None and qmv3_key in segmentation_bank.lookup:
            cached = segmentation_bank.task_neutral_record(qmv3_key)
            yield {
                "lookup_key": description_cache_key("multisource_parent", parent_id),
                "component": "multisource_parent",
                "parent_sample_id": parent_id,
                "source_ref": str(row["source_parent_index"]),
                "source_content_hash": str(cached["cache_fingerprint"]),
                "views": None,
                "preencoded_views": cached["views"],
                "modality_families": {
                    str(name): str(family)
                    for name, family in (
                        (segmentation_bank.lookup[qmv3_key].get("modality_families") or {}).items()
                    )
                },
                "source_cache": qmv3_key,
            }
            continue
        modality_metadata = row.get("modality_metadata", {})
        instances = []
        for name, item in sorted(modality_metadata.items()):
            if item.get("available", True) and item.get("path"):
                aligned = positive_float(item.get("native_gsd_m"))
                instances.append(build_modality_instance(str(name), item, aligned))
        if not instances:
            raise ValueError(f"Bridge parent 没有可编码模态: {parent_id}")
        yield {
            "lookup_key": description_cache_key("multisource_parent", parent_id),
            "component": "multisource_parent",
            "parent_sample_id": parent_id,
            "source_ref": str(row["source_parent_index"]),
            "source_content_hash": _multisource_content_hash(modality_metadata),
            "views": render_sensor_views(instances, render_size, strict=True),
            "modality_families": {item.name: item.family for item in instances},
            "source_cache": None,
        }


def _serialize_encoded(
    record: dict[str, Any],
    encoded_views: list[dict[str, Any]],
    spatial_sizes: tuple[int, ...],
    view_tokens: int,
    revision: str,
    processor_revision: str,
) -> dict[str, Any]:
    views = []
    if len(record["views"]) != len(encoded_views):
        raise ValueError(f"description cache view encoding 数量不一致: {record['lookup_key']}")
    for view, encoded in zip(record["views"], encoded_views):
        tokens = encoded["tokens"]
        if tokens.shape[0] > view_tokens:
            tokens = F.adaptive_avg_pool1d(tokens.float().T[None], view_tokens)[0].T.half()
        views.append({
            "name": view.name,
            "description": view.description,
            "source_modalities": list(view.source_modalities),
            "source_families": [],
            "quality_flags": list(view.quality_flags),
            "content_hash": view.content_hash,
            "render_transform": view.render_transform,
            "vision_grid_thw": list(encoded["vision_grid_thw"]),
            "merged_grid_hw": list(encoded["merged_grid_hw"]),
            "valid_mask": F.adaptive_avg_pool2d(view.valid_mask.float()[None], spatial_sizes[0])[0].half(),
            "spatial_features": encoded["spatial"],
            "view_tokens": tokens,
        })
    for source, target in zip(record["views"], views):
        target["source_families"] = sorted({
            str(record["modality_families"].get(name) or "unknown")
            for name in source.source_modalities
        })
    return _finalize_record(record, views, revision, processor_revision)


def _serialize_preencoded(
    record: dict[str, Any], revision: str, processor_revision: str,
) -> dict[str, Any]:
    views = []
    for source in record["preencoded_views"]:
        views.append({
            key: copy.deepcopy(source[key])
            for key in (
                "name", "description", "source_modalities", "source_families", "quality_flags",
                "content_hash", "render_transform", "vision_grid_thw", "merged_grid_hw",
                "valid_mask", "spatial_features", "view_tokens",
            )
        })
    return _finalize_record(record, views, revision, processor_revision)


def _finalize_record(
    record: dict[str, Any], views: list[dict[str, Any]], revision: str, processor_revision: str,
) -> dict[str, Any]:
    payload = "|".join([
        DESCRIPTION_CACHE_PROTOCOL,
        record["lookup_key"],
        record["source_content_hash"],
        revision,
        processor_revision,
        *sorted(str(view["content_hash"]) for view in views),
    ])
    return {
        "lookup_key": record["lookup_key"],
        "component": record["component"],
        "parent_sample_id": record["parent_sample_id"],
        "source_ref": record["source_ref"],
        "source_content_hash": record["source_content_hash"],
        "source_cache": record.get("source_cache"),
        "cache_fingerprint": hashlib.sha256(payload.encode()).hexdigest(),
        "views": views,
    }


def _write_shard(output: Path, index: int, rows: list[dict[str, Any]], lookup: dict[str, Any]) -> str:
    for local_index, row in enumerate(rows):
        lookup[row["lookup_key"]] = {
            "shard": index,
            "index": local_index,
            "component": row["component"],
            "parent_sample_id": row["parent_sample_id"],
        }
    path = output / f"shard_{index:05d}.pt"
    temporary = output / f".{path.name}.tmp"
    try:
        torch.save({"format": DESCRIPTION_CACHE_FORMAT, "records": rows}, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path.name


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output = resolve_project_path(args.output_dir) or Path(args.output_dir)
    if args.verify_only:
        bank = DescriptionVisionFeatureBank(output)
        print(json.dumps({
            "output_dir": str(output), "format": bank.manifest["format"],
            "num_samples": bank.manifest["num_samples"], "status": "valid",
        }, ensure_ascii=False))
        return
    components = tuple(value.strip() for value in args.components.split(",") if value.strip())
    if not components or set(components) - {"single_image", "multisource_parent"}:
        raise ValueError("--components 只支持 single_image,multisource_parent")
    description_dir = resolve_project_path(args.description_benchmark)
    bridge_dir = resolve_project_path(args.bridge_benchmark) if args.bridge_benchmark else None
    if "single_image" in components and (description_dir is None or not description_dir.is_dir()):
        raise FileNotFoundError(f"Description benchmark 不存在: {args.description_benchmark}")
    if "multisource_parent" in components and (bridge_dir is None or not bridge_dir.is_dir()):
        raise FileNotFoundError("multisource_parent cache 需要 --bridge-benchmark")
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"cache exists: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    layers = tuple(int(value) for value in args.layers.split(","))
    spatial_sizes = tuple(int(value) for value in args.spatial_sizes.split(","))
    if layers != (5, 11, 17, 23) or len(spatial_sizes) != 4:
        raise ValueError("description cache v1 固定使用 layers=5,11,17,23 和四个 spatial sizes")
    if args.shard_size <= 0 or args.view_tokens <= 0:
        raise ValueError("--shard-size/--view-tokens 必须为正整数")
    segmentation_bank = (
        QwenVisionFeatureBank(args.segmentation_vision_cache, decoder_dim=config.decoder_dim)
        if args.segmentation_vision_cache else None
    )
    encoder = (
        QwenVisionEncoder(config.qwen_model_path, torch.device(args.device), layers, spatial_sizes)
        if args.backend == "qwen" else
        HashVisionEncoder(layers, spatial_sizes, args.view_tokens)
    )
    if segmentation_bank is not None:
        expected_backend = "hash-smoke" if args.backend == "hash-smoke" else "qwen"
        if segmentation_bank.manifest.get("backend") != expected_backend:
            raise ValueError("segmentation cache backend 与 description cache backend 不一致")
        if list(segmentation_bank.manifest.get("layers") or []) != list(layers):
            raise ValueError("segmentation cache layers 与 description cache 不一致")
        if list(segmentation_bank.manifest.get("spatial_sizes") or []) != list(spatial_sizes):
            raise ValueError("segmentation cache spatial sizes 与 description cache 不一致")
        if int(segmentation_bank.manifest.get("view_tokens_per_view") or 0) != int(args.view_tokens):
            raise ValueError("segmentation cache view token 数与 description cache 不一致")
        if int(segmentation_bank.manifest.get("render_size") or 0) != int(args.render_size):
            raise ValueError("segmentation cache render size 与 description cache 不一致")
        if segmentation_bank.manifest.get("renderer_version") != RENDERER_VERSION:
            raise ValueError("segmentation cache renderer version 与当前 renderer 不一致")
        if segmentation_bank.manifest.get("model_revision") != encoder.revision:
            raise ValueError("segmentation cache model revision 与 description encoder 不一致")
        if segmentation_bank.manifest.get("processor_revision") != encoder.processor_revision:
            raise ValueError("segmentation cache processor revision 与 description encoder 不一致")
    iterators: list[Iterator[dict[str, Any]]] = []
    input_fingerprints: dict[str, Any] = {}
    if "single_image" in components:
        index = description_dir / "indexes/all.jsonl"
        input_fingerprints["single_image"] = {"path": str(args.description_benchmark), "sha256": _sha256_file(index)}
        iterators.append(_single_image_records(description_dir, args.render_size))
    if "multisource_parent" in components:
        index = bridge_dir / "indexes/candidate_all.jsonl"
        input_fingerprints["multisource_parent"] = {"path": str(args.bridge_benchmark), "sha256": _sha256_file(index)}
        iterators.append(_bridge_records(bridge_dir, args.render_size, segmentation_bank))

    lookup: dict[str, Any] = {}
    shards: list[str] = []
    buffer: list[dict[str, Any]] = []
    count = 0
    try:
        for iterator in iterators:
            for record in tqdm(iterator, desc="description-vision-v1", unit="parent"):
                if args.max_samples > 0 and count >= args.max_samples:
                    break
                serialized = (
                    _serialize_preencoded(record, encoder.revision, encoder.processor_revision)
                    if record.get("preencoded_views") is not None else
                    _serialize_encoded(
                        record, encoder.encode(record), spatial_sizes, args.view_tokens,
                        encoder.revision, encoder.processor_revision,
                    )
                )
                if serialized["lookup_key"] in lookup or any(
                    row["lookup_key"] == serialized["lookup_key"] for row in buffer
                ):
                    raise ValueError(f"description cache key 重复: {serialized['lookup_key']}")
                buffer.append(serialized)
                count += 1
                if len(buffer) >= args.shard_size:
                    shards.append(_write_shard(output, len(shards), buffer, lookup))
                    buffer = []
            if args.max_samples > 0 and count >= args.max_samples:
                break
        if buffer:
            shards.append(_write_shard(output, len(shards), buffer, lookup))
    finally:
        encoder.close()

    manifest = {
        "format": DESCRIPTION_CACHE_FORMAT,
        "protocol": DESCRIPTION_CACHE_PROTOCOL,
        "builder_version": BUILDER_VERSION,
        "renderer_version": RENDERER_VERSION,
        "model_revision": encoder.revision,
        "processor_revision": encoder.processor_revision,
        "layers": list(layers),
        "spatial_sizes": list(spatial_sizes),
        "render_size": args.render_size,
        "view_tokens_per_view": args.view_tokens,
        "spatial_channels": encoder.spatial_channels,
        "token_dim": encoder.token_dim,
        "backend": args.backend,
        "input_fingerprints": input_fingerprints,
        "num_samples": count,
        "components": list(components),
        "lookup": lookup,
        "shards": shards,
        "shard_size": args.shard_size,
        "forbidden_state": ["instruction", "condition", "region_geometry", "segmentation_state"],
    }
    temporary = output / ".manifest.json.tmp"
    try:
        temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(output / "manifest.json")
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps({
        "output_dir": str(output), "format": DESCRIPTION_CACHE_FORMAT,
        "num_samples": count, "num_shards": len(shards), "components": list(components),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
