#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build task-neutral Qwen vision cache v1 for segmentation-grounded description.

用途：按 parent 缓存 single-image 与 multisource Bridge 的视觉特征，不缓存 instruction、region 或分割状态。
推荐运行命令：使用统一入口 ``qpsalm-segdesc cache build``；完整参数见根目录 README。
主要输入：Description M1、Bridge M2、可选 segmentation vision cache v3 和本地 Qwen 权重。
主要输出：独立的 qpsalm_description_vision_cache_v1 manifest、shards 与深度 validation_report。
写入行为：只写 --output-dir；--overwrite 只清理该 cache，不修改 benchmark 或 segmentation cache。
Reusable M3 data-plane implementation; command routing lives in workflows/cache_build.py.
"""

from __future__ import annotations

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
from qpsalm_seg.description.data.datasets import (
    DESCRIPTION_BUILDER_VERSION,
)
from qpsalm_seg.description.data.expert_contracts import BRIDGE_BUILDER_VERSION
from qpsalm_seg.description.protocols.io import (
    atomic_write_json,
    read_jsonl,
    strict_json_loads,
)
from qpsalm_seg.description.data.vision_cache import (
    DESCRIPTION_CACHE_BUILDER_VERSION,
    DESCRIPTION_CACHE_FORMAT,
    DESCRIPTION_CACHE_PROTOCOL,
    DESCRIPTION_CACHE_VALIDATION_PROTOCOL,
    DescriptionVisionFeatureBank,
    description_cache_key,
    sha256_file,
    source_cache_snapshot,
    validate_source_cache_snapshot,
)
from qpsalm_seg.models.qwen_vision_encoder import HashVisionEncoder, QwenVisionEncoder
from qpsalm_seg.models.vision_cache import QwenVisionFeatureBank
from qpsalm_seg.paths import (
    resolve_project_path,
    validate_output_replacement_safety,
)
from qpsalm_seg.rendering import RENDERER_VERSION, render_sensor_views


BUILDER_VERSION = DESCRIPTION_CACHE_BUILDER_VERSION


def _input_fingerprint(
    benchmark_ref: str,
    benchmark_dir: Path,
    index: Path,
    relative_index: str,
    *,
    expected_builder: str,
    component: str,
) -> dict[str, Any]:
    """Reject stale benchmark generations before expensive visual encoding."""

    report_relative = "reports/validation_report.json"
    report_path = benchmark_dir / report_relative
    if not index.is_file():
        raise FileNotFoundError(f"{component} 输入索引不存在: {index}")
    if not report_path.is_file():
        raise FileNotFoundError(
            f"{component} benchmark 缺少 validation report: {report_path}"
        )
    report = strict_json_loads(report_path.read_text(encoding="utf-8"))
    if report.get("builder_version") != expected_builder:
        raise RuntimeError(
            f"{component} benchmark builder 过期: "
            f"{report.get('builder_version')!r} != {expected_builder!r}"
        )
    if report.get("errors"):
        raise RuntimeError(f"{component} benchmark validation errors 非空")
    if component == "single_image":
        if int(
            report.get(
                "verified_perceptual_duplicate_cross_split_groups", -1
            )
        ) != 0:
            raise RuntimeError(
                "Description benchmark verified cross-split cluster 必须为零"
            )
        validation_status = "engineering-valid"
    else:
        validation_status = str(report.get("status") or "")
        if (
            validation_status not in {
                "awaiting_expert_review", "expert_pilot_frozen",
            }
            or report.get("pilot_protocol_complete") is not True
        ):
            raise RuntimeError(
                "Bridge benchmark 必须 engineering-valid、Pilot 完整，且状态为 "
                "awaiting_expert_review/expert_pilot_frozen"
            )
    return {
        "benchmark": str(benchmark_ref),
        "index": relative_index,
        "size": int(index.stat().st_size),
        "sha256": sha256_file(index),
        "validation_report": report_relative,
        "validation_report_size": int(report_path.stat().st_size),
        "validation_report_sha256": sha256_file(report_path),
        "validation_builder_version": expected_builder,
        "validation_status": validation_status,
    }


def build_input_fingerprints(
    components: tuple[str, ...],
    *,
    description_ref: str,
    description_dir: Path | None,
    bridge_ref: str | None,
    bridge_dir: Path | None,
) -> dict[str, dict[str, Any]]:
    fingerprints: dict[str, dict[str, Any]] = {}
    if "single_image" in components:
        if description_dir is None or not description_dir.is_dir():
            raise FileNotFoundError(f"Description benchmark 不存在: {description_ref}")
        relative = "indexes/all.jsonl"
        index = description_dir / relative
        fingerprints["single_image"] = _input_fingerprint(
            description_ref,
            description_dir,
            index,
            relative,
            expected_builder=DESCRIPTION_BUILDER_VERSION,
            component="single_image",
        )
    if "multisource_parent" in components:
        if bridge_dir is None or not bridge_dir.is_dir():
            raise FileNotFoundError("multisource_parent cache 需要 --bridge-benchmark")
        relative = "indexes/candidate_all.jsonl"
        index = bridge_dir / relative
        fingerprints["multisource_parent"] = _input_fingerprint(
            str(bridge_ref),
            bridge_dir,
            index,
            relative,
            expected_builder=BRIDGE_BUILDER_VERSION,
            component="multisource_parent",
        )
    return fingerprints


def source_record_fingerprint(bank: QwenVisionFeatureBank, key: str) -> str:
    return str(bank.task_neutral_record(key)["cache_fingerprint"])


def deep_validation_report(
    output: Path,
    *,
    input_fingerprints: dict[str, dict[str, Any]],
    source_bank: QwenVisionFeatureBank | None,
    source_cache_path: Path | None,
) -> dict[str, Any]:
    # 深度扫描只保留一个 shard，避免验证阶段额外占用多 GB 主存。
    # 构建阶段报告尚未发布；verify-only 会在进入本函数前先严格核验已发布报告。
    bank = DescriptionVisionFeatureBank(
        output,
        max_open_shards=1,
        require_validation_report=False,
    )
    report = bank.validate_all(
        expected_input_fingerprints=input_fingerprints,
        source_record_fingerprint=(
            (lambda key: source_record_fingerprint(source_bank, key))
            if source_bank is not None else None
        ),
    )
    provenance = bank.manifest["source_cache_provenance"]
    added_errors = 0
    if bool(provenance["provided"]):
        if source_cache_path is None:
            report["errors"].append("缺少用于深度验证的 segmentation cache v3 路径")
            added_errors += 1
            report["source_cache"]["isolation_unchanged"] = False
        else:
            snapshot_errors = validate_source_cache_snapshot(provenance, source_cache_path)
            report["errors"].extend(snapshot_errors)
            added_errors += len(snapshot_errors)
            if snapshot_errors:
                report["source_cache"]["isolation_unchanged"] = False
    report["num_errors"] = int(report.get("num_errors") or 0) + added_errors
    report["status"] = "valid" if not report["errors"] else "invalid"
    return report


def multisource_content_hash(items: dict[str, dict[str, Any]]) -> str:
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
            "value_sha256": sha256_file(value_path),
            "valid_sha256": sha256_file(valid_path),
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
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def single_image_records(benchmark: Path, render_size: int) -> Iterator[dict[str, Any]]:
    rows = read_jsonl(benchmark / "indexes/all.jsonl")
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


def bridge_records(
    benchmark: Path,
    render_size: int,
    segmentation_bank: QwenVisionFeatureBank | None,
) -> Iterator[dict[str, Any]]:
    rows = read_jsonl(benchmark / "indexes/candidate_all.jsonl")
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
            "source_content_hash": multisource_content_hash(modality_metadata),
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


def _write_shard(
    output: Path,
    index: int,
    rows: list[dict[str, Any]],
    lookup: dict[str, Any],
) -> dict[str, Any]:
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
    stat = path.stat()
    return {
        "path": path.name,
        "size": int(stat.st_size),
        "records": len(rows),
        "sha256": sha256_file(path),
    }


def build_or_verify_cache(args: Any) -> dict[str, Any]:
    """Execute the data-plane operation for a validated workflow namespace."""
    config = load_config(args.config)
    output = resolve_project_path(args.output_dir) or Path(args.output_dir)
    requested_components = tuple(
        value.strip() for value in args.components.split(",") if value.strip()
    )
    if not requested_components or set(requested_components) - {
        "single_image", "multisource_parent",
    }:
        raise ValueError("--components 只支持 single_image,multisource_parent")
    description_dir = resolve_project_path(args.description_benchmark)
    bridge_dir = resolve_project_path(args.bridge_benchmark) if args.bridge_benchmark else None

    if args.verify_only:
        bank = DescriptionVisionFeatureBank(output)
        components = tuple(str(value) for value in bank.manifest["components"])
        verify_bridge_ref = args.bridge_benchmark
        if "multisource_parent" in components and not verify_bridge_ref:
            verify_bridge_ref = str(
                bank.manifest["input_fingerprints"]["multisource_parent"]["benchmark"]
            )
            bridge_dir = resolve_project_path(verify_bridge_ref)
        current_inputs = build_input_fingerprints(
            components,
            description_ref=args.description_benchmark,
            description_dir=description_dir,
            bridge_ref=verify_bridge_ref,
            bridge_dir=bridge_dir,
        )
        provenance = bank.manifest["source_cache_provenance"]
        source_ref = args.segmentation_vision_cache or provenance.get("path")
        source_path = resolve_project_path(source_ref) if source_ref else None
        source_bank = None
        source_init_error = None
        if bool(provenance["provided"]) and source_path is not None:
            try:
                source_bank = QwenVisionFeatureBank(
                    source_path, decoder_dim=config.decoder_dim
                )
            except Exception as exc:
                source_init_error = f"源 segmentation cache 无法加载: {exc}"
        report = deep_validation_report(
            output,
            input_fingerprints=current_inputs,
            source_bank=source_bank,
            source_cache_path=source_path,
        )
        if source_init_error is not None:
            report["errors"].append(source_init_error)
            report["num_errors"] = int(report.get("num_errors") or 0) + 1
            report["status"] = "invalid"
        return report

    components = requested_components
    validate_output_replacement_safety(output, {
        "config": args.config,
        "Qwen model": config.qwen_model_path,
        "Description benchmark": args.description_benchmark,
        "Bridge benchmark": args.bridge_benchmark,
        "segmentation Vision Cache v3": args.segmentation_vision_cache,
    })
    input_fingerprints = build_input_fingerprints(
        components,
        description_ref=args.description_benchmark,
        description_dir=description_dir,
        bridge_ref=args.bridge_benchmark,
        bridge_dir=bridge_dir,
    )
    source_cache_path = (
        resolve_project_path(args.segmentation_vision_cache)
        if args.segmentation_vision_cache else None
    )
    if source_cache_path is not None and (
        source_cache_path == output
        or source_cache_path in output.parents
        or output in source_cache_path.parents
    ):
        raise ValueError("Description cache output 必须与 segmentation cache v3 完全隔离")
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
        QwenVisionFeatureBank(source_cache_path, decoder_dim=config.decoder_dim)
        if source_cache_path is not None else None
    )
    source_before = (
        source_cache_snapshot(source_cache_path)
        if source_cache_path is not None else None
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
    if "single_image" in components:
        assert description_dir is not None
        iterators.append(single_image_records(description_dir, args.render_size))
    if "multisource_parent" in components:
        assert bridge_dir is not None
        iterators.append(bridge_records(bridge_dir, args.render_size, segmentation_bank))

    lookup: dict[str, Any] = {}
    shards: list[str] = []
    shard_fingerprints: list[dict[str, Any]] = []
    buffer: list[dict[str, Any]] = []
    count = 0
    reused_source_records = 0
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
                if serialized.get("source_cache"):
                    reused_source_records += 1
                buffer.append(serialized)
                count += 1
                if len(buffer) >= args.shard_size:
                    shard = _write_shard(output, len(shards), buffer, lookup)
                    shards.append(str(shard["path"]))
                    shard_fingerprints.append(shard)
                    buffer = []
            if args.max_samples > 0 and count >= args.max_samples:
                break
        if buffer:
            shard = _write_shard(output, len(shards), buffer, lookup)
            shards.append(str(shard["path"]))
            shard_fingerprints.append(shard)
    finally:
        encoder.close()

    source_after = (
        source_cache_snapshot(source_cache_path)
        if source_cache_path is not None else None
    )
    source_unchanged = source_before == source_after
    if not source_unchanged:
        invalid_report = {
            "protocol": DESCRIPTION_CACHE_VALIDATION_PROTOCOL,
            "format": DESCRIPTION_CACHE_FORMAT,
            "cache_protocol": DESCRIPTION_CACHE_PROTOCOL,
            "builder_version": BUILDER_VERSION,
            "status": "invalid",
            "num_records": count,
            "num_shards": len(shards),
            "shard_integrity": {
                "protocol": "sha256_size_record_count_v1",
                "manifest_entries": len(shard_fingerprints),
                "verified_shards": 0,
                "verified_bytes": 0,
                "all_verified": False,
            },
            "records_by_component": {},
            "input_fingerprints": input_fingerprints,
            "source_cache": {
                "provided": True,
                "reused_records": reused_source_records,
                "validated_records": 0,
                "isolation_unchanged": False,
            },
            "errors": [
                "构建 Description cache 期间源 segmentation cache v3 文件元数据发生变化"
            ],
            "num_errors": 1,
            "num_warnings": 0,
            "errors_truncated": False,
            "warnings": [],
        }
        atomic_write_json(output / "validation_report.json", invalid_report)
        raise RuntimeError(invalid_report["errors"][0])

    source_provenance = {
        "provided": source_cache_path is not None,
        "path": str(args.segmentation_vision_cache) if source_cache_path is not None else None,
        "manifest_sha256": (
            source_before["manifest_sha256"] if source_before is not None else None
        ),
        "metadata_fingerprint": (
            source_before["metadata_fingerprint"] if source_before is not None else None
        ),
        "file_count": source_before["file_count"] if source_before is not None else None,
        "reused_records": reused_source_records,
        "isolation_unchanged": True,
    }
    published_components = tuple(
        component for component in components
        if any(location["component"] == component for location in lookup.values())
    )
    published_input_fingerprints = {
        component: input_fingerprints[component] for component in published_components
    }

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
        "input_fingerprints": published_input_fingerprints,
        "source_cache_provenance": source_provenance,
        "num_samples": count,
        "components": list(published_components),
        "lookup": lookup,
        "shards": shards,
        "shard_fingerprints": shard_fingerprints,
        "shard_size": args.shard_size,
        "forbidden_state": ["instruction", "condition", "region_geometry", "segmentation_state"],
    }
    atomic_write_json(output / "manifest.json", manifest)
    validation_report = deep_validation_report(
        output,
        input_fingerprints=published_input_fingerprints,
        source_bank=segmentation_bank,
        source_cache_path=source_cache_path,
    )
    atomic_write_json(output / "validation_report.json", validation_report)
    if validation_report["errors"]:
        raise RuntimeError(
            "Description vision cache v1 深度验证失败；"
            f"详见 {output / 'validation_report.json'}"
        )
    return {
        "output_dir": str(output), "format": DESCRIPTION_CACHE_FORMAT,
        "num_samples": count, "num_shards": len(shards),
        "components": list(published_components),
        "validation_report": str(output / "validation_report.json"),
        "status": validation_report["status"],
        "errors": [],
    }
