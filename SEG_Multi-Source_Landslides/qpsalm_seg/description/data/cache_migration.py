#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Strict, side-by-side M3 v2 to current M3 v3 cache migration.

用途：仅在所有 parent 的视觉内容绑定仍一致时复用旧 Description Cache 张量。
推荐命令：使用 ``qpsalm-segdesc cache migrate``，输入旧 cache 与当前 benchmark。
输入：M3 v2 cache、Description v4、Bridge v7、只读 segmentation cache v3。
输出：新的 M3 v3 cache 目录、migration audit、深度 validation report。
写入行为：只创建新的输出目录；旧 cache、benchmark 与 source cache 永不修改。
工作流阶段：M3 artifact migration gate。
Reusable M3 data-plane implementation; command routing lives in workflows/cache_migration.py.
"""

from __future__ import annotations

from collections import Counter
import errno
import os
from pathlib import Path
import shutil
from typing import Any

import torch

from qpsalm_seg.config import load_config
from qpsalm_seg.models.vision_cache import QwenVisionFeatureBank
from qpsalm_seg.paths import resolve_project_path, validate_output_replacement_safety

from .vision_cache import (
    DESCRIPTION_CACHE_BUILDER_VERSION,
    DESCRIPTION_CACHE_FORMAT,
    DESCRIPTION_CACHE_PROTOCOL,
    DESCRIPTION_CACHE_SHARD_REPLAY_PROTOCOL,
    DescriptionVisionFeatureBank,
    description_cache_key,
    sha256_file,
    source_cache_snapshot,
    validate_description_cache_record,
    validate_source_cache_snapshot,
)
from ..protocols.io import atomic_write_json, read_json, read_jsonl
from .cache_builder import (
    build_input_fingerprints,
    deep_validation_report,
    multisource_content_hash,
)


LEGACY_CACHE_BUILDER_VERSION = "description_vision_cache_m3_v2_deep_validation"
LEGACY_CACHE_VALIDATION_PROTOCOL = "qpsalm_description_vision_cache_validation_v1"
CACHE_MIGRATION_PROTOCOL = (
    "qpsalm_description_vision_cache_migration_v2_published_replay_bound"
)


def revalidate_source_cache_provenance(
    bank: DescriptionVisionFeatureBank,
) -> dict[str, Any]:
    """Replay the segmentation cache v3 snapshot that justified M3 reuse."""
    provenance = dict(bank.manifest.get("source_cache_provenance") or {})
    components = {
        str(value) for value in bank.manifest.get("components") or []
    }
    source_required = "multisource_parent" in components
    source_ref = str(provenance.get("path") or "")
    source = resolve_project_path(source_ref) if source_ref else None
    errors: list[str] = []
    if source_required:
        if provenance.get("provided") is not True:
            errors.append(
                "multisource M3 cache 未绑定 segmentation Vision Cache v3"
            )
        if source is None or not source.is_dir():
            errors.append(
                f"segmentation Vision Cache v3 不存在: {source_ref!r}"
            )
        else:
            errors.extend(validate_source_cache_snapshot(provenance, source))
        if provenance.get("isolation_unchanged") is not True:
            errors.append("M3 cache 未证明 segmentation source cache 隔离")
    elif provenance.get("provided") is True:
        if source is None or not source.is_dir():
            errors.append(
                f"已声明的 segmentation source cache 不存在: {source_ref!r}"
            )
        else:
            errors.extend(validate_source_cache_snapshot(provenance, source))
    return {
        "required": source_required,
        "provided": provenance.get("provided") is True,
        "path": (
            str(source.resolve(strict=False))
            if source is not None else None
        ),
        "provenance": provenance,
        "errors": errors,
        "current": not errors,
    }


def _resolved_dir(value: str | Path, *, label: str) -> Path:
    path = resolve_project_path(value) or Path(value)
    if not path.is_dir():
        raise FileNotFoundError(f"{label} 不存在: {path}")
    return path.resolve(strict=False)


def _validate_legacy_manifest(
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = root / "manifest.json"
    report_path = root / "validation_report.json"
    manifest = read_json(manifest_path, label="legacy cache manifest")
    report = read_json(report_path, label="legacy cache validation report")
    expected = {
        "format": DESCRIPTION_CACHE_FORMAT,
        "protocol": DESCRIPTION_CACHE_PROTOCOL,
        "builder_version": LEGACY_CACHE_BUILDER_VERSION,
    }
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise ValueError(
                f"legacy cache {field} 不匹配: {manifest.get(field)!r} != {value!r}"
            )
    required = {
        "renderer_version", "model_revision", "processor_revision", "layers",
        "spatial_sizes", "render_size", "view_tokens_per_view",
        "spatial_channels", "token_dim", "backend", "input_fingerprints",
        "source_cache_provenance", "num_samples", "components", "lookup",
        "shards", "shard_size", "forbidden_state",
    }
    missing = sorted(required - set(manifest))
    if missing:
        raise ValueError(f"legacy cache manifest 字段不完整: {missing}")
    if (
        report.get("protocol") != LEGACY_CACHE_VALIDATION_PROTOCOL
        or report.get("format") != DESCRIPTION_CACHE_FORMAT
        or report.get("cache_protocol") != DESCRIPTION_CACHE_PROTOCOL
        or report.get("builder_version") != LEGACY_CACHE_BUILDER_VERSION
        or report.get("status") != "valid"
        or report.get("errors") != []
        or int(report.get("num_errors", -1)) != 0
    ):
        raise ValueError("legacy cache validation report 未通过或协议不匹配")
    if (
        int(report.get("num_records", -1)) != int(manifest["num_samples"])
        or int(report.get("num_shards", -1)) != len(manifest["shards"])
        or report.get("input_fingerprints") != manifest["input_fingerprints"]
    ):
        raise ValueError("legacy cache validation report 与 manifest 不一致")
    if set(manifest["forbidden_state"]) != {
        "instruction", "condition", "region_geometry", "segmentation_state",
    }:
        raise ValueError("legacy cache task-neutral forbidden_state 不完整")
    if int(manifest["num_samples"]) != len(manifest["lookup"]):
        raise ValueError("legacy cache lookup/sample 数量不一致")
    shards = [str(value) for value in manifest["shards"]]
    if not shards or len(shards) != len(set(shards)):
        raise ValueError("legacy cache shards 为空或重复")
    if any(Path(name).name != name for name in shards):
        raise ValueError("legacy cache shard 必须是 cache 根目录内文件")
    return manifest, report


def _current_source_content(
    description_dir: Path,
    bridge_dir: Path,
    source_bank: QwenVisionFeatureBank,
) -> dict[str, str]:
    bindings: dict[str, str] = {}
    for row in read_jsonl(
        description_dir / "indexes/all.jsonl", label="Description all index"
    ):
        parent = str(row["parent_sample_id"])
        key = description_cache_key("single_image", parent)
        content_hash = str((row.get("visual_ref") or {}).get("sha256") or "")
        if len(content_hash) != 64:
            raise ValueError(f"Description visual sha256 非法: {key}")
        previous = bindings.setdefault(key, content_hash)
        if previous != content_hash:
            raise ValueError(f"Description parent 出现冲突视觉绑定: {key}")

    seen_bridge: set[str] = set()
    for row in read_jsonl(
        bridge_dir / "indexes/candidate_all.jsonl", label="Bridge candidate index"
    ):
        parent = str(row["parent_sample_id"])
        if parent in seen_bridge:
            continue
        seen_bridge.add(parent)
        key = description_cache_key("multisource_parent", parent)
        source_key = f"qmv3-parent:{parent}"
        if source_key in source_bank.lookup:
            content_hash = str(
                source_bank.task_neutral_record(source_key)["cache_fingerprint"]
            )
        else:
            content_hash = multisource_content_hash(row.get("modality_metadata") or {})
        bindings[key] = content_hash
    return bindings


def hardlink_shard(source: Path, target: Path) -> dict[str, Any]:
    try:
        os.link(source, target)
    except OSError as exc:
        if exc.errno == errno.EXDEV:
            raise RuntimeError(
                "legacy/output 不在同一文件系统，严格迁移禁止复制张量；请完整重建 M3 v3"
            ) from exc
        raise
    source_stat = source.stat()
    target_stat = target.stat()
    if (
        source_stat.st_dev != target_stat.st_dev
        or source_stat.st_ino != target_stat.st_ino
        or source_stat.st_size != target_stat.st_size
    ):
        raise RuntimeError(f"hardlink inode/size 审计失败: {source.name}")
    return {
        "source": str(source),
        "target": target.name,
        "device": int(source_stat.st_dev),
        "source_inode": int(source_stat.st_ino),
        "target_inode": int(target_stat.st_ino),
        "same_inode": True,
    }


def load_legacy_shard(path: Path) -> list[dict[str, Any]]:
    """Open one legacy shard and reject unreadable or foreign payloads."""
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise ValueError(f"legacy cache shard 无法读取: {path}") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("format") != DESCRIPTION_CACHE_FORMAT
        or not isinstance(payload.get("records"), list)
        or not all(isinstance(row, dict) for row in payload["records"])
    ):
        raise ValueError(f"legacy shard 结构损坏: {path}")
    return payload["records"]


def validate_migration_record(
    row: dict[str, Any],
    *,
    legacy_manifest: dict[str, Any],
    shard_index: int,
    local_index: int,
    current_content: dict[str, str],
    source_record_fingerprint: Any,
) -> bool:
    """Validate one reused tensor record against its current parent source.

    Returns whether the record directly reuses a segmentation cache v3 record.
    """
    validate_description_cache_record(row, legacy_manifest)
    key = str(row["lookup_key"])
    location = legacy_manifest["lookup"].get(key)
    expected_location = {
        "shard": int(shard_index),
        "index": int(local_index),
        "component": row["component"],
        "parent_sample_id": row["parent_sample_id"],
    }
    if location != expected_location:
        raise ValueError(
            f"legacy lookup/shard 位置不一致: {key} "
            f"lookup={location!r} actual={expected_location!r}"
        )
    if key not in current_content:
        raise RuntimeError(f"当前 benchmark 缺少 legacy parent: {key}")
    expected_content = current_content[key]
    if str(row["source_content_hash"]) != expected_content:
        raise RuntimeError(
            "parent source_content_hash 已漂移，禁止复用张量；"
            f"请完整重建 M3 v3: {key}"
        )
    source_key = row.get("source_cache")
    if not source_key:
        return False
    current_source_fingerprint = str(
        source_record_fingerprint(str(source_key))
    )
    if current_source_fingerprint != expected_content:
        raise RuntimeError(
            f"segmentation cache record 已漂移，禁止复用: {source_key}"
        )
    return True


def revalidate_published_cache_migration(
    cache: str | Path,
    bank: DescriptionVisionFeatureBank,
) -> dict[str, Any]:
    """Replay the side-by-side hardlink audit without modifying either cache."""
    cache_root = _resolved_dir(cache, label="M3 v3 cache")
    migration_path = cache_root / "migration_report.json"
    migration = read_json(migration_path, label="M3 migration report")
    manifest_migration = dict(bank.manifest.get("migration") or {})
    published_replay = dict(migration.get("published_replay") or {})
    legacy = resolve_project_path(
        str(manifest_migration.get("source_cache_dir") or "")
    )
    legacy = (legacy or Path(".")).resolve(strict=False)
    hardlinks = manifest_migration.get("hardlinks")
    hardlinks = hardlinks if isinstance(hardlinks, list) else []
    expected_targets = set(bank.shards)
    observed_targets: set[str] = set()
    hardlink_errors: list[str] = []
    for index, raw in enumerate(hardlinks):
        item = dict(raw) if isinstance(raw, dict) else {}
        source = Path(str(item.get("source") or "")).resolve(strict=False)
        target_name = str(item.get("target") or "")
        target = cache_root / target_name
        observed_targets.add(target_name)
        try:
            source.relative_to(legacy)
        except ValueError:
            hardlink_errors.append(f"hardlink[{index}] source 越出 legacy cache")
            continue
        if source.name != target_name:
            hardlink_errors.append(f"hardlink[{index}] shard 名称不一致")
            continue
        if not source.is_file() or not target.is_file():
            hardlink_errors.append(f"hardlink[{index}] source/target 缺失")
            continue
        source_stat = source.stat()
        target_stat = target.stat()
        if (
            source_stat.st_dev != target_stat.st_dev
            or source_stat.st_ino != target_stat.st_ino
            or source_stat.st_size != target_stat.st_size
            or int(item.get("device", -1)) != source_stat.st_dev
            or int(item.get("source_inode", -1)) != source_stat.st_ino
            or int(item.get("target_inode", -1)) != target_stat.st_ino
            or item.get("same_inode") is not True
        ):
            hardlink_errors.append(f"hardlink[{index}] inode/size audit 漂移")

    legacy_manifest = legacy / "manifest.json"
    legacy_validation = legacy / "validation_report.json"
    source_cache_audit = revalidate_source_cache_provenance(bank)
    checks = {
        "report_current": (
            migration.get("protocol") == CACHE_MIGRATION_PROTOCOL
            and migration.get("status") == "engineering-valid"
            and migration.get("errors") == []
            and Path(str(migration.get("output_dir") or "")).resolve(
                strict=False
            ) == cache_root
        ),
        "population_current": (
            int(migration.get("records", -1))
            == int(bank.manifest.get("num_samples", -2))
            and int(migration.get("shards", -1)) == len(bank.shards)
            and int(manifest_migration.get("reused_records", -1))
            == int(bank.manifest.get("num_samples", -2))
            and int(manifest_migration.get("reused_shards", -1))
            == len(bank.shards)
            and int(migration.get("reused_bytes", -1))
            == int(manifest_migration.get("reused_bytes", -2))
        ),
        "published_terminal_replay": (
            published_replay.get("protocol")
            == DESCRIPTION_CACHE_SHARD_REPLAY_PROTOCOL
            and published_replay.get("all_verified") is True
            and int(published_replay.get("verified_shards", -1))
            == len(bank.shards)
            and int(published_replay.get("verified_bytes", -1))
            == int(manifest_migration.get("reused_bytes", -2))
            and published_replay.get("metadata_snapshot")
            == bank.file_metadata_snapshot()
        ),
        "legacy_reports_current": (
            legacy != cache_root
            and legacy.is_dir()
            and legacy_manifest.is_file()
            and legacy_validation.is_file()
            and sha256_file(legacy_manifest)
            == manifest_migration.get("source_manifest_sha256")
            and sha256_file(legacy_validation)
            == manifest_migration.get("source_validation_report_sha256")
        ),
        "segmentation_source_cache_current": (
            source_cache_audit.get("current") is True
        ),
        "hardlinks_current": (
            manifest_migration.get("protocol") == CACHE_MIGRATION_PROTOCOL
            and manifest_migration.get("reuse_method") == "hardlink"
            and manifest_migration.get("all_same_inode") is True
            and len(hardlinks) == len(bank.shards)
            and observed_targets == expected_targets
            and not hardlink_errors
        ),
    }
    return {
        "origin": "strict_migration",
        "path": str(migration_path.resolve(strict=False)),
        "sha256": sha256_file(migration_path),
        "report": migration,
        "legacy_cache": str(legacy),
        "segmentation_source_cache": source_cache_audit,
        "hardlink_errors": hardlink_errors,
        "checks": checks,
    }


def revalidate_published_cache_origin(
    cache: str | Path,
    bank: DescriptionVisionFeatureBank,
) -> dict[str, Any]:
    """Accept either a strict migration or a native current-builder artifact."""
    cache_root = _resolved_dir(cache, label="M3 v3 cache")
    migration_path = cache_root / "migration_report.json"
    if migration_path.is_file():
        return revalidate_published_cache_migration(cache_root, bank)
    validation = dict(bank.validation_report or {})
    source_cache_audit = revalidate_source_cache_provenance(bank)
    checks = {
        "current_builder": (
            bank.manifest.get("builder_version")
            == DESCRIPTION_CACHE_BUILDER_VERSION
        ),
        "current_validation": (
            validation.get("status") == "valid"
            and validation.get("errors") == []
        ),
        "native_build_has_no_migration_metadata": (
            "migration" not in bank.manifest
        ),
        "segmentation_source_cache_current": (
            source_cache_audit.get("current") is True
        ),
    }
    return {
        "origin": "native_m3_v3_build",
        "path": None,
        "sha256": None,
        "report": None,
        "legacy_cache": None,
        "segmentation_source_cache": source_cache_audit,
        "hardlink_errors": [],
        "checks": checks,
    }


def migrate_cache(args: Any) -> dict[str, Any]:
    config = load_config(args.config)
    legacy = _resolved_dir(args.legacy_cache, label="legacy M3 v2 cache")
    description_dir = _resolved_dir(
        args.description_benchmark, label="Description benchmark"
    )
    bridge_dir = _resolved_dir(args.bridge_benchmark, label="Bridge benchmark")
    source_dir = _resolved_dir(
        args.segmentation_vision_cache, label="segmentation Vision Cache v3"
    )
    output = resolve_project_path(args.output_dir) or Path(args.output_dir)
    output = output.resolve(strict=False)
    validate_output_replacement_safety(output, {
        "legacy cache": legacy,
        "Description benchmark": description_dir,
        "Bridge benchmark": bridge_dir,
        "segmentation Vision Cache v3": source_dir,
    })
    if output.exists():
        raise FileExistsError(
            f"迁移输出已存在，拒绝覆盖: {output}；请换新目录或人工清理失败产物"
        )
    staging = output.with_name(f".{output.name}.migration.part")
    if staging.exists():
        raise FileExistsError(f"残留 migration staging 目录: {staging}")

    legacy_manifest, _ = _validate_legacy_manifest(legacy)
    components = tuple(str(value) for value in legacy_manifest["components"])
    if set(components) != {"single_image", "multisource_parent"}:
        raise ValueError("本迁移只接受完整 single_image + multisource_parent cache")
    current_inputs = build_input_fingerprints(
        components,
        description_ref=str(args.description_benchmark),
        description_dir=description_dir,
        bridge_ref=str(args.bridge_benchmark),
        bridge_dir=bridge_dir,
    )
    source_snapshot = source_cache_snapshot(source_dir)
    source_errors = validate_source_cache_snapshot(
        legacy_manifest["source_cache_provenance"], source_dir
    )
    if source_errors:
        raise RuntimeError(
            "legacy cache 的 segmentation cache v3 provenance 已漂移；请完整重建 M3 v3: "
            + "; ".join(source_errors)
        )
    source_bank = QwenVisionFeatureBank(source_dir, decoder_dim=config.decoder_dim)
    current_content = _current_source_content(
        description_dir, bridge_dir, source_bank
    )
    if set(current_content) != set(legacy_manifest["lookup"]):
        missing = sorted(set(legacy_manifest["lookup"]) - set(current_content))
        added = sorted(set(current_content) - set(legacy_manifest["lookup"]))
        raise RuntimeError(
            "当前 benchmark parent population 与 legacy cache 不一致；请完整重建 M3 v3: "
            f"missing={missing[:8]} added={added[:8]}"
        )

    staging.mkdir(parents=True)
    shard_fingerprints: list[dict[str, Any]] = []
    hardlinks: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    by_component: Counter[str] = Counter()
    reused_source_records = 0
    try:
        for shard_index, shard_name in enumerate(legacy_manifest["shards"]):
            source_path = legacy / str(shard_name)
            if not source_path.is_file():
                raise FileNotFoundError(f"legacy cache 缺少 shard: {source_path}")
            source_hash = sha256_file(source_path)
            rows = load_legacy_shard(source_path)
            expected_records = min(
                int(legacy_manifest["shard_size"]),
                int(legacy_manifest["num_samples"])
                - shard_index * int(legacy_manifest["shard_size"]),
            )
            if len(rows) != expected_records:
                raise ValueError(
                    f"legacy shard record 数不一致: {shard_name} "
                    f"expected={expected_records} actual={len(rows)}"
                )
            for local_index, row in enumerate(rows):
                key = str(row["lookup_key"])
                if key in seen_keys:
                    raise ValueError(f"legacy cache lookup_key 重复: {key}")
                seen_keys.add(key)
                if validate_migration_record(
                    row,
                    legacy_manifest=legacy_manifest,
                    shard_index=shard_index,
                    local_index=local_index,
                    current_content=current_content,
                    source_record_fingerprint=lambda source_key: (
                        source_bank.task_neutral_record(source_key)["cache_fingerprint"]
                    ),
                ):
                    reused_source_records += 1
                by_component[str(row["component"])] += 1
            target_path = staging / str(shard_name)
            hardlinks.append(hardlink_shard(source_path, target_path))
            shard_fingerprints.append({
                "path": str(shard_name),
                "size": int(source_path.stat().st_size),
                "records": len(rows),
                "sha256": source_hash,
            })
            del rows

        if seen_keys != set(legacy_manifest["lookup"]):
            raise ValueError("legacy shard population 与 lookup 不一致")
        provenance = {
            "provided": True,
            "path": str(args.segmentation_vision_cache),
            "manifest_sha256": source_snapshot["manifest_sha256"],
            "metadata_fingerprint": source_snapshot["metadata_fingerprint"],
            "file_count": source_snapshot["file_count"],
            "reused_records": reused_source_records,
            "isolation_unchanged": True,
        }
        manifest = dict(legacy_manifest)
        manifest.update({
            "builder_version": DESCRIPTION_CACHE_BUILDER_VERSION,
            "input_fingerprints": current_inputs,
            "source_cache_provenance": provenance,
            "shard_fingerprints": shard_fingerprints,
            "migration": {
                "protocol": CACHE_MIGRATION_PROTOCOL,
                "source_cache_dir": str(legacy),
                "source_manifest_sha256": sha256_file(legacy / "manifest.json"),
                "source_validation_report_sha256": sha256_file(
                    legacy / "validation_report.json"
                ),
                "reuse_method": "hardlink",
                "reused_shards": len(hardlinks),
                "reused_records": len(seen_keys),
                "reused_bytes": sum(
                    int(value["size"]) for value in shard_fingerprints
                ),
                "all_same_inode": all(
                    value["same_inode"] for value in hardlinks
                ),
                "hardlinks": hardlinks,
            },
        })
        atomic_write_json(staging / "manifest.json", manifest)
        report = deep_validation_report(
            staging,
            input_fingerprints=current_inputs,
            source_bank=source_bank,
            source_cache_path=source_dir,
        )
        report["migration"] = manifest["migration"]
        atomic_write_json(staging / "validation_report.json", report)
        if report["errors"]:
            raise RuntimeError(
                "迁移 cache 深度验证失败；staging 将被移除: "
                + "; ".join(str(value) for value in report["errors"][:8])
            )
        # 发布前再确认只读 source cache 没有发生任何元数据变化。
        final_source_errors = validate_source_cache_snapshot(provenance, source_dir)
        if final_source_errors:
            raise RuntimeError("迁移期间 source cache 发生变化: " + "; ".join(final_source_errors))
        result = {
            "protocol": CACHE_MIGRATION_PROTOCOL,
            "status": "engineering-valid",
            "output_dir": str(output),
            "records": len(seen_keys),
            "records_by_component": dict(sorted(by_component.items())),
            "shards": len(shard_fingerprints),
            "reused_bytes": manifest["migration"]["reused_bytes"],
            "errors": [],
        }
        atomic_write_json(staging / "migration_report.json", result)
        output.parent.mkdir(parents=True, exist_ok=True)
        staging.replace(output)
        # 正式路径下重新加载并遍历全部 shard；staging 成功不能替代发布终态。
        try:
            published = DescriptionVisionFeatureBank(output, max_open_shards=1)
            published.artifact_binding()
            published_replay = published.verify_all_shards()
            if published_replay.get("all_verified") is not True:
                raise RuntimeError(
                    "M3 v3 cache 发布终态未通过全 shard replay"
                )
            result["published_replay"] = {
                "protocol": published_replay.get("protocol"),
                "all_verified": published_replay.get("all_verified"),
                "verified_shards": published_replay.get("verified_shards"),
                "verified_bytes": published_replay.get("verified_bytes"),
                "metadata_snapshot": published_replay.get(
                    "metadata_snapshot"
                ),
            }
            atomic_write_json(output / "migration_report.json", result)
            published_migration = revalidate_published_cache_migration(
                output, published
            )
            if not all(published_migration["checks"].values()):
                raise RuntimeError(
                    "M3 v3 cache 发布终态未通过 inode/source replay"
                )
            result["published_migration_checks"] = dict(
                published_migration["checks"]
            )
            atomic_write_json(output / "migration_report.json", result)
            final_migration = revalidate_published_cache_migration(
                output, published
            )
            if not all(final_migration["checks"].values()):
                raise RuntimeError(
                    "M3 v3 migration report 发布后 live replay 失败"
                )
        except Exception:
            # 只删除本次新建的 hardlink 目录；legacy shard 本体不会受到影响。
            shutil.rmtree(output)
            raise
        return result
    except Exception:
        if staging.is_dir():
            shutil.rmtree(staging)
        raise
