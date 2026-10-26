"""将已冻结的 External-only 发布树确定性重发布为 RS-GeneralDesc native v1。"""

from __future__ import annotations

import json
import os
import shutil
import sys
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from .builder import STAGING_SENTINEL
from .io import (
    atomic_write_json,
    canonical_json,
    first_symlink_component,
    portable_relative_path,
    read_json,
    sha256_bytes,
    sha256_file,
    sha256_text,
)
from .contracts import (
    BENCHMARK_SCOPE,
    BUILD_CONFIG_VERSION,
    CANONICAL_SCHEMA_VERSION,
    EXTERNAL_SPLIT_SCHEMA_VERSION,
    HASH_SCHEMA_VERSION,
    MANIFEST_VERSION,
    PROVENANCE_SCHEMA_VERSION,
    RELEASE_EQUIVALENCE_SCHEMA_VERSION,
    RS_GENERALDESC_SOURCES,
    STATISTICS_SCHEMA_VERSION,
    canonical_schema_document,
    parent_content_fingerprint,
    parent_id_from_fingerprint,
    record_hash,
    record_id,
    validate_canonical_record,
)
from .errors import BuildError, ReasonCode
from .hash_ledger import HashLedgerVerifier


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise BuildError(
                ReasonCode.SCHEMA_MISMATCH,
                f"duplicate JSON key: {key}",
            )
        output[key] = value
    return output


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise BuildError(
                    ReasonCode.SCHEMA_MISMATCH,
                    f"{path}: line {line_number} 为空",
                )
            try:
                value = json.loads(line, object_pairs_hook=_unique_object)
            except (json.JSONDecodeError, ValueError) as error:
                raise BuildError(
                    ReasonCode.SCHEMA_MISMATCH,
                    f"{path}: line {line_number} 非法 JSON",
                ) from error
            if not isinstance(value, dict):
                raise BuildError(
                    ReasonCode.TYPE_MISMATCH,
                    f"{path}: line {line_number} 必须是对象",
                )
            yield value


def _atomic_write_jsonl_stream(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    count = 0
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o644,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(canonical_json(row))
                handle.write("\n")
                count += 1
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return count


def _safe_file(root: Path, relative: str) -> Path:
    pure = portable_relative_path(relative, location=relative)
    path = root.joinpath(*pure.parts)
    linked = first_symlink_component(path)
    if linked is not None:
        raise BuildError(
            ReasonCode.OUTPUT_LINK,
            f"source path 含 symlink：{relative}",
        )
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise BuildError(ReasonCode.PATH_ESCAPE, f"source path 逃逸：{relative}") from error
    if not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1:
        raise BuildError(
            ReasonCode.OUTPUT_LINK,
            f"source path 必须是普通单链接文件：{relative}",
        )
    return path


def _source_records(
    manifest: Mapping[str, Any],
    verifier: HashLedgerVerifier,
) -> Iterable[tuple[str, dict[str, Any]]]:
    layout = manifest.get("layout")
    if not isinstance(layout, dict) or not isinstance(layout.get("record_shards"), list):
        raise BuildError(ReasonCode.SCHEMA_MISMATCH, "source manifest 缺少 record shards")
    for shard in layout["record_shards"]:
        if not isinstance(shard, dict) or set(shard) != {"path", "record_count"}:
            raise BuildError(ReasonCode.SCHEMA_MISMATCH, "source record shard 合同非法")
        relative = shard.get("path")
        if not isinstance(relative, str):
            raise BuildError(ReasonCode.TYPE_MISMATCH, "source shard path 非字符串")
        rows = list(_iter_jsonl(verifier.verify_file(relative)))
        if shard.get("record_count") != len(rows):
            raise BuildError(
                ReasonCode.SCHEMA_MISMATCH,
                f"source shard count 不一致：{relative}",
            )
        for row in rows:
            yield relative, row


def _record_projection(record: Mapping[str, Any]) -> dict[str, Any]:
    omitted = {
        "schema_version",
        "record_id",
        "parent_id",
        "logical_role",
        "provenance_ids",
        "record_sha256",
    }
    projected = {key: value for key, value in record.items() if key not in omitted}
    coordinate = projected.get("coordinate_convention")
    if isinstance(coordinate, dict):
        projected["coordinate_convention"] = {
            key: coordinate[key]
            for key in ("image_origin", "bbox_canonical")
            if key in coordinate
        }
    return projected


def _record_identity(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": CANONICAL_SCHEMA_VERSION,
        "parent_id": record["parent_id"],
        "source": record["source"],
        "logical_role": record["logical_role"],
        "task_family": record["task_family"],
        "supervision_kind": record["supervision_kind"],
        "input_layout": record["input_layout"],
        "output_modality": record["output_modality"],
        "media_asset_ids": sorted(item["asset_id"] for item in record["media"]),
        "target": record["target"],
        "instruction": record["instruction"],
        "training_responses": list(record["training_responses"]),
        "reference_responses": list(record["reference_responses"]),
        "annotation_layer": record["annotation"]["layer"],
    }


def _transform_build_config(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise BuildError(ReasonCode.TYPE_MISMATCH, "source build_config 必须是对象")
    expected = {
        "schema_version",
        "profile",
        "seed",
        "sources",
        "external_split",
        "text_policy",
        "asset_policy",
        "sharding",
        "workers",
        "limits",
        "deletion_allowlist",
    }
    retired_fields = set(value) - expected
    if set(value) - retired_fields != expected or len(retired_fields) != 1:
        raise BuildError(ReasonCode.SCHEMA_MISMATCH, "source build_config 字段异常")
    sources = value.get("sources")
    if not isinstance(sources, dict) or set(sources) != RS_GENERALDESC_SOURCES:
        raise BuildError(ReasonCode.SCHEMA_MISMATCH, "source build_config 三源合同异常")
    if any(item != {"enabled": True} for item in sources.values()):
        raise BuildError(ReasonCode.SCHEMA_MISMATCH, "repackage 只接受三源均启用的 full build")
    asset_policy = value.get("asset_policy")
    if not isinstance(asset_policy, dict):
        raise BuildError(ReasonCode.TYPE_MISMATCH, "source asset_policy 必须是对象")
    native_asset_fields = {
        "image_format",
        "image_quality",
        "jpeg_subsampling",
        "target_size",
    }
    if not native_asset_fields <= set(asset_policy):
        raise BuildError(ReasonCode.SCHEMA_MISMATCH, "source asset_policy 字段异常")
    native_asset_policy = {
        key: asset_policy[key] for key in sorted(native_asset_fields)
    }
    transformed = {
        key: item
        for key, item in value.items()
        if key not in {"schema_version", "asset_policy"} | retired_fields
    }
    transformed.update(
        {
            "schema_version": BUILD_CONFIG_VERSION,
            "asset_policy": native_asset_policy,
        }
    )
    return transformed


def _source_payload_paths(
    layout: Mapping[str, Any],
    asset_inventory: list[Any],
) -> set[str]:
    """从 manifest layout 与 asset inventory 推导完整 ledger 文件集合。"""

    payload_paths: set[str] = set()
    for key, value in layout.items():
        if key in {"hashes", "validation", "record_shards", "role_to_record_shards"}:
            continue
        if value is None:
            continue
        if not isinstance(value, str):
            raise BuildError(
                ReasonCode.SCHEMA_MISMATCH,
                f"source manifest layout.{key} 必须是路径或 null",
            )
        payload_paths.add(value)
    shards = layout.get("record_shards")
    if not isinstance(shards, list) or not shards:
        raise BuildError(ReasonCode.SCHEMA_MISMATCH, "source record shards 为空")
    for shard in shards:
        if not isinstance(shard, dict) or set(shard) != {"path", "record_count"}:
            raise BuildError(ReasonCode.SCHEMA_MISMATCH, "source record shard 合同非法")
        relative = shard.get("path")
        if not isinstance(relative, str) or relative in payload_paths:
            raise BuildError(ReasonCode.SCHEMA_MISMATCH, "source record shard path 非法或重复")
        payload_paths.add(relative)
    for asset in asset_inventory:
        if not isinstance(asset, dict) or not isinstance(asset.get("path"), str):
            raise BuildError(ReasonCode.SCHEMA_MISMATCH, "source asset path 非法")
        relative = str(asset["path"])
        if relative in payload_paths:
            raise BuildError(ReasonCode.SCHEMA_MISMATCH, "source asset path 重复")
        payload_paths.add(relative)
    return payload_paths


def _native_hashes(
    staging: Path,
    copied_assets: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    excluded = {
        "manifest.json",
        "metadata/hashes.json",
        "metadata/validation.json",
    }
    rows: list[dict[str, Any]] = []
    for path in sorted(item for item in staging.rglob("*") if item.is_file()):
        relative = path.relative_to(staging).as_posix()
        if relative in excluded or relative.startswith("."):
            continue
        known = copied_assets.get(relative)
        if known is None:
            row = {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        else:
            row = {
                "path": relative,
                "size_bytes": known["size_bytes"],
                "sha256": known["sha256"],
            }
        rows.append(row)
    root_sha256 = sha256_bytes(canonical_json(rows).encode("utf-8"))
    return {
        "schema_version": HASH_SCHEMA_VERSION,
        "files": rows,
        "root_sha256": root_sha256,
    }


def repackage_benchmark(
    source_root: Path | str,
    target_root: Path | str,
    *,
    expected_manifest_sha256: str,
    expected_build_id: str,
    expected_payload_sha256: str,
    expected_hash_manifest_sha256: str,
) -> Path:
    """重发布冻结 payload；拒绝覆盖、链接与任一前代身份漂移。"""

    source = Path(source_root).absolute()
    target = Path(target_root).absolute()
    if first_symlink_component(source) is not None or not source.is_dir():
        raise BuildError(ReasonCode.OUTPUT_LINK, "source root 必须是无链接普通目录")
    if target.exists() or target.is_symlink():
        raise BuildError(ReasonCode.OUTPUT_EXISTS, f"拒绝覆盖 target root：{target}")
    linked_target = first_symlink_component(target)
    if linked_target is not None:
        raise BuildError(
            ReasonCode.OUTPUT_LINK,
            f"target root 含链接组件：{linked_target}",
        )
    try:
        target.resolve().relative_to(source.resolve())
    except ValueError:
        pass
    else:
        raise BuildError(ReasonCode.PATH_ESCAPE, "target root 不能位于 source root 内")
    manifest_path = _safe_file(source, "manifest.json")
    if sha256_file(manifest_path) != expected_manifest_sha256:
        raise BuildError(ReasonCode.HASH_MISMATCH, "source manifest SHA-256 漂移")
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise BuildError(ReasonCode.TYPE_MISMATCH, "source manifest 必须是对象")
    if (
        manifest.get("build_id") != expected_build_id
        or manifest.get("payload_root_sha256") != expected_payload_sha256
        or manifest.get("hash_manifest_sha256") != expected_hash_manifest_sha256
    ):
        raise BuildError(ReasonCode.HASH_MISMATCH, "source build/payload/hash identity 漂移")
    layout = manifest.get("layout")
    if not isinstance(layout, dict):
        raise BuildError(ReasonCode.SCHEMA_MISMATCH, "source manifest layout 非法")
    hashes_relative = layout.get("hashes")
    if not isinstance(hashes_relative, str):
        raise BuildError(ReasonCode.SCHEMA_MISMATCH, "source hashes path 非法")
    verifier = HashLedgerVerifier(
        source,
        ledger_path=hashes_relative,
        expected_ledger_sha256=expected_hash_manifest_sha256,
        expected_root_sha256=expected_payload_sha256,
        error_type=BuildError,
    )
    saved_validation = read_json(_safe_file(source, str(layout.get("validation"))))
    counts = manifest.get("counts")
    if (
        not isinstance(saved_validation, dict)
        or saved_validation.get("deep") is not True
        or saved_validation.get("errors") != []
        or saved_validation.get("warnings") != []
        or saved_validation.get("payload_root_sha256") != expected_payload_sha256
        or not isinstance(counts, dict)
        or saved_validation.get("record_count") != counts.get("records")
        or saved_validation.get("parent_count") != counts.get("parents")
    ):
        raise BuildError(ReasonCode.HASH_MISMATCH, "source saved deep validation 不干净")

    assets_relative = layout.get("assets")
    if not isinstance(assets_relative, str):
        raise BuildError(ReasonCode.SCHEMA_MISMATCH, "source assets path 非法")
    asset_inventory = read_json(verifier.verify_file(assets_relative))
    if not isinstance(asset_inventory, list):
        raise BuildError(ReasonCode.TYPE_MISMATCH, "source assets inventory 必须是列表")
    source_payload_paths = _source_payload_paths(layout, asset_inventory)
    verifier.assert_entry_set(source_payload_paths)
    verifier.assert_actual_file_set()
    record_paths = {
        str(shard["path"])
        for shard in layout["record_shards"]
        if isinstance(shard, dict) and isinstance(shard.get("path"), str)
    }
    asset_paths = {
        str(asset["path"])
        for asset in asset_inventory
        if isinstance(asset, dict) and isinstance(asset.get("path"), str)
    }
    for relative in sorted(source_payload_paths - record_paths - asset_paths):
        verifier.verify_file(relative)

    staging = target.parent / f".{target.name}.staging-{uuid.uuid4().hex}"
    if staging.exists() or staging.is_symlink():
        raise BuildError(ReasonCode.OUTPUT_EXISTS, f"staging 已存在：{staging}")
    staging.mkdir(parents=True)
    (staging / STAGING_SENTINEL).write_text(
        "owned-by-rs-generaldesc-repackage\n",
        encoding="utf-8",
    )
    try:
        copied_assets: dict[str, dict[str, Any]] = {}
        copied_bytes = 0
        for index, asset in enumerate(asset_inventory, start=1):
            if not isinstance(asset, dict) or asset.get("media_type") != "image":
                raise BuildError(ReasonCode.INVALID_ENUM, "source 仅允许 image asset")
            relative = asset.get("path")
            if not isinstance(relative, str) or not relative.startswith("assets/image/"):
                raise BuildError(ReasonCode.PATH_ESCAPE, "source asset path 非法")
            ledger_entry = verifier.assert_declared_identity(
                relative,
                size_bytes=asset.get("size_bytes"),
                sha256=asset.get("sha256"),
            )
            verifier.copy_verified_file(relative, staging / relative)
            copied_assets[relative] = {
                "size_bytes": ledger_entry.size_bytes,
                "sha256": ledger_entry.sha256,
            }
            copied_bytes += ledger_entry.size_bytes
            if index % 10000 == 0 or index == len(asset_inventory):
                print(
                    f"repackage assets {index}/{len(asset_inventory)} bytes={copied_bytes}",
                    file=sys.stderr,
                    flush=True,
                )
        if (
            len(asset_inventory) != counts.get("assets")
            or copied_bytes != counts.get("asset_bytes")
        ):
            raise BuildError(ReasonCode.HASH_MISMATCH, "source asset count/bytes 漂移")

        split_provenance: dict[str, str] = {}
        provenance_relative = layout.get("provenance")
        if not isinstance(provenance_relative, str):
            raise BuildError(ReasonCode.SCHEMA_MISMATCH, "source provenance path 非法")
        provenance_path = verifier.verify_file(provenance_relative)
        for row in _iter_jsonl(provenance_path):
            details = row.get("details")
            if isinstance(details, dict) and details.get("type") == "external_parent_split":
                provenance_id = row.get("provenance_id")
                component_id = details.get("content_component_id")
                if not isinstance(provenance_id, str) or not isinstance(component_id, str):
                    raise BuildError(ReasonCode.SCHEMA_MISMATCH, "source split provenance 非法")
                split_provenance[provenance_id] = component_id

        parent_map: dict[str, str] = {}
        native_parent_ids: set[str] = set()
        parent_source: dict[str, str] = {}
        parent_role: dict[str, str] = {}
        split_parent: dict[str, str] = {}
        parent_mapping_rows: list[dict[str, str]] = []
        current_parent: str | None = None
        current_rows: list[dict[str, Any]] = []
        closed_parents: set[str] = set()

        def finalize_parent() -> None:
            nonlocal current_parent, current_rows
            if current_parent is None:
                return
            fingerprint = parent_content_fingerprint(current_rows)
            native_parent = parent_id_from_fingerprint(fingerprint)
            if native_parent in native_parent_ids:
                raise BuildError(ReasonCode.HASH_MISMATCH, "native parent ID collision")
            native_parent_ids.add(native_parent)
            sources = {str(row.get("source")) for row in current_rows}
            roles = {str(row.get("logical_role")) for row in current_rows}
            if len(sources) != 1 or len(roles) != 1:
                raise BuildError(ReasonCode.EXTERNAL_SPLIT_LEAKAGE, "source parent 合同不一致")
            source_name = next(iter(sources))
            role_name = next(iter(roles))
            if source_name not in RS_GENERALDESC_SOURCES or role_name not in {"external_train", "external_val"}:
                raise BuildError(ReasonCode.ROLE_CONTAMINATION, "source parent role/source 非法")
            parent_map[current_parent] = native_parent
            parent_source[current_parent] = source_name
            parent_role[current_parent] = role_name
            parent_mapping_rows.append({"from": current_parent, "to": native_parent})
            closed_parents.add(current_parent)
            current_parent = None
            current_rows = []

        source_record_count = 0
        for _, row in _source_records(manifest, verifier):
            source_record_count += 1
            old_parent = row.get("parent_id")
            if not isinstance(old_parent, str):
                raise BuildError(ReasonCode.SCHEMA_MISMATCH, "source parent_id 非法")
            if current_parent != old_parent:
                finalize_parent()
                if old_parent in closed_parents:
                    raise BuildError(ReasonCode.SCHEMA_MISMATCH, "source parent records 不连续")
                current_parent = old_parent
            if any(
                not isinstance(media, dict) or media.get("media_type") != "image"
                for media in row.get("media", [])
            ) or not isinstance(row.get("target"), dict) or row["target"].get("type") not in {"none", "bbox"}:
                raise BuildError(ReasonCode.ROLE_CONTAMINATION, "source record 含非 RS-GeneralDesc media/target")
            current_rows.append(row)
            for provenance_id in row.get("provenance_ids", []):
                if provenance_id in split_provenance:
                    previous = split_parent.setdefault(provenance_id, old_parent)
                    if previous != old_parent:
                        raise BuildError(ReasonCode.EXTERNAL_SPLIT_LEAKAGE, "split provenance 跨 parent")
        finalize_parent()
        if source_record_count != counts.get("records") or len(parent_map) != counts.get("parents"):
            raise BuildError(ReasonCode.HASH_MISMATCH, "source record/parent count 漂移")

        component_parents: dict[str, set[str]] = defaultdict(set)
        parent_components: dict[str, set[str]] = defaultdict(set)
        for provenance_id, component_id in split_provenance.items():
            old_parent = split_parent.get(provenance_id)
            if old_parent is None:
                raise BuildError(ReasonCode.SCHEMA_MISMATCH, "orphan split provenance")
            component_parents[component_id].add(old_parent)
            parent_components[old_parent].add(component_id)
        if set(parent_components) != set(parent_map) or any(
            len(values) != 1 for values in parent_components.values()
        ):
            raise BuildError(ReasonCode.SCHEMA_MISMATCH, "source component/parent 映射不完整")
        component_map = {
            old_component: "component_"
            + sha256_text(
                canonical_json(sorted(parent_map[parent] for parent in parents))
            )
            for old_component, parents in component_parents.items()
        }
        assignment_rows = [
            {
                "parent_id": parent_map[old_parent],
                "source": parent_source[old_parent],
                "logical_role": parent_role[old_parent],
                "content_component_id": component_map[
                    next(iter(parent_components[old_parent]))
                ],
            }
            for old_parent in sorted(parent_map, key=lambda value: parent_map[value])
        ]
        assignment_sha256 = sha256_text(canonical_json(assignment_rows))
        source_assignment_sha256 = {
            source_name: sha256_text(
                canonical_json(
                    [row for row in assignment_rows if row["source"] == source_name]
                )
            )
            for source_name in sorted(RS_GENERALDESC_SOURCES)
        }

        provenance_map: dict[str, str] = {}
        native_provenance_ids: set[str] = set()

        def transformed_provenance() -> Iterable[dict[str, Any]]:
            for row in _iter_jsonl(provenance_path):
                old_id = row.get("provenance_id")
                if not isinstance(old_id, str) or old_id in provenance_map:
                    raise BuildError(ReasonCode.SCHEMA_MISMATCH, "source provenance ID 非法或重复")
                details = row.get("details")
                if not isinstance(details, dict):
                    raise BuildError(ReasonCode.TYPE_MISMATCH, "source provenance details 必须是对象")
                native_details = dict(details)
                if old_id in split_provenance:
                    native_details["content_component_id"] = component_map[
                        split_provenance[old_id]
                    ]
                    native_details["assignment_sha256"] = assignment_sha256
                identity = {
                    "schema_version": PROVENANCE_SCHEMA_VERSION,
                    "source": row.get("source"),
                    "source_record_id": row.get("source_record_id"),
                    "source_split": row.get("source_split"),
                    "details": native_details,
                }
                native_id = "provenance_" + sha256_text(canonical_json(identity))
                if native_id in native_provenance_ids:
                    raise BuildError(ReasonCode.HASH_MISMATCH, "native provenance ID collision")
                provenance_map[old_id] = native_id
                native_provenance_ids.add(native_id)
                yield {**identity, "provenance_id": native_id}

        provenance_count = _atomic_write_jsonl_stream(
            staging / "metadata/provenance.jsonl",
            transformed_provenance(),
        )
        if not provenance_count:
            raise BuildError(ReasonCode.SCHEMA_MISMATCH, "source provenance 为空")

        record_mapping_rows: list[dict[str, str]] = []
        seen_native_records: set[str] = set()
        role_shards: dict[str, list[str]] = defaultdict(list)
        native_record_count = 0
        for shard_index, shard in enumerate(layout["record_shards"], start=1):
            relative = str(shard["path"])

            def transformed_records() -> Iterable[dict[str, Any]]:
                nonlocal native_record_count
                for old in _iter_jsonl(verifier.verify_file(relative)):
                    old_id = old.get("record_id")
                    old_parent = old.get("parent_id")
                    if not isinstance(old_id, str) or old_parent not in parent_map:
                        raise BuildError(ReasonCode.SCHEMA_MISMATCH, "source record identity 非法")
                    native = dict(old)
                    native["schema_version"] = CANONICAL_SCHEMA_VERSION
                    native["parent_id"] = parent_map[old_parent]
                    native["provenance_ids"] = sorted(
                        provenance_map[value] for value in old["provenance_ids"]
                    )
                    native["coordinate_convention"] = {
                        "image_origin": "top_left",
                        "bbox_canonical": "xyxy_normalized",
                    }
                    native["record_id"] = record_id(_record_identity(native))
                    native["record_sha256"] = record_hash(native)
                    validate_canonical_record(native)
                    if _record_projection(old) != _record_projection(native):
                        raise BuildError(ReasonCode.HASH_MISMATCH, "record content projection 漂移")
                    if native["logical_role"] != old["logical_role"]:
                        raise BuildError(ReasonCode.EXTERNAL_SPLIT_LEAKAGE, "record role 漂移")
                    if native["record_id"] in seen_native_records:
                        raise BuildError(ReasonCode.HASH_MISMATCH, "native record ID collision")
                    seen_native_records.add(native["record_id"])
                    record_mapping_rows.append({"from": old_id, "to": native["record_id"]})
                    native_record_count += 1
                    yield native

            written = _atomic_write_jsonl_stream(
                staging / relative,
                transformed_records(),
            )
            if written != shard.get("record_count"):
                raise BuildError(ReasonCode.HASH_MISMATCH, f"native shard count 漂移：{relative}")
            shard_roles = {
                role
                for role, paths in layout.get("role_to_record_shards", {}).items()
                if relative in paths
            }
            for role in sorted(shard_roles):
                role_shards[role].append(relative)
            if shard_index % 25 == 0 or shard_index == len(layout["record_shards"]):
                print(
                    f"repackage records shards={shard_index}/{len(layout['record_shards'])} records={native_record_count}",
                    file=sys.stderr,
                    flush=True,
                )
        if native_record_count != counts.get("records"):
            raise BuildError(ReasonCode.HASH_MISMATCH, "native record count 漂移")

        statistics_relative = layout.get("statistics")
        if not isinstance(statistics_relative, str):
            raise BuildError(ReasonCode.SCHEMA_MISMATCH, "source statistics path 非法")
        source_statistics = read_json(verifier.verify_file(statistics_relative))
        if not isinstance(source_statistics, dict):
            raise BuildError(ReasonCode.TYPE_MISMATCH, "source statistics 必须是对象")
        statistics = dict(source_statistics)
        statistics["schema_version"] = STATISTICS_SCHEMA_VERSION
        external_split = dict(statistics["external_split"])
        external_split["schema_version"] = EXTERNAL_SPLIT_SCHEMA_VERSION
        external_split["assignment_sha256"] = assignment_sha256
        statistics["external_split"] = external_split

        source_audits_relative = layout.get("source_audits")
        if not isinstance(source_audits_relative, str):
            raise BuildError(ReasonCode.SCHEMA_MISMATCH, "source audits path 非法")
        source_audits = read_json(verifier.verify_file(source_audits_relative))
        if not isinstance(source_audits, dict) or set(source_audits) != RS_GENERALDESC_SOURCES:
            raise BuildError(ReasonCode.SCHEMA_MISMATCH, "source audits 三源合同异常")
        native_audits: dict[str, Any] = {}
        for source_name, audit in source_audits.items():
            if not isinstance(audit, dict):
                raise BuildError(ReasonCode.TYPE_MISMATCH, "source audit 必须是对象")
            native_audit = dict(audit)
            split_audit = native_audit.get("external_split")
            if not isinstance(split_audit, dict):
                raise BuildError(ReasonCode.SCHEMA_MISMATCH, "source split audit 缺失")
            native_audit["external_split"] = {
                **split_audit,
                "assignment_sha256": source_assignment_sha256[source_name],
            }
            native_audits[source_name] = native_audit

        build_config_relative = layout.get("build_config")
        if not isinstance(build_config_relative, str):
            raise BuildError(ReasonCode.SCHEMA_MISMATCH, "source build config path 非法")
        build_config = _transform_build_config(
            read_json(verifier.verify_file(build_config_relative))
        )
        semantic_config_sha256 = sha256_text(canonical_json(build_config))
        atomic_write_json(staging / "schemas/canonical.schema.json", canonical_schema_document())
        atomic_write_json(staging / "metadata/assets.json", asset_inventory)
        atomic_write_json(staging / "metadata/statistics.json", statistics)
        atomic_write_json(staging / "metadata/source_audits.json", native_audits)
        atomic_write_json(staging / "metadata/build_config.json", build_config)
        skips_relative = layout.get("skips")
        if not isinstance(skips_relative, str):
            raise BuildError(ReasonCode.SCHEMA_MISMATCH, "source skips path 非法")
        _atomic_write_jsonl_stream(
            staging / "metadata/skips.jsonl",
            _iter_jsonl(verifier.verify_file(skips_relative)),
        )

        parent_mapping_rows.sort(key=lambda row: row["from"])
        record_mapping_rows.sort(key=lambda row: row["from"])
        asset_set_rows = [
            {
                "path": str(row["path"]),
                "sha256": str(row["sha256"]),
                "size_bytes": int(row["size_bytes"]),
            }
            for row in asset_inventory
        ]
        asset_set_rows.sort(key=lambda row: row["path"])
        verifier.assert_all_verified()
        equivalence = {
            "schema_version": RELEASE_EQUIVALENCE_SCHEMA_VERSION,
            "predecessor_identity": {
                "manifest_sha256": expected_manifest_sha256,
                "build_id": expected_build_id,
                "payload_root_sha256": expected_payload_sha256,
                "hash_manifest_sha256": expected_hash_manifest_sha256,
            },
            "record_count": counts["records"],
            "parent_count": counts["parents"],
            "asset_count": counts["assets"],
            "asset_bytes": counts["asset_bytes"],
            "record_content_equivalent": True,
            "parent_grouping_equivalent": True,
            "logical_roles_equivalent": True,
            "asset_bytes_equivalent": True,
            "parent_mapping_sha256": sha256_text(canonical_json(parent_mapping_rows)),
            "record_mapping_sha256": sha256_text(canonical_json(record_mapping_rows)),
            "asset_set_sha256": sha256_text(canonical_json(asset_set_rows)),
        }
        equivalence_path = "metadata/release_equivalence.json"
        atomic_write_json(staging / equivalence_path, equivalence)

        hashes = _native_hashes(staging, copied_assets)
        atomic_write_json(staging / "metadata/hashes.json", hashes)
        hash_manifest_sha256 = sha256_file(staging / "metadata/hashes.json")
        build_id_value = "build_" + sha256_text(
            canonical_json([semantic_config_sha256, hashes["root_sha256"]])
        )
        native_layout = {
            "canonical_schema": "schemas/canonical.schema.json",
            "record_shards": list(layout["record_shards"]),
            "role_to_record_shards": {
                role: paths for role, paths in sorted(role_shards.items())
            },
            "provenance": "metadata/provenance.jsonl",
            "skips": "metadata/skips.jsonl",
            "statistics": "metadata/statistics.json",
            "assets": "metadata/assets.json",
            "source_audits": "metadata/source_audits.json",
            "hashes": "metadata/hashes.json",
            "build_config": "metadata/build_config.json",
            "validation": "metadata/validation.json",
            "deletion_allowlist": None,
            "release_equivalence": equivalence_path,
        }
        native_manifest = {
            "schema_version": MANIFEST_VERSION,
            "canonical_schema_version": CANONICAL_SCHEMA_VERSION,
            "build_profile": "full",
            "benchmark_scope": BENCHMARK_SCOPE,
            "scope_validation_complete": True,
            "build_id": build_id_value,
            "semantic_config_sha256": semantic_config_sha256,
            "payload_root_sha256": hashes["root_sha256"],
            "hash_manifest_sha256": hash_manifest_sha256,
            "reference_project": manifest["reference_project"],
            "environment_versions": manifest["environment_versions"],
            "layout": native_layout,
            "counts": dict(counts),
            "source_audits": native_audits,
            "formal_acceptance_eligible": True,
            "formal_acceptance_blockers": [],
            "source_roots_embedded": False,
            "external_test_retained": False,
            "upstream_split_policy": manifest["upstream_split_policy"],
            "official_upstream_evaluation_reproducible": False,
        }
        atomic_write_json(staging / "manifest.json", native_manifest)

        from .validator import validate_benchmark

        validation = validate_benchmark(staging, deep=True)
        if validation["errors"]:
            raise BuildError(
                ReasonCode.STAGING_FAILURE,
                "native staging deep validation 失败",
                details={"errors": validation["errors"][:50]},
            )
        atomic_write_json(staging / "metadata/validation.json", validation)
        if target.exists() or target.is_symlink():
            raise BuildError(ReasonCode.OUTPUT_EXISTS, f"发布前 target 已存在：{target}")
        (staging / STAGING_SENTINEL).unlink()
        staging.replace(target)
        parent_descriptor = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    except BaseException:
        if staging.exists() and (staging / STAGING_SENTINEL).is_file():
            shutil.rmtree(staging)
        raise
    return target
