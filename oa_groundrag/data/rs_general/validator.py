"""RS-GeneralDesc native v1 的严格只读 validator。"""

from __future__ import annotations

import io
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from PIL import Image, UnidentifiedImageError

from .common import (
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
    EXTERNAL_SPLIT_VERSION,
    HF_DATASETS_REFERENCE,
    MANIFEST_VERSION,
    PROVENANCE_SCHEMA_VERSION,
    RELEASE_EQUIVALENCE_SCHEMA_VERSION,
    RS_GENERALDESC_SOURCES,
    STATISTICS_SCHEMA_VERSION,
    VALIDATION_VERSION,
    canonical_schema_document,
    validate_canonical_record,
)


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_WINDOWS_ABSOLUTE = re.compile(r"^[A-Za-z]:[\\/]")


def _error(report: dict[str, Any], message: str) -> None:
    report["errors"].append(message)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON key: {key}")
        output[key] = value
    return output


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"{path}: line {line_number} 为空")
            value = json.loads(line, object_pairs_hook=_unique_object)
            if not isinstance(value, dict):
                raise ValueError(f"{path}: line {line_number} 必须是对象")
            yield value


def _exact_keys(
    value: Any,
    expected: set[str],
    report: dict[str, Any],
    location: str,
) -> bool:
    if not isinstance(value, dict):
        _error(report, f"{location}: 必须是对象")
        return False
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing or unknown:
        _error(report, f"{location}: missing={missing} unknown={unknown}")
        return False
    return True


def _absolute_strings(value: Any, *, location: str = "$") -> list[str]:
    found: list[str] = []
    if isinstance(value, str):
        if value.startswith("/") or _WINDOWS_ABSOLUTE.match(value):
            found.append(location)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(_absolute_strings(item, location=f"{location}[{index}]"))
    elif isinstance(value, dict):
        for key, item in value.items():
            found.extend(_absolute_strings(item, location=f"{location}.{key}"))
    return found


def _safe_file(root: Path, relative: str) -> Path:
    pure = portable_relative_path(relative, location=relative)
    path = root.joinpath(*pure.parts)
    linked = first_symlink_component(path)
    if linked is not None:
        raise ValueError(f"路径含 symlink：{relative}")
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"路径逃逸：{relative}") from error
    if not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1:
        raise ValueError(f"必须是普通单链接文件：{relative}")
    return path


def _count_mapping(value: Any) -> bool:
    return isinstance(value, dict) and all(
        isinstance(key, str)
        and key
        and isinstance(count, int)
        and not isinstance(count, bool)
        and count >= 0
        for key, count in value.items()
    )


def _nested_counts(
    records: Iterable[Mapping[str, Any]],
    *,
    outer: str,
) -> dict[str, dict[str, int]]:
    counts: Counter[tuple[str, str]] = Counter()
    for row in records:
        counts[(str(row[outer]), str(row["logical_role"]))] += 1
    output: dict[str, dict[str, int]] = defaultdict(dict)
    for (outer_value, role), count in sorted(counts.items()):
        output[outer_value][role] = count
    return dict(output)


def _expected_blockers(manifest: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    if manifest.get("build_profile") != "full":
        blockers.append("bounded_smoke_profile")
    if manifest.get("scope_validation_complete") is not True:
        blockers.append("deep_validation_pending")
    return blockers


def _validate_release_equivalence(
    value: Any,
    report: dict[str, Any],
    manifest: Mapping[str, Any],
) -> None:
    required = {
        "schema_version",
        "predecessor_identity",
        "record_count",
        "parent_count",
        "asset_count",
        "asset_bytes",
        "record_content_equivalent",
        "parent_grouping_equivalent",
        "logical_roles_equivalent",
        "asset_bytes_equivalent",
        "parent_mapping_sha256",
        "record_mapping_sha256",
        "asset_set_sha256",
    }
    if not _exact_keys(value, required, report, "release_equivalence"):
        return
    assert isinstance(value, dict)
    predecessor = value["predecessor_identity"]
    if not _exact_keys(
        predecessor,
        {
            "manifest_sha256",
            "build_id",
            "payload_root_sha256",
            "hash_manifest_sha256",
        },
        report,
        "release_equivalence.predecessor_identity",
    ):
        return
    if value.get("schema_version") != RELEASE_EQUIVALENCE_SCHEMA_VERSION:
        _error(report, "release_equivalence schema 错误")
    for key in (
        "record_content_equivalent",
        "parent_grouping_equivalent",
        "logical_roles_equivalent",
        "asset_bytes_equivalent",
    ):
        if value.get(key) is not True:
            _error(report, f"release_equivalence.{key} 必须为 true")
    counts = manifest.get("counts", {})
    for key, manifest_key in (
        ("record_count", "records"),
        ("parent_count", "parents"),
        ("asset_count", "assets"),
        ("asset_bytes", "asset_bytes"),
    ):
        if value.get(key) != counts.get(manifest_key):
            _error(report, f"release_equivalence.{key} 与 manifest 不一致")
    for key in (
        "parent_mapping_sha256",
        "record_mapping_sha256",
        "asset_set_sha256",
    ):
        if not isinstance(value.get(key), str) or _SHA256.fullmatch(value[key]) is None:
            _error(report, f"release_equivalence.{key} 非法")
    if _absolute_strings(value):
        _error(report, "release_equivalence 不得嵌入绝对路径")


def validate_benchmark(root: Path | str, *, deep: bool = False) -> dict[str, Any]:
    """验证 native v1；deep 时每个唯一 image asset 只读取并解码一次。"""

    report: dict[str, Any] = {
        "schema_version": VALIDATION_VERSION,
        "deep": deep,
        "errors": [],
        "warnings": [],
        "checked_records": 0,
        "checked_media_references": 0,
        "checked_unique_assets": 0,
        "hash_files_checked": 0,
        "record_count": 0,
        "parent_count": 0,
        "source_counts": {},
        "role_counts": {},
        "task_counts": {},
        "supervision_kind_counts": {},
        "input_layout_counts": {},
        "output_modality_counts": {},
        "parent_role_counts": {},
        "source_role_counts": {},
        "task_role_counts": {},
        "payload_root_sha256": None,
        "hash_manifest_sha256": None,
    }
    benchmark_root = Path(root).absolute()
    linked_root = first_symlink_component(benchmark_root)
    if linked_root is not None or not benchmark_root.is_dir():
        _error(report, "Benchmark root 必须是无链接的普通目录")
        return report
    try:
        manifest_path = _safe_file(benchmark_root, "manifest.json")
        manifest = read_json(manifest_path)
    except Exception as error:
        _error(report, f"manifest 读取失败：{error}")
        return report
    manifest_fields = {
        "schema_version",
        "canonical_schema_version",
        "build_profile",
        "benchmark_scope",
        "scope_validation_complete",
        "build_id",
        "semantic_config_sha256",
        "payload_root_sha256",
        "hash_manifest_sha256",
        "reference_project",
        "environment_versions",
        "layout",
        "counts",
        "source_audits",
        "formal_acceptance_eligible",
        "formal_acceptance_blockers",
        "source_roots_embedded",
        "external_test_retained",
        "upstream_split_policy",
        "official_upstream_evaluation_reproducible",
    }
    if not _exact_keys(manifest, manifest_fields, report, "manifest"):
        return report
    if (
        manifest.get("schema_version") != MANIFEST_VERSION
        or manifest.get("canonical_schema_version") != CANONICAL_SCHEMA_VERSION
        or manifest.get("benchmark_scope") != BENCHMARK_SCOPE
    ):
        _error(report, "manifest native schema/scope 不匹配")
    if manifest.get("build_profile") not in {"full", "smoke"}:
        _error(report, "manifest build_profile 非法")
    expected_blockers = _expected_blockers(manifest)
    if manifest.get("formal_acceptance_blockers") != expected_blockers:
        _error(report, "manifest formal blockers 与 profile/deep 状态不一致")
    if manifest.get("formal_acceptance_eligible") is not (not expected_blockers):
        _error(report, "manifest formal acceptance flag 与 blockers 不一致")
    if manifest.get("source_roots_embedded") is not False:
        _error(report, "manifest source_roots_embedded 必须为 false")
    if manifest.get("external_test_retained") is not False:
        _error(report, "manifest external_test_retained 必须为 false")
    if manifest.get("reference_project") != HF_DATASETS_REFERENCE:
        _error(report, "manifest reference_project 不匹配")
    if _absolute_strings(manifest):
        _error(report, "manifest 不得嵌入绝对路径")

    layout_fields = {
        "canonical_schema",
        "record_shards",
        "role_to_record_shards",
        "provenance",
        "skips",
        "statistics",
        "assets",
        "source_audits",
        "hashes",
        "build_config",
        "validation",
        "deletion_allowlist",
        "release_equivalence",
    }
    layout = manifest.get("layout")
    if not _exact_keys(layout, layout_fields, report, "manifest.layout"):
        return report
    assert isinstance(layout, dict)
    fixed_paths = {
        "canonical_schema": "schemas/canonical.schema.json",
        "provenance": "metadata/provenance.jsonl",
        "skips": "metadata/skips.jsonl",
        "statistics": "metadata/statistics.json",
        "assets": "metadata/assets.json",
        "source_audits": "metadata/source_audits.json",
        "hashes": "metadata/hashes.json",
        "build_config": "metadata/build_config.json",
        "validation": "metadata/validation.json",
    }
    for key, expected in fixed_paths.items():
        if layout.get(key) != expected:
            _error(report, f"manifest.layout.{key} 必须为 {expected}")

    try:
        schema = read_json(_safe_file(benchmark_root, str(layout["canonical_schema"])))
        build_config = read_json(_safe_file(benchmark_root, str(layout["build_config"])))
        statistics = read_json(_safe_file(benchmark_root, str(layout["statistics"])))
        asset_inventory = read_json(_safe_file(benchmark_root, str(layout["assets"])))
        source_audits = read_json(_safe_file(benchmark_root, str(layout["source_audits"])))
        hashes_path = _safe_file(benchmark_root, str(layout["hashes"]))
        hashes = read_json(hashes_path)
        provenance_path = _safe_file(benchmark_root, str(layout["provenance"]))
        skips_path = _safe_file(benchmark_root, str(layout["skips"]))
    except Exception as error:
        _error(report, f"metadata 读取失败：{error}")
        return report
    if schema != canonical_schema_document():
        _error(report, "canonical schema document 不匹配 native v1")

    enabled_sources = set(RS_GENERALDESC_SOURCES)
    build_fields = {
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
    if _exact_keys(build_config, build_fields, report, "build_config"):
        if build_config.get("schema_version") != BUILD_CONFIG_VERSION:
            _error(report, "build_config schema 错误")
        sources = build_config.get("sources")
        if not isinstance(sources, dict) or set(sources) != RS_GENERALDESC_SOURCES:
            _error(report, "build_config sources 必须精确列出三类 RS source")
        elif any(
            not isinstance(value, dict)
            or set(value) != {"enabled"}
            or not isinstance(value["enabled"], bool)
            for value in sources.values()
        ):
            _error(report, "build_config source 合同非法")
        else:
            enabled_sources = {
                name for name, value in sources.items() if value["enabled"]
            }
            if not enabled_sources:
                _error(report, "build_config 至少启用一个 source")
        if _absolute_strings(build_config):
            _error(report, "build_config 不得嵌入绝对路径")
    semantic_hash = sha256_text(canonical_json(build_config))
    if manifest.get("semantic_config_sha256") != semantic_hash:
        _error(report, "semantic config SHA-256 不匹配")

    hash_fields = {"schema_version", "files", "root_sha256"}
    if not _exact_keys(hashes, hash_fields, report, "hashes"):
        return report
    if hashes.get("schema_version") != HASH_SCHEMA_VERSION:
        _error(report, "hashes schema 错误")
    actual_hash_manifest = sha256_file(hashes_path)
    report["hash_manifest_sha256"] = actual_hash_manifest
    if manifest.get("hash_manifest_sha256") != actual_hash_manifest:
        _error(report, "hash manifest SHA-256 不匹配")
    hash_rows = hashes.get("files")
    if not isinstance(hash_rows, list):
        _error(report, "hashes.files 必须是列表")
        return report
    expected_rows: list[dict[str, Any]] = []
    seen_hash_paths: set[str] = set()
    asset_geometry: dict[str, tuple[int, int, str]] = {}
    forbidden_hash_paths = {
        "manifest.json",
        "metadata/hashes.json",
        "metadata/validation.json",
    }
    for index, row in enumerate(hash_rows):
        if not _exact_keys(row, {"path", "size_bytes", "sha256"}, report, f"hashes.files[{index}]"):
            continue
        assert isinstance(row, dict)
        relative = row.get("path")
        if not isinstance(relative, str) or relative in seen_hash_paths or relative in forbidden_hash_paths:
            _error(report, f"hashes.files[{index}].path 非法或重复")
            continue
        seen_hash_paths.add(relative)
        try:
            path = _safe_file(benchmark_root, relative)
            size = path.stat().st_size
            if deep and relative.startswith("assets/image/"):
                payload = path.read_bytes()
                digest = sha256_bytes(payload)
                try:
                    with Image.open(io.BytesIO(payload)) as image:
                        image.load()
                        asset_geometry[relative] = (image.width, image.height, image.mode)
                except (UnidentifiedImageError, OSError, ValueError) as error:
                    _error(report, f"asset 无法解码：{relative}: {error}")
            else:
                digest = sha256_file(path)
        except Exception as error:
            _error(report, f"hash payload 检查失败：{relative}: {error}")
            continue
        report["hash_files_checked"] += 1
        actual_row = {"path": relative, "size_bytes": size, "sha256": digest}
        expected_rows.append(actual_row)
        if row != actual_row:
            _error(report, f"hash payload identity 不匹配：{relative}")
    actual_payload_files = {
        path.relative_to(benchmark_root).as_posix()
        for path in benchmark_root.rglob("*")
        if path.is_file()
        and path.relative_to(benchmark_root).as_posix() not in forbidden_hash_paths
        and not path.relative_to(benchmark_root).as_posix().startswith(".")
    }
    if seen_hash_paths != actual_payload_files:
        _error(report, "hash manifest 文件集合与 Benchmark payload 不一致")
    expected_rows.sort(key=lambda row: row["path"])
    payload_root = sha256_bytes(canonical_json(expected_rows).encode("utf-8"))
    report["payload_root_sha256"] = payload_root
    if hashes.get("root_sha256") != payload_root or manifest.get("payload_root_sha256") != payload_root:
        _error(report, "payload root SHA-256 不匹配")
    expected_build_id = "build_" + sha256_text(canonical_json([semantic_hash, payload_root]))
    if manifest.get("build_id") != expected_build_id:
        _error(report, "build ID 推导不匹配")

    if not isinstance(asset_inventory, list):
        _error(report, "assets inventory 必须是列表")
        return report
    inventory_by_path: dict[str, dict[str, Any]] = {}
    inventory_by_id: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(asset_inventory):
        if not _exact_keys(
            row,
            {"asset_id", "path", "media_type", "sha256", "size_bytes", "extension"},
            report,
            f"assets[{index}]",
        ):
            continue
        assert isinstance(row, dict)
        relative = row.get("path")
        digest = row.get("sha256")
        asset_id = row.get("asset_id")
        if (
            not isinstance(relative, str)
            or not relative.startswith("assets/image/")
            or row.get("media_type") != "image"
            or not isinstance(digest, str)
            or _SHA256.fullmatch(digest) is None
            or asset_id != f"asset_{digest}"
            or relative in inventory_by_path
            or asset_id in inventory_by_id
        ):
            _error(report, f"assets[{index}] identity 非法或重复")
            continue
        hash_entry = next((item for item in expected_rows if item["path"] == relative), None)
        if hash_entry is None or hash_entry["sha256"] != digest or hash_entry["size_bytes"] != row.get("size_bytes"):
            _error(report, f"asset inventory 与 payload 不一致：{relative}")
        inventory_by_path[relative] = row
        inventory_by_id[str(asset_id)] = row
    if deep:
        report["checked_unique_assets"] = len(asset_geometry)
        if set(asset_geometry) != set(inventory_by_path):
            _error(report, "deep unique asset 集合与 inventory 不一致")

    provenance_ids: set[str] = set()
    try:
        for row in _iter_jsonl(provenance_path):
            if set(row) != {
                "schema_version",
                "source",
                "source_record_id",
                "source_split",
                "details",
                "provenance_id",
            }:
                _error(report, "provenance 字段非法")
                continue
            identity = {
                "schema_version": PROVENANCE_SCHEMA_VERSION,
                "source": row.get("source"),
                "source_record_id": row.get("source_record_id"),
                "source_split": row.get("source_split"),
                "details": row.get("details"),
            }
            expected_id = "provenance_" + sha256_text(canonical_json(identity))
            provenance_id = row.get("provenance_id")
            if row.get("schema_version") != PROVENANCE_SCHEMA_VERSION or provenance_id != expected_id:
                _error(report, "provenance schema/ID 不匹配")
            if not isinstance(provenance_id, str) or provenance_id in provenance_ids:
                _error(report, "provenance ID 重复或非法")
            else:
                provenance_ids.add(provenance_id)
            if row.get("source") not in RS_GENERALDESC_SOURCES or _absolute_strings(row):
                _error(report, "provenance source/path 合同非法")
    except Exception as error:
        _error(report, f"provenance 读取失败：{error}")

    record_shards = layout.get("record_shards")
    if not isinstance(record_shards, list) or not record_shards:
        _error(report, "manifest record_shards 必须非空")
        return report
    records: list[dict[str, Any]] = []
    seen_record_ids: set[str] = set()
    referenced_provenance: set[str] = set()
    shard_roles: dict[str, set[str]] = defaultdict(set)
    for shard_index, shard in enumerate(record_shards):
        if not _exact_keys(shard, {"path", "record_count"}, report, f"record_shards[{shard_index}]"):
            continue
        assert isinstance(shard, dict)
        relative = shard.get("path")
        if not isinstance(relative, str) or not relative.startswith("records/part-"):
            _error(report, "record shard path 非法")
            continue
        try:
            shard_path = _safe_file(benchmark_root, relative)
            rows = list(_iter_jsonl(shard_path))
        except Exception as error:
            _error(report, f"record shard 读取失败：{relative}: {error}")
            continue
        if shard.get("record_count") != len(rows):
            _error(report, f"record shard count 不一致：{relative}")
        for row in rows:
            try:
                validate_canonical_record(row)
            except Exception as error:
                _error(report, f"canonical record 非法：{row.get('record_id')}: {error}")
                continue
            record_identifier = str(row["record_id"])
            if record_identifier in seen_record_ids:
                _error(report, f"record_id 重复：{record_identifier}")
            seen_record_ids.add(record_identifier)
            records.append(row)
            shard_roles[relative].add(str(row["logical_role"]))
            referenced_provenance.update(str(value) for value in row["provenance_ids"])
            for media in row["media"]:
                report["checked_media_references"] += 1
                relative_asset = str(media["path"])
                inventory = inventory_by_path.get(relative_asset)
                if inventory is None or inventory.get("asset_id") != media.get("asset_id"):
                    _error(report, f"media 未绑定 asset inventory：{relative_asset}")
                if deep:
                    geometry = asset_geometry.get(relative_asset)
                    expected_geometry = (media["width"], media["height"], media["mode"])
                    if geometry != expected_geometry:
                        _error(report, f"media geometry 与 asset 不一致：{relative_asset}")
    report["checked_records"] = len(records)
    report["record_count"] = len(records)
    parents = {str(row["parent_id"]) for row in records}
    report["parent_count"] = len(parents)
    if referenced_provenance != provenance_ids:
        _error(report, "record/provenance 引用集合不一致")
    expected_role_shards: dict[str, list[str]] = defaultdict(list)
    for path, roles in shard_roles.items():
        for role in sorted(roles):
            expected_role_shards[role].append(path)
    expected_role_shards = {
        role: paths for role, paths in sorted(expected_role_shards.items())
    }
    if layout.get("role_to_record_shards") != expected_role_shards:
        _error(report, "role_to_record_shards 与 records 不一致")

    parent_roles: dict[str, set[str]] = defaultdict(set)
    for row in records:
        parent_roles[str(row["parent_id"])].add(str(row["logical_role"]))
    if any(len(roles) != 1 for roles in parent_roles.values()):
        _error(report, "parent 跨 external_train/external_val")
    source_counts = dict(sorted(Counter(str(row["source"]) for row in records).items()))
    role_counts = dict(sorted(Counter(str(row["logical_role"]) for row in records).items()))
    task_counts = dict(sorted(Counter(str(row["task_family"]) for row in records).items()))
    supervision_counts = dict(sorted(Counter(str(row["supervision_kind"]) for row in records).items()))
    input_counts = dict(sorted(Counter(str(row["input_layout"]) for row in records).items()))
    output_counts = dict(sorted(Counter(str(row["output_modality"]) for row in records).items()))
    annotation_counts = dict(sorted(Counter(str(row["annotation"]["layer"]) for row in records).items()))
    parent_role_counts = dict(
        sorted(Counter(next(iter(roles)) for roles in parent_roles.values()).items())
    )
    source_role_counts = _nested_counts(records, outer="source")
    task_role_counts = _nested_counts(records, outer="task_family")
    report.update(
        {
            "source_counts": source_counts,
            "role_counts": role_counts,
            "task_counts": task_counts,
            "supervision_kind_counts": supervision_counts,
            "input_layout_counts": input_counts,
            "output_modality_counts": output_counts,
            "parent_role_counts": parent_role_counts,
            "source_role_counts": source_role_counts,
            "task_role_counts": task_role_counts,
        }
    )

    statistics_fields = {
        "schema_version",
        "record_count",
        "parent_count",
        "asset_count",
        "asset_bytes",
        "source_counts",
        "role_counts",
        "task_counts",
        "supervision_kind_counts",
        "input_layout_counts",
        "output_modality_counts",
        "annotation_layer_counts",
        "parent_role_counts",
        "source_role_counts",
        "task_role_counts",
        "external_split",
        "skip_count_total",
        "skip_records_materialized",
        "skip_reason_counts",
    }
    if _exact_keys(statistics, statistics_fields, report, "statistics"):
        if statistics.get("schema_version") != STATISTICS_SCHEMA_VERSION:
            _error(report, "statistics schema 错误")
        comparisons = {
            "record_count": len(records),
            "parent_count": len(parents),
            "asset_count": len(inventory_by_path),
            "asset_bytes": sum(int(row["size_bytes"]) for row in inventory_by_path.values()),
            "source_counts": source_counts,
            "role_counts": role_counts,
            "task_counts": task_counts,
            "supervision_kind_counts": supervision_counts,
            "input_layout_counts": input_counts,
            "output_modality_counts": output_counts,
            "annotation_layer_counts": annotation_counts,
            "parent_role_counts": parent_role_counts,
            "source_role_counts": source_role_counts,
            "task_role_counts": task_role_counts,
        }
        for key, expected in comparisons.items():
            if statistics.get(key) != expected:
                _error(report, f"statistics.{key} 与 canonical 不一致")
        external_split = statistics.get("external_split")
        if not isinstance(external_split, dict):
            _error(report, "statistics.external_split 必须是对象")
        else:
            if (
                external_split.get("schema_version") != EXTERNAL_SPLIT_SCHEMA_VERSION
                or external_split.get("version") != EXTERNAL_SPLIT_VERSION
                or external_split.get("content_dedup_checked") is not True
                or external_split.get("content_fingerprint_error_count") != 0
                or external_split.get("parent_role_counts") != parent_role_counts
                or external_split.get("record_role_counts") != role_counts
                or external_split.get("source_role_counts") != source_role_counts
                or external_split.get("task_role_counts") != task_role_counts
            ):
                _error(report, "statistics.external_split 合同或计数不一致")
    counts = manifest.get("counts")
    if counts != {
        "records": len(records),
        "parents": len(parents),
        "assets": len(inventory_by_path),
        "asset_bytes": sum(int(row["size_bytes"]) for row in inventory_by_path.values()),
    }:
        _error(report, "manifest counts 与 payload 不一致")
    if manifest.get("source_audits") != source_audits:
        _error(report, "manifest/source_audits metadata 不一致")
    if not isinstance(source_audits, dict) or set(source_audits) != enabled_sources:
        _error(report, "source_audits 必须精确包含已启用 source")
    if _absolute_strings(source_audits):
        _error(report, "source_audits 不得嵌入绝对路径")

    skip_count = 0
    try:
        for skip in _iter_jsonl(skips_path):
            skip_count += 1
            if _absolute_strings(skip):
                _error(report, "skips 不得嵌入绝对路径")
    except Exception as error:
        _error(report, f"skips 读取失败：{error}")
    if isinstance(statistics, dict) and statistics.get("skip_records_materialized") != skip_count:
        _error(report, "statistics skip_records_materialized 不匹配")

    release_path = layout.get("release_equivalence")
    if release_path is not None:
        if not isinstance(release_path, str):
            _error(report, "release_equivalence path 必须是字符串或 null")
        else:
            try:
                equivalence = read_json(_safe_file(benchmark_root, release_path))
                _validate_release_equivalence(equivalence, report, manifest)
            except Exception as error:
                _error(report, f"release_equivalence 读取失败：{error}")
    deletion_path = layout.get("deletion_allowlist")
    if deletion_path is not None:
        try:
            _safe_file(benchmark_root, str(deletion_path))
        except Exception as error:
            _error(report, f"deletion_allowlist 读取失败：{error}")
    return report
