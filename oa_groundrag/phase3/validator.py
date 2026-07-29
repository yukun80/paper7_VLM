"""OA-LandslideDesc manifest、hash、schema、几何与 source-hidden 验证。"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image

from .common import (
    canonical_json,
    portable_relative_path,
    read_json,
    read_jsonl,
    sha256_bytes,
    sha256_file,
    sha256_text,
)
from .contracts import (
    BUILD_CONFIG_VERSION,
    CANONICAL_SCHEMA_VERSION,
    EXTERNAL_SPLIT_SCHEMA_VERSION,
    EXTERNAL_SPLIT_VERSION,
    HF_DATASETS_REFERENCE,
    MANIFEST_VERSION,
    PROVENANCE_SCHEMA_VERSION,
    STATISTICS_SCHEMA_VERSION,
    VALIDATION_VERSION,
    InputLayout,
    LogicalRole,
    MediaType,
    OutputModality,
    TaskFamily,
    canonical_schema_document,
    validate_canonical_record,
)
from .errors import ReasonCode


_WINDOWS_ABSOLUTE = re.compile(r"^[A-Za-z]:[\\/]")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _error(report: dict[str, Any], message: str) -> None:
    report["errors"].append(message)


def _exact_keys(
    report: dict[str, Any],
    value: Any,
    expected: set[str],
    *,
    location: str,
) -> bool:
    if not isinstance(value, dict):
        _error(report, f"{location} 必须是对象")
        return False
    if set(value) != expected:
        _error(
            report,
            f"{location} 字段错误 missing={sorted(expected - set(value))} "
            f"unknown={sorted(set(value) - expected)}",
        )
        return False
    return True


def _strict_integer(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _valid_count_mapping(value: Any) -> bool:
    return isinstance(value, dict) and all(
        isinstance(key, str)
        and key
        and not isinstance(count, bool)
        and isinstance(count, int)
        and count >= 0
        for key, count in value.items()
    )


def _valid_nested_count_mapping(value: Any) -> bool:
    return isinstance(value, dict) and all(
        isinstance(key, str)
        and key
        and _valid_count_mapping(counts)
        for key, counts in value.items()
    )


def _parent_role_counts(
    records: list[dict[str, Any]],
) -> dict[str, int]:
    return dict(
        sorted(
            Counter(
                role
                for _, role in {
                    (str(row["parent_id"]), str(row["logical_role"]))
                    for row in records
                }
            ).items()
        )
    )


def _nested_role_counts(
    records: list[dict[str, Any]],
    *,
    outer: str,
) -> dict[str, dict[str, int]]:
    counts: Counter[tuple[str, str]] = Counter()
    for row in records:
        outer_value = (
            str(row["source"])
            if outer == "source"
            else str(row["task_family"])
        )
        counts[(outer_value, str(row["logical_role"]))] += 1
    output: dict[str, dict[str, int]] = defaultdict(dict)
    for (outer_value, role), count in sorted(counts.items()):
        output[outer_value][role] = count
    return dict(output)


def _absolute_strings(value: Any, *, location: str = "$") -> list[str]:
    findings: list[str] = []
    if isinstance(value, str):
        if value.startswith("/") or _WINDOWS_ABSOLUTE.match(value):
            findings.append(f"{location}={value}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            findings.extend(_absolute_strings(child, location=f"{location}[{index}]"))
    elif isinstance(value, dict):
        for key, child in value.items():
            findings.extend(_absolute_strings(child, location=f"{location}.{key}"))
    return findings


def _safe_file(root: Path, relative: str) -> Path:
    pure = portable_relative_path(relative, location="manifest path")
    path = root.joinpath(*pure.parts)
    path.resolve().relative_to(root.resolve())
    return path


def _validate_media(
    report: dict[str, Any],
    root: Path,
    media: Mapping[str, Any],
    *,
    deep: bool,
) -> None:
    try:
        path = _safe_file(root, str(media["path"]))
    except Exception as error:
        _error(report, f"asset path 非法：{error}")
        return
    if not path.is_file():
        _error(report, f"asset 缺失：{media.get('path')}")
        return
    if path.is_symlink() or path.stat().st_nlink != 1:
        _error(report, f"asset 不能是 symlink/hardlink：{media.get('path')}")
    if path.stat().st_size != int(media.get("size_bytes", -1)):
        _error(report, f"asset size 不一致：{media.get('path')}")
    if sha256_file(path) != media.get("sha256"):
        _error(report, f"asset hash 不一致：{media.get('path')}")
    if not deep:
        return
    media_type = media.get("media_type")
    try:
        if media_type == "image":
            with Image.open(path) as image:
                image.load()
                if image.mode != "RGB":
                    _error(report, f"image mode 不是 RGB：{media.get('path')}")
                if list(image.size) != [
                    int(media.get("width", -1)),
                    int(media.get("height", -1)),
                ]:
                    _error(report, f"image size 不一致：{media.get('path')}")
        elif media_type == "mask":
            with Image.open(path) as image:
                array = np.asarray(image.convert("L"))
                values = set(np.unique(array).tolist())
                if not values.issubset({0, 255}):
                    _error(report, f"mask 非法值：{media.get('path')}")
                if list(image.size) != [
                    int(media.get("width", -1)),
                    int(media.get("height", -1)),
                ]:
                    _error(report, f"mask size 不一致：{media.get('path')}")
        elif media_type == "array":
            with np.load(path, allow_pickle=False) as archive:
                if not archive.files:
                    _error(report, f"NPZ 为空：{media.get('path')}")
                for name in archive.files:
                    value = np.asarray(archive[name])
                    if value.dtype.kind == "f" and not np.isfinite(value).all():
                        _error(report, f"NPZ 非有限值：{media.get('path')}:{name}")
                    if "valid" in name:
                        unique = set(np.unique(value).tolist())
                        if not unique.issubset({0, 1}):
                            _error(
                                report,
                                f"NPZ validity 非二值：{media.get('path')}:{name}",
                            )
        else:
            _error(report, f"未知 media_type：{media_type}")
    except Exception as error:
        _error(report, f"asset 深度解码失败 {media.get('path')}：{error}")


def validate_benchmark(root: Path | str, *, deep: bool = False) -> dict[str, Any]:
    input_root = Path(root)
    input_root_is_symlink = input_root.is_symlink()
    benchmark_root = input_root.resolve()
    report: dict[str, Any] = {
        "schema_version": VALIDATION_VERSION,
        "deep": bool(deep),
        "errors": [],
        "warnings": [],
        "checked_records": 0,
        "checked_assets": 0,
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
    }
    if not benchmark_root.is_dir():
        _error(report, "Benchmark root 不存在")
        return report
    if input_root_is_symlink:
        _error(report, "Benchmark root 不能是 symlink")
        return report
    manifest_path = benchmark_root / "manifest.json"
    if not manifest_path.is_file():
        _error(report, "缺少 manifest.json")
        return report
    try:
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
        "oa_split_repartitioned",
        "upstream_split_policy",
        "official_upstream_evaluation_reproducible",
    }
    if not _exact_keys(
        report,
        manifest,
        manifest_fields,
        location="manifest",
    ):
        return report
    if manifest.get("schema_version") != MANIFEST_VERSION:
        _error(report, "manifest schema_version 不受支持")
    if manifest.get("canonical_schema_version") != CANONICAL_SCHEMA_VERSION:
        _error(report, "manifest canonical_schema_version 不受支持")
    if manifest.get("build_profile") not in {"smoke", "full"}:
        _error(report, "manifest build_profile 非法")
    if manifest.get("benchmark_scope") not in {
        "external_train_val",
        "oa_landslidedesc_complete",
    }:
        _error(report, "manifest benchmark_scope 非法")
    if not isinstance(manifest.get("scope_validation_complete"), bool):
        _error(report, "manifest scope_validation_complete 必须是 bool")
    for key in (
        "build_id",
        "semantic_config_sha256",
        "payload_root_sha256",
        "hash_manifest_sha256",
    ):
        value = manifest.get(key)
        prefix = "build_" if key == "build_id" else ""
        digest = value.removeprefix(prefix) if isinstance(value, str) else ""
        if _SHA256.fullmatch(digest) is None:
            _error(report, f"manifest {key} 格式非法")
    if manifest.get("reference_project") != HF_DATASETS_REFERENCE:
        _error(report, "manifest Hugging Face reference revision 不一致")
    environment = manifest.get("environment_versions")
    expected_environment_keys = {
        "python",
        "h5py",
        "numpy",
        "pillow",
        "pyyaml",
        "scipy",
        "torch",
    }
    if not _exact_keys(
        report,
        environment,
        expected_environment_keys,
        location="manifest.environment_versions",
    ) or not all(
        isinstance(value, str) and value for value in environment.values()
    ):
        _error(report, "manifest environment_versions 非严格字符串映射")
    if not _exact_keys(
        report,
        manifest.get("counts"),
        {"records", "parents", "assets", "asset_bytes"},
        location="manifest.counts",
    ):
        return report
    if (
        not isinstance(manifest.get("source_audits"), dict)
        or not all(
            isinstance(key, str) and isinstance(value, dict)
            for key, value in manifest["source_audits"].items()
        )
    ):
        _error(report, "manifest source_audits 必须是对象映射")
    if not isinstance(manifest.get("formal_acceptance_eligible"), bool):
        _error(report, "manifest formal_acceptance_eligible 必须是 bool")
    if not isinstance(manifest.get("formal_acceptance_blockers"), list) or not all(
        isinstance(value, str) and value
        for value in manifest.get("formal_acceptance_blockers", [])
    ):
        _error(report, "manifest formal_acceptance_blockers 必须是字符串列表")
    for key in (
        "source_roots_embedded",
        "external_test_retained",
        "oa_split_repartitioned",
    ):
        if manifest.get(key) is not False:
            _error(report, f"manifest {key} 必须为 false")
    if manifest.get("upstream_split_policy") != (
        "provenance_only_repartitioned_parent_content_component"
    ):
        _error(report, "manifest upstream_split_policy 非法")
    if manifest.get("official_upstream_evaluation_reproducible") is not False:
        _error(
            report,
            "重新划分后不得声明可复现 RSGPT RSIEval/DisasterM3 Bench 官方结果",
        )
    absolute = _absolute_strings(manifest)
    if absolute:
        _error(report, f"manifest 包含绝对路径：{absolute[:5]}")
    layout = manifest.get("layout")
    if not isinstance(layout, dict):
        _error(report, "manifest.layout 必须是对象")
        return report
    required_layout = {
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
    }
    if set(layout) != required_layout:
        _error(report, "manifest.layout 字段不完整或含未知项")
        return report
    try:
        hashes_path = _safe_file(benchmark_root, str(layout["hashes"]))
        hashes = read_json(hashes_path)
    except Exception as error:
        _error(report, f"hash manifest 读取失败：{error}")
        return report
    if not _exact_keys(
        report,
        hashes,
        {"schema_version", "files", "root_sha256"},
        location="hashes",
    ):
        return report
    if hashes.get("schema_version") != "oa_landslidedesc.hashes.v1":
        _error(report, "hashes schema_version 不受支持")
    if sha256_file(hashes_path) != manifest.get("hash_manifest_sha256"):
        _error(report, "hash manifest SHA-256 与 manifest 不一致")
    files = hashes.get("files")
    if not isinstance(files, list):
        _error(report, "hashes.files 必须为列表")
        return report
    seen_paths: set[str] = set()
    for item in files:
        if not isinstance(item, dict) or set(item) != {
            "path",
            "size_bytes",
            "sha256",
        }:
            _error(report, "hash file entry 合同非法")
            continue
        if (
            isinstance(item["size_bytes"], bool)
            or not isinstance(item["size_bytes"], int)
            or item["size_bytes"] < 0
            or not isinstance(item["sha256"], str)
            or _SHA256.fullmatch(item["sha256"]) is None
        ):
            _error(report, f"hash file entry 类型非法：{item}")
            continue
        relative = str(item["path"])
        if relative in seen_paths:
            _error(report, f"hash path 重复：{relative}")
            continue
        seen_paths.add(relative)
        try:
            path = _safe_file(benchmark_root, relative)
        except Exception as error:
            _error(report, f"hash path 非法 {relative}：{error}")
            continue
        if not path.is_file():
            _error(report, f"hash payload 缺失：{relative}")
            continue
        if path.is_symlink() or path.stat().st_nlink != 1:
            _error(report, f"payload 不能是 symlink/hardlink：{relative}")
        if path.stat().st_size != int(item["size_bytes"]):
            _error(report, f"payload size 不符：{relative}")
        if sha256_file(path) != item["sha256"]:
            _error(report, f"payload hash 不符：{relative}")
    expected_root = sha256_bytes(canonical_json(files).encode("utf-8"))
    if hashes.get("root_sha256") != expected_root:
        _error(report, "payload root_sha256 计算不一致")
    if manifest.get("payload_root_sha256") != expected_root:
        _error(report, "manifest payload_root_sha256 不一致")
    expected_build_id = (
        "build_"
        + sha256_text(
            canonical_json(
                [manifest.get("semantic_config_sha256"), expected_root]
            )
        )
    )
    if manifest.get("build_id") != expected_build_id:
        _error(report, "manifest build_id 计算不一致")
    unhashed_allowed = {
        "manifest.json",
        str(layout["hashes"]),
        str(layout["validation"]),
    }
    staging_sentinel = ".oa_landslidedesc_staging"
    if benchmark_root.name.startswith(".") and ".staging-" in benchmark_root.name:
        unhashed_allowed.add(staging_sentinel)
    elif (benchmark_root / staging_sentinel).exists():
        _error(report, "正式输出仍包含 staging ownership sentinel")
    actual_files = {
        path.relative_to(benchmark_root).as_posix()
        for path in benchmark_root.rglob("*")
        if path.is_file()
    }
    unhashed = sorted(actual_files - seen_paths - unhashed_allowed)
    if unhashed:
        _error(report, f"输出树包含未登记 payload 文件：{unhashed[:10]}")

    try:
        schema = read_json(_safe_file(benchmark_root, layout["canonical_schema"]))
        statistics = read_json(_safe_file(benchmark_root, layout["statistics"]))
        asset_inventory = read_json(_safe_file(benchmark_root, layout["assets"]))
        source_audits = read_json(
            _safe_file(benchmark_root, layout["source_audits"])
        )
        provenance_rows = read_jsonl(
            _safe_file(benchmark_root, layout["provenance"])
        )
        skip_rows = read_jsonl(_safe_file(benchmark_root, layout["skips"]))
        build_config = read_json(_safe_file(benchmark_root, layout["build_config"]))
    except Exception as error:
        _error(report, f"metadata 读取失败：{error}")
        return report
    if schema != canonical_schema_document():
        _error(report, "canonical schema 文档与内置版本不一致")
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
    if not _exact_keys(
        report,
        statistics,
        statistics_fields,
        location="statistics",
    ):
        return report
    if statistics.get("schema_version") != STATISTICS_SCHEMA_VERSION:
        _error(report, "statistics schema_version 不受支持")
    sanitized_fields = {
        "schema_version",
        "profile",
        "seed",
        "sources",
        "oa",
        "external_split",
        "text_policy",
        "asset_policy",
        "sharding",
        "workers",
        "limits",
        "deletion_allowlist",
    }
    if not _exact_keys(
        report,
        build_config,
        sanitized_fields,
        location="build_config",
    ):
        return report
    if (
        build_config.get("schema_version") != BUILD_CONFIG_VERSION
        or build_config.get("profile") != manifest.get("build_profile")
    ):
        _error(report, "build_config schema/profile 与 manifest 不一致")
    if sha256_text(canonical_json(build_config)) != manifest.get(
        "semantic_config_sha256"
    ):
        _error(report, "build_config semantic SHA-256 与 manifest 不一致")
    if _exact_keys(
        report,
        build_config.get("sources"),
        {"rsgpt", "mmrs1m", "disasterm3"},
        location="build_config.sources",
    ):
        for source_name, source_config in build_config["sources"].items():
            _exact_keys(
                report,
                source_config,
                {"enabled"},
                location=f"build_config.sources.{source_name}",
            )
    _exact_keys(
        report,
        build_config.get("oa"),
        {
            "enabled",
            "expected_gold_count",
            "gold_annotations_configured",
            "silver_annotations_configured",
        },
        location="build_config.oa",
    )
    if _exact_keys(
        report,
        build_config.get("external_split"),
        {"version", "validation_percent"},
        location="build_config.external_split",
    ):
        external_split_config = build_config["external_split"]
        if (
            external_split_config.get("version") != EXTERNAL_SPLIT_VERSION
            or _strict_integer(
                external_split_config.get("validation_percent")
            )
            is None
            or not 0
            <= external_split_config["validation_percent"]
            <= 50
        ):
            _error(report, "build_config.external_split 合同非法")
    _exact_keys(
        report,
        build_config.get("text_policy"),
        {"version", "forbidden_patterns", "reasoning_forbidden_patterns"},
        location="build_config.text_policy",
    )
    _exact_keys(
        report,
        build_config.get("asset_policy"),
        {
            "image_format",
            "image_quality",
            "jpeg_subsampling",
            "target_size",
            "mask_format",
            "oa_preview_clip",
        },
        location="build_config.asset_policy",
    )
    _exact_keys(
        report,
        build_config.get("sharding"),
        {"records_per_shard"},
        location="build_config.sharding",
    )
    _exact_keys(
        report,
        build_config.get("limits"),
        {
            "max_parents",
            "max_records",
            "max_assets",
            "max_copied_bytes",
            "max_validation_candidates_per_task",
            "per_task",
        },
        location="build_config.limits",
    )
    split_statistics = statistics.get("external_split")
    split_fields = {
        "schema_version",
        "version",
        "seed",
        "validation_percent",
        "content_dedup_checked",
        "assignment_sha256",
        "parent_role_counts",
        "record_role_counts",
        "source_role_counts",
        "task_role_counts",
        "content_component_count",
        "duplicate_content_component_count",
        "content_fingerprint_error_count",
    }
    if _exact_keys(
        report,
        split_statistics,
        split_fields,
        location="statistics.external_split",
    ):
        if (
            split_statistics.get("schema_version")
            != EXTERNAL_SPLIT_SCHEMA_VERSION
            or split_statistics.get("version") != EXTERNAL_SPLIT_VERSION
            or split_statistics.get("seed") != build_config.get("seed")
            or split_statistics.get("validation_percent")
            != build_config.get("external_split", {}).get(
                "validation_percent"
            )
            or not isinstance(
                split_statistics.get("content_dedup_checked"),
                bool,
            )
            or _SHA256.fullmatch(
                str(split_statistics.get("assignment_sha256", ""))
            )
            is None
        ):
            _error(report, "statistics.external_split identity 合同非法")
        for key in ("parent_role_counts", "record_role_counts"):
            if not _valid_count_mapping(split_statistics.get(key)):
                _error(
                    report,
                    f"statistics.external_split.{key} 计数合同非法",
                )
        for key in ("source_role_counts", "task_role_counts"):
            if not _valid_nested_count_mapping(split_statistics.get(key)):
                _error(
                    report,
                    f"statistics.external_split.{key} 计数合同非法",
                )
        component_count = _strict_integer(
            split_statistics.get("content_component_count")
        )
        duplicate_count = _strict_integer(
            split_statistics.get("duplicate_content_component_count")
        )
        fingerprint_errors = _strict_integer(
            split_statistics.get("content_fingerprint_error_count")
        )
        if (
            component_count is None
            or component_count < 0
            or duplicate_count is None
            or duplicate_count < 0
            or duplicate_count > component_count
            or fingerprint_errors is None
            or fingerprint_errors < 0
        ):
            _error(report, "statistics.external_split component 计数非法")
    for name, value in (
        ("schema", schema),
        ("statistics", statistics),
        ("assets", asset_inventory),
        ("source_audits", source_audits),
        ("provenance", provenance_rows),
        ("skips", skip_rows),
        ("build_config", build_config),
    ):
        findings = _absolute_strings(value)
        if findings:
            _error(report, f"{name} 包含绝对路径：{findings[:5]}")
    if (
        not isinstance(source_audits, dict)
        or source_audits != manifest.get("source_audits")
    ):
        _error(report, "source_audits metadata 与 manifest 不一致")
    deletion_allowlist = layout["deletion_allowlist"]
    if deletion_allowlist is not None:
        if manifest.get("build_profile") != "full":
            _error(report, "只有 full build 可以包含 deletion allowlist")
        try:
            allowlist_rows = read_jsonl(
                _safe_file(benchmark_root, str(deletion_allowlist))
            )
        except Exception as error:
            _error(report, f"deletion allowlist 读取失败：{error}")
            allowlist_rows = []
        allowlist_fields = {
            "source",
            "source_record_id",
            "relative_path",
            "size_bytes",
            "reason_code",
            "evidence",
            "action",
        }
        for index, item in enumerate(allowlist_rows):
            if not _exact_keys(
                report,
                item,
                allowlist_fields,
                location=f"deletion_allowlist[{index}]",
            ):
                continue
            if (
                item["action"] != "human_review_only"
                or item["reason_code"]
                not in {
                    ReasonCode.UNUSED_ASSET.value,
                    ReasonCode.ASSET_CORRUPT.value,
                    ReasonCode.ASSET_ZERO_BYTES.value,
                    ReasonCode.DUPLICATE_RECORD.value,
                    ReasonCode.ASSET_MISSING.value,
                }
                or isinstance(item["size_bytes"], bool)
                or not isinstance(item["size_bytes"], int)
                or item["size_bytes"] < 0
            ):
                _error(report, f"deletion_allowlist[{index}] 合同非法")
            try:
                portable_relative_path(
                    str(item["relative_path"]),
                    location=f"deletion_allowlist[{index}].relative_path",
                )
            except Exception as error:
                _error(report, f"deletion_allowlist[{index}] path 非法：{error}")
        findings = _absolute_strings(allowlist_rows)
        if findings:
            _error(report, f"deletion allowlist 包含绝对路径：{findings[:5]}")
    if not isinstance(asset_inventory, list):
        _error(report, "asset inventory 必须是列表")
        return report
    asset_fields = {
        "asset_id",
        "path",
        "media_type",
        "sha256",
        "size_bytes",
        "extension",
    }
    for index, item in enumerate(asset_inventory):
        if not _exact_keys(
            report,
            item,
            asset_fields,
            location=f"assets[{index}]",
        ):
            continue
        if (
            not isinstance(item["asset_id"], str)
            or item["asset_id"] != f"asset_{item['sha256']}"
            or not isinstance(item["sha256"], str)
            or _SHA256.fullmatch(item["sha256"]) is None
            or item["media_type"] not in {value.value for value in MediaType}
            or isinstance(item["size_bytes"], bool)
            or not isinstance(item["size_bytes"], int)
            or item["size_bytes"] <= 0
        ):
            _error(report, f"assets[{index}] 类型/hash 合同非法")
        try:
            _safe_file(benchmark_root, str(item["path"]))
        except Exception as error:
            _error(report, f"assets[{index}] path 非法：{error}")
    provenance_fields = {
        "schema_version",
        "source",
        "source_record_id",
        "source_split",
        "details",
        "provenance_id",
    }
    for index, item in enumerate(provenance_rows):
        if not _exact_keys(
            report,
            item,
            provenance_fields,
            location=f"provenance[{index}]",
        ):
            continue
        identity = {key: item[key] for key in provenance_fields - {"provenance_id"}}
        expected_id = f"provenance_{sha256_text(canonical_json(identity))}"
        if (
            item["schema_version"] != PROVENANCE_SCHEMA_VERSION
            or item["provenance_id"] != expected_id
            or not isinstance(item["details"], dict)
            or not all(
                isinstance(item[key], str) and item[key]
                for key in ("source", "source_record_id", "source_split")
            )
        ):
            _error(report, f"provenance[{index}] identity/schema 非法")
    skip_fields = {
        "source",
        "source_record_id",
        "source_split",
        "task_family",
        "reason_code",
        "evidence",
    }
    for index, item in enumerate(skip_rows):
        if not _exact_keys(
            report,
            item,
            skip_fields,
            location=f"skips[{index}]",
        ):
            continue
        try:
            ReasonCode(item["reason_code"])
        except (TypeError, ValueError):
            _error(report, f"skips[{index}].reason_code 非法")
        if (
            not isinstance(item["evidence"], dict)
            or not all(
                isinstance(item[key], str) and item[key]
                for key in (
                    "source",
                    "source_record_id",
                    "source_split",
                    "task_family",
                )
            )
        ):
            _error(report, f"skips[{index}] 字段类型非法")
    inventory_by_id = {
        str(item.get("asset_id")): item
        for item in asset_inventory
        if isinstance(item, dict)
    }
    if len(inventory_by_id) != len(asset_inventory):
        _error(report, "asset inventory asset_id 缺失或重复")
    provenance_ids = {
        str(item.get("provenance_id"))
        for item in provenance_rows
        if isinstance(item, dict)
    }
    provenance_by_id = {
        str(item.get("provenance_id")): item
        for item in provenance_rows
        if isinstance(item, dict)
    }
    if len(provenance_ids) != len(provenance_rows):
        _error(report, "provenance_id 缺失或重复")

    records: list[dict[str, Any]] = []
    shard_rows: dict[str, list[dict[str, Any]]] = {}
    record_shards = layout["record_shards"]
    if not isinstance(record_shards, list):
        _error(report, "record_shards 必须是列表")
        return report
    for shard in record_shards:
        if not isinstance(shard, dict) or set(shard) != {"path", "record_count"}:
            _error(report, "record shard entry 合同非法")
            continue
        if (
            not isinstance(shard["path"], str)
            or not shard["path"]
            or isinstance(shard["record_count"], bool)
            or not isinstance(shard["record_count"], int)
            or shard["record_count"] <= 0
        ):
            _error(report, "record shard entry 类型非法")
            continue
        try:
            shard_path = str(shard["path"])
            if shard_path in shard_rows:
                _error(report, f"record shard path 重复：{shard_path}")
                continue
            rows = read_jsonl(_safe_file(benchmark_root, shard_path))
        except Exception as error:
            _error(report, f"record shard 读取失败：{error}")
            continue
        if len(rows) != shard["record_count"]:
            _error(report, f"record shard count 不一致：{shard['path']}")
        shard_rows[shard_path] = rows
        records.extend(rows)
    expected_role_shards: dict[str, list[str]] = defaultdict(list)
    for shard_path, rows in shard_rows.items():
        for role in sorted({str(row.get("logical_role")) for row in rows}):
            expected_role_shards[role].append(shard_path)
    actual_role_shards = layout["role_to_record_shards"]
    if not isinstance(actual_role_shards, dict) or actual_role_shards != dict(
        sorted(expected_role_shards.items())
    ):
        _error(report, "manifest role_to_record_shards 与实际分片不一致")
    expected_record_order = sorted(
        records,
        key=lambda row: (
            str(row.get("logical_role")),
            str(row.get("source")),
            str(row.get("parent_id")),
            str(row.get("task_family")),
            str(row.get("record_id")),
        ),
    )
    if records != expected_record_order:
        _error(report, "canonical records 未按固定构建顺序分片")
    seen_records: set[str] = set()
    external_parent_roles: dict[str, set[str]] = defaultdict(set)
    oa_parent_roles: dict[str, set[str]] = defaultdict(set)
    external_image_roles: dict[str, set[str]] = defaultdict(set)
    used_assets: set[str] = set()
    used_provenance: set[str] = set()
    for index, record in enumerate(records):
        try:
            validate_canonical_record(record)
        except Exception as error:
            _error(report, f"record[{index}] schema 失败：{error}")
            continue
        record_id = str(record["record_id"])
        if record_id in seen_records:
            _error(report, f"record_id 重复：{record_id}")
        seen_records.add(record_id)
        if any(key in record for key in ("messages", "qwen", "conversations")):
            _error(report, f"canonical 被 Qwen 字段污染：{record_id}")
        if not set(record["provenance_ids"]).issubset(provenance_ids):
            _error(report, f"record provenance 引用缺失：{record_id}")
        used_provenance.update(str(value) for value in record["provenance_ids"])
        logical_role = str(record["logical_role"])
        if logical_role in {
            LogicalRole.EXTERNAL_TRAIN.value,
            LogicalRole.EXTERNAL_VAL.value,
        }:
            external_parent_roles[str(record["parent_id"])].add(logical_role)
            split_details = [
                provenance_by_id[provenance_id]["details"]
                for provenance_id in record["provenance_ids"]
                if provenance_id in provenance_by_id
                and isinstance(
                    provenance_by_id[provenance_id].get("details"),
                    dict,
                )
                and provenance_by_id[provenance_id]["details"].get("type")
                == "external_parent_split"
            ]
            if len(split_details) != 1:
                _error(
                    report,
                    f"external record 缺少唯一 split provenance：{record_id}",
                )
            else:
                split_detail = split_details[0]
                expected_split_detail_fields = {
                    "type",
                    "version",
                    "seed",
                    "validation_percent",
                    "logical_role",
                    "content_component_id",
                    "content_dedup_checked",
                    "assignment_sha256",
                }
                if (
                    set(split_detail) != expected_split_detail_fields
                    or split_detail.get("version") != EXTERNAL_SPLIT_VERSION
                    or split_detail.get("seed") != build_config.get("seed")
                    or split_detail.get("validation_percent")
                    != build_config.get("external_split", {}).get(
                        "validation_percent"
                    )
                    or split_detail.get("logical_role") != logical_role
                    or not isinstance(
                        split_detail.get("content_component_id"),
                        str,
                    )
                    or not split_detail["content_component_id"].startswith(
                        "component_"
                    )
                    or split_detail.get("content_dedup_checked") is not True
                    or split_detail.get("assignment_sha256")
                    != (
                        split_statistics.get("assignment_sha256")
                        if isinstance(split_statistics, dict)
                        else None
                    )
                ):
                    _error(
                        report,
                        f"external split provenance 非法：{record_id}",
                    )
        elif logical_role.startswith("oa_"):
            oa_parent_roles[str(record["parent_id"])].add(logical_role)
        by_role = {str(media["role"]): media for media in record["media"]}
        for media in record["media"]:
            asset_id = str(media["asset_id"])
            used_assets.add(asset_id)
            inventory = inventory_by_id.get(asset_id)
            if inventory is None:
                _error(report, f"record 引用未登记 asset：{asset_id}")
            else:
                for key in ("path", "sha256", "size_bytes", "media_type"):
                    if media.get(key) != inventory.get(key):
                        _error(report, f"media/inventory {key} 不一致：{asset_id}")
            _validate_media(report, benchmark_root, media, deep=deep)
            report["checked_assets"] += 1
            if (
                logical_role
                in {
                    LogicalRole.EXTERNAL_TRAIN.value,
                    LogicalRole.EXTERNAL_VAL.value,
                }
                and media.get("media_type") == MediaType.IMAGE.value
                and isinstance(media.get("sha256"), str)
            ):
                external_image_roles[str(media["sha256"])].add(logical_role)
        target = record["target"]
        if target["type"] == "bbox":
            for bbox in target.get("boxes", []):
                canonical = bbox.get("canonical_xyxy_norm")
                if (
                    not isinstance(canonical, list)
                    or len(canonical) != 4
                    or min(canonical) < 0
                    or max(canonical) > 1
                    or canonical[2] <= canonical[0]
                    or canonical[3] <= canonical[1]
                ):
                    _error(report, f"canonical bbox 非法：{record_id}")
        elif target["type"] == "mask":
            image = by_role.get(str(target.get("image_role")))
            mask = by_role.get(str(target.get("mask_role")))
            if image is None or mask is None:
                _error(report, f"mask target media 缺失：{record_id}")
            elif (image.get("width"), image.get("height")) != (
                mask.get("width"),
                mask.get("height"),
            ):
                _error(report, f"mask/image geometry 不一致：{record_id}")
            elif deep and not target.get("empty_allowed", False):
                path = _safe_file(benchmark_root, str(mask["path"]))
                with Image.open(path) as image_handle:
                    if not np.any(np.asarray(image_handle.convert("L"))):
                        _error(report, f"非空 mask target 实际为空：{record_id}")
        report["checked_records"] += 1
    orphan_provenance = sorted(provenance_ids - used_provenance)
    if orphan_provenance:
        _error(report, f"存在未引用 provenance：{orphan_provenance[:10]}")
    external_conflicts = {
        parent: sorted(roles)
        for parent, roles in external_parent_roles.items()
        if len(roles) > 1
    }
    if external_conflicts:
        _error(
            report,
            "external parent 跨 train/val："
            f"{list(external_conflicts.items())[:5]}",
        )
    content_conflicts = {
        digest: sorted(roles)
        for digest, roles in external_image_roles.items()
        if len(roles) > 1
    }
    if content_conflicts:
        _error(
            report,
            "相同规范图像 hash 跨 external train/val："
            f"{list(content_conflicts.items())[:5]}",
        )
    conflicts = {
        parent: sorted(roles)
        for parent, roles in oa_parent_roles.items()
        if len(
            {
                "train" if role == "oa_train" else role.removeprefix("oa_")
                for role in roles
            }
        )
        > 1
    }
    if conflicts:
        _error(report, f"OA parent 跨 split：{list(conflicts.items())[:5]}")
    orphan_assets = sorted(set(inventory_by_id) - used_assets)
    if orphan_assets:
        _error(report, f"存在未引用 canonical assets：{orphan_assets[:10]}")
    if _strict_integer(statistics.get("record_count")) != len(records):
        _error(report, "statistics record_count 不一致")
    if _strict_integer(statistics.get("parent_count")) != len(
        {record["parent_id"] for record in records}
    ):
        _error(report, "statistics parent_count 不一致")
    if _strict_integer(statistics.get("asset_count")) != len(asset_inventory):
        _error(report, "statistics asset_count 不一致")
    actual_source_counts = dict(
        sorted(Counter(record["source"] for record in records).items())
    )
    actual_role_counts = dict(
        sorted(Counter(record["logical_role"] for record in records).items())
    )
    actual_task_counts = dict(
        sorted(Counter(record["task_family"] for record in records).items())
    )
    actual_supervision_kind_counts = dict(
        sorted(Counter(record["supervision_kind"] for record in records).items())
    )
    actual_input_layout_counts = dict(
        sorted(Counter(record["input_layout"] for record in records).items())
    )
    actual_output_modality_counts = dict(
        sorted(Counter(record["output_modality"] for record in records).items())
    )
    actual_layer_counts = dict(
        sorted(
            Counter(record["annotation"]["layer"] for record in records).items()
        )
    )
    actual_parent_role_counts = _parent_role_counts(records)
    actual_source_role_counts = _nested_role_counts(records, outer="source")
    actual_task_role_counts = _nested_role_counts(records, outer="task")
    for key, actual in (
        ("source_counts", actual_source_counts),
        ("role_counts", actual_role_counts),
        ("task_counts", actual_task_counts),
        ("supervision_kind_counts", actual_supervision_kind_counts),
        ("input_layout_counts", actual_input_layout_counts),
        ("output_modality_counts", actual_output_modality_counts),
        ("annotation_layer_counts", actual_layer_counts),
        ("parent_role_counts", actual_parent_role_counts),
        ("source_role_counts", actual_source_role_counts),
        ("task_role_counts", actual_task_role_counts),
    ):
        if statistics.get(key) != actual:
            _error(report, f"statistics {key} 不一致")
    external_records = [
        record
        for record in records
        if record["logical_role"]
        in {
            LogicalRole.EXTERNAL_TRAIN.value,
            LogicalRole.EXTERNAL_VAL.value,
        }
    ]
    actual_external_parent_roles = _parent_role_counts(external_records)
    actual_external_record_roles = dict(
        sorted(
            Counter(
                str(record["logical_role"]) for record in external_records
            ).items()
        )
    )
    actual_external_source_roles = _nested_role_counts(
        external_records,
        outer="source",
    )
    actual_external_task_roles = _nested_role_counts(
        external_records,
        outer="task",
    )
    if isinstance(split_statistics, dict):
        for key, actual in (
            ("parent_role_counts", actual_external_parent_roles),
            ("record_role_counts", actual_external_record_roles),
            ("source_role_counts", actual_external_source_roles),
            ("task_role_counts", actual_external_task_roles),
        ):
            if split_statistics.get(key) != actual:
                _error(
                    report,
                    f"statistics.external_split.{key} 与 records 不一致",
                )

    benchmark_scope = manifest.get("benchmark_scope")
    configured_sources = {
        source
        for source, source_config in build_config.get("sources", {}).items()
        if isinstance(source_config, dict)
        and source_config.get("enabled") is True
    }
    if benchmark_scope == "external_train_val":
        if build_config.get("oa") != {
            "enabled": False,
            "expected_gold_count": 0,
            "gold_annotations_configured": False,
            "silver_annotations_configured": False,
        }:
            _error(report, "external_train_val scope 必须完全禁用 OA")
        if (
            not isinstance(split_statistics, dict)
            or split_statistics.get("content_dedup_checked") is not True
        ):
            _error(report, "external_train_val 必须完成规范图像内容去重检查")
        if any(record["source"] == "oa" for record in records):
            _error(report, "external_train_val scope 包含 OA record")
        invalid_roles = sorted(
            {
                str(record["logical_role"])
                for record in records
                if record["logical_role"]
                not in {
                    LogicalRole.EXTERNAL_TRAIN.value,
                    LogicalRole.EXTERNAL_VAL.value,
                }
            }
        )
        if invalid_roles:
            _error(
                report,
                f"external_train_val scope 含非法角色：{invalid_roles}",
            )
        adopted_tasks = {
            "rsgpt": {
                TaskFamily.GLOBAL_CAPTION.value,
                TaskFamily.VISUAL_QA.value,
                TaskFamily.SCENE_UNDERSTANDING.value,
                TaskFamily.OBJECT_COUNT.value,
            },
            "mmrs1m": {
                TaskFamily.GLOBAL_CAPTION.value,
                TaskFamily.VISUAL_QA.value,
                TaskFamily.BBOX_REGION_CAPTION.value,
            },
            "disasterm3": {
                TaskFamily.SCENE_UNDERSTANDING.value,
                TaskFamily.OBJECT_COUNT.value,
                TaskFamily.SPATIAL_RELATION.value,
                TaskFamily.VISIBLE_CHANGE_REPORT.value,
            },
        }
        for record in external_records:
            source = str(record["source"])
            task = str(record["task_family"])
            if task not in adopted_tasks.get(source, set()):
                _error(
                    report,
                    f"external adoption matrix 非法：{source}/{task}",
                )
            if record["output_modality"] != OutputModality.TEXT.value:
                _error(report, f"external record 非 text 输出：{record['record_id']}")
            if (
                record["target"]["type"] == "mask"
                or record["input_layout"] == InputLayout.MASK_GROUNDED.value
                or any(
                    media["media_type"] == MediaType.MASK.value
                    for media in record["media"]
                )
            ):
                _error(
                    report,
                    f"external description_multitask 不接受 pixel mask："
                    f"{record['record_id']}",
                )
        if set(actual_source_counts) != configured_sources:
            _error(
                report,
                "external_train_val 实际 source 与启用 source 不一致",
            )
        if set(source_audits) != configured_sources:
            _error(
                report,
                "external_train_val source_audits 与启用 source 不一致",
            )
        validation_percent = build_config.get("external_split", {}).get(
            "validation_percent"
        )
        if (
            isinstance(validation_percent, int)
            and not isinstance(validation_percent, bool)
            and validation_percent > 0
        ):
            for source in sorted(configured_sources):
                source_parents = {
                    str(record["parent_id"])
                    for record in external_records
                    if record["source"] == source
                }
                source_roles = {
                    str(record["logical_role"])
                    for record in external_records
                    if record["source"] == source
                }
                if len(source_parents) >= 2 and source_roles != {
                    LogicalRole.EXTERNAL_TRAIN.value,
                    LogicalRole.EXTERNAL_VAL.value,
                }:
                    _error(
                        report,
                        f"{source}: parents 足够但未同时覆盖 external_train/val",
                    )
            minimum_task_parents = (100 + validation_percent - 1) // (
                validation_percent
            )
            source_tasks = sorted(
                {
                    (str(record["source"]), str(record["task_family"]))
                    for record in external_records
                }
            )
            for source, task in source_tasks:
                task_rows = [
                    record
                    for record in external_records
                    if record["source"] == source
                    and record["task_family"] == task
                ]
                task_parents = {
                    str(record["parent_id"]) for record in task_rows
                }
                task_roles = {
                    str(record["logical_role"]) for record in task_rows
                }
                if (
                    len(task_parents) >= minimum_task_parents
                    and task_roles
                    != {
                        LogicalRole.EXTERNAL_TRAIN.value,
                        LogicalRole.EXTERNAL_VAL.value,
                    }
                ):
                    _error(
                        report,
                        f"{source}/{task}: parents 足够但任务未覆盖 "
                        "external_train/val",
                    )
        blockers = manifest.get("formal_acceptance_blockers", [])
        if manifest.get("formal_acceptance_eligible") is not False:
            _error(report, "external_train_val 不得声明完整 Phase 2 formal eligible")
        if "oa_component_disabled" not in blockers:
            _error(report, "external_train_val 必须记录 oa_component_disabled")
    elif benchmark_scope == "oa_landslidedesc_complete":
        if build_config.get("oa", {}).get("enabled") is not True:
            _error(report, "完整 OA-LandslideDesc scope 必须启用 OA")
    actual_asset_bytes = sum(
        int(item["size_bytes"])
        for item in asset_inventory
        if isinstance(item, dict)
        and isinstance(item.get("size_bytes"), int)
        and not isinstance(item.get("size_bytes"), bool)
    )
    if _strict_integer(statistics.get("asset_bytes")) != actual_asset_bytes:
        _error(report, "statistics asset_bytes 不一致")
    if _strict_integer(statistics.get("skip_records_materialized")) != len(skip_rows):
        _error(report, "statistics skip_records_materialized 不一致")
    total_skip_count = _strict_integer(statistics.get("skip_count_total"))
    if total_skip_count is None or total_skip_count < len(skip_rows):
        _error(report, "statistics skip_count_total 小于物化 skip 数")
    materialized_skip_counts = Counter(
        str(item.get("reason_code"))
        for item in skip_rows
        if isinstance(item, dict)
    )
    total_skip_counts = statistics.get("skip_reason_counts")
    if not isinstance(total_skip_counts, dict) or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in total_skip_counts.values()
    ):
        _error(report, "statistics skip_reason_counts 类型非法")
    else:
        if sum(total_skip_counts.values()) != total_skip_count:
            _error(report, "statistics skip_reason_counts 总数不一致")
        for reason, count in materialized_skip_counts.items():
            total_reason_count = _strict_integer(total_skip_counts.get(reason))
            if total_reason_count is None or total_reason_count < count:
                _error(report, f"statistics skip reason {reason} 小于物化数")
    if _strict_integer(manifest.get("counts", {}).get("records")) != len(records):
        _error(report, "manifest record count 不一致")
    if _strict_integer(manifest.get("counts", {}).get("assets")) != len(asset_inventory):
        _error(report, "manifest asset count 不一致")
    if _strict_integer(manifest.get("counts", {}).get("parents")) != len(
        {record["parent_id"] for record in records}
    ):
        _error(report, "manifest parent count 不一致")
    if _strict_integer(
        manifest.get("counts", {}).get("asset_bytes")
    ) != actual_asset_bytes:
        _error(report, "manifest asset bytes 不一致")
    if manifest.get("formal_acceptance_eligible") and manifest.get(
        "formal_acceptance_blockers"
    ):
        _error(report, "formal eligible 与 blockers 矛盾")
    if (
        not manifest.get("formal_acceptance_eligible")
        and not manifest.get("formal_acceptance_blockers")
    ):
        _error(report, "formal ineligible 必须给出 blocker")
    if manifest.get("build_profile") != "full" and manifest.get(
        "formal_acceptance_eligible"
    ):
        _error(report, "非 full build 不得 formal eligible")
    deep_pending = "deep_validation_pending" in manifest.get(
        "formal_acceptance_blockers",
        [],
    )
    if manifest.get("scope_validation_complete") is deep_pending:
        _error(
            report,
            "scope_validation_complete 与 deep_validation_pending 矛盾",
        )
    validation_path = _safe_file(benchmark_root, str(layout["validation"]))
    if validation_path.exists():
        try:
            saved_validation = read_json(validation_path)
        except Exception as error:
            _error(report, f"保存的 validation summary 读取失败：{error}")
        else:
            validation_fields = {
                "schema_version",
                "deep",
                "errors",
                "warnings",
                "checked_records",
                "checked_assets",
                "record_count",
                "parent_count",
                "source_counts",
                "role_counts",
                "task_counts",
                "supervision_kind_counts",
                "input_layout_counts",
                "output_modality_counts",
                "parent_role_counts",
                "source_role_counts",
                "task_role_counts",
                "payload_root_sha256",
            }
            if _exact_keys(
                report,
                saved_validation,
                validation_fields,
                location="validation_summary",
            ):
                if (
                    saved_validation["schema_version"] != VALIDATION_VERSION
                    or saved_validation["deep"] is not True
                    or saved_validation["errors"] != []
                    or saved_validation["warnings"] != []
                    or saved_validation["checked_records"] != len(records)
                    or saved_validation["checked_assets"]
                    != sum(len(record["media"]) for record in records)
                    or saved_validation["record_count"] != len(records)
                    or saved_validation["parent_count"]
                    != len({record["parent_id"] for record in records})
                    or saved_validation["source_counts"] != actual_source_counts
                    or saved_validation["role_counts"] != actual_role_counts
                    or saved_validation["task_counts"] != actual_task_counts
                    or saved_validation["supervision_kind_counts"]
                    != actual_supervision_kind_counts
                    or saved_validation["input_layout_counts"]
                    != actual_input_layout_counts
                    or saved_validation["output_modality_counts"]
                    != actual_output_modality_counts
                    or saved_validation["parent_role_counts"]
                    != actual_parent_role_counts
                    or saved_validation["source_role_counts"]
                    != actual_source_role_counts
                    or saved_validation["task_role_counts"]
                    != actual_task_role_counts
                    or saved_validation["payload_root_sha256"] != expected_root
                ):
                    _error(report, "保存的 validation summary 与当前产物不一致")

    for path in benchmark_root.rglob("*"):
        if path.is_symlink():
            _error(report, f"输出树包含 symlink：{path.relative_to(benchmark_root)}")
        elif path.is_file() and path.stat().st_nlink != 1:
            _error(report, f"输出树包含 hardlink：{path.relative_to(benchmark_root)}")

    if deep and not report["errors"]:
        try:
            from .dataset import OALandslideDescDataset

            dataset = OALandslideDescDataset(benchmark_root, load_assets=True)
            for index in range(len(dataset)):
                dataset[index]
        except Exception as error:
            _error(report, f"source-hidden Dataset API 遍历失败：{error}")

    report["record_count"] = len(records)
    report["parent_count"] = len({record["parent_id"] for record in records})
    report["source_counts"] = dict(
        sorted(Counter(record["source"] for record in records).items())
    )
    report["role_counts"] = dict(
        sorted(Counter(record["logical_role"] for record in records).items())
    )
    report["task_counts"] = dict(
        sorted(Counter(record["task_family"] for record in records).items())
    )
    report["supervision_kind_counts"] = actual_supervision_kind_counts
    report["input_layout_counts"] = actual_input_layout_counts
    report["output_modality_counts"] = actual_output_modality_counts
    report["parent_role_counts"] = actual_parent_role_counts
    report["source_role_counts"] = actual_source_role_counts
    report["task_role_counts"] = actual_task_role_counts
    report["payload_root_sha256"] = expected_root
    return report
