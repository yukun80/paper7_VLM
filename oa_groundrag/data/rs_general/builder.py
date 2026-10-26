"""统一 audit/build 流水线、确定性分片、staging 与原子发布。"""

from __future__ import annotations

import shutil
import sys
import uuid
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .adapters import (
    DisasterM3Adapter,
    MMRS1MAdapter,
    RSGPTAdapter,
)
from .assets import AssetStore, normalize_bbox_target
from .io import (
    atomic_write_json,
    atomic_write_jsonl,
    canonical_json,
    sha256_bytes,
    sha256_file,
    sha256_text,
)
from .config import BuildConfig
from .contracts import (
    BENCHMARK_SCOPE,
    CANONICAL_SCHEMA_VERSION,
    HASH_SCHEMA_VERSION,
    HF_DATASETS_REFERENCE,
    MANIFEST_VERSION,
    PROVENANCE_SCHEMA_VERSION,
    STATISTICS_SCHEMA_VERSION,
    AdapterResult,
    LogicalRole,
    SkipRecord,
    SourceExample,
    canonical_schema_document,
    parent_id,
    record_hash,
    record_id,
    validate_canonical_record,
    validate_role_layer,
)
from .errors import AssetError, BuildError, ReasonCode
from .splitter import assign_external_splits


AUDIT_SCHEMA_VERSION = "rs_generaldesc.audit.v1"
STAGING_SENTINEL = ".rs_generaldesc_staging"


def _enabled_adapters(config: BuildConfig) -> list[Any]:
    adapters: list[Any] = []
    if config.sources["rsgpt"].enabled:
        adapters.append(RSGPTAdapter(config))
    if config.sources["mmrs1m"].enabled:
        adapters.append(MMRS1MAdapter(config))
    if config.sources["disasterm3"].enabled:
        adapters.append(DisasterM3Adapter(config))
    return adapters


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _preflight_paths(config: BuildConfig, *, for_build: bool) -> None:
    roots: list[Path] = []
    for name, source in config.sources.items():
        if not source.enabled:
            continue
        if not source.root.is_dir():
            raise BuildError(
                ReasonCode.ASSET_MISSING,
                f"{name}: source root 不存在",
            )
        if source.root.is_symlink():
            raise BuildError(
                ReasonCode.SOURCE_SYMLINK,
                f"{name}: source root 不能是 symlink",
            )
        roots.append(source.root)
    if any(_inside(config.output_root, root) for root in roots):
        raise BuildError(
            ReasonCode.PATH_ESCAPE,
            "output_root 不能位于任一只读 source root 内",
        )
    if for_build and (
        config.output_root.exists() or config.output_root.is_symlink()
    ):
        raise BuildError(
            ReasonCode.OUTPUT_EXISTS,
            f"拒绝覆盖已有 output_root：{config.output_root}",
        )


def _scan(
    config: BuildConfig,
    *,
    deep: bool,
    for_build: bool,
) -> tuple[list[AdapterResult], dict[str, Any]]:
    _preflight_paths(config, for_build=False)
    adapters = _enabled_adapters(config)
    if config.workers == 1 or len(adapters) <= 1:
        results = [
            adapter.scan(deep=deep, for_build=for_build)
            for adapter in adapters
        ]
    else:
        with ThreadPoolExecutor(
            max_workers=min(config.workers, len(adapters)),
            thread_name_prefix="rs-generaldesc-audit",
        ) as executor:
            futures = [
                executor.submit(
                    adapter.scan,
                    deep=deep,
                    for_build=for_build,
                )
                for adapter in adapters
            ]
            results = [future.result() for future in futures]
    ordered = sorted(results, key=lambda result: result.source)
    split_summary = assign_external_splits(
        ordered,
        config,
        check_content=deep or for_build,
    )
    return ordered, split_summary


def _environment_versions() -> dict[str, str]:
    packages = ("h5py", "numpy", "Pillow", "PyYAML", "scipy", "torch")
    versions: dict[str, str] = {
        "python": ".".join(str(value) for value in sys.version_info[:3]),
    }
    for package in packages:
        try:
            versions[package.lower()] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package.lower()] = "not_installed"
    return versions


def audit_sources(config: BuildConfig, *, deep: bool = False) -> dict[str, Any]:
    results, split_summary = _scan(config, deep=deep, for_build=False)
    source_reports = {
        result.source: dict(result.audit)
        for result in sorted(results, key=lambda item: item.source)
    }
    skip_counter = Counter(
        skip.reason_code.value for result in results for skip in result.skips
    )
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "profile": config.profile,
        "semantic_config_sha256": config.semantic_hash,
        "deep": deep,
        "sources": source_reports,
        "external_split": split_summary,
        "totals": {
            "selected_examples": sum(
                int(result.audit.get("selected_examples", len(result.examples)))
                for result in results
            ),
            "candidate_examples": sum(
                int(result.audit.get("full_candidate_examples", len(result.examples)))
                for result in results
            ),
            "skip_count": sum(len(result.skips) for result in results),
            "skip_reason_counts": dict(sorted(skip_counter.items())),
            "role_counts": dict(
                sorted(
                    Counter(
                        example.logical_role.value
                        for result in results
                        for example in result.examples
                    ).items()
                )
            ),
        },
        "formal_acceptance_eligible": False,
        "formal_acceptance_blockers": ["audit_only"],
        "reference_project": dict(HF_DATASETS_REFERENCE),
    }


def _apply_global_limits(
    examples: Sequence[SourceExample],
    config: BuildConfig,
) -> list[SourceExample]:
    groups: dict[tuple[str, str, str], deque[SourceExample]] = defaultdict(deque)
    for example in sorted(
        examples,
        key=lambda row: (
            row.source,
            row.task_family.value,
            row.parent_key,
            row.source_record_id,
        ),
    ):
        groups[
            (
                example.source,
                example.task_family.value,
                example.logical_role.value,
            )
        ].append(example)
    ordered_keys = sorted(groups)
    selected: list[SourceExample] = []
    parents: set[tuple[str, str]] = set()
    while True:
        progressed = False
        for key in ordered_keys:
            if not groups[key]:
                continue
            candidate = groups[key].popleft()
            parent = (candidate.source, candidate.parent_key)
            if (
                config.limits.max_parents is not None
                and parent not in parents
                and len(parents) >= config.limits.max_parents
            ):
                continue
            selected.append(candidate)
            parents.add(parent)
            progressed = True
            if (
                config.limits.max_records is not None
                and len(selected) >= config.limits.max_records
            ):
                return selected
        if not progressed:
            return selected


def _provenance_row(
    example: SourceExample,
    item: Mapping[str, Any],
) -> dict[str, Any]:
    identity = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "source": example.source,
        "source_record_id": example.source_record_id,
        "source_split": example.source_split,
        "details": dict(item),
    }
    provenance_id = f"provenance_{sha256_text(canonical_json(identity))}"
    return {**identity, "provenance_id": provenance_id}


def _canonical_from_example(
    example: SourceExample,
    *,
    store: AssetStore,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    validate_role_layer(
        example.logical_role,
        example.annotation_layer,
        example.review_status,
    )
    media, geometries, transforms = store.materialize(example.assets)
    target = normalize_bbox_target(example.target, geometries)
    provenance_rows = [
        _provenance_row(example, item) for item in example.provenance
    ]
    provenance_ids = sorted(row["provenance_id"] for row in provenance_rows)
    p_id = parent_id(example.source, example.parent_key)
    identity = {
        "schema_version": CANONICAL_SCHEMA_VERSION,
        "parent_id": p_id,
        "source": example.source,
        "logical_role": example.logical_role.value,
        "task_family": example.task_family.value,
        "supervision_kind": example.supervision_kind.value,
        "input_layout": example.input_layout.value,
        "output_modality": example.output_modality.value,
        "media_asset_ids": sorted(row["asset_id"] for row in media),
        "target": target,
        "instruction": example.instruction,
        "training_responses": list(example.training_responses),
        "reference_responses": list(example.reference_responses),
        "annotation_layer": example.annotation_layer.value,
    }
    canonical: dict[str, Any] = {
        "schema_version": CANONICAL_SCHEMA_VERSION,
        "record_id": record_id(identity),
        "parent_id": p_id,
        "source": example.source,
        "source_record_id": example.source_record_id,
        "source_split": example.source_split,
        "logical_role": example.logical_role.value,
        "task_family": example.task_family.value,
        "supervision_kind": example.supervision_kind.value,
        "input_layout": example.input_layout.value,
        "output_modality": example.output_modality.value,
        "media": media,
        "target": target,
        "instruction": example.instruction,
        "training_responses": list(example.training_responses),
        "reference_responses": list(example.reference_responses),
        "deterministic_facts": dict(example.deterministic_facts),
        "annotation": {
            "layer": example.annotation_layer.value,
            "review_status": example.review_status.value,
        },
        "coordinate_convention": {
            "image_origin": "top_left",
            "bbox_canonical": "xyxy_normalized",
        },
        "transforms": transforms,
        "quality_flags": sorted(set(example.quality_flags)),
        "provenance_ids": provenance_ids,
    }
    canonical["record_sha256"] = record_hash(canonical)
    validate_canonical_record(canonical)
    return canonical, provenance_rows


def _bounded_skip_rows(
    skips: Sequence[SkipRecord],
    *,
    profile: str,
) -> list[dict[str, Any]]:
    if profile == "full":
        return [
            row.to_dict()
            for row in sorted(
                skips,
                key=lambda item: (
                    item.source,
                    item.reason_code.value,
                    item.source_record_id,
                ),
            )
        ]
    groups: dict[tuple[str, str, str], list[SkipRecord]] = defaultdict(list)
    for row in skips:
        groups[(row.source, row.task_family, row.reason_code.value)].append(row)
    output: list[dict[str, Any]] = []
    for key in sorted(groups):
        rows = sorted(groups[key], key=lambda item: item.source_record_id)
        output.extend(row.to_dict() for row in rows[:10])
    return output


def _nested_record_counts(
    records: Sequence[Mapping[str, Any]],
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


def _statistics(
    records: Sequence[Mapping[str, Any]],
    all_skips: Sequence[SkipRecord],
    *,
    materialized_skip_count: int,
    asset_store: AssetStore,
    external_split: Mapping[str, Any],
) -> dict[str, Any]:
    final_external_split = dict(external_split)
    external_records = [
        row
        for row in records
        if row["logical_role"]
        in {
            LogicalRole.EXTERNAL_TRAIN.value,
            LogicalRole.EXTERNAL_VAL.value,
        }
    ]
    final_external_split.update(
        {
            "parent_role_counts": dict(
                sorted(
                    Counter(
                        role
                        for _, role in {
                            (str(row["parent_id"]), str(row["logical_role"]))
                            for row in external_records
                        }
                    ).items()
                )
            ),
            "record_role_counts": dict(
                sorted(
                    Counter(
                        str(row["logical_role"]) for row in external_records
                    ).items()
                )
            ),
            "source_role_counts": {
                source: dict(sorted(counts.items()))
                for source, counts in sorted(
                    _nested_record_counts(
                        external_records,
                        outer="source",
                    ).items()
                )
            },
            "task_role_counts": {
                task: dict(sorted(counts.items()))
                for task, counts in sorted(
                    _nested_record_counts(
                        external_records,
                        outer="task",
                    ).items()
                )
            },
        }
    )
    return {
        "schema_version": STATISTICS_SCHEMA_VERSION,
        "record_count": len(records),
        "parent_count": len({row["parent_id"] for row in records}),
        "asset_count": len(asset_store.inventory),
        "asset_bytes": asset_store.total_bytes,
        "source_counts": dict(sorted(Counter(row["source"] for row in records).items())),
        "role_counts": dict(
            sorted(Counter(row["logical_role"] for row in records).items())
        ),
        "task_counts": dict(
            sorted(Counter(row["task_family"] for row in records).items())
        ),
        "supervision_kind_counts": dict(
            sorted(Counter(row["supervision_kind"] for row in records).items())
        ),
        "input_layout_counts": dict(
            sorted(Counter(row["input_layout"] for row in records).items())
        ),
        "output_modality_counts": dict(
            sorted(Counter(row["output_modality"] for row in records).items())
        ),
        "annotation_layer_counts": dict(
            sorted(Counter(row["annotation"]["layer"] for row in records).items())
        ),
        "parent_role_counts": dict(
            sorted(
                Counter(
                    role
                    for _, role in {
                        (str(row["parent_id"]), str(row["logical_role"]))
                        for row in records
                    }
                ).items()
            )
        ),
        "source_role_counts": {
            source: dict(sorted(counts.items()))
            for source, counts in sorted(
                _nested_record_counts(records, outer="source").items()
            )
        },
        "task_role_counts": {
            task: dict(sorted(counts.items()))
            for task, counts in sorted(
                _nested_record_counts(records, outer="task").items()
            )
        },
        "external_split": final_external_split,
        "skip_count_total": len(all_skips),
        "skip_records_materialized": materialized_skip_count,
        "skip_reason_counts": dict(
            sorted(Counter(row.reason_code.value for row in all_skips).items())
        ),
    }


def _payload_hashes(staging: Path, excluded: set[str]) -> dict[str, Any]:
    files = []
    for path in sorted(
        item for item in staging.rglob("*") if item.is_file()
    ):
        relative = path.relative_to(staging).as_posix()
        if relative in excluded or relative.startswith("."):
            continue
        files.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    root_digest = sha256_bytes(canonical_json(files).encode("utf-8"))
    return {
        "schema_version": HASH_SCHEMA_VERSION,
        "files": files,
        "root_sha256": root_digest,
    }


def _write_hashes(staging: Path) -> dict[str, Any]:
    relative = "metadata/hashes.json"
    hashes = _payload_hashes(
        staging,
        {
            relative,
            "manifest.json",
            "metadata/validation.json",
        },
    )
    atomic_write_json(staging / relative, hashes)
    return hashes


def _record_shards(
    staging: Path,
    records: Sequence[Mapping[str, Any]],
    *,
    rows_per_shard: int,
) -> list[dict[str, Any]]:
    shards: list[dict[str, Any]] = []
    for shard_index, start in enumerate(range(0, len(records), rows_per_shard)):
        rows = records[start : start + rows_per_shard]
        relative = f"records/part-{shard_index:05d}.jsonl"
        atomic_write_jsonl(staging / relative, rows)
        shards.append({"path": relative, "record_count": len(rows)})
    return shards


def _manifest(
    *,
    config: BuildConfig,
    records: Sequence[Mapping[str, Any]],
    shards: Sequence[Mapping[str, Any]],
    statistics: Mapping[str, Any],
    audits: Mapping[str, Any],
    hashes: Mapping[str, Any],
    blockers: Sequence[str],
    deletion_allowlist_path: str | None,
    release_equivalence_path: str | None = None,
) -> dict[str, Any]:
    role_shards: dict[str, list[str]] = defaultdict(list)
    record_by_shard: dict[str, list[Mapping[str, Any]]] = {}
    offset = 0
    for shard in shards:
        count = int(shard["record_count"])
        record_by_shard[str(shard["path"])] = list(records[offset : offset + count])
        offset += count
    for path, rows in record_by_shard.items():
        for role in sorted({str(row["logical_role"]) for row in rows}):
            role_shards[role].append(path)
    return {
        "schema_version": MANIFEST_VERSION,
        "canonical_schema_version": CANONICAL_SCHEMA_VERSION,
        "build_profile": config.profile,
        "benchmark_scope": BENCHMARK_SCOPE,
        "scope_validation_complete": "deep_validation_pending" not in blockers,
        "build_id": (
            "build_"
            + sha256_text(
                canonical_json([config.semantic_hash, hashes["root_sha256"]])
            )
        ),
        "semantic_config_sha256": config.semantic_hash,
        "payload_root_sha256": hashes["root_sha256"],
        "hash_manifest_sha256": None,
        "reference_project": dict(HF_DATASETS_REFERENCE),
        "environment_versions": _environment_versions(),
        "layout": {
            "canonical_schema": "schemas/canonical.schema.json",
            "record_shards": list(shards),
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
            "deletion_allowlist": deletion_allowlist_path,
            "release_equivalence": release_equivalence_path,
        },
        "counts": {
            "records": len(records),
            "parents": statistics["parent_count"],
            "assets": statistics["asset_count"],
            "asset_bytes": statistics["asset_bytes"],
        },
        "source_audits": dict(audits),
        "formal_acceptance_eligible": not blockers,
        "formal_acceptance_blockers": list(blockers),
        "source_roots_embedded": False,
        "external_test_retained": False,
        "upstream_split_policy": (
            "provenance_only_repartitioned_parent_content_component"
        ),
        "official_upstream_evaluation_reproducible": False,
    }


def _ensure_parent_split_integrity(records: Sequence[Mapping[str, Any]]) -> None:
    external_roles_by_parent: dict[str, set[str]] = defaultdict(set)
    for row in records:
        role = str(row["logical_role"])
        parent = str(row["parent_id"])
        if role in {
            LogicalRole.EXTERNAL_TRAIN.value,
            LogicalRole.EXTERNAL_VAL.value,
        }:
            external_roles_by_parent[parent].add(role)
        else:
            raise BuildError(
                ReasonCode.ROLE_CONTAMINATION,
                f"RS-GeneralDesc builder 不接受 role={role}",
            )
    external_conflicts = {
        parent: sorted(roles)
        for parent, roles in external_roles_by_parent.items()
        if len(roles) > 1
    }
    if external_conflicts:
        raise BuildError(
            ReasonCode.EXTERNAL_SPLIT_LEAKAGE,
            "同一 external parent 跨 train/val",
            details={"conflict_count": len(external_conflicts)},
        )
def _formal_blockers(
    config: BuildConfig,
    *,
    deep_pending: bool,
) -> list[str]:
    blockers: list[str] = []
    if config.profile != "full":
        blockers.append("bounded_smoke_profile")
    if deep_pending:
        blockers.append("deep_validation_pending")
    return blockers


def _allowlist_rows(skips: Sequence[SkipRecord]) -> list[dict[str, Any]]:
    allowed = {
        ReasonCode.UNUSED_ASSET,
        ReasonCode.ASSET_CORRUPT,
        ReasonCode.ASSET_ZERO_BYTES,
        ReasonCode.DUPLICATE_RECORD,
        ReasonCode.ASSET_MISSING,
    }
    rows: list[dict[str, Any]] = []
    for skip in skips:
        if skip.reason_code not in allowed:
            continue
        path = skip.evidence.get("asset")
        size = skip.evidence.get("size_bytes")
        if not isinstance(path, str) or not isinstance(size, int):
            continue
        rows.append(
            {
                "source": skip.source,
                "source_record_id": skip.source_record_id,
                "relative_path": path,
                "size_bytes": size,
                "reason_code": skip.reason_code.value,
                "evidence": dict(skip.evidence),
                "action": "human_review_only",
            }
        )
    return sorted(rows, key=lambda row: (row["source"], row["relative_path"]))


def build_benchmark(config: BuildConfig) -> Path:
    _preflight_paths(config, for_build=True)
    target = config.output_root
    staging = target.parent / f".{target.name}.staging-{uuid.uuid4().hex}"
    if staging.exists() or staging.is_symlink():
        raise BuildError(ReasonCode.OUTPUT_EXISTS, f"staging 已存在：{staging}")
    staging.mkdir(parents=True)
    (staging / STAGING_SENTINEL).write_text(
        "owned-by-rs-generaldesc-builder\n", encoding="utf-8"
    )
    try:
        results, split_summary = _scan(config, deep=True, for_build=True)
        examples = _apply_global_limits(
            [example for result in results for example in result.examples],
            config,
        )
        if not examples:
            raise BuildError(ReasonCode.SCHEMA_MISMATCH, "没有可构建 canonical record")
        all_skips = [skip for result in results for skip in result.skips]
        store = AssetStore(
            staging,
            policy=config.asset_policy,
            limits=config.limits,
        )
        records_by_id: dict[str, dict[str, Any]] = {}
        provenance_by_id: dict[str, dict[str, Any]] = {}
        for example in examples:
            checkpoint = store.checkpoint()
            try:
                record, provenance_rows = _canonical_from_example(
                    example,
                    store=store,
                )
            except AssetError as error:
                store.rollback(checkpoint)
                if error.code in {
                    ReasonCode.ASSET_LIMIT_EXCEEDED,
                    ReasonCode.OUTPUT_LINK,
                    ReasonCode.HASH_MISMATCH,
                }:
                    raise BuildError(
                        error.code,
                        str(error),
                        details=error.details,
                    ) from error
                all_skips.append(
                    SkipRecord(
                        source=example.source,
                        source_record_id=example.source_record_id,
                        source_split=example.source_split,
                        task_family=example.task_family.value,
                        reason_code=error.code,
                        evidence=dict(error.details),
                    )
                )
                continue
            for row in provenance_rows:
                provenance_by_id[row["provenance_id"]] = row
            existing = records_by_id.get(record["record_id"])
            if existing is None:
                records_by_id[record["record_id"]] = record
            else:
                existing["provenance_ids"] = sorted(
                    set(existing["provenance_ids"]) | set(record["provenance_ids"])
                )
                existing["record_sha256"] = record_hash(existing)
                all_skips.append(
                    SkipRecord(
                        source=example.source,
                        source_record_id=example.source_record_id,
                        source_split=example.source_split,
                        task_family=example.task_family.value,
                        reason_code=ReasonCode.DUPLICATE_RECORD,
                        evidence={"canonical_record_id": record["record_id"]},
                    )
                )
        records = sorted(
            records_by_id.values(),
            key=lambda row: (
                row["logical_role"],
                row["source"],
                row["parent_id"],
                row["task_family"],
                row["record_id"],
            ),
        )
        if not records:
            raise BuildError(
                ReasonCode.SCHEMA_MISMATCH,
                "全部候选均被严格资产验证拒绝",
            )
        for row in records:
            validate_canonical_record(row)
        _ensure_parent_split_integrity(records)

        atomic_write_json(
            staging / "schemas/canonical.schema.json",
            canonical_schema_document(),
        )
        shards = _record_shards(
            staging,
            records,
            rows_per_shard=config.sharding.records_per_shard,
        )
        provenance_rows = [
            provenance_by_id[key] for key in sorted(provenance_by_id)
        ]
        atomic_write_jsonl(staging / "metadata/provenance.jsonl", provenance_rows)
        bounded_skips = _bounded_skip_rows(all_skips, profile=config.profile)
        atomic_write_jsonl(staging / "metadata/skips.jsonl", bounded_skips)
        statistics = _statistics(
            records,
            all_skips,
            materialized_skip_count=len(bounded_skips),
            asset_store=store,
            external_split=split_summary,
        )
        atomic_write_json(staging / "metadata/statistics.json", statistics)
        atomic_write_json(staging / "metadata/assets.json", store.inventory)
        atomic_write_json(staging / "metadata/build_config.json", config.sanitized())
        audits = {
            result.source: dict(result.audit)
            for result in sorted(results, key=lambda item: item.source)
        }
        atomic_write_json(staging / "metadata/source_audits.json", audits)
        allowlist_path: str | None = None
        if config.profile == "full" and config.deletion_allowlist:
            allowlist_path = "metadata/deletion_allowlist.jsonl"
            atomic_write_jsonl(
                staging / allowlist_path,
                _allowlist_rows(all_skips),
            )
        hashes = _write_hashes(staging)
        blockers = _formal_blockers(config, deep_pending=False)
        manifest = _manifest(
            config=config,
            records=records,
            shards=shards,
            statistics=statistics,
            audits=audits,
            hashes=hashes,
            blockers=blockers,
            deletion_allowlist_path=allowlist_path,
        )
        manifest["hash_manifest_sha256"] = sha256_file(
            staging / "metadata/hashes.json"
        )
        atomic_write_json(staging / "manifest.json", manifest)

        from .validator import validate_benchmark

        validation = validate_benchmark(staging, deep=True)
        if validation["errors"]:
            raise BuildError(
                ReasonCode.STAGING_FAILURE,
                "最终 staging deep validation 失败",
                details={"errors": validation["errors"][:20]},
            )
        atomic_write_json(staging / "metadata/validation.json", validation)
        if target.exists() or target.is_symlink():
            raise BuildError(
                ReasonCode.OUTPUT_EXISTS,
                f"发布前发现 output_root 已存在：{target}",
            )
        (staging / STAGING_SENTINEL).unlink()
        staging.replace(target)
    except BaseException:
        if staging.exists() and (staging / STAGING_SENTINEL).is_file():
            shutil.rmtree(staging)
        raise
    return target
