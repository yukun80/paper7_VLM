"""Atomic end-to-end Canonical Benchmark v3 assembler."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sami_gsd.contracts.config import BenchmarkAuditConfig
from sami_gsd.contracts.language import DescriptionSourceRecord
from sami_gsd.data.duplicates import DUPLICATE_PROTOCOL_VERSION, build_duplicate_analysis
from sami_gsd.data.materialize import MATERIALIZER_VERSION, SpatialParentInput, materialize_spatial_parent
from sami_gsd.data.split import SPLIT_PROTOCOL_VERSION, apply_parent_splits, assign_parent_splits
from sami_gsd.data.tasks import FixedRegionPrediction, RegionAnswer, TASK_EXPANSION_VERSION, expand_task_views
from sami_gsd.data.validation import VALIDATION_VERSION, validate_benchmark_payload
from sami_gsd.utilities.artifacts import (
    atomic_output_directory,
    atomic_write_bytes,
    canonical_json_bytes,
    canonical_yaml_bytes,
    sha256_bytes,
    sha256_file,
)


BENCHMARK_BUILDER_VERSION = "sami_canonical_benchmark_builder_v1"


class BenchmarkBuildError(ValueError):
    """Raised when a formal build gate cannot be satisfied."""


def _jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    """Serialize stable strict JSONL with one final newline per record."""

    return b"".join(canonical_json_bytes(row) for row in rows)


def _write_json(staging: Path, relative: str, payload: Any) -> None:
    """Atomically write one canonical JSON artifact in the private staging root."""

    atomic_write_bytes(staging / relative, canonical_json_bytes(payload))


def _write_jsonl(staging: Path, relative: str, rows: list[dict[str, Any]]) -> None:
    """Atomically write one canonical JSONL projection."""

    atomic_write_bytes(staging / relative, _jsonl_bytes(rows))


def _registry(config: BenchmarkAuditConfig) -> dict[str, Any]:
    """Publish the complete configured source/license registry."""

    return {
        "schema_version": "sami_source_registry_v1",
        "entries": [
            {
                "source_key": source.source_key,
                "display_name": source.display_name,
                "local_path": source.local_path,
                "enabled": source.enabled,
                "allowed_task_roles": list(source.allowed_task_roles),
                **source.license.model_dump(mode="json"),
            }
            for source in sorted(config.sources, key=lambda item: item.source_key)
        ],
    }


def _license_report(config: BenchmarkAuditConfig, parent_inputs: tuple[SpatialParentInput, ...]) -> dict[str, Any]:
    """Prove every materialized source is explicitly reviewed and eligible."""

    entries = [source.license for source in config.sources]
    violations = [
        license.source_key
        for license in entries
        if license.allowed_for_training
        and (
            license.license_status == "unknown"
            or license.license_name.lower() == "unknown"
            or license.reviewed_by is None
            or license.review_date is None
        )
    ]
    configured = {source.source_key: source.license for source in config.sources}
    materialized_sources = sorted({item.license.source_key for item in parent_inputs})
    for item in parent_inputs:
        if item.license.source_key not in configured or item.license != configured[item.license.source_key]:
            violations.append(f"unbound_input_license:{item.license.source_key}")
    return {
        "schema_version": "sami_license_report_v1",
        "builder_version": BENCHMARK_BUILDER_VERSION,
        "training_eligible_sources": sorted(license.source_key for license in entries if license.allowed_for_training),
        "materialized_training_sources": materialized_sources,
        "unknown_license_sources": sorted(license.source_key for license in entries if license.license_status == "unknown"),
        "training_eligible_unknown_count": len(violations),
        "errors": [f"training_eligible_unknown:{value}" for value in sorted(violations)],
        "warnings": [],
    }


def build_canonical_benchmark(
    config: BenchmarkAuditConfig,
    *,
    parent_inputs: tuple[SpatialParentInput, ...],
    description_records: tuple[DescriptionSourceRecord, ...],
    output_dir: Path,
    schemas_root: Path,
    forced_splits: dict[str, str] | None = None,
    region_answers: dict[tuple[str, str], RegionAnswer] | None = None,
    fixed_predictions: dict[tuple[str, str], FixedRegionPrediction] | None = None,
) -> dict[str, Any]:
    """Build, validate and atomically publish one new benchmark directory."""

    if not parent_inputs:
        raise BenchmarkBuildError("Small build requires at least one resolved spatial parent")
    if config.mode == "small":
        counts: dict[str, int] = {}
        for item in parent_inputs:
            counts[item.license.source_key] = counts.get(item.license.source_key, 0) + 1
        if any(count > config.build.small_max_parents_per_source for count in counts.values()):
            raise BenchmarkBuildError("Small parent count exceeds the configured per-source limit")
    license_report = _license_report(config, parent_inputs)
    if license_report["errors"]:
        raise BenchmarkBuildError("license report blocks canonical materialization")
    if any(record.training_eligible for record in description_records):
        raise BenchmarkBuildError("training language images require canonical materialization before publication")

    config_hash = sha256_bytes(canonical_json_bytes(config.model_dump(mode="json")))
    source_input_hash = sha256_bytes(
        canonical_json_bytes(
            [
                {
                    "parent_id": item.parent_id,
                    "source_record_sha256": item.source_record_sha256,
                    "license": item.license.model_dump(mode="json"),
                }
                for item in sorted(parent_inputs, key=lambda value: value.parent_id)
            ]
        )
    )
    with atomic_output_directory(output_dir) as staging:
        materialized = tuple(
            materialize_spatial_parent(
                item,
                benchmark_root=staging,
                canvas_hw=config.build.materialization.canvas_hw,
            )
            for item in sorted(parent_inputs, key=lambda value: value.parent_id)
        )
        audit_parents = tuple(item.parent for item in materialized)
        duplicates = build_duplicate_analysis(
            audit_parents,
            benchmark_root=staging,
            settings=config.build.duplicates,
        )
        split = assign_parent_splits(
            audit_parents,
            duplicate_clusters=duplicates.parent_to_cluster,
            settings=config.build.split,
            seed=config.seed,
            forced_splits=forced_splits,
        )
        parents = apply_parent_splits(audit_parents, split)
        tasks = expand_task_views(
            parents,
            region_answers=region_answers,
            fixed_predictions=fixed_predictions,
        )
        parent_by_id = {parent.parent_id: parent for parent in parents}

        parent_rows = [parent.model_dump(mode="json") for parent in parents]
        _write_jsonl(staging, "parents/all.jsonl", parent_rows)
        for split_name in ("train", "val", "test"):
            _write_jsonl(
                staging,
                f"parents/{split_name}.jsonl",
                [row for row in parent_rows if row["split"] == split_name],
            )
        for task_type in ("t1_global", "t2_referring", "t3_gt_region", "t4_predicted_region"):
            for split_name in ("train", "val", "test"):
                rows = [
                    task.model_dump(mode="json")
                    for task in tasks.tasks
                    if task.task_type == task_type and parent_by_id[task.parent_id].split == split_name
                ]
                _write_jsonl(staging, f"tasks/{task_type}/{split_name}.jsonl", rows)

        registry = _registry(config)
        atomic_write_bytes(staging / "manifests/source_registry.yaml", canonical_yaml_bytes(registry))
        _write_json(
            staging,
            "manifests/split_manifest.json",
            {
                "schema_version": "sami_split_manifest_v1",
                "protocol": SPLIT_PROTOCOL_VERSION,
                "seed": config.seed,
                "aggregate_sha256": split.aggregate_sha256,
                "components": list(split.components),
                "parent_to_split": split.parent_to_split,
            },
        )
        _write_jsonl(staging, "manifests/duplicate_clusters.jsonl", list(duplicates.clusters))
        _write_json(staging, "manifests/evaluation_conditions.json", tasks.evaluation_conditions)
        description_rows = [
            record.model_dump(mode="json")
            for record in sorted(description_records, key=lambda item: item.record_id)
        ]
        _write_jsonl(staging, "manifests/description_source_subset.jsonl", description_rows)
        _write_jsonl(
            staging,
            "manifests/description_train_eligible.jsonl",
            [row for row in description_rows if row["training_eligible"] and row["split_policy"] != "permanent_test_only"],
        )
        _write_json(staging, "reports/license_report.json", license_report)
        cross_split = sum(
            len({split.parent_to_split[parent_id] for parent_id in cluster["parent_ids"]}) > 1
            for cluster in duplicates.clusters
        )
        duplicate_report = {
            "schema_version": "sami_duplicate_report_v1",
            "protocol": DUPLICATE_PROTOCOL_VERSION,
            "aggregate_sha256": duplicates.aggregate_sha256,
            "cluster_count": len(duplicates.clusters),
            "exact_edge_count": duplicates.exact_edge_count,
            "perceptual_candidate_edge_count": duplicates.perceptual_candidate_edge_count,
            "verified_perceptual_edge_count": duplicates.verified_perceptual_edge_count,
            "verified_duplicate_cross_split_count": cross_split,
            "errors": [] if cross_split == 0 else [f"verified_duplicate_cross_split:{cross_split}"],
        }
        _write_json(staging, "reports/duplicate_report.json", duplicate_report)
        summary = {
            "schema_version": "sami_benchmark_summary_v1",
            "builder_version": BENCHMARK_BUILDER_VERSION,
            "mode": config.mode,
            "parent_count": len(parents),
            "parents_by_split": {
                split_name: sum(parent.split == split_name for parent in parents)
                for split_name in ("train", "val", "test")
            },
            "task_count": len(tasks.tasks),
            "tasks_by_type": {
                task_type: sum(task.task_type == task_type for task in tasks.tasks)
                for task_type in ("t1_global", "t2_referring", "t3_gt_region", "t4_predicted_region")
            },
            "description_source_record_count": len(description_records),
            "valid_pixel_count": sum(item.valid_pixel_count for item in materialized),
            "excluded_pixel_count": sum(item.excluded_pixel_count for item in materialized),
            "positive_valid_pixel_count": sum(item.positive_valid_pixel_count for item in materialized),
            "warnings": [],
            "errors": [],
        }
        _write_json(staging, "reports/summary_report.json", summary)
        validation = validate_benchmark_payload(staging, schemas_root=schemas_root)
        _write_json(staging, "reports/validation_report.json", validation)
        if validation["errors"]:
            raise BenchmarkBuildError(f"benchmark validation failed: {validation['errors']}")

        output_hashes = {
            path.relative_to(staging).as_posix(): sha256_file(path)
            for path in sorted(staging.rglob("*"))
            if path.is_file()
        }
        manifest_core = {
            "builder_version": BENCHMARK_BUILDER_VERSION,
            "materializer_version": MATERIALIZER_VERSION,
            "duplicate_version": DUPLICATE_PROTOCOL_VERSION,
            "split_version": SPLIT_PROTOCOL_VERSION,
            "task_expansion_version": TASK_EXPANSION_VERSION,
            "validation_version": VALIDATION_VERSION,
            "mode": config.mode,
            "seed": config.seed,
            "config_sha256": config_hash,
            "source_input_sha256": source_input_hash,
            "parent_record_sha256": sha256_bytes(_jsonl_bytes(parent_rows)),
            "output_sha256": output_hashes,
        }
        manifest = {
            "schema_version": "sami_benchmark_manifest_v1",
            **manifest_core,
            "aggregate_sha256": sha256_bytes(canonical_json_bytes(manifest_core)),
            "errors": [],
            "warnings": validation["warnings"],
        }
        _write_json(staging, "manifests/benchmark_manifest.json", manifest)
    return manifest


__all__ = ["BENCHMARK_BUILDER_VERSION", "BenchmarkBuildError", "build_canonical_benchmark"]
