"""Atomic end-to-end Canonical Benchmark v3 assembler."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sami_gsd.contracts.canonical import ArtifactRef, CanonicalParentV3
from sami_gsd.contracts.config import BenchmarkAuditConfig
from sami_gsd.contracts.language import (
    CanonicalDescriptionRecord,
    CanonicalLanguageAnswer,
    DescriptionSourceRecord,
)
from sami_gsd.data.duplicates import DUPLICATE_PROTOCOL_VERSION, build_duplicate_analysis
from sami_gsd.data.materialize import (
    MATERIALIZER_VERSION,
    LanguageParentInput,
    SpatialParentInput,
    materialize_language_parent,
    materialize_spatial_parent,
)
from sami_gsd.data.split import SPLIT_PROTOCOL_VERSION, apply_parent_splits, assign_parent_splits
from sami_gsd.data.tasks import FixedRegionPrediction, RegionAnswer, TASK_EXPANSION_VERSION, expand_task_views
from sami_gsd.data.transforms import forward_box, quantize_covering_box
from sami_gsd.data.validation import VALIDATION_VERSION, validate_benchmark_payload
from sami_gsd.utilities.artifacts import (
    atomic_output_directory,
    atomic_write_bytes,
    canonical_json_bytes,
    canonical_yaml_bytes,
    sha256_bytes,
    sha256_file,
)


BENCHMARK_BUILDER_VERSION = "sami_canonical_benchmark_builder_v3_component_license_bound"


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
        "schema_version": "sami_source_registry_v2_component_license_bound",
        "entries": [
            {
                "source_key": source.source_key,
                "display_name": source.display_name,
                "local_path": source.local_path,
                "enabled": source.enabled,
                "allowed_task_roles": list(source.allowed_task_roles),
                "language_components": [
                    {
                        "component": component.component,
                        "component_key": component.component_key,
                        "allowed_task_roles": list(component.allowed_task_roles),
                        "split_policy": component.split_policy,
                        **component.license.model_dump(mode="json"),
                    }
                    for component in source.language_components
                ],
                **source.license.model_dump(mode="json"),
            }
            for source in sorted(config.sources, key=lambda item: item.source_key)
        ],
    }


def _license_is_reviewed_for_use(license_record: Any) -> bool:
    """Return whether one license has named, non-unknown review evidence."""

    return (
        license_record.license_status != "unknown"
        and license_record.license_name.lower() != "unknown"
        and license_record.reviewed_by is not None
        and license_record.review_date is not None
    )


def _language_parent_inputs(
    config: BenchmarkAuditConfig,
    records: tuple[DescriptionSourceRecord, ...],
    *,
    datasets_root: Path | None,
) -> tuple[tuple[LanguageParentInput, ...], dict[str, str], dict[str, str]]:
    """Resolve licensed language records into exact-image parent groups."""

    configured = {source.source_key: source for source in config.sources}
    selected: list[tuple[DescriptionSourceRecord, Path]] = []
    for record in sorted(records, key=lambda item: item.record_id):
        source = configured.get(record.source_key)
        component = (
            next(
                (item for item in source.language_components if item.component == record.component),
                None,
            )
            if source is not None
            else None
        )
        if (
            source is None
            or component is None
            or record.component_license_key != component.component_key
            or record.license != component.license
            or record.split_policy != component.split_policy
        ):
            raise BenchmarkBuildError(f"unbound language component/license: {record.record_id}")
        materialize_for_training = record.training_eligible
        materialize_for_test = record.split_policy == "permanent_test_only" and record.license.allowed_for_evaluation
        if not materialize_for_training and not materialize_for_test:
            continue
        role = "language_region" if record.role == "region_short_phrase" else "language_global"
        if role not in component.allowed_task_roles:
            raise BenchmarkBuildError(f"language role is not approved by the component registry: {record.record_id}")
        if not _license_is_reviewed_for_use(record.license):
            raise BenchmarkBuildError(f"language materialization requires reviewed license evidence: {record.record_id}")
        if datasets_root is None:
            raise BenchmarkBuildError("datasets_root is required for licensed language materialization")
        prefix = f"datasets/{source.local_path}/"
        if not record.image.logical_path.startswith(prefix):
            raise BenchmarkBuildError(f"language image path is outside its configured source root: {record.record_id}")
        physical = datasets_root / record.image.logical_path.removeprefix("datasets/")
        if not physical.is_file():
            raise BenchmarkBuildError(f"language image is missing: {record.record_id}")
        try:
            resolved_physical = physical.resolve(strict=True)
        except OSError as error:
            raise BenchmarkBuildError(f"language image cannot be resolved: {record.record_id}") from error
        if not resolved_physical.is_relative_to(datasets_root.resolve()):
            raise BenchmarkBuildError(f"language image escapes datasets_root: {record.record_id}")
        if sha256_file(physical) != record.image.sha256:
            raise BenchmarkBuildError(f"language image hash changed after subset selection: {record.record_id}")
        selected.append((record, physical))

    grouped: dict[tuple[str, str, str], list[tuple[DescriptionSourceRecord, Path]]] = {}
    for record, physical in selected:
        license_sha256 = sha256_bytes(canonical_json_bytes(record.license.model_dump(mode="json")))
        grouped.setdefault((record.source_key, record.image.sha256, license_sha256), []).append((record, physical))
    inputs: list[LanguageParentInput] = []
    record_to_parent: dict[str, str] = {}
    forced: dict[str, str] = {}
    for (source_key, image_sha256, license_sha256), members in sorted(grouped.items()):
        parent_id = f"language-{source_key}-{image_sha256[:16]}-{license_sha256[:8]}"
        ordered_records = tuple(sorted((record for record, _ in members), key=lambda item: item.record_id))
        physical = min((path for _, path in members), key=lambda path: path.as_posix())
        inputs.append(LanguageParentInput(parent_id=parent_id, records=ordered_records, raw_image_path=physical))
        for record in ordered_records:
            record_to_parent[record.record_id] = parent_id
        if any(record.split_policy == "permanent_test_only" for record in ordered_records):
            forced[parent_id] = "test"
    return tuple(inputs), dict(sorted(record_to_parent.items())), dict(sorted(forced.items()))


def _canonical_description_records(
    records: tuple[DescriptionSourceRecord, ...],
    *,
    record_to_parent: dict[str, str],
    parent_by_id: dict[str, CanonicalParentV3],
) -> tuple[CanonicalDescriptionRecord, ...]:
    """Bind selected targets to final parent splits and Benchmark assets."""

    canonical: list[CanonicalDescriptionRecord] = []
    for record in sorted(records, key=lambda item: item.record_id):
        parent_id = record_to_parent.get(record.record_id)
        if parent_id is None:
            continue
        parent = parent_by_id[parent_id]
        modality = next(
            item for item in parent.modalities if item.modality_id == parent.reference_canvas.reference_modality_id
        )
        if modality.aligned_asset_path is None or modality.valid_mask_path is None:
            raise BenchmarkBuildError(f"materialized language parent lost its reference assets: {parent_id}")
        region_box = None
        if record.normalized_box_xyxy is not None:
            original_h, original_w = parent.reference_canvas.original_hw
            source_box = (
                record.normalized_box_xyxy[0] * original_w,
                record.normalized_box_xyxy[1] * original_h,
                record.normalized_box_xyxy[2] * original_w,
                record.normalized_box_xyxy[3] * original_h,
            )
            region_box = quantize_covering_box(
                forward_box(source_box, parent.reference_canvas.transform_chain),
                parent.reference_canvas.canvas_hw,
            )
        source_record_sha256 = sha256_bytes(canonical_json_bytes(record.model_dump(mode="json")))
        canonical.append(
            CanonicalDescriptionRecord(
                schema_version="sami_canonical_description_v2_component_license_bound",
                record_id=record.record_id,
                parent_id=parent_id,
                source_key=record.source_key,
                component=record.component,
                component_license_key=record.component_license_key,
                component_license_sha256=sha256_bytes(
                    canonical_json_bytes(record.license.model_dump(mode="json"))
                ),
                role=record.role,
                split_policy=record.split_policy,
                split=parent.split,
                image_ref=ArtifactRef(path=modality.aligned_asset_path, sha256=modality.hashes["aligned"]),
                valid_mask_ref=ArtifactRef(path=modality.valid_mask_path, sha256=modality.hashes["valid"]),
                source_image_sha256=record.image.sha256,
                source_record_sha256=source_record_sha256,
                answers=tuple(
                    CanonicalLanguageAnswer(
                        source_answer_id=answer.answer_id,
                        text=answer.text,
                        annotation_origin=answer.annotation_origin,
                        source_index_sha256=answer.index_sha256,
                    )
                    for answer in record.answers
                ),
                region_box_half_open=region_box,
                training_eligible=record.training_eligible,
            )
        )
    return tuple(canonical)


def _license_report(
    config: BenchmarkAuditConfig,
    parent_inputs: tuple[SpatialParentInput, ...],
    description_records: tuple[DescriptionSourceRecord, ...],
    *,
    materialized_description_ids: set[str],
) -> dict[str, Any]:
    """Prove every materialized source is explicitly reviewed and eligible."""

    top_level_licenses = {
        source.source_key: source.license
        for source in config.sources
    }
    component_licenses = {
        component.component_key: component.license
        for source in config.sources
        for component in source.language_components
    }
    scopes = {**top_level_licenses, **component_licenses}
    violations = [
        scope_key
        for scope_key, license_record in scopes.items()
        if license_record.allowed_for_training
        and (
            license_record.license_status == "unknown"
            or license_record.license_name.lower() == "unknown"
            or license_record.reviewed_by is None
            or license_record.review_date is None
        )
    ]
    materialized_sources = sorted({item.license.source_key for item in parent_inputs})
    for item in parent_inputs:
        if (
            item.license.source_key not in top_level_licenses
            or item.license != top_level_licenses[item.license.source_key]
        ):
            violations.append(f"unbound_input_license:{item.license.source_key}")
    materialized_language_sources: set[str] = set()
    for record in description_records:
        if record.record_id not in materialized_description_ids:
            continue
        materialized_language_sources.add(record.component_license_key)
        if (
            record.component_license_key not in component_licenses
            or record.license != component_licenses[record.component_license_key]
        ):
            violations.append(f"unbound_language_license:{record.record_id}")
        if record.training_eligible and not record.license.allowed_for_training:
            violations.append(f"language_training_not_allowed:{record.record_id}")
        if record.split_policy == "permanent_test_only" and not record.license.allowed_for_evaluation:
            violations.append(f"language_evaluation_not_allowed:{record.record_id}")
        if not _license_is_reviewed_for_use(record.license):
            violations.append(f"language_license_unreviewed:{record.record_id}")
    return {
        "schema_version": "sami_license_report_v2_component_license_bound",
        "builder_version": BENCHMARK_BUILDER_VERSION,
        "training_eligible_sources": sorted(
            scope_key
            for scope_key, license_record in scopes.items()
            if license_record.allowed_for_training
        ),
        "materialized_training_sources": materialized_sources,
        "materialized_language_sources": sorted(materialized_language_sources),
        "unknown_license_sources": sorted(
            scope_key
            for scope_key, license_record in scopes.items()
            if license_record.license_status == "unknown"
        ),
        "training_eligible_unknown_count": len(set(violations)),
        "errors": [f"license_gate_violation:{value}" for value in sorted(set(violations))],
        "warnings": [],
    }


def build_canonical_benchmark(
    config: BenchmarkAuditConfig,
    *,
    parent_inputs: tuple[SpatialParentInput, ...],
    description_records: tuple[DescriptionSourceRecord, ...],
    output_dir: Path,
    schemas_root: Path,
    datasets_root: Path | None = None,
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
    language_inputs, description_parent_ids, language_forced_splits = _language_parent_inputs(
        config,
        description_records,
        datasets_root=datasets_root,
    )
    license_report = _license_report(
        config,
        parent_inputs,
        description_records,
        materialized_description_ids=set(description_parent_ids),
    )
    if license_report["errors"]:
        raise BenchmarkBuildError("license report blocks canonical materialization")

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
            + [
                {
                    "record_id": record.record_id,
                    "image_sha256": record.image.sha256,
                    "source_record_sha256": sha256_bytes(
                        canonical_json_bytes(record.model_dump(mode="json"))
                    ),
                }
                for record in sorted(description_records, key=lambda value: value.record_id)
            ]
        )
    )
    with atomic_output_directory(output_dir) as staging:
        materialized_spatial = tuple(
            materialize_spatial_parent(
                item,
                benchmark_root=staging,
                canvas_hw=config.build.materialization.canvas_hw,
            )
            for item in sorted(parent_inputs, key=lambda value: value.parent_id)
        )
        materialized_language = tuple(
            materialize_language_parent(
                item,
                benchmark_root=staging,
                canvas_hw=config.build.materialization.canvas_hw,
            )
            for item in language_inputs
        )
        audit_parents = tuple(item.parent for item in materialized_spatial) + tuple(
            item.parent for item in materialized_language
        )
        duplicates = build_duplicate_analysis(
            audit_parents,
            benchmark_root=staging,
            settings=config.build.duplicates,
        )
        merged_forced_splits = dict(language_forced_splits)
        for parent_id, split_name in (forced_splits or {}).items():
            previous = merged_forced_splits.get(parent_id)
            if previous is not None and previous != split_name:
                raise BenchmarkBuildError(f"conflicting forced split for parent: {parent_id}")
            merged_forced_splits[parent_id] = split_name
        split = assign_parent_splits(
            audit_parents,
            duplicate_clusters=duplicates.parent_to_cluster,
            settings=config.build.split,
            seed=config.seed,
            forced_splits=merged_forced_splits,
        )
        parents = apply_parent_splits(audit_parents, split)
        tasks = expand_task_views(
            parents,
            region_answers=region_answers,
            fixed_predictions=fixed_predictions,
        )
        parent_by_id = {parent.parent_id: parent for parent in parents}
        canonical_descriptions = _canonical_description_records(
            description_records,
            record_to_parent=description_parent_ids,
            parent_by_id=parent_by_id,
        )

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
                "settings": config.build.split.model_dump(mode="json"),
                "forced_splits": dict(sorted(merged_forced_splits.items())),
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
        canonical_description_rows = [record.model_dump(mode="json") for record in canonical_descriptions]
        _write_jsonl(staging, "descriptions/all.jsonl", canonical_description_rows)
        for split_name in ("train", "val", "test"):
            _write_jsonl(
                staging,
                f"descriptions/{split_name}.jsonl",
                [row for row in canonical_description_rows if row["split"] == split_name],
            )
        _write_jsonl(
            staging,
            "descriptions/train_eligible.jsonl",
            [
                row
                for row in canonical_description_rows
                if row["training_eligible"] and row["split"] == "train"
            ],
        )
        _write_json(staging, "reports/license_report.json", license_report)
        cross_split = sum(
            len({split.parent_to_split[parent_id] for parent_id in cluster["parent_ids"]}) > 1
            for cluster in duplicates.clusters
        )
        duplicate_report = {
            "schema_version": "sami_duplicate_report_v1",
            "protocol": DUPLICATE_PROTOCOL_VERSION,
            "settings": config.build.duplicates.model_dump(mode="json"),
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
            "canonical_description_record_count": len(canonical_descriptions),
            "language_parent_count": len(materialized_language),
            "spatial_parent_count": len(materialized_spatial),
            "description_train_eligible_count": sum(
                record.training_eligible and record.split == "train" for record in canonical_descriptions
            ),
            "valid_pixel_count": sum(item.valid_pixel_count for item in materialized_spatial)
            + sum(item.valid_pixel_count for item in materialized_language),
            "excluded_pixel_count": sum(item.excluded_pixel_count for item in materialized_spatial)
            + sum(item.excluded_pixel_count for item in materialized_language),
            "positive_valid_pixel_count": sum(
                item.positive_valid_pixel_count for item in materialized_spatial
            ),
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
            "description_contract_version": "sami_canonical_description_v2_component_license_bound",
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
