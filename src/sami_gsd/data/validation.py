"""Independent Canonical Benchmark v3 on-disk validator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator

from sami_gsd.contracts.canonical import CanonicalParentV3, TaskViewV3
from sami_gsd.contracts.language import DescriptionSourceRecord
from sami_gsd.utilities.artifacts import canonical_json_bytes, sha256_bytes, sha256_file


VALIDATION_VERSION = "sami_benchmark_validation_v1"


def _strict_json(text: str) -> Any:
    """Decode JSON while rejecting non-standard constants."""

    return json.loads(
        text,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"non-finite JSON constant: {value}")),
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file with no blank records."""

    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"blank JSONL row: {path}:{line_number}")
        value = _strict_json(line)
        if not isinstance(value, dict):
            raise ValueError(f"JSONL row is not an object: {path}:{line_number}")
        rows.append(value)
    return rows


def _forbidden_field_paths(value: Any, *, location: str = "$") -> list[str]:
    """Find exact pre/post/change model fields without substring false positives."""

    errors: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            lowered = key.lower()
            if lowered in {"pre", "post", "change", "pre_image", "post_image", "change_mask"}:
                errors.append(f"forbidden_field:{location}.{key}")
            errors.extend(_forbidden_field_paths(item, location=f"{location}.{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            errors.extend(_forbidden_field_paths(item, location=f"{location}[{index}]"))
    return errors


def validate_benchmark_payload(
    benchmark_root: Path,
    *,
    schemas_root: Path,
) -> dict[str, Any]:
    """Reopen indexes/assets and recompute the P1 semantic validation report."""

    errors: list[str] = []
    warnings: list[str] = []
    required = (
        "manifests/source_registry.yaml",
        "manifests/split_manifest.json",
        "manifests/duplicate_clusters.jsonl",
        "manifests/evaluation_conditions.json",
        "manifests/description_source_subset.jsonl",
        "manifests/description_train_eligible.jsonl",
        "parents/all.jsonl",
        "parents/train.jsonl",
        "parents/val.jsonl",
        "parents/test.jsonl",
        "reports/license_report.json",
        "reports/duplicate_report.json",
        "reports/summary_report.json",
    )
    for relative in required:
        if not (benchmark_root / relative).is_file():
            errors.append(f"missing_required_artifact:{relative}")
    part_files = sorted(path.relative_to(benchmark_root).as_posix() for path in benchmark_root.rglob("*.part-*"))
    errors.extend(f"part_file_present:{path}" for path in part_files)
    if errors:
        report: dict[str, Any] = {
            "schema_version": "sami_benchmark_validation_report_v1",
            "validator_version": VALIDATION_VERSION,
            "parent_count": 0,
            "task_count": 0,
            "verified_duplicate_cross_split_count": 0,
            "training_eligible_unknown_count": 0,
            "errors": sorted(errors),
            "warnings": warnings,
        }
        report["aggregate_sha256"] = sha256_bytes(canonical_json_bytes(report))
        return report

    parent_schema = _strict_json((schemas_root / "canonical_parent_v3.schema.json").read_text(encoding="utf-8"))
    task_schema = _strict_json((schemas_root / "task_view_v3.schema.json").read_text(encoding="utf-8"))
    parent_validator = Draft202012Validator(parent_schema)
    task_validator = Draft202012Validator(task_schema)

    raw_parents = _read_jsonl(benchmark_root / "parents/all.jsonl")
    parents: list[CanonicalParentV3] = []
    for index, payload in enumerate(raw_parents):
        try:
            parent_validator.validate(payload)
            parents.append(CanonicalParentV3.model_validate(payload))
        except Exception as error:
            errors.append(f"invalid_parent:{index}:{error}")
        errors.extend(_forbidden_field_paths(payload, location=f"parents[{index}]"))
    parent_ids = [parent.parent_id for parent in parents]
    if len(parent_ids) != len(set(parent_ids)):
        errors.append("duplicate_parent_id")
    parent_by_id = {parent.parent_id: parent for parent in parents}
    if any(parent.split == "audit" for parent in parents):
        errors.append("audit_split_parent_in_final_index")

    for split in ("train", "val", "test"):
        rows = _read_jsonl(benchmark_root / f"parents/{split}.jsonl")
        expected = [
            parent.model_dump(mode="json")
            for parent in sorted(parents, key=lambda item: item.parent_id)
            if parent.split == split
        ]
        if rows != expected:
            errors.append(f"parent_split_projection_mismatch:{split}")

    for parent in parents:
        for relative, expected_hash in parent.hashes.assets.items():
            del relative
            if expected_hash not in {
                reference.sha256
                for reference in (
                    [parent.annotations.global_landslide_mask]
                    if parent.annotations.global_landslide_mask is not None
                    else []
                )
            } | {region.mask_ref.sha256 for region in parent.annotations.referring_regions} | {
                value for modality in parent.modalities for value in modality.hashes.values()
            }:
                errors.append(f"unbound_parent_asset_hash:{parent.parent_id}:{expected_hash}")
        asset_refs = []
        if parent.annotations.global_landslide_mask is not None:
            asset_refs.append(parent.annotations.global_landslide_mask)
        asset_refs.extend(region.mask_ref for region in parent.annotations.referring_regions)
        for modality in parent.modalities:
            for path_value, hash_key in (
                (modality.native_asset_path, "native"),
                (modality.aligned_asset_path, "aligned"),
                (modality.valid_mask_path, "valid"),
            ):
                if path_value is not None:
                    expected = modality.hashes[hash_key]
                    physical = benchmark_root / path_value
                    if not physical.is_file() or sha256_file(physical) != expected:
                        errors.append(f"asset_hash_mismatch:{parent.parent_id}:{path_value}")
        for reference in asset_refs:
            physical = benchmark_root / reference.path
            if not physical.is_file() or sha256_file(physical) != reference.sha256:
                errors.append(f"asset_hash_mismatch:{parent.parent_id}:{reference.path}")

    tasks: list[TaskViewV3] = []
    for task_type in ("t1_global", "t2_referring", "t3_gt_region", "t4_predicted_region"):
        for split in ("train", "val", "test"):
            path = benchmark_root / f"tasks/{task_type}/{split}.jsonl"
            if not path.is_file():
                errors.append(f"missing_task_index:{task_type}/{split}")
                continue
            for row_number, payload in enumerate(_read_jsonl(path)):
                try:
                    task_validator.validate(payload)
                    task = TaskViewV3.model_validate(payload)
                except Exception as error:
                    errors.append(f"invalid_task:{task_type}:{split}:{row_number}:{error}")
                    continue
                parent = parent_by_id.get(task.parent_id)
                if parent is None:
                    errors.append(f"task_parent_missing:{task.task_id}")
                elif parent.split != split:
                    errors.append(f"task_parent_split_mismatch:{task.task_id}")
                if task.task_type != task_type:
                    errors.append(f"task_type_projection_mismatch:{task.task_id}")
                tasks.append(task)
                errors.extend(_forbidden_field_paths(payload, location=f"tasks[{task.task_id}]"))
    task_ids = [task.task_id for task in tasks]
    if len(task_ids) != len(set(task_ids)):
        errors.append("duplicate_task_id")

    duplicate_rows = _read_jsonl(benchmark_root / "manifests/duplicate_clusters.jsonl")
    cross_split = 0
    covered: set[str] = set()
    for row in duplicate_rows:
        members = row.get("parent_ids", [])
        covered.update(members)
        splits = {parent_by_id[parent_id].split for parent_id in members if parent_id in parent_by_id}
        if len(splits) > 1:
            cross_split += 1
    if covered != set(parent_ids):
        errors.append("duplicate_cluster_parent_coverage_mismatch")
    if cross_split:
        errors.append(f"verified_duplicate_cross_split:{cross_split}")

    registry = yaml.safe_load((benchmark_root / "manifests/source_registry.yaml").read_text(encoding="utf-8"))
    entries = registry.get("entries", []) if isinstance(registry, dict) else []
    violations = [
        entry.get("source_key", "unknown")
        for entry in entries
        if entry.get("allowed_for_training")
        and (
            entry.get("license_status") == "unknown"
            or str(entry.get("license_name", "")).lower() == "unknown"
            or entry.get("reviewed_by") is None
            or entry.get("review_date") is None
        )
    ]
    errors.extend(f"training_eligible_unknown:{key}" for key in violations)

    description_rows = _read_jsonl(benchmark_root / "manifests/description_source_subset.jsonl")
    train_description_rows = _read_jsonl(benchmark_root / "manifests/description_train_eligible.jsonl")
    descriptions: list[DescriptionSourceRecord] = []
    for index, payload in enumerate(description_rows):
        try:
            descriptions.append(DescriptionSourceRecord.model_validate(payload))
        except Exception as error:
            errors.append(f"invalid_description_source:{index}:{error}")
    expected_train = [
        record.model_dump(mode="json")
        for record in descriptions
        if record.training_eligible and record.split_policy != "permanent_test_only"
    ]
    if train_description_rows != expected_train:
        errors.append("description_train_projection_mismatch")
    if train_description_rows:
        errors.append("training_description_assets_not_materialized_as_canonical_parents")

    if not parents:
        errors.append("small_has_no_canonical_parents")
    if not any(parent.license.allowed_for_training for parent in parents):
        errors.append("small_has_no_training_eligible_parent")
    for task_type in ("t3_gt_region", "t4_predicted_region"):
        if not any(task.task_type == task_type for task in tasks):
            warnings.append(f"task_view_empty_until_bound_inputs:{task_type}")

    report = {
        "schema_version": "sami_benchmark_validation_report_v1",
        "validator_version": VALIDATION_VERSION,
        "parent_count": len(parents),
        "task_count": len(tasks),
        "verified_duplicate_cross_split_count": cross_split,
        "training_eligible_unknown_count": len(violations),
        "errors": sorted(set(errors)),
        "warnings": sorted(set(warnings)),
    }
    report["aggregate_sha256"] = sha256_bytes(canonical_json_bytes(report))
    return report


def validate_published_benchmark(
    benchmark_root: Path,
    *,
    schemas_root: Path,
) -> dict[str, Any]:
    """Validate manifest file hashes and replay the semantic report exactly."""

    report = validate_benchmark_payload(benchmark_root, schemas_root=schemas_root)
    report_path = benchmark_root / "reports/validation_report.json"
    manifest_path = benchmark_root / "manifests/benchmark_manifest.json"
    if not report_path.is_file() or not manifest_path.is_file():
        report["errors"] = sorted(set(report["errors"] + ["published_manifest_or_validation_report_missing"]))
        report["aggregate_sha256"] = sha256_bytes(canonical_json_bytes({k: v for k, v in report.items() if k != "aggregate_sha256"}))
        return report
    published_report = _strict_json(report_path.read_text(encoding="utf-8"))
    if published_report != report:
        report["errors"] = sorted(set(report["errors"] + ["validation_report_replay_mismatch"]))
    manifest = _strict_json(manifest_path.read_text(encoding="utf-8"))
    for relative, expected in manifest.get("output_sha256", {}).items():
        physical = benchmark_root / relative
        if not physical.is_file() or sha256_file(physical) != expected:
            report["errors"] = sorted(set(report["errors"] + [f"manifest_hash_mismatch:{relative}"]))
    report["aggregate_sha256"] = sha256_bytes(
        canonical_json_bytes({key: value for key, value in report.items() if key != "aggregate_sha256"})
    )
    return report


__all__ = ["VALIDATION_VERSION", "validate_benchmark_payload", "validate_published_benchmark"]
