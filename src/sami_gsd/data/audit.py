"""Read-only raw-source scanner for Canonical Benchmark v3 P1."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from sami_gsd.contracts.config import BenchmarkAuditConfig
from sami_gsd.utilities.artifacts import (
    atomic_output_directory,
    atomic_write_bytes,
    canonical_json_bytes,
    canonical_yaml_bytes,
    sha256_bytes,
    sha256_file,
)


AUDIT_BUILDER_VERSION = "sami_source_audit_v2_component_license_bound"


def _raise_walk_error(error: OSError) -> None:
    """Turn an ``os.walk`` access failure into an explicit audit failure."""

    raise error


def _scan_source_files(
    *,
    source_root: Path,
    logical_root: str,
    include_hidden: bool,
) -> tuple[list[dict[str, Any]], int]:
    """Return stable file records and an ignored-symlink count.

    Source bytes and metadata are only read. Symbolic links are never followed,
    which keeps the configured source root as the scanner's hard boundary.
    """

    records: list[dict[str, Any]] = []
    ignored_symlinks = 0
    for directory, directory_names, file_names in os.walk(
        source_root,
        followlinks=False,
        onerror=_raise_walk_error,
    ):
        directory_path = Path(directory)
        retained_directories: list[str] = []
        for name in sorted(directory_names):
            candidate = directory_path / name
            if candidate.is_symlink():
                ignored_symlinks += 1
            elif include_hidden or not name.startswith("."):
                retained_directories.append(name)
        directory_names[:] = retained_directories
        for name in sorted(file_names):
            candidate = directory_path / name
            if not include_hidden and name.startswith("."):
                continue
            if candidate.is_symlink():
                ignored_symlinks += 1
                continue
            if not candidate.is_file():
                continue
            relative_path = candidate.relative_to(source_root).as_posix()
            stat = candidate.stat()
            records.append(
                {
                    "logical_path": f"{logical_root}/{relative_path}",
                    "sha256": sha256_file(candidate),
                    "size_bytes": stat.st_size,
                    "suffix": candidate.suffix.lower(),
                }
            )
    records.sort(key=lambda item: item["logical_path"])
    return records, ignored_symlinks


def _build_inventory(config: BenchmarkAuditConfig, *, datasets_root: Path) -> dict[str, Any]:
    """Build a deterministic inventory without exposing the machine root."""

    source_records: list[dict[str, Any]] = []
    warnings: list[str] = []
    for source in sorted(config.sources, key=lambda item: item.source_key):
        if not source.enabled:
            source_records.append(
                {
                    "source_key": source.source_key,
                    "status": "disabled",
                    "logical_root": f"datasets/{source.local_path}",
                    "file_count": 0,
                    "total_bytes": 0,
                    "ignored_symlinks": 0,
                    "files": [],
                    "aggregate_sha256": sha256_bytes(canonical_json_bytes([])),
                }
            )
            continue
        source_root = datasets_root / source.local_path
        logical_root = f"datasets/{source.local_path}"
        if not source_root.exists():
            warnings.append(f"source_missing:{source.source_key}")
            files: list[dict[str, Any]] = []
            ignored_symlinks = 0
            status = "missing"
        elif source_root.is_symlink():
            raise ValueError(f"configured source root must not be a symbolic link: {source.source_key}")
        elif not source_root.is_dir():
            raise NotADirectoryError(f"configured source is not a directory: {source.source_key}")
        else:
            files, ignored_symlinks = _scan_source_files(
                source_root=source_root,
                logical_root=logical_root,
                include_hidden=config.audit.include_hidden,
            )
            status = "present"
        source_records.append(
            {
                "source_key": source.source_key,
                "status": status,
                "logical_root": logical_root,
                "file_count": len(files),
                "total_bytes": sum(item["size_bytes"] for item in files),
                "ignored_symlinks": ignored_symlinks,
                "files": files,
                "aggregate_sha256": sha256_bytes(canonical_json_bytes(files)),
            }
        )
    payload: dict[str, Any] = {
        "schema_version": "sami_raw_source_inventory_v1",
        "builder_version": AUDIT_BUILDER_VERSION,
        "mode": config.mode,
        "seed": config.seed,
        "sources": source_records,
        "warnings": sorted(warnings),
        "errors": [],
    }
    payload["aggregate_sha256"] = sha256_bytes(canonical_json_bytes(payload))
    return payload


def _build_registry(config: BenchmarkAuditConfig) -> dict[str, Any]:
    """Project configured source licenses into the canonical registry shape."""

    entries: list[dict[str, Any]] = []
    for source in sorted(config.sources, key=lambda item: item.source_key):
        license_payload = source.license.model_dump(mode="json")
        entries.append(
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
                **license_payload,
            }
        )
    return {
        "schema_version": "sami_source_registry_v2_component_license_bound",
        "entries": entries,
    }


def _build_license_report(registry: dict[str, Any]) -> dict[str, Any]:
    """Summarize eligibility and prove unknown sources cannot enter training."""

    entries = registry["entries"]
    scopes = [
        (entry["source_key"], entry)
        for entry in entries
    ] + [
        (component["component_key"], component)
        for entry in entries
        for component in entry["language_components"]
    ]
    unknown = [scope_key for scope_key, license_payload in scopes if license_payload["license_status"] == "unknown"]
    eligible = [scope_key for scope_key, license_payload in scopes if license_payload["allowed_for_training"]]
    violations = [
        scope_key
        for scope_key, license_payload in scopes
        if license_payload["allowed_for_training"]
        and (
            license_payload["license_status"] == "unknown"
            or license_payload["license_name"].lower() == "unknown"
            or license_payload["reviewed_by"] is None
            or license_payload["review_date"] is None
        )
    ]
    return {
        "schema_version": "sami_license_report_v2_component_license_bound",
        "builder_version": AUDIT_BUILDER_VERSION,
        "source_count": len(entries),
        "language_component_count": sum(len(entry["language_components"]) for entry in entries),
        "training_eligible_sources": sorted(eligible),
        "unknown_license_sources": sorted(unknown),
        "training_eligible_unknown_count": len(violations),
        "errors": [f"training_eligible_unknown:{source_key}" for source_key in sorted(violations)],
        "warnings": [f"license_unknown:{source_key}" for source_key in sorted(unknown)],
    }


def audit_sources(
    config: BenchmarkAuditConfig,
    *,
    datasets_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Scan configured raw roots and atomically publish a deterministic audit.

    Args:
        config: Validated P1 audit configuration.
        datasets_root: Runtime-only filesystem root containing raw source roots.
        output_dir: New output directory; existing paths are rejected.

    Returns:
        The published audit manifest.

    Raises:
        FileExistsError: The output directory already exists.
        NotADirectoryError: A configured source path exists but is not a directory.
        OSError: Source reads, hashing or publication fail.
    """

    inventory = _build_inventory(config, datasets_root=datasets_root.resolve())
    registry = _build_registry(config)
    license_report = _build_license_report(registry)
    if license_report["errors"]:
        raise ValueError("license registry contains a training-eligible unknown source")

    inventory_bytes = canonical_json_bytes(inventory)
    registry_bytes = canonical_yaml_bytes(registry)
    license_bytes = canonical_json_bytes(license_report)
    config_bytes = canonical_json_bytes(config.model_dump(mode="json"))
    output_hashes = {
        "inventory.json": sha256_bytes(inventory_bytes),
        "license_report.json": sha256_bytes(license_bytes),
        "source_registry.yaml": sha256_bytes(registry_bytes),
    }
    aggregate_input = {
        "builder_version": AUDIT_BUILDER_VERSION,
        "config_sha256": sha256_bytes(config_bytes),
        "output_sha256": output_hashes,
        "seed": config.seed,
    }
    manifest: dict[str, Any] = {
        "schema_version": "sami_source_audit_manifest_v1",
        "builder_version": AUDIT_BUILDER_VERSION,
        "benchmark_name": config.benchmark_name,
        "mode": config.mode,
        "seed": config.seed,
        "config_sha256": aggregate_input["config_sha256"],
        "output_sha256": output_hashes,
        "aggregate_sha256": sha256_bytes(canonical_json_bytes(aggregate_input)),
        "errors": [],
        "warnings": inventory["warnings"] + license_report["warnings"],
    }
    manifest_bytes = canonical_json_bytes(manifest)

    with atomic_output_directory(output_dir) as staging:
        atomic_write_bytes(staging / "inventory.json", inventory_bytes)
        atomic_write_bytes(staging / "source_registry.yaml", registry_bytes)
        atomic_write_bytes(staging / "license_report.json", license_bytes)
        atomic_write_bytes(staging / "audit_manifest.json", manifest_bytes)
    return manifest
