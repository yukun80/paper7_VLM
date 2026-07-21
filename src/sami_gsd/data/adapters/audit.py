"""Sample-bounded live audit over the unique P1.3 source adapter registry."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sami_gsd.contracts.config import BenchmarkAuditConfig
from sami_gsd.data.adapters.base import SourceAdapterError
from sami_gsd.data.adapters.registry import build_source_adapter_registry
from sami_gsd.utilities.artifacts import canonical_json_bytes, sha256_bytes, sha256_file


SOURCE_SAMPLE_AUDIT_VERSION = "sami_source_sample_audit_p1_3_v1"


def _physical_asset_path(source_root: Path, logical_path: str, configured_local_path: str) -> Path:
    """Resolve a previously validated logical asset inside its configured root."""

    prefix = f"datasets/{configured_local_path}/"
    if not logical_path.startswith(prefix):
        raise ValueError("adapter emitted an asset outside its configured logical root")
    relative = logical_path[len(prefix) :]
    path = source_root / relative
    if not path.is_file():
        raise FileNotFoundError(f"sample asset disappeared during audit: {logical_path}")
    return path


def audit_source_samples(
    config: BenchmarkAuditConfig,
    *,
    datasets_root: Path,
    limit_per_source: int = 8,
) -> dict[str, Any]:
    """Account for all configured sources with bounded, read-only sample extraction.

    Blocked adapters are reported without being called.  Implemented adapters
    hash only their selected sample assets, then re-hash those same files after
    projection to verify that raw bytes were not changed.
    """

    if type(limit_per_source) is not int or limit_per_source <= 0:
        raise ValueError("limit_per_source must be a positive integer")
    registry = build_source_adapter_registry()
    configured_keys = tuple(sorted(source.source_key for source in config.sources))
    if configured_keys != registry.keys():
        raise ValueError("configured sources and the unique adapter registry do not match exactly")

    source_reports: list[dict[str, Any]] = []
    for source in sorted(config.sources, key=lambda value: value.source_key):
        adapter = registry.get(source.source_key)
        descriptor = adapter.descriptor
        source_root = datasets_root / source.local_path
        blockers = list(descriptor.blockers)
        errors: list[str] = []
        projections = ()
        raw_bytes_unchanged: bool | None = None
        if not source_root.exists():
            status = "missing"
            blockers.append("configured_source_root_missing")
        elif source_root.is_symlink() or not source_root.is_dir():
            status = "blocked_error"
            errors.append("configured_source_root_not_a_plain_directory")
        elif descriptor.implementation_status == "blocked":
            status = "blocked"
        else:
            try:
                projections = adapter.extract_samples(
                    source_root,
                    source,
                    limit=limit_per_source,
                )
            except SourceAdapterError as error:
                status = "blocked_error"
                errors.append(str(error))
            else:
                if not projections:
                    status = "blocked_error"
                    errors.append("implemented_adapter_emitted_no_sample")
                else:
                    status = "sampled"
                    comparisons: list[bool] = []
                    for projection in projections:
                        for asset in projection.raw_record.assets:
                            physical_path = _physical_asset_path(source_root, asset.logical_path, source.local_path)
                            comparisons.append(sha256_file(physical_path) == asset.sha256)
                    raw_bytes_unchanged = all(comparisons)
                    if not raw_bytes_unchanged:
                        errors.append("sample_source_bytes_changed_during_audit")
                        status = "blocked_error"

        projection_payloads = [projection.model_dump(mode="json") for projection in projections]
        projection_hashes = [projection.projection_sha256 for projection in projections]
        candidate_blockers = sorted(
            {
                blocker
                for projection in projections
                for blocker in projection.canonical_candidate.blockers
            }
        )
        grouping_evidence = sorted(
            {
                evidence
                for projection in projections
                for evidence in projection.raw_record.grouping_evidence
            }
        )
        ambiguity_flags = sorted(
            {
                flag
                for projection in projections
                for flag in projection.raw_record.ambiguity_flags
            }
        )
        grouping_blockers = sorted(
            blocker
            for blocker in candidate_blockers
            if "group" in blocker or "scene_identity" in blocker or "provenance" in blocker
        )
        source_report = {
            "source_key": source.source_key,
            "logical_root": f"datasets/{source.local_path}",
            "present": source_root.is_dir() and not source_root.is_symlink(),
            "adapter_version": descriptor.adapter_version,
            "implementation_status": descriptor.implementation_status,
            "status": status,
            "sample_limit": limit_per_source,
            "sample_count": len(projections),
            "sample_projection_sha256": projection_hashes,
            "sample_aggregate_sha256": sha256_bytes(canonical_json_bytes(projection_payloads)),
            "raw_bytes_unchanged": raw_bytes_unchanged,
            "training_eligible": False,
            "blockers": sorted(set(blockers) | set(candidate_blockers)),
            "grouping_evidence": grouping_evidence,
            "ambiguity_flags": ambiguity_flags,
            "grouping_status": "closed" if projections and not grouping_blockers else "blocked",
            "grouping_blockers": grouping_blockers,
            "audit_canonical_candidate_count": len(projections),
            "canonical_parent_materialization_eligible_count": sum(
                not projection.canonical_candidate.blockers
                and projection.canonical_candidate.license.allowed_for_training
                for projection in projections
            ),
            "errors": errors,
        }
        source_reports.append(source_report)

    report: dict[str, Any] = {
        "schema_version": "sami_source_sample_audit_report_v1",
        "builder_version": SOURCE_SAMPLE_AUDIT_VERSION,
        "mode": config.mode,
        "seed": config.seed,
        "sample_limit_per_source": limit_per_source,
        "source_count": len(source_reports),
        "implemented_source_count": sum(
            report["implementation_status"] == "implemented" for report in source_reports
        ),
        "sampled_source_count": sum(report["status"] == "sampled" for report in source_reports),
        "blocked_source_count": sum(report["status"].startswith("blocked") for report in source_reports),
        "missing_source_count": sum(report["status"] == "missing" for report in source_reports),
        "sources": source_reports,
        "canonical_dry_run": {
            "candidate_count": sum(report["audit_canonical_candidate_count"] for report in source_reports),
            "materialization_eligible_count": sum(
                report["canonical_parent_materialization_eligible_count"] for report in source_reports
            ),
            "source_group_closed_count": sum(report["grouping_status"] == "closed" for report in source_reports),
            "source_group_blocked_count": sum(report["grouping_status"] == "blocked" for report in source_reports),
            "training_promotion_performed": False,
        },
        "errors": [
            f"{report['source_key']}:{error}"
            for report in source_reports
            for error in report["errors"]
        ],
    }
    report["aggregate_sha256"] = sha256_bytes(canonical_json_bytes(report))
    return report


__all__ = ["SOURCE_SAMPLE_AUDIT_VERSION", "audit_source_samples"]
