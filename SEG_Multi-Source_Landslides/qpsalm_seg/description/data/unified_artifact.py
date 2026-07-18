"""Live Unified v3 index replay for SegDesc artifact readiness."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from qpsalm_seg.paths import resolve_project_path

from ..protocols.io import (
    nested_file_bindings_current,
    read_json,
    read_jsonl,
    sha256_file,
)


UNIFIED_BUILDER_VERSION = (
    "qpsalm_segdesc_index_builder_v3_component_contract_bound"
)
UNIFIED_VALIDATION_PROTOCOL = (
    "qpsalm_segdesc_index_validation_v3_component_contract_bound"
)
UNIFIED_STATISTICS_PROTOCOL = (
    "qpsalm_segdesc_index_statistics_v3_component_contract_bound"
)


def _report_binding(path: Path, report: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": str(path.resolve(strict=False)),
        "sha256": sha256_file(path),
        "bytes": int(path.stat().st_size),
        "protocol": report.get("protocol"),
        "builder_version": report.get("builder_version"),
        "status": report.get("status"),
    }


def revalidate_unified_artifact(
    unified: Path,
    *,
    mode: str,
    bridge_status: str,
    description_root: Path,
    bridge_root: Path,
) -> dict[str, Any]:
    """Replay current Unified v3 publications and their component bindings."""
    validation_path = unified / "reports/validation_report.json"
    statistics_path = unified / "reports/statistics.json"
    manifest_path = unified / "manifests/component_manifest.json"
    build_path = unified / "reports/build_report.json"
    all_index_path = unified / "indexes/all.jsonl"
    validation = read_json(validation_path, label="Unified validation")
    statistics = read_json(statistics_path, label="Unified statistics")
    manifest = read_json(manifest_path, label="Unified component manifest")
    build_report = read_json(build_path, label="Unified build report")
    all_rows = read_jsonl(all_index_path, label="Unified all index")
    record_ids = [str(row.get("unified_record_id") or "") for row in all_rows]
    live_by_split = dict(sorted(Counter(
        str(row.get("split") or "") for row in all_rows
    ).items()))
    live_by_task = dict(sorted(Counter(
        str(row.get("task_group") or "") for row in all_rows
    ).items()))
    split_rows: list[dict[str, Any]] = []
    split_files: dict[str, dict[str, Any]] = {}
    split_labels_current = True
    for split in ("train", "dev", "val", "test"):
        path = unified / f"indexes/{split}.jsonl"
        rows = read_jsonl(path, label=f"Unified {split} index")
        split_labels_current = split_labels_current and all(
            str(row.get("split") or "") == split for row in rows
        )
        split_rows.extend(rows)
        split_files[split] = {
            "path": str(path.resolve(strict=False)),
            "sha256": sha256_file(path),
            "records": len(rows),
        }
    split_by_id = {
        str(row.get("unified_record_id") or ""): row
        for row in split_rows
    }
    expert_expected = bridge_status == "expert_pilot_frozen"
    live_expert_records = sum(
        int(bool(row.get("expert_supervision"))) for row in all_rows
    )
    actual_expert_index_present = (
        bridge_root / "indexes/expert_all.jsonl"
    ).is_file()
    actual_bridge_gate_present = (
        bridge_root / "manifests/evaluation_gate_manifest.json"
    ).is_file()
    required_components = {"segmentation", "description", "bridge"}
    components = dict(manifest.get("components") or {})
    component_reports = dict(
        manifest.get("component_validation_reports") or {}
    )
    resolved_components = {
        name: resolve_project_path(str(value or ""))
        for name, value in components.items()
    }
    resolved_component_reports = {
        name: resolve_project_path(str(
            (binding or {}).get("path") or ""
        ))
        for name, binding in component_reports.items()
        if isinstance(binding, dict)
    }
    segmentation_instruction = dict(
        (component_reports.get("segmentation") or {}).get(
            "instruction_validation"
        ) or {}
    )
    checks = {
        "builder_current": (
            validation.get("builder_version") == UNIFIED_BUILDER_VERSION
            and statistics.get("builder_version") == UNIFIED_BUILDER_VERSION
        ),
        "protocols_current": (
            validation.get("protocol") == UNIFIED_VALIDATION_PROTOCOL
            and statistics.get("protocol") == UNIFIED_STATISTICS_PROTOCOL
        ),
        "mode_consistent": (
            validation.get("mode") == mode
            and statistics.get("mode") == mode
        ),
        "manifest_build_report_identical": manifest == build_report,
        "manifest_current": (
            manifest.get("builder_version") == UNIFIED_BUILDER_VERSION
            and manifest.get("schema_version") == "qpsalm_segdesc_index_v1"
            and manifest.get("mode") == mode
            and manifest.get("storage_mode") == "component_references_only"
        ),
        "component_inventory_current": (
            set(components) == required_components
            and set(component_reports) == required_components
            and all(
                isinstance(value, str) and bool(value.strip())
                for value in components.values()
            )
            and all(
                path is not None and path.is_dir()
                for path in resolved_components.values()
            )
            and resolved_components.get("description") is not None
            and resolved_components["description"].resolve(strict=False)
            == description_root.resolve(strict=False)
            and resolved_components.get("bridge") is not None
            and resolved_components["bridge"].resolve(strict=False)
            == bridge_root.resolve(strict=False)
        ),
        "component_report_paths_current": (
            set(resolved_components) == required_components
            and set(resolved_component_reports) == required_components
            and all(
                resolved_components.get(name) is not None
                and resolved_component_reports.get(name) is not None
                and isinstance(component_reports[name], dict)
                and len(str(component_reports[name].get("sha256") or "")) == 64
                and resolved_component_reports[name].resolve(strict=False)
                == (
                    resolved_components[name]
                    / "reports/validation_report.json"
                ).resolve(strict=False)
                for name in required_components
            )
            and len(str(segmentation_instruction.get("sha256") or "")) == 64
            and bool(str(segmentation_instruction.get("path") or "").strip())
        ),
        "component_reports_mirrored": (
            validation.get("component_validation_reports")
            == manifest.get("component_validation_reports")
            == statistics.get("component_validation_reports")
        ),
        "component_report_bindings_current": nested_file_bindings_current(
            component_reports
        ),
        "validation_clean": (
            validation.get("status") == "valid"
            and validation.get("errors") == []
            and validation.get("component_contracts_verified") is True
        ),
        "population_consistent": (
            int(validation.get("num_records", -1))
            == int(statistics.get("num_records", -2)) > 0
            and len(all_rows) == int(validation.get("num_records", -1))
            and len(record_ids) == len(set(record_ids))
            and all(record_ids)
            and live_by_split == manifest.get("by_split")
            == validation.get("by_split")
            == statistics.get("by_split")
            and live_by_task == manifest.get("by_task_group")
            == validation.get("by_task_group")
            == statistics.get("by_task_group")
        ),
        "split_indexes_exact_partition": (
            split_labels_current
            and len(split_rows) == len(all_rows)
            and set(split_by_id) == set(record_ids)
            and all(
                split_by_id[record_id] == row
                for record_id, row in zip(record_ids, all_rows)
            )
        ),
        "bridge_status_consistent": (
            manifest.get("bridge_status") == bridge_status
            and validation.get("bridge_status") == bridge_status
            and statistics.get("bridge_status") == bridge_status
        ),
        "expert_publication_consistent": (
            manifest.get("expert_index_published") is expert_expected
            and validation.get("expert_index_published") is expert_expected
            and statistics.get("expert_index_published") is expert_expected
            and int(statistics.get("expert_records", -1))
            == live_expert_records
            and (live_expert_records > 0) is expert_expected
            and manifest.get("contains_expert_bridge") is expert_expected
        ),
        "expert_artifacts_consistent": (
            manifest.get("bridge_gate") == validation.get("bridge_gate")
            == statistics.get("bridge_gate")
            and nested_file_bindings_current(manifest.get("bridge_gate"))
            and (
                isinstance(manifest.get("bridge_gate"), dict)
                if expert_expected else manifest.get("bridge_gate") is None
            )
            and bool(manifest.get("expert_index_present"))
            == bool(statistics.get("expert_index_present"))
            == actual_expert_index_present
            and bool(manifest.get("stale_expert_index_ignored"))
            == bool(statistics.get("stale_expert_index_ignored"))
            == (actual_expert_index_present and not expert_expected)
            and bool(manifest.get("stale_bridge_gate_ignored"))
            == bool(statistics.get("stale_bridge_gate_ignored"))
            == (actual_bridge_gate_present and not expert_expected)
        ),
    }
    return {
        "manifest": _report_binding(manifest_path, manifest),
        "build_report": _report_binding(build_path, build_report),
        "validation": _report_binding(validation_path, validation),
        "statistics": _report_binding(statistics_path, statistics),
        "all_index": {
            "path": str(all_index_path.resolve(strict=False)),
            "sha256": sha256_file(all_index_path),
            "records": len(all_rows),
        },
        "split_indexes": split_files,
        "checks": checks,
        "num_records": validation.get("num_records"),
        "expert_index_published": validation.get("expert_index_published"),
    }
