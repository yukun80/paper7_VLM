"""Pure gate primitives shared by training report builders and validators."""

from __future__ import annotations

import hashlib
import math
from typing import Any, Iterable, Mapping

from .output import canonical_description_json, parse_description_output
from .versions import (
    DESCRIPTION_GRADIENT_GATE_PROTOCOL,
    STRICT_RELOAD_PROBE_PROTOCOL,
    STRUCTURED_GENERATION_PROTOCOL,
)


CAUSAL_LABEL_HISTORY_FIELDS = (
    "causal_label_audit_passed",
    "causal_prefix_masked",
    "causal_target_contiguous",
    "causal_padding_masked",
    "causal_eos_supervised",
)

_DESCRIPTION_GRADIENT_MODULES = {
    "desc_adapter", "description_backbone", "mgrr", "region_projection",
    "global_visual_projection", "alignment",
}
_D_MINUS_ONE_TASK_PATHS = {"global_caption", "region_description"}
_STRUCTURED_ENUM_PATHS = {
    "target_status": ("target_status",),
    "region.location": ("region", "location"),
    "region.size_class": ("region", "size_class"),
    "region.shape": ("region", "shape"),
    "region.elongation": ("region", "elongation"),
    "region.compactness": ("region", "compactness"),
    "region.fragmentation": ("region", "fragmentation"),
    "evidence.terrain_support": ("evidence", "terrain_support"),
    "evidence.sar_support": ("evidence", "sar_support"),
    "evidence.deformation_support": ("evidence", "deformation_support"),
    "evidence.evidence_sufficiency": ("evidence", "evidence_sufficiency"),
}
_STRUCTURED_TEXT_PATHS = {
    "evidence.surface_observation",
    "evidence.surrounding_context",
    "summary",
}


def _nested_string(value: Mapping[str, Any], path: tuple[str, ...]) -> str | None:
    current: Any = value
    for field in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(field)
    return current if isinstance(current, str) else None


def structured_generation_audits_current(
    rows: Iterable[Mapping[str, Any]] | None,
) -> bool:
    """Replay that every structured row came from the current raw decoder."""

    structured = [
        row for row in (rows or []) if row.get("structured_output") is True
    ]
    if not structured:
        return False
    for row in structured:
        audit = row.get("generation_audit")
        raw = row.get("raw_generation")
        if not isinstance(audit, Mapping) or not isinstance(raw, str):
            return False
        parsed_result = parse_description_output(raw)
        parsed = parsed_result.parsed
        if not parsed_result.schema_valid or not isinstance(parsed, Mapping):
            return False
        try:
            canonical_raw = canonical_description_json(dict(parsed))
        except ValueError:
            return False
        if raw != canonical_raw:
            return False
        forced = audit.get("forced_tokens")
        selected = audit.get("model_selected_tokens")
        total = audit.get("total_tokens")
        maximum = audit.get("max_new_tokens")
        advance_calls = audit.get("decoder_advance_calls")
        if (
            audit.get("protocol") != STRUCTURED_GENERATION_PROTOCOL
            or audit.get("mode") != "schema_constrained_raw_generation"
            or audit.get("raw_schema_valid") is not True
            or audit.get("repair_used") is not False
            or audit.get("raw_sha256")
            != hashlib.sha256(raw.encode("utf-8")).hexdigest()
            or audit.get("token_stream_sha256")
            != hashlib.sha256(raw.encode("utf-8")).hexdigest()
            or audit.get("token_stream_matches_raw") is not True
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in (forced, selected, total, maximum, advance_calls)
            )
            or int(forced) <= 0
            or int(selected) <= 0
            or int(total) != int(forced) + int(selected)
            or not 0 < int(total) <= int(maximum)
            or not 0 < int(advance_calls) <= int(total)
            or not isinstance(audit.get("enum_choices"), Mapping)
            or dict(audit["enum_choices"]) != {
                name: _nested_string(parsed, path)
                for name, path in _STRUCTURED_ENUM_PATHS.items()
            }
            or not isinstance(audit.get("text_termination"), Mapping)
            or set(audit["text_termination"]) != _STRUCTURED_TEXT_PATHS
            or not all(
                isinstance(value, str) and bool(value)
                for value in audit["text_termination"].values()
            )
        ):
            return False
    return True


def d_minus_one_gradient_gate_passed(
    audit: Mapping[str, Any] | None,
) -> bool:
    """Deeply replay the two-path D-1 gradient proof, not summary booleans."""

    value = dict(audit or {})
    streams = value.get("streams")
    if (
        value.get("protocol") != DESCRIPTION_GRADIENT_GATE_PROTOCOL
        or value.get("run_stage") != "overfit"
        or value.get("required_streams") != ["main"]
        or value.get("checked_streams") != ["main"]
        or value.get("all_required_streams_checked") is not True
        or value.get("passed") is not True
        or not isinstance(streams, Mapping)
        or set(streams) != {"main"}
    ):
        return False
    stream = streams["main"]
    if not isinstance(stream, Mapping):
        return False
    path_reports = stream.get("path_reports")
    if (
        stream.get("required_task_paths")
        != sorted(_D_MINUS_ONE_TASK_PATHS)
        or stream.get("observed_task_paths")
        != sorted(_D_MINUS_ONE_TASK_PATHS)
        or stream.get("passed") is not True
        or not isinstance(path_reports, Mapping)
        or set(path_reports) != _D_MINUS_ONE_TASK_PATHS
    ):
        return False

    for task_path, raw_report in path_reports.items():
        if not isinstance(raw_report, Mapping):
            return False
        report = dict(raw_report)
        observed = report.get("observed_task_paths")
        if (
            report.get("protocol") != DESCRIPTION_GRADIENT_GATE_PROTOCOL
            or report.get("run_stage") != "overfit"
            or report.get("stream_name") != "main"
            or report.get("stream_stage") != "overfit"
            or report.get("passed") is not True
            or not isinstance(observed, list)
            or observed != [task_path]
        ):
            return False
        expected_nonzero = {
            "desc_adapter", "global_visual_projection",
        }
        expected_zero = {"alignment"}
        if "region_description" in observed:
            expected_nonzero.update({
                "description_backbone", "mgrr", "region_projection",
            })
        else:
            expected_zero.update({
                "description_backbone", "mgrr", "region_projection",
            })
        if (
            report.get("required_nonzero") != sorted(expected_nonzero)
            or report.get("required_zero") != sorted(expected_zero)
        ):
            return False
        modules = report.get("modules")
        checks = report.get("checks")
        expected_checks = {
            *(f"{name}_nonzero" for name in expected_nonzero),
            *(f"{name}_zero" for name in expected_zero),
            "all_trainable_gradients_finite",
        }
        if (
            not isinstance(modules, Mapping)
            or set(modules) != _DESCRIPTION_GRADIENT_MODULES
            or not isinstance(checks, Mapping)
            or set(checks) != expected_checks
            or not all(item is True for item in checks.values())
        ):
            return False
        for name, summary in modules.items():
            if not isinstance(summary, Mapping):
                return False
            count_values = [
                summary.get("num_parameters"),
                summary.get("num_with_grad"),
                summary.get("num_nonzero"),
            ]
            raw_norm = summary.get("norm_sum")
            if (
                any(
                    isinstance(item, bool) or not isinstance(item, int)
                    for item in count_values
                )
                or isinstance(raw_norm, bool)
                or not isinstance(raw_norm, (int, float))
            ):
                return False
            try:
                num_parameters = int(summary["num_parameters"])
                num_with_grad = int(summary["num_with_grad"])
                num_nonzero = int(summary["num_nonzero"])
                norm_sum = float(summary["norm_sum"])
            except (KeyError, TypeError, ValueError):
                return False
            if (
                set(summary) != {
                    "num_parameters", "num_with_grad", "num_nonzero",
                    "norm_sum", "all_finite",
                }
                or num_parameters < 0
                or not 0 <= num_nonzero <= num_with_grad <= num_parameters
                or not math.isfinite(norm_sum)
                or norm_sum < 0.0
                or summary.get("all_finite") is not True
                or (name in expected_nonzero and num_nonzero <= 0)
                or (name in expected_zero and num_nonzero != 0)
            ):
                return False
    return True


def audit_causal_label_history(
    history_rows: Iterable[Mapping[str, Any]] | None,
) -> dict[str, bool]:
    """Replay the per-step causal-label evidence stored in training history."""
    rows = list(history_rows or [])
    return {
        name: bool(rows) and all(
            not isinstance(row.get(name), bool)
            and isinstance(row.get(name), (int, float))
            and math.isfinite(float(row[name]))
            and float(row[name]) == 1.0
            for row in rows
        )
        for name in CAUSAL_LABEL_HISTORY_FIELDS
    }


def strict_reload_state_replay_passed(audit: Mapping[str, Any] | None) -> bool:
    """Require independently perturbed state to match the saved state after reload."""

    value = dict(audit or {})
    probe = value.get("state_probe")
    if not isinstance(probe, Mapping):
        return False
    expected = probe.get("expected_sha256")
    before = probe.get("before_sha256")
    corrupted = probe.get("corrupted_sha256")
    restored = probe.get("restored_sha256")
    fields = probe.get("corrupted_fields")
    if not all(
        isinstance(item, Mapping)
        for item in (expected, before, corrupted, restored, fields)
    ):
        return False
    required = {"optimizer", "scheduler", "rng", "scaler"}
    if any(set(item) != required for item in (expected, before, corrupted, restored, fields)):
        return False
    scaler_requested = value.get("grad_scaler_state_requested") is True
    perturbed_roles = {"optimizer", "scheduler", "rng"}
    if scaler_requested:
        perturbed_roles.add("scaler")
    state_matches = (
        dict(before) == dict(expected)
        and dict(restored) == dict(expected)
        and all(corrupted[name] != expected[name] for name in perturbed_roles)
        and all(isinstance(fields[name], str) and fields[name] for name in perturbed_roles)
    )
    if not scaler_requested:
        state_matches = state_matches and all(
            item.get("scaler") is None
            for item in (expected, before, corrupted, restored, fields)
        )
    return bool(
        value.get("protocol") == STRICT_RELOAD_PROBE_PROTOCOL
        and value.get("passed") is True
        and value.get("optimizer_state_restored") is True
        and value.get("scheduler_state_restored") is True
        and value.get("rng_state_restored") is True
        and value.get("grad_scaler_state_restored") is True
        and state_matches
    )
