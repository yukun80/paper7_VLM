"""Mask/modality counterfactual rows and formal sensitivity gate."""

from __future__ import annotations

from collections import Counter, defaultdict
import math
from pathlib import Path
from typing import Any

from qpsalm_seg.paths import resolve_project_path

from ..protocols.io import canonical_sha256, strict_json_loads
from ..protocols.output import parse_description_output
from .contracts import COUNTERFACTUAL_INPUT_AUDIT_PROTOCOL
from .metrics import (
    bootstrap_mean_ci,
    caption_token_f1,
    structured_disagreement,
    unsupported_claim_counts,
)




def _score(row: dict[str, Any]) -> float:
    metrics = row.get("raw_metrics") or {}
    if metrics.get("raw_field_accuracy") is not None:
        return float(metrics["raw_field_accuracy"])
    return float(metrics.get("caption_token_f1") or 0.0)


def claim_rate(rows: list[dict[str, Any]]) -> float:
    unsupported = sum(int((row.get("raw_metrics") or {}).get("unsupported_claims") or 0) for row in rows)
    claims = sum(int((row.get("raw_metrics") or {}).get("factual_claims") or 0) for row in rows)
    return unsupported / max(claims, 1)


def load_counterfactual_rows(directory: str | Path) -> list[dict[str, Any]]:
    root = resolve_project_path(directory) or Path(directory)
    path = root / "counterfactual_generations.jsonl"
    rows = [
        strict_json_loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    keys = [
        (str(row.get("sample_id") or ""), str(row.get("mode") or ""))
        for row in rows
    ]
    duplicates = sorted(value for value, count in Counter(keys).items() if count > 1)
    if duplicates:
        raise ValueError(f"counterfactual generations 存在重复 sample/mode: {duplicates[:8]}")
    return rows


def _parent_counterfactual_values(
    rows: list[dict[str, Any]], mode: str, field: str,
) -> list[float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if str(row.get("mode") or "") == mode:
            grouped[str(row["parent_sample_id"])].append(float(row[field]))
    return [sum(values) / len(values) for _parent, values in sorted(grouped.items())]


def _validate_counterfactual_row_bindings(
    rows: list[dict[str, Any]],
    generation_rows: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Replay deltas and prove every counted perturbation changed model input."""

    mask_modes = {
        "full_mask", "zero_mask", "shuffled_mask", "region_swap",
        "cross_parent_region_swap",
    }
    state_modes = {"modality_removal", "cross_parent_modality_swap"}
    allowed_modes = mask_modes | state_modes
    for index, row in enumerate(rows):
        sample_id = str(row.get("sample_id") or "")
        mode = str(row.get("mode") or "")
        base = generation_rows.get(sample_id)
        if base is None or mode not in allowed_modes:
            raise ValueError(
                f"counterfactual row 未绑定正式 generation/mode: row={index} "
                f"sample={sample_id!r} mode={mode!r}"
            )
        if (
            str(row.get("parent_sample_id") or "")
            != str(base.get("parent_sample_id") or "")
            or str(row.get("baseline_generation") or "")
            != str(base.get("raw_generation") or "")
        ):
            raise ValueError("counterfactual baseline generation/parent 已与正式行漂移")

        change = row.get("input_change_audit")
        if (
            not isinstance(change, dict)
            or change.get("protocol") != COUNTERFACTUAL_INPUT_AUDIT_PROTOCOL
            or change.get("mode") != mode
            or change.get("changed") is not True
        ):
            raise ValueError("counterfactual row 缺少可重放 input-change audit")
        dimensions = change.get("changed_dimensions")
        if (
            not isinstance(dimensions, list)
            or len(dimensions) != len(set(dimensions))
            or not set(dimensions).issubset({"region_mask", "backbone_state"})
            or (mode in mask_modes and "region_mask" not in dimensions)
            or (mode in state_modes and "backbone_state" not in dimensions)
        ):
            raise ValueError("counterfactual input-change dimension 与 mode 不一致")
        for prefix in ("region_mask", "backbone_state"):
            left = str(change.get(f"baseline_{prefix}_sha256") or "")
            right = str(change.get(f"counterfactual_{prefix}_sha256") or "")
            if len(left) != 64 or len(right) != 64:
                raise ValueError("counterfactual input fingerprint 不完整")
            expected_changed = prefix in dimensions
            if (left != right) != expected_changed:
                raise ValueError("counterfactual input fingerprint 与 changed_dimensions 冲突")

        donor = row.get("counterfactual_input")
        parent = str(base.get("parent_sample_id") or "")
        if mode == "region_swap" and (
            not isinstance(donor, dict)
            or donor.get("protocol") != "qpsalm_same_parent_region_swap_v1"
            or str(donor.get("parent_sample_id") or "") != parent
        ):
            raise ValueError("same-parent region swap donor audit 非法")
        if mode == "cross_parent_region_swap" and (
            not isinstance(donor, dict)
            or donor.get("protocol") != "qpsalm_cross_parent_region_swap_v1"
            or str(donor.get("target_parent_sample_id") or "") != parent
            or not str(donor.get("donor_parent_sample_id") or "")
            or str(donor.get("donor_parent_sample_id")) == parent
        ):
            raise ValueError("cross-parent region swap donor audit 非法")
        if mode == "cross_parent_modality_swap" and (
            not isinstance(donor, dict)
            or donor.get("protocol") != "qpsalm_cross_parent_modality_donor_v1"
            or str(donor.get("target_parent_sample_id") or "") != parent
            or not str(donor.get("donor_parent_sample_id") or "")
            or str(donor.get("donor_parent_sample_id")) == parent
            or not isinstance(donor.get("applied_swap"), dict)
            or (donor.get("applied_swap") or {}).get("protocol")
            != "qpsalm_cross_parent_modality_swap_v1"
        ):
            raise ValueError("cross-parent modality swap donor audit 非法")

        baseline = str(row.get("baseline_generation") or "")
        changed = str(row.get("counterfactual_generation") or "")
        target = str(base.get("target_text") or "")
        structured = str(base.get("task_family") or "") in {
            "region_description_auto", "region_description_expert",
        }
        if structured:
            baseline_parsed = parse_description_output(baseline).parsed
            changed_parsed = parse_description_output(changed).parsed
            target_parsed = parse_description_output(target).parsed
            expected_sensitivity = structured_disagreement(
                baseline_parsed, changed_parsed
            )
            baseline_score = 1.0 - structured_disagreement(
                baseline_parsed, target_parsed
            )
            changed_score = 1.0 - structured_disagreement(
                changed_parsed, target_parsed
            )
            baseline_claims = unsupported_claim_counts(
                baseline_parsed, target_parsed
            )[1]
            changed_claims = unsupported_claim_counts(
                changed_parsed, target_parsed
            )[1]
        else:
            references = list(base.get("reference_texts") or [])
            expected_sensitivity = 1.0 - caption_token_f1(changed, [baseline])
            baseline_score = caption_token_f1(baseline, references)
            changed_score = caption_token_f1(changed, references)
            baseline_claims = changed_claims = 0
        expected = {
            "sensitivity": expected_sensitivity,
            "baseline_target_score": baseline_score,
            "counterfactual_target_score": changed_score,
            "target_score_delta": changed_score - baseline_score,
            "factual_claim_count_delta": float(changed_claims - baseline_claims),
        }
        for field, value in expected.items():
            try:
                observed = float(row[field])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"counterfactual row 缺少派生字段: {field}") from exc
            if not math.isclose(observed, float(value), rel_tol=0.0, abs_tol=1.0e-12):
                raise ValueError(f"反事实派生指标无法重新计算: {field}")
    return {
        "protocol": "qpsalm_counterfactual_row_binding_v1_generation_replayed",
        "num_rows": len(rows),
        "num_generation_rows": len(generation_rows),
        "rows_sha256": canonical_sha256(rows),
        "passed": True,
    }


def counterfactual_gate(
    report: dict[str, Any],
    rows: list[dict[str, Any]] | None = None,
    scientific_protocol: dict[str, Any] | None = None,
    generation_rows: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    values = report.get("counterfactual_sensitivity") or {}

    def upper(mode: str, metric: str) -> float | None:
        value = (((values.get(mode) or {}).get(metric) or {}).get("high"))
        return float(value) if value is not None else None

    required_modes = (
        "shuffled_mask", "region_swap", "cross_parent_region_swap",
        "cross_parent_modality_swap", "modality_removal",
    )
    frozen_statistics: dict[str, Any] | None = None
    row_binding = (
        _validate_counterfactual_row_bindings(rows, generation_rows)
        if rows is not None and generation_rows is not None else None
    )
    if rows is None or scientific_protocol is None:
        coverage = {
            mode: bool(
                int((values.get(mode) or {}).get("requested") or 0) > 0
                and int((values.get(mode) or {}).get("n") or 0)
                >= int((values.get(mode) or {}).get("requested") or 0)
                and (values.get(mode) or {}).get("coverage_complete") is True
            )
            for mode in required_modes
        }
    else:
        bootstrap = scientific_protocol["bootstrap"]
        minimums = scientific_protocol["counterfactual_minimum_effective_parents"]
        frozen_statistics = {}
        coverage = {}
        for index, mode in enumerate(required_modes):
            score_values = _parent_counterfactual_values(
                rows, mode, "target_score_delta"
            )
            claim_values = _parent_counterfactual_values(
                rows, mode, "factual_claim_count_delta"
            )
            row_count = sum(str(row.get("mode") or "") == mode for row in rows)
            report_mode = values.get(mode) or {}
            coverage[mode] = bool(
                len(score_values) >= int(minimums[mode])
                and len(score_values) == len(claim_values)
                and int(report_mode.get("n", -1)) == row_count
                and int(report_mode.get("num_effective_parents", -1)) == len(score_values)
                and report_mode.get("aggregation_unit") == "parent"
            )
            frozen_statistics[mode] = {
                "minimum_effective_parents": int(minimums[mode]),
                "num_effective_parents": len(score_values),
                "num_effective_rows": row_count,
                "paired_target_score_delta_ci": bootstrap_mean_ci(
                    score_values,
                    seed=int(bootstrap["seed"]) + 104729 * (index + 1),
                    samples=int(bootstrap["samples"]),
                    confidence=float(bootstrap["confidence"]),
                ),
                "paired_factual_claim_count_delta_ci": bootstrap_mean_ci(
                    claim_values,
                    seed=int(bootstrap["seed"]) + 130363 * (index + 1),
                    samples=int(bootstrap["samples"]),
                    confidence=float(bootstrap["confidence"]),
                ),
            }

    def frozen_upper(mode: str, metric: str) -> float | None:
        if frozen_statistics is None:
            return upper(mode, metric)
        value = ((frozen_statistics.get(mode) or {}).get(metric) or {}).get("high")
        return float(value) if value is not None else None

    checks = {
        "counterfactual_coverage_complete": all(coverage.values()),
        "shuffled_mask_degrades_target_score": (
            frozen_upper("shuffled_mask", "paired_target_score_delta_ci") is not None
            and frozen_upper("shuffled_mask", "paired_target_score_delta_ci") < 0
        ),
        "region_swap_degrades_target_score": (
            frozen_upper("region_swap", "paired_target_score_delta_ci") is not None
            and frozen_upper("region_swap", "paired_target_score_delta_ci") < 0
        ),
        "cross_parent_region_swap_degrades_target_score": (
            frozen_upper(
                "cross_parent_region_swap", "paired_target_score_delta_ci"
            ) is not None
            and frozen_upper(
                "cross_parent_region_swap", "paired_target_score_delta_ci"
            ) < 0
        ),
        "cross_parent_swap_degrades_target_score": (
            frozen_upper("cross_parent_modality_swap", "paired_target_score_delta_ci") is not None
            and frozen_upper("cross_parent_modality_swap", "paired_target_score_delta_ci") < 0
        ),
        "modality_removal_reduces_factual_claims": (
            frozen_upper("modality_removal", "paired_factual_claim_count_delta_ci") is not None
            and frozen_upper("modality_removal", "paired_factual_claim_count_delta_ci") < 0
        ),
    }
    return {
        "checks": checks,
        "coverage_by_mode": coverage,
        "passed": all(checks.values()),
        "counterfactual_sensitivity": values,
        "frozen_parent_statistics": frozen_statistics,
        "row_binding": row_binding,
    }
