"""分任务 evaluator、证据约束与反事实检查。"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from oa_groundrag.phase3.common import first_symlink_component, read_jsonl

from .artifacts import AtomicArtifactDirectory
from .contracts import (
    PREDICTION_SCHEMA_VERSION,
    RUN_MANIFEST_SCHEMA_VERSION,
    EvidenceSufficiency,
    MaskMode,
    TargetStatus,
)
from .errors import EvaluationError, ReasonCode
from .outputs import parse_model_output, serialize_model_output


_PREDICTION_FIELDS = {
    "schema_version",
    "record_id",
    "parent_id",
    "logical_role",
    "task_family",
    "mask_mode",
    "generated_text",
    "model_output",
    "reference_responses",
    "evidence_ids",
    "provenance",
    "counterfactual",
}


def _normalize(value: str) -> str:
    return " ".join(re.findall(r"[\w]+", value.lower(), flags=re.UNICODE))


def _token_f1(prediction: str, reference: str) -> float:
    predicted = _normalize(prediction).split()
    expected = _normalize(reference).split()
    if not predicted or not expected:
        return float(predicted == expected)
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


def _validate_row(
    row: Mapping[str, Any],
    *,
    expected_mask_mode: MaskMode,
) -> Any:
    if set(row) != _PREDICTION_FIELDS:
        raise EvaluationError(
            ReasonCode.PREDICTION_INVALID,
            "prediction 字段不匹配",
            details={
                "unknown": sorted(set(row) - _PREDICTION_FIELDS),
                "missing": sorted(_PREDICTION_FIELDS - set(row)),
            },
        )
    if row.get("schema_version") != PREDICTION_SCHEMA_VERSION:
        raise EvaluationError(
            ReasonCode.PREDICTION_INVALID,
            "prediction schema 不匹配",
        )
    if row.get("mask_mode") != expected_mask_mode.value:
        raise EvaluationError(
            ReasonCode.MASK_MODE_MIXED,
            "prediction mask_mode 混合或与 evaluator 不一致",
        )
    for name in (
        "record_id",
        "parent_id",
        "logical_role",
        "task_family",
        "generated_text",
    ):
        if not isinstance(row.get(name), str) or not row[name].strip():
            raise EvaluationError(
                ReasonCode.PREDICTION_INVALID,
                f"prediction.{name} 必须非空",
            )
    references = row.get("reference_responses")
    evidence_ids = row.get("evidence_ids")
    if (
        not isinstance(references, list)
        or not all(isinstance(value, str) and value.strip() for value in references)
        or not isinstance(evidence_ids, list)
        or not all(isinstance(value, str) and value for value in evidence_ids)
        or len(evidence_ids) != len(set(evidence_ids))
        or not isinstance(row.get("provenance"), dict)
    ):
        raise EvaluationError(
            ReasonCode.PREDICTION_INVALID,
            "prediction references/evidence/provenance 合同非法",
        )
    if expected_mask_mode is MaskMode.EXTERNAL_GENERIC:
        if row["model_output"] is not None or evidence_ids:
            raise EvaluationError(
                ReasonCode.EXTERNAL_MASK_FORBIDDEN,
                "External prediction 不允许 mask-grounded model_output/evidence",
            )
        return None
    if not isinstance(row["model_output"], dict):
        raise EvaluationError(
            ReasonCode.PREDICTION_INVALID,
            "mask-grounded prediction 缺少 structured model_output",
        )
    output = parse_model_output(
        row["model_output"],
        valid_evidence_ids=evidence_ids,
    )
    if row["generated_text"].strip() != serialize_model_output(output):
        raise EvaluationError(
            ReasonCode.PREDICTION_INVALID,
            "generated_text 与 structured model_output 不一致",
        )
    return output


def _external_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_task: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[str(row["task_family"])].append(row)
    metrics: dict[str, Any] = {}
    for task, values in sorted(by_task.items()):
        exact: list[float] = []
        f1: list[float] = []
        for row in values:
            references = row["reference_responses"]
            prediction = str(row["generated_text"])
            exact.append(
                float(
                    _normalize(prediction)
                    in {_normalize(reference) for reference in references}
                )
            )
            f1.append(
                max(
                    _token_f1(prediction, reference)
                    for reference in references
                )
            )
        metrics[task] = {
            "count": len(values),
            "normalized_exact_match": sum(exact) / len(exact),
            "best_reference_token_f1": sum(f1) / len(f1),
        }
    return metrics


def _no_target_hallucination(description: str) -> bool:
    normalized = description.lower()
    patterns = (
        r"\bbbox\b",
        r"\bbounding box\b",
        r"\bcentroid\b",
        r"\barea ratio\b",
        r"\bcontext crop\b",
        r"\bmask shape\b",
        r"\bthe target is located\b",
    )
    return any(re.search(pattern, normalized) for pattern in patterns)


def _counterfactual_metrics(
    rows: Sequence[Mapping[str, Any]],
    outputs: Mapping[str, Any],
) -> dict[str, Any]:
    groups: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    group_parents: dict[str, str] = {}
    for row in rows:
        counterfactual = row.get("counterfactual")
        if counterfactual is None:
            continue
        variants = {
            "baseline",
            "mask_swap",
            "wrong_region",
            "empty_mask",
            "modality_removal",
        }
        if (
            not isinstance(counterfactual, dict)
            or not isinstance(counterfactual.get("group_id"), str)
            or not counterfactual["group_id"].strip()
            or counterfactual.get("variant") not in variants
        ):
            raise EvaluationError(
                ReasonCode.COUNTERFACTUAL_INVALID,
                "counterfactual group_id/variant 非法",
            )
        group_id = counterfactual["group_id"]
        variant = counterfactual["variant"]
        allowed_fields = {"group_id", "variant"}
        if variant == "modality_removal":
            allowed_fields.add("removed_evidence_ids")
        if set(counterfactual) - allowed_fields:
            raise EvaluationError(
                ReasonCode.COUNTERFACTUAL_INVALID,
                "counterfactual 含 variant 不允许的字段",
            )
        removed = counterfactual.get("removed_evidence_ids", [])
        if (
            not isinstance(removed, list)
            or not all(isinstance(value, str) and value for value in removed)
            or len(removed) != len(set(removed))
        ):
            raise EvaluationError(
                ReasonCode.COUNTERFACTUAL_INVALID,
                "removed_evidence_ids 必须是唯一非空字符串列表",
            )
        parent_id = str(row["parent_id"])
        previous_parent = group_parents.setdefault(group_id, parent_id)
        if previous_parent != parent_id:
            raise EvaluationError(
                ReasonCode.COUNTERFACTUAL_INVALID,
                "同一 counterfactual group 不得跨 parent",
                details={
                    "group_id": group_id,
                    "parents": sorted({previous_parent, parent_id}),
                },
            )
        group = groups[group_id]
        if variant in group:
            raise EvaluationError(
                ReasonCode.COUNTERFACTUAL_INVALID,
                "同一 counterfactual group/variant 不得重复",
                details={"group_id": group_id, "variant": variant},
            )
        group[variant] = row
    missing_baseline = sorted(
        group_id for group_id, variants in groups.items()
        if "baseline" not in variants
    )
    if missing_baseline:
        raise EvaluationError(
            ReasonCode.COUNTERFACTUAL_INVALID,
            "counterfactual group 缺少 baseline",
            details={"group_ids": missing_baseline},
        )
    checks = Counter()
    failures = Counter()
    for variants in groups.values():
        baseline = variants["baseline"]
        base_output = outputs[str(baseline["record_id"])]
        for variant in ("mask_swap", "wrong_region"):
            row = variants.get(variant)
            if row is None:
                continue
            checks[variant] += 1
            output = outputs[str(row["record_id"])]
            if _normalize(output.description) == _normalize(
                base_output.description
            ):
                failures[variant] += 1
        empty = variants.get("empty_mask")
        if empty is not None:
            checks["empty_mask"] += 1
            output = outputs[str(empty["record_id"])]
            if (
                output.target_status is not TargetStatus.NO_TARGET
                or _no_target_hallucination(output.description)
            ):
                failures["empty_mask"] += 1
        removed = variants.get("modality_removal")
        if removed is not None:
            checks["modality_removal"] += 1
            output = outputs[str(removed["record_id"])]
            ordering = {
                EvidenceSufficiency.INSUFFICIENT: 0,
                EvidenceSufficiency.LIMITED: 1,
                EvidenceSufficiency.SUFFICIENT: 2,
            }
            if (
                ordering[output.evidence_sufficiency]
                > ordering[base_output.evidence_sufficiency]
            ):
                failures["modality_removal"] += 1
            removed_ids = set(
                removed.get("counterfactual", {}).get(
                    "removed_evidence_ids", []
                )
            )
            cited = {
                evidence_id
                for claim in output.claims
                for evidence_id in claim.evidence_ids
            }
            if cited & removed_ids:
                failures["modality_removal"] += 1
    return {
        "invalid_counterfactual_rows": 0,
        "group_count": len(groups),
        "checks": dict(sorted(checks.items())),
        "failures": dict(sorted(failures.items())),
    }


def evaluate_predictions(
    prediction_path: Path,
    *,
    output_root: Path,
    expected_mask_mode: MaskMode,
    formal: bool = False,
) -> Path:
    if formal:
        raise EvaluationError(
            ReasonCode.FORMAL_EVALUATION_FORBIDDEN,
            "phase4 本轮不执行正式评价",
        )
    prediction_path = Path(prediction_path)
    linked = first_symlink_component(prediction_path)
    if linked is not None:
        raise EvaluationError(
            ReasonCode.OUTPUT_LINK,
            f"prediction path 含链接组件：{linked}",
        )
    if (
        not prediction_path.is_file()
        or prediction_path.is_symlink()
        or prediction_path.stat().st_nlink != 1
    ):
        raise EvaluationError(
            ReasonCode.OUTPUT_LINK,
            "prediction path 必须是普通单链接文件",
        )
    rows = read_jsonl(prediction_path)
    if not rows:
        raise EvaluationError(
            ReasonCode.PREDICTION_INVALID,
            "prediction JSONL 不能为空",
        )
    if len({str(row.get("record_id")) for row in rows}) != len(rows):
        raise EvaluationError(
            ReasonCode.PREDICTION_INVALID,
            "prediction record_id 不能重复",
        )
    outputs: dict[str, Any] = {}
    evidence_violations = 0
    no_target_hallucinations = 0
    target_status_mismatches = 0
    mask_task_metrics: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        output = _validate_row(row, expected_mask_mode=expected_mask_mode)
        if output is None:
            continue
        task_metrics = mask_task_metrics[str(row["task_family"])]
        task_metrics["count"] += 1
        outputs[str(row["record_id"])] = output
        if any(not claim.evidence_ids for claim in output.claims):
            evidence_violations += 1
            task_metrics["evidence_constraint_violations"] += 1
        expected_status = row["provenance"].get("expected_target_status")
        if (
            expected_status is not None
            and output.target_status.value != expected_status
        ):
            target_status_mismatches += 1
            task_metrics["target_status_mismatches"] += 1
        if (
            output.target_status is TargetStatus.NO_TARGET
            and _no_target_hallucination(output.description)
        ):
            no_target_hallucinations += 1
            task_metrics["no_target_hallucinations"] += 1
    if expected_mask_mode is MaskMode.EXTERNAL_GENERIC:
        per_task = _external_metrics(rows)
        counterfactual = {
            "not_applicable": True,
            "reason": "external_generic_has_no_masks",
        }
    else:
        per_task = {
            task: {
                "count": values["count"],
                "evidence_constraint_violations": values[
                    "evidence_constraint_violations"
                ],
                "target_status_mismatches": values[
                    "target_status_mismatches"
                ],
                "no_target_hallucinations": values[
                    "no_target_hallucinations"
                ],
            }
            for task, values in sorted(mask_task_metrics.items())
        }
        counterfactual = _counterfactual_metrics(rows, outputs)
    report = {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "run_kind": "evaluation",
        "mask_mode": expected_mask_mode.value,
        "prediction_count": len(rows),
        "parent_count": len({str(row["parent_id"]) for row in rows}),
        "per_task": per_task,
        "evidence_constraint_violations": evidence_violations,
        "target_status_mismatches": target_status_mismatches,
        "no_target_hallucinations": no_target_hallucinations,
        "counterfactual": counterfactual,
        "formal_acceptance": False,
    }
    with AtomicArtifactDirectory(Path(output_root)) as writer:
        writer.write_json("metrics.json", report)
        writer.write_json(
            "manifest.json",
            {
                "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
                "run_kind": "evaluation",
                "mask_mode": expected_mask_mode.value,
                "prediction_path": str(Path(prediction_path).resolve()),
                "formal_acceptance": False,
            },
        )
        return writer.publish()
