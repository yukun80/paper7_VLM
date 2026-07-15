#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Expert Region Factuality Score aggregation for frozen description outputs."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Iterable

from qpsalm_seg.paths import resolve_project_path

from .metrics import bootstrap_mean_ci


ALLOWED_SUPPORT = {"supported": 1.0, "partially_supported": 0.5, "unsupported": 0.0}
EXPERT_FAMILIES = (
    "target_status", "region_geometry", "surface", "terrain",
    "sar", "deformation", "surrounding_context", "summary",
)


def _cohen_kappa(left: list[str], right: list[str]) -> float | None:
    if len(left) != len(right) or not left:
        return None
    labels = sorted(set(left) | set(right))
    observed = sum(a == b for a, b in zip(left, right)) / len(left)
    expected = sum(
        (left.count(label) / len(left)) * (right.count(label) / len(right))
        for label in labels
    )
    if expected >= 1.0:
        return 1.0 if observed >= 1.0 else None
    return (observed - expected) / (1.0 - expected)


def _krippendorff_alpha_nominal(units: list[list[str]]) -> float | None:
    comparable = [values for values in units if len(values) >= 2]
    if not comparable:
        return None
    disagreements = sum(
        sum(a != b for index, a in enumerate(values) for b in values[index + 1:])
        for values in comparable
    )
    pairs = sum(len(values) * (len(values) - 1) / 2 for values in comparable)
    observed = disagreements / max(pairs, 1.0)
    all_values = [value for values in comparable for value in values]
    total = len(all_values)
    if total < 2:
        return None
    counts = {value: all_values.count(value) for value in set(all_values)}
    expected_agreement = sum(count * (count - 1) for count in counts.values()) / (
        total * (total - 1)
    )
    expected_disagreement = 1.0 - expected_agreement
    if expected_disagreement <= 0:
        return 1.0 if observed <= 0 else None
    return 1.0 - observed / expected_disagreement


def _jsonl(path_ref: str | Path) -> list[dict[str, Any]]:
    path = resolve_project_path(path_ref) or Path(path_ref)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_expert_review_template(eval_dir: str | Path) -> list[dict[str, Any]]:
    root = resolve_project_path(eval_dir) or Path(eval_dir)
    rows = [
        row for row in _jsonl(root / "raw_generations.jsonl")
        if (row.get("raw_metrics") or {}).get("raw_schema_valid") is not None
    ]
    if not rows:
        raise ValueError("ERFS template 只支持 structured region-description generations")
    return [{
        "sample_id": str(row["sample_id"]),
        "parent_sample_id": str(row["parent_sample_id"]),
        "reviewer_id": "",
        "family_scores": {family: None for family in EXPERT_FAMILIES},
        "claims": [],
        "notes": "",
    } for row in rows]


def aggregate_expert_factuality(
    eval_dir: str | Path,
    review_files: Iterable[str | Path],
    *,
    seed: int,
    minimum_reviewers: int = 2,
) -> dict[str, Any]:
    root = resolve_project_path(eval_dir) or Path(eval_dir)
    generation = {
        str(row["sample_id"]): row
        for row in _jsonl(root / "raw_generations.jsonl")
    }
    reviews: dict[str, list[dict[str, Any]]] = defaultdict(list)
    reviewer_ids: set[str] = set()
    seen: set[tuple[str, str]] = set()
    for review_file in review_files:
        for row in _jsonl(review_file):
            sample = str(row.get("sample_id") or "")
            reviewer = str(row.get("reviewer_id") or "").strip()
            if sample not in generation:
                raise ValueError(f"expert review 引用了 eval 中不存在的 sample={sample}")
            if not reviewer:
                raise ValueError(f"expert review 缺少 reviewer_id: sample={sample}")
            key = (sample, reviewer)
            if key in seen:
                raise ValueError(f"同一 reviewer 重复审核 sample: {key}")
            seen.add(key)
            reviewer_ids.add(reviewer)
            families = row.get("family_scores")
            if not isinstance(families, dict) or not families:
                raise ValueError(f"family_scores 必须为非空 object: sample={sample}")
            normalized = {}
            for family, value in families.items():
                score = float(value)
                if score not in {0.0, 0.5, 1.0}:
                    raise ValueError(f"ERFS family score 只允许 0/0.5/1: {sample}:{family}={score}")
                normalized[str(family)] = score
            if set(normalized) != set(EXPERT_FAMILIES):
                raise ValueError(
                    f"family_scores 必须完整匹配 ontology families: sample={sample} "
                    f"missing={sorted(set(EXPERT_FAMILIES) - set(normalized))} "
                    f"unexpected={sorted(set(normalized) - set(EXPERT_FAMILIES))}"
                )
            claims = row.get("claims") or []
            if not isinstance(claims, list):
                raise ValueError(f"claims 必须为 list: sample={sample}")
            unsupported = factual = 0
            for claim in claims:
                status = str((claim or {}).get("support") or "")
                if status not in ALLOWED_SUPPORT:
                    raise ValueError(f"claim support 非法: sample={sample} value={status!r}")
                factual += 1
                unsupported += int(status == "unsupported")
            reviews[sample].append({
                "reviewer_id": reviewer,
                "family_scores": normalized,
                "sample_score": sum(normalized.values()) / len(normalized),
                "unsupported_claims": unsupported,
                "factual_claims": factual,
            })
    missing = sorted(set(generation) - set(reviews))
    if missing:
        raise ValueError(f"expert reviews 未覆盖全部 frozen generations: count={len(missing)} examples={missing[:8]}")
    insufficient = sorted(sample for sample, values in reviews.items() if len(values) < minimum_reviewers)
    if insufficient:
        raise ValueError(
            f"expert reviews 少于 {minimum_reviewers} 人: count={len(insufficient)} examples={insufficient[:8]}"
        )
    per_sample = {}
    parent_values: dict[str, list[float]] = defaultdict(list)
    exact_agreements = []
    family_units: dict[str, list[list[str]]] = defaultdict(list)
    unsupported = claims = 0
    for sample, values in sorted(reviews.items()):
        score = sum(value["sample_score"] for value in values) / len(values)
        parent = str(generation[sample]["parent_sample_id"])
        per_sample[sample] = {
            "parent_sample_id": parent,
            "reviewers": [value["reviewer_id"] for value in values],
            "expert_region_factuality_score": score,
        }
        parent_values[parent].append(score)
        first = values[0]["family_scores"]
        exact_agreements.extend(
            float(first.get(family) == other["family_scores"].get(family))
            for other in values[1:]
            for family in sorted(set(first) & set(other["family_scores"]))
        )
        families = sorted({
            family for value in values for family in value["family_scores"]
        })
        for family in families:
            family_units[family].append([
                str(value["family_scores"][family])
                for value in values
                if family in value["family_scores"]
            ])
        unsupported += sum(value["unsupported_claims"] for value in values)
        claims += sum(value["factual_claims"] for value in values)
    per_parent = {
        parent: sum(values) / len(values)
        for parent, values in sorted(parent_values.items())
    }
    reviewer_order = sorted(reviewer_ids)
    family_agreement = {}
    for family, units in sorted(family_units.items()):
        left: list[str] = []
        right: list[str] = []
        if len(reviewer_order) == 2:
            first_reviewer, second_reviewer = reviewer_order
            for sample_values in reviews.values():
                by_reviewer = {
                    value["reviewer_id"]: value["family_scores"].get(family)
                    for value in sample_values
                }
                if by_reviewer.get(first_reviewer) is not None and by_reviewer.get(second_reviewer) is not None:
                    left.append(str(by_reviewer[first_reviewer]))
                    right.append(str(by_reviewer[second_reviewer]))
        comparable = [values for values in units if len(values) >= 2]
        family_agreement[family] = {
            "num_comparable_items": len(comparable),
            "exact_agreement": (
                sum(len(set(values)) == 1 for values in comparable) / len(comparable)
                if comparable else None
            ),
            "cohen_kappa": _cohen_kappa(left, right) if len(reviewer_order) == 2 else None,
            "krippendorff_alpha_nominal": _krippendorff_alpha_nominal(units),
        }
    return {
        "protocol": "qpsalm_expert_region_factuality_v1",
        "eval_dir": str(root),
        "num_samples": len(per_sample),
        "num_parents": len(per_parent),
        "reviewer_ids": sorted(reviewer_ids),
        "minimum_reviewers": int(minimum_reviewers),
        "expert_region_factuality_score": sum(per_parent.values()) / max(len(per_parent), 1),
        "parent_bootstrap_ci": bootstrap_mean_ci(per_parent.values(), seed=seed, samples=10000),
        "field_exact_agreement": sum(exact_agreements) / max(len(exact_agreements), 1),
        "field_agreement": family_agreement,
        "expert_unsupported_claim_rate": unsupported / max(claims, 1),
        "expert_unsupported_claims": unsupported,
        "expert_factual_claims": claims,
        "per_parent_scores": per_parent,
        "per_sample_scores": per_sample,
    }
