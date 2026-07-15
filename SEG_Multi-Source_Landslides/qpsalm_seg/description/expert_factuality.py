#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Expert Region Factuality Score aggregation for frozen description outputs."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from qpsalm_seg.paths import resolve_project_path

from .metrics import bootstrap_mean_ci
from .output_protocol import parse_description_output


ALLOWED_SUPPORT = {"supported": 1.0, "partially_supported": 0.5, "unsupported": 0.0}
EXPERT_FAMILIES = (
    "target_status", "region_geometry", "surface", "terrain",
    "sar", "deformation", "surrounding_context", "summary",
)
CLAIM_FIELDS = (
    "target_status",
    "region.location", "region.size_class", "region.shape", "region.elongation",
    "region.compactness", "region.fragmentation",
    "evidence.surface_observation", "evidence.terrain_support", "evidence.sar_support",
    "evidence.deformation_support", "evidence.surrounding_context",
    "evidence.evidence_sufficiency", "summary",
)
NON_CLAIM_VALUES = {
    "", "unknown", "unavailable", "insufficient", "insufficient_evidence",
    "no reliable description is available.",
}


def _nested(value: dict[str, Any], path: str) -> Any:
    current: Any = value
    for part in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _expected_claims(generation: dict[str, Any]) -> list[dict[str, Any]]:
    raw = str(generation.get("raw_generation") or "")
    parsed = parse_description_output(raw).parsed
    if not isinstance(parsed, dict):
        return [{
            "claim_id": "unparsed_generation",
            "source_field": "raw_generation",
            "text": raw,
            "support": None,
        }]
    claims = []
    for field in CLAIM_FIELDS:
        value = _nested(parsed, field)
        text = str(value or "").strip()
        if text.casefold() in NON_CLAIM_VALUES:
            continue
        claims.append({
            "claim_id": field,
            "source_field": field,
            "text": text,
            "support": None,
        })
    return claims


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_expert_review_template(eval_dir: str | Path) -> list[dict[str, Any]]:
    root = resolve_project_path(eval_dir) or Path(eval_dir)
    report_path = root / "eval_report.json"
    generation_path = root / "raw_generations.jsonl"
    if not report_path.is_file():
        raise FileNotFoundError(f"ERFS 缺少 eval_report.json: {root}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not bool((report.get("generation_coverage") or {}).get("complete")):
        raise ValueError(
            "ERFS 只能审核完整 frozen generation；请用 --max-generate-samples 0 重跑评估"
        )
    rows = [
        row for row in _jsonl(generation_path)
        if (row.get("raw_metrics") or {}).get("raw_schema_valid") is not None
    ]
    if not rows:
        raise ValueError("ERFS template 只支持 structured region-description generations")
    templates = [{
        "sample_id": str(row["sample_id"]),
        "parent_sample_id": str(row["parent_sample_id"]),
        "reviewer_id": "",
        "review_protocol": "blind_region_factuality_v2",
        "instruction": str(row.get("instruction") or ""),
        "model_generation": str(row.get("raw_generation") or ""),
        "evaluation_mode": str(row.get("evaluation_mode") or report.get("evaluation_mode") or "unknown"),
        "region_source": str(row.get("region_source") or "unknown"),
        "region_id": str(row.get("region_id") or "unknown"),
        "expert_review_panel_path": row.get("expert_review_panel_path"),
        "region_mask_path": row.get("region_mask_path"),
        "visual_preview_path": row.get("visual_preview_path"),
        "multimodal_preview_path": row.get("multimodal_preview_path"),
        "reference_target_hidden": True,
        "frozen_generation_sha256": _sha256(generation_path),
        "family_scores": {family: None for family in EXPERT_FAMILIES},
        "claims": _expected_claims(row),
        "notes": "",
    } for row in rows]
    visual_fields = (
        "expert_review_panel_path", "multimodal_preview_path", "visual_preview_path"
    )
    missing_visuals = []
    for value in templates:
        existing = False
        for field in visual_fields:
            path_ref = value.get(field)
            if not path_ref:
                continue
            path = resolve_project_path(str(path_ref)) or Path(str(path_ref))
            existing |= path.is_file()
        if not existing:
            missing_visuals.append(value["sample_id"])
    if missing_visuals:
        raise ValueError(
            "ERFS 审核模板缺少可视证据路径: "
            f"count={len(missing_visuals)} examples={missing_visuals[:8]}"
        )
    return templates


def aggregate_expert_factuality(
    eval_dir: str | Path,
    review_files: Iterable[str | Path],
    *,
    seed: int,
    minimum_reviewers: int = 2,
) -> dict[str, Any]:
    root = resolve_project_path(eval_dir) or Path(eval_dir)
    generation_path = root / "raw_generations.jsonl"
    generation_hash = _sha256(generation_path)
    generation = {
        str(row["sample_id"]): row
        for row in _jsonl(generation_path)
    }
    reviews: dict[str, list[dict[str, Any]]] = defaultdict(list)
    reviewer_ids: set[str] = set()
    seen: set[tuple[str, str]] = set()
    review_hashes: dict[str, str] = {}
    for review_file in review_files:
        review_path = resolve_project_path(review_file) or Path(review_file)
        review_hashes[str(review_path.resolve(strict=False))] = _sha256(review_path)
        for row in _jsonl(review_file):
            sample = str(row.get("sample_id") or "")
            reviewer = str(row.get("reviewer_id") or "").strip()
            if sample not in generation:
                raise ValueError(f"expert review 引用了 eval 中不存在的 sample={sample}")
            if not reviewer:
                raise ValueError(f"expert review 缺少 reviewer_id: sample={sample}")
            if row.get("review_protocol") != "blind_region_factuality_v2":
                raise ValueError(f"expert review protocol 不一致: sample={sample}")
            if str(row.get("frozen_generation_sha256") or "") != generation_hash:
                raise ValueError(f"expert review 绑定了不同 frozen generation: sample={sample}")
            if str(row.get("model_generation") or "") != str(
                generation[sample].get("raw_generation") or ""
            ):
                raise ValueError(f"expert review 改写了冻结模型输出: sample={sample}")
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
            expected_claims = _expected_claims(generation[sample])
            expected_ids = {str(value["claim_id"]) for value in expected_claims}
            expected_by_id = {
                str(value["claim_id"]): value for value in expected_claims
            }
            observed_ids = [str((value or {}).get("claim_id") or "") for value in claims]
            if len(observed_ids) != len(set(observed_ids)) or set(observed_ids) != expected_ids:
                raise ValueError(
                    f"claims 必须完整覆盖冻结输出: sample={sample} "
                    f"missing={sorted(expected_ids - set(observed_ids))} "
                    f"unexpected={sorted(set(observed_ids) - expected_ids)}"
                )
            unsupported = factual = 0
            for claim in claims:
                claim_id = str((claim or {}).get("claim_id") or "")
                expected_claim = expected_by_id[claim_id]
                if (
                    str((claim or {}).get("source_field") or "")
                    != str(expected_claim["source_field"])
                    or str((claim or {}).get("text") or "")
                    != str(expected_claim["text"])
                ):
                    raise ValueError(
                        f"expert review 改写了冻结 claim: sample={sample} claim={claim_id}"
                    )
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
        "raw_generations_sha256": generation_hash,
        "eval_report_sha256": _sha256(root / "eval_report.json"),
        "review_file_sha256": dict(sorted(review_hashes.items())),
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
