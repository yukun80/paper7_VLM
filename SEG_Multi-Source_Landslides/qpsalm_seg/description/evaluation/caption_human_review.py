"""Blind two-rater review protocol for frozen RSIEval captions."""

from __future__ import annotations

from collections import defaultdict
import hashlib
from pathlib import Path
from typing import Any, Iterable

from qpsalm_seg.paths import resolve_project_path

from .caption_metrics import caption_metric_population
from ..protocols.io import (
    atomic_write_jsonl,
    sha256_file as _sha256,
    strict_json_loads,
)
from .metrics import bootstrap_mean_ci


CAPTION_HUMAN_REVIEW_PROTOCOL = (
    "qpsalm_rsieval_caption_human_review_v1_blind_two_rater"
)
CAPTION_HUMAN_REPORT_PROTOCOL = (
    "qpsalm_rsieval_caption_human_review_report_v1_parent_macro"
)
CAPTION_REVIEW_DIMENSIONS = ("factuality", "detail", "readability")
CAPTION_REVIEW_SCALE = {
    "minimum": 1,
    "maximum": 5,
    "anchors": {
        "1": "严重错误或不可用",
        "3": "基本可接受但有明显不足",
        "5": "准确、充分且清晰",
    },
}
_FROZEN_REVIEW_FIELDS = (
    "review_item_id",
    "review_protocol",
    "sample_id",
    "parent_sample_id",
    "source_dataset",
    "visual_image_path",
    "visual_image_sha256",
    "model_generation",
    "model_generation_sha256",
    "reference_target_hidden",
    "frozen_eval_report_sha256",
    "frozen_generations_sha256",
    "metric_population_sha256",
    "score_scale",
)


def _jsonl(path_ref: str | Path) -> list[dict[str, Any]]:
    path = resolve_project_path(path_ref) or Path(path_ref)
    if not path.is_file():
        raise FileNotFoundError(f"caption human review 文件不存在: {path}")
    return [
        strict_json_loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _review_item_id(sample_id: str, audit: dict[str, Any], image_hash: str) -> str:
    encoded = "\n".join((
        CAPTION_HUMAN_REVIEW_PROTOCOL,
        sample_id,
        str(audit["eval_report_sha256"]),
        str(audit["raw_generations_sha256"]),
        str(audit["metric_population_sha256"]),
        image_hash,
    )).encode("utf-8")
    return "rsieval_caption_review_" + hashlib.sha256(encoded).hexdigest()[:24]


def build_caption_human_review_template(
    eval_dir: str | Path,
    *,
    expected_samples: int = 100,
) -> list[dict[str, Any]]:
    """Create a blind template bound to images and frozen raw generations."""
    rows, audit = caption_metric_population(
        eval_dir,
        source_dataset="RSIEval",
        expected_samples=expected_samples,
    )
    templates: list[dict[str, Any]] = []
    for row in rows:
        path_ref = str(row.get("visual_image_path") or "")
        image_path = resolve_project_path(path_ref) or Path(path_ref)
        if not path_ref or not image_path.is_file():
            raise FileNotFoundError(
                "RSIEval caption review 缺少 materialized visual image: "
                f"sample={row.get('sample_id')} path={path_ref!r}"
            )
        image_hash = _sha256(image_path)
        generation = str(row["raw_generation"])
        sample_id = str(row["sample_id"])
        templates.append({
            "review_item_id": _review_item_id(sample_id, audit, image_hash),
            "review_protocol": CAPTION_HUMAN_REVIEW_PROTOCOL,
            "sample_id": sample_id,
            "parent_sample_id": str(row["parent_sample_id"]),
            "source_dataset": "RSIEval",
            "visual_image_path": path_ref,
            "visual_image_sha256": image_hash,
            "model_generation": generation,
            "model_generation_sha256": hashlib.sha256(
                generation.encode("utf-8")
            ).hexdigest(),
            # 参考 caption 不进入审核文件，避免评分被 reference wording 锚定。
            "reference_target_hidden": True,
            "frozen_eval_report_sha256": audit["eval_report_sha256"],
            "frozen_generations_sha256": audit["raw_generations_sha256"],
            "metric_population_sha256": audit["metric_population_sha256"],
            "score_scale": CAPTION_REVIEW_SCALE,
            "reviewer_id": "",
            "scores": {dimension: None for dimension in CAPTION_REVIEW_DIMENSIONS},
            "notes": "",
        })
    return templates


def write_caption_review_jsonl(
    path_ref: str | Path, rows: Iterable[dict[str, Any]]
) -> None:
    atomic_write_jsonl(path_ref, rows)


def _score_value(value: Any, *, sample: str, dimension: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"caption review score 不能为 bool: {sample}:{dimension}")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"caption review score 必须为 1..5: {sample}:{dimension}={value!r}"
        ) from exc
    if not numeric.is_integer() or int(numeric) not in range(1, 6):
        raise ValueError(
            f"caption review score 必须为 1..5: {sample}:{dimension}={value!r}"
        )
    return int(numeric)


def _quadratic_weighted_kappa(left: list[int], right: list[int]) -> float | None:
    if len(left) != len(right) or not left:
        return None
    labels = tuple(range(1, 6))
    observed = sum(((a - b) / 4.0) ** 2 for a, b in zip(left, right)) / len(left)
    left_counts = {label: left.count(label) / len(left) for label in labels}
    right_counts = {label: right.count(label) / len(right) for label in labels}
    expected = sum(
        left_counts[a] * right_counts[b] * ((a - b) / 4.0) ** 2
        for a in labels for b in labels
    )
    if expected <= 0.0:
        return 1.0 if observed <= 0.0 else None
    return 1.0 - observed / expected


def aggregate_caption_human_reviews(
    eval_dir: str | Path,
    review_files: Iterable[str | Path],
    *,
    seed: int,
    minimum_reviewers: int = 2,
    expected_samples: int = 100,
) -> dict[str, Any]:
    """Validate complete independent reviews and aggregate by image parent."""
    if int(minimum_reviewers) < 2:
        raise ValueError("caption human review protocol 要求至少两名独立 reviewer")
    expected_rows = build_caption_human_review_template(
        eval_dir, expected_samples=expected_samples
    )
    expected = {str(row["sample_id"]): row for row in expected_rows}
    files = list(review_files)
    if len(files) < int(minimum_reviewers):
        raise ValueError(
            f"正式 caption human review 至少需要 {int(minimum_reviewers)} 份独立文件"
        )
    reviews: dict[str, dict[str, dict[str, int]]] = defaultdict(dict)
    reviewer_ids: set[str] = set()
    review_hashes: dict[str, str] = {}
    for file_ref in files:
        path = resolve_project_path(file_ref) or Path(file_ref)
        rows = _jsonl(path)
        if len(rows) != len(expected):
            raise ValueError(
                f"caption review 文件未完整覆盖 population: {path} "
                f"rows={len(rows)} expected={len(expected)}"
            )
        file_reviewers = {
            str(row.get("reviewer_id") or "").strip() for row in rows
        }
        if len(file_reviewers) != 1 or "" in file_reviewers:
            raise ValueError(f"每份 caption review 文件必须且只能有一个 reviewer_id: {path}")
        reviewer = next(iter(file_reviewers))
        if reviewer in reviewer_ids:
            raise ValueError(f"caption reviewer_id 跨文件重复: {reviewer}")
        reviewer_ids.add(reviewer)
        review_hashes[str(path.resolve(strict=False))] = _sha256(path)
        observed_samples: set[str] = set()
        for row in rows:
            sample = str(row.get("sample_id") or "")
            if sample not in expected or sample in observed_samples:
                raise ValueError(f"caption review sample 非法或重复: {sample!r}")
            observed_samples.add(sample)
            template = expected[sample]
            if set(row) != set(template):
                raise ValueError(f"caption review schema fields 被修改: sample={sample}")
            for field in _FROZEN_REVIEW_FIELDS:
                if row.get(field) != template.get(field):
                    raise ValueError(
                        f"caption review 修改了冻结字段: sample={sample} field={field}"
                    )
            scores = row.get("scores")
            if not isinstance(scores, dict) or set(scores) != set(CAPTION_REVIEW_DIMENSIONS):
                raise ValueError(f"caption review scores 字段不完整: sample={sample}")
            reviews[sample][reviewer] = {
                dimension: _score_value(
                    scores[dimension], sample=sample, dimension=dimension
                )
                for dimension in CAPTION_REVIEW_DIMENSIONS
            }
    insufficient = [
        sample for sample, values in reviews.items()
        if len(values) < int(minimum_reviewers)
    ]
    if len(reviews) != len(expected) or insufficient:
        raise ValueError(
            "caption reviews 未达到完整双人覆盖: "
            f"covered={len(reviews)}/{len(expected)} insufficient={insufficient[:8]}"
        )

    ordered_reviewers = sorted(reviewer_ids)
    per_sample: dict[str, Any] = {}
    parent_dimension_values: dict[str, dict[str, list[float]]] = {
        dimension: defaultdict(list) for dimension in CAPTION_REVIEW_DIMENSIONS
    }
    parent_overall_values: dict[str, list[float]] = defaultdict(list)
    dimension_agreement: dict[str, Any] = {}
    for dimension in CAPTION_REVIEW_DIMENSIONS:
        pair_kappas = []
        exact = within_one = comparisons = high_disagreement = 0
        for left_index, left_reviewer in enumerate(ordered_reviewers):
            for right_reviewer in ordered_reviewers[left_index + 1:]:
                left = [reviews[sample][left_reviewer][dimension] for sample in sorted(expected)]
                right = [reviews[sample][right_reviewer][dimension] for sample in sorted(expected)]
                kappa = _quadratic_weighted_kappa(left, right)
                if kappa is not None:
                    pair_kappas.append(kappa)
                for a, b in zip(left, right):
                    comparisons += 1
                    exact += int(a == b)
                    within_one += int(abs(a - b) <= 1)
                    high_disagreement += int(abs(a - b) >= 3)
        dimension_agreement[dimension] = {
            "pairwise_comparisons": comparisons,
            "exact_agreement": exact / comparisons if comparisons else None,
            "within_one_agreement": within_one / comparisons if comparisons else None,
            "mean_pairwise_quadratic_weighted_kappa": (
                sum(pair_kappas) / len(pair_kappas) if pair_kappas else None
            ),
            "high_disagreement_count": high_disagreement,
        }

    for sample in sorted(expected):
        parent = str(expected[sample]["parent_sample_id"])
        by_dimension = {
            dimension: sum(
                reviews[sample][reviewer][dimension]
                for reviewer in ordered_reviewers
            ) / len(ordered_reviewers)
            for dimension in CAPTION_REVIEW_DIMENSIONS
        }
        overall = sum(by_dimension.values()) / len(by_dimension)
        per_sample[sample] = {
            "parent_sample_id": parent,
            "dimension_scores": by_dimension,
            "overall_score": overall,
        }
        for dimension, value in by_dimension.items():
            parent_dimension_values[dimension][parent].append(value)
        parent_overall_values[parent].append(overall)

    per_parent = {
        parent: {
            "dimension_scores": {
                dimension: sum(parent_dimension_values[dimension][parent])
                / len(parent_dimension_values[dimension][parent])
                for dimension in CAPTION_REVIEW_DIMENSIONS
            },
            "overall_score": sum(values) / len(values),
        }
        for parent, values in sorted(parent_overall_values.items())
    }
    metrics = {}
    for index, dimension in enumerate(CAPTION_REVIEW_DIMENSIONS):
        values = [value["dimension_scores"][dimension] for value in per_parent.values()]
        metrics[dimension] = {
            "parent_macro_1_to_5": sum(values) / len(values),
            "parent_macro_normalized_0_to_1": (
                (sum(values) / len(values)) - 1.0
            ) / 4.0,
            "parent_bootstrap_95ci": bootstrap_mean_ci(
                values, seed=int(seed) + 1009 * (index + 1), samples=10000
            ),
        }
    overall_values = [value["overall_score"] for value in per_parent.values()]
    root = resolve_project_path(eval_dir) or Path(eval_dir)
    return {
        "protocol": CAPTION_HUMAN_REPORT_PROTOCOL,
        "status": "expert_review_complete",
        "role": "secondary_human_caption_metrics",
        "checkpoint_selection_allowed": False,
        "input_audit": {
            "eval_dir": str(root.resolve(strict=False)),
            "eval_report_sha256": expected_rows[0]["frozen_eval_report_sha256"],
            "raw_generations_sha256": expected_rows[0]["frozen_generations_sha256"],
            "metric_population_sha256": expected_rows[0]["metric_population_sha256"],
            "review_file_sha256": dict(sorted(review_hashes.items())),
        },
        "num_samples": len(per_sample),
        "num_parents": len(per_parent),
        "reviewer_ids": ordered_reviewers,
        "minimum_reviewers": int(minimum_reviewers),
        "metrics": metrics,
        "overall": {
            "parent_macro_1_to_5": sum(overall_values) / len(overall_values),
            "parent_macro_normalized_0_to_1": (
                (sum(overall_values) / len(overall_values)) - 1.0
            ) / 4.0,
            "parent_bootstrap_95ci": bootstrap_mean_ci(
                overall_values, seed=int(seed) + 7919, samples=10000
            ),
        },
        "agreement": dimension_agreement,
        "per_parent_scores": per_parent,
        "per_sample_scores": per_sample,
        "errors": [],
    }
