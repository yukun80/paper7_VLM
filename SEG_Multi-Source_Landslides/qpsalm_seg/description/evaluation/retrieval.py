"""Same-image retrieval metrics and small evaluation helpers."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch

from ..protocols.config import SegDescConfig
from ..protocols.io import atomic_write_jsonl, canonical_sha256
from .contracts import SAME_IMAGE_RETRIEVAL_PROTOCOL
from .counterfactuals import COUNTERFACTUAL_MODES


def same_image_retrieval(
    region_embeddings: list[torch.Tensor],
    text_embeddings: list[torch.Tensor],
    parent_ids: list[str],
    phrase_labels: list[str] | None = None,
    sample_ids: list[str] | None = None,
) -> dict[str, Any]:
    if not region_embeddings:
        return {
            "protocol": SAME_IMAGE_RETRIEVAL_PROTOCOL,
            "population_identity_complete": sample_ids is not None,
            "population_sha256": canonical_sha256([]) if sample_ids is not None else None,
            "num_queries": 0,
            "num_multi_candidate_queries": 0,
            "region_to_text_r1": None,
            "text_to_region_r1": None,
            "mean_r1": None,
            "region_to_text_r5": None,
            "text_to_region_r5": None,
            "mean_r5": None,
            "normalized_phrase_match": None,
            "modifier_accuracy": None,
            "region_to_text_ranking_margin": None,
            "text_to_region_ranking_margin": None,
            "mean_ranking_margin": None,
            "aggregation_unit": "parent",
            "per_parent": {},
            "per_parent_mean_r1": {},
        }
    region = torch.cat(region_embeddings).float()
    text = torch.cat(text_embeddings).float()
    if region.shape != text.shape or region.shape[0] != len(parent_ids):
        raise ValueError("DIOR retrieval embedding/metadata 数量不一致")
    labels = (
        [" ".join(str(value).casefold().split()) for value in phrase_labels]
        if phrase_labels is not None else [f"pair:{index}" for index in range(len(parent_ids))]
    )
    if len(labels) != len(parent_ids):
        raise ValueError("DIOR retrieval phrase label 数量不一致")
    identity_complete = sample_ids is not None
    resolved_sample_ids = list(sample_ids or [f"row:{index}" for index in range(len(parent_ids))])
    if len(resolved_sample_ids) != len(parent_ids):
        raise ValueError("DIOR retrieval sample identity 数量不一致")
    if identity_complete and len(set(resolved_sample_ids)) != len(resolved_sample_ids):
        raise ValueError("DIOR retrieval sample_id 必须唯一")
    population = sorted(
        [
            {
                "sample_id": str(sample),
                "parent_sample_id": str(parent),
                "normalized_phrase": str(label),
            }
            for sample, parent, label in zip(resolved_sample_ids, parent_ids, labels)
        ],
        key=lambda value: value["sample_id"],
    )
    def modifiers(value: str) -> tuple[str, ...]:
        tokens = [
            token for token in str(value).casefold().split()
            if token not in {"a", "an", "the", "of", "in", "on", "at"}
        ]
        return tuple(tokens[:-1]) if len(tokens) > 1 else ()

    eligible = 0
    ambiguous = 0
    parent_metrics: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for index, parent in enumerate(parent_ids):
        candidates = [value for value, current in enumerate(parent_ids) if current == parent]
        if len(candidates) < 2:
            continue
        candidate_tensor = torch.tensor(candidates, device=region.device)
        text_to_region_scores = text[index] @ region[candidate_tensor].T
        region_to_text_scores = region[index] @ text[candidate_tensor].T
        text_to_region_rank = [
            candidates[int(value)]
            for value in torch.argsort(text_to_region_scores, descending=True).tolist()
        ]
        region_to_text_rank = [
            candidates[int(value)]
            for value in torch.argsort(region_to_text_scores, descending=True).tolist()
        ]
        selected_region = text_to_region_rank[0]
        selected_text = region_to_text_rank[0]
        positives = {value for value in candidates if labels[value] == labels[index]}
        ambiguous += int(len(positives) > 1)
        negatives = [value for value in candidates if value not in positives]
        t2r_correct = float(selected_region in positives)
        r2t_correct = float(selected_text in positives)
        values = parent_metrics[str(parent)]
        values["text_to_region_r1"].append(t2r_correct)
        values["region_to_text_r1"].append(r2t_correct)
        values["text_to_region_r5"].append(
            float(bool(set(text_to_region_rank[:5]) & positives))
        )
        values["region_to_text_r5"].append(
            float(bool(set(region_to_text_rank[:5]) & positives))
        )
        values["normalized_phrase_match"].append(
            float(labels[selected_text] == labels[index])
        )
        values["modifier_accuracy"].append(
            float(modifiers(labels[selected_text]) == modifiers(labels[index]))
        )
        if negatives:
            negative_positions = [candidates.index(value) for value in negatives]
            positive_positions = [candidates.index(value) for value in positives]
            values["text_to_region_margin"].append(float(
                text_to_region_scores[positive_positions].max()
                - text_to_region_scores[negative_positions].max()
            ))
            values["region_to_text_margin"].append(float(
                region_to_text_scores[positive_positions].max()
                - region_to_text_scores[negative_positions].max()
            ))
        eligible += 1
    per_parent = {
        parent: {
            name: sum(values) / len(values)
            for name, values in sorted(metrics.items()) if values
        }
        for parent, metrics in sorted(parent_metrics.items())
    }

    def parent_macro(name: str) -> float | None:
        values = [metrics[name] for metrics in per_parent.values() if name in metrics]
        return sum(values) / len(values) if values else None

    r2t_r1 = parent_macro("region_to_text_r1")
    t2r_r1 = parent_macro("text_to_region_r1")
    return {
        "protocol": SAME_IMAGE_RETRIEVAL_PROTOCOL,
        "population_identity_complete": identity_complete,
        "population_sha256": canonical_sha256(population),
        "num_queries": len(parent_ids),
        "num_multi_candidate_queries": eligible,
        "num_ambiguous_phrase_queries": ambiguous,
        "aggregation_unit": "parent",
        "region_to_text_r1": r2t_r1,
        "text_to_region_r1": t2r_r1,
        "mean_r1": (
            (r2t_r1 + t2r_r1) * 0.5
            if r2t_r1 is not None and t2r_r1 is not None else None
        ),
        "region_to_text_r5": parent_macro("region_to_text_r5"),
        "text_to_region_r5": parent_macro("text_to_region_r5"),
        "mean_r5": (
            (
                parent_macro("region_to_text_r5")
                + parent_macro("text_to_region_r5")
            ) * 0.5
            if parent_macro("region_to_text_r5") is not None
            and parent_macro("text_to_region_r5") is not None else None
        ),
        "normalized_phrase_match": parent_macro("normalized_phrase_match"),
        "modifier_accuracy": parent_macro("modifier_accuracy"),
        "region_to_text_ranking_margin": parent_macro("region_to_text_margin"),
        "text_to_region_ranking_margin": parent_macro("text_to_region_margin"),
        "mean_ranking_margin": (
            (
                parent_macro("region_to_text_margin")
                + parent_macro("text_to_region_margin")
            ) * 0.5
            if parent_macro("region_to_text_margin") is not None
            and parent_macro("text_to_region_margin") is not None else None
        ),
        "per_parent": per_parent,
        "per_parent_mean_r1": {
            parent: 0.5 * (
                metrics["region_to_text_r1"] + metrics["text_to_region_r1"]
            )
            for parent, metrics in per_parent.items()
        },
    }


def counterfactual_modes(config: SegDescConfig) -> tuple[str, ...]:
    values = tuple(config.evaluation.counterfactual_modes or COUNTERFACTUAL_MODES)
    invalid = sorted(set(values) - set(COUNTERFACTUAL_MODES))
    if invalid:
        raise ValueError(f"未知 counterfactual modes: {invalid}")
    return values


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    atomic_write_jsonl(path, rows)
