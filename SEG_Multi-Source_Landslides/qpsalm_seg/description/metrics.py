#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Raw-generation metrics for caption, region alignment and Bridge outputs.

Primary structured metrics intentionally consume the un-repaired parse.  The
deterministic repair path is reported separately and can never improve the
scientific score used for checkpoint selection.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
import re
from typing import Any, Iterable

import torch
import numpy as np

from .output_protocol import ParsedDescription, parse_description_output


REGION_FIELDS = (
    "location", "size_class", "shape", "elongation", "compactness", "fragmentation",
)
EVIDENCE_FIELDS = (
    "surface_observation", "terrain_support", "sar_support", "deformation_support",
    "surrounding_context", "evidence_sufficiency",
)
STRUCTURED_FIELDS = ("target_status",) + tuple(
    f"region.{name}" for name in REGION_FIELDS
) + tuple(f"evidence.{name}" for name in EVIDENCE_FIELDS)
UNSUPPORTED_VALUES = {"unknown", "unavailable", "insufficient", "insufficient_evidence", ""}


def _field(value: dict[str, Any] | None, path: str) -> Any:
    current: Any = value
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(text).casefold())


def caption_token_f1(prediction: str, references: Iterable[str]) -> float:
    """Best-reference bag-of-words F1, used as a lightweight secondary metric."""
    predicted = Counter(_tokens(prediction))
    best = 0.0
    for reference in references:
        target = Counter(_tokens(reference))
        overlap = sum((predicted & target).values())
        precision = overlap / max(sum(predicted.values()), 1)
        recall = overlap / max(sum(target.values()), 1)
        score = 2 * precision * recall / max(precision + recall, 1.0e-12)
        best = max(best, score)
    return best


def _is_factual_claim(value: Any) -> bool:
    text = str(value or "").strip().casefold()
    return bool(text and text not in UNSUPPORTED_VALUES and text != "no reliable description is available.")


def unsupported_claim_counts(
    prediction: dict[str, Any] | None,
    target: dict[str, Any] | None,
) -> tuple[int, int]:
    """Count unsupported evidence claims per factual claim, not per sample."""
    if not isinstance(prediction, dict):
        return 0, 0
    unsupported = claims = 0
    categorical = {
        "terrain_support", "sar_support", "deformation_support", "evidence_sufficiency",
    }
    for name in EVIDENCE_FIELDS:
        predicted = _field(prediction, f"evidence.{name}")
        if not _is_factual_claim(predicted):
            continue
        claims += 1
        expected = _field(target, f"evidence.{name}")
        if not _is_factual_claim(expected) or (
            name in categorical and str(predicted).strip().casefold() != str(expected).strip().casefold()
        ):
            unsupported += 1
    return unsupported, claims


def structured_disagreement(
    first: dict[str, Any] | None,
    second: dict[str, Any] | None,
) -> float:
    if not isinstance(first, dict) or not isinstance(second, dict):
        return float(first != second)
    changed = sum(_field(first, name) != _field(second, name) for name in STRUCTURED_FIELDS)
    return changed / len(STRUCTURED_FIELDS)


class DescriptionMetricAccumulator:
    """Accumulate scientific raw metrics and clearly separated repair diagnostics."""

    def __init__(self) -> None:
        self.samples = 0
        self.structured_samples = 0
        self.caption_samples = 0
        self.raw_parse_valid = 0
        self.raw_schema_valid = 0
        self.repair_schema_valid = 0
        self.field_correct = Counter()
        self.field_total = Counter()
        self.field_confusion = Counter()
        self.status_confusion = Counter()
        self.unsupported = 0
        self.claims = 0
        self.caption_f1_sum = 0.0
        self.by_task: dict[str, list[float]] = defaultdict(list)

    def update(
        self,
        *,
        prediction: str,
        target_text: str,
        structured: bool,
        metadata: dict[str, Any],
        references: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        self.samples += 1
        task = str(metadata.get("task_family") or "unknown")
        if not structured:
            score = caption_token_f1(prediction, references or [target_text])
            self.caption_samples += 1
            self.caption_f1_sum += score
            self.by_task[task].append(score)
            return {"caption_token_f1": score, "raw_schema_valid": None}

        self.structured_samples += 1
        predicted = parse_description_output(prediction)
        target = parse_description_output(target_text)
        if target.parsed is None or not target.schema_valid:
            raise ValueError(f"Bridge target 不是合法 schema output: {metadata.get('sample_id')}")
        self.raw_parse_valid += int(predicted.parsed is not None)
        self.raw_schema_valid += int(predicted.schema_valid)
        # deterministic_repair always follows the schema by construction; keep
        # this as an engineering diagnostic rather than a scientific score.
        repaired = parse_description_output(json.dumps(predicted.repaired, ensure_ascii=False))
        self.repair_schema_valid += int(repaired.schema_valid)
        sample_correct = 0
        for name in STRUCTURED_FIELDS:
            self.field_total[name] += 1
            match = predicted.parsed is not None and _field(predicted.parsed, name) == _field(target.parsed, name)
            expected = str(_field(target.parsed, name))
            observed = str(_field(predicted.parsed, name)) if predicted.parsed is not None else "__invalid__"
            self.field_confusion[(name, expected, observed)] += 1
            self.field_correct[name] += int(match)
            sample_correct += int(match)
        expected_status = str(_field(target.parsed, "target_status") or "invalid")
        observed_status = str(_field(predicted.parsed, "target_status") or "invalid")
        self.status_confusion[(expected_status, observed_status)] += 1
        unsupported, claims = unsupported_claim_counts(predicted.parsed, target.parsed)
        self.unsupported += unsupported
        self.claims += claims
        field_accuracy = sample_correct / len(STRUCTURED_FIELDS)
        self.by_task[task].append(field_accuracy)
        return {
            "raw_schema_valid": predicted.schema_valid,
            "raw_field_accuracy": field_accuracy,
            "unsupported_claims": unsupported,
            "factual_claims": claims,
            "parse_errors": list(predicted.parse_errors),
            "repair_actions": list(predicted.repair_actions),
            "parsed": predicted.parsed,
            "repaired": predicted.repaired,
        }

    def _status_metrics(self) -> dict[str, Any]:
        labels = ("present", "absent", "uncertain")
        per_label = {}
        recalls = []
        f1s = []
        active_labels = []
        for label in labels:
            tp = self.status_confusion[(label, label)]
            fn = sum(self.status_confusion[(label, value)] for value in (*labels, "invalid") if value != label)
            fp = sum(self.status_confusion[(value, label)] for value in labels if value != label)
            recall = tp / max(tp + fn, 1)
            precision = tp / max(tp + fp, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1.0e-12)
            per_label[label] = {"precision": precision, "recall": recall, "f1": f1, "support": tp + fn}
            if tp + fn > 0:
                active_labels.append(label)
                recalls.append(recall)
                f1s.append(f1)
        absent_total = sum(self.status_confusion[("absent", value)] for value in (*labels, "invalid"))
        absent_false_description = sum(
            count for (expected, predicted), count in self.status_confusion.items()
            if expected == "absent" and predicted not in {"absent"}
        )
        present_total = sum(self.status_confusion[("present", value)] for value in (*labels, "invalid"))
        present_false_rejection = self.status_confusion[("present", "absent")]
        return {
            "macro_f1": sum(f1s) / max(len(f1s), 1),
            "balanced_accuracy": sum(recalls) / max(len(recalls), 1),
            "active_labels": active_labels,
            "per_label": per_label,
            "false_description_rate": absent_false_description / max(absent_total, 1),
            "positive_false_rejection_rate": present_false_rejection / max(present_total, 1),
            "confusion": {
                f"{expected}->{predicted}": count
                for (expected, predicted), count in sorted(self.status_confusion.items())
            },
        }

    def compute(self) -> dict[str, Any]:
        field_accuracy = {
            name: self.field_correct[name] / max(self.field_total[name], 1)
            for name in STRUCTURED_FIELDS
        }
        macro = sum(field_accuracy.values()) / max(len(field_accuracy), 1)
        field_macro_f1 = {}
        for field in STRUCTURED_FIELDS:
            labels = sorted({
                expected for (name, expected, _observed) in self.field_confusion if name == field
            })
            f1_values = []
            for label in labels:
                tp = self.field_confusion[(field, label, label)]
                fn = sum(
                    count for (name, expected, observed), count in self.field_confusion.items()
                    if name == field and expected == label and observed != label
                )
                fp = sum(
                    count for (name, expected, observed), count in self.field_confusion.items()
                    if name == field and expected != label and observed == label
                )
                precision = tp / max(tp + fp, 1)
                recall = tp / max(tp + fn, 1)
                f1_values.append(2 * precision * recall / max(precision + recall, 1.0e-12))
            field_macro_f1[field] = sum(f1_values) / max(len(f1_values), 1)
        return {
            "num_samples": self.samples,
            "num_structured": self.structured_samples,
            "num_caption": self.caption_samples,
            "raw_json_parse_rate": self.raw_parse_valid / max(self.structured_samples, 1),
            "raw_schema_valid_rate": self.raw_schema_valid / max(self.structured_samples, 1),
            "repair_schema_valid_rate": self.repair_schema_valid / max(self.structured_samples, 1),
            "structured_field_macro_accuracy": macro,
            "structured_field_accuracy": field_accuracy,
            "structured_field_macro_f1": sum(field_macro_f1.values()) / max(len(field_macro_f1), 1),
            "structured_field_f1": field_macro_f1,
            "unsupported_factual_claim_rate": self.unsupported / max(self.claims, 1),
            "unsupported_factual_claims": self.unsupported,
            "factual_claims": self.claims,
            "caption_token_f1": self.caption_f1_sum / max(self.caption_samples, 1),
            "target_status": self._status_metrics(),
            "by_task": {
                name: {"n": len(values), "mean_primary_score": sum(values) / max(len(values), 1)}
                for name, values in sorted(self.by_task.items())
            },
        }


def retrieval_metrics(logits: torch.Tensor) -> dict[str, float | int]:
    if logits.ndim != 2 or logits.shape[0] != logits.shape[1]:
        raise ValueError(f"retrieval logits 必须为方阵，当前 {tuple(logits.shape)}")
    targets = torch.arange(logits.shape[0], device=logits.device)
    region_to_text = (logits.argmax(1) == targets).float().mean()
    text_to_region = (logits.argmax(0) == targets).float().mean()
    return {
        "num_pairs": int(logits.shape[0]),
        "region_to_text_r1": float(region_to_text.detach().cpu()),
        "text_to_region_r1": float(text_to_region.detach().cpu()),
        "mean_r1": float(((region_to_text + text_to_region) * 0.5).detach().cpu()),
    }


def finite_mean(values: Iterable[float]) -> float | None:
    selected = [float(value) for value in values if math.isfinite(float(value))]
    return sum(selected) / len(selected) if selected else None


def bootstrap_mean_ci(
    values: Iterable[float],
    *,
    seed: int,
    samples: int = 2000,
    confidence: float = 0.95,
) -> dict[str, float | int | None]:
    array = np.asarray([float(value) for value in values if math.isfinite(float(value))], dtype=np.float64)
    if not len(array):
        return {"n": 0, "mean": None, "low": None, "high": None, "confidence": confidence}
    rng = np.random.default_rng(int(seed))
    draws = rng.choice(array, size=(max(1, int(samples)), len(array)), replace=True).mean(1)
    alpha = (1.0 - float(confidence)) / 2.0
    return {
        "n": int(len(array)),
        "mean": float(array.mean()),
        "low": float(np.quantile(draws, alpha)),
        "high": float(np.quantile(draws, 1.0 - alpha)),
        "confidence": float(confidence),
    }


def paired_bootstrap_delta_ci(
    baseline: Iterable[float],
    candidate: Iterable[float],
    *,
    seed: int,
    samples: int = 5000,
) -> dict[str, float | int | None]:
    first = np.asarray(list(baseline), dtype=np.float64)
    second = np.asarray(list(candidate), dtype=np.float64)
    if first.shape != second.shape:
        raise ValueError(f"paired bootstrap shape 不一致: {first.shape} vs {second.shape}")
    return bootstrap_mean_ci(second - first, seed=seed, samples=samples)
