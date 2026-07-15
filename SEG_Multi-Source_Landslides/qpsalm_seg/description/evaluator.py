#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Description evaluation with GT, fixed prediction and end-to-end region protocols."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import time
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from qpsalm_seg.data import MultiSourceLandslideDataset, qpsalm_collate
from qpsalm_seg.paths import resolve_project_path

from .common import description_amp_dtype, move_description_batch, write_json
from .config import SegDescConfig
from .counterfactuals import (
    COUNTERFACTUAL_MODES,
    counterfactual_backbone,
    counterfactual_region_masks,
    select_backbone_state,
)
from .metrics import (
    DescriptionMetricAccumulator,
    caption_token_f1,
    bootstrap_mean_ci,
    finite_mean,
    structured_disagreement,
)
from .data import bridge_region_metadata
from .model import SegmentationGroundedDescriptionModel
from .output_protocol import parse_description_output


class EndToEndTargetResolver:
    """Map one Bridge region to the exact segmentation instruction that names it."""

    PROTOCOL = "qpsalm_end_to_end_region_target_v2"
    GLOBAL_FAMILY_PRIORITY = {
        "global_landslide_segmentation": 0,
        "negative_aware_segmentation": 1,
        "multisource_evidence_segmentation": 2,
    }

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        ranked_global: dict[str, tuple[int, int]] = {}
        self.referring: dict[tuple[str, str], int] = {}
        for index, row in enumerate(rows):
            parent = str(row.get("parent_sample_id") or row.get("sample_id"))
            family = str(row.get("task_family") or "")
            if family in self.GLOBAL_FAMILY_PRIORITY:
                priority = self.GLOBAL_FAMILY_PRIORITY[family]
                if parent not in ranked_global or priority < ranked_global[parent][0]:
                    ranked_global[parent] = (priority, index)
            target_id = row.get("parent_referring_target_sample_id")
            if target_id:
                key = (parent, str(target_id))
                previous = self.referring.setdefault(key, index)
                if previous != index:
                    previous_row = rows[previous]
                    if str(previous_row.get("sample_id")) != str(row.get("sample_id")):
                        raise ValueError(f"重复 referring instruction identity: {key}")
        self.global_indices = {
            parent: index for parent, (_priority, index) in ranked_global.items()
        }

    @staticmethod
    def _empty_target(row: dict[str, Any]) -> bool:
        mask = row.get("mask") or {}
        if bool(mask.get("empty_mask")):
            return True
        positive = mask.get("positive_pixels")
        return positive is not None and int(positive) == 0

    @staticmethod
    def _aliases(metadata: dict[str, Any]) -> list[dict[str, Any]]:
        return sorted(
            (
                dict(value) for value in (metadata.get("source_region_aliases") or [])
                if isinstance(value, dict) and value.get("sample_id")
            ),
            key=lambda value: str(value["sample_id"]),
        )

    def _global(self, parent: str) -> tuple[int, str, str | None]:
        index = self.global_indices.get(parent)
        if index is None:
            raise KeyError(f"segmentation split 缺少 global instruction: parent={parent}")
        return index, "global_instruction", None

    def _referring(
        self,
        parent: str,
        aliases: list[dict[str, Any]],
        *,
        expected_family: str,
    ) -> tuple[int, str, str | None]:
        for alias in aliases:
            target_id = str(alias["sample_id"])
            index = self.referring.get((parent, target_id))
            if index is None:
                continue
            family = str(self.rows[index].get("task_family") or "")
            if family == expected_family:
                return index, "referring_alias", target_id
        raise KeyError(
            "segmentation split 缺少精确 referring instruction: "
            f"parent={parent} family={expected_family} "
            f"aliases={[value['sample_id'] for value in aliases[:8]]}"
        )

    def resolve(self, metadata: dict[str, Any]) -> dict[str, Any]:
        parent = str(metadata.get("parent_sample_id") or "")
        source = str(metadata.get("region_source") or "unknown")
        aliases = self._aliases(metadata)
        alias_id: str | None
        if source == "gt_global_mask":
            index, kind, alias_id = self._global(parent)
        elif source in {"gt_referring_mask", "pseudo_instance_component"}:
            if not aliases:
                raise KeyError(
                    f"{source} 没有可识别的 referring alias: "
                    f"parent={parent} region={metadata.get('region_id')}"
                )
            index, kind, alias_id = self._referring(
                parent, aliases, expected_family="referring_landslide_segmentation"
            )
        elif source == "no_target":
            if aliases:
                index, kind, alias_id = self._referring(
                    parent, aliases, expected_family="no_target_segmentation"
                )
            else:
                index, kind, alias_id = self._global(parent)
                if not self._empty_target(self.rows[index]):
                    raise KeyError(
                        "no_target region 既无 no-target alias，parent global target 也非空: "
                        f"parent={parent}"
                    )
                kind = "empty_global_instruction"
        else:
            raise KeyError(
                f"region_source={source!r} 没有端到端 segmentation target protocol"
            )
        row = self.rows[index]
        return {
            "dataset_index": int(index),
            "mapping_kind": kind,
            "alias_sample_id": alias_id,
            "segmentation_sample_id": str(row.get("sample_id")),
            "segmentation_task_family": str(row.get("task_family")),
            "parent_sample_id": parent,
            "bridge_region_id": str(metadata.get("region_id") or "unknown"),
            "bridge_region_source": source,
        }


class EndToEndMaskProvider:
    """Run frozen segmentation only for an exactly resolved Bridge target."""

    def __init__(
        self,
        model: SegmentationGroundedDescriptionModel,
        split: str,
        threshold: float,
    ) -> None:
        config = replace(
            model.segmentation.config,
            modality_dropout=0.0,
            train_hflip_prob=0.0,
            train_vflip_prob=0.0,
        )
        self.dataset = MultiSourceLandslideDataset(config, split)
        self.resolver = EndToEndTargetResolver(self.dataset.rows)
        self.model = model
        self.threshold = float(threshold)
        self.cache: dict[str, torch.Tensor] = {}
        self.mapping_counts: Counter[str] = Counter()

    def require_targets(self, metadata_rows: Iterable[dict[str, Any]]) -> None:
        errors = []
        for metadata in metadata_rows:
            try:
                self.resolver.resolve(metadata)
            except KeyError as exc:
                errors.append(str(exc))
        if errors:
            raise KeyError(
                "end-to-end segmentation target 映射不完整: "
                f"count={len(errors)} examples={errors[:8]}"
            )

    @torch.no_grad()
    def predict(
        self, metadata: dict[str, Any], output_hw: tuple[int, int]
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        audit = self.resolver.resolve(metadata)
        cache_key = str(audit["segmentation_sample_id"])
        probability = self.cache.get(cache_key)
        if probability is None:
            batch = qpsalm_collate([self.dataset[int(audit["dataset_index"])]])
            with self.model.controller.adapter_scope("default"):
                output = self.model.segmentation(batch)
            probability = torch.sigmoid(output.final_mask_logits.float()).detach().cpu()
            self.cache[cache_key] = probability
        self.mapping_counts[str(audit["mapping_kind"])] += 1
        device = next(self.model.parameters()).device
        resized_probability = F.interpolate(
            probability.to(device=device), size=output_hw, mode="bilinear", align_corners=False
        )
        audit = {**audit, "mask_threshold": self.threshold}
        return (resized_probability >= self.threshold).to(resized_probability.dtype), audit

    def summary(self, dataset: Any) -> dict[str, Any]:
        return {
            "protocol": self.resolver.PROTOCOL,
            "source_bridge_rows": int(getattr(dataset, "end_to_end_source_count", len(dataset))),
            "eligible_bridge_rows_before_limit": int(
                getattr(dataset, "end_to_end_eligible_count", len(dataset))
            ),
            "evaluated_rows": len(dataset),
            "excluded_by_reason": dict(sorted(
                getattr(dataset, "end_to_end_exclusion_counts", {}).items()
            )),
            "mapping_counts": dict(sorted(self.mapping_counts.items())),
            "unique_segmentation_inferences": len(self.cache),
            "mask_threshold": self.threshold,
        }


def _same_image_retrieval(
    region_embeddings: list[torch.Tensor],
    text_embeddings: list[torch.Tensor],
    parent_ids: list[str],
) -> dict[str, Any]:
    if not region_embeddings:
        return {"num_queries": 0, "num_multi_candidate_queries": 0, "region_to_text_r1": None, "text_to_region_r1": None}
    region = torch.cat(region_embeddings).float()
    text = torch.cat(text_embeddings).float()
    if region.shape != text.shape or region.shape[0] != len(parent_ids):
        raise ValueError("DIOR retrieval embedding/metadata 数量不一致")
    r2t = t2r = eligible = 0
    per_parent: dict[str, list[float]] = {}
    for index, parent in enumerate(parent_ids):
        candidates = [value for value, current in enumerate(parent_ids) if current == parent]
        if len(candidates) < 2:
            continue
        candidate_tensor = torch.tensor(candidates, device=region.device)
        t2r_correct = int(
            candidates[int((text[index] @ region[candidate_tensor].T).argmax())] == index
        )
        r2t_correct = int(
            candidates[int((region[index] @ text[candidate_tensor].T).argmax())] == index
        )
        t2r += t2r_correct
        r2t += r2t_correct
        eligible += 1
        per_parent.setdefault(parent, []).append(0.5 * (r2t_correct + t2r_correct))
    return {
        "num_queries": len(parent_ids),
        "num_multi_candidate_queries": eligible,
        "region_to_text_r1": r2t / eligible if eligible else None,
        "text_to_region_r1": t2r / eligible if eligible else None,
        "mean_r1": (r2t + t2r) / (2 * eligible) if eligible else None,
        "per_parent_mean_r1": {
            parent: sum(values) / len(values)
            for parent, values in sorted(per_parent.items())
        },
    }


def _counterfactual_modes(config: SegDescConfig) -> tuple[str, ...]:
    values = tuple(config.counterfactual_modes or COUNTERFACTUAL_MODES)
    invalid = sorted(set(values) - set(COUNTERFACTUAL_MODES))
    if invalid:
        raise ValueError(f"未知 counterfactual modes: {invalid}")
    return values


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


@torch.no_grad()
def evaluate_description(
    model: SegmentationGroundedDescriptionModel,
    loader: DataLoader,
    config: SegDescConfig,
    device: torch.device,
    *,
    split: str,
    output_dir: str | Path | None = None,
    max_generate_samples: int | None = None,
    run_counterfactuals: bool = True,
) -> dict[str, Any]:
    model.eval()
    amp_dtype = description_amp_dtype(config, device)
    autocast = device.type == "cuda" and config.amp_dtype != "fp32"
    metric = DescriptionMetricAccumulator()
    losses: list[float] = []
    generation_rows: list[dict[str, Any]] = []
    counterfactual_rows: list[dict[str, Any]] = []
    end_to_end_rows: list[dict[str, Any]] = []
    counterfactual_values: dict[str, list[float]] = {name: [] for name in _counterfactual_modes(config)}
    region_embeddings: list[torch.Tensor] = []
    text_embeddings: list[torch.Tensor] = []
    retrieval_parents: list[str] = []
    generated = 0
    generate_limit = int(config.max_generate_samples if max_generate_samples is None else max_generate_samples)
    counterfactual_counts = {name: 0 for name in _counterfactual_modes(config)}
    e2e = (
        EndToEndMaskProvider(model, split, config.segmentation_mask_threshold)
        if config.evaluation_mode == "end_to_end" else None
    )
    if e2e is not None:
        e2e.require_targets(
            bridge_region_metadata(row)
            for row in getattr(loader.dataset, "rows", [])
        )
    started = time.perf_counter()

    for batch_index, cpu_batch in enumerate(loader):
        batch = move_description_batch(cpu_batch, device)
        backbone = model.encode_description_requests(batch["requests"])
        region_masks = batch["region_masks"]
        batch_e2e_audits: list[dict[str, Any] | None] = [None] * len(batch["metadata"])
        if e2e is not None:
            resolved = [
                e2e.predict(row, tuple(region_masks.shape[-2:]))
                for row in batch["metadata"]
            ]
            predicted = [value[0][0] for value in resolved]
            batch_e2e_audits = [value[1] for value in resolved]
            end_to_end_rows.extend(value[1] for value in resolved)
            region_masks = torch.stack(predicted).to(device=device)

        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast):
            if config.stage == "dior_alignment":
                regions, texts = model.region_alignment_embeddings(
                    backbone, region_masks, batch["target_texts"]
                )
                temperature = model.alignment_temperature.float().clamp(0.01, 1.0)
                logits = regions @ texts.T / temperature
                targets = torch.arange(logits.shape[0], device=logits.device)
                loss = 0.5 * (
                    F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets)
                )
                region_embeddings.append(regions.detach().cpu())
                text_embeddings.append(texts.detach().cpu())
                retrieval_parents.extend(str(row["parent_sample_id"]) for row in batch["metadata"])
            else:
                output = model.describe_from_state(
                    backbone,
                    region_masks,
                    batch["instructions"],
                    target_texts=batch["target_texts"],
                    region_valid_mask=backbone.valid_mask,
                    protocol=config.region_protocol,
                    structured_output=batch["structured_outputs"],
                )
                if output.per_sample_loss is None:
                    raise RuntimeError("description validation 未产生 per-sample loss")
                loss = (output.per_sample_loss * batch["weights"]).sum() / batch["weights"].sum().clamp_min(1.0)
        losses.append(float(loss.detach().cpu()))

        if config.stage == "dior_alignment" or generate_limit <= generated:
            continue
        baseline_texts: list[str] = []
        batch_generation_count = min(region_masks.shape[0], generate_limit - generated)
        for sample_index in range(batch_generation_count):
            one_state = select_backbone_state(backbone, [sample_index])
            one_mask = region_masks[sample_index:sample_index + 1]
            structured = bool(batch["structured_outputs"][sample_index])
            raw = model.generate_from_state(
                one_state,
                one_mask,
                batch["instructions"][sample_index],
                max_new_tokens=config.max_new_tokens,
                protocol=config.region_protocol,
                structured_output=structured,
            )
            baseline_texts.append(raw)
            details = metric.update(
                prediction=raw,
                target_text=batch["target_texts"][sample_index],
                references=batch["reference_texts"][sample_index],
                structured=structured,
                metadata=batch["metadata"][sample_index],
            )
            generation_rows.append({
                **batch["metadata"][sample_index],
                "end_to_end_segmentation_target": batch_e2e_audits[sample_index],
                "split": split,
                "evaluation_mode": config.evaluation_mode,
                "instruction": batch["instructions"][sample_index],
                "target_text": batch["target_texts"][sample_index],
                "raw_generation": raw,
                "raw_metrics": details,
                "region_area_fraction": float(one_mask.float().mean().cpu()),
            })
            generated += 1

        if not run_counterfactuals or all(
            value >= int(config.counterfactual_samples)
            for value in counterfactual_counts.values()
        ):
            continue
        for mode in _counterfactual_modes(config):
            if counterfactual_counts[mode] >= int(config.counterfactual_samples):
                continue
            if mode == "cross_parent_modality_swap" and backbone.valid_mask.shape[0] < 2:
                continue
            try:
                if mode in {"full_mask", "zero_mask", "shuffled_mask", "region_swap"}:
                    cf_backbone = backbone
                    cf_masks = counterfactual_region_masks(region_masks, mode)
                else:
                    cf_backbone = counterfactual_backbone(backbone, mode)
                    cf_masks = region_masks
            except ValueError:
                continue
            for sample_index, baseline in enumerate(baseline_texts):
                if counterfactual_counts[mode] >= int(config.counterfactual_samples):
                    break
                one_state = select_backbone_state(cf_backbone, [sample_index])
                structured = bool(batch["structured_outputs"][sample_index])
                changed = model.generate_from_state(
                    one_state,
                    cf_masks[sample_index:sample_index + 1],
                    batch["instructions"][sample_index],
                    max_new_tokens=config.max_new_tokens,
                    protocol=config.region_protocol,
                    structured_output=structured,
                )
                if structured:
                    sensitivity = structured_disagreement(
                        parse_description_output(baseline).parsed,
                        parse_description_output(changed).parsed,
                    )
                else:
                    sensitivity = 1.0 - caption_token_f1(changed, [baseline])
                counterfactual_values[mode].append(sensitivity)
                counterfactual_rows.append({
                    **batch["metadata"][sample_index],
                    "mode": mode,
                    "baseline_generation": baseline,
                    "counterfactual_generation": changed,
                    "sensitivity": sensitivity,
                })
                counterfactual_counts[mode] += 1

    report = {
        "protocol": "qpsalm_description_evaluation_v1",
        "stage": config.stage,
        "split": split,
        "evaluation_mode": config.evaluation_mode,
        "region_protocol": config.region_protocol,
        "num_samples": len(loader.dataset),
        "num_generated": generated,
        "mean_teacher_forced_loss": finite_mean(losses),
        "generation_metrics": metric.compute(),
        "primary_score_bootstrap_ci": bootstrap_mean_ci(
            [
                float((row.get("raw_metrics") or {}).get("raw_field_accuracy"))
                if (row.get("raw_metrics") or {}).get("raw_field_accuracy") is not None
                else float((row.get("raw_metrics") or {}).get("caption_token_f1", 0.0))
                for row in generation_rows
            ],
            seed=config.seed + 7919,
        ),
        "same_image_retrieval": _same_image_retrieval(
            region_embeddings, text_embeddings, retrieval_parents
        ),
        "counterfactual_sensitivity": {
            name: {"n": len(values), "mean_disagreement": finite_mean(values)}
            for name, values in counterfactual_values.items()
        },
        "end_to_end_coverage": e2e.summary(loader.dataset) if e2e is not None else None,
        "elapsed_seconds": time.perf_counter() - started,
    }
    if output_dir is not None:
        resolved = resolve_project_path(output_dir) or Path(output_dir)
        resolved.mkdir(parents=True, exist_ok=True)
        _write_jsonl(resolved / "raw_generations.jsonl", generation_rows)
        _write_jsonl(resolved / "counterfactual_generations.jsonl", counterfactual_rows)
        if e2e is not None:
            _write_jsonl(resolved / "end_to_end_target_audit.jsonl", end_to_end_rows)
        write_json(resolved / "eval_report.json", report)
    return report


def description_selection_score(report: dict[str, Any], stage: str, metric_name: str = "auto") -> float:
    if metric_name != "auto":
        path: Any = report
        for part in metric_name.split("."):
            path = path.get(part) if isinstance(path, dict) else None
        if path is None:
            raise KeyError(f"checkpoint metric 不存在: {metric_name}")
        return float(path)
    if stage == "dior_alignment":
        return float((report.get("same_image_retrieval") or {}).get("mean_r1") or 0.0)
    generation = report.get("generation_metrics") or {}
    if stage in {"bridge_auto", "bridge_expert", "predicted_mask", "overfit"}:
        return float(generation.get("structured_field_macro_f1") or 0.0)
    return float(generation.get("caption_token_f1") or 0.0)
