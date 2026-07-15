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
from torch.utils.data import DataLoader
import numpy as np

from qpsalm_seg.data import MultiSourceLandslideDataset, qpsalm_collate
from qpsalm_seg.paths import resolve_project_path
from qpsalm_seg.visualize import restore_mask_to_original

from .common import description_amp_dtype, move_description_batch, write_json
from .backbone import transform_region_mask_to_cache
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
    unsupported_claim_counts,
)
from .data import bridge_region_metadata
from .model import (
    SegmentationGroundedDescriptionModel,
    alignment_positive_mask,
    multi_positive_alignment_loss,
)
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
        self.cache: dict[str, tuple[torch.Tensor, dict[str, Any]]] = {}
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
        cached = self.cache.get(cache_key)
        if cached is None:
            item = self.dataset[int(audit["dataset_index"])]
            batch = qpsalm_collate([item])
            with self.model.controller.adapter_scope("default"):
                output = self.model.segmentation(batch)
            canvas = (
                torch.sigmoid(output.final_mask_logits[0, 0].float()).detach().cpu().numpy()
                >= self.threshold
            ).astype(np.uint8)
            segmentation_transform = dict(item["metadata"].get("resize_transform") or {})
            restored = restore_mask_to_original(canvas, segmentation_transform)
            if restored is None:
                raise ValueError(
                    "end-to-end mask 无法按 segmentation resize transform 恢复: "
                    f"sample={cache_key}"
                )
            original_mask = torch.from_numpy(restored.astype(np.float32))[None]
            cached = (original_mask, segmentation_transform)
            self.cache[cache_key] = cached
        original_mask, segmentation_transform = cached
        self.mapping_counts[str(audit["mapping_kind"])] += 1
        device = next(self.model.parameters()).device
        parent = str(audit["parent_sample_id"])
        cache_record = self.model.description_backbone.bank.record(
            "multisource_parent", parent
        )
        description_transform = dict(cache_record["views"][0]["render_transform"])
        description_mask = transform_region_mask_to_cache(
            original_mask, description_transform
        ).to(device=device)
        if tuple(description_mask.shape[-2:]) != tuple(output_hw):
            raise ValueError(
                "end-to-end description mask canvas 不一致: "
                f"parent={parent} observed={tuple(description_mask.shape[-2:])} "
                f"expected={tuple(output_hw)}"
            )
        audit = {
            **audit,
            "mask_threshold": self.threshold,
            "segmentation_resize_transform": segmentation_transform,
            "description_render_transform": description_transform,
            "original_mask_shape": list(original_mask.shape[-2:]),
        }
        return description_mask.unsqueeze(0), audit

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
    phrase_labels: list[str] | None = None,
) -> dict[str, Any]:
    if not region_embeddings:
        return {"num_queries": 0, "num_multi_candidate_queries": 0, "region_to_text_r1": None, "text_to_region_r1": None}
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
    r2t = t2r = eligible = 0
    ambiguous = 0
    per_parent: dict[str, list[float]] = {}
    for index, parent in enumerate(parent_ids):
        candidates = [value for value, current in enumerate(parent_ids) if current == parent]
        if len(candidates) < 2:
            continue
        candidate_tensor = torch.tensor(candidates, device=region.device)
        selected_region = candidates[int((text[index] @ region[candidate_tensor].T).argmax())]
        selected_text = candidates[int((region[index] @ text[candidate_tensor].T).argmax())]
        positives = {value for value in candidates if labels[value] == labels[index]}
        ambiguous += int(len(positives) > 1)
        t2r_correct = int(selected_region in positives)
        r2t_correct = int(selected_text in positives)
        t2r += t2r_correct
        r2t += r2t_correct
        eligible += 1
        per_parent.setdefault(parent, []).append(0.5 * (r2t_correct + t2r_correct))
    return {
        "num_queries": len(parent_ids),
        "num_multi_candidate_queries": eligible,
        "num_ambiguous_phrase_queries": ambiguous,
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
    counterfactual_score_deltas: dict[str, list[float]] = {
        name: [] for name in _counterfactual_modes(config)
    }
    counterfactual_claim_deltas: dict[str, list[float]] = {
        name: [] for name in _counterfactual_modes(config)
    }
    region_embeddings: list[torch.Tensor] = []
    text_embeddings: list[torch.Tensor] = []
    retrieval_parents: list[str] = []
    retrieval_phrases: list[str] = []
    generated = 0
    requested_generate = int(
        config.max_generate_samples if max_generate_samples is None else max_generate_samples
    )
    generate_limit = len(loader.dataset) if requested_generate <= 0 else min(
        requested_generate, len(loader.dataset)
    )
    counterfactual_counts = {name: 0 for name in _counterfactual_modes(config)}
    counterfactual_skipped_no_effect = {
        name: 0 for name in _counterfactual_modes(config)
    }
    counterfactual_skipped_unavailable = {
        name: 0 for name in _counterfactual_modes(config)
    }
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
        backbone = model.encode_description_requests(
            batch["requests"],
            include_spatial=config.stage not in {"mmrs_caption", "rsicap_caption"},
        )
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
                positive_mask = alignment_positive_mask(
                    batch["target_texts"],
                    [str(row["parent_sample_id"]) for row in batch["metadata"]],
                    device=logits.device,
                )
                loss = multi_positive_alignment_loss(logits, positive_mask)
                region_embeddings.append(regions.detach().cpu())
                text_embeddings.append(texts.detach().cpu())
                retrieval_parents.extend(str(row["parent_sample_id"]) for row in batch["metadata"])
                retrieval_phrases.extend(str(value) for value in batch["target_texts"])
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
            counterfactual_inputs: list[dict[str, Any] | None] = [
                None for _ in batch["metadata"]
            ]
            counterfactual_unavailable = [False for _ in batch["metadata"]]
            try:
                if mode == "region_swap":
                    cf_backbone = backbone
                    cf_masks = region_masks.clone()
                    resolver = getattr(loader.dataset, "same_parent_region_swap", None)
                    if resolver is None:
                        continue
                    for sample_index, metadata in enumerate(batch["metadata"]):
                        resolved = resolver(
                            str(metadata["sample_id"]),
                            region_masks[sample_index],
                        )
                        if resolved is not None:
                            alternate, counterfactual_inputs[sample_index] = resolved
                            cf_masks[sample_index] = alternate.to(
                                device=region_masks.device,
                                dtype=region_masks.dtype,
                            )
                        else:
                            counterfactual_unavailable[sample_index] = True
                elif mode in {"full_mask", "zero_mask", "shuffled_mask"}:
                    cf_backbone = backbone
                    cf_masks = counterfactual_region_masks(region_masks, mode)
                elif mode == "modality_removal":
                    cf_backbone = counterfactual_backbone(backbone, mode)
                    cf_masks = region_masks
                else:
                    # Cross-parent donors are resolved per sample below. This
                    # remains valid when formal evaluation uses batch_size=1.
                    cf_backbone = backbone
                    cf_masks = region_masks
            except ValueError:
                continue
            for sample_index, baseline in enumerate(baseline_texts):
                if counterfactual_counts[mode] >= int(config.counterfactual_samples):
                    break
                one_state = None
                if mode == "cross_parent_modality_swap":
                    donor_resolver = getattr(
                        loader.dataset, "cross_parent_modality_swap_request", None
                    )
                    resolved_donor = (
                        donor_resolver(str(batch["metadata"][sample_index]["sample_id"]))
                        if donor_resolver is not None else None
                    )
                    if resolved_donor is None:
                        effective = False
                        counterfactual_unavailable[sample_index] = True
                    else:
                        donor_request, donor_audit = resolved_donor
                        pair_backbone = model.encode_description_requests([
                            batch["requests"][sample_index], donor_request,
                        ])
                        swapped_pair = counterfactual_backbone(
                            pair_backbone, "cross_parent_modality_swap"
                        )
                        one_state = select_backbone_state(swapped_pair, [0])
                        swap_audit = swapped_pair.metadata[0].get(
                            "counterfactual_modality_swap"
                        )
                        counterfactual_inputs[sample_index] = {
                            **donor_audit,
                            "applied_swap": swap_audit,
                        }
                        effective = swap_audit is not None
                elif mode in {"full_mask", "zero_mask", "shuffled_mask", "region_swap"}:
                    effective = not torch.equal(
                        cf_masks[sample_index], region_masks[sample_index]
                    )
                elif mode == "modality_removal":
                    effective = (
                        cf_backbone.active_subsets[sample_index].active_names
                        != backbone.active_subsets[sample_index].active_names
                    )
                else:
                    effective = not cf_backbone.active_subsets[
                        sample_index
                    ].signature.endswith(":none")
                if not effective:
                    if counterfactual_unavailable[sample_index]:
                        counterfactual_skipped_unavailable[mode] += 1
                    else:
                        counterfactual_skipped_no_effect[mode] += 1
                    continue
                if one_state is None:
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
                    baseline_parsed = parse_description_output(baseline).parsed
                    changed_parsed = parse_description_output(changed).parsed
                    target_parsed = parse_description_output(
                        batch["target_texts"][sample_index]
                    ).parsed
                    sensitivity = structured_disagreement(
                        baseline_parsed,
                        changed_parsed,
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
                    sensitivity = 1.0 - caption_token_f1(changed, [baseline])
                    references = batch["reference_texts"][sample_index]
                    baseline_score = caption_token_f1(baseline, references)
                    changed_score = caption_token_f1(changed, references)
                    baseline_claims = changed_claims = 0
                score_delta = changed_score - baseline_score
                claim_delta = float(changed_claims - baseline_claims)
                counterfactual_values[mode].append(sensitivity)
                counterfactual_score_deltas[mode].append(score_delta)
                counterfactual_claim_deltas[mode].append(claim_delta)
                counterfactual_rows.append({
                    **batch["metadata"][sample_index],
                    "mode": mode,
                    "counterfactual_input": counterfactual_inputs[sample_index],
                    "baseline_generation": baseline,
                    "counterfactual_generation": changed,
                    "sensitivity": sensitivity,
                    "baseline_target_score": baseline_score,
                    "counterfactual_target_score": changed_score,
                    "target_score_delta": score_delta,
                    "factual_claim_count_delta": claim_delta,
                })
                counterfactual_counts[mode] += 1

    report = {
        "protocol": "qpsalm_description_evaluation_v3",
        "stage": config.stage,
        "split": split,
        "evaluation_mode": config.evaluation_mode,
        "region_protocol": config.region_protocol,
        "num_samples": len(loader.dataset),
        "num_generated": generated,
        "generation_coverage": {
            "requested": requested_generate,
            "eligible_samples": len(loader.dataset),
            "generated_samples": generated,
            "fraction": generated / max(len(loader.dataset), 1),
            "complete": generated == len(loader.dataset),
        },
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
            region_embeddings, text_embeddings, retrieval_parents, retrieval_phrases
        ),
        "counterfactual_sensitivity": {
            name: {
                "requested": int(config.counterfactual_samples),
                "n": len(values),
                "coverage_complete": (
                    int(config.counterfactual_samples) > 0
                    and len(values) >= int(config.counterfactual_samples)
                ),
                "mean_disagreement": finite_mean(values),
                "mean_target_score_delta": finite_mean(
                    counterfactual_score_deltas[name]
                ),
                "paired_target_score_delta_ci": bootstrap_mean_ci(
                    counterfactual_score_deltas[name],
                    seed=config.seed + 104729 * (1 + list(counterfactual_values).index(name)),
                    samples=5000,
                ),
                "mean_factual_claim_count_delta": finite_mean(
                    counterfactual_claim_deltas[name]
                ),
                "paired_factual_claim_count_delta_ci": bootstrap_mean_ci(
                    counterfactual_claim_deltas[name],
                    seed=config.seed + 130363 * (1 + list(counterfactual_values).index(name)),
                    samples=5000,
                ),
                "skipped_no_effect": counterfactual_skipped_no_effect[name],
                "skipped_unavailable": counterfactual_skipped_unavailable[name],
            }
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
