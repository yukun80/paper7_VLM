#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Description evaluation with GT, fixed prediction and end-to-end region protocols."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
import time
from typing import Any, Iterable

import torch
from torch.utils.data import DataLoader
import numpy as np

from qpsalm_seg.data import MultiSourceLandslideDataset, qpsalm_collate
from qpsalm_seg.paths import resolve_project_path
from qpsalm_seg.visualize import restore_mask_to_original

from ..data.loaders import description_amp_dtype, move_description_batch
from ..protocols.io import atomic_write_json as write_json
from ..protocols.region_geometry import project_native_region_mask_to_cache
from ..protocols.config import SegDescConfig
from .counterfactuals import (
    counterfactual_backbone,
    counterfactual_region_masks,
    select_backbone_state,
)
from .cycle_localization import (
    CycleLocalizationProvider,
    cycle_region_iou,
    summarize_cycle_localization,
)
from .metrics import (
    DescriptionMetricAccumulator,
    caption_token_f1,
    bootstrap_mean_ci,
    finite_mean,
    structured_disagreement,
    unsupported_claim_counts,
)
from .targets import EndToEndTargetResolver
from ..data.datasets import bridge_region_metadata
from ..modeling.model import (
    SegmentationGroundedDescriptionModel,
    alignment_positive_mask,
    multi_positive_alignment_loss,
)
from ..protocols.output import parse_description_output
from ..data.source_binding import build_segmentation_instruction_source_binding
from .contracts import (
    DESCRIPTION_EVALUATION_PROTOCOL,
    EVALUATION_POPULATION_FIELDS,
)
from .artifacts import (
    counterfactual_input_change_audit,
    evaluation_mask_artifact_inventory,
    write_evaluation_mask_artifact,
)
from .publication import evaluation_population_sha256
from .retrieval import counterfactual_modes, same_image_retrieval, write_jsonl




def _counterfactual_parent_values(
    rows: list[dict[str, Any]], mode: str, field: str,
) -> list[float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if str(row.get("mode") or "") != mode:
            continue
        grouped[str(row["parent_sample_id"])].append(float(row[field]))
    return [sum(values) / len(values) for _parent, values in sorted(grouped.items())]


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
        self.segmentation_source_binding = (
            build_segmentation_instruction_source_binding(
                config, split, self.dataset.rows
            )
        )
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
        self,
        metadata: dict[str, Any],
        output_hw: tuple[int, int],
        *,
        return_source: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any]] | tuple[
        torch.Tensor, dict[str, Any], torch.Tensor
    ]:
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
        description_mask, source_mapping = project_native_region_mask_to_cache(
            original_mask, description_transform
        )
        description_mask = description_mask.to(device=device)
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
            "source_to_render_mapping": source_mapping,
            "original_mask_shape": list(original_mask.shape[-2:]),
        }
        if return_source:
            return description_mask.unsqueeze(0), audit, original_mask.clone()
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
            "segmentation_source_binding": self.segmentation_source_binding,
        }


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
    publish_report: bool = True,
) -> dict[str, Any]:
    model.eval()
    resolved_output = (
        resolve_project_path(output_dir) or Path(output_dir)
        if output_dir is not None else None
    )
    if resolved_output is not None:
        resolved_output.mkdir(parents=True, exist_ok=True)
    amp_dtype = description_amp_dtype(config, device)
    autocast = device.type == "cuda" and config.training.amp_dtype != "fp32"
    metric = DescriptionMetricAccumulator()
    unavailable_metric = DescriptionMetricAccumulator()
    losses: list[float] = []
    generation_rows: list[dict[str, Any]] = []
    counterfactual_rows: list[dict[str, Any]] = []
    end_to_end_rows: list[dict[str, Any]] = []
    cycle_localization_rows: list[dict[str, Any]] = []
    mask_artifacts: list[dict[str, Any]] = []
    counterfactual_values: dict[str, list[float]] = {name: [] for name in counterfactual_modes(config)}
    counterfactual_score_deltas: dict[str, list[float]] = {
        name: [] for name in counterfactual_modes(config)
    }
    counterfactual_claim_deltas: dict[str, list[float]] = {
        name: [] for name in counterfactual_modes(config)
    }
    region_embeddings: list[torch.Tensor] = []
    text_embeddings: list[torch.Tensor] = []
    retrieval_parents: list[str] = []
    retrieval_phrases: list[str] = []
    retrieval_sample_ids: list[str] = []
    generated = 0
    requested_generate = int(
        config.evaluation.max_generate_samples if max_generate_samples is None else max_generate_samples
    )
    generate_limit = len(loader.dataset) if requested_generate <= 0 else min(
        requested_generate, len(loader.dataset)
    )
    counterfactual_counts = {name: 0 for name in counterfactual_modes(config)}
    counterfactual_skipped_no_effect = {
        name: 0 for name in counterfactual_modes(config)
    }
    counterfactual_skipped_unavailable = {
        name: 0 for name in counterfactual_modes(config)
    }
    e2e = (
        EndToEndMaskProvider(model, split, config.evaluation.segmentation_mask_threshold)
        if config.evaluation.evaluation_mode == "end_to_end" else None
    )
    if e2e is not None:
        e2e.require_targets(
            bridge_region_metadata(row)
            for row in getattr(loader.dataset, "rows", [])
        )
    cycle = (
        CycleLocalizationProvider(
            model, split, config.evaluation.segmentation_mask_threshold
        )
        if int(config.evaluation.cycle_localization_samples) >= 0 else None
    )
    if cycle is not None:
        cycle.prepare(getattr(loader.dataset, "rows", []))
    cycle_target = (
        0 if cycle is None else (
            cycle.eligible_rows
            if int(config.evaluation.cycle_localization_samples) == 0
            else min(
                int(config.evaluation.cycle_localization_samples), cycle.eligible_rows
            )
        )
    )
    started = time.perf_counter()

    for batch_index, cpu_batch in enumerate(loader):
        batch = move_description_batch(cpu_batch, device)
        backbone = model.encode_description_requests(
            batch["requests"],
            include_spatial=config.training.stage not in {"mmrs_caption", "rsicap_caption"},
        )
        region_masks = batch["region_masks"]
        batch_e2e_audits: list[dict[str, Any] | None] = [None] * len(batch["metadata"])
        batch_e2e_source_masks: list[torch.Tensor | None] = [
            None
        ] * len(batch["metadata"])
        if e2e is not None:
            resolved = [
                e2e.predict(
                    row,
                    tuple(region_masks.shape[-2:]),
                    return_source=True,
                )
                for row in batch["metadata"]
            ]
            predicted = [value[0][0] for value in resolved]
            batch_e2e_audits = [value[1] for value in resolved]
            batch_e2e_source_masks = [value[2] for value in resolved]
            region_masks = torch.stack(predicted).to(device=device)

        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast):
            if config.training.stage == "dior_alignment":
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
                retrieval_sample_ids.extend(str(row["sample_id"]) for row in batch["metadata"])
            else:
                output = model.describe_from_state(
                    backbone,
                    region_masks,
                    batch["instructions"],
                    target_texts=batch["target_texts"],
                    region_valid_mask=backbone.valid_mask,
                    protocol=config.model.region_protocol,
                    structured_output=batch["structured_outputs"],
                    use_region_tokens=batch["use_region_tokens"],
                )
                if output.per_sample_loss is None:
                    raise RuntimeError("description validation 未产生 per-sample loss")
                loss = (output.per_sample_loss * batch["weights"]).sum() / batch["weights"].sum().clamp_min(1.0)
        losses.append(float(loss.detach().cpu()))

        if config.training.stage == "dior_alignment" or generate_limit <= generated:
            continue
        baseline_texts: list[str] = []
        batch_generation_count = min(region_masks.shape[0], generate_limit - generated)
        for sample_index in range(batch_generation_count):
            one_state = select_backbone_state(backbone, [sample_index])
            one_mask = region_masks[sample_index:sample_index + 1]
            sample_id = str(batch["metadata"][sample_index]["sample_id"])
            region_mask_artifact = (
                write_evaluation_mask_artifact(
                    resolved_output,
                    role="region_input",
                    sample_id=sample_id,
                    mask=one_mask,
                )
                if resolved_output is not None else None
            )
            if region_mask_artifact is not None:
                mask_artifacts.append(region_mask_artifact)
            region_source_binding = batch["metadata"][sample_index].get(
                "region_input_source_binding"
            )
            end_to_end_audit = batch_e2e_audits[sample_index]
            if end_to_end_audit is not None:
                source_artifact = (
                    write_evaluation_mask_artifact(
                        resolved_output,
                        role="end_to_end_source",
                        sample_id=sample_id,
                        mask=batch_e2e_source_masks[sample_index],
                    )
                    if resolved_output is not None else None
                )
                if source_artifact is not None:
                    mask_artifacts.append(source_artifact)
                original_binding = dict(region_source_binding or {})
                region_source_binding = {
                    **original_binding,
                    "source_mask": {
                        "kind": "evaluation_artifact",
                        "artifact": source_artifact,
                        "shape": list(source_artifact["shape"]),
                        "positive_pixels": int(source_artifact["positive_pixels"]),
                    } if source_artifact is not None else None,
                    "render_transform": dict(
                        end_to_end_audit["description_render_transform"]
                    ),
                    "source_to_render_mapping": dict(
                        end_to_end_audit["source_to_render_mapping"]
                    ),
                }
                end_to_end_audit = {
                    **end_to_end_audit,
                    "region_input_mask_artifact": region_mask_artifact,
                    "region_input_source_binding": region_source_binding,
                }
                end_to_end_rows.append(end_to_end_audit)
            structured = bool(batch["structured_outputs"][sample_index])
            region_enabled = bool(batch["use_region_tokens"][sample_index])
            generation = model.generate_from_state_with_audit(
                one_state,
                one_mask,
                batch["instructions"][sample_index],
                max_new_tokens=config.evaluation.max_new_tokens,
                protocol=config.model.region_protocol,
                structured_output=structured,
                use_region_tokens=region_enabled,
            )
            raw = generation.text
            baseline_texts.append(raw)
            details = metric.update(
                prediction=raw,
                target_text=batch["target_texts"][sample_index],
                references=batch["reference_texts"][sample_index],
                structured=structured,
                metadata=batch["metadata"][sample_index],
            )
            if bool(batch["metadata"][sample_index].get("has_unavailable_modality")):
                unavailable_metric.update(
                    prediction=raw,
                    target_text=batch["target_texts"][sample_index],
                    references=batch["reference_texts"][sample_index],
                    structured=structured,
                    metadata=batch["metadata"][sample_index],
                )
            generation_rows.append({
                **batch["metadata"][sample_index],
                "end_to_end_segmentation_target": end_to_end_audit,
                "region_input_mask_artifact": region_mask_artifact,
                "region_input_source_binding": region_source_binding,
                "split": split,
                "evaluation_mode": config.evaluation.evaluation_mode,
                "instruction": batch["instructions"][sample_index],
                "target_text": batch["target_texts"][sample_index],
                "reference_texts": list(batch["reference_texts"][sample_index]),
                "structured_output": structured,
                "use_region_tokens": region_enabled,
                "raw_generation": raw,
                "generation_audit": generation.audit,
                "raw_metrics": details,
                "region_area_fraction": float(one_mask.float().mean().cpu()),
            })
            if (
                cycle is not None
                and len(cycle_localization_rows) < cycle_target
                and cycle.eligible(str(batch["metadata"][sample_index]["sample_id"]))
            ):
                if not str(raw).strip():
                    cycle.runtime_skip_counts["empty_raw_generation"] += 1
                else:
                    cycle_mask, cycle_audit, cycle_source_mask = cycle.localize(
                        batch["metadata"][sample_index],
                        raw,
                        tuple(one_mask.shape[-2:]),
                        return_source=True,
                    )
                    cycle_metrics = cycle_region_iou(
                        cycle_mask,
                        one_mask[0],
                        one_state.valid_mask[0],
                    )
                    valid = one_state.valid_mask[0].detach().bool()
                    effective_prediction = cycle_mask.detach().bool() & valid
                    effective_target = one_mask[0].detach().bool() & valid
                    prediction_artifact = (
                        write_evaluation_mask_artifact(
                            resolved_output,
                            role="cycle_prediction",
                            sample_id=sample_id,
                            mask=effective_prediction,
                        )
                        if resolved_output is not None else None
                    )
                    target_artifact = (
                        write_evaluation_mask_artifact(
                            resolved_output,
                            role="cycle_target",
                            sample_id=sample_id,
                            mask=effective_target,
                        )
                        if resolved_output is not None else None
                    )
                    source_artifact = (
                        write_evaluation_mask_artifact(
                            resolved_output,
                            role="cycle_source",
                            sample_id=sample_id,
                            mask=cycle_source_mask,
                        )
                        if resolved_output is not None else None
                    )
                    valid_artifact = (
                        write_evaluation_mask_artifact(
                            resolved_output,
                            role="cycle_valid",
                            sample_id=sample_id,
                            mask=valid,
                        )
                        if resolved_output is not None else None
                    )
                    if all(value is not None for value in (
                        prediction_artifact,
                        target_artifact,
                        source_artifact,
                        valid_artifact,
                    )):
                        mask_artifacts.extend((
                            prediction_artifact,
                            target_artifact,
                            source_artifact,
                            valid_artifact,
                        ))
                    cycle_localization_rows.append({
                        **batch["metadata"][sample_index],
                        "split": split,
                        "evaluation_mode": config.evaluation.evaluation_mode,
                        "region_protocol": config.model.region_protocol,
                        "cycle_audit": cycle_audit,
                        "prediction_mask_artifact": prediction_artifact,
                        "target_mask_artifact": target_artifact,
                        "source_mask_artifact": source_artifact,
                        "valid_mask_artifact": valid_artifact,
                        **cycle_metrics,
                    })
            generated += 1

        if not run_counterfactuals or all(
            value >= int(config.evaluation.counterfactual_samples)
            for value in counterfactual_counts.values()
        ):
            continue
        for mode in counterfactual_modes(config):
            if counterfactual_counts[mode] >= int(config.evaluation.counterfactual_samples):
                continue
            counterfactual_inputs: list[dict[str, Any] | None] = [
                None for _ in batch["metadata"]
            ]
            counterfactual_unavailable = [False for _ in batch["metadata"]]
            try:
                if mode in {"region_swap", "cross_parent_region_swap"}:
                    cf_backbone = backbone
                    cf_masks = region_masks.clone()
                    resolver_name = (
                        "same_parent_region_swap"
                        if mode == "region_swap" else "cross_parent_region_swap"
                    )
                    resolver = getattr(loader.dataset, resolver_name, None)
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
                if counterfactual_counts[mode] >= int(config.evaluation.counterfactual_samples):
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
                elif mode in {
                    "full_mask", "zero_mask", "shuffled_mask", "region_swap",
                    "cross_parent_region_swap",
                }:
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
                baseline_state = select_backbone_state(backbone, [sample_index])
                input_change_audit = counterfactual_input_change_audit(
                    mode=mode,
                    baseline_state=baseline_state,
                    counterfactual_state=one_state,
                    baseline_mask=region_masks[
                        sample_index:sample_index + 1
                    ],
                    counterfactual_mask=cf_masks[
                        sample_index:sample_index + 1
                    ],
                )
                if input_change_audit["changed"] is not True:
                    counterfactual_skipped_no_effect[mode] += 1
                    continue
                structured = bool(batch["structured_outputs"][sample_index])
                region_enabled = bool(batch["use_region_tokens"][sample_index])
                changed = model.generate_from_state(
                    one_state,
                    cf_masks[sample_index:sample_index + 1],
                    batch["instructions"][sample_index],
                    max_new_tokens=config.evaluation.max_new_tokens,
                    protocol=config.model.region_protocol,
                    structured_output=structured,
                    use_region_tokens=region_enabled,
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
                    "input_change_audit": input_change_audit,
                    "baseline_generation": baseline,
                    "counterfactual_generation": changed,
                    "sensitivity": sensitivity,
                    "baseline_target_score": baseline_score,
                    "counterfactual_target_score": changed_score,
                    "target_score_delta": score_delta,
                    "factual_claim_count_delta": claim_delta,
                })
                counterfactual_counts[mode] += 1

    population_sha256 = evaluation_population_sha256(generation_rows)
    report = {
        "protocol": DESCRIPTION_EVALUATION_PROTOCOL,
        "stage": config.training.stage,
        "split": split,
        "evaluation_mode": config.evaluation.evaluation_mode,
        "region_protocol": config.model.region_protocol,
        "expert_gate_audit": getattr(loader.dataset, "expert_gate_audit", None),
        "predicted_index_audit": getattr(loader.dataset, "predicted_index_audit", None),
        "source_filter_audit": getattr(loader.dataset, "source_filter_audit", None),
        "region_source_filter_audit": getattr(
            loader.dataset, "region_source_filter_audit", None
        ),
        "num_samples": len(loader.dataset),
        "num_generated": generated,
        "evaluation_limit_audit": {
            "protocol": "qpsalm_description_evaluation_limit_v1",
            "requested_max_samples": int(config.data.max_val_samples),
            "full_population_requested": int(config.data.max_val_samples) == 0,
            "dataset_rows_evaluated": len(loader.dataset),
        },
        "generation_coverage": {
            "requested": requested_generate,
            "eligible_samples": len(loader.dataset),
            "generated_samples": generated,
            "fraction": generated / max(len(loader.dataset), 1),
            "complete": generated == len(loader.dataset),
            "population_sha256": population_sha256,
            "population_identity_fields": list(EVALUATION_POPULATION_FIELDS),
        },
        "evaluation_mask_artifacts": evaluation_mask_artifact_inventory(
            mask_artifacts,
            materialized=resolved_output is not None,
        ),
        "statistics_protocol": {
            "aggregation_unit": "parent",
            "confidence": 0.95,
            "bootstrap_samples": 10000,
            "runtime_seed": int(config.training.seed),
            "formal_gate_recomputes_with_frozen_pilot_seed": True,
        },
        "mean_teacher_forced_loss": finite_mean(losses),
        "generation_metrics": metric.compute(),
        "unavailable_modality_generation_metrics": unavailable_metric.compute(),
        "primary_score_bootstrap_ci": bootstrap_mean_ci(
            [
                float((row.get("raw_metrics") or {}).get("raw_field_accuracy"))
                if (row.get("raw_metrics") or {}).get("raw_field_accuracy") is not None
                else float((row.get("raw_metrics") or {}).get("caption_token_f1", 0.0))
                for row in generation_rows
            ],
            seed=config.training.seed + 7919,
        ),
        "same_image_retrieval": same_image_retrieval(
            region_embeddings,
            text_embeddings,
            retrieval_parents,
            retrieval_phrases,
            retrieval_sample_ids,
        ),
        "counterfactual_sensitivity": {
            name: {
                "requested": int(config.evaluation.counterfactual_samples),
                "n": len(values),
                "num_effective_parents": len(_counterfactual_parent_values(
                    counterfactual_rows, name, "target_score_delta"
                )),
                "aggregation_unit": "parent",
                "coverage_complete": (
                    int(config.evaluation.counterfactual_samples) > 0
                    and len(values) >= int(config.evaluation.counterfactual_samples)
                ),
                "mean_disagreement": finite_mean(_counterfactual_parent_values(
                    counterfactual_rows, name, "sensitivity"
                )),
                "mean_target_score_delta": finite_mean(_counterfactual_parent_values(
                    counterfactual_rows, name, "target_score_delta"
                )),
                "paired_target_score_delta_ci": bootstrap_mean_ci(
                    _counterfactual_parent_values(
                        counterfactual_rows, name, "target_score_delta"
                    ),
                    seed=config.training.seed + 104729 * (1 + list(counterfactual_values).index(name)),
                    samples=10000,
                ),
                "mean_factual_claim_count_delta": finite_mean(_counterfactual_parent_values(
                    counterfactual_rows, name, "factual_claim_count_delta"
                )),
                "paired_factual_claim_count_delta_ci": bootstrap_mean_ci(
                    _counterfactual_parent_values(
                        counterfactual_rows, name, "factual_claim_count_delta"
                    ),
                    seed=config.training.seed + 130363 * (1 + list(counterfactual_values).index(name)),
                    samples=10000,
                ),
                "skipped_no_effect": counterfactual_skipped_no_effect[name],
                "skipped_unavailable": counterfactual_skipped_unavailable[name],
            }
            for name, values in counterfactual_values.items()
        },
        "end_to_end_coverage": e2e.summary(loader.dataset) if e2e is not None else None,
        "cycle_localization": (
            summarize_cycle_localization(
                cycle_localization_rows,
                cycle,
                requested=int(config.evaluation.cycle_localization_samples),
                seed=int(config.training.seed) + 524287,
            )
            if cycle is not None else None
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }
    if resolved_output is not None:
        resolved = resolved_output
        write_jsonl(resolved / "raw_generations.jsonl", generation_rows)
        write_jsonl(resolved / "counterfactual_generations.jsonl", counterfactual_rows)
        if e2e is not None:
            write_jsonl(resolved / "end_to_end_target_audit.jsonl", end_to_end_rows)
        if cycle is not None:
            write_jsonl(
                resolved / "cycle_localization.jsonl", cycle_localization_rows
            )
        if publish_report:
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
