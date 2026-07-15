#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deterministic region and evidence counterfactuals for description evaluation."""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

import torch

from qpsalm_seg.schema import (
    ActiveModalitySubset,
    MODALITY_FAMILY_IDS,
    MultiScaleFeatures,
    MultisourceBackboneState,
    TaskNeutralVisualEvidence,
)


COUNTERFACTUAL_MODES = (
    "full_mask", "zero_mask", "shuffled_mask", "region_swap",
    "modality_removal", "cross_parent_modality_swap",
)


def select_backbone_state(
    state: MultisourceBackboneState,
    indices: Sequence[int],
) -> MultisourceBackboneState:
    selected = [int(value) for value in indices]
    if not selected:
        raise ValueError("backbone state selection 不能为空")
    visual = state.visual_evidence
    selected_visual = None
    if visual is not None:
        tensor_indices = torch.tensor(selected, device=visual.tokens.device, dtype=torch.long)
        selected_visual = TaskNeutralVisualEvidence(
            tokens=visual.tokens.index_select(0, tensor_indices),
            token_mask=visual.token_mask.index_select(0, tensor_indices),
            family_ids=visual.family_ids.index_select(0, tensor_indices),
            token_counts=tuple(visual.token_counts[index] for index in selected),
            view_segments=[visual.view_segments[index] for index in selected],
            cache_keys=tuple(visual.cache_keys[index] for index in selected),
            cache_format=visual.cache_format,
        )
    valid_indices = torch.tensor(selected, device=state.valid_mask.device, dtype=torch.long)
    return MultisourceBackboneState(
        features=MultiScaleFeatures(
            samples=[state.features.samples[index] for index in selected],
            reference_hw=state.features.reference_hw,
        ),
        valid_mask=state.valid_mask.index_select(0, valid_indices),
        active_subsets=tuple(state.active_subsets[index] for index in selected),
        metadata=tuple(state.metadata[index] for index in selected),
        reference_hw=state.reference_hw,
        use_full_evidence=state.use_full_evidence,
        visual_evidence=selected_visual,
    )


def counterfactual_region_masks(region_masks: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "full_mask":
        return torch.ones_like(region_masks)
    if mode == "zero_mask":
        return torch.zeros_like(region_masks)
    if mode == "shuffled_mask":
        # Keep area constant while breaking local region-to-feature alignment.
        flat = region_masks.flatten(-2)
        index = torch.arange(flat.shape[-1], device=flat.device)
        permutation = (index * 104729 + 15485863) % max(flat.shape[-1], 1)
        return flat.index_select(-1, permutation).view_as(region_masks)
    if mode == "region_swap":
        if region_masks.shape[0] > 1:
            return region_masks.roll(1, 0)
        return torch.flip(region_masks, dims=(-1,))
    raise ValueError(f"未知 region counterfactual={mode!r}")


def _remove_one_modality(state: MultisourceBackboneState) -> MultisourceBackboneState:
    samples = []
    subsets = []
    removed_families: list[str | None] = []
    for pyramids, subset in zip(state.features.samples, state.active_subsets):
        if not pyramids:
            samples.append([])
            subsets.append(subset)
            removed_families.append(None)
            continue
        remove_index = max(
            range(len(pyramids)),
            key=lambda index: (float(pyramids[index].instance.quality), pyramids[index].instance.name),
        )
        removed = pyramids[remove_index]
        kept = [value for index, value in enumerate(pyramids) if index != remove_index]
        samples.append(kept)
        active_names = tuple(value.instance.name for value in kept)
        subsets.append(ActiveModalitySubset(
            active_names=active_names,
            dropped_names=tuple(sorted(set(subset.dropped_names) | {removed.instance.name})),
            signature=f"description-remove:{removed.instance.name}",
            is_full=False,
        ))
        removed_families.append(removed.instance.family)
    visual = state.visual_evidence
    if visual is not None:
        mask = visual.token_mask.clone()
        for index, family in enumerate(removed_families):
            if family is None:
                continue
            else:
                family_id = MODALITY_FAMILY_IDS.get(family)
                if family_id is not None:
                    mask[index] &= visual.family_ids[index] != int(family_id)
        visual = replace(
            visual,
            token_mask=mask,
            token_counts=tuple(int(value.sum().item()) for value in mask),
        )
    return replace(
        state,
        features=MultiScaleFeatures(samples=samples, reference_hw=state.reference_hw),
        active_subsets=tuple(subsets),
        visual_evidence=visual,
    )


def _cross_parent_modality_swap(state: MultisourceBackboneState) -> MultisourceBackboneState:
    if len(state.features.samples) < 2:
        raise ValueError("cross-parent modality swap 至少需要两个 parent")
    samples = [list(values) for values in state.features.samples]
    swapped_families: list[str | None] = [None] * len(samples)
    for index in range(len(samples)):
        other = (index + 1) % len(samples)
        common = sorted({value.instance.family for value in samples[index]} & {
            value.instance.family for value in samples[other]
        })
        if not common:
            continue
        family = common[0]
        swapped_families[index] = family
        left = next(i for i, value in enumerate(samples[index]) if value.instance.family == family)
        right = next(i for i, value in enumerate(samples[other]) if value.instance.family == family)
        samples[index][left] = state.features.samples[other][right]
    subsets = []
    for index, (sample, subset) in enumerate(zip(samples, state.active_subsets)):
        active_names = tuple(value.instance.name for value in sample)
        subsets.append(ActiveModalitySubset(
            active_names=active_names,
            dropped_names=subset.dropped_names,
            signature=f"description-cross-parent:{index}:" + "+".join(active_names),
            is_full=False,
        ))
    visual = state.visual_evidence
    if visual is not None:
        tokens = visual.tokens.clone()
        for index in range(tokens.shape[0]):
            other = (index + 1) % tokens.shape[0]
            family = swapped_families[index]
            family_id = MODALITY_FAMILY_IDS.get(str(family)) if family is not None else None
            if family_id is not None:
                target_indices = torch.nonzero(
                    visual.token_mask[index]
                    & (visual.family_ids[index] == family_id),
                    as_tuple=False,
                ).flatten()
                source_indices = torch.nonzero(
                    visual.token_mask[other]
                    & (visual.family_ids[other] == family_id),
                    as_tuple=False,
                ).flatten()
                count = min(target_indices.numel(), source_indices.numel())
                if count:
                    tokens[index, target_indices[:count]] = visual.tokens[
                        other, source_indices[:count]
                    ]
        visual = replace(visual, tokens=tokens)
    return replace(
        state,
        features=MultiScaleFeatures(samples=samples, reference_hw=state.reference_hw),
        active_subsets=tuple(subsets),
        visual_evidence=visual,
    )


def counterfactual_backbone(
    state: MultisourceBackboneState,
    mode: str,
) -> MultisourceBackboneState:
    if mode == "modality_removal":
        return _remove_one_modality(state)
    if mode == "cross_parent_modality_swap":
        return _cross_parent_modality_swap(state)
    raise ValueError(f"未知 evidence counterfactual={mode!r}")
