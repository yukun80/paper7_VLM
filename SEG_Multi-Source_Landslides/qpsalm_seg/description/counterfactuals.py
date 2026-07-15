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
        # A reverse-and-roll map is a true permutation for every canvas size.
        # The previous modular map could collide when its multiplier and the
        # number of pixels were not coprime.
        flat = region_masks.flatten(-2)
        shift = max(flat.shape[-1] // 3, 1)
        return torch.roll(flat.flip(-1), shifts=shift, dims=-1).view_as(region_masks)
    if mode == "region_swap":
        raise ValueError(
            "region_swap 必须由 DescriptionTaskDataset 解析同一 parent 的真实区域；"
            "禁止跨 parent 滚动或几何翻转伪造区域"
        )
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
    donor_indices: list[int | None] = [None] * len(samples)
    parent_ids = [str(row.get("parent_sample_id") or "") for row in state.metadata]
    for index in range(len(samples)):
        if not parent_ids[index]:
            continue
        other = None
        common: list[str] = []
        for offset in range(1, len(samples)):
            candidate = (index + offset) % len(samples)
            if not parent_ids[candidate] or parent_ids[candidate] == parent_ids[index]:
                continue
            candidate_common = sorted(
                {value.instance.family for value in samples[index]}
                & {value.instance.family for value in samples[candidate]}
            )
            if candidate_common:
                other = candidate
                common = candidate_common
                break
        if other is None:
            continue
        family = common[0]
        swapped_families[index] = family
        donor_indices[index] = other
        left = next(i for i, value in enumerate(samples[index]) if value.instance.family == family)
        right = next(i for i, value in enumerate(samples[other]) if value.instance.family == family)
        samples[index][left] = state.features.samples[other][right]
    subsets = []
    for index, (sample, subset) in enumerate(zip(samples, state.active_subsets)):
        active_names = tuple(value.instance.name for value in sample)
        family = swapped_families[index]
        subsets.append(ActiveModalitySubset(
            active_names=active_names,
            dropped_names=subset.dropped_names,
            signature=(
                f"description-cross-parent:{parent_ids[donor_indices[index]]}:{family}:"
                + "+".join(active_names)
                if family is not None and donor_indices[index] is not None
                else "description-cross-parent:none"
            ),
            is_full=family is None and subset.is_full,
        ))
    visual = state.visual_evidence
    if visual is not None:
        tokens = visual.tokens.clone()
        token_mask = visual.token_mask.clone()
        for index in range(tokens.shape[0]):
            other = donor_indices[index]
            family = swapped_families[index]
            family_id = MODALITY_FAMILY_IDS.get(str(family)) if family is not None else None
            if family_id is not None and other is not None:
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
                # Never leave target-parent evidence in surplus slots.  If the
                # source view has fewer tokens, those slots become padding.
                if target_indices.numel() > count:
                    token_mask[index, target_indices[count:]] = False
        visual = replace(
            visual,
            tokens=tokens,
            token_mask=token_mask,
            token_counts=tuple(int(value.sum().item()) for value in token_mask),
        )
    return replace(
        state,
        features=MultiScaleFeatures(samples=samples, reference_hw=state.reference_hw),
        active_subsets=tuple(subsets),
        metadata=tuple({
            **row,
            "counterfactual_modality_swap": (
                {
                    "protocol": "qpsalm_cross_parent_modality_swap_v1",
                    "donor_parent_sample_id": parent_ids[donor_indices[index]],
                    "modality_family": swapped_families[index],
                }
                if donor_indices[index] is not None else None
            ),
        } for index, row in enumerate(state.metadata)),
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
