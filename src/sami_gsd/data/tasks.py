"""T1--T4 task-view expansion after parent splits are frozen."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from sami_gsd.contracts.canonical import (
    ArtifactRef,
    CanonicalParentV3,
    RegionGeometry,
    TaskViewV3,
)
from sami_gsd.utilities.artifacts import canonical_json_bytes, sha256_bytes


TASK_EXPANSION_VERSION = "sami_task_expansion_v1_t1_t4"


class TaskExpansionError(ValueError):
    """Raised when task views would precede or violate their parent contract."""


@dataclass(frozen=True)
class RegionAnswer:
    """Reviewed region-understanding target bound to a parent region."""

    answer_ref: ArtifactRef
    annotation_origin: Literal["human", "expert"]


@dataclass(frozen=True)
class FixedRegionPrediction:
    """OOF fixed predicted mask used by a T4 region-understanding view."""

    mask_ref: ArtifactRef
    bbox_half_open: tuple[int, int, int, int]
    checkpoint_fingerprint: str


@dataclass(frozen=True)
class TaskExpansion:
    """Stable task rows and modality-axis evaluation conditions."""

    tasks: tuple[TaskViewV3, ...]
    evaluation_conditions: dict[str, Any]
    aggregate_sha256: str


def _task_id(parent_id: str, task_type: str, region_id: str | None) -> str:
    """Derive one stable task ID without source names in instructions."""

    digest = sha256_bytes(
        canonical_json_bytes(
            {"parent_id": parent_id, "task_type": task_type, "region_id": region_id}
        )
    )
    return f"task-{digest[:24]}"


def _evaluation_conditions(parents: tuple[CanonicalParentV3, ...]) -> dict[str, Any]:
    """Publish modality combinations as conditions, never duplicated task rows."""

    rows: list[dict[str, Any]] = []
    for parent in sorted(parents, key=lambda item: item.parent_id):
        modality_ids = tuple(sorted(modality.modality_id for modality in parent.modalities))
        reference = parent.reference_canvas.reference_modality_id
        subsets = {modality_ids, (reference,)}
        for modality_id in modality_ids:
            retained = tuple(value for value in modality_ids if value != modality_id)
            if reference in retained:
                subsets.add(retained)
        rows.append(
            {
                "parent_id": parent.parent_id,
                "split": parent.split,
                "active_modality_subsets": [list(values) for values in sorted(subsets)],
            }
        )
    return {
        "schema_version": "sami_evaluation_conditions_v1",
        "task_rows_are_not_duplicated_by_modality": True,
        "mask_modes": ["gt_mask", "fixed_prediction", "end_to_end"],
        "parents": rows,
    }


def expand_task_views(
    parents: tuple[CanonicalParentV3, ...],
    *,
    region_answers: dict[tuple[str, str], RegionAnswer] | None = None,
    fixed_predictions: dict[tuple[str, str], FixedRegionPrediction] | None = None,
) -> TaskExpansion:
    """Expand T1/T2 and eligible T3/T4 views in stable parent/region order.

    T3 requires a reviewed region answer. T4 additionally requires an OOF
    fixed prediction. End-to-end is retained as an evaluation condition and
    does not fabricate a precomputed input-mask artifact.
    """

    ordered = tuple(sorted(parents, key=lambda item: item.parent_id))
    if not ordered or any(parent.split == "audit" for parent in ordered):
        raise TaskExpansionError("task expansion requires non-empty, frozen parent splits")
    if len({parent.parent_id for parent in ordered}) != len(ordered):
        raise TaskExpansionError("task expansion requires unique parent IDs")
    answers = region_answers or {}
    predictions = fixed_predictions or {}
    valid_region_keys = {
        (parent.parent_id, region.region_id)
        for parent in ordered
        for region in parent.annotations.referring_regions
    }
    if not set(answers).issubset(valid_region_keys):
        raise TaskExpansionError("region answer references an unknown parent region")
    if not set(predictions).issubset(valid_region_keys):
        raise TaskExpansionError("fixed prediction references an unknown parent region")
    if not set(predictions).issubset(answers):
        raise TaskExpansionError("T4 fixed predictions require a matching reviewed answer")

    tasks: list[TaskViewV3] = []
    for parent in ordered:
        mask_ref = parent.annotations.global_landslide_mask
        if mask_ref is not None and parent.annotations.global_target_status in {"positive", "no_target"}:
            no_target = parent.annotations.global_target_status == "no_target"
            instruction = (
                "Segment all landslide regions. If no landslide is present, return an empty mask."
                if no_target
                else "Segment all landslide regions."
            )
            origin = (
                "derived_no_target"
                if no_target
                else parent.annotations.global_mask_origin
            )
            if origin is None:
                raise TaskExpansionError("global task lost its mask annotation origin")
            tasks.append(
                TaskViewV3(
                    task_id=_task_id(parent.parent_id, "t1_global", None),
                    parent_id=parent.parent_id,
                    task_type="t1_global",
                    instruction=instruction,
                    target_status=parent.annotations.global_target_status,
                    region_geometry=None,
                    target_mask_ref=mask_ref,
                    target_box_ref=None,
                    answer_ref=None,
                    annotation_origin=origin,
                    weight=1.0,
                )
            )
        for region in sorted(parent.annotations.referring_regions, key=lambda item: item.region_id):
            key = (parent.parent_id, region.region_id)
            geometry = RegionGeometry(
                coordinate_space="reference_pixel_half_open",
                region_id=region.region_id,
                bbox_half_open=region.bbox_half_open,
            )
            tasks.append(
                TaskViewV3(
                    task_id=_task_id(parent.parent_id, "t2_referring", region.region_id),
                    parent_id=parent.parent_id,
                    task_type="t2_referring",
                    instruction=f"Segment {region.expression.rstrip('.')}.",
                    target_status="positive",
                    region_geometry=geometry,
                    target_mask_ref=region.mask_ref,
                    target_box_ref=region.bbox_half_open,
                    answer_ref=None,
                    annotation_origin=region.annotation_origin,
                    weight=1.0,
                )
            )
            if key not in answers:
                continue
            answer = answers[key]
            tasks.append(
                TaskViewV3(
                    task_id=_task_id(parent.parent_id, "t3_gt_region", region.region_id),
                    parent_id=parent.parent_id,
                    task_type="t3_gt_region",
                    instruction="Describe the evidence inside the supplied landslide-region mask.",
                    target_status="positive",
                    region_geometry=geometry,
                    target_mask_ref=region.mask_ref,
                    target_box_ref=region.bbox_half_open,
                    answer_ref=answer.answer_ref,
                    annotation_origin=answer.annotation_origin,
                    weight=1.0,
                )
            )
            if key in predictions:
                prediction = predictions[key]
                predicted_geometry = RegionGeometry(
                    coordinate_space="reference_pixel_half_open",
                    region_id=region.region_id,
                    bbox_half_open=prediction.bbox_half_open,
                )
                tasks.append(
                    TaskViewV3(
                        task_id=_task_id(parent.parent_id, "t4_predicted_region", region.region_id),
                        parent_id=parent.parent_id,
                        task_type="t4_predicted_region",
                        instruction="Describe the evidence inside the supplied fixed predicted region mask.",
                        target_status="positive",
                        region_geometry=predicted_geometry,
                        target_mask_ref=prediction.mask_ref,
                        target_box_ref=prediction.bbox_half_open,
                        answer_ref=answer.answer_ref,
                        annotation_origin="oof_prediction",
                        weight=1.0,
                    )
                )
    tasks_tuple = tuple(sorted(tasks, key=lambda item: item.task_id))
    conditions = _evaluation_conditions(ordered)
    payload = {
        "version": TASK_EXPANSION_VERSION,
        "tasks": [task.model_dump(mode="json") for task in tasks_tuple],
        "evaluation_conditions": conditions,
    }
    return TaskExpansion(
        tasks=tasks_tuple,
        evaluation_conditions=conditions,
        aggregate_sha256=sha256_bytes(canonical_json_bytes(payload)),
    )


__all__ = [
    "TASK_EXPANSION_VERSION",
    "FixedRegionPrediction",
    "RegionAnswer",
    "TaskExpansion",
    "TaskExpansionError",
    "expand_task_views",
]
