"""Deterministic reference-canvas selection for P1.2.

No filesystem or raster reads occur here.  Source-specific scanners must first
project their evidence into :class:`ReferenceCanvasCandidate` records.
"""

from __future__ import annotations

from collections.abc import Sequence

from sami_gsd.contracts.spatial import ReferenceCanvasCandidate, ReferenceCanvasDecision
from sami_gsd.utilities.artifacts import canonical_json_bytes, sha256_bytes


class ReferenceSelectionError(ValueError):
    """Raised when the frozen rules cannot resolve a reference grid safely."""


def _candidate_fingerprint(candidates: Sequence[ReferenceCanvasCandidate]) -> str:
    """Hash a stable projection of every considered candidate."""

    payload = [candidate.model_dump(mode="json") for candidate in candidates]
    return sha256_bytes(canonical_json_bytes(payload))


def _select_mask_candidate(
    candidates: Sequence[ReferenceCanvasCandidate],
    *,
    rule: str,
) -> ReferenceCanvasCandidate:
    """Choose complete coverage and finest GSD, with an explicit lexical tie-break."""

    if len(candidates) == 1:
        return candidates[0]
    complete = [candidate for candidate in candidates if candidate.valid_coverage == 1.0]
    if not complete:
        raise ReferenceSelectionError(f"{rule} has multiple grids but none has complete valid coverage")
    if any(candidate.native_gsd_m is None for candidate in complete):
        raise ReferenceSelectionError(f"{rule} has multiple complete grids without fully comparable native GSD")
    return min(complete, key=lambda candidate: (candidate.native_gsd_m, candidate.modality_id))


def select_reference_canvas(
    candidates: Sequence[ReferenceCanvasCandidate],
) -> ReferenceCanvasDecision:
    """Apply the frozen reference-canvas priorities without randomness.

    Priority is authoritative native mask, then registered mask with complete
    coverage and finest GSD, then a sole language image.  Calling order never
    affects the decision.  A selected grid without a coordinate inverse is
    retained for audit/global-language use but explicitly excluded from T1--T4.

    Args:
        candidates: Source-grid metadata.  No raster bytes are accepted.

    Returns:
        A frozen decision containing the input-set fingerprint and eligibility.

    Raises:
        ReferenceSelectionError: The candidate set is empty, duplicated or not
            uniquely resolvable under the frozen rules.
    """

    ordered = tuple(sorted(candidates, key=lambda candidate: candidate.modality_id))
    if not ordered:
        raise ReferenceSelectionError("reference selection requires at least one candidate")
    modality_ids = tuple(candidate.modality_id for candidate in ordered)
    if len(set(modality_ids)) != len(modality_ids):
        raise ReferenceSelectionError("reference candidate modality_id values must be unique")

    native = tuple(candidate for candidate in ordered if candidate.mask_grid == "native")
    registered = tuple(candidate for candidate in ordered if candidate.mask_grid == "registered")
    if native:
        selected = _select_mask_candidate(native, rule="authoritative_native_mask")
        selection_rule = "authoritative_native_mask"
    elif registered:
        selected = _select_mask_candidate(registered, rule="registered_mask_complete_finest_gsd")
        selection_rule = "registered_mask_complete_finest_gsd"
    elif len(ordered) == 1 and ordered[0].single_image_language:
        selected = ordered[0]
        selection_rule = "single_image_original"
    else:
        raise ReferenceSelectionError(
            "no authoritative/registered mask grid or sole single-image language candidate can define the reference"
        )

    eligible = selected.coordinate_inverse_available
    return ReferenceCanvasDecision(
        reference_modality_id=selected.modality_id,
        original_hw=selected.original_hw,
        selection_rule=selection_rule,
        considered_modality_ids=modality_ids,
        candidate_set_sha256=_candidate_fingerprint(ordered),
        inverse_transform_available=eligible,
        spatial_tasks_eligible=eligible,
        spatial_exclusion_reason=None if eligible else "coordinate_inverse_unavailable",
    )


def require_spatial_task_eligibility(decision: ReferenceCanvasDecision) -> None:
    """Fail closed before deriving a T1--T4 view from an ineligible parent."""

    if not decision.spatial_tasks_eligible:
        raise ReferenceSelectionError(
            f"reference modality {decision.reference_modality_id!r} has no coordinate inverse; T1--T4 are forbidden"
        )


__all__ = [
    "ReferenceSelectionError",
    "require_spatial_task_eligibility",
    "select_reference_canvas",
]
