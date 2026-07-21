"""Typed, file-I/O-free P2 model inputs and Qwen backbone states."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal

import torch
from PIL import Image
from torch import Tensor

from sami_gsd.contracts.canonical import CanonicalParentV3, ModalityRecord
from sami_gsd.contracts.model import PixelBudgetProfile


AvailabilityReason = Literal["inactive_dropout", "missing", "present_zero_valid"]


@dataclass(frozen=True, slots=True)
class SensorCard:
    """Compact scientific metadata placed before exactly one effective view.

    Dataset identity, split, normalization, labels and target geometry are
    intentionally absent from this public contract.
    """

    view_id: str
    family: str
    sensor: str
    product_type: str
    bands_or_polarizations: tuple[str, ...]
    orbit: str | None
    gsd_m: float | None
    units: str | None
    sign_convention: str | None
    valid_coverage: float
    quality_status: str
    quality_flags: tuple[str, ...]

    @classmethod
    def from_modality(cls, modality: ModalityRecord) -> "SensorCard":
        """Build the frozen allowed-field projection of one modality record."""

        return cls(
            view_id=modality.modality_id,
            family=modality.family,
            sensor=modality.sensor,
            product_type=modality.product_type,
            bands_or_polarizations=tuple(modality.band_names),
            orbit=modality.orbit,
            gsd_m=modality.aligned_gsd_m or modality.native_gsd_m,
            units=modality.units,
            sign_convention=modality.sign_convention,
            valid_coverage=float(modality.valid_coverage),
            quality_status=modality.quality.status,
            quality_flags=tuple(modality.quality.flags),
        )

    def payload(self) -> dict[str, Any]:
        """Return the stable prompt/cache projection in a fixed field order."""

        return {
            "view_id": self.view_id,
            "family": self.family,
            "sensor": self.sensor,
            "product_type": self.product_type,
            "bands_or_polarizations": list(self.bands_or_polarizations),
            "orbit": self.orbit,
            "gsd_m": self.gsd_m,
            "units": self.units,
            "sign_convention": self.sign_convention,
            "valid_coverage": self.valid_coverage,
            "quality_status": self.quality_status,
            "quality_flags": list(self.quality_flags),
        }


@dataclass(frozen=True, slots=True)
class LoadedView:
    """One decoded modality before model-side ordering and pixel budgeting.

    ``image`` is RGB and ``valid_mask`` has shape ``[H, W]``. Both remain on
    CPU; no file access occurs in the adapter or backbone forward.
    """

    modality: ModalityRecord
    image: Image.Image
    valid_mask: Tensor
    image_sha256: str
    valid_sha256: str

    def validate(self) -> None:
        """Validate shape, dtype and finite/valid semantics."""

        if self.image.mode != "RGB":
            raise ValueError(f"view {self.modality.modality_id} image must be RGB")
        width, height = self.image.size
        if tuple(self.valid_mask.shape) != (height, width):
            raise ValueError(f"view {self.modality.modality_id} valid mask does not match image size")
        if self.valid_mask.dtype is not torch.bool:
            raise TypeError(f"view {self.modality.modality_id} valid mask must have dtype bool")
        if self.valid_mask.device.type != "cpu":
            raise ValueError(f"view {self.modality.modality_id} must be prepared on CPU")
        if not bool(self.valid_mask.any()):
            raise ValueError(f"view {self.modality.modality_id} has zero valid pixels")


@dataclass(frozen=True, slots=True)
class LoadedParent:
    """One canonical record plus decoded present, non-zero-valid views."""

    record: CanonicalParentV3
    views: dict[str, LoadedView]

    def validate(self) -> None:
        """Ensure decoded views are an exact subset of declared usable modalities."""

        declared = {modality.modality_id: modality for modality in self.record.modalities}
        unknown = set(self.views) - set(declared)
        if unknown:
            raise ValueError(f"loaded parent contains undeclared modalities: {sorted(unknown)}")
        for modality_id, view in self.views.items():
            if view.modality != declared[modality_id]:
                raise ValueError(f"loaded view {modality_id} does not match canonical modality metadata")
            if view.modality.availability_status not in {"present_partial_valid", "present_valid"}:
                raise ValueError(f"loaded view {modality_id} must be present with non-zero valid coverage")
            view.validate()


@dataclass(frozen=True, slots=True)
class ExcludedModality:
    """An unavailable or dropped modality retained only as explicit state."""

    modality_id: str
    availability_status: str
    reason: AvailabilityReason


@dataclass(frozen=True, slots=True)
class PreparedView:
    """One effective, ordered and pixel-budgeted Qwen image view."""

    parent_id: str
    modality: ModalityRecord
    role: Literal["reference", "support"]
    sensor_card: SensorCard
    image: Image.Image
    valid_mask: Tensor
    source_hw: tuple[int, int]
    rendered_hw: tuple[int, int]
    pixel_budget: int
    image_sha256: str
    valid_sha256: str


@dataclass(frozen=True, slots=True)
class PreparedParent:
    """One parent after active-subset filtering and stable view ordering."""

    parent_id: str
    canonical_reference_view_id: str
    active_modality_ids: tuple[str, ...]
    views: tuple[PreparedView, ...]
    excluded_modalities: tuple[ExcludedModality, ...]

    @property
    def view_ids(self) -> tuple[str, ...]:
        """Return the stable identity order used by prompt, processor and cache."""

        return tuple(view.modality.modality_id for view in self.views)


@dataclass(frozen=True, slots=True)
class MultiImageBatch:
    """File-I/O-free input for an official native Qwen3-VL forward."""

    schema_version: Literal["sami_multi_image_batch_v1"]
    profile: PixelBudgetProfile
    parents: tuple[PreparedParent, ...]

    @property
    def flat_views(self) -> tuple[PreparedView, ...]:
        """Flatten views in parent order without changing per-parent identity."""

        return tuple(view for parent in self.parents for view in parent.views)

    def identity_payload(self) -> dict[str, Any]:
        """Return every input field that can affect visual/state bytes."""

        return {
            "schema_version": self.schema_version,
            "profile": self.profile.model_dump(mode="json"),
            "parents": [
                {
                    "parent_id": parent.parent_id,
                    "canonical_reference_view_id": parent.canonical_reference_view_id,
                    "active_modality_ids": list(parent.active_modality_ids),
                    "view_order": list(parent.view_ids),
                    "excluded_modalities": [
                        {
                            "modality_id": excluded.modality_id,
                            "availability_status": excluded.availability_status,
                            "reason": excluded.reason,
                        }
                        for excluded in parent.excluded_modalities
                    ],
                    "views": [
                        {
                            "view_id": view.modality.modality_id,
                            "role": view.role,
                            "sensor_card": view.sensor_card.payload(),
                            "source_hw": list(view.source_hw),
                            "rendered_hw": list(view.rendered_hw),
                            "pixel_budget": view.pixel_budget,
                            "image_sha256": view.image_sha256,
                            "valid_sha256": view.valid_sha256,
                        }
                        for view in parent.views
                    ],
                }
                for parent in self.parents
            ],
        }


@dataclass(frozen=True, slots=True)
class SpatialFeatureLevel:
    """One reconstructable per-view feature grid with shape ``[C, H, W]``."""

    level: str
    features: Tensor

    def validate(self) -> None:
        """Reject malformed or non-finite feature maps."""

        if self.features.ndim != 3:
            raise ValueError(f"spatial level {self.level} must have shape [C,H,W]")
        if not bool(torch.isfinite(self.features).all()):
            raise ValueError(f"spatial level {self.level} contains non-finite values")


@dataclass(frozen=True, slots=True)
class ViewTransform:
    """Processor-grid and canonical-transform metadata for one encoded view."""

    source_hw: tuple[int, int]
    rendered_hw: tuple[int, int]
    processor_grid_thw: tuple[int, int, int]
    merged_grid_hw: tuple[int, int]
    alignment_status: str
    source_to_reference_transform: tuple[dict[str, Any], ...] | None
    reference_to_source_transform: tuple[dict[str, Any], ...] | None


@dataclass(frozen=True, slots=True)
class ViewBackboneState:
    """Task-neutral language-aligned and spatial state for one Qwen view."""

    parent_id: str
    view_id: str
    role: Literal["reference", "support"]
    sensor_card: SensorCard
    language_aligned_visual_tokens: Tensor
    spatial_features: tuple[SpatialFeatureLevel, ...]
    valid_mask: Tensor
    transform: ViewTransform
    image_sha256: str
    valid_sha256: str

    def validate(self) -> None:
        """Validate token/grid identity and numerical safety."""

        if self.language_aligned_visual_tokens.ndim != 2:
            raise ValueError("language_aligned_visual_tokens must have shape [N,D]")
        if not bool(torch.isfinite(self.language_aligned_visual_tokens).all()):
            raise ValueError("language-aligned tokens contain non-finite values")
        expected_tokens = self.transform.merged_grid_hw[0] * self.transform.merged_grid_hw[1]
        if self.language_aligned_visual_tokens.shape[0] != expected_tokens:
            raise ValueError("language-aligned token count does not reconstruct the merged spatial grid")
        if tuple(self.valid_mask.shape) != self.transform.merged_grid_hw or self.valid_mask.dtype is not torch.bool:
            raise ValueError("view valid mask must be bool on the reconstructed merged grid")
        for level in self.spatial_features:
            level.validate()
            if tuple(level.features.shape[-2:]) != self.transform.merged_grid_hw:
                raise ValueError(f"spatial level {level.level} does not match merged grid")


@dataclass(frozen=True, slots=True)
class QwenBackboneState:
    """Task-neutral official-forward state consumed by later P3/P6 modules."""

    schema_version: Literal["sami_qwen_backbone_state_v1"]
    parent_ids: tuple[str, ...]
    view_order: tuple[tuple[str, ...], ...]
    reference_view_ids: tuple[str, ...]
    active_modality_ids: tuple[tuple[str, ...], ...]
    excluded_modalities: tuple[tuple[ExcludedModality, ...], ...]
    views: tuple[ViewBackboneState, ...]
    prompt_sha256: tuple[str, ...]
    model_fingerprint: str
    processor_fingerprint: str
    qwen_code_revision: str
    profile: str
    dtype: str
    cache_key: str
    from_cache: bool

    def validate(self) -> None:
        """Validate parent/view partition, ordering and every tensor contract."""

        parent_count = len(self.parent_ids)
        if parent_count == 0 or len(set(self.parent_ids)) != parent_count:
            raise ValueError("QwenBackboneState requires unique non-empty parent identities")
        parallel = (
            self.view_order,
            self.reference_view_ids,
            self.active_modality_ids,
            self.excluded_modalities,
            self.prompt_sha256,
        )
        if any(len(field) != parent_count for field in parallel):
            raise ValueError("QwenBackboneState parent-level fields have inconsistent lengths")
        expected_flat = tuple(view_id for order in self.view_order for view_id in order)
        if tuple(view.view_id for view in self.views) != expected_flat:
            raise ValueError("QwenBackboneState view tensor order does not match view_order")
        if len(self.views) != len({(view.parent_id, view.view_id) for view in self.views}):
            raise ValueError("QwenBackboneState contains duplicate parent/view identities")
        offset = 0
        for parent_id, order, reference_view_id, active_ids, excluded in zip(
            self.parent_ids,
            self.view_order,
            self.reference_view_ids,
            self.active_modality_ids,
            self.excluded_modalities,
            strict=True,
        ):
            if not order or order[0] != reference_view_id:
                raise ValueError(f"parent {parent_id} reference view must be first")
            if len(order) != len(set(order)):
                raise ValueError(f"parent {parent_id} contains duplicate view identities")
            if not set(order).issubset(active_ids):
                raise ValueError(f"parent {parent_id} effective views are not contained in active subset")
            excluded_ids = {item.modality_id for item in excluded}
            if set(order) & excluded_ids:
                raise ValueError(f"parent {parent_id} excluded modality leaked into visual state")
            parent_views = self.views[offset : offset + len(order)]
            if any(view.parent_id != parent_id for view in parent_views):
                raise ValueError(f"parent {parent_id} view partition contains another parent identity")
            offset += len(order)
        for view in self.views:
            view.validate()

    def metadata_payload(self) -> dict[str, Any]:
        """Return exact non-tensor metadata used by cache equivalence checks."""

        return {
            "schema_version": self.schema_version,
            "parent_ids": list(self.parent_ids),
            "view_order": [list(order) for order in self.view_order],
            "reference_view_ids": list(self.reference_view_ids),
            "active_modality_ids": [list(ids) for ids in self.active_modality_ids],
            "excluded_modalities": [
                [
                    {
                        "modality_id": item.modality_id,
                        "availability_status": item.availability_status,
                        "reason": item.reason,
                    }
                    for item in group
                ]
                for group in self.excluded_modalities
            ],
            "views": [
                {
                    "parent_id": view.parent_id,
                    "view_id": view.view_id,
                    "role": view.role,
                    "sensor_card": view.sensor_card.payload(),
                    "spatial_levels": [level.level for level in view.spatial_features],
                    "transform": {
                        "source_hw": list(view.transform.source_hw),
                        "rendered_hw": list(view.transform.rendered_hw),
                        "processor_grid_thw": list(view.transform.processor_grid_thw),
                        "merged_grid_hw": list(view.transform.merged_grid_hw),
                        "alignment_status": view.transform.alignment_status,
                        "source_to_reference_transform": view.transform.source_to_reference_transform,
                        "reference_to_source_transform": view.transform.reference_to_source_transform,
                    },
                    "image_sha256": view.image_sha256,
                    "valid_sha256": view.valid_sha256,
                }
                for view in self.views
            ],
            "prompt_sha256": list(self.prompt_sha256),
            "model_fingerprint": self.model_fingerprint,
            "processor_fingerprint": self.processor_fingerprint,
            "qwen_code_revision": self.qwen_code_revision,
            "profile": self.profile,
            "dtype": self.dtype,
            "cache_key": self.cache_key,
        }

    def tensor_items(self) -> tuple[tuple[str, Tensor], ...]:
        """Enumerate all publishable tensors using stable role-bound names."""

        items: list[tuple[str, Tensor]] = []
        for view_index, view in enumerate(self.views):
            prefix = f"views.{view_index}.{view.parent_id}.{view.view_id}"
            items.append((f"{prefix}.language_aligned_visual_tokens", view.language_aligned_visual_tokens))
            items.append((f"{prefix}.valid_mask", view.valid_mask))
            for level_index, level in enumerate(view.spatial_features):
                items.append((f"{prefix}.spatial.{level_index}.{level.level}", level.features))
        return tuple(items)

    def as_cached(self) -> "QwenBackboneState":
        """Return the same immutable state marked as a cache hit."""

        return replace(self, from_cache=True)


__all__ = [
    "ExcludedModality",
    "LoadedParent",
    "LoadedView",
    "MultiImageBatch",
    "PreparedParent",
    "PreparedView",
    "QwenBackboneState",
    "SensorCard",
    "SpatialFeatureLevel",
    "ViewBackboneState",
    "ViewTransform",
]
