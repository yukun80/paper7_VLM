"""Sensor-Aware Multi-Image Adapter for the P2 greenfield skeleton."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from torch import nn

from sami_gsd.contracts.model import SamiModelConfig
from sami_gsd.model.rendering import image_content_sha256, resize_to_pixel_budget, valid_mask_sha256
from sami_gsd.model.states import (
    ExcludedModality,
    LoadedParent,
    MultiImageBatch,
    PreparedParent,
    PreparedView,
    QwenBackboneState,
    SensorCard,
)


ADAPTER_REVISION = "sami_sensor_aware_multi_image_adapter_v1"


class BackboneEncoder(Protocol):
    """Minimal dependency boundary implemented by ``QwenBackboneWrapper``."""

    def encode(
        self,
        batch: MultiImageBatch,
        *,
        return_spatial_features: bool,
        cache_store: object | None = None,
    ) -> QwenBackboneState:
        """Encode one prepared batch without reading image files."""


_FAMILY_ORDER = {
    "optical": 0,
    "multispectral": 1,
    "sar": 2,
    "dem": 3,
    "slope": 3,
    "insar": 4,
    "deformation": 4,
    "other": 5,
}


class SensorAwareMultiImageAdapter(nn.Module):
    """Prepare deterministic reference/support views and delegate official encoding.

    The adapter has no learned reliability, mask query, proposal, segmentation or
    description behavior. It consumes already decoded CPU views so model forward
    remains free of filesystem and NumPy I/O.
    """

    def __init__(self, config: SamiModelConfig, backbone: BackboneEncoder | None = None) -> None:
        super().__init__()
        self.config = config
        self.backbone = backbone

    def prepare(
        self,
        parents: Sequence[LoadedParent],
        active_modalities: Sequence[Sequence[str]],
    ) -> MultiImageBatch:
        """Build a stable variable-cardinality native multi-image batch.

        Args:
            parents: Canonical parents with decoded effective CPU RGB views.
            active_modalities: One modality-id subset per parent. Caller order is
                ignored; identity order is derived from the frozen protocol.

        Returns:
            ``MultiImageBatch`` with reference-first stable views.

        Raises:
            ValueError: Parent/subset counts, identities, reference validity or
                pixel/view budgets violate the contract.
        """

        if not parents:
            raise ValueError("at least one parent is required")
        if len(parents) != len(active_modalities):
            raise ValueError("active_modalities must contain exactly one subset per parent")
        profile = self.config.active_pixel_budget()
        prepared_parents: list[PreparedParent] = []
        seen_parent_ids: set[str] = set()
        for loaded_parent, requested_sequence in zip(parents, active_modalities, strict=True):
            loaded_parent.validate()
            record = loaded_parent.record
            if record.parent_id in seen_parent_ids:
                raise ValueError(f"duplicate parent_id in model batch: {record.parent_id}")
            seen_parent_ids.add(record.parent_id)
            if len(requested_sequence) != len(set(requested_sequence)):
                raise ValueError(f"active subset for {record.parent_id} contains duplicate modality ids")
            requested = set(requested_sequence)
            declared = {modality.modality_id: modality for modality in record.modalities}
            unknown = requested - set(declared)
            if unknown:
                raise ValueError(f"active subset for {record.parent_id} contains undeclared ids: {sorted(unknown)}")

            reference_id = record.reference_canvas.reference_modality_id
            if reference_id not in requested:
                raise ValueError(f"active subset for {record.parent_id} must retain canonical reference {reference_id}")
            excluded: list[ExcludedModality] = []
            effective_ids: list[str] = []
            for modality_id in sorted(declared):
                modality = declared[modality_id]
                if modality_id not in requested:
                    excluded.append(
                        ExcludedModality(modality_id, modality.availability_status, "inactive_dropout")
                    )
                elif modality.availability_status == "missing":
                    excluded.append(ExcludedModality(modality_id, modality.availability_status, "missing"))
                elif modality.availability_status == "present_zero_valid":
                    excluded.append(
                        ExcludedModality(modality_id, modality.availability_status, "present_zero_valid")
                    )
                else:
                    effective_ids.append(modality_id)
            if reference_id not in effective_ids:
                raise ValueError(f"canonical reference {reference_id} is missing or zero-valid")
            effective_ids.sort(
                key=lambda modality_id: (
                    0 if modality_id == reference_id else 1,
                    _FAMILY_ORDER[declared[modality_id].family],
                    modality_id,
                )
            )
            if len(effective_ids) > profile.max_views:
                raise ValueError(
                    f"parent {record.parent_id} has {len(effective_ids)} effective views, "
                    f"exceeding Profile {profile.profile} max_views={profile.max_views}"
                )

            prepared_views: list[PreparedView] = []
            for modality_id in effective_ids:
                try:
                    loaded_view = loaded_parent.views[modality_id]
                except KeyError as error:
                    raise ValueError(f"effective view {modality_id} was not decoded before model prepare") from error
                role = "reference" if modality_id == reference_id else "support"
                pixel_budget = (
                    profile.reference_max_pixels if role == "reference" else profile.support_max_pixels
                )
                source_hw = (loaded_view.image.height, loaded_view.image.width)
                image, valid_mask = resize_to_pixel_budget(
                    loaded_view.image,
                    loaded_view.valid_mask,
                    pixel_budget,
                )
                prepared_views.append(
                    PreparedView(
                        parent_id=record.parent_id,
                        modality=loaded_view.modality,
                        role=role,
                        sensor_card=SensorCard.from_modality(loaded_view.modality),
                        image=image,
                        valid_mask=valid_mask,
                        source_hw=source_hw,
                        rendered_hw=(image.height, image.width),
                        pixel_budget=pixel_budget,
                        image_sha256=image_content_sha256(image),
                        valid_sha256=valid_mask_sha256(valid_mask),
                    )
                )
            prepared_parents.append(
                PreparedParent(
                    parent_id=record.parent_id,
                    canonical_reference_view_id=reference_id,
                    active_modality_ids=tuple(sorted(requested)),
                    views=tuple(prepared_views),
                    excluded_modalities=tuple(excluded),
                )
            )
        return MultiImageBatch(
            schema_version="sami_multi_image_batch_v1",
            profile=profile,
            parents=tuple(prepared_parents),
        )

    def encode(
        self,
        batch: MultiImageBatch,
        *,
        return_spatial_features: bool,
        cache_store: object | None = None,
    ) -> QwenBackboneState:
        """Run the configured official backbone or fail explicitly."""

        if self.backbone is None:
            raise RuntimeError("SensorAwareMultiImageAdapter requires an explicit QwenBackboneWrapper")
        return self.backbone.encode(
            batch,
            return_spatial_features=return_spatial_features,
            cache_store=cache_store,
        )


__all__ = ["ADAPTER_REVISION", "BackboneEncoder", "SensorAwareMultiImageAdapter"]
