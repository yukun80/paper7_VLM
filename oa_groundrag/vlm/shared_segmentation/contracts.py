"""Explicit spatial and physical contracts for shared-model observations."""
from dataclasses import dataclass
from typing import Mapping
import torch
from torch import Tensor

LENS_REVISION = "05106853ed6222763233b4cc31de117b30adef49"
SCHEMA = "geo_groundrag.shared_segmentation.v1"
MODALITIES = ("dem", "insar_velocity", "slope")

@dataclass(frozen=True)
class AuxiliaryObservation:
    values: Tensor
    valid: Tensor
    source: str
    alignment: str
    unit: str
    unit_verified: bool = False
    sign_convention: str | None = None
    sign_verified: bool = False

    def validate(self, name: str, image_hw: tuple[int, int]) -> None:
        if name not in MODALITIES or not self.source or not self.unit:
            raise ValueError("Unknown modality or missing source/unit")
        if self.alignment != "rgb_grid_verified":
            raise ValueError("Spatial alignment must be verified, not inferred from shape")
        if self.values.ndim != 3 or self.values.shape[0] != 1:
            raise ValueError("Auxiliary values must have shape (1,H,W)")
        if tuple(self.values.shape[-2:]) != image_hw or self.valid.shape != self.values.shape:
            raise ValueError("Auxiliary observation must be aligned to the RGB grid")
        if self.valid.dtype != torch.bool:
            raise ValueError("Validity must be boolean")
        if not torch.isfinite(self.values[self.valid]).all():
            raise ValueError("Non-finite values marked valid")
        verified_units = {"dem": {"m"}, "insar_velocity": {"mm/year", "m/year"}, "slope": {"degree", "radian", "percent"}}
        if self.unit_verified and self.unit not in verified_units[name]:
            raise ValueError("Unsupported or unknown physical units cannot be marked verified")
        if self.sign_verified and (not self.unit_verified or not self.sign_convention):
            raise ValueError("Verified direction requires verified units and a convention")

    def evidence(self, name: str) -> dict:
        physical = self.unit_verified and (name != "insar_velocity" or self.sign_verified)
        return {"modality": name, "source": self.source, "unit": self.unit,
                "valid_fraction": float(self.valid.float().mean()),
                "physical_interpretation_allowed": physical,
                "sign_convention": self.sign_convention if self.sign_verified else None,
                "limitation": "" if physical else "Encoded observation only; no physical magnitude or direction claim."}


def validate_auxiliaries(aux: Mapping[str, AuxiliaryObservation], image_hw: tuple[int, int]) -> None:
    for name, observation in aux.items():
        observation.validate(name, image_hw)


def validate_mask_binding(input_ids: Tensor, positions: Tensor, image_indices: Tensor,
                          images_per_sample: list[int], mask_token_id: int,
                          valid: Tensor | None = None) -> None:
    if positions.ndim != 2 or positions.shape != image_indices.shape or positions.shape[0] != input_ids.shape[0]:
        raise ValueError("Mask binding shape mismatch")
    if len(images_per_sample) != input_ids.shape[0]:
        raise ValueError("Image count batch mismatch")
    valid = torch.ones_like(positions, dtype=torch.bool) if valid is None else valid.bool()
    if valid.shape != positions.shape:
        raise ValueError("Mask validity shape mismatch")
    for b in range(len(images_per_sample)):
        selected = positions[b][valid[b]].tolist()
        if len(selected) != len(set(selected)):
            raise ValueError("Duplicate mask token positions")
        for k in range(positions.shape[1]):
            if not valid[b, k]:
                continue
            p, image = int(positions[b, k]), int(image_indices[b, k])
            if not 0 <= p < input_ids.shape[1] or int(input_ids[b, p]) != mask_token_id:
                raise ValueError("Mask position does not point to a mask token")
            if not 0 <= image < images_per_sample[b]:
                raise ValueError("Mask image binding out of range")
