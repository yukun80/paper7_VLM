"""Synthetic P2 fixtures independent of raw datasets and generated benchmarks."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from PIL import Image

from sami_gsd.contracts.canonical import CanonicalParentV3
from sami_gsd.contracts.model import SamiModelConfig, load_model_config
from sami_gsd.model.rendering import image_content_sha256, valid_mask_sha256
from sami_gsd.model.states import LoadedParent, LoadedView
from tests.p1.conftest import canonical_parent_payload


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def model_config() -> SamiModelConfig:
    """Load the checked-in strict P2 configuration."""

    return load_model_config(REPOSITORY_ROOT / "configs" / "model_sami.yaml")


def parent_payload(
    specifications: tuple[tuple[str, str, str], ...],
    *,
    reference_id: str,
    parent_id: str = "synthetic-p2-parent",
) -> dict[str, Any]:
    """Create a valid canonical parent with requested family/availability views."""

    payload = canonical_parent_payload()
    payload["parent_id"] = parent_id
    payload["source"]["record_id"] = parent_id
    payload["reference_canvas"]["reference_modality_id"] = reference_id
    template = payload["modalities"][0]
    modalities: list[dict[str, Any]] = []
    for modality_id, family, availability in specifications:
        modality = deepcopy(template)
        modality["modality_id"] = modality_id
        modality["family"] = family
        modality["sensor"] = f"synthetic-{family}-sensor"
        modality["product_type"] = f"synthetic-{family}-product"
        modality["alignment_status"] = "reference" if modality_id == reference_id else "aligned"
        modality["native_asset_path"] = f"assets/{parent_id}/{modality_id}.npy"
        modality["aligned_asset_path"] = f"assets/{parent_id}/{modality_id}.npy"
        modality["valid_mask_path"] = f"assets/{parent_id}/{modality_id}-valid.npy"
        modality["availability_status"] = availability
        modality["valid_coverage"] = {
            "present_valid": 1.0,
            "present_partial_valid": 0.5,
            "present_zero_valid": 0.0,
            "missing": 0.0,
        }[availability]
        if family in {"sar", "insar", "deformation"}:
            modality["orbit"] = "ascending"
            modality["units"] = "dB"
            modality["sign_convention"] = "larger_is_stronger_backscatter"
        if availability == "missing":
            modality["native_asset_path"] = None
            modality["aligned_asset_path"] = None
            modality["valid_mask_path"] = None
            modality["source_to_reference_transform"] = None
            modality["reference_to_source_transform"] = None
            modality["alignment_status"] = "unavailable"
        modalities.append(modality)
    payload["modalities"] = modalities
    return payload


def loaded_parent(
    specifications: tuple[tuple[str, str, str], ...],
    *,
    reference_id: str,
    parent_id: str = "synthetic-p2-parent",
    view_insertion_order: tuple[str, ...] | None = None,
) -> LoadedParent:
    """Create decoded RGB/valid views for all non-zero-valid present modalities."""

    record = CanonicalParentV3.model_validate(
        parent_payload(specifications, reference_id=reference_id, parent_id=parent_id)
    )
    declared = {modality.modality_id: modality for modality in record.modalities}
    order = view_insertion_order or tuple(modality.modality_id for modality in record.modalities)
    views: dict[str, LoadedView] = {}
    for index, modality_id in enumerate(order):
        modality = declared[modality_id]
        if modality.availability_status not in {"present_valid", "present_partial_valid"}:
            continue
        image = Image.new("RGB", (48, 32), color=(20 + index, 40 + index, 60 + index))
        valid = torch.ones((32, 48), dtype=torch.bool)
        if modality.availability_status == "present_partial_valid":
            valid[:, 24:] = False
        views[modality_id] = LoadedView(
            modality=modality,
            image=image,
            valid_mask=valid,
            image_sha256=image_content_sha256(image),
            valid_sha256=valid_mask_sha256(valid),
        )
    result = LoadedParent(record=record, views=views)
    result.validate()
    return result


class Qwen3VLProcessor:
    """Minimal official-name processor double with deterministic merged grids."""

    def __init__(self) -> None:
        self.image_processor = SimpleNamespace(patch_size=16, merge_size=2)

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        """Represent each native image item by one parseable placeholder."""

        if tokenize or add_generation_prompt:
            raise AssertionError("P2 wrapper must request an untokenized, non-generation prompt")
        content = messages[1]["content"]
        return " ".join("<image>" if item["type"] == "image" else item["text"] for item in content)

    def __call__(
        self,
        *,
        text: list[str],
        images: list[Image.Image],
        padding: bool,
        return_tensors: str,
        min_pixels: int,
        max_pixels: int,
    ) -> dict[str, torch.Tensor]:
        """Emit four merged tokens per image and parent-aware input IDs."""

        if not padding or return_tensors != "pt" or min_pixels <= 0 or max_pixels <= 0:
            raise AssertionError("official processor call contract changed")
        image_counts = [prompt.count("<image>") for prompt in text]
        if sum(image_counts) != len(images):
            raise AssertionError("prompt/image cardinality differs")
        rows = [[1] + [99] * (count * 4) + [2] for count in image_counts]
        width = max(len(row) for row in rows)
        input_ids = torch.tensor([row + [0] * (width - len(row)) for row in rows], dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": input_ids.ne(0),
            "pixel_values": torch.zeros((len(images), 1), dtype=torch.float32),
            "image_grid_thw": torch.tensor([[1, 4, 4]] * len(images), dtype=torch.long),
        }


class _FakeVisual(torch.nn.Module):
    """Official visual-tower surface with two DeepStack levels."""

    patch_size = 16
    spatial_merge_size = 2

    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> Any:
        token_count = int(grid_thw.shape[0]) * 4
        values = torch.arange(token_count * 8, dtype=torch.float32).reshape(token_count, 8)
        return SimpleNamespace(
            pooler_output=values + 2.0,
            deepstack_features=(values, values + 1.0),
        )


class _FakeLanguage(torch.nn.Module):
    """Language surface returning deterministic hidden states."""

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor]:
        base = input_ids.to(dtype=torch.float32).unsqueeze(-1)
        return (base.repeat(1, 1, 8) / 100.0,)


class Qwen3VLForConditionalGeneration(torch.nn.Module):
    """Minimal official-name model double that counts complete forwards."""

    def __init__(self) -> None:
        super().__init__()
        self.model = torch.nn.Module()
        self.model.visual = _FakeVisual()
        self.model.language_model = _FakeLanguage()
        self.config = SimpleNamespace(
            image_token_id=99,
            vision_config=SimpleNamespace(deepstack_visual_indexes=(1, 3)),
        )
        self.forward_count = 0

    def forward(self, **inputs: Any) -> dict[str, torch.Tensor]:
        self.forward_count += 1
        visual = self.model.visual(inputs["pixel_values"], inputs["image_grid_thw"])
        _ = visual
        hidden = self.model.language_model(inputs["input_ids"])
        return {"last_hidden_state": hidden[0]}
