"""Strict write-once memoization for P2 ``QwenBackboneState``."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from sami_gsd.contracts.model import CacheEquivalenceSettings
from sami_gsd.model.states import (
    ExcludedModality,
    QwenBackboneState,
    SensorCard,
    SpatialFeatureLevel,
    ViewBackboneState,
    ViewTransform,
)
from sami_gsd.utilities.artifacts import (
    atomic_output_directory,
    atomic_write_json,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
)


def tensor_sha256(tensor: Tensor) -> str:
    """Hash exact CPU tensor dtype, shape and storage bytes."""

    cpu = tensor.detach().contiguous().cpu()
    header = f"{cpu.dtype}|{tuple(cpu.shape)}|".encode("utf-8")
    payload = cpu.view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(header + payload).hexdigest()


def build_cache_key(payload: dict[str, Any]) -> tuple[str, str]:
    """Return cache key and exact canonical key-payload digest."""

    payload_bytes = canonical_json_bytes(payload)
    payload_sha256 = sha256_bytes(payload_bytes)
    return payload_sha256, payload_sha256


def _state_payload(state: QwenBackboneState) -> dict[str, Any]:
    """Project one state into primitives and CPU tensors for strict loading."""

    return {
        "schema_version": "sami_qwen_backbone_cache_payload_v1",
        "metadata": state.metadata_payload(),
        "views": [
            {
                "language_aligned_visual_tokens": view.language_aligned_visual_tokens.detach().cpu(),
                "valid_mask": view.valid_mask.detach().cpu(),
                "spatial_features": [level.features.detach().cpu() for level in view.spatial_features],
            }
            for view in state.views
        ],
    }


def _sensor_card(payload: dict[str, Any]) -> SensorCard:
    """Reconstruct a strict sensor-card dataclass from cache metadata."""

    expected = {
        "view_id",
        "family",
        "sensor",
        "product_type",
        "bands_or_polarizations",
        "orbit",
        "gsd_m",
        "units",
        "sign_convention",
        "valid_coverage",
        "quality_status",
        "quality_flags",
    }
    if set(payload) != expected:
        raise ValueError("cache sensor-card fields do not match P2 contract")
    return SensorCard(
        view_id=str(payload["view_id"]),
        family=str(payload["family"]),
        sensor=str(payload["sensor"]),
        product_type=str(payload["product_type"]),
        bands_or_polarizations=tuple(str(item) for item in payload["bands_or_polarizations"]),
        orbit=None if payload["orbit"] is None else str(payload["orbit"]),
        gsd_m=None if payload["gsd_m"] is None else float(payload["gsd_m"]),
        units=None if payload["units"] is None else str(payload["units"]),
        sign_convention=(
            None if payload["sign_convention"] is None else str(payload["sign_convention"])
        ),
        valid_coverage=float(payload["valid_coverage"]),
        quality_status=str(payload["quality_status"]),
        quality_flags=tuple(str(item) for item in payload["quality_flags"]),
    )


def _transform(payload: dict[str, Any]) -> ViewTransform:
    """Reconstruct exact grid/canonical transform metadata."""

    expected = {
        "source_hw",
        "rendered_hw",
        "processor_grid_thw",
        "merged_grid_hw",
        "alignment_status",
        "source_to_reference_transform",
        "reference_to_source_transform",
    }
    if set(payload) != expected:
        raise ValueError("cache view-transform fields do not match P2 contract")

    def transform_chain(value: Any) -> tuple[dict[str, Any], ...] | None:
        if value is None:
            return None
        if not isinstance(value, (list, tuple)) or not all(isinstance(item, dict) for item in value):
            raise ValueError("cached canonical transform chain must be a sequence of mappings or null")
        return tuple(dict(item) for item in value)

    return ViewTransform(
        source_hw=tuple(int(item) for item in payload["source_hw"]),
        rendered_hw=tuple(int(item) for item in payload["rendered_hw"]),
        processor_grid_thw=tuple(int(item) for item in payload["processor_grid_thw"]),
        merged_grid_hw=tuple(int(item) for item in payload["merged_grid_hw"]),
        alignment_status=str(payload["alignment_status"]),
        source_to_reference_transform=transform_chain(payload["source_to_reference_transform"]),
        reference_to_source_transform=transform_chain(payload["reference_to_source_transform"]),
    )


def _reconstruct_state(payload: dict[str, Any]) -> QwenBackboneState:
    """Reconstruct and deeply validate a cached state payload."""

    if set(payload) != {"schema_version", "metadata", "views"}:
        raise ValueError("cache payload has unexpected fields")
    if payload["schema_version"] != "sami_qwen_backbone_cache_payload_v1":
        raise ValueError("unsupported cache payload schema")
    metadata = payload["metadata"]
    view_tensors = payload["views"]
    if not isinstance(metadata, dict) or not isinstance(view_tensors, list):
        raise ValueError("cache payload metadata/views have invalid types")
    expected_metadata_fields = {
        "schema_version",
        "parent_ids",
        "view_order",
        "reference_view_ids",
        "active_modality_ids",
        "excluded_modalities",
        "views",
        "prompt_sha256",
        "model_fingerprint",
        "processor_fingerprint",
        "qwen_code_revision",
        "profile",
        "dtype",
        "cache_key",
    }
    if set(metadata) != expected_metadata_fields:
        raise ValueError("cache state metadata fields do not match greenfield v1")
    metadata_views = metadata.get("views")
    if not isinstance(metadata_views, list) or len(metadata_views) != len(view_tensors):
        raise ValueError("cache metadata/tensor view counts differ")
    views: list[ViewBackboneState] = []
    for view_metadata, tensors in zip(metadata_views, view_tensors, strict=True):
        if not isinstance(view_metadata, dict) or not isinstance(tensors, dict):
            raise ValueError("cache view entry must contain mappings")
        if set(tensors) != {"language_aligned_visual_tokens", "valid_mask", "spatial_features"}:
            raise ValueError("cache tensor view has unexpected fields")
        levels = view_metadata.get("spatial_levels")
        spatial_tensors = tensors["spatial_features"]
        if not isinstance(levels, list) or not isinstance(spatial_tensors, list) or len(levels) != len(spatial_tensors):
            raise ValueError("cache spatial level names/tensors differ")
        role = str(view_metadata["role"])
        if role not in {"reference", "support"}:
            raise ValueError("cached view role is not registered")
        views.append(
            ViewBackboneState(
                parent_id=str(view_metadata["parent_id"]),
                view_id=str(view_metadata["view_id"]),
                role=role,  # type: ignore[arg-type]
                sensor_card=_sensor_card(view_metadata["sensor_card"]),
                language_aligned_visual_tokens=tensors["language_aligned_visual_tokens"],
                spatial_features=tuple(
                    SpatialFeatureLevel(level=str(level), features=tensor)
                    for level, tensor in zip(levels, spatial_tensors, strict=True)
                ),
                valid_mask=tensors["valid_mask"],
                transform=_transform(view_metadata["transform"]),
                image_sha256=str(view_metadata["image_sha256"]),
                valid_sha256=str(view_metadata["valid_sha256"]),
            )
        )
    excluded_groups: list[tuple[ExcludedModality, ...]] = []
    for group in metadata["excluded_modalities"]:
        reconstructed_group: list[ExcludedModality] = []
        for item in group:
            if set(item) != {"modality_id", "availability_status", "reason"}:
                raise ValueError("cached excluded-modality fields do not match P2 contract")
            reason = str(item["reason"])
            if reason not in {"inactive_dropout", "missing", "present_zero_valid"}:
                raise ValueError("cached excluded-modality reason is not registered")
            reconstructed_group.append(
                ExcludedModality(
                    modality_id=str(item["modality_id"]),
                    availability_status=str(item["availability_status"]),
                    reason=reason,  # type: ignore[arg-type]
                )
            )
        excluded_groups.append(tuple(reconstructed_group))
    excluded = tuple(excluded_groups)
    state = QwenBackboneState(
        schema_version="sami_qwen_backbone_state_v1",
        parent_ids=tuple(str(item) for item in metadata["parent_ids"]),
        view_order=tuple(tuple(str(item) for item in order) for order in metadata["view_order"]),
        reference_view_ids=tuple(str(item) for item in metadata["reference_view_ids"]),
        active_modality_ids=tuple(
            tuple(str(item) for item in group) for group in metadata["active_modality_ids"]
        ),
        excluded_modalities=excluded,
        views=tuple(views),
        prompt_sha256=tuple(str(item) for item in metadata["prompt_sha256"]),
        model_fingerprint=str(metadata["model_fingerprint"]),
        processor_fingerprint=str(metadata["processor_fingerprint"]),
        qwen_code_revision=str(metadata["qwen_code_revision"]),
        profile=str(metadata["profile"]),
        dtype=str(metadata["dtype"]),
        cache_key=str(metadata["cache_key"]),
        from_cache=True,
    )
    state.validate()
    return state


class QwenBackboneCache:
    """Write-once directory cache with payload byte and tensor inventory replay."""

    def __init__(self, root: Path, *, schema_version: str) -> None:
        if schema_version != "sami_qwen_backbone_cache_v1":
            raise ValueError("QwenBackboneCache only accepts the greenfield v1 schema")
        self.root = root.resolve()
        self.schema_version = schema_version

    def _entry_dir(self, cache_key: str) -> Path:
        if len(cache_key) != 64 or any(character not in "0123456789abcdef" for character in cache_key):
            raise ValueError("cache key must be lowercase SHA-256")
        return self.root / cache_key[:2] / cache_key

    def get(self, cache_key: str, *, key_payload_sha256: str) -> QwenBackboneState | None:
        """Return a validated hit, ``None`` for absence, and never hide corruption."""

        entry = self._entry_dir(cache_key)
        if not entry.exists():
            return None
        if not entry.is_dir():
            raise ValueError(f"cache entry is not a directory: {entry}")
        manifest_path = entry / "manifest.json"
        state_path = entry / "state.pt"
        expected_names = {"manifest.json", "state.pt"}
        actual_names = {path.name for path in entry.iterdir()}
        if actual_names != expected_names:
            raise ValueError(f"cache entry contains unbound files: {sorted(actual_names - expected_names)}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_manifest_fields = {
            "schema_version",
            "cache_key",
            "key_payload_sha256",
            "state_sha256",
            "tensor_inventory",
        }
        if set(manifest) != expected_manifest_fields:
            raise ValueError("cache manifest fields do not match greenfield v1")
        if manifest["schema_version"] != self.schema_version:
            raise ValueError("cache manifest schema mismatch")
        if manifest["cache_key"] != cache_key or manifest["key_payload_sha256"] != key_payload_sha256:
            raise ValueError("cache key payload binding mismatch")
        if sha256_file(state_path) != manifest["state_sha256"]:
            raise ValueError("cache state bytes do not match manifest SHA-256")
        payload = torch.load(state_path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict):
            raise ValueError("cache state payload must be a mapping")
        state = _reconstruct_state(payload)
        actual_inventory = [
            {
                "name": name,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "sha256": tensor_sha256(tensor),
            }
            for name, tensor in state.tensor_items()
        ]
        if actual_inventory != manifest["tensor_inventory"]:
            raise ValueError("cache tensor inventory replay mismatch")
        return state

    def write(self, state: QwenBackboneState, *, key_payload_sha256: str) -> Path:
        """Atomically publish a new cache entry without overwriting any prior bytes."""

        state.validate()
        entry = self._entry_dir(state.cache_key)
        payload = _state_payload(state)
        with atomic_output_directory(entry) as staging:
            state_path = staging / "state.pt"
            torch.save(payload, state_path)
            tensor_inventory = [
                {
                    "name": name,
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "sha256": tensor_sha256(tensor),
                }
                for name, tensor in state.tensor_items()
            ]
            atomic_write_json(
                staging / "manifest.json",
                {
                    "schema_version": self.schema_version,
                    "cache_key": state.cache_key,
                    "key_payload_sha256": key_payload_sha256,
                    "state_sha256": sha256_file(state_path),
                    "tensor_inventory": tensor_inventory,
                },
            )
        return entry


def compare_backbone_states(
    online: QwenBackboneState,
    cached: QwenBackboneState,
    settings: CacheEquivalenceSettings,
) -> dict[str, Any]:
    """Enforce exact metadata/shape and protocol numerical equivalence."""

    online.validate()
    cached.validate()
    if online.metadata_payload() != cached.metadata_payload():
        raise ValueError("online/cache non-tensor metadata differ")
    online_items = online.tensor_items()
    cached_items = cached.tensor_items()
    if tuple(name for name, _ in online_items) != tuple(name for name, _ in cached_items):
        raise ValueError("online/cache tensor inventories differ")
    comparisons: list[dict[str, Any]] = []
    for (name, online_tensor), (_, cached_tensor) in zip(online_items, cached_items, strict=True):
        if online_tensor.shape != cached_tensor.shape or online_tensor.dtype != cached_tensor.dtype:
            raise ValueError(f"online/cache shape or dtype differs for {name}")
        if online_tensor.dtype == torch.bool:
            if not torch.equal(online_tensor.cpu(), cached_tensor.cpu()):
                raise ValueError(f"online/cache bool tensor differs for {name}")
            comparisons.append({"name": name, "cosine_similarity": 1.0, "max_abs": 0.0})
            continue
        left = online_tensor.detach().to(dtype=torch.float64, device="cpu").reshape(-1)
        right = cached_tensor.detach().to(dtype=torch.float64, device="cpu").reshape(-1)
        max_abs = float((left - right).abs().max().item()) if left.numel() else 0.0
        left_norm = float(torch.linalg.vector_norm(left).item())
        right_norm = float(torch.linalg.vector_norm(right).item())
        if left_norm == 0.0 or right_norm == 0.0:
            cosine = 1.0 if torch.equal(left, right) else 0.0
        else:
            raw_cosine = float(torch.dot(left, right).item() / (left_norm * right_norm))
            cosine = max(-1.0, min(1.0, raw_cosine))
        threshold = settings.bf16_max_abs if online_tensor.dtype == torch.bfloat16 else settings.fp32_max_abs
        if cosine < settings.cosine_similarity_min or max_abs > threshold:
            raise ValueError(
                f"online/cache numerical mismatch for {name}: cosine={cosine}, max_abs={max_abs}"
            )
        comparisons.append({"name": name, "cosine_similarity": cosine, "max_abs": max_abs})
    return {
        "passed": True,
        "metadata_exact": True,
        "tensor_count": len(comparisons),
        "minimum_cosine_similarity": min(item["cosine_similarity"] for item in comparisons),
        "maximum_abs_difference": max(item["max_abs"] for item in comparisons),
        "comparisons": comparisons,
    }


__all__ = [
    "QwenBackboneCache",
    "build_cache_key",
    "compare_backbone_states",
    "tensor_sha256",
]
