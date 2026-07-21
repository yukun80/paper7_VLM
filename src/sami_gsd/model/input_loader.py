"""Read accepted Benchmark v3 assets into file-I/O-free P2 model inputs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from sami_gsd.contracts.canonical import CanonicalParentV3, ModalityRecord, validate_portable_path
from sami_gsd.contracts.model import BenchmarkBinding
from sami_gsd.model.rendering import (
    array_to_hwc,
    image_content_sha256,
    render_modality_rgb,
    valid_mask_sha256,
)
from sami_gsd.model.states import LoadedParent, LoadedView
from sami_gsd.utilities.artifacts import sha256_file


def validate_benchmark_binding(benchmark_root: Path, binding: BenchmarkBinding) -> dict[str, object]:
    """Replay the accepted manifest/report identity without rebuilding data."""

    root = benchmark_root.resolve()
    manifest_path = root / "manifests" / "benchmark_manifest.json"
    validation_path = root / "reports" / "validation_report.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != binding.schema_version:
        raise ValueError("benchmark manifest schema does not match P2 config")
    if manifest.get("mode") != binding.mode:
        raise ValueError("benchmark mode does not match P2 config")
    if manifest.get("aggregate_sha256") != binding.aggregate_sha256:
        raise ValueError("benchmark aggregate SHA-256 does not match accepted P1 binding")
    if validation.get("aggregate_sha256") != binding.validation_aggregate_sha256:
        raise ValueError("benchmark validation SHA-256 does not match accepted P1 binding")
    if validation.get("errors") != []:
        raise ValueError("accepted benchmark validation report contains errors")
    return {
        "manifest_path": "manifests/benchmark_manifest.json",
        "manifest_aggregate_sha256": manifest["aggregate_sha256"],
        "validation_path": "reports/validation_report.json",
        "validation_aggregate_sha256": validation["aggregate_sha256"],
    }


def load_canonical_parent(benchmark_root: Path, parent_id: str) -> CanonicalParentV3:
    """Load exactly one parent from the accepted stable all-parent index."""

    if not parent_id:
        raise ValueError("parent_id must be non-empty")
    index_path = benchmark_root.resolve() / "parents" / "all.jsonl"
    found: dict[str, object] | None = None
    with index_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            payload = json.loads(line)
            if payload.get("parent_id") == parent_id:
                if found is not None:
                    raise ValueError(f"duplicate parent_id {parent_id!r} in {index_path}")
                found = payload
    if found is None:
        raise KeyError(f"parent_id {parent_id!r} is not present in {index_path}")
    return CanonicalParentV3.model_validate(found)


def _resolve_asset(benchmark_root: Path, logical_path: str) -> Path:
    """Resolve one portable benchmark-local path without traversal."""

    validate_portable_path(logical_path)
    root = benchmark_root.resolve()
    resolved = (root / logical_path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"asset path escapes benchmark root: {logical_path}")
    if not resolved.is_file():
        raise FileNotFoundError(f"benchmark asset is missing: {resolved}")
    return resolved


def _load_array(path: Path) -> np.ndarray:
    """Decode a benchmark image array or ordinary RGB image explicitly."""

    if path.suffix.lower() == ".npy":
        return np.load(path, allow_pickle=False)
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def _expected_asset_hash(modality: ModalityRecord, *, aligned: bool) -> str:
    """Return the canonical source hash for the selected array path."""

    key = "aligned" if aligned else "native"
    try:
        return modality.hashes[key]
    except KeyError as error:
        raise ValueError(f"{modality.modality_id} lacks required {key} asset hash") from error


def load_model_parent(
    benchmark_root: Path,
    parent: CanonicalParentV3,
    *,
    requested_modality_ids: tuple[str, ...],
) -> LoadedParent:
    """Decode only requested effective views and verify every source byte hash."""

    declared = {modality.modality_id: modality for modality in parent.modalities}
    unknown = set(requested_modality_ids) - set(declared)
    if unknown:
        raise ValueError(f"requested undeclared modalities: {sorted(unknown)}")
    views: dict[str, LoadedView] = {}
    for modality_id in requested_modality_ids:
        modality = declared[modality_id]
        if modality.availability_status not in {"present_partial_valid", "present_valid"}:
            continue
        aligned = modality.aligned_asset_path is not None
        image_logical_path = modality.aligned_asset_path or modality.native_asset_path
        if image_logical_path is None or modality.valid_mask_path is None:
            raise ValueError(f"effective modality {modality_id} lacks an image or valid-mask path")
        image_path = _resolve_asset(benchmark_root, image_logical_path)
        valid_path = _resolve_asset(benchmark_root, modality.valid_mask_path)
        if sha256_file(image_path) != _expected_asset_hash(modality, aligned=aligned):
            raise ValueError(f"{modality_id} image bytes do not match canonical hash")
        expected_valid_hash = modality.hashes.get("valid")
        if expected_valid_hash is None or sha256_file(valid_path) != expected_valid_hash:
            raise ValueError(f"{modality_id} valid-mask bytes do not match canonical hash")

        raw_array = _load_array(image_path)
        array_hwc = array_to_hwc(
            raw_array,
            band_count=len(modality.band_names),
            modality_id=modality_id,
        )
        valid_array = np.load(valid_path, allow_pickle=False)
        if valid_array.ndim == 3 and valid_array.shape[-1] == 1:
            valid_array = valid_array[:, :, 0]
        if valid_array.ndim != 2:
            raise ValueError(f"{modality_id} valid mask must be 2D")
        image, valid_mask = render_modality_rgb(array_hwc, valid_array != 0, modality)
        view = LoadedView(
            modality=modality,
            image=image,
            valid_mask=valid_mask,
            image_sha256=image_content_sha256(image),
            valid_sha256=valid_mask_sha256(valid_mask),
        )
        view.validate()
        views[modality_id] = view
    loaded = LoadedParent(record=parent, views=views)
    loaded.validate()
    return loaded


__all__ = [
    "load_canonical_parent",
    "load_model_parent",
    "validate_benchmark_binding",
]
