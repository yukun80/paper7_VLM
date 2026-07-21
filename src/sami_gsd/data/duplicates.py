"""Exact and verified perceptual duplicate grouping before parent splitting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sami_gsd.contracts.canonical import CanonicalParentV3
from sami_gsd.contracts.config import DuplicateSettings
from sami_gsd.utilities.artifacts import canonical_json_bytes, sha256_bytes


DUPLICATE_PROTOCOL_VERSION = "sami_duplicate_protocol_v1_sha256_dhash_rgb64_mae"


class DuplicateDetectionError(ValueError):
    """Raised when a parent cannot enter deterministic duplicate analysis."""


@dataclass(frozen=True)
class DuplicateAnalysis:
    """Stable duplicate components and their exact/perceptual evidence."""

    clusters: tuple[dict[str, Any], ...]
    parent_to_cluster: dict[str, str]
    exact_edge_count: int
    perceptual_candidate_edge_count: int
    verified_perceptual_edge_count: int
    aggregate_sha256: str


class _UnionFind:
    """Small deterministic disjoint-set implementation."""

    def __init__(self, values: tuple[str, ...]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        """Return a root with path compression."""

        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            following = self.parent[value]
            self.parent[value] = root
            value = following
        return root

    def union(self, left: str, right: str) -> None:
        """Join roots using lexical order for deterministic representatives."""

        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        low, high = sorted((left_root, right_root))
        self.parent[high] = low


def _numpy() -> Any:
    """Load the P1 data dependency only when duplicate analysis runs."""

    try:
        import numpy as np
    except ImportError as error:  # pragma: no cover - minimal installs
        raise DuplicateDetectionError("duplicate analysis requires the sami-groundsegdesc[data] extra") from error
    return np


def _resize_bilinear(image: Any, output_hw: tuple[int, int]) -> Any:
    """Resize HWC using the same half-pixel-center policy as materialization."""

    np = _numpy()
    input_h, input_w, _ = image.shape
    output_h, output_w = output_hw
    source_y = np.clip((np.arange(output_h) + 0.5) * input_h / output_h - 0.5, 0.0, input_h - 1)
    source_x = np.clip((np.arange(output_w) + 0.5) * input_w / output_w - 0.5, 0.0, input_w - 1)
    y0 = np.floor(source_y).astype(int)
    x0 = np.floor(source_x).astype(int)
    y1 = np.minimum(input_h - 1, y0 + 1)
    x1 = np.minimum(input_w - 1, x0 + 1)
    wy = (source_y - y0)[:, None, None]
    wx = (source_x - x0)[None, :, None]
    top = (1.0 - wx) * image[y0[:, None], x0[None, :]] + wx * image[y0[:, None], x1[None, :]]
    bottom = (1.0 - wx) * image[y1[:, None], x0[None, :]] + wx * image[y1[:, None], x1[None, :]]
    return (1.0 - wy) * top + wy * bottom


def _reference_modality(parent: CanonicalParentV3) -> Any:
    """Return the unique reference modality already enforced by the contract."""

    return next(
        modality
        for modality in parent.modalities
        if modality.modality_id == parent.reference_canvas.reference_modality_id
    )


def normalize_parent_rgb64(parent: CanonicalParentV3, *, benchmark_root: Path) -> Any:
    """Normalize one reference view to uint8 RGB64 for MAE verification.

    Per-channel min/max uses only valid pixels. The normalization is duplicate
    evidence, not a physical measurement and never enters model prompts.
    """

    np = _numpy()
    modality = _reference_modality(parent)
    if modality.aligned_asset_path is None or modality.valid_mask_path is None:
        raise DuplicateDetectionError(f"parent has no materialized reference view: {parent.parent_id}")
    image = np.load(benchmark_root / modality.aligned_asset_path, allow_pickle=False)
    valid = np.load(benchmark_root / modality.valid_mask_path, allow_pickle=False).astype(bool)
    if image.ndim != 3 or valid.ndim != 2 or tuple(image.shape[:2]) != tuple(valid.shape):
        raise DuplicateDetectionError(f"reference image/valid shape conflict: {parent.parent_id}")
    if not valid.any():
        raise DuplicateDetectionError(f"reference view is zero-valid: {parent.parent_id}")
    if image.shape[2] >= 3:
        rgb = image[..., :3].astype("float64", copy=True)
    elif image.shape[2] == 1:
        rgb = np.repeat(image.astype("float64", copy=False), 3, axis=2)
    else:
        mean = image.astype("float64", copy=False).mean(axis=2, keepdims=True)
        rgb = np.repeat(mean, 3, axis=2)
    normalized = np.zeros_like(rgb, dtype="float64")
    for channel in range(3):
        values = rgb[..., channel][valid]
        low = float(values.min())
        high = float(values.max())
        if high > low:
            normalized[..., channel] = (rgb[..., channel] - low) * (255.0 / (high - low))
    normalized[~valid] = 0.0
    resized = _resize_bilinear(normalized, (64, 64))
    return np.clip(np.floor(resized + 0.5), 0, 255).astype("u1")


def _dhash(rgb64: Any) -> int:
    """Compute a 64-bit dHash used only for candidate recall."""

    np = _numpy()
    grayscale = rgb64.astype("float64").mean(axis=2, keepdims=True)
    reduced = _resize_bilinear(grayscale, (8, 9))[..., 0]
    bits = (reduced[:, 1:] >= reduced[:, :-1]).reshape(-1)
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def _cluster_id(members: tuple[str, ...]) -> str:
    """Derive a stable identifier from sorted member IDs."""

    return f"dup-{sha256_bytes(canonical_json_bytes(list(members)))[:20]}"


def build_duplicate_analysis(
    parents: tuple[CanonicalParentV3, ...],
    *,
    benchmark_root: Path,
    settings: DuplicateSettings,
) -> DuplicateAnalysis:
    """Run SHA exact grouping, dHash recall and RGB64 MAE verification."""

    np = _numpy()
    ordered = tuple(sorted(parents, key=lambda item: item.parent_id))
    parent_ids = tuple(parent.parent_id for parent in ordered)
    if not parent_ids or len(parent_ids) != len(set(parent_ids)):
        raise DuplicateDetectionError("duplicate analysis requires non-empty unique parent IDs")
    union = _UnionFind(parent_ids)
    rgb64 = {parent.parent_id: normalize_parent_rgb64(parent, benchmark_root=benchmark_root) for parent in ordered}
    exact_hashes = {
        parent.parent_id: _reference_modality(parent).hashes["native"]
        for parent in ordered
    }
    dhashes = {parent_id: _dhash(image) for parent_id, image in rgb64.items()}
    exact_edges: list[tuple[str, str]] = []
    candidates: list[tuple[str, str, int]] = []
    verified: list[tuple[str, str, float]] = []
    for index, left in enumerate(parent_ids):
        for right in parent_ids[index + 1 :]:
            if exact_hashes[left] == exact_hashes[right]:
                exact_edges.append((left, right))
                union.union(left, right)
                continue
            distance = (dhashes[left] ^ dhashes[right]).bit_count()
            if distance > settings.dhash_candidate_max_distance:
                continue
            candidates.append((left, right, distance))
            mae = float(np.abs(rgb64[left].astype("int16") - rgb64[right].astype("int16")).mean())
            if mae <= settings.verified_rgb64_mae_threshold:
                verified.append((left, right, mae))
                union.union(left, right)

    members_by_root: dict[str, list[str]] = {}
    for parent_id in parent_ids:
        members_by_root.setdefault(union.find(parent_id), []).append(parent_id)
    clusters: list[dict[str, Any]] = []
    mapping: dict[str, str] = {}
    for members_list in sorted(members_by_root.values(), key=lambda values: tuple(values)):
        members = tuple(sorted(members_list))
        cluster_id = _cluster_id(members)
        for parent_id in members:
            mapping[parent_id] = cluster_id
        clusters.append(
            {
                "schema_version": "sami_duplicate_cluster_v1",
                "cluster_id": cluster_id,
                "parent_ids": list(members),
                "exact_edges": [list(edge) for edge in exact_edges if edge[0] in members and edge[1] in members],
                "verified_perceptual_edges": [
                    {"left": left, "right": right, "rgb64_mae": mae}
                    for left, right, mae in verified
                    if left in members and right in members
                ],
            }
        )
    payload = {
        "protocol": DUPLICATE_PROTOCOL_VERSION,
        "settings": settings.model_dump(mode="json"),
        "clusters": clusters,
        "perceptual_candidates": [
            {"left": left, "right": right, "dhash_distance": distance}
            for left, right, distance in candidates
        ],
    }
    return DuplicateAnalysis(
        clusters=tuple(clusters),
        parent_to_cluster=dict(sorted(mapping.items())),
        exact_edge_count=len(exact_edges),
        perceptual_candidate_edge_count=len(candidates),
        verified_perceptual_edge_count=len(verified),
        aggregate_sha256=sha256_bytes(canonical_json_bytes(payload)),
    )


__all__ = [
    "DUPLICATE_PROTOCOL_VERSION",
    "DuplicateAnalysis",
    "DuplicateDetectionError",
    "build_duplicate_analysis",
    "normalize_parent_rgb64",
]
