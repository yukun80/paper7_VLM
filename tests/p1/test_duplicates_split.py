"""Synthetic exact/perceptual duplicate and parent-level split tests."""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from sami_gsd.contracts.canonical import SourceIdentity
from sami_gsd.contracts.config import DuplicateSettings, SplitSettings
from sami_gsd.data.duplicates import build_duplicate_analysis
from sami_gsd.data.materialize import materialize_spatial_parent
from sami_gsd.data.split import SplitAssignmentError, apply_parent_splits, assign_parent_splits
from sami_gsd.utilities.artifacts import canonical_json_bytes, sha256_bytes
from tests.p1.test_materialization import spatial_input


def parent_variant(
    parent_id: str,
    *,
    group_id: str,
    image_delta: float = 0.0,
    scene_id: str | None = None,
) -> object:
    """Return a distinct source identity and optionally changed reference bytes."""

    item = spatial_input()
    modality = item.modalities[0]
    image = modality.array.copy()
    valid = modality.valid.astype(bool)
    image[valid] += image_delta
    changed_modality = replace(
        modality,
        array=image,
        source_logical_path=f"datasets/synthetic/{parent_id}.npy",
        source_sha256=(parent_id[0].encode("utf-8").hex() * 64)[:64],
    )
    identity = SourceIdentity(
        dataset="synthetic",
        record_id=parent_id,
        scene_id=scene_id,
        event_id=None,
        region_id=None,
        source_group_id=group_id,
    )
    return replace(item, parent_id=parent_id, source=identity, modalities=(changed_modality,))


class DuplicateAndSplitTests(unittest.TestCase):
    """Verify connected duplicate groups are assigned before task generation."""

    def test_exact_then_verified_perceptual_edges_form_one_component(self) -> None:
        """dHash only recalls candidates; RGB64 MAE decides the verified edge."""

        settings = DuplicateSettings(
            dhash_candidate_max_distance=8,
            verified_rgb64_mae_threshold=3.0,
            normalized_rgb_hw=(64, 64),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = (
                parent_variant("a-parent", group_id="group-a"),
                parent_variant("b-parent", group_id="group-b"),
                parent_variant("c-parent", group_id="group-c", image_delta=1.0),
            )
            parents = tuple(
                materialize_spatial_parent(item, benchmark_root=root, canvas_hw=(8, 8)).parent
                for item in inputs
            )
            first = build_duplicate_analysis(parents, benchmark_root=root, settings=settings)
            second = build_duplicate_analysis(tuple(reversed(parents)), benchmark_root=root, settings=settings)
            self.assertEqual(first.aggregate_sha256, second.aggregate_sha256)
            self.assertEqual(first.exact_edge_count, 1)
            self.assertGreaterEqual(first.perceptual_candidate_edge_count, 1)
            self.assertGreaterEqual(first.verified_perceptual_edge_count, 1)
            self.assertEqual(len(set(first.parent_to_cluster.values())), 1)

    def test_union_constraints_and_forced_test_role_are_split_together(self) -> None:
        """Duplicate, source-group and scene constraints cannot cross a split."""

        duplicate_settings = DuplicateSettings(
            dhash_candidate_max_distance=8,
            verified_rgb64_mae_threshold=3.0,
            normalized_rgb_hw=(64, 64),
        )
        split_settings = SplitSettings(train=0.7, val=0.15, test=0.15)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = (
                parent_variant("a-parent", group_id="group-a", scene_id="scene-shared"),
                parent_variant("b-parent", group_id="group-b"),
                parent_variant("c-parent", group_id="group-c", image_delta=1.0),
                parent_variant("d-parent", group_id="group-d", image_delta=13.0, scene_id="scene-shared"),
            )
            parents = tuple(
                materialize_spatial_parent(item, benchmark_root=root, canvas_hw=(8, 8)).parent
                for item in inputs
            )
            duplicates = build_duplicate_analysis(parents, benchmark_root=root, settings=duplicate_settings)
            assignment = assign_parent_splits(
                parents,
                duplicate_clusters=duplicates.parent_to_cluster,
                settings=split_settings,
                seed=42,
                forced_splits={"a-parent": "test"},
            )
            frozen = apply_parent_splits(parents, assignment)
            self.assertTrue(all(parent.split == "test" for parent in frozen))
            self.assertEqual(
                {assignment.parent_to_split[parent_id] for parent_id in duplicates.parent_to_cluster},
                {"test"},
            )
            repeated = assign_parent_splits(
                tuple(reversed(parents)),
                duplicate_clusters=duplicates.parent_to_cluster,
                settings=split_settings,
                seed=42,
                forced_splits={"a-parent": "test"},
            )
            self.assertEqual(assignment.aggregate_sha256, repeated.aggregate_sha256)

            with self.assertRaisesRegex(SplitAssignmentError, "conflicting forced splits"):
                assign_parent_splits(
                    parents,
                    duplicate_clusters=duplicates.parent_to_cluster,
                    settings=split_settings,
                    seed=42,
                    forced_splits={"a-parent": "train", "b-parent": "test"},
                )

    def test_coverage_rebalance_populates_all_splits_without_breaking_groups(self) -> None:
        """A missing hash bucket moves whole unforced components deterministically."""

        split_settings = SplitSettings(train=0.7, val=0.15, test=0.15)
        parent_ids: list[str] = []
        candidate_index = 0
        while len(parent_ids) < 6:
            parent_id = f"coverage-{candidate_index}"
            fraction = int(
                sha256_bytes(canonical_json_bytes({"seed": 42, "members": (parent_id,)}))[:16],
                16,
            ) / float(16**16)
            if fraction < split_settings.train:
                parent_ids.append(parent_id)
            candidate_index += 1
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parents = tuple(
                materialize_spatial_parent(
                    parent_variant(parent_id, group_id=f"group-{parent_id}"),
                    benchmark_root=root,
                    canvas_hw=(8, 8),
                ).parent
                for parent_id in parent_ids
            )
            clusters = {parent.parent_id: f"unique-{parent.parent_id}" for parent in parents}
            coverage = tuple(parent.parent_id for parent in parents)
            assignment = assign_parent_splits(
                parents,
                duplicate_clusters=clusters,
                settings=split_settings,
                seed=42,
                coverage_parent_ids=coverage,
            )
            repeated = assign_parent_splits(
                tuple(reversed(parents)),
                duplicate_clusters=clusters,
                settings=split_settings,
                seed=42,
                coverage_parent_ids=tuple(reversed(coverage)),
            )
            self.assertEqual(set(assignment.parent_to_split.values()), {"train", "val", "test"})
            self.assertTrue(any(
                component["assignment_reason"] == "seeded_component_hash_coverage_rebalance"
                for component in assignment.components
            ))
            self.assertEqual(assignment.aggregate_sha256, repeated.aggregate_sha256)


if __name__ == "__main__":
    unittest.main()
