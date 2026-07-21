"""Synthetic preprocessing, valid/nodata and materialization tests."""

from __future__ import annotations

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import numpy as np

from sami_gsd.contracts.canonical import SourceIdentity
from sami_gsd.data.materialize import (
    MaterializationError,
    SourceModalityInput,
    SourceReferringInput,
    SpatialParentInput,
    build_fit_pad_transform,
    materialize_spatial_parent,
    transform_geotransform,
)
from sami_gsd.utilities.artifacts import canonical_json_bytes, sha256_bytes


SHA = "b" * 64


def spatial_input() -> SpatialParentInput:
    """Build a non-square HWC record with explicit nodata and a referring mask."""

    image = np.arange(2 * 4 * 3, dtype=np.float32).reshape(2, 4, 3)
    image[0, 0] = np.nan
    valid = np.ones((2, 4), dtype=np.uint8)
    valid[0, 0] = 0
    mask = np.zeros((2, 4), dtype=np.uint8)
    mask[:, 2:] = 1
    region = np.zeros((2, 4), dtype=np.uint8)
    region[1, 2:] = 1
    modality = SourceModalityInput(
        modality_id="reference_rgb",
        family="optical",
        sensor="synthetic-sensor",
        product_type="synthetic-rgb",
        band_names=("R", "G", "B"),
        array=image,
        valid=valid,
        source_logical_path="datasets/synthetic/image.npy",
        source_sha256=SHA,
        crs="EPSG:4326",
        geotransform=(100.0, 0.1, 0.0, 30.0, 0.0, -0.1),
    )
    return SpatialParentInput(
        parent_id="synthetic-parent-001",
        source_registry_key="synthetic",
        source=SourceIdentity(
            dataset="synthetic",
            record_id="record-001",
            scene_id="scene-001",
            event_id=None,
            region_id="region-001",
            source_group_id="group-001",
        ),
        reference_modality_id="reference_rgb",
        modalities=(modality,),
        global_mask=mask,
        global_mask_origin="official",
        referring_regions=(
            SourceReferringInput(
                region_id="region-001",
                expression="the right-side landslide",
                mask=region,
                annotation_origin="official",
            ),
        ),
        source_record_sha256=SHA,
        annotation_status="gold",
    )


class MaterializationTests(unittest.TestCase):
    """Verify deterministic array publication and evidence exclusion."""

    def test_fit_pad_materialization_is_deterministic_and_raw_read_only(self) -> None:
        """Two new roots receive byte-identical assets and parent records."""

        item = spatial_input()
        original_image = item.modalities[0].array.copy()
        original_valid = item.modalities[0].valid.copy()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = materialize_spatial_parent(item, benchmark_root=root / "one", canvas_hw=(6, 6))
            second = materialize_spatial_parent(item, benchmark_root=root / "two", canvas_hw=(6, 6))

            first_payload = first.parent.model_dump(mode="json")
            second_payload = second.parent.model_dump(mode="json")
            self.assertEqual(first_payload, second_payload)
            self.assertEqual(
                sha256_bytes(canonical_json_bytes(first_payload)),
                sha256_bytes(canonical_json_bytes(second_payload)),
            )
            first_files = {
                path.relative_to(root / "one").as_posix(): path.read_bytes()
                for path in (root / "one").rglob("*.npy")
            }
            second_files = {
                path.relative_to(root / "two").as_posix(): path.read_bytes()
                for path in (root / "two").rglob("*.npy")
            }
            self.assertEqual(first_files, second_files)
            self.assertEqual(list(root.rglob("*.part-*")), [])

        np.testing.assert_equal(item.modalities[0].array, original_image)
        np.testing.assert_equal(item.modalities[0].valid, original_valid)

    def test_nodata_and_padding_are_zeroed_and_excluded(self) -> None:
        """Invalid source pixels and canvas padding never become target evidence."""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = materialize_spatial_parent(spatial_input(), benchmark_root=root, canvas_hw=(6, 6))
            valid = np.load(root / result.parent.reference_canvas.valid_mask_path, allow_pickle=False)
            mask_ref = result.parent.annotations.global_landslide_mask
            assert mask_ref is not None
            mask = np.load(root / mask_ref.path, allow_pickle=False)
            aligned_path = result.parent.modalities[0].aligned_asset_path
            assert aligned_path is not None
            aligned = np.load(root / aligned_path, allow_pickle=False)
            self.assertEqual(valid.dtype, np.dtype("uint8"))
            self.assertEqual(mask.dtype, np.dtype("uint8"))
            self.assertTrue(np.all(aligned[valid == 0] == 0.0))
            self.assertTrue(np.all(mask[valid == 0] == 0))
            self.assertEqual(result.excluded_pixel_count, 36 - result.valid_pixel_count)
            self.assertEqual(result.positive_valid_pixel_count, int(mask.sum()))
            self.assertTrue(np.all(valid[0] == 0))
            self.assertTrue(np.all(valid[-1] == 0))

    def test_reference_geotransform_tracks_resize_and_padding(self) -> None:
        """The output affine refers to the materialized canvas, not the raw grid."""

        chain = build_fit_pad_transform((2, 4), (6, 6))
        transformed = transform_geotransform((100.0, 0.1, 0.0, 30.0, 0.0, -0.1), chain)
        self.assertEqual(chain[-1].output_hw, (6, 6))
        self.assertIsNotNone(transformed)
        assert transformed is not None
        self.assertAlmostEqual(transformed[1], 0.1 * 4 / 6)
        self.assertAlmostEqual(transformed[5], -0.1 * 2 / 3)
        self.assertAlmostEqual(transformed[3], 30.0 + (-0.1) * (-1 * 2 / 3))

    def test_grid_mismatch_fails_before_publication(self) -> None:
        """Implicit support reprojection remains closed."""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "benchmark"
            item = spatial_input()
            support = deepcopy(item.modalities[0])
            object.__setattr__(support, "modality_id", "support")
            object.__setattr__(support, "geotransform", (101.0, 0.1, 0.0, 30.0, 0.0, -0.1))
            mismatch = deepcopy(item)
            object.__setattr__(mismatch, "modalities", (item.modalities[0], support))
            with self.assertRaisesRegex(MaterializationError, "geotransform differs"):
                materialize_spatial_parent(mismatch, benchmark_root=root, canvas_hw=(6, 6))


if __name__ == "__main__":
    unittest.main()
