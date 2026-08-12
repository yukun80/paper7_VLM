from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from oa_groundrag.landslide_evidence.contracts import LandslideEvidenceError
from oa_groundrag.landslide_evidence.region_contracts import parent_identity
from oa_groundrag.landslide_evidence.region_pipeline import RegionBenchmarkAccess
from oa_groundrag.phase4.artifacts import AtomicArtifactDirectory
from oa_groundrag.phase4.evidence import (
    binary_mask_array,
    boundary_contrast_proxy,
    deterministic_shift_mask,
    render_binary_mask,
    render_context_crop,
)
from oa_groundrag.phase4.errors import ContractError


class RegionPipelineTest(unittest.TestCase):
    def test_binary_mask_mode_values_and_bool_roundtrip(self) -> None:
        mask = np.zeros((8, 9), dtype=bool)
        mask[2:6, 3:7] = True
        image = render_binary_mask(mask)
        self.assertEqual(image.mode, "L")
        self.assertEqual(set(np.unique(np.asarray(image)).tolist()), {0, 255})
        self.assertTrue(np.array_equal(binary_mask_array(image), mask))

    def test_image_mask_crop_shapes_and_reconstruction(self) -> None:
        optical = Image.fromarray(np.arange(10 * 12 * 3, dtype=np.uint8).reshape(10, 12, 3))
        mask = np.zeros((10, 12), dtype=bool)
        mask[3:7, 4:9] = True
        first, window = render_context_crop(optical, mask, margin_ratio=0.15)
        second, second_window = render_context_crop(optical, mask, margin_ratio=0.15)
        self.assertEqual(window, second_window)
        self.assertTrue(np.array_equal(np.asarray(first), np.asarray(second)))
        self.assertTrue(np.array_equal(np.asarray(first), np.asarray(optical.crop(window))))

    def test_shift_is_deterministic_low_overlap_and_in_bounds(self) -> None:
        mask = np.zeros((20, 24), dtype=bool)
        mask[7:12, 9:15] = True
        first, first_meta = deterministic_shift_mask(mask)
        second, second_meta = deterministic_shift_mask(mask)
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(first_meta, second_meta)
        self.assertEqual(int(first.sum()), int(mask.sum()))
        self.assertLessEqual(first_meta["mask_iou_with_gt"], 0.01)
        self.assertEqual(first.shape, mask.shape)

    def test_boundary_proxy_is_finite(self) -> None:
        optical = Image.new("RGB", (10, 10), (30, 60, 90))
        mask = np.zeros((10, 10), dtype=bool)
        mask[2:8, 2:8] = True
        self.assertGreaterEqual(boundary_contrast_proxy(optical, mask), 0.0)

    def test_unknown_parent_is_not_inferred_from_filename(self) -> None:
        parent, status = parent_identity({
            "sample_id": "looks_like_region_42", "source_group_id": None, "group_status": "unknown",
        })
        self.assertEqual(parent, "looks_like_region_42")
        self.assertEqual(status.value, "unknown")

    def test_test_access_is_rejected_before_payload_read(self) -> None:
        with self.assertRaises(LandslideEvidenceError) as raised:
            RegionBenchmarkAccess(Path("/does/not/matter"), {}, allowed_split="test")
        self.assertEqual(raised.exception.code, "SPLIT_FORBIDDEN")

    def test_atomic_output_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "existing"
            target.mkdir()
            with self.assertRaises(ContractError):
                with AtomicArtifactDirectory(target):
                    pass


if __name__ == "__main__":
    unittest.main()
