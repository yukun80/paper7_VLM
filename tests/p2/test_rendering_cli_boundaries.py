"""P2 deterministic rendering, CLI and greenfield-boundary tests."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import torch

from sami_gsd.cli import build_parser
from sami_gsd.model.rendering import array_to_hwc, render_modality_rgb, resize_to_pixel_budget
from tests.p2.conftest import REPOSITORY_ROOT, loaded_parent


class RenderingCliBoundaryTests(unittest.TestCase):
    """Verify deterministic preprocessing and absence of legacy runtime coupling."""

    def test_declared_channel_axis_is_explicit_and_ambiguity_fails(self) -> None:
        """The loader uses band metadata and never guesses an ambiguous layout."""

        chw = np.zeros((3, 10, 12), dtype=np.float32)
        hwc = array_to_hwc(chw, band_count=3, modality_id="view")
        self.assertEqual(hwc.shape, (10, 12, 3))
        with self.assertRaisesRegex(ValueError, "cannot be uniquely resolved"):
            array_to_hwc(np.zeros((3, 8, 3)), band_count=3, modality_id="ambiguous")

    def test_rendering_zeros_invalid_pixels_and_preserves_bool_validity(self) -> None:
        """Invalid/nodata pixels contribute neither display values nor valid evidence."""

        modality = loaded_parent(
            (("opt-ref", "optical", "present_valid"),), reference_id="opt-ref"
        ).record.modalities[0]
        array = np.arange(32 * 48 * 3, dtype=np.float32).reshape(32, 48, 3)
        valid = np.ones((32, 48), dtype=bool)
        valid[0, 0] = False
        image, mask = render_modality_rgb(array, valid, modality)
        self.assertEqual(image.getpixel((0, 0)), (0, 0, 0))
        self.assertEqual(mask.dtype, torch.bool)
        self.assertFalse(bool(mask[0, 0]))

    def test_pixel_budget_downscales_without_upscaling(self) -> None:
        """Reference/support budgets are hard upper bounds and retain valid pixels."""

        loaded = loaded_parent(
            (("opt-ref", "optical", "present_valid"),), reference_id="opt-ref"
        ).views["opt-ref"]
        same_image, same_valid = resize_to_pixel_budget(loaded.image, loaded.valid_mask, 10_000)
        self.assertEqual(same_image.size, loaded.image.size)
        self.assertTrue(torch.equal(same_valid, loaded.valid_mask))
        small_image, small_valid = resize_to_pixel_budget(loaded.image, loaded.valid_mask, 100)
        self.assertLessEqual(small_image.width * small_image.height, 100)
        self.assertTrue(bool(small_valid.any()))

    def test_unique_cli_registers_one_forward_smoke(self) -> None:
        """The root CLI exposes the P2 command without importing a legacy launcher."""

        arguments = build_parser().parse_args(
            [
                "model",
                "smoke",
                "--config",
                "configs/model_sami.yaml",
                "--parent-id",
                "p",
                "--output",
                "report.json",
            ]
        )
        self.assertEqual(arguments.model_command, "smoke")
        self.assertEqual(arguments.device, "cuda:0")

    def test_greenfield_model_has_no_legacy_runtime_imports(self) -> None:
        """P2 source/config never depends on SANE/QMEF/PMRD/MGRR or qpsalm_seg."""

        paths = sorted((REPOSITORY_ROOT / "src" / "sami_gsd" / "model").glob("*.py"))
        paths += [REPOSITORY_ROOT / "src" / "sami_gsd" / "contracts" / "model.py"]
        forbidden = (
            "qpsalm_seg",
            "SEG_Multi-Source_Landslides",
            "from .controllers",
            "import controllers",
            "SANE",
            "QMEF",
            "PMRD",
            "MGRR",
        )
        for path in paths:
            text = path.read_text(encoding="utf-8")
            for marker in forbidden:
                with self.subTest(path=path.name, marker=marker):
                    self.assertNotIn(marker, text)


if __name__ == "__main__":
    unittest.main()
