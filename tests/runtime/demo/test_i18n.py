from __future__ import annotations

from types import SimpleNamespace
import unittest

from oa_groundrag.runtime.contracts import RegionSource, UnifiedTask
from oa_groundrag.runtime.demo.i18n import (
    DEFAULT_LOCALE,
    RUN_MODE_SINGLE,
    RUN_MODE_SUITE,
    SUPPORTED_LOCALES,
    TEXT,
    DemoI18nError,
    MessageSpec,
    candidate_choices,
    preview_caption,
    region_source_choices,
    render_messages,
    run_mode_choices,
    task_choices,
    tr,
    validate_catalogs,
)


class DemoI18nTest(unittest.TestCase):
    def test_catalog_keys_and_placeholders_are_strictly_aligned(self) -> None:
        validate_catalogs()
        self.assertEqual(DEFAULT_LOCALE, "zh")
        self.assertEqual(SUPPORTED_LOCALES, ("zh", "en"))
        self.assertEqual(set(TEXT["zh"]), set(TEXT["en"]))
        with self.assertRaises(DemoI18nError):
            tr("fr", "app.header")
        with self.assertRaises(DemoI18nError):
            tr("zh", "missing.key")
        with self.assertRaises(DemoI18nError):
            tr("en", "status.browser.filtered", total=3)
        with self.assertRaises(DemoI18nError):
            tr("en", "status.browser.loaded", unexpected=True)

    def test_message_specs_retranslate_without_changing_stable_parameters(self) -> None:
        specs = (
            MessageSpec.create(
                "status.run.published",
                run_id="demo_fixed",
                success=3,
                failed=0,
                waiting=1,
            ),
            MessageSpec.create("status.run.pending"),
        )
        chinese = render_messages("zh", specs)
        english = render_messages("en", specs)
        self.assertIn("已发布", chinese)
        self.assertIn("published", english)
        self.assertIn("demo_fixed", chinese)
        self.assertIn("demo_fixed", english)
        self.assertEqual(dict(specs[0].params)["success"], 3)

    def test_translated_choices_preserve_runtime_values(self) -> None:
        for helper in (task_choices, region_source_choices, run_mode_choices):
            chinese = helper("zh")
            english = helper("en")
            self.assertEqual(
                [value for _, value in chinese],
                [value for _, value in english],
            )
            self.assertNotEqual(
                [label for label, _ in chinese],
                [label for label, _ in english],
            )
        self.assertEqual(
            [value for _, value in task_choices("en")],
            [task.value for task in UnifiedTask],
        )
        self.assertEqual(
            [value for _, value in region_source_choices("en")],
            [
                RegionSource.OA_AUXSEG_CANDIDATE.value,
                RegionSource.USER_MASK.value,
            ],
        )
        self.assertEqual(
            [value for _, value in run_mode_choices("en")],
            [RUN_MODE_SINGLE, RUN_MODE_SUITE],
        )

    def test_candidate_and_preview_labels_are_presentation_only(self) -> None:
        payload = {
            "options": [{
                "kind": "CANDIDATE",
                "token": "dss_fixed::CANDIDATE::2",
                "candidate_id": 2,
                "area_pixels": 19,
                "confidence": 0.875,
                "overlay_path": "/tmp/overlay.png",
            }, {
                "kind": "EXPLICIT_GLOBAL",
                "token": "dss_fixed::EXPLICIT_GLOBAL",
                "candidate_id": None,
                "area_pixels": None,
                "confidence": None,
                "overlay_path": None,
            }],
        }
        chinese = candidate_choices("zh", payload)
        english = candidate_choices("en", payload)
        self.assertEqual([value for _, value in chinese], [value for _, value in english])
        self.assertIn("显式使用", chinese[1][0])
        self.assertIn("Use OA-AuxSeg", english[1][0])
        with self.assertRaises(DemoI18nError):
            candidate_choices("en", {"options": [{"kind": "UNKNOWN", "token": "x"}]})

        preview = SimpleNamespace(
            is_auxiliary=True,
            modality="dem",
            channel_name="elevation",
            valid_fraction=0.625,
        )
        self.assertIn("空间专家输入预览", preview_caption("zh", preview))
        self.assertIn("Spatial Expert Input Preview", preview_caption("en", preview))
        self.assertIn("dem / elevation", preview_caption("zh", preview))


if __name__ == "__main__":
    unittest.main()
