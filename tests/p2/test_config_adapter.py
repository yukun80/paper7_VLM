"""Frozen P2 config, sensor-card and active-view adapter tests."""

from __future__ import annotations

import unittest

from pydantic import ValidationError

from sami_gsd.contracts.model import SamiModelConfig
from sami_gsd.model.sensor_adapter import SensorAwareMultiImageAdapter
from tests.p2.conftest import loaded_parent, model_config


PRESENT = "present_valid"


class ConfigAdapterTests(unittest.TestCase):
    """Exercise every required modality composition before real Qwen loading."""

    def setUp(self) -> None:
        self.config = model_config()
        self.adapter = SensorAwareMultiImageAdapter(self.config)

    def test_frozen_profiles_and_backend(self) -> None:
        """The public config registers only the official backend and exact S/M budgets."""

        self.assertEqual(self.config.model.backend, "qwen3_vl_official")
        self.assertEqual(self.config.model.family, "Qwen3-VL-2B")
        self.assertEqual(
            [(item.profile, item.reference_max_pixels, item.support_max_pixels, item.max_views)
             for item in self.config.pixel_budgets],
            [("S", 512**2, 384**2, 4), ("M", 768**2, 448**2, 6)],
        )

    def test_config_rejects_hidden_profile_change(self) -> None:
        """A familiar profile name cannot hide a different pixel policy."""

        payload = self.config.model_dump(mode="json")
        payload["pixel_budgets"][0]["max_views"] = 5
        with self.assertRaises(ValidationError):
            SamiModelConfig.model_validate(payload)

    def test_optical_sar_and_terrain_only_parents_prepare(self) -> None:
        """Each required single-family parent keeps its own canonical reference identity."""

        cases = (("optical", "optical-ref"), ("sar", "sar-ref"), ("dem", "dem-ref"))
        for family, modality_id in cases:
            with self.subTest(family=family):
                parent = loaded_parent(((modality_id, family, PRESENT),), reference_id=modality_id)
                batch = self.adapter.prepare((parent,), ((modality_id,),))
                self.assertEqual(batch.parents[0].view_ids, (modality_id,))
                self.assertEqual(batch.parents[0].views[0].role, "reference")
                self.assertEqual(batch.parents[0].views[0].sensor_card.family, family)

    def test_multi_view_order_is_stable_under_both_input_permutations(self) -> None:
        """Caller list/dict order cannot change reference/support view identity."""

        specs = (
            ("dem-z", "dem", PRESENT),
            ("sar-z", "sar", PRESENT),
            ("opt-ref", "optical", PRESENT),
            ("sar-a", "sar", PRESENT),
        )
        first = loaded_parent(specs, reference_id="opt-ref")
        second = loaded_parent(
            specs,
            reference_id="opt-ref",
            view_insertion_order=("sar-a", "opt-ref", "dem-z", "sar-z"),
        )
        order_a = self.adapter.prepare(
            (first,), (("dem-z", "sar-z", "opt-ref", "sar-a"),)
        ).parents[0].view_ids
        order_b = self.adapter.prepare(
            (second,), (("sar-a", "opt-ref", "sar-z", "dem-z"),)
        ).parents[0].view_ids
        self.assertEqual(order_a, ("opt-ref", "sar-a", "sar-z", "dem-z"))
        self.assertEqual(order_a, order_b)

    def test_dropout_missing_and_zero_valid_never_create_visual_views(self) -> None:
        """Inactive, missing and zero-valid states stay distinct with no token/card leakage."""

        parent = loaded_parent(
            (
                ("opt-ref", "optical", PRESENT),
                ("sar-dropped", "sar", PRESENT),
                ("dem-missing", "dem", "missing"),
                ("slope-empty", "slope", "present_zero_valid"),
            ),
            reference_id="opt-ref",
        )
        prepared = self.adapter.prepare(
            (parent,), (("opt-ref", "dem-missing", "slope-empty"),)
        ).parents[0]
        self.assertEqual(prepared.view_ids, ("opt-ref",))
        reasons = {item.modality_id: item.reason for item in prepared.excluded_modalities}
        self.assertEqual(
            reasons,
            {
                "dem-missing": "missing",
                "sar-dropped": "inactive_dropout",
                "slope-empty": "present_zero_valid",
            },
        )
        prompt_payload = prepared.views[0].sensor_card.payload()
        self.assertNotIn("dataset", prompt_payload)
        self.assertNotIn("split", prompt_payload)
        self.assertNotIn("normalization", prompt_payload)

    def test_reference_dropout_and_silent_truncation_are_rejected(self) -> None:
        """The adapter never invents a reference or silently discards excess active views."""

        pair = loaded_parent(
            (("opt-ref", "optical", PRESENT), ("sar-a", "sar", PRESENT)),
            reference_id="opt-ref",
        )
        with self.assertRaisesRegex(ValueError, "must retain canonical reference"):
            self.adapter.prepare((pair,), (("sar-a",),))

        specifications = tuple(
            [("opt-ref", "optical", PRESENT)]
            + [(f"sar-{index}", "sar", PRESENT) for index in range(4)]
        )
        crowded = loaded_parent(specifications, reference_id="opt-ref")
        with self.assertRaisesRegex(ValueError, "exceeding Profile S"):
            self.adapter.prepare((crowded,), (tuple(item[0] for item in specifications),))


if __name__ == "__main__":
    unittest.main()
