from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from oa_groundrag.phase4.contracts import (
    MODEL_OUTPUT_SCHEMA_VERSION,
    AlignmentStatus,
    AuxiliaryView,
    MaskMode,
    RegionCandidate,
    RegionInventory,
    SelectionMode,
    SelectionRequest,
    TargetStatus,
)
from oa_groundrag.phase4.evidence import (
    EvidenceBuilder,
    deterministic_mask_facts,
)
from oa_groundrag.phase4.errors import (
    ContractError,
    EvidenceError,
    ReasonCode,
    SelectionError,
)
from oa_groundrag.phase4.messages import build_mask_grounded_messages
from oa_groundrag.phase4.outputs import parse_model_output
from oa_groundrag.phase4.regions import RegionSelector


def inventory() -> RegionInventory:
    first = np.zeros((12, 12), dtype=np.uint8)
    first[1:4, 1:4] = 1
    second = np.zeros((12, 12), dtype=np.uint8)
    second[7:11, 7:11] = 1
    global_mask = first | second
    return RegionInventory(
        sample_id="sample",
        mask_mode=MaskMode.GT_MASK,
        global_mask=global_mask,
        candidates=(
            RegionCandidate("r1", first, 0.9),
            RegionCandidate("r2", second, 0.8),
        ),
        source_identity="fixture",
    )


class RegionSelectorTests(unittest.TestCase):
    def test_all_deterministic_selection_modes(self) -> None:
        selector = RegionSelector()
        value = inventory()
        for mode in (SelectionMode.GLOBAL, SelectionMode.ALL):
            selected = selector.select(value, SelectionRequest(mode))
            self.assertEqual(selected.region_id, "global")
            np.testing.assert_array_equal(selected.mask, value.global_mask)
        selected = selector.select(
            value,
            SelectionRequest(SelectionMode.REGION_ID, region_id="r2"),
        )
        self.assertEqual(selected.region_id, "r2")
        selected = selector.select(
            value,
            SelectionRequest(
                SelectionMode.BBOX,
                bbox_xyxy=(0.0, 0.0, 5.0, 5.0),
            ),
        )
        self.assertEqual(selected.region_id, "r1")
        selected = selector.select(
            value,
            SelectionRequest(SelectionMode.CLICK, click_xy=(8.0, 8.0)),
        )
        self.assertEqual(selected.region_id, "r2")
        selected = selector.select(
            value,
            SelectionRequest(SelectionMode.AREA_RANK, area_rank=1),
        )
        self.assertEqual(selected.region_id, "r2")
        selected = selector.select(
            value,
            SelectionRequest(SelectionMode.LOCATION, location="northwest"),
        )
        self.assertEqual(selected.region_id, "r1")
        selected = selector.select(
            value,
            SelectionRequest(
                SelectionMode.NUMBERED_OVERLAY,
                numbered_response='{"region_id":"r2"}',
                numbered_selector_frozen=True,
            ),
        )
        self.assertEqual(selected.region_id, "r2")
        with self.assertRaises(ContractError) as caught:
            SelectionRequest(
                SelectionMode.NUMBERED_OVERLAY,
                numbered_response='{"region_id":"r2"}',
                numbered_selector_frozen=False,
            )
        self.assertEqual(
            caught.exception.code,
            ReasonCode.MODEL_IDENTITY_MISMATCH,
        )

    def test_ambiguity_wrong_region_and_no_match_are_explicit(self) -> None:
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[1:4, 1:4] = 1
        value = RegionInventory(
            "tie",
            MaskMode.GT_MASK,
            mask,
            (
                RegionCandidate("a", mask),
                RegionCandidate("b", mask),
            ),
            "fixture",
        )
        with self.assertRaises(SelectionError) as caught:
            RegionSelector().select(
                value,
                SelectionRequest(
                    SelectionMode.BBOX,
                    bbox_xyxy=(1.0, 1.0, 4.0, 4.0),
                ),
            )
        self.assertEqual(caught.exception.code, ReasonCode.AMBIGUOUS_REGION)
        with self.assertRaises(SelectionError) as caught:
            RegionSelector().select(
                inventory(),
                SelectionRequest(
                    SelectionMode.REGION_ID,
                    region_id="wrong",
                ),
            )
        self.assertEqual(caught.exception.code, ReasonCode.REGION_ID_INVALID)
        with self.assertRaises(SelectionError) as caught:
            RegionSelector().select(
                inventory(),
                SelectionRequest(
                    SelectionMode.CLICK,
                    click_xy=(5.0, 5.0),
                ),
            )
        self.assertEqual(caught.exception.code, ReasonCode.REGION_NOT_FOUND)

    def test_empty_global_is_semantic_no_target(self) -> None:
        value = RegionInventory(
            "empty",
            MaskMode.GT_MASK,
            np.zeros((6, 7), dtype=np.uint8),
            (),
            "fixture",
        )
        selected = RegionSelector().select(
            value,
            SelectionRequest(SelectionMode.GLOBAL),
        )
        self.assertEqual(selected.target_status, TargetStatus.NO_TARGET)
        self.assertIsNone(selected.mask)
        self.assertIsNone(selected.region_id)

    def test_candidate_cannot_escape_global_or_create_external_inventory(self) -> None:
        global_mask = np.zeros((5, 5), dtype=np.uint8)
        candidate = np.zeros((5, 5), dtype=np.uint8)
        candidate[2, 2] = 1
        with self.assertRaises(ContractError) as caught:
            RegionInventory(
                "bad",
                MaskMode.GT_MASK,
                global_mask,
                (RegionCandidate("r", candidate),),
                "fixture",
            )
        self.assertEqual(caught.exception.code, ReasonCode.MASK_INVALID)
        with self.assertRaises(ContractError) as caught:
            RegionInventory(
                "external",
                MaskMode.EXTERNAL_GENERIC,
                candidate,
                (),
                "fixture",
            )
        self.assertEqual(
            caught.exception.code,
            ReasonCode.EXTERNAL_MASK_FORBIDDEN,
        )


class EvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_geometry_overlay_crop_and_auxiliary_limits(self) -> None:
        selected = RegionSelector().select(
            inventory(),
            SelectionRequest(SelectionMode.REGION_ID, region_id="r1"),
        )
        auxiliary = AuxiliaryView(
            evidence_id="ev_dem",
            name="dem",
            image=np.arange(144, dtype=np.float32).reshape(12, 12),
            alignment=AlignmentStatus.UNKNOWN,
            coverage=0.5,
            unit=None,
            sign_convention=None,
        )
        result = EvidenceBuilder().build(
            optical_image=Image.new("RGB", (12, 12), (30, 60, 90)),
            selected=selected,
            mask_mode=MaskMode.GT_MASK,
            output_root=self.base / "evidence",
            auxiliary_views=(auxiliary,),
        )
        facts = result.bundle.deterministic_facts
        self.assertEqual(
            facts["bbox_xyxy_pixel_half_open"],
            [1, 1, 4, 4],
        )
        self.assertEqual(facts["area_pixels"], 9)
        self.assertEqual(facts["fragment_connectivity"], 8)
        self.assertEqual(facts["perimeter_connectivity"], 4)
        self.assertEqual(facts["perimeter_pixels"], 12)
        self.assertEqual(facts["covariance_elongation"], 1.0)
        self.assertTrue((result.root / "mask_overlay.png").is_file())
        self.assertTrue((result.root / "context_crop.png").is_file())
        self.assertIn("AUX_ALIGNMENT_UNKNOWN:dem", result.bundle.limitations)
        self.assertIn("AUX_COVERAGE_INSUFFICIENT:dem", result.bundle.limitations)
        self.assertIn("AUX_UNIT_UNKNOWN:dem", result.bundle.limitations)
        self.assertFalse(
            result.bundle.auxiliary_metadata[0][
                "quantitative_claims_allowed"
            ]
        )
        self.assertEqual(
            result.bundle.excluded_internal_signals,
            ("attention", "modality_quality_weight", "region_feature"),
        )

    def test_no_target_has_no_overlay_crop_or_geometry(self) -> None:
        empty = RegionInventory(
            "empty",
            MaskMode.GT_MASK,
            np.zeros((6, 6), dtype=np.uint8),
            (),
            "fixture",
        )
        selected = RegionSelector().select(
            empty,
            SelectionRequest(SelectionMode.GLOBAL),
        )
        result = EvidenceBuilder().build(
            optical_image=Image.new("RGB", (6, 6)),
            selected=selected,
            mask_mode=MaskMode.GT_MASK,
            output_root=self.base / "no-target",
        )
        self.assertEqual(result.bundle.deterministic_facts, {})
        self.assertEqual(
            [asset.role for asset in result.bundle.assets],
            ["optical_full"],
        )
        messages = build_mask_grounded_messages(
            result.bundle,
            evidence_root=result.root,
            instruction="Describe the selected target.",
        )
        text = messages[0]["content"][-1]["text"]
        self.assertIn("no target", text.lower())
        self.assertIn("do not imply target presence", text.lower())

    def test_rag_and_auxiliary_overflow_fail_instead_of_truncating(self) -> None:
        selected = RegionSelector().select(
            inventory(),
            SelectionRequest(SelectionMode.GLOBAL),
        )
        view = AuxiliaryView(
            "ev_a",
            "a",
            Image.new("RGB", (12, 12)),
            AlignmentStatus.REGISTERED,
            1.0,
            "m",
            "positive_up",
        )
        with self.assertRaises(EvidenceError) as caught:
            EvidenceBuilder(max_auxiliary_views=1).build(
                optical_image=Image.new("RGB", (12, 12)),
                selected=selected,
                mask_mode=MaskMode.GT_MASK,
                output_root=self.base / "too-many",
                auxiliary_views=(view, view),
            )
        self.assertEqual(
            caught.exception.code,
            ReasonCode.AUXILIARY_LIMIT_EXCEEDED,
        )
        with self.assertRaises(EvidenceError) as caught:
            EvidenceBuilder().build(
                optical_image=Image.new("RGB", (12, 12)),
                selected=selected,
                mask_mode=MaskMode.GT_MASK,
                output_root=self.base / "rag",
                rag_context=("forbidden",),
            )
        self.assertEqual(caught.exception.code, ReasonCode.RAG_FORBIDDEN)

    def test_mask_swap_changes_program_facts(self) -> None:
        value = inventory()
        first = RegionSelector().select(
            value,
            SelectionRequest(SelectionMode.REGION_ID, region_id="r1"),
        )
        second = RegionSelector().select(
            value,
            SelectionRequest(SelectionMode.REGION_ID, region_id="r2"),
        )
        self.assertNotEqual(
            deterministic_mask_facts(first.mask)["area_pixels"],
            deterministic_mask_facts(second.mask)["area_pixels"],
        )


class StructuredOutputTests(unittest.TestCase):
    def valid(self) -> dict:
        return {
            "schema_version": MODEL_OUTPUT_SCHEMA_VERSION,
            "target_status": "target_present",
            "description": "A visible region is described.",
            "answer": None,
            "evidence_sufficiency": "sufficient",
            "claims": [
                {
                    "text": "The region is visible.",
                    "evidence_ids": ["ev_overlay"],
                }
            ],
            "limitations": [],
        }

    def test_strict_output_and_evidence_references(self) -> None:
        output = parse_model_output(
            self.valid(),
            valid_evidence_ids={"ev_overlay"},
        )
        self.assertEqual(output.target_status, TargetStatus.TARGET_PRESENT)
        wrong = self.valid()
        wrong["claims"][0]["evidence_ids"] = ["unknown"]
        with self.assertRaises(ContractError) as caught:
            parse_model_output(wrong, valid_evidence_ids={"ev_overlay"})
        self.assertEqual(
            caught.exception.code,
            ReasonCode.EVIDENCE_REFERENCE_INVALID,
        )

    def test_geometry_fields_duplicate_keys_and_nonfinite_are_rejected(self) -> None:
        geometry = self.valid()
        geometry["bbox"] = [1, 2, 3, 4]
        with self.assertRaises(ContractError) as caught:
            parse_model_output(geometry, valid_evidence_ids={"ev_overlay"})
        self.assertEqual(caught.exception.code, ReasonCode.FORBIDDEN_MODEL_FACT)
        geometry_text = self.valid()
        geometry_text["claims"][0]["text"] = "The centroid is at (2, 3)."
        with self.assertRaises(ContractError) as caught:
            parse_model_output(
                geometry_text,
                valid_evidence_ids={"ev_overlay"},
            )
        self.assertEqual(caught.exception.code, ReasonCode.FORBIDDEN_MODEL_FACT)
        duplicate = (
            '{"schema_version":"'
            + MODEL_OUTPUT_SCHEMA_VERSION
            + '","schema_version":"x"}'
        )
        with self.assertRaises(ContractError):
            parse_model_output(duplicate, valid_evidence_ids=set())
        nonfinite = json.dumps(self.valid()).replace(
            '"answer": null',
            '"answer": NaN',
        )
        with self.assertRaises(ContractError):
            parse_model_output(nonfinite, valid_evidence_ids={"ev_overlay"})


if __name__ == "__main__":
    unittest.main()
