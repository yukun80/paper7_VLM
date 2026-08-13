from __future__ import annotations

import json
import unittest
from pathlib import Path

from tests.data.grounded.fixture_helpers import no_target_output, target_output

from oa_groundrag.data.grounded.contracts import load_config
from oa_groundrag.data.grounded.region_contracts import RepresentationMode, load_region_corpus_config
from oa_groundrag.vlm.errors import ContractError, ReasonCode
from oa_groundrag.grounding.outputs import (
    detect_forbidden_region_claims,
    parse_region_model_output,
    serialize_region_model_output,
)


REPO = Path(__file__).resolve().parents[3]


class RegionContractTest(unittest.TestCase):
    def test_live_configs_are_strict_and_frozen(self) -> None:
        corpus = load_region_corpus_config(
            REPO / "configs/grounding/region_corpus_train_v1.yaml"
        )
        self.assertEqual(corpus.stage4a_selection.sample_count, 500)
        benchmark_stage4 = REPO.parent / "benchmark/oa_grounded_stage4_v1"
        self.assertEqual(
            corpus.output_root,
            benchmark_stage4
            / "region_corpus/mask_grounded_region_corpus_train_v1_500",
        )
        self.assertEqual(RepresentationMode.FULL_PLUS_MASK_PLUS_CROP.value, "full_plus_mask_plus_crop")

    def test_retired_eval_config_is_not_shipped(self) -> None:
        self.assertFalse(
            (REPO / "configs/grounding/oa_grounded_eval_dev_v1.yaml").exists()
        )

    def test_v1_config_contract_still_loads(self) -> None:
        config = load_config(REPO / "configs/grounding/pilot_500.yaml")
        self.assertEqual(config.sample_count, 500)

    def test_region_output_round_trip(self) -> None:
        parsed = parse_region_model_output(target_output())
        self.assertEqual(parse_region_model_output(serialize_region_model_output(parsed)), parsed)

    def test_no_target_contract(self) -> None:
        parsed = parse_region_model_output(no_target_output())
        self.assertEqual(parsed.target_status.value, "no_target")
        invalid = no_target_output()
        invalid["target_appearance"]["tone"] = "brown"
        with self.assertRaises(ContractError):
            parse_region_model_output(invalid)

    def test_unknown_duplicate_and_nonfinite_are_rejected(self) -> None:
        unknown = target_output() | {"bbox": [1, 2, 3, 4]}
        with self.assertRaises(ContractError) as raised:
            parse_region_model_output(unknown)
        self.assertEqual(raised.exception.code, ReasonCode.UNKNOWN_FIELD)
        text = json.dumps(target_output())
        duplicate = text[:-1] + ',"schema_version":"x"}'
        with self.assertRaises(ContractError):
            parse_region_model_output(duplicate)
        with self.assertRaises(ContractError):
            parse_region_model_output(text[:-1] + ',"x":NaN}')

    def test_forbidden_claim_and_uncertain_limitation(self) -> None:
        invalid = target_output()
        invalid["short_summary"] = "This area was triggered by rainfall."
        with self.assertRaises(ContractError) as raised:
            parse_region_model_output(invalid)
        self.assertEqual(raised.exception.code, ReasonCode.FORBIDDEN_CLAIM)
        self.assertEqual(
            detect_forbidden_region_claims("Insufficient evidence; cannot determine whether rainfall triggered it."),
            (),
        )


if __name__ == "__main__":
    unittest.main()
