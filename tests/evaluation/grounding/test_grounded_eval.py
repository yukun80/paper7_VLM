from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.data.grounded.fixture_helpers import no_target_output, target_output

from oa_groundrag.data.rs_general.io import atomic_write_json, atomic_write_jsonl, canonical_json, sha256_file
from oa_groundrag.evaluation.grounding.observations import evaluate_dev
from oa_groundrag.grounding.outputs import (
    REGION_PREDICTION_SCHEMA_VERSION,
    REGION_PROVENANCE_SCHEMA_VERSION,
)


class GroundedEvaluationTest(unittest.TestCase):
    def test_dev_evaluator_checks_counterfactuals_and_never_accepts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            eval_root, output_root = root / "eval", root / "report"
            eval_root.mkdir()
            atomic_write_json(eval_root / "manifest.json", {
                "schema_version": "oa_groundrag.oa_grounded_eval_dev.manifest.v1",
                "train_corpus": {"root": str(root / "train")},
            })
            records = []
            kinds = (
                ("base", "target_present", "full_plus_mask_plus_crop", ["optical_full", "binary_mask", "context_crop"]),
                ("empty", "no_target", "full_plus_mask", ["optical_full", "binary_mask"]),
                ("shift", "target_present", "full_plus_mask_plus_crop", ["optical_full", "binary_mask", "context_crop"]),
                ("context", "target_present", "crop_only", ["context_crop"]),
            )
            for record_id, status, mode, roles in kinds:
                records.append({
                    "record_id": record_id, "parent_id": "parent", "target_status": status,
                    "representation_mode": mode, "formal_model_input_roles": roles,
                })
            atomic_write_jsonl(eval_root / "records.jsonl", records)
            atomic_write_jsonl(eval_root / "counterfactual_groups.jsonl", [{
                "group_id": "group",
                "variants": {
                    "baseline_correct_mask": "base", "empty_mask": "empty",
                    "deterministic_mask_shift": "shift", "context_removal": "context",
                },
            }])
            atomic_write_jsonl(eval_root / "annotation_queue.jsonl", [
                {"record_id": record["record_id"], "asset_identity_sha256": record["record_id"].ljust(64, "0")}
                for record in records
            ])
            manifest_sha = sha256_file(eval_root / "manifest.json")
            predictions = []
            for record in records:
                output = no_target_output() if record["target_status"] == "no_target" else target_output()
                if record["record_id"] == "shift":
                    output["short_summary"] = "The shifted region has a different visible texture."
                if record["record_id"] == "context":
                    output["limitations"] = ["Full-scene context is unavailable in this input."]
                provenance = {
                    "schema_version": REGION_PROVENANCE_SCHEMA_VERSION,
                    "record_id": record["record_id"],
                    "asset_manifest_sha256": manifest_sha,
                    "asset_identity_sha256": record["record_id"].ljust(64, "0"),
                    "representation_mode": record["representation_mode"],
                    "formal_model_input_roles": record["formal_model_input_roles"],
                    "prompt_sha256": "a" * 64,
                    "generation": {"do_sample": False},
                }
                predictions.append({
                    "schema_version": REGION_PREDICTION_SCHEMA_VERSION,
                    "record_id": record["record_id"],
                    "parent_id": "parent",
                    "model_output": output,
                    "generated_text": canonical_json(output),
                    "provenance": provenance,
                    "counterfactual_group_id": "group",
                })
            predictions_path = root / "predictions.jsonl"
            atomic_write_jsonl(predictions_path, predictions)
            with patch(
                "oa_groundrag.evaluation.grounding.observations.validate_eval_dev",
                return_value={"valid": True, "formal_acceptance": False},
            ):
                result = evaluate_dev(
                    eval_root=eval_root,
                    predictions_path=predictions_path,
                    output_root=output_root,
                )
            self.assertFalse(result["formal_acceptance"])
            self.assertEqual(result["automatic_metrics"]["counterfactual_group_completeness"], 1.0)
            self.assertEqual(result["automatic_metrics"]["empty_mask_refusal_rate"], 1.0)
            report = __import__("json").loads((output_root / "report.json").read_text())
            self.assertFalse(report["formal_acceptance"])
            self.assertFalse(report["thresholds_frozen"])
            self.assertFalse(report["scientific_acceptance"])
            self.assertFalse(report["sealed_test_evaluated"])
            self.assertIsNone(report["reference_authority"])


if __name__ == "__main__":
    unittest.main()
