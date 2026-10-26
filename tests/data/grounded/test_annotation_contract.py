from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests.data.grounded.fixture_helpers import target_output

from oa_groundrag.data.grounded.annotation.queue import validate_annotation_row
from oa_groundrag.data.grounded.contracts import LandslideEvidenceError
from oa_groundrag.data.grounded.region_contracts import ANNOTATION_SCHEMA


def annotation_row() -> dict:
    return {
        "schema_version": ANNOTATION_SCHEMA,
        "record_id": "record-1",
        "asset_identity_sha256": "a" * 64,
        "annotation_version": "region_annotation.v1",
        "annotation_status": "reviewed",
        "annotator_id": "annotator-a",
        "reviewer_id": "reviewer-b",
        "adjudicator_id": None,
        "adjudication_status": "not_required",
        "description": target_output(),
        "scores": {
            "target_appearance_accuracy": "correct",
            "target_morphology_accuracy": "partially_correct",
            "surrounding_environment_accuracy": "correct",
            "region_context_relation_accuracy": "correct",
            "confuser_recognition": "not_applicable",
            "evidence_sufficiency_accuracy": "correct",
            "expert_factuality": "correct",
        },
        "unsupported_claims": [],
    }


class AnnotationContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.queue = {
            "record_id": "record-1",
            "asset_identity_sha256": "a" * 64,
            "target_status": "target_present",
        }

    def test_reviewed_annotation_is_valid(self) -> None:
        row = annotation_row()
        self.assertEqual(
            validate_annotation_row(row, queue_row=self.queue, location="annotation"),
            row,
        )

    def test_reviewer_must_differ_from_annotator(self) -> None:
        row = annotation_row()
        row["reviewer_id"] = row["annotator_id"]
        with self.assertRaises(LandslideEvidenceError):
            validate_annotation_row(row, queue_row=self.queue, location="annotation")

    def test_unknown_field_and_asset_drift_are_rejected(self) -> None:
        unknown = annotation_row() | {"extra": True}
        with self.assertRaises(LandslideEvidenceError):
            validate_annotation_row(unknown, queue_row=self.queue, location="annotation")
        drift = annotation_row()
        drift["asset_identity_sha256"] = "b" * 64
        with self.assertRaises(LandslideEvidenceError):
            validate_annotation_row(drift, queue_row=self.queue, location="annotation")

    def test_queued_annotation_cannot_be_imported(self) -> None:
        row = annotation_row()
        row["annotation_status"] = "queued"
        with self.assertRaises(LandslideEvidenceError):
            validate_annotation_row(row, queue_row=self.queue, location="annotation")


if __name__ == "__main__":
    unittest.main()
