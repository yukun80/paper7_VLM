"""Canonical Parent v3 and task-view contract tests."""

from __future__ import annotations

import json
import unittest
from copy import deepcopy
from pathlib import Path

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError as JsonSchemaValidationError
from pydantic import ValidationError

from sami_gsd.contracts.canonical import CanonicalParentV3, TaskViewV3
from tests.p1.conftest import canonical_parent_payload, task_view_payload


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class ContractTests(unittest.TestCase):
    """Validate static and Python-owned Canonical Benchmark contracts."""

    def test_json_schemas_accept_canonical_fixtures(self) -> None:
        """Static draft-2020-12 schemas accept their matching fixtures."""

        cases = [
            ("canonical_parent_v3.schema.json", canonical_parent_payload()),
            ("task_view_v3.schema.json", task_view_payload()),
        ]
        for schema_name, payload in cases:
            with self.subTest(schema=schema_name):
                schema = json.loads((REPOSITORY_ROOT / "schemas" / schema_name).read_text(encoding="utf-8"))
                Draft202012Validator.check_schema(schema)
                Draft202012Validator(schema).validate(payload)

    def test_python_contracts_accept_canonical_fixtures(self) -> None:
        """Pydantic and JSON contracts share their public field names."""

        parent = CanonicalParentV3.model_validate(canonical_parent_payload())
        task = TaskViewV3.model_validate(task_view_payload())
        self.assertEqual(parent.reference_canvas.coordinate_space, "reference_pixel_half_open")
        self.assertEqual(task.task_type, "t2_referring")

    def test_component_license_bound_description_schema_is_draft_2020_12(self) -> None:
        """The runtime description index has one valid, strict static schema."""

        schema = json.loads(
            (REPOSITORY_ROOT / "schemas" / "canonical_description_v2.schema.json").read_text(
                encoding="utf-8"
            )
        )
        Draft202012Validator.check_schema(schema)
        self.assertEqual(
            schema["properties"]["schema_version"]["const"],
            "sami_canonical_description_v2_component_license_bound",
        )

    def test_contracts_reject_extra_top_level_fields(self) -> None:
        """Public record types reject silent compatibility extensions."""

        parent = canonical_parent_payload()
        parent["legacy_protocol"] = "v2"
        task = task_view_payload()
        task["modality_condition_copy"] = True
        with self.assertRaises(ValidationError):
            CanonicalParentV3.model_validate(parent)
        with self.assertRaises(ValidationError):
            TaskViewV3.model_validate(task)

        parent_schema = json.loads(
            (REPOSITORY_ROOT / "schemas" / "canonical_parent_v3.schema.json").read_text(encoding="utf-8")
        )
        task_schema = json.loads((REPOSITORY_ROOT / "schemas" / "task_view_v3.schema.json").read_text(encoding="utf-8"))
        with self.assertRaises(JsonSchemaValidationError):
            Draft202012Validator(parent_schema).validate(parent)
        with self.assertRaises(JsonSchemaValidationError):
            Draft202012Validator(task_schema).validate(task)

    def test_static_schema_rejects_training_eligible_unknown_license(self) -> None:
        """The published JSON schema carries the fail-closed license rule."""

        parent = canonical_parent_payload()
        parent["license"]["license_status"] = "unknown"
        parent["license"]["license_name"] = "unknown"
        schema = json.loads(
            (REPOSITORY_ROOT / "schemas" / "canonical_parent_v3.schema.json").read_text(encoding="utf-8")
        )
        with self.assertRaises(JsonSchemaValidationError):
            Draft202012Validator(schema).validate(parent)

    def test_machine_paths_and_parent_traversal_are_rejected(self) -> None:
        """Canonical records contain portable logical paths only."""

        for invalid_path in ("/tmp/mask.npy", "../mask.npy", "assets//mask.npy", "assets\\mask.npy"):
            with self.subTest(path=invalid_path):
                parent = canonical_parent_payload()
                parent["annotations"]["global_landslide_mask"]["path"] = invalid_path
                with self.assertRaises(ValidationError):
                    CanonicalParentV3.model_validate(parent)

    def test_half_open_boxes_and_reference_modality_are_checked(self) -> None:
        """Reject reversed boxes and an undeclared reference modality."""

        reversed_box = canonical_parent_payload()
        reversed_box["annotations"]["referring_regions"][0]["bbox_half_open"] = [20, 5, 4, 24]
        with self.assertRaises(ValidationError):
            CanonicalParentV3.model_validate(reversed_box)

        missing_reference = canonical_parent_payload()
        missing_reference["reference_canvas"]["reference_modality_id"] = "not-declared"
        with self.assertRaises(ValidationError):
            CanonicalParentV3.model_validate(missing_reference)

    def test_missing_and_zero_valid_are_not_interchangeable(self) -> None:
        """A zero-valid present asset cannot masquerade as missing."""

        invalid_missing = deepcopy(canonical_parent_payload())
        modality = invalid_missing["modalities"][0]
        modality["availability_status"] = "missing"
        modality["valid_coverage"] = 0.0
        with self.assertRaises(ValidationError):
            CanonicalParentV3.model_validate(invalid_missing)

        zero_valid = deepcopy(canonical_parent_payload())
        modality = zero_valid["modalities"][0]
        modality["availability_status"] = "present_zero_valid"
        modality["valid_coverage"] = 0.0
        result = CanonicalParentV3.model_validate(zero_valid)
        self.assertEqual(result.modalities[0].availability_status, "present_zero_valid")

    def test_global_only_support_cannot_expose_pixel_transforms(self) -> None:
        """Unregistered support may remain global but cannot claim pixel evidence."""

        global_only = deepcopy(canonical_parent_payload())
        modality = deepcopy(global_only["modalities"][0])
        modality["modality_id"] = "sar-global-only"
        modality["family"] = "sar"
        modality["alignment_status"] = "global_only"
        modality["source_to_reference_transform"] = None
        modality["reference_to_source_transform"] = None
        global_only["modalities"].append(modality)
        result = CanonicalParentV3.model_validate(global_only)
        self.assertEqual(result.modalities[1].alignment_status, "global_only")

        leaking_transform = deepcopy(global_only)
        leaking_transform["modalities"][1]["source_to_reference_transform"] = [
            canonical_parent_payload()["reference_canvas"]["transform_chain"][0]
        ]
        with self.assertRaisesRegex(ValidationError, "must not expose pixel-level transforms"):
            CanonicalParentV3.model_validate(leaking_transform)


if __name__ == "__main__":
    unittest.main()
