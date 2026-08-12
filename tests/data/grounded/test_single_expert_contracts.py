from __future__ import annotations

from copy import deepcopy
import json
import tempfile
import unittest
from pathlib import Path

from fixture_helpers import no_target_output, target_output
from single_expert_fixture_helpers import (
    build_annotation_asset,
    draft_run,
)

from oa_groundrag.landslide_evidence.contracts import LandslideEvidenceError
from oa_groundrag.landslide_evidence.single_expert import (
    AnnotationIntendedUse,
    VERIFIED_ANNOTATION_SCHEMA,
    create_annotation_project,
    default_region_description,
    load_annotation_project,
    load_model_drafts,
    validate_verified_annotation_row,
    verify_annotation_work,
    write_draft_results,
)
from oa_groundrag.phase3.common import canonical_json, sha256_text
from oa_groundrag.phase4.errors import ContractError
from oa_groundrag.phase4.outputs import (
    RegionDraftQualityStatus,
    assess_region_draft_quality,
    parse_region_model_output,
    region_output_contract,
    region_output_template,
)


class SingleExpertContractTest(unittest.TestCase):
    def test_prompt_excludes_answers_while_ui_template_matches_parser(self) -> None:
        prompt_path = (
            Path(__file__).resolve().parents[2]
            / "configs/stage4_landslide_evidence/single_expert_prompt_v1.txt"
        )
        prompt_text = prompt_path.read_text(encoding="utf-8")
        self.assertNotIn("TARGET_PRESENT_TEMPLATE_JSON", prompt_text)
        self.assertNotIn("NO_TARGET_TEMPLATE_JSON", prompt_text)
        for status in ("target_present", "no_target"):
            template = region_output_template(status)
            self.assertEqual(default_region_description(status), template)
            self.assertNotIn("json_template", region_output_contract(status))
            self.assertNotIn(canonical_json(template), prompt_text)
            self.assertEqual(
                parse_region_model_output(template).target_status.value,
                status,
            )
        first = region_output_template("target_present")
        first["target_appearance"]["tone"] = "changed"
        self.assertEqual(
            region_output_template("target_present")["target_appearance"]["tone"],
            "无法判断",
        )

    def test_draft_quality_distinguishes_information_and_template_copy(self) -> None:
        informative = target_output()
        informative["evidence_sufficiency"] = "sufficient"
        informative["limitations"] = []
        self.assertEqual(
            assess_region_draft_quality(informative).status,
            RegionDraftQualityStatus.INFORMATIVE,
        )

        limited = region_output_template("target_present")
        limited["surrounding_environment"]["land_cover"] = ["林地覆盖的山地场景"]
        limited["short_summary"] = "目标区域像素较少，局部外观难以可靠分辨。"
        limited["limitations"] = ["目标区域过小且分辨率有限，无法看清可靠局部纹理。"]
        self.assertEqual(
            assess_region_draft_quality(limited).status,
            RegionDraftQualityStatus.LIMITED_BUT_SPECIFIC,
        )

        copied = assess_region_draft_quality(
            region_output_template("target_present")
        )
        self.assertEqual(copied.status, RegionDraftQualityStatus.LOW_INFORMATION)
        self.assertIn("template_copy", copied.issues)

        generic = region_output_template("target_present")
        generic["short_summary"] = "当前目标仍无法判断。"
        self.assertEqual(
            assess_region_draft_quality(generic).status,
            RegionDraftQualityStatus.LOW_INFORMATION,
        )
        self.assertEqual(
            assess_region_draft_quality(
                region_output_template("no_target")
            ).status,
            RegionDraftQualityStatus.NOT_APPLICABLE_NO_TARGET,
        )

    def _project_with_one_draft(self, root: Path) -> tuple[Path, dict, dict]:
        asset_root = build_annotation_asset(root / "asset", kind="train")
        prompt = root / "prompt.txt"
        prompt.write_text("fixture prompt", encoding="utf-8")
        project_root = root / "project"
        create_annotation_project(
            asset_root=asset_root,
            output_root=project_root,
            intended_use=AnnotationIntendedUse.TRAIN_SUPERVISION.value,
            prompt_path=prompt,
        )
        _, assignments = load_annotation_project(project_root)
        assignment = next(row for row in assignments if row["target_status"] == "no_target")
        description = no_target_output()
        run = draft_run(
            [assignment["record_id"]],
            partition=assignment["partition"],
            suffix="one",
            prompt_text=(project_root / "prompt.txt").read_text(encoding="utf-8"),
        )
        draft = {
            "schema_version": "oa_groundrag.mask_grounded_region.model_draft.v1",
            "draft_id": "draft-one",
            "draft_run_id": run["draft_run_id"],
            "record_id": assignment["record_id"],
            "asset_identity_sha256": assignment["asset_identity_sha256"],
            "messages_sha256": sha256_text("messages"),
            "raw_output": canonical_json(description),
            "parse_status": "valid",
            "description": description,
            "failure": None,
        }
        write_draft_results(
            project_root,
            draft_run=run,
            new_drafts=[draft],
            freeze_prompt=False,
        )
        return project_root, assignment, draft

    def test_verified_schema_has_only_minimal_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project_root, assignment, draft = self._project_with_one_draft(Path(temporary))
            result = verify_annotation_work(
                project_root=project_root,
                record_id=assignment["record_id"],
                editor_text=canonical_json(no_target_output()),
            )
            self.assertTrue(result["verified"])
            row = json.loads(
                (project_root / "verified" / f"{assignment['record_id']}.json").read_text()
            )
            self.assertEqual(row["schema_version"], VERIFIED_ANNOTATION_SCHEMA)
            self.assertEqual(row["annotator"], "expert")
            self.assertEqual(
                set(row),
                {
                    "schema_version", "record_id", "asset_identity_sha256", "draft_id",
                    "annotator", "verification_status", "description",
                },
            )
            self.assertNotIn("reviewer_id", row)
            self.assertNotIn("adjudication_status", row)

    def test_unknown_identity_and_no_target_mismatch_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project_root, assignment, draft = self._project_with_one_draft(Path(temporary))
            del project_root
            valid = {
                "schema_version": VERIFIED_ANNOTATION_SCHEMA,
                "record_id": assignment["record_id"],
                "asset_identity_sha256": assignment["asset_identity_sha256"],
                "draft_id": draft["draft_id"],
                "annotator": "expert",
                "verification_status": "expert_verified",
                "description": no_target_output(),
            }
            self.assertEqual(
                validate_verified_annotation_row(
                    valid, assignment=assignment, draft=draft, location="annotation"
                )["verification_status"],
                "expert_verified",
            )
            unknown = deepcopy(valid)
            unknown["reviewer_id"] = "not-allowed"
            with self.assertRaises(LandslideEvidenceError):
                validate_verified_annotation_row(
                    unknown, assignment=assignment, draft=draft, location="annotation"
                )
            legacy = deepcopy(valid)
            legacy["annotator_id"] = legacy.pop("annotator")
            with self.assertRaises(LandslideEvidenceError):
                validate_verified_annotation_row(
                    legacy, assignment=assignment, draft=draft, location="annotation"
                )
            other_expert = deepcopy(valid)
            other_expert["annotator"] = "expert-one"
            with self.assertRaises(LandslideEvidenceError):
                validate_verified_annotation_row(
                    other_expert,
                    assignment=assignment,
                    draft=draft,
                    location="annotation",
                )
            drift = deepcopy(valid)
            drift["draft_id"] = "wrong-draft"
            with self.assertRaises(LandslideEvidenceError):
                validate_verified_annotation_row(
                    drift, assignment=assignment, draft=draft, location="annotation"
                )
            mismatch = deepcopy(valid)
            mismatch["description"] = target_output()
            with self.assertRaises(LandslideEvidenceError):
                validate_verified_annotation_row(
                    mismatch, assignment=assignment, draft=draft, location="annotation"
                )

    def test_duplicate_nonfinite_and_forbidden_claims_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project_root, assignment, _ = self._project_with_one_draft(Path(temporary))
            duplicate = '{"schema_version":"x","schema_version":"y"}'
            with self.assertRaises(ContractError):
                verify_annotation_work(
                    project_root=project_root,
                    record_id=assignment["record_id"],
                    editor_text=duplicate,
                )
            nonfinite = canonical_json(no_target_output()).replace(
                '"evidence_sufficiency":"insufficient"',
                '"evidence_sufficiency":NaN',
            )
            with self.assertRaises(ContractError):
                verify_annotation_work(
                    project_root=project_root,
                    record_id=assignment["record_id"],
                    editor_text=nonfinite,
                )
            forbidden = no_target_output()
            forbidden["short_summary"] = "该区域由暴雨触发。"
            with self.assertRaises(ContractError):
                verify_annotation_work(
                    project_root=project_root,
                    record_id=assignment["record_id"],
                    editor_text=canonical_json(forbidden),
                )
            smuggled = no_target_output()
            smuggled["short_summary"] = "证据不足，但是该区域由暴雨触发。"
            with self.assertRaises(ContractError):
                verify_annotation_work(
                    project_root=project_root,
                    record_id=assignment["record_id"],
                    editor_text=canonical_json(smuggled),
                )
            uncertain = no_target_output()
            uncertain["limitations"] = ["无法判断该区域是否由暴雨触发。"]
            verify_annotation_work(
                project_root=project_root,
                record_id=assignment["record_id"],
                editor_text=canonical_json(uncertain),
            )
            self.assertEqual(load_model_drafts(project_root)[assignment["record_id"]]["draft_id"], "draft-one")


if __name__ == "__main__":
    unittest.main()
