from __future__ import annotations

from collections import Counter
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.data.grounded.fixture_helpers import no_target_output, target_output

from tests.data.grounded.single_expert_fixture_helpers import (
    FakeDraftRuntime,
    SOURCES,
    build_annotation_asset,
    draft_run,
    populate_project,
)

from oa_groundrag.data.grounded.contracts import LandslideEvidenceError
from oa_groundrag.data.grounded.annotation.project import (
    AnnotationIntendedUse,
    MODEL_DRAFT_SCHEMA,
    VERIFIED_ANNOTATION_SCHEMA,
    _write_status,
    create_annotation_project,
    load_annotation_project,
    load_model_drafts,
    register_draft_run,
    write_draft_results,
)
from oa_groundrag.data.grounded.annotation.package import (
    export_verified_annotations,
    validate_verified_annotation_package,
)
from oa_groundrag.data.grounded.annotation.training import (
    MaskGroundedTrainingMessageDataset,
    export_training_messages,
    load_training_message_artifact,
)
from oa_groundrag.data.grounded.annotation.drafting import (
    generate_annotation_drafts,
    load_local_draft_config,
)
from oa_groundrag.data.rs_general.io import (
    atomic_write_json,
    canonical_json,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_text,
)
from oa_groundrag.vlm.errors import ContractError
from oa_groundrag.evaluation.grounding.observations import _human_metrics


REPO_ROOT = Path(__file__).resolve().parents[3]
PROMPT_CONFIG = REPO_ROOT / "configs/grounding/prompts/single_expert_prompt_v1.txt"
DRAFT_CONFIG = REPO_ROOT / "configs/grounding/single_expert_qwen3vl_8b_v1.yaml"


class SingleExpertPipelineTest(unittest.TestCase):
    def test_calibration_may_adjust_bounded_greedy_generation_config(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = DRAFT_CONFIG.read_text(encoding="utf-8")
            adjusted_path = root / "adjusted.yaml"
            adjusted_path.write_text(
                original.replace("max_new_tokens: 768", "max_new_tokens: 512"),
                encoding="utf-8",
            )
            adjusted = load_local_draft_config(adjusted_path)
            baseline = load_local_draft_config(DRAFT_CONFIG)
            self.assertEqual(baseline.max_images, 3)
            self.assertEqual(baseline.max_input_tokens, 4096)
            self.assertEqual(baseline.max_new_tokens, 768)
            self.assertEqual(adjusted.max_new_tokens, 512)
            self.assertNotEqual(adjusted.semantic_sha256, baseline.semantic_sha256)
            unsafe = root / "unsafe.yaml"
            unsafe.write_text(
                original.replace("do_sample: false", "do_sample: true"),
                encoding="utf-8",
            )
            with self.assertRaises(LandslideEvidenceError):
                load_local_draft_config(unsafe)

    def test_calibration_is_deterministic_stratified_and_train_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            asset_root = build_annotation_asset(root / "asset", kind="train")
            first_root, second_root = root / "first", root / "second"
            for output in (first_root, second_root):
                create_annotation_project(
                    asset_root=asset_root,
                    output_root=output,
                    intended_use=AnnotationIntendedUse.TRAIN_SUPERVISION.value,
                    prompt_path=PROMPT_CONFIG,
                )
            first, first_assignments = load_annotation_project(first_root)
            second, second_assignments = load_annotation_project(second_root)
            self.assertEqual(first["calibration_record_ids"], second["calibration_record_ids"])
            calibration = [row for row in first_assignments if row["partition"] == "calibration"]
            self.assertEqual(len(calibration), 20)
            self.assertEqual(Counter(row["source"] for row in calibration), Counter({source: 4 for source in SOURCES}))
            self.assertEqual({row["split"] for row in calibration}, {"train"})
            no_target_sources = {row["source"] for row in calibration if row["target_status"] == "no_target"}
            self.assertEqual(no_target_sources, set(SOURCES) - {"multimodal_landslide"})
            self.assertEqual(
                [row["record_id"] for row in first_assignments],
                [row["record_id"] for row in second_assignments],
            )

    def test_test_split_is_rejected_at_project_creation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            test_root = build_annotation_asset(root / "test_asset", kind="train", split="test")
            with self.assertRaises(LandslideEvidenceError) as raised:
                create_annotation_project(
                    asset_root=test_root,
                    output_root=root / "project",
                    intended_use=AnnotationIntendedUse.TRAIN_SUPERVISION.value,
                    prompt_path=PROMPT_CONFIG,
                )
            self.assertEqual(raised.exception.code, "SPLIT_FORBIDDEN")

    def test_source_ledger_tamper_is_rejected_before_project_creation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            asset_root = build_annotation_asset(root / "asset", kind="train")
            records_path = asset_root / "records.jsonl"
            records_path.write_text(
                records_path.read_text(encoding="utf-8") + " ",
                encoding="utf-8",
            )
            with self.assertRaises(LandslideEvidenceError) as raised:
                create_annotation_project(
                    asset_root=asset_root,
                    output_root=root / "project",
                    intended_use=AnnotationIntendedUse.TRAIN_SUPERVISION.value,
                    prompt_path=PROMPT_CONFIG,
                )
            self.assertEqual(raised.exception.code, "LEDGER_MISMATCH")

    def test_fake_local_drafting_is_single_attempt_and_remaining_is_gated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            asset_root = build_annotation_asset(root / "asset", kind="train")
            project_root = root / "project"
            create_annotation_project(
                asset_root=asset_root,
                output_root=project_root,
                intended_use=AnnotationIntendedUse.TRAIN_SUPERVISION.value,
                prompt_path=PROMPT_CONFIG,
            )
            first = generate_annotation_drafts(
                project_root=project_root,
                config_path=DRAFT_CONFIG,
                partition="calibration",
                limit=1,
                runtime=FakeDraftRuntime(),
            )
            self.assertEqual(first["generated"], 1)
            self.assertTrue((project_root / "draft_run.json").is_file())
            first_ids = set(load_model_drafts(project_root))
            second = generate_annotation_drafts(
                project_root=project_root,
                config_path=DRAFT_CONFIG,
                partition="calibration",
                limit=1,
                runtime=FakeDraftRuntime(),
            )
            self.assertEqual(second["generated"], 1)
            second_ids = set(load_model_drafts(project_root))
            self.assertEqual(len(second_ids), 2)
            self.assertTrue(first_ids < second_ids)
            with self.assertRaises(LandslideEvidenceError):
                generate_annotation_drafts(
                    project_root=project_root,
                    config_path=DRAFT_CONFIG,
                    partition="remaining",
                    limit=1,
                    runtime=FakeDraftRuntime(),
                )

    def test_remaining_freezes_prompt_and_config_before_model_load_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            asset_root = build_annotation_asset(root / "asset", kind="train")
            project_root = root / "project"
            create_annotation_project(
                asset_root=asset_root,
                output_root=project_root,
                intended_use=AnnotationIntendedUse.TRAIN_SUPERVISION.value,
                prompt_path=PROMPT_CONFIG,
            )
            _, assignments = load_annotation_project(project_root)
            calibration = [
                row for row in assignments if row["partition"] == "calibration"
            ]
            run = draft_run(
                [row["record_id"] for row in calibration],
                partition="calibration",
                suffix="freeze-before-load",
                prompt_text=(project_root / "prompt.txt").read_text(encoding="utf-8"),
            )
            drafts = []
            for assignment in calibration:
                description = (
                    no_target_output()
                    if assignment["target_status"] == "no_target"
                    else target_output()
                )
                draft_id = f"draft-freeze-{assignment['record_id']}"
                drafts.append({
                    "schema_version": MODEL_DRAFT_SCHEMA,
                    "draft_id": draft_id,
                    "draft_run_id": run["draft_run_id"],
                    "record_id": assignment["record_id"],
                    "asset_identity_sha256": assignment["asset_identity_sha256"],
                    "messages_sha256": sha256_text(draft_id),
                    "raw_output": canonical_json(description),
                    "parse_status": "valid",
                    "description": description,
                    "failure": None,
                })
            write_draft_results(
                project_root,
                draft_run=run,
                new_drafts=drafts,
                freeze_prompt=False,
            )
            by_id = {row["record_id"]: row for row in drafts}
            for assignment in calibration:
                atomic_write_json(
                    project_root / "verified" / f"{assignment['record_id']}.json",
                    {
                        "schema_version": VERIFIED_ANNOTATION_SCHEMA,
                        "record_id": assignment["record_id"],
                        "asset_identity_sha256": assignment["asset_identity_sha256"],
                        "draft_id": by_id[assignment["record_id"]]["draft_id"],
                        "annotator": "expert",
                        "verification_status": "expert_verified",
                        "description": by_id[assignment["record_id"]]["description"],
                    },
                )
            _write_status(project_root)
            with patch(
                "oa_groundrag.data.grounded.annotation.drafting.LocalQwenDraftRuntime",
                side_effect=RuntimeError("simulated model load failure"),
            ):
                with self.assertRaisesRegex(RuntimeError, "model load failure"):
                    generate_annotation_drafts(
                        project_root=project_root,
                        config_path=DRAFT_CONFIG,
                        partition="remaining",
                        limit=1,
                    )
            project, _ = load_annotation_project(project_root)
            config = load_local_draft_config(DRAFT_CONFIG)
            self.assertEqual(
                project["frozen_prompt_sha256"],
                sha256_file(project_root / "prompt.txt"),
            )
            self.assertEqual(
                project["frozen_draft_config_sha256"],
                config.semantic_sha256,
            )

    def test_full_train_package_exports_expert_answer_as_assistant(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            asset_root = build_annotation_asset(root / "asset", kind="train")
            project_root = root / "project"
            create_annotation_project(
                asset_root=asset_root,
                output_root=project_root,
                intended_use=AnnotationIntendedUse.TRAIN_SUPERVISION.value,
                prompt_path=PROMPT_CONFIG,
            )
            _, assignments = load_annotation_project(project_root)
            edited_id = next(row["record_id"] for row in assignments if row["target_status"] == "target_present")
            populate_project(project_root, edited_record_id=edited_id)
            package_root = root / "package"
            result = export_verified_annotations(
                project_root=project_root,
                output_root=package_root,
            )
            self.assertTrue(result["training_eligible"])
            package = validate_verified_annotation_package(
                asset_root=asset_root,
                package_root=package_root,
            )
            self.assertEqual(len(package.annotations), 500)
            self.assertFalse(package.manifest["formal_acceptance"])
            messages_root = root / "messages"
            exported = export_training_messages(
                asset_root=asset_root,
                annotations_root=package_root,
                output_root=messages_root,
            )
            self.assertEqual(exported["record_count"], 500)
            rows = {row["record_id"]: row for row in read_jsonl(messages_root / "messages.jsonl")}
            assistant = rows[edited_id]["messages"][-1]["content"][0]["text"]
            self.assertIn("专家核验后保留", assistant)
            draft = next(row for row in package.drafts if row["record_id"] == edited_id)
            self.assertNotEqual(assistant, canonical_json(draft["description"]))
            manifest = read_json(messages_root / "manifest.json")
            self.assertEqual(manifest["reference_authority"], "single_expert")
            self.assertFalse(manifest["formal_acceptance"])
            dataset = MaskGroundedTrainingMessageDataset(messages_root)
            self.assertEqual(len(dataset), 500)
            dataset_index = next(
                index for index, row in enumerate(dataset.records)
                if row["record_id"] == edited_id
            )
            sample = dataset[dataset_index]
            self.assertEqual(sample.logical_role, "train")
            self.assertEqual(sample.task_family, "mask_grounded_region_description")
            self.assertIn("专家核验后保留", sample.reference_responses[0])
            with self.assertRaises(ContractError):
                export_training_messages(
                    asset_root=asset_root,
                    annotations_root=package_root,
                    output_root=messages_root,
                )
            messages_path = messages_root / "messages.jsonl"
            messages_path.write_text(
                messages_path.read_text(encoding="utf-8").replace(
                    "专家核验后保留", "被篡改的监督答案", 1
                ),
                encoding="utf-8",
            )
            with self.assertRaises(LandslideEvidenceError):
                load_training_message_artifact(messages_root)

    def test_val_reference_is_baseline_only_and_cannot_enter_training(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train_asset = build_annotation_asset(root / "train_asset", kind="train")
            train_project = root / "train_project"
            create_annotation_project(
                asset_root=train_asset,
                output_root=train_project,
                intended_use=AnnotationIntendedUse.TRAIN_SUPERVISION.value,
                prompt_path=PROMPT_CONFIG,
            )
            _, train_assignments = load_annotation_project(train_project)
            remaining = [row for row in train_assignments if row["partition"] == "remaining"]
            run = draft_run(
                [remaining[0]["record_id"]],
                partition="remaining",
                suffix="freeze",
                prompt_text=(train_project / "prompt.txt").read_text(encoding="utf-8"),
            )
            register_draft_run(train_project, draft_run=run, freeze_prompt=True)
            eval_asset = build_annotation_asset(root / "eval_asset", kind="eval")
            dev_project = root / "dev_project"
            created = create_annotation_project(
                asset_root=eval_asset,
                output_root=dev_project,
                intended_use=AnnotationIntendedUse.DEV_REFERENCE.value,
                train_project_root=train_project,
            )
            self.assertEqual(created["record_count"], 100)
            project, assignments = load_annotation_project(dev_project)
            self.assertTrue(project["baseline_only"])
            self.assertEqual({row["split"] for row in assignments}, {"val"})
            populate_project(dev_project)
            package_root = root / "dev_package"
            export_verified_annotations(project_root=dev_project, output_root=package_root)
            package = validate_verified_annotation_package(
                asset_root=eval_asset,
                package_root=package_root,
            )
            self.assertFalse(package.manifest["training_eligible"])
            self.assertEqual(package.manifest["intended_use"], "single_expert_dev_reference")
            reference_metrics = _human_metrics(
                package_root,
                eval_root=eval_asset,
                valid_outputs={
                    row["record_id"]: row["description"]
                    for row in package.annotations
                },
            )
            assert reference_metrics is not None
            self.assertEqual(reference_metrics["reference_authority"], "single_expert")
            self.assertEqual(reference_metrics["auxiliary_structured_exact_match_rate"], 1.0)
            self.assertEqual(reference_metrics["auxiliary_field_exact_match_rate"], 1.0)
            with self.assertRaises(LandslideEvidenceError) as raised:
                export_training_messages(
                    asset_root=eval_asset,
                    annotations_root=package_root,
                    output_root=root / "forbidden_messages",
                )
            self.assertEqual(raised.exception.code, "SPLIT_FORBIDDEN")


if __name__ == "__main__":
    unittest.main()
