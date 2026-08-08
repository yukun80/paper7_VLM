"""Stage 4 v2 模型辅助监督、排除规则和动态训练消息永久回归。"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from tests.stage4_landslide_evidence.fixture_helpers import (
    no_target_output,
    target_output,
    target_record,
)

from oa_groundrag.landslide_evidence.model_assisted import (
    EXPERT_AUTHORITY,
    MODEL_AUTHORITY,
    MODEL_ASSISTED_ASSIGNMENT_SCHEMA,
    MODEL_ASSISTED_PROJECT_SCHEMA,
    ModelAssistedCollectionContext,
    ModelAssistedCollectionEntry,
    ModelAssistedProjectContext,
    ModelAssistedTrainingMessageDataset,
    _assignment,
    _load_collection_context,
    assess_supervision_eligibility,
    export_model_assisted_supervision,
    export_model_assisted_training_messages,
    load_model_assisted_training_messages,
    validate_model_assisted_supervision,
)
from oa_groundrag.landslide_evidence.region_pipeline import region_asset_identity
from oa_groundrag.landslide_evidence.single_expert import (
    DRAFT_MODEL_REVISION,
    MODEL_DRAFT_FAILURE_SCHEMA,
    MODEL_DRAFT_RUN_SCHEMA,
    MODEL_DRAFT_SCHEMA,
    VERIFIED_ANNOTATION_SCHEMA,
)
from oa_groundrag.phase3.common import (
    atomic_write_json,
    canonical_json,
    sha256_file,
    sha256_text,
)
from oa_groundrag.phase4.outputs import region_output_template


class ModelAssistedTests(unittest.TestCase):
    @staticmethod
    def _draft_run(record_ids: list[str], *, prompt_text: str) -> dict[str, object]:
        config = {
            "schema_version": "oa_groundrag.mask_grounded_region.draft_config.v1",
            "model": {
                "path": "/fixture/Qwen3-VL-8B-Instruct",
                "processor_path": "/fixture/Qwen3-VL-8B-Instruct",
                "repository": "Qwen/Qwen3-VL-8B-Instruct",
                "revision": DRAFT_MODEL_REVISION,
                "local_files_only": True,
                "trust_remote_code": False,
                "dtype": "bfloat16",
                "attn_implementation": "sdpa",
            },
            "processor": {
                "min_pixels": 12544,
                "max_pixels": 200704,
                "max_images": 3,
                "max_input_tokens": 4096,
            },
            "generation": {
                "max_new_tokens": 768,
                "do_sample": False,
                "temperature": 0.0,
                "top_p": 1.0,
                "seed": 20260804,
            },
        }
        return {
            "schema_version": MODEL_DRAFT_RUN_SCHEMA,
            "draft_run_id": "draft_run_fixture",
            "config_sha256": "1" * 64,
            "config_semantic_sha256": sha256_text(canonical_json(config)),
            "config": config,
            "model_repository": "Qwen/Qwen3-VL-8B-Instruct",
            "model_revision": DRAFT_MODEL_REVISION,
            "model_identity": {"fixture": True},
            "processor_identity": {"fixture": True},
            "prompt_text": prompt_text,
            "prompt_sha256": sha256_text(prompt_text),
            "generation": {
                "max_new_tokens": 768,
                "do_sample": False,
                "temperature": 0.0,
                "top_p": 1.0,
                "seed": 20260804,
                "single_attempt": True,
            },
            "partition": "all",
            "record_ids": record_ids,
            "record_ids_sha256": sha256_text(canonical_json(record_ids)),
            "formal_acceptance": False,
        }

    def _draft(
        self,
        assignment: dict[str, object],
        description: dict[str, object] | None,
        *,
        run_id: str = "draft_run_fixture",
        messages_sha256: str = "b" * 64,
    ) -> dict[str, object]:
        if description is None:
            return {
                "schema_version": MODEL_DRAFT_SCHEMA,
                "draft_id": f"draft_{assignment['record_id']}",
                "draft_run_id": run_id,
                "record_id": assignment["record_id"],
                "asset_identity_sha256": assignment["asset_identity_sha256"],
                "messages_sha256": messages_sha256,
                "raw_output": "not-json",
                "parse_status": "invalid",
                "description": None,
                "failure": {
                    "schema_version": MODEL_DRAFT_FAILURE_SCHEMA,
                    "code": "INVALID_MODEL_OUTPUT",
                    "message": "fixture invalid JSON",
                    "details": {},
                },
            }
        return {
            "schema_version": MODEL_DRAFT_SCHEMA,
            "draft_id": f"draft_{assignment['record_id']}",
            "draft_run_id": run_id,
            "record_id": assignment["record_id"],
            "asset_identity_sha256": assignment["asset_identity_sha256"],
            "messages_sha256": messages_sha256,
            "raw_output": canonical_json(description),
            "parse_status": "valid",
            "description": description,
            "failure": None,
        }

    @staticmethod
    def _simple_assignment(*, status: str = "target_present") -> dict[str, object]:
        return {
            "schema_version": MODEL_ASSISTED_ASSIGNMENT_SCHEMA,
            "ordinal": 0,
            "record_id": "record_a",
            "parent_id": "parent_a",
            "source": "synthetic",
            "split": "train",
            "target_status": status,
            "member": "base",
            "member_root": "/fixture",
            "member_manifest_sha256": "c" * 64,
            "asset_identity_sha256": "a" * 64,
        }

    def test_eligibility_distinguishes_expert_model_and_exclusions(self) -> None:
        assignment = self._simple_assignment()
        informative = self._draft(assignment, target_output())
        decision = assess_supervision_eligibility(
            draft=informative,
            assignment=assignment,
        )
        self.assertTrue(decision.eligible)
        self.assertEqual(decision.authority, MODEL_AUTHORITY)

        verified_description = target_output()
        verified_description["short_summary"] = "专家确认该区域与周围植被存在可见纹理差异。"
        verified = {
            "schema_version": VERIFIED_ANNOTATION_SCHEMA,
            "record_id": assignment["record_id"],
            "asset_identity_sha256": assignment["asset_identity_sha256"],
            "draft_id": informative["draft_id"],
            "annotator": "expert",
            "verification_status": "expert_verified",
            "description": verified_description,
        }
        expert = assess_supervision_eligibility(
            draft=informative,
            assignment=assignment,
            verified=verified,
        )
        self.assertTrue(expert.eligible)
        self.assertEqual(expert.authority, EXPERT_AUTHORITY)
        self.assertEqual(expert.description["short_summary"], verified_description["short_summary"])

        invalid = assess_supervision_eligibility(
            draft=self._draft(assignment, None),
            assignment=assignment,
        )
        self.assertFalse(invalid.eligible)
        self.assertEqual(invalid.reason_code, "parse_invalid")

        copied = assess_supervision_eligibility(
            draft=self._draft(assignment, region_output_template("target_present")),
            assignment=assignment,
        )
        self.assertFalse(copied.eligible)
        self.assertEqual(copied.reason_code, "template_copy")

    def test_specific_low_information_and_normal_uncertainty_are_trainable(self) -> None:
        assignment = self._simple_assignment()
        specific_low = region_output_template("target_present")
        specific_low["target_appearance"]["tone"] = "mask 内可见局部浅棕色斑块"
        specific_low["short_summary"] = "仅能确认关注区存在局部浅棕色视觉斑块。"
        decision = assess_supervision_eligibility(
            draft=self._draft(assignment, specific_low),
            assignment=assignment,
        )
        self.assertTrue(decision.eligible)
        self.assertEqual(decision.quality_status, "low_information")

        uncertain = target_output()
        uncertain["target_morphology"]["qualitative_orientation"] = "无法判断"
        decision = assess_supervision_eligibility(
            draft=self._draft(assignment, uncertain),
            assignment=assignment,
        )
        self.assertTrue(decision.eligible)

        no_target_assignment = self._simple_assignment(status="no_target")
        no_target = assess_supervision_eligibility(
            draft=self._draft(no_target_assignment, no_target_output()),
            assignment=no_target_assignment,
        )
        self.assertTrue(no_target.eligible)
        self.assertEqual(no_target.quality_status, "not_applicable_no_target")

    def _project_fixture(
        self,
        root: Path,
    ) -> tuple[ModelAssistedProjectContext, Path]:
        member_root = root / "member"
        member_root.mkdir()
        entries = []
        for ordinal, record_id in enumerate(("r_expert", "r_model", "r_invalid", "r_template")):
            record = target_record(member_root, record_id=record_id)
            record["split"] = "train"
            record["parent_id"] = f"parent_{record_id}"
            record["source"] = "synthetic"
            asset_id = region_asset_identity(member_root, record["assets"])
            entries.append(ModelAssistedCollectionEntry(
                ordinal=ordinal,
                member="base",
                member_root=member_root,
                member_manifest_sha256="c" * 64,
                record=record,
                queue={
                    "record_id": record_id,
                    "split": "train",
                    "asset_identity_sha256": asset_id,
                },
            ))
        collection_root = root / "collection"
        collection_root.mkdir()
        collection = ModelAssistedCollectionContext(
            root=collection_root,
            manifest={"schema_version": "oa_groundrag.mask_grounded_region.train_collection.v2"},
            manifest_sha256="d" * 64,
            entries=tuple(entries),
        )
        assignments = tuple(_assignment(entry) for entry in entries)
        prompt = "fixture model-assisted prompt"
        run = self._draft_run(
            [str(row["record_id"]) for row in assignments],
            prompt_text=prompt,
        )
        drafts = {}
        descriptions = (
            target_output(), target_output(), None, region_output_template("target_present")
        )
        for assignment, entry, description in zip(assignments, entries, descriptions, strict=True):
            from oa_groundrag.landslide_evidence.single_expert import build_annotation_draft_messages

            messages_sha = sha256_text(canonical_json(build_annotation_draft_messages(
                entry.record,
                asset_root=entry.member_root,
                prompt_text=prompt,
            )))
            drafts[str(assignment["record_id"])] = self._draft(
                assignment,
                description,
                run_id=run["draft_run_id"],
                messages_sha256=messages_sha,
            )
        verified_description = deepcopy(target_output())
        verified_description["short_summary"] = "专家最终确认关注区与周边植被具有可见纹理差异。"
        first = assignments[0]
        verified = {
            str(first["record_id"]): {
                "schema_version": VERIFIED_ANNOTATION_SCHEMA,
                "record_id": first["record_id"],
                "asset_identity_sha256": first["asset_identity_sha256"],
                "draft_id": drafts[str(first["record_id"])]["draft_id"],
                "annotator": "expert",
                "verification_status": "expert_verified",
                "description": verified_description,
            }
        }
        legacy_root = root / "legacy"
        legacy_root.mkdir()
        atomic_write_json(legacy_root / "snapshot.json", {"frozen": True})
        file_row = {
            "path": "snapshot.json",
            "size_bytes": (legacy_root / "snapshot.json").stat().st_size,
            "sha256": sha256_file(legacy_root / "snapshot.json"),
        }
        provenance = {
            "schema_version": "oa_groundrag.mask_grounded_region.legacy_import_provenance.v2",
            "source_project_root": str(legacy_root),
            "files": [file_row],
            "files_root_sha256": sha256_text(canonical_json([file_row])),
            "imported_draft_count": 1,
            "imported_verified_count": 1,
            "formal_acceptance": False,
        }
        work_root = root / "work"
        work_root.mkdir()
        atomic_write_json(work_root / "project.json", {"schema_version": MODEL_ASSISTED_PROJECT_SCHEMA})
        atomic_write_json(work_root / "import_provenance.json", provenance)
        project = {
            "schema_version": MODEL_ASSISTED_PROJECT_SCHEMA,
            "project_id": "fixture",
            "collection_root": str(collection_root),
            "collection_manifest_sha256": collection.manifest_sha256,
            "split": "train",
            "record_count": 4,
            "ordered_record_ids_sha256": sha256_text(canonical_json([
                row["record_id"] for row in assignments
            ])),
            "prompt_sha256": sha256_text(prompt),
            "config_semantic_sha256": run["config_semantic_sha256"],
            "legacy_import_sha256": sha256_text(canonical_json(provenance)),
            "formal_acceptance": False,
        }
        return ModelAssistedProjectContext(
            root=work_root,
            project=project,
            collection=collection,
            assignments=assignments,
            entries_by_id={str(entry.record["record_id"]): entry for entry in entries},
            drafts=drafts,
            draft_runs=(run,),
            verified=verified,
        ), collection_root

    def test_package_and_dynamic_training_messages_preserve_authority(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project, collection_root = self._project_fixture(root)
            package_root = root / "package"
            messages_root = root / "messages"
            with (
                patch(
                    "oa_groundrag.landslide_evidence.model_assisted.EXPECTED_COLLECTION_COUNT",
                    4,
                ),
                patch(
                    "oa_groundrag.landslide_evidence.model_assisted.load_model_assisted_project",
                    return_value=project,
                ),
                patch(
                    "oa_groundrag.landslide_evidence.model_assisted._load_collection_context",
                    return_value=project.collection,
                ),
                patch(
                    "oa_groundrag.landslide_evidence.model_assisted._legacy_import_rows",
                    return_value=(
                        {"r_expert": project.drafts["r_expert"]},
                        {"r_expert": project.verified["r_expert"]},
                    ),
                ),
            ):
                result = export_model_assisted_supervision(
                    project_root=project.root,
                    output_root=package_root,
                )
                self.assertEqual(result["eligible_count"], 2)
                self.assertEqual(result["excluded_count"], 2)
                package = validate_model_assisted_supervision(
                    collection_root=collection_root,
                    package_root=package_root,
                )
                self.assertEqual(
                    package.manifest["authority_counts"],
                    {EXPERT_AUTHORITY: 1, MODEL_AUTHORITY: 1},
                )
                self.assertEqual(
                    {row["reason_code"] for row in package.exclusions},
                    {"parse_invalid", "template_copy"},
                )
                export_model_assisted_training_messages(
                    collection_root=collection_root,
                    package_root=package_root,
                    output_root=messages_root,
                )
                artifact = load_model_assisted_training_messages(messages_root)
                self.assertEqual(len(artifact.rows), 2)
                self.assertEqual(
                    [row["supervision_authority"] for row in artifact.rows],
                    [EXPERT_AUTHORITY, MODEL_AUTHORITY],
                )
                dataset = ModelAssistedTrainingMessageDataset(messages_root)
                self.assertEqual(len(dataset), 2)
                self.assertIn(
                    "专家最终确认",
                    dataset[0].reference_responses[0],
                )
                self.assertFalse(dataset[0].provenance["gold"])
                atomic_write_json(root / "legacy" / "unexpected.json", {"drift": True})
                with self.assertRaises(Exception):
                    validate_model_assisted_supervision(
                        collection_root=collection_root,
                        package_root=package_root,
                    )
                (root / "legacy" / "unexpected.json").unlink()

    def test_collection_loader_rejects_val_before_training_assignment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record = target_record(root, record_id="val_record")
            asset_id = region_asset_identity(root, record["assets"])
            expanded_entry = SimpleNamespace(
                index={
                    "ordinal": 0,
                    "record_id": "val_record",
                    "member": "base",
                    "asset_identity_sha256": asset_id,
                },
                record=record,
                queue={
                    "record_id": "val_record",
                    "split": "val",
                    "asset_identity_sha256": asset_id,
                },
                member_root=root,
                member_manifest_sha256="a" * 64,
            )
            expanded = SimpleNamespace(
                root=root,
                manifest={},
                manifest_sha256="b" * 64,
                entries=(expanded_entry,),
            )
            with patch(
                "oa_groundrag.landslide_evidence.expanded_region.load_expanded_collection_context",
                return_value=expanded,
            ):
                with self.assertRaises(Exception):
                    _load_collection_context(root)


if __name__ == "__main__":
    unittest.main()
