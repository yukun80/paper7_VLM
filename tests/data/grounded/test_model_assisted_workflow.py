from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

from oa_groundrag.data.grounded.contracts import LandslideEvidenceError
from oa_groundrag.data.grounded.supervision.workflow import (
    MODEL_ASSISTED_RECORD_COUNT,
    ModelAssistedWorkflowPaths,
    prepare_expanded_corpus,
    run_model_assisted_train_workflow,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
CLI_PATH = (
    REPO_ROOT
    / "scripts"
    / "data"
    / "grounded_supervision.py"
)
MODEL_MODULE = "oa_groundrag.data.grounded.supervision.model_assisted"
EXPANDED_MODULE = "oa_groundrag.data.grounded.supervision.expanded_region"


class ModelAssistedWorkflowTest(unittest.TestCase):
    @staticmethod
    def _paths(root: Path) -> ModelAssistedWorkflowPaths:
        return ModelAssistedWorkflowPaths(
            extension_config_path=root / "extension.yaml",
            extension_root=root / "region_corpus" / "extension",
            collection_root=root / "region_collection" / "collection",
            legacy_project_root=root / "legacy" / "project",
            project_root=root / "work" / "project",
            annotation_package_root=root / "annotations" / "package",
            training_messages_root=root / "training_messages" / "messages",
            prompt_path=root / "prompt.txt",
            draft_config_path=root / "draft.yaml",
        )

    @staticmethod
    def _status(drafted: int, *, invalid: int = 0) -> dict[str, object]:
        return {
            "total": MODEL_ASSISTED_RECORD_COUNT,
            "drafted": drafted,
            "valid_drafts": drafted - invalid,
            "invalid_drafts": invalid,
            "verified": 500,
            "pending": MODEL_ASSISTED_RECORD_COUNT - drafted,
            "complete": drafted == MODEL_ASSISTED_RECORD_COUNT,
            "formal_acceptance": False,
        }

    def test_prepare_builds_then_strictly_validates_both_roots(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._paths(Path(temporary)).absolute()
            calls: list[tuple[object, ...]] = []
            expanded = ModuleType(EXPANDED_MODULE)

            def prepare(config_path: Path, *, verify_source: bool) -> object:
                calls.append(("prepare", config_path, verify_source))
                return SimpleNamespace(
                    extension_root=paths.extension_root,
                    collection_root=paths.collection_root,
                    extension_manifest_sha256="a" * 64,
                    collection_manifest_sha256="b" * 64,
                    extension_record_count=7_950,
                    collection_record_count=8_450,
                )

            def validate_extension(
                root: Path,
                *,
                config_path: Path,
                verify_source: bool,
            ) -> dict[str, object]:
                calls.append(
                    ("validate_extension", root, config_path, verify_source)
                )
                return {"valid": True, "record_count": 7_950}

            def validate_collection(
                root: Path,
                *,
                verify_members: bool,
                verify_source: bool,
            ) -> dict[str, object]:
                calls.append(
                    ("validate_collection", root, verify_members, verify_source)
                )
                return {"valid": True, "record_count": 8_450}

            expanded.prepare_expanded_region_assets = prepare  # type: ignore[attr-defined]
            expanded.validate_region_extension = validate_extension  # type: ignore[attr-defined]
            expanded.validate_expanded_region_collection = validate_collection  # type: ignore[attr-defined]
            with patch.dict(sys.modules, {EXPANDED_MODULE: expanded}):
                result = prepare_expanded_corpus(paths=paths)
            self.assertEqual(result["stage"], "expanded_corpus_ready")
            self.assertEqual(result["collection_record_count"], 8_450)
            self.assertEqual(
                calls,
                [
                    ("prepare", paths.extension_config_path, True),
                    (
                        "validate_extension",
                        paths.extension_root,
                        paths.extension_config_path,
                        True,
                    ),
                    (
                        "validate_collection",
                        paths.collection_root,
                        True,
                        False,
                    ),
                ],
            )

    def test_one_runtime_fills_missing_then_publishes_in_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._paths(Path(temporary)).absolute()
            calls: list[str] = []
            drafted = 500
            invalid = 0
            runtime = object()
            core = ModuleType(MODEL_MODULE)

            def create(**kwargs: object) -> dict[str, object]:
                calls.append("create")
                self.assertEqual(kwargs["collection_root"], paths.collection_root)
                self.assertEqual(
                    kwargs["legacy_project_root"], paths.legacy_project_root
                )
                self.assertEqual(kwargs["config_path"], paths.draft_config_path)
                paths.project_root.mkdir(parents=True)
                return {"ok": True}

            def status(root: Path) -> dict[str, object]:
                calls.append("status")
                self.assertEqual(root, paths.project_root)
                return self._status(drafted, invalid=invalid)

            def generate(**kwargs: object) -> dict[str, object]:
                nonlocal drafted, invalid
                calls.append("generate")
                self.assertIs(kwargs["runtime"], runtime)
                self.assertEqual(kwargs["config_path"], paths.draft_config_path)
                drafted = MODEL_ASSISTED_RECORD_COUNT
                invalid = 3
                callback = kwargs["progress_callback"]
                self.assertTrue(callable(callback))
                callback({"event": "generation_complete", "generated": 7_950})
                return {"ok": True, "generated": 7_950}

            def export_package(**_: object) -> dict[str, object]:
                calls.append("export_package")
                self.assertEqual(drafted, MODEL_ASSISTED_RECORD_COUNT)
                paths.annotation_package_root.mkdir(parents=True)
                return {"ok": True}

            def validate_package(**_: object) -> object:
                calls.append("validate_package")
                return SimpleNamespace(manifest={
                    "record_count": MODEL_ASSISTED_RECORD_COUNT,
                    "eligible_count": 8_447,
                    "excluded_count": 3,
                    "authority_counts": {
                        "expert_verified": 500,
                        "model_generated_unreviewed": 7_947,
                    },
                    "exclusion_counts": {"parse_invalid": 3},
                    "reference_authority": "mixed_model_and_single_expert",
                })

            def export_messages(**_: object) -> dict[str, object]:
                calls.append("export_messages")
                paths.training_messages_root.mkdir(parents=True)
                return {"ok": True}

            def load_messages(root: Path) -> object:
                calls.append("load_messages")
                self.assertEqual(root, paths.training_messages_root)
                return SimpleNamespace(rows=range(8_447))

            core.create_model_assisted_project = create  # type: ignore[attr-defined]
            core.model_assisted_project_status = status  # type: ignore[attr-defined]
            core.generate_model_assisted_drafts = generate  # type: ignore[attr-defined]
            core.export_model_assisted_supervision = export_package  # type: ignore[attr-defined]
            core.validate_model_assisted_supervision = validate_package  # type: ignore[attr-defined]
            core.export_model_assisted_training_messages = export_messages  # type: ignore[attr-defined]
            core.load_model_assisted_training_messages = load_messages  # type: ignore[attr-defined]
            events: list[str] = []
            with patch.dict(sys.modules, {MODEL_MODULE: core}), patch(
                "oa_groundrag.data.grounded.supervision.workflow.prepare_expanded_corpus",
                return_value={"ok": True},
            ):
                result = run_model_assisted_train_workflow(
                    paths=paths,
                    runtime=runtime,
                    progress_callback=lambda row: events.append(str(row["event"])),
                )
            self.assertEqual(result["stage"], "complete")
            self.assertEqual(result["training_message_count"], 8_447)
            self.assertEqual(result["expert_count"], 500)
            self.assertEqual(result["model_count"], 7_947)
            self.assertEqual(result["excluded_count"], 3)
            self.assertEqual(result["reason_counts"], {"parse_invalid": 3})
            self.assertEqual(
                calls,
                [
                    "create",
                    "status",
                    "generate",
                    "status",
                    "export_package",
                    "validate_package",
                    "export_messages",
                    "load_messages",
                ],
            )
            self.assertIn("draft_generation_complete", events)

    def test_incomplete_drafts_never_publish(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._paths(Path(temporary)).absolute()
            paths.project_root.mkdir(parents=True)
            core = ModuleType(MODEL_MODULE)
            core.create_model_assisted_project = lambda **_: None  # type: ignore[attr-defined]
            core.model_assisted_project_status = (  # type: ignore[attr-defined]
                lambda _: self._status(8_449)
            )
            core.generate_model_assisted_drafts = lambda **_: {  # type: ignore[attr-defined]
                "ok": True
            }

            def forbidden(**_: object) -> object:
                self.fail("incomplete drafts must not publish")

            core.export_model_assisted_supervision = forbidden  # type: ignore[attr-defined]
            core.validate_model_assisted_supervision = forbidden  # type: ignore[attr-defined]
            core.export_model_assisted_training_messages = forbidden  # type: ignore[attr-defined]
            core.load_model_assisted_training_messages = forbidden  # type: ignore[attr-defined]
            with patch.dict(sys.modules, {MODEL_MODULE: core}), patch(
                "oa_groundrag.data.grounded.supervision.workflow.prepare_expanded_corpus",
                return_value={"ok": True},
            ):
                with self.assertRaises(LandslideEvidenceError):
                    run_model_assisted_train_workflow(paths=paths)
            self.assertFalse(paths.annotation_package_root.exists())
            self.assertFalse(paths.training_messages_root.exists())

    def test_valid_published_roots_are_reused_without_project_or_generation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._paths(Path(temporary)).absolute()
            paths.annotation_package_root.mkdir(parents=True)
            paths.training_messages_root.mkdir(parents=True)
            core = ModuleType(MODEL_MODULE)

            def forbidden(*_: object, **__: object) -> object:
                self.fail("published-root reuse must not create or generate")

            core.create_model_assisted_project = forbidden  # type: ignore[attr-defined]
            core.model_assisted_project_status = forbidden  # type: ignore[attr-defined]
            core.generate_model_assisted_drafts = forbidden  # type: ignore[attr-defined]
            core.export_model_assisted_supervision = forbidden  # type: ignore[attr-defined]
            core.validate_model_assisted_supervision = (  # type: ignore[attr-defined]
                lambda **_: SimpleNamespace(manifest={
                    "record_count": MODEL_ASSISTED_RECORD_COUNT,
                    "eligible_count": 8_450,
                    "excluded_count": 0,
                    "authority_counts": {
                        "expert_verified": 500,
                        "model_generated_unreviewed": 7_950,
                    },
                    "exclusion_counts": {},
                    "reference_authority": "mixed_model_and_single_expert",
                })
            )
            core.export_model_assisted_training_messages = forbidden  # type: ignore[attr-defined]
            core.load_model_assisted_training_messages = (  # type: ignore[attr-defined]
                lambda _: SimpleNamespace(rows=range(MODEL_ASSISTED_RECORD_COUNT))
            )
            with patch.dict(sys.modules, {MODEL_MODULE: core}), patch(
                "oa_groundrag.data.grounded.supervision.workflow.prepare_expanded_corpus",
                return_value={"ok": True},
            ):
                result = run_model_assisted_train_workflow(paths=paths)
            self.assertEqual(result["stage"], "complete")

    def test_cli_has_fixed_argument_free_commands_and_v2_roots(self) -> None:
        spec = importlib.util.spec_from_file_location(
            "stage4_model_assisted_cli",
            CLI_PATH,
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(
            vars(module._parser().parse_args(["prepare-expanded-corpus"])),
            {"command": "prepare-expanded-corpus"},
        )
        self.assertEqual(
            vars(module._parser().parse_args(["run-train-workflow"])),
            {"command": "run-train-workflow"},
        )
        self.assertEqual(
            vars(module._parser().parse_args(["publish-compact-training"])),
            {"command": "publish-compact-training"},
        )
        self.assertEqual(
            vars(module._parser().parse_args(["validate-compact-training"])),
            {"command": "validate-compact-training"},
        )
        paths = module.MODEL_ASSISTED_WORKFLOW_PATHS
        expected = REPO_ROOT.parent / "benchmark" / "oa_grounded_stage4_v2"
        self.assertEqual(
            paths.extension_root,
            expected
            / "region_corpus"
            / "mask_grounded_region_corpus_train_extension_v2_7950",
        )
        self.assertEqual(
            paths.collection_root,
            expected
            / "region_collection"
            / "mask_grounded_region_train_collection_v2_8450",
        )
        self.assertEqual(
            paths.legacy_project_root,
            REPO_ROOT.parent
            / "benchmark"
            / "oa_grounded_stage4_v1"
            / "work"
            / "stage4_train_expert_v1",
        )
        self.assertEqual(
            paths.project_root,
            expected
            / "work"
            / "mask_grounded_region_model_assisted_train_v2_8450",
        )
        self.assertEqual(
            paths.annotation_package_root,
            expected
            / "annotations"
            / "mask_grounded_region_model_assisted_supervision_train_v2_source8450",
        )
        self.assertEqual(
            paths.training_messages_root,
            expected
            / "training_messages"
            / "mask_grounded_region_model_assisted_training_messages_train_v2_source8450",
        )
        self.assertEqual(
            module.COMPACT_TRAINING_ROOT,
            expected
            / "training_messages"
            / "mask_grounded_region_compact_training_messages_train_v3_6974",
        )
        source = CLI_PATH.read_text(encoding="utf-8")
        self.assertNotIn("nvidia-smi", source)
        self.assertNotIn("requests", source)
        self.assertNotIn("gradio", source.lower())


if __name__ == "__main__":
    unittest.main()
