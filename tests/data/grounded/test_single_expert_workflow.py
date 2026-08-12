from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from fixture_helpers import no_target_output, target_output
from single_expert_fixture_helpers import (
    build_annotation_asset,
    draft_run,
)

from oa_groundrag.landslide_evidence.single_expert import (
    AnnotationIntendedUse,
    MODEL_DRAFT_SCHEMA,
    VERIFIED_ANNOTATION_SCHEMA,
    _write_status,
    load_draft_runs,
    load_annotation_project,
    load_model_drafts,
    write_draft_results,
)
from oa_groundrag.landslide_evidence.contracts import LandslideEvidenceError
from oa_groundrag.landslide_evidence.single_expert_workflow import (
    TrainWorkflowPaths,
    run_train_annotation_workflow,
)
from oa_groundrag.phase3.common import (
    atomic_write_json,
    canonical_json,
    read_json,
    sha256_file,
    sha256_text,
)
from oa_groundrag.phase4.outputs import region_output_template


REPO_ROOT = Path(__file__).resolve().parents[2]
PROMPT = REPO_ROOT / "configs/stage4_landslide_evidence/single_expert_prompt_v1.txt"
DRAFT_CONFIG = REPO_ROOT / "configs/stage4_landslide_evidence/single_expert_qwen3vl_8b_v1.yaml"
CLI_PATH = REPO_ROOT / "scripts/stage4_landslide_evidence/run_single_expert_annotation.py"


class SingleExpertWorkflowTest(unittest.TestCase):
    def _paths(
        self,
        root: Path,
        *,
        prompt_path: Path = PROMPT,
    ) -> TrainWorkflowPaths:
        return TrainWorkflowPaths(
            corpus_root=build_annotation_asset(root / "corpus", kind="train"),
            project_root=root / "work" / "project",
            annotation_package_root=root / "annotations" / "package",
            training_messages_root=root / "training_messages" / "messages",
            prompt_path=prompt_path,
            draft_config_path=DRAFT_CONFIG,
        )

    @staticmethod
    def _bulk_generate(**kwargs: object) -> dict[str, object]:
        project_root = Path(kwargs["project_root"])
        partition = str(kwargs["partition"])
        _, assignments = load_annotation_project(project_root)
        existing = load_model_drafts(project_root, assignments=assignments)
        selected = [
            row for row in assignments
            if row["partition"] == partition and row["record_id"] not in existing
        ]
        run = draft_run(
            [row["record_id"] for row in selected],
            partition=partition,
            suffix=f"workflow-{partition}",
            prompt_text=(project_root / "prompt.txt").read_text(encoding="utf-8"),
        )
        drafts = []
        for assignment in selected:
            description = (
                no_target_output()
                if assignment["target_status"] == "no_target"
                else target_output()
            )
            drafts.append({
                "schema_version": MODEL_DRAFT_SCHEMA,
                "draft_id": f"draft_{assignment['record_id']}",
                "draft_run_id": run["draft_run_id"],
                "record_id": assignment["record_id"],
                "asset_identity_sha256": assignment["asset_identity_sha256"],
                "messages_sha256": sha256_text(f"workflow:{assignment['record_id']}"),
                "raw_output": canonical_json(description),
                "parse_status": "valid",
                "description": description,
                "failure": None,
            })
        write_draft_results(
            project_root,
            draft_run=run,
            new_drafts=drafts,
            freeze_prompt=partition == "remaining",
        )
        callback = kwargs.get("progress_callback")
        if callable(callback):
            callback({"event": "generation_complete", "partition": partition})
        return {"ok": True, "partition": partition, "generated": len(drafts)}

    @staticmethod
    def _verify_partition(**kwargs: object) -> None:
        project_root = Path(kwargs["project_root"])
        partition = str(kwargs["partition"])
        _, assignments = load_annotation_project(project_root)
        drafts = load_model_drafts(project_root, assignments=assignments)
        for assignment in assignments:
            if assignment["partition"] != partition:
                continue
            description = (
                no_target_output()
                if assignment["target_status"] == "no_target"
                else target_output()
            )
            atomic_write_json(
                project_root / "verified" / f"{assignment['record_id']}.json",
                {
                    "schema_version": VERIFIED_ANNOTATION_SCHEMA,
                    "record_id": assignment["record_id"],
                    "asset_identity_sha256": assignment["asset_identity_sha256"],
                    "draft_id": drafts[assignment["record_id"]]["draft_id"],
                    "annotator": "expert",
                    "verification_status": "expert_verified",
                    "description": description,
                },
            )
        _write_status(project_root)

    def test_one_command_stops_after_calibration_then_resumes_and_recovers_publish(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._paths(Path(temporary))
            generated_partitions: list[str] = []
            served_partitions: list[str] = []
            progress_events: list[str] = []

            def generate(**kwargs: object) -> dict[str, object]:
                generated_partitions.append(str(kwargs["partition"]))
                return self._bulk_generate(**kwargs)

            def serve(**kwargs: object) -> None:
                served_partitions.append(str(kwargs["partition"]))
                self.assertEqual(kwargs["view"], "pending")
                self.assertIs(kwargs["auto_close_partition"], True)
                self._verify_partition(**kwargs)

            def publish_package(**kwargs: object) -> dict[str, object]:
                output = Path(kwargs["output_root"])
                output.mkdir(parents=True, exist_ok=False)
                return {"ok": True}

            training_attempts = 0

            def publish_training(**kwargs: object) -> dict[str, object]:
                nonlocal training_attempts
                training_attempts += 1
                if training_attempts == 1:
                    raise RuntimeError("simulated interruption after package publish")
                output = Path(kwargs["output_root"])
                output.mkdir(parents=True, exist_ok=False)
                return {"ok": True}

            package = SimpleNamespace(manifest={"training_eligible": True})
            artifact = SimpleNamespace(rows=tuple(range(500)))
            patches = (
                patch(
                    "oa_groundrag.landslide_evidence.single_expert_workflow.generate_annotation_drafts",
                    side_effect=generate,
                ),
                patch(
                    "oa_groundrag.landslide_evidence.single_expert_workflow.export_verified_annotations",
                    side_effect=publish_package,
                ),
                patch(
                    "oa_groundrag.landslide_evidence.single_expert_workflow.validate_verified_annotation_package",
                    return_value=package,
                ),
                patch(
                    "oa_groundrag.landslide_evidence.single_expert_workflow.export_training_messages",
                    side_effect=publish_training,
                ),
                patch(
                    "oa_groundrag.landslide_evidence.single_expert_workflow.load_training_message_artifact",
                    return_value=artifact,
                ),
            )
            with patches[0], patches[1], patches[2], patches[3], patches[4]:
                first = run_train_annotation_workflow(
                    paths=paths,
                    port=7860,
                    serve_callback=serve,
                    progress_callback=lambda row: progress_events.append(str(row["event"])),
                )
                self.assertEqual(
                    first["stage"],
                    "calibration_complete_prompt_adjustment_required",
                )
                self.assertFalse(paths.annotation_package_root.exists())
                workflow_state = paths.project_root / "workflow_state.json"
                self.assertEqual(
                    read_json(workflow_state)["phase"],
                    "awaiting_prompt_confirmation",
                )
                # 模拟第 20 条完成后、状态机边界落盘前异常退出；重跑仍必须先停一次。
                workflow_state.unlink()
                recovered_boundary = run_train_annotation_workflow(
                    paths=paths,
                    port=7860,
                    serve_callback=serve,
                )
                self.assertEqual(
                    recovered_boundary["stage"],
                    "calibration_complete_prompt_adjustment_required",
                )
                with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                    run_train_annotation_workflow(
                        paths=paths,
                        port=7860,
                        serve_callback=serve,
                    )
                self.assertTrue(paths.annotation_package_root.is_dir())
                self.assertFalse(paths.training_messages_root.exists())
                completed = run_train_annotation_workflow(paths=paths, port=7860)
                self.assertEqual(completed["stage"], "complete")
                self.assertEqual(completed["training_message_count"], 500)
                repeated = run_train_annotation_workflow(paths=paths, port=7860)
                self.assertEqual(repeated["stage"], "complete")

            self.assertEqual(generated_partitions, ["calibration", "remaining"])
            self.assertEqual(served_partitions, ["calibration", "remaining"])
            project, _ = load_annotation_project(paths.project_root)
            self.assertIsNotNone(project["frozen_prompt_sha256"])
            self.assertIsNotNone(project["frozen_draft_config_sha256"])
            self.assertIn("calibration_quality_assessed", progress_events)
            self.assertIn("calibration_ui_starting", progress_events)

    def test_target_template_copy_stops_before_calibration_ui(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._paths(Path(temporary))
            served = False

            def template_generate(**kwargs: object) -> dict[str, object]:
                project_root = Path(kwargs["project_root"])
                partition = str(kwargs["partition"])
                _, assignments = load_annotation_project(project_root)
                selected = [
                    row for row in assignments if row["partition"] == partition
                ]
                run = draft_run(
                    [row["record_id"] for row in selected],
                    partition=partition,
                    suffix="template-copy",
                    prompt_text=(project_root / "prompt.txt").read_text(encoding="utf-8"),
                )
                drafts = []
                for assignment in selected:
                    description = region_output_template(
                        assignment["target_status"]
                    )
                    drafts.append({
                        "schema_version": MODEL_DRAFT_SCHEMA,
                        "draft_id": f"copy_{assignment['record_id']}",
                        "draft_run_id": run["draft_run_id"],
                        "record_id": assignment["record_id"],
                        "asset_identity_sha256": assignment["asset_identity_sha256"],
                        "messages_sha256": sha256_text(assignment["record_id"]),
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
                return {"ok": True}

            def serve(**_: object) -> None:
                nonlocal served
                served = True

            with patch(
                "oa_groundrag.landslide_evidence.single_expert_workflow.generate_annotation_drafts",
                side_effect=template_generate,
            ):
                with self.assertRaises(LandslideEvidenceError) as raised:
                    run_train_annotation_workflow(
                        paths=paths,
                        serve_callback=serve,
                    )
            self.assertEqual(raised.exception.code, "DRAFT_QUALITY_FAILED")
            self.assertFalse(served)
            self.assertEqual(raised.exception.details["target_template_copies"], 16)

    def test_cli_freezes_benchmark_paths_and_exposes_only_port_for_one_click(self) -> None:
        spec = importlib.util.spec_from_file_location("stage4_single_expert_cli", CLI_PATH)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        args = vars(module._parser().parse_args(["run-train-workflow"]))
        self.assertEqual(args, {"command": "run-train-workflow", "port": 7860})
        self.assertEqual(
            module.TRAIN_WORKFLOW_PATHS.project_root,
            REPO_ROOT.parent / "benchmark/oa_grounded_stage4_v1/work/stage4_train_expert_v1",
        )
        source = CLI_PATH.read_text(encoding="utf-8")
        self.assertNotIn("annotator_id", source)
        self.assertNotIn("nvidia-smi", source)
        self.assertNotIn("22 GiB", source)

    def test_repo_prompt_syncs_only_at_confirmation_and_then_remains_frozen(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo_prompt = root / "repo_prompt.txt"
            initial_prompt = "initial calibration prompt\n"
            intermediate_prompt = "must not enter calibration\n"
            remaining_prompt = "confirmed remaining prompt\n"
            post_freeze_prompt = "must not replace frozen prompt\n"
            repo_prompt.write_text(initial_prompt, encoding="utf-8")
            paths = self._paths(root, prompt_path=repo_prompt)

            with patch(
                "oa_groundrag.landslide_evidence.single_expert_workflow.generate_annotation_drafts",
                side_effect=self._bulk_generate,
            ):
                # 首次 UI 中断时仍在 calibration，repo prompt 后续变化不得
                # 改写建项时副本或已登记的 calibration provenance。
                with self.assertRaisesRegex(RuntimeError, "calibration interrupted"):
                    run_train_annotation_workflow(
                        paths=paths,
                        serve_callback=lambda **_: (_ for _ in ()).throw(
                            RuntimeError("calibration interrupted")
                        ),
                    )
                calibration_runs = load_draft_runs(paths.project_root)
                self.assertEqual(len(calibration_runs), 1)
                calibration_run = calibration_runs[0]
                self.assertEqual(calibration_run["partition"], "calibration")
                self.assertEqual(calibration_run["prompt_text"], initial_prompt)
                repo_prompt.write_text(intermediate_prompt, encoding="utf-8")

                calibration_events: list[str] = []

                def verify_calibration(**kwargs: object) -> None:
                    self.assertEqual(
                        (paths.project_root / "prompt.txt").read_text(encoding="utf-8"),
                        initial_prompt,
                    )
                    self._verify_partition(**kwargs)

                boundary = run_train_annotation_workflow(
                    paths=paths,
                    serve_callback=verify_calibration,
                    progress_callback=lambda row: calibration_events.append(str(row["event"])),
                )
                self.assertEqual(
                    boundary["stage"],
                    "calibration_complete_prompt_adjustment_required",
                )
                self.assertNotIn("prompt_synchronized", calibration_events)
                self.assertEqual(load_draft_runs(paths.project_root)[0], calibration_run)

                repo_prompt.write_text("   \n", encoding="utf-8")
                with self.assertRaisesRegex(LandslideEvidenceError, "prompt 不能为空"):
                    run_train_annotation_workflow(paths=paths)
                self.assertEqual(
                    read_json(paths.project_root / "workflow_state.json")["phase"],
                    "awaiting_prompt_confirmation",
                )
                self.assertEqual(
                    (paths.project_root / "prompt.txt").read_text(encoding="utf-8"),
                    initial_prompt,
                )

                repo_prompt.write_text(remaining_prompt, encoding="utf-8")
                confirmation_events: list[dict[str, object]] = []
                with self.assertRaisesRegex(RuntimeError, "remaining interrupted"):
                    run_train_annotation_workflow(
                        paths=paths,
                        serve_callback=lambda **_: (_ for _ in ()).throw(
                            RuntimeError("remaining interrupted")
                        ),
                        progress_callback=lambda row: confirmation_events.append(dict(row)),
                    )
                synchronized = [
                    row for row in confirmation_events
                    if row["event"] == "prompt_synchronized"
                ]
                self.assertEqual(len(synchronized), 1)
                self.assertEqual(
                    synchronized[0]["prompt_sha256"],
                    sha256_file(paths.project_root / "prompt.txt"),
                )
                self.assertEqual(
                    (paths.project_root / "prompt.txt").read_text(encoding="utf-8"),
                    remaining_prompt,
                )
                project, _ = load_annotation_project(paths.project_root)
                self.assertEqual(
                    project["frozen_prompt_sha256"],
                    sha256_text(remaining_prompt),
                )
                self.assertIsNotNone(project["frozen_draft_config_sha256"])
                self.assertEqual(
                    [run["prompt_text"] for run in load_draft_runs(paths.project_root)],
                    [initial_prompt, remaining_prompt],
                )

                # phase=remaining 且身份已冻结后，后续 repo 文件漂移不再同步。
                repo_prompt.write_text(post_freeze_prompt, encoding="utf-8")
                resumed_events: list[str] = []
                with self.assertRaisesRegex(RuntimeError, "remaining interrupted again"):
                    run_train_annotation_workflow(
                        paths=paths,
                        serve_callback=lambda **_: (_ for _ in ()).throw(
                            RuntimeError("remaining interrupted again")
                        ),
                        progress_callback=lambda row: resumed_events.append(str(row["event"])),
                    )
                self.assertNotIn("prompt_synchronized", resumed_events)
                self.assertEqual(
                    (paths.project_root / "prompt.txt").read_text(encoding="utf-8"),
                    remaining_prompt,
                )
                self.assertEqual(load_draft_runs(paths.project_root)[0], calibration_run)

    def test_invalid_existing_publish_root_is_rejected_without_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._paths(Path(temporary))
            paths.annotation_package_root.mkdir(parents=True)
            sentinel = paths.annotation_package_root / "sentinel.txt"
            sentinel.write_text("keep", encoding="utf-8")
            with self.assertRaises(LandslideEvidenceError):
                run_train_annotation_workflow(paths=paths)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")
            self.assertFalse(paths.project_root.exists())


if __name__ == "__main__":
    unittest.main()
