from __future__ import annotations

import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from oa_groundrag.runtime.contracts import (
    RegionSource,
    UnifiedInferenceError,
    UnifiedReasonCode,
    UnifiedRequest,
    UnifiedTask,
)
from oa_groundrag.runtime.demo.access import (
    DemoAuthorizedSpatialInput,
    DemoInferenceAccess,
    DemoTestAccessController,
)
from oa_groundrag.runtime.demo.catalog import BenchmarkCatalog
from oa_groundrag.runtime.demo.runner import (
    WAITING_FOR_CANDIDATE,
    DemoCandidateKind,
    DemoCandidateSelection,
    DemoRunnerError,
    TASK_ORDER,
    UnifiedDemoRunner,
)

from tests.runtime.demo.helpers import (
    DatasetReadCounter,
    build_benchmark,
    fake_runtime,
    make_demo_config,
)


class TestAccessRuntimeTest(unittest.TestCase):
    def test_ordinary_runtime_rejects_demo_test_input_and_context_manifest_is_honest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binding = build_benchmark(root / "benchmark")
            controller = DemoTestAccessController(
                demo_root=root / "demo",
                allow_test_demo=True,
                benchmark_identity=binding.identity,
                config_sha256="c" * 64,
            )
            receipt = controller.issue(sample_id="test-small", action="INFERENCE")
            spatial = DemoAuthorizedSpatialInput(
                batch=object(),
                optical_image=Image.new("RGB", (16, 16)),
                sample_id="test-small",
                source="source_c",
                split="test",
                receipt_id=receipt.receipt_id,
            )
            request = UnifiedRequest(
                task=UnifiedTask.SEGMENT_ONLY,
                request_id="test-access",
                spatial_input=spatial,
                include_audit=True,
            )
            runtime, _calls = fake_runtime()
            with self.assertRaises(UnifiedInferenceError) as caught:
                runtime.plan(request)
            self.assertEqual(
                caught.exception.reason_code,
                UnifiedReasonCode.TEST_OR_SEALED_PATH_FORBIDDEN,
            )
            access = DemoInferenceAccess(
                receipt=receipt,
                demo_root=root / "demo",
                benchmark_identity=binding.identity,
                config_sha256="c" * 64,
            )
            wrong_binding = DemoInferenceAccess(
                receipt=receipt,
                demo_root=root / "demo",
                benchmark_identity={**binding.identity, "index_sha256": "d" * 64},
                config_sha256="c" * 64,
            )
            with self.assertRaises(UnifiedInferenceError) as binding_error:
                runtime.plan(request, access_context=wrong_binding)
            self.assertEqual(
                binding_error.exception.reason_code,
                UnifiedReasonCode.ARTIFACT_IDENTITY_MISMATCH,
            )
            runtime.plan(request, access_context=access)
            output = root / "demo" / "runtime_request"
            runtime.infer(request, output_root=output, access_context=access)
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertTrue(manifest["sealed_test_accessed"])
            self.assertFalse(manifest["blind_or_sealed_evaluation_property"])
            self.assertFalse(manifest["formal_test_evaluation"])
            self.assertEqual(manifest["access_receipt_id"], receipt.receipt_id)

            class FailingSpatial:
                def infer(self, request: UnifiedRequest, *, output_dir: Path):
                    raise RuntimeError("synthetic provider failure")

                def release(self) -> None:
                    return None

            runtime.spatial = FailingSpatial()
            failed_request = UnifiedRequest(
                task=UnifiedTask.SEGMENT_ONLY,
                request_id="test-access-failure",
                spatial_input=spatial,
                include_audit=True,
            )
            failed_output = root / "demo" / "runtime_failure"
            with self.assertRaises(UnifiedInferenceError):
                runtime.infer(
                    failed_request,
                    output_root=failed_output,
                    access_context=access,
                )
            failure_manifest = json.loads(
                (failed_output / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertTrue(failure_manifest["sealed_test_accessed"])
            self.assertFalse(failure_manifest["formal_test_evaluation"])


class UserMaskContractTest(unittest.TestCase):
    def test_png_l_binary_nonempty_is_copied_byte_for_byte(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "mask.png"
            values = np.zeros((16, 16), dtype=np.uint8)
            values[4:12, 5:10] = 255
            Image.fromarray(values).save(source, format="PNG")
            payload = source.read_bytes()
            destination = root / "staged" / "mask.png"
            audit = UnifiedDemoRunner.validate_and_stage_user_mask(
                source,
                destination,
                expected_size=(16, 16),
            )
            self.assertEqual(destination.read_bytes(), payload)
            self.assertTrue(audit["byte_preserving_copy"])

    def test_user_mask_rejects_palette_wrong_size_nonbinary_and_empty(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases: list[Path] = []
            palette = root / "palette.png"
            Image.new("P", (16, 16)).save(palette)
            cases.append(palette)
            wrong_size = root / "wrong.png"
            Image.new("L", (8, 8), color=255).save(wrong_size)
            cases.append(wrong_size)
            nonbinary = root / "nonbinary.png"
            Image.new("L", (16, 16), color=127).save(nonbinary)
            cases.append(nonbinary)
            empty = root / "empty.png"
            Image.new("L", (16, 16), color=0).save(empty)
            cases.append(empty)
            for index, source in enumerate(cases):
                with self.subTest(source=source.name), self.assertRaises(DemoRunnerError):
                    UnifiedDemoRunner.validate_and_stage_user_mask(
                        source,
                        root / f"out-{index}.png",
                        expected_size=(16, 16),
                    )


class UnifiedDemoRunnerTest(unittest.TestCase):
    def _setup(self, root: Path, *, fail_describe_once: bool = False):
        binding = build_benchmark(root / "benchmark")
        config = make_demo_config(root, binding)
        catalog = BenchmarkCatalog(binding)
        runtime, calls = fake_runtime(fail_describe_once=fail_describe_once)
        return config, catalog, UnifiedDemoRunner(config, catalog, runtime=runtime), calls

    @staticmethod
    def _mask(path: Path) -> Path:
        values = np.zeros((16, 16), dtype=np.uint8)
        values[3:12, 4:11] = 255
        Image.fromarray(values).save(path)
        return path

    def test_six_task_matrix_and_fixed_suite_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _config, catalog, runner, calls = self._setup(root)
            summary = runner.run(
                record=catalog.locate("val-large"),
                tasks=tuple(reversed(TASK_ORDER)),
                user_mask=self._mask(root / "user.png"),
                region_interpretation_source=RegionSource.USER_MASK,
            )
            self.assertEqual(tuple(item.task for item in summary.tasks), TASK_ORDER)
            self.assertTrue(all(item.status == "SUCCESS" for item in summary.tasks))
            self.assertEqual(
                runner.candidate_choices("val-large", split="val"),
                (5,),
            )
            self.assertEqual(
                runner.candidate_choices("val-large", split="train"),
                (),
            )
            self.assertTrue((summary.run_root / "manifest.json").is_file())
            self.assertTrue((summary.run_root / "viewer/region_interpretation.json").is_file())
            self.assertIn("spatial:SEGMENT_ONLY", calls)
            self.assertIn("evidence:REGION_UNDERSTANDING", calls)
            self.assertIn("rag:KNOWLEDGE_QA", calls)
            knowledge_viewer = runner.load_viewer(
                summary.run_root,
                UnifiedTask.KNOWLEDGE_QA,
            )
            self.assertFalse(
                knowledge_viewer["input_preview"][
                    "benchmark_payload_consumed_by_task"
                ]
            )
            self.assertIsNone(
                knowledge_viewer["input_preview"]["spatial_expert_inputs"][
                    "full_optical"
                ]
            )
            self.assertEqual(
                knowledge_viewer["input_preview"]["spatial_expert_inputs"][
                    "auxiliary_channel_previews"
                ],
                [],
            )

    def test_pure_knowledge_qa_does_not_load_benchmark_or_spatial_payload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _config, catalog, runner, calls = self._setup(root)
            with patch.object(catalog, "load", wraps=catalog.load) as load:
                summary = runner.run(
                    record=None,
                    tasks=(UnifiedTask.KNOWLEDGE_QA,),
                    instructions={
                        UnifiedTask.KNOWLEDGE_QA: "Why can InSAR alone be insufficient?"
                    },
                )
            load.assert_not_called()
            self.assertFalse(summary.benchmark_payload_consumed)
            self.assertFalse(summary.benchmark_payload_loaded_this_run)
            self.assertEqual(summary.input_scope, "KNOWLEDGE_ONLY")
            self.assertIsNone(summary.sample_id)
            self.assertFalse(summary.sealed_test_accessed)
            self.assertEqual(
                [value for value in calls if value.startswith("spatial:")],
                [],
            )
            manifest = json.loads(
                (summary.run_root / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertFalse(manifest["selection_unchanged"])
            self.assertIsNone(manifest["frozen_evaluation_name"])
            self.assertIsNone(manifest["frozen_baseline_record_id"])
            viewer = runner.load_viewer(summary.run_root, UnifiedTask.KNOWLEDGE_QA)
            self.assertFalse(
                viewer["input_preview"]["benchmark_payload_consumed_by_task"]
            )
            self.assertEqual(
                viewer["request_preview"]["effective_instruction"],
                "Why can InSAR alone be insufficient?",
            )

    def test_pure_knowledge_qa_with_test_metadata_creates_no_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binding = build_benchmark(root / "benchmark")
            config = make_demo_config(root, binding, allow_test_demo=False)
            controller = DemoTestAccessController(
                demo_root=config.demo_root,
                allow_test_demo=False,
                benchmark_identity=binding.identity,
                config_sha256=config.config_sha256,
            )
            reads = DatasetReadCounter()
            from oa_groundrag.data.oa_auxseg.dataset import BenchmarkDataset

            catalog = BenchmarkCatalog(
                binding,
                access_controller=controller,
                dataset_factory=reads.wrap(BenchmarkDataset),
            )
            test_metadata = catalog.locate("test-small")
            self.assertEqual(test_metadata.split, "test")
            runtime, calls = fake_runtime()
            runner = UnifiedDemoRunner(config, catalog, runtime=runtime)
            summary = runner.run(
                record=test_metadata,
                tasks=(UnifiedTask.KNOWLEDGE_QA,),
            )
            self.assertEqual(reads.calls, [])
            self.assertFalse(summary.sealed_test_accessed)
            self.assertIsNone(summary.access_receipt_id)
            receipts = config.demo_root / "test_access_receipts"
            self.assertTrue(not receipts.exists() or not any(receipts.iterdir()))
            self.assertNotIn("spatial:KNOWLEDGE_QA", calls)

    def test_candidate_waits_then_replays_same_spatial_result_without_rerun(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _config, catalog, runner, calls = self._setup(root)
            record = catalog.locate("val-large")
            first = runner.run(
                record=record,
                tasks=(UnifiedTask.SEGMENT_ONLY, UnifiedTask.REGION_INTERPRETATION),
            )
            self.assertEqual(
                [item.status for item in first.tasks],
                ["SUCCESS", WAITING_FOR_CANDIDATE],
            )
            self.assertNotIn("evidence:REGION_INTERPRETATION", calls)
            snapshot = runner.active_snapshot
            self.assertIsNotNone(snapshot)
            assert snapshot is not None
            self.assertEqual(len(snapshot.candidate_previews), 1)
            preview = snapshot.candidate_previews[0]
            self.assertEqual(preview["candidate_id"], 5)
            self.assertEqual(preview["bbox_xyxy"], [2, 2, 14, 14])
            self.assertGreater(preview["area_pixels"], 0)
            self.assertEqual(preview["confidence"], 0.9)
            self.assertTrue(Path(str(preview["mask_path"])).is_file())
            self.assertTrue(Path(str(preview["overlay_path"])).is_file())
            ui_payload = runner.candidate_ui_payload(
                sample_id=record.sample_id,
                split=record.split,
            )
            self.assertNotIn("choices", ui_payload)
            self.assertNotIn("gallery", ui_payload)
            self.assertEqual(
                [value["kind"] for value in ui_payload["options"]],
                ["CANDIDATE", "EXPLICIT_GLOBAL"],
            )
            self.assertEqual(ui_payload["options"][0]["candidate_id"], 5)
            self.assertEqual(ui_payload["options"][0]["area_pixels"], preview["area_pixels"])
            selected = DemoCandidateSelection(
                snapshot.snapshot_id,
                DemoCandidateKind.CANDIDATE,
                5,
            )
            spatial_calls = calls.count("spatial:SEGMENT_ONLY")
            second = runner.run(
                record=record,
                tasks=(UnifiedTask.REGION_INTERPRETATION,),
                candidate_selection=selected,
            )
            self.assertEqual(second.tasks[0].status, "SUCCESS")
            self.assertEqual(calls.count("spatial:SEGMENT_ONLY"), spatial_calls)
            self.assertNotIn("spatial:REGION_INTERPRETATION", calls)
            response = second.tasks[0].response or {}
            self.assertEqual(
                response["region_selection"]["selected_candidate_id"],
                5,
            )
            viewer = runner.load_viewer(second.run_root, UnifiedTask.REGION_INTERPRETATION)
            self.assertTrue(viewer["execution_trace"]["spatial_result_replayed"])
            self.assertFalse(
                viewer["execution_trace"]["candidate_decision"][
                    "explicit_global_confirmed"
                ]
            )

    def test_explicit_global_is_audited_and_not_implicit_candidate_interpretation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _config, catalog, runner, _calls = self._setup(root)
            record = catalog.locate("val-large")
            runner.run(record=record, tasks=(UnifiedTask.SEGMENT_ONLY,))
            assert runner.active_snapshot is not None
            selected = DemoCandidateSelection(
                runner.active_snapshot.snapshot_id,
                DemoCandidateKind.EXPLICIT_GLOBAL,
            )
            summary = runner.run(
                record=record,
                tasks=(UnifiedTask.REGION_INTERPRETATION,),
                candidate_selection=selected,
            )
            response = summary.tasks[0].response or {}
            self.assertEqual(
                response["region_selection"]["status"],
                "FALLBACK_GLOBAL",
            )
            viewer = runner.load_viewer(summary.run_root, UnifiedTask.REGION_INTERPRETATION)
            decision = viewer["execution_trace"]["candidate_decision"]
            self.assertTrue(decision["explicit_global_confirmed"])
            self.assertEqual(decision["fallback_reason"], "CANDIDATE_ID_MISSING")

    def test_candidate_token_is_bound_to_sample_and_latest_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _config, catalog, runner, _calls = self._setup(root)
            runner.run(
                record=catalog.locate("val-large"),
                tasks=(UnifiedTask.SEGMENT_ONLY,),
            )
            assert runner.active_snapshot is not None
            old = DemoCandidateSelection(
                runner.active_snapshot.snapshot_id,
                DemoCandidateKind.CANDIDATE,
                5,
            )
            with self.assertRaises(DemoRunnerError):
                runner.resolve_candidate_selection(
                    old.token,
                    sample_id="val-empty",
                    split="val",
                )
            runner.run(
                record=catalog.locate("val-empty"),
                tasks=(UnifiedTask.SEGMENT_ONLY,),
            )
            with self.assertRaises(DemoRunnerError):
                runner.resolve_candidate_selection(
                    old.token,
                    sample_id="val-large",
                    split="val",
                )

    def test_suite_saves_failure_and_continues_independent_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _config, catalog, runner, calls = self._setup(root, fail_describe_once=True)
            summary = runner.run(
                record=catalog.locate("val-large"),
                tasks=(UnifiedTask.VLM_ONLY, UnifiedTask.SEGMENT_ONLY),
            )
            self.assertEqual([item.status for item in summary.tasks], ["FAILED", "SUCCESS"])
            self.assertIn("spatial:SEGMENT_ONLY", calls)
            assert summary.tasks[0].task_root is not None
            failure = json.loads((
                summary.run_root / summary.tasks[0].task_root / "failure.json"
            ).read_text(encoding="utf-8"))
            self.assertEqual(failure["reason_code"], "SHARED_MLLM_FAILED")

    def test_full_preflight_happens_before_any_provider_call(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, catalog, runner, calls = self._setup(root)
            with self.assertRaises(DemoRunnerError):
                runner.run(
                    record=catalog.locate("val-large"),
                    tasks=(UnifiedTask.SEGMENT_ONLY, UnifiedTask.REGION_UNDERSTANDING),
                )
            self.assertEqual(calls, [])
            runs = config.demo_root / "runs"
            self.assertTrue(not runs.exists() or not any(runs.iterdir()))

    def test_retired_frozen_data_mode_fails_closed_before_provider_or_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, catalog, runner, calls = self._setup(root)
            record = catalog.locate("val-large")
            with self.assertRaisesRegex(DemoRunnerError, "旧 Frozen data_mode 已退役"):
                runner.run(
                    record=record,
                    tasks=(UnifiedTask.SEGMENT_ONLY,),
                    data_mode="Frozen Evaluation / Read Only",
                )
            self.assertEqual(calls, [])
            runs = config.demo_root / "runs"
            self.assertTrue(not runs.exists() or not any(runs.iterdir()))


if __name__ == "__main__":
    unittest.main()
