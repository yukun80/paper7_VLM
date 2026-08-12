from __future__ import annotations

import json
import tempfile
from pathlib import Path
import unittest

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
from oa_groundrag.runtime.demo.catalog import BenchmarkCatalog, FrozenEvaluationItem
from oa_groundrag.runtime.demo.runner import DemoRunnerError, TASK_ORDER, UnifiedDemoRunner

from tests.runtime.demo.helpers import build_benchmark, fake_runtime, make_demo_config


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
                candidate_region_id=5,
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
            failure = json.loads(
                (summary.run_root / summary.tasks[0].task_root / "failure.json").read_text(encoding="utf-8")
            )
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

    def test_frozen_run_never_consumes_reference_mask_and_marks_selection_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _config, catalog, runner, _calls = self._setup(root)
            record = catalog.locate("val-large")
            frozen = FrozenEvaluationItem(
                ordinal=0,
                evaluation_name="synthetic-frozen",
                baseline_record={
                    "sample_id": record.sample_id,
                    "source": record.source,
                    "split": "val",
                    "record_id": "baseline",
                    "target_status": "target_present",
                },
                counterfactual_records={},
                root=root,
            )
            summary = runner.run(
                record=record,
                tasks=(UnifiedTask.SEGMENT_ONLY,),
                data_mode="Frozen Evaluation / Read Only",
                frozen_item=frozen,
            )
            manifest = json.loads((summary.run_root / "manifest.json").read_text(encoding="utf-8"))
            run_summary = json.loads((summary.run_root / "run_summary.json").read_text(encoding="utf-8"))
            self.assertTrue(manifest["selection_unchanged"])
            self.assertFalse(manifest["formal_evaluation"])
            self.assertFalse(run_summary["reference_or_gt_mask_consumed"])


if __name__ == "__main__":
    unittest.main()
