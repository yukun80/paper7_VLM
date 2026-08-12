from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any, Mapping, Sequence

from oa_groundrag.unified.contracts import (
    BenchmarkSampleRef,
    RegionSelectionStatus,
    RegionSource,
    UnifiedInferenceError,
    UnifiedReasonCode,
    UnifiedRequest,
    UnifiedTask,
)
from oa_groundrag.unified.providers import (
    GroundedEvidenceResult,
    SpatialCandidate,
    SpatialResult,
    TextRAGResult,
)
from oa_groundrag.unified.router import CapabilityRouter
from oa_groundrag.unified.runtime import UnifiedInferenceRuntime


class FakeSpatial:
    def __init__(self, calls: list[str], *, candidates: bool = True, no_target: bool = False) -> None:
        self.calls = calls
        self.with_candidates = candidates
        self.no_target = no_target

    def infer(self, request: UnifiedRequest, *, output_dir: Path) -> SpatialResult:
        self.calls.append("spatial")
        output_dir.mkdir(parents=True)
        (output_dir / "global-mask.txt").write_text("mask", encoding="utf-8")
        (output_dir / "probability.txt").write_text("probability", encoding="utf-8")
        candidates = (
            SpatialCandidate(7, "candidate-7", (1, 2, 5, 7), (3.0, 4.0), 20, 0.9, "spatial/candidate-7.txt"),
            SpatialCandidate(3, "candidate-3", (8, 9, 12, 14), (10.0, 11.0), 15, 0.8, "spatial/candidate-3.txt"),
        ) if self.with_candidates else ()
        return SpatialResult(
            sample_id="sample-1",
            source="source-a",
            split="val",
            optical_image="optical-image",
            global_mask="global-mask",
            mask_probability="probability",
            no_target=self.no_target,
            no_target_score=0.75 if self.no_target else 0.1,
            candidates=candidates,
            active_modalities=("dem",),
            mask_reference="spatial/global-mask.txt",
            mask_probability_reference="spatial/probability.txt",
        )

    def release(self) -> None:
        self.calls.append("release:spatial")


class FailingSpatial(FakeSpatial):
    def infer(self, request: UnifiedRequest, *, output_dir: Path) -> SpatialResult:
        self.calls.append("spatial")
        raise RuntimeError("checkpoint missing")


class FakeShared:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def describe(self, request: UnifiedRequest) -> str:
        self.calls.append("mllm:describe")
        return "scene description"

    def generate_visual(self, messages: Sequence[Mapping[str, Any]]) -> str:
        self.calls.append("mllm:observation")
        return "raw observation"

    def generate_text(self, messages: Sequence[Mapping[str, Any]], *, packet: Mapping[str, Any]) -> str:
        self.calls.append("mllm:interpretation")
        return "raw interpretation"

    def release(self) -> None:
        self.calls.append("release:mllm")


class FakeEvidence:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls
        self.last_selection = None
        self.last_mask = None

    def build(
        self,
        request: UnifiedRequest,
        *,
        optical_image: Any,
        mask: Any,
        selection: Any,
        output_dir: Path,
        sample_id: str,
        source: str,
        split: str,
        no_target: bool,
    ) -> GroundedEvidenceResult:
        self.calls.append("evidence")
        self.last_selection = selection
        self.last_mask = mask
        output_dir.mkdir(parents=True)
        (output_dir / "mask.png").write_bytes(b"mask")
        return GroundedEvidenceResult(
            messages=({"role": "user", "content": []},),
            program_facts={"mask": {"area_pixels": 0 if no_target else 20}},
            target_status="no_target" if no_target else "target_present",
            mask_reference="grounded/mask.png",
            limitations=("NO_TARGET_NO_REGION_GEOMETRY",) if no_target else (),
            metadata={},
        )

    def parse_observation(self, raw_output: str, *, evidence: GroundedEvidenceResult) -> Mapping[str, Any]:
        self.calls.append("parse:observation")
        return {
            "target_status": evidence.target_status,
            "short_summary": "visual observation",
            "limitations": ["empty mask"] if evidence.target_status == "no_target" else [],
        }


class FakeRAG:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def retrieve(
        self,
        request: UnifiedRequest,
        *,
        observation: Mapping[str, Any],
        program_facts: Mapping[str, Any],
        target_status: str,
        available_modalities: Sequence[str],
        candidate_count: int,
    ) -> TextRAGResult:
        self.calls.append("rag")
        packet = {
            "packet_id": "packet-1",
            "items": [{
                "evidence_id": "ev-1",
                "source_id": "source-1",
                "source_title": "Title",
                "pdf_page": 2,
                "section": "S",
            }],
        }
        return TextRAGResult(
            packet=packet,
            messages=({"role": "user", "content": []},),
            citations=({"evidence_id": "ev-1", "source_title": "Title", "pdf_page": 2},),
            metadata={"target_status": target_status},
        )

    def parse_generation(self, raw_output: str, *, result: TextRAGResult) -> Mapping[str, Any]:
        self.calls.append("parse:interpretation")
        return {
            "summary": {"text": "professional interpretation", "evidence_ids": ["ev-1"]},
            "supporting_interpretations": [],
            "alternative_explanations": [],
            "limitations": [],
            "recommended_verification": [],
        }

    def release(self) -> None:
        self.calls.append("release:rag")


def spatial_ref() -> BenchmarkSampleRef:
    return BenchmarkSampleRef(split="val", sample_id="sample-1")


class UnifiedRuntimeTest(unittest.TestCase):
    def make_runtime(
        self,
        *,
        candidates: bool = True,
        no_target: bool = False,
        failing_spatial: bool = False,
    ) -> tuple[UnifiedInferenceRuntime, list[str], FakeEvidence]:
        calls: list[str] = []
        spatial_class = FailingSpatial if failing_spatial else FakeSpatial
        evidence = FakeEvidence(calls)
        runtime = UnifiedInferenceRuntime(
            spatial=spatial_class(calls, candidates=candidates, no_target=no_target),
            shared_mllm=FakeShared(calls),
            evidence=evidence,
            text_rag=FakeRAG(calls),
        )
        return runtime, calls, evidence

    def run_request(self, runtime: UnifiedInferenceRuntime, request: UnifiedRequest) -> Any:
        with tempfile.TemporaryDirectory() as directory:
            return runtime.infer(request, output_root=Path(directory) / "result")

    @staticmethod
    def capability_calls(calls: Sequence[str]) -> list[str]:
        return [value for value in calls if not value.startswith("release:")]

    def test_router_matrix(self) -> None:
        router = CapabilityRouter()
        requests = (
            UnifiedRequest(UnifiedTask.VLM_ONLY, "1", "describe", (Path("/tmp/a.png"),)),
            UnifiedRequest(UnifiedTask.SEGMENT_ONLY, "2", spatial_input=spatial_ref()),
            UnifiedRequest(UnifiedTask.REGION_UNDERSTANDING, "3", "region", (Path("/tmp/a.png"),), Path("/tmp/m.png"), region_source=RegionSource.USER_MASK),
            UnifiedRequest(UnifiedTask.SEGMENT_AND_UNDERSTAND, "4", "observe", spatial_input=spatial_ref(), region_source=RegionSource.OA_AUXSEG_GLOBAL),
            UnifiedRequest(UnifiedTask.KNOWLEDGE_QA, "5", "why"),
            UnifiedRequest(UnifiedTask.REGION_INTERPRETATION, "6", "interpret", spatial_input=spatial_ref(), region_source=RegionSource.OA_AUXSEG_CANDIDATE),
        )
        observed = [router.route(request).to_dict() for request in requests]
        self.assertEqual(
            [(row["needs_spatial"], row["needs_shared_mllm"], row["needs_region"], row["needs_rag"]) for row in observed],
            [(False, True, False, False), (True, False, False, False), (False, True, True, False), (True, True, True, False), (False, True, False, True), (True, True, True, True)],
        )

    def test_vlm_only_calls_only_shared(self) -> None:
        runtime, calls, _ = self.make_runtime()
        self.run_request(runtime, UnifiedRequest(UnifiedTask.VLM_ONLY, "1", "describe", (Path("/tmp/a.png"),)))
        self.assertEqual(self.capability_calls(calls), ["mllm:describe"])

    def test_segment_only_calls_only_spatial(self) -> None:
        runtime, calls, _ = self.make_runtime()
        response = self.run_request(runtime, UnifiedRequest(UnifiedTask.SEGMENT_ONLY, "2", spatial_input=spatial_ref()))
        self.assertEqual(self.capability_calls(calls), ["spatial"])
        self.assertEqual(response.mask_reference, "spatial/global-mask.txt")

    def test_region_understanding_never_calls_spatial_or_rag(self) -> None:
        runtime, calls, _ = self.make_runtime()
        request = UnifiedRequest(UnifiedTask.REGION_UNDERSTANDING, "3", "region", (Path("/tmp/a.png"),), Path("/tmp/m.png"), region_source=RegionSource.USER_MASK)
        self.run_request(runtime, request)
        self.assertEqual(self.capability_calls(calls), ["evidence", "mllm:observation", "parse:observation"])

    def test_segment_and_understand_order(self) -> None:
        runtime, calls, _ = self.make_runtime()
        request = UnifiedRequest(UnifiedTask.SEGMENT_AND_UNDERSTAND, "4", "observe", spatial_input=spatial_ref(), region_source=RegionSource.OA_AUXSEG_GLOBAL)
        self.run_request(runtime, request)
        self.assertEqual(self.capability_calls(calls), ["spatial", "evidence", "mllm:observation", "parse:observation"])
        self.assertEqual(
            calls,
            ["spatial", "release:spatial", "evidence", "mllm:observation", "parse:observation", "release:mllm"],
        )

    def test_knowledge_qa_calls_rag_then_mllm(self) -> None:
        runtime, calls, _ = self.make_runtime()
        self.run_request(runtime, UnifiedRequest(UnifiedTask.KNOWLEDGE_QA, "5", "why"))
        self.assertEqual(self.capability_calls(calls), ["rag", "mllm:interpretation", "parse:interpretation"])

    def test_region_interpretation_order_and_exact_candidate(self) -> None:
        runtime, calls, evidence = self.make_runtime()
        request = UnifiedRequest(UnifiedTask.REGION_INTERPRETATION, "6", "interpret", spatial_input=spatial_ref(), region_source=RegionSource.OA_AUXSEG_CANDIDATE, candidate_region_id=3)
        response = self.run_request(runtime, request)
        self.assertEqual(self.capability_calls(calls), ["spatial", "evidence", "mllm:observation", "parse:observation", "rag", "mllm:interpretation", "parse:interpretation"])
        self.assertEqual(evidence.last_mask, "candidate-3")
        self.assertEqual(response.region_selection.status, RegionSelectionStatus.CANDIDATE_SELECTED)
        self.assertEqual(
            calls,
            [
                "spatial", "release:spatial", "evidence", "mllm:observation",
                "parse:observation", "release:mllm", "rag", "release:rag",
                "mllm:interpretation", "parse:interpretation", "release:mllm",
            ],
        )

    def test_missing_candidate_id_falls_back_global(self) -> None:
        runtime, _, evidence = self.make_runtime()
        request = UnifiedRequest(UnifiedTask.REGION_INTERPRETATION, "7", "interpret", spatial_input=spatial_ref(), region_source=RegionSource.OA_AUXSEG_CANDIDATE, include_audit=True)
        response = self.run_request(runtime, request)
        self.assertEqual(evidence.last_mask, "global-mask")
        self.assertEqual(response.region_selection.status, RegionSelectionStatus.FALLBACK_GLOBAL)
        self.assertEqual(response.region_selection.reason.value, "CANDIDATE_ID_MISSING")
        self.assertIn("CANDIDATE_ID_MISSING", response.limitations[0])

    def test_unknown_candidate_id_does_not_select_top1(self) -> None:
        runtime, _, evidence = self.make_runtime()
        request = UnifiedRequest(UnifiedTask.REGION_INTERPRETATION, "8", "interpret", spatial_input=spatial_ref(), region_source=RegionSource.OA_AUXSEG_CANDIDATE, candidate_region_id=999)
        response = self.run_request(runtime, request)
        self.assertEqual(evidence.last_mask, "global-mask")
        self.assertEqual(response.region_selection.reason.value, "CANDIDATE_ID_NOT_FOUND")

    def test_no_candidates_falls_back_global(self) -> None:
        runtime, _, evidence = self.make_runtime(candidates=False)
        request = UnifiedRequest(UnifiedTask.REGION_INTERPRETATION, "9", "interpret", spatial_input=spatial_ref(), region_source=RegionSource.OA_AUXSEG_CANDIDATE, candidate_region_id=7)
        response = self.run_request(runtime, request)
        self.assertEqual(evidence.last_mask, "global-mask")
        self.assertEqual(response.region_selection.reason.value, "NO_CANDIDATES")

    def test_no_target_still_runs_full_interpretation(self) -> None:
        runtime, calls, _ = self.make_runtime(candidates=False, no_target=True)
        request = UnifiedRequest(UnifiedTask.REGION_INTERPRETATION, "10", "interpret", spatial_input=spatial_ref(), region_source=RegionSource.OA_AUXSEG_CANDIDATE)
        response = self.run_request(runtime, request)
        self.assertEqual(self.capability_calls(calls), ["spatial", "evidence", "mllm:observation", "parse:observation", "rag", "mllm:interpretation", "parse:interpretation"])
        self.assertTrue(response.no_target)
        self.assertIn("OA_AUXSEG_NO_TARGET_OR_EMPTY_GLOBAL_MASK", response.limitations)

    def test_unknown_task_fails_closed(self) -> None:
        with self.assertRaises(UnifiedInferenceError) as raised:
            UnifiedRequest.from_dict({"task": "AUTO", "request_id": "x"})
        self.assertEqual(raised.exception.reason_code, UnifiedReasonCode.INVALID_TASK)

    def test_test_and_sealed_paths_rejected_before_provider(self) -> None:
        runtime, calls, _ = self.make_runtime()
        request = UnifiedRequest(UnifiedTask.VLM_ONLY, "11", "describe", (Path("/tmp/sealed/a.png"),))
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(UnifiedInferenceError) as raised:
                runtime.infer(request, output_root=Path(directory) / "result")
        self.assertEqual(raised.exception.reason_code, UnifiedReasonCode.TEST_OR_SEALED_PATH_FORBIDDEN)
        self.assertEqual(calls, [])

    def test_provider_failure_keeps_reason_and_does_not_fallback(self) -> None:
        runtime, calls, _ = self.make_runtime(failing_spatial=True)
        request = UnifiedRequest(UnifiedTask.REGION_INTERPRETATION, "12", "interpret", spatial_input=spatial_ref(), region_source=RegionSource.OA_AUXSEG_CANDIDATE)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result"
            with self.assertRaises(UnifiedInferenceError) as raised:
                runtime.infer(request, output_root=output)
            self.assertTrue((output / "failure.json").is_file())
        self.assertEqual(raised.exception.reason_code, UnifiedReasonCode.SPATIAL_PROVIDER_FAILED)
        self.assertEqual(self.capability_calls(calls), ["spatial"])

    def test_gt_mask_is_evaluation_only(self) -> None:
        with self.assertRaises(UnifiedInferenceError) as raised:
            UnifiedRequest(UnifiedTask.REGION_UNDERSTANDING, "13", "region", (Path("/tmp/a.png"),), Path("/tmp/m.png"), region_source=RegionSource.GT_MASK)
        self.assertEqual(raised.exception.reason_code, UnifiedReasonCode.FORBIDDEN_REGION_SOURCE)

    def test_plan_loads_no_provider(self) -> None:
        runtime, calls, _ = self.make_runtime()
        plan = runtime.plan(UnifiedRequest(UnifiedTask.KNOWLEDGE_QA, "14", "why"))
        self.assertTrue(plan.needs_rag)
        self.assertEqual(calls, [])

    def test_existing_output_root_rejected(self) -> None:
        runtime, calls, _ = self.make_runtime()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(Exception):
                runtime.infer(
                    UnifiedRequest(UnifiedTask.KNOWLEDGE_QA, "15", "why"),
                    output_root=Path(directory),
                )
        self.assertEqual(calls, [])

    def test_symbolic_output_root_is_rejected_before_provider(self) -> None:
        runtime, calls, _ = self.make_runtime()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target"
            target.mkdir()
            link = root / "result"
            link.symlink_to(target, target_is_directory=True)
            with self.assertRaises(Exception):
                runtime.infer(
                    UnifiedRequest(UnifiedTask.KNOWLEDGE_QA, "16", "why"),
                    output_root=link,
                )
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
