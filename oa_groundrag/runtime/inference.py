"""ExecutionPlan 驱动的统一推理编排；不实现模型数学。"""

from __future__ import annotations

import gc
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, TypeVar

from oa_groundrag.artifacts.identity import (
    canonical_json,
    sha256_file,
    sha256_text,
)
from oa_groundrag.artifacts.directory import AtomicArtifactDirectory

from .contracts import (
    ExecutionPlan,
    RegionSelection,
    RegionSelectionReason,
    RegionSelectionStatus,
    RegionSource,
    UnifiedInferenceError,
    UnifiedReasonCode,
    UnifiedRequest,
    UnifiedResponse,
    UnifiedTask,
    reject_test_or_sealed_path,
)
from .providers import (
    GroundedEvidenceProvider,
    GroundedEvidenceResult,
    SharedMLLMProvider,
    SpatialCandidate,
    SpatialProvider,
    SpatialResult,
    TextRAGProvider,
    TextRAGResult,
)
from .router import CapabilityRouter


T = TypeVar("T")


_PROVIDER_REASON = {
    "spatial": UnifiedReasonCode.SPATIAL_PROVIDER_FAILED,
    "evidence": UnifiedReasonCode.EVIDENCE_PROVIDER_FAILED,
    "shared_mllm": UnifiedReasonCode.SHARED_MLLM_FAILED,
    "text_rag": UnifiedReasonCode.TEXT_RAG_FAILED,
}

_IDENTITY_REASON_CODES = {
    "ARTIFACT_IDENTITY_MISMATCH",
    "BENCHMARK_IDENTITY_MISMATCH",
    "CHECKPOINT_CORRUPT",
    "CHECKPOINT_INCOMPATIBLE",
    "MODEL_IDENTITY_MISMATCH",
    "PREDICTION_IDENTITY_MISMATCH",
}

_RESIDENT_OPERATIONS = {
    "spatial": {"infer"},
    "shared_mllm": {"describe", "visual_observation", "knowledge_generation"},
    "text_rag": {"retrieve"},
}


class UnifiedInferenceRuntime:
    def __init__(
        self,
        *,
        spatial: SpatialProvider,
        shared_mllm: SharedMLLMProvider,
        evidence: GroundedEvidenceProvider,
        text_rag: TextRAGProvider,
        router: CapabilityRouter | None = None,
    ) -> None:
        self.spatial = spatial
        self.shared_mllm = shared_mllm
        self.evidence = evidence
        self.text_rag = text_rag
        self.router = router or CapabilityRouter()

    def plan(self, request: UnifiedRequest) -> ExecutionPlan:
        request.validate_paths()
        return self.router.route(request)

    def _event(self, trace: list[dict[str, Any]], event: str, **details: Any) -> None:
        trace.append({"index": len(trace), "event": event, **details})

    def _call(
        self,
        trace: list[dict[str, Any]],
        *,
        provider: str,
        operation: str,
        task: UnifiedTask,
        function: Callable[[], T],
    ) -> T:
        peak = 0
        allocated = 0
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except (ImportError, RuntimeError):
            pass
        self._event(trace, "provider_call_started", provider=provider, operation=operation)
        try:
            result = function()
        except UnifiedInferenceError:
            raise
        except Exception as error:
            code_value = getattr(getattr(error, "code", None), "value", None)
            if code_value == "INVALID_MODEL_OUTPUT":
                reason = UnifiedReasonCode.INVALID_MODEL_OUTPUT
            elif code_value in _IDENTITY_REASON_CODES:
                reason = UnifiedReasonCode.ARTIFACT_IDENTITY_MISMATCH
            else:
                reason = _PROVIDER_REASON[provider]
            self._event(
                trace,
                "provider_call_failed",
                provider=provider,
                operation=operation,
                cause_type=type(error).__name__,
            )
            raise UnifiedInferenceError(
                reason,
                str(error),
                task=task,
                provider=provider,
                completed_trace=trace,
                cause_type=type(error).__name__,
                details={"upstream_reason_code": code_value},
            ) from error
        try:
            import torch

            if torch.cuda.is_available():
                peak = int(torch.cuda.max_memory_allocated())
                allocated = int(torch.cuda.memory_allocated())
        except (ImportError, RuntimeError):
            pass
        self._event(
            trace,
            "provider_call_completed",
            provider=provider,
            operation=operation,
            cuda_peak_allocated_bytes=peak,
            cuda_allocated_bytes=allocated,
        )
        return result

    def _metadata(
        self,
        trace: list[dict[str, Any]],
        *,
        provider: str,
        value: Mapping[str, Any],
    ) -> None:
        self._event(
            trace,
            "provider_metadata",
            provider=provider,
            identity=dict(value),
        )

    def _release(
        self,
        trace: list[dict[str, Any]],
        *,
        provider: str,
        task: UnifiedTask,
    ) -> None:
        target = {
            "spatial": self.spatial,
            "shared_mllm": self.shared_mllm,
            "text_rag": self.text_rag,
        }[provider]
        self._call(
            trace,
            provider=provider,
            operation="release",
            task=task,
            function=target.release,
        )
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        self._event(trace, "unused_memory_cleared", provider=provider)

    @staticmethod
    def _active_heavy_providers(trace: Sequence[Mapping[str, Any]]) -> set[str]:
        active: set[str] = set()
        for row in trace:
            provider = str(row.get("provider"))
            if provider not in {"spatial", "shared_mllm", "text_rag"}:
                continue
            if (
                row.get("event") == "provider_call_started"
                and row.get("operation") in _RESIDENT_OPERATIONS[provider]
            ):
                active.add(provider)
            elif (
                row.get("event") == "provider_call_completed"
                and row.get("operation") == "release"
            ):
                active.discard(provider)
        return active

    def _release_remaining(
        self,
        trace: list[dict[str, Any]],
        *,
        task: UnifiedTask,
        strict: bool,
    ) -> None:
        providers = {
            "spatial": self.spatial,
            "shared_mllm": self.shared_mllm,
            "text_rag": self.text_rag,
        }
        for provider in sorted(self._active_heavy_providers(trace)):
            if strict:
                self._release(trace, provider=provider, task=task)
                continue
            try:
                providers[provider].release()
                self._event(trace, "provider_cleanup_completed", provider=provider)
            except Exception as error:
                self._event(
                    trace,
                    "provider_cleanup_failed",
                    provider=provider,
                    cause_type=type(error).__name__,
                )

    @staticmethod
    def _candidate_selection(
        request: UnifiedRequest,
        spatial: SpatialResult,
    ) -> tuple[RegionSelection, Any, str | None]:
        candidates = spatial.candidates
        requested = request.candidate_region_id
        if not candidates:
            reason = RegionSelectionReason.NO_CANDIDATES
        elif requested is None:
            reason = RegionSelectionReason.CANDIDATE_ID_MISSING
        else:
            selected = next(
                (candidate for candidate in candidates if candidate.region_id == requested),
                None,
            )
            if selected is not None:
                return (
                    RegionSelection(
                        requested_source=RegionSource.OA_AUXSEG_CANDIDATE,
                        effective_source=RegionSource.OA_AUXSEG_CANDIDATE,
                        status=RegionSelectionStatus.CANDIDATE_SELECTED,
                        requested_candidate_id=requested,
                        selected_candidate_id=selected.region_id,
                        candidate_count=len(candidates),
                    ),
                    selected.mask,
                    selected.mask_reference,
                )
            reason = RegionSelectionReason.CANDIDATE_ID_NOT_FOUND
        return (
            RegionSelection(
                requested_source=RegionSource.OA_AUXSEG_CANDIDATE,
                effective_source=RegionSource.OA_AUXSEG_GLOBAL,
                status=RegionSelectionStatus.FALLBACK_GLOBAL,
                reason=reason,
                requested_candidate_id=requested,
                selected_candidate_id=None,
                candidate_count=len(candidates),
            ),
            spatial.global_mask,
            spatial.mask_reference,
        )

    @staticmethod
    def _fallback_limitation(selection: RegionSelection | None) -> tuple[str, ...]:
        if selection is None or selection.status is not RegionSelectionStatus.FALLBACK_GLOBAL:
            return ()
        return (
            "REQUESTED_CANDIDATE_UNRESOLVED:"
            f"{selection.reason.value};OA_AUXSEG_GLOBAL_USED",
        )

    @staticmethod
    def _observation_limitations(observation: Mapping[str, Any] | None) -> tuple[str, ...]:
        if observation is None:
            return ()
        values = observation.get("limitations", [])
        if not isinstance(values, list):
            return ()
        return tuple(str(value) for value in values if isinstance(value, str) and value)

    @staticmethod
    def _knowledge_limitations(output: Mapping[str, Any] | None) -> tuple[str, ...]:
        if output is None:
            return ()
        values = output.get("limitations", [])
        if not isinstance(values, list):
            return ()
        result: list[str] = []
        for value in values:
            if isinstance(value, Mapping) and isinstance(value.get("text"), str):
                result.append(str(value["text"]))
        return tuple(result)

    def _run(
        self,
        request: UnifiedRequest,
        plan: ExecutionPlan,
        *,
        writer: AtomicArtifactDirectory,
        trace: list[dict[str, Any]],
    ) -> UnifiedResponse:
        spatial: SpatialResult | None = None
        evidence: GroundedEvidenceResult | None = None
        observation: Mapping[str, Any] | None = None
        rag_result: TextRAGResult | None = None
        knowledge_output: Mapping[str, Any] | None = None
        selection: RegionSelection | None = None
        text: str | None = None
        mask: Any = None
        mask_reference: str | None = None
        no_target: bool | None = None
        sample_id = request.request_id
        source = "user"
        split = "runtime"
        optical_image: Any = request.images[0] if request.images else None

        if plan.needs_spatial:
            spatial = self._call(
                trace,
                provider="spatial",
                operation="infer",
                task=request.task,
                function=lambda: self.spatial.infer(request, output_dir=writer.path("spatial")),
            )
            self._metadata(
                trace,
                provider="spatial",
                value={} if spatial.identity is None else spatial.identity,
            )
            sample_id, source, split = spatial.sample_id, spatial.source, spatial.split
            optical_image = spatial.optical_image
            no_target = spatial.no_target
            if plan.needs_shared_mllm:
                self._release(trace, provider="spatial", task=request.task)

        if request.task is UnifiedTask.VLM_ONLY:
            text = self._call(
                trace,
                provider="shared_mllm",
                operation="describe",
                task=request.task,
                function=lambda: self.shared_mllm.describe(request),
            )
            metadata = getattr(self.shared_mllm, "runtime_metadata", lambda: {})()
            self._metadata(trace, provider="shared_mllm", value=metadata)
        elif request.task is UnifiedTask.SEGMENT_ONLY:
            assert spatial is not None
            mask_reference = spatial.mask_reference
        elif request.task is UnifiedTask.REGION_UNDERSTANDING:
            selection = RegionSelection(
                requested_source=RegionSource.USER_MASK,
                effective_source=RegionSource.USER_MASK,
                status=RegionSelectionStatus.DIRECT,
            )
            mask = request.user_mask
            mask_reference = None if request.user_mask is None else str(request.user_mask)
            no_target = False
        elif request.task is UnifiedTask.SEGMENT_AND_UNDERSTAND:
            assert spatial is not None
            selection = RegionSelection(
                requested_source=RegionSource.OA_AUXSEG_GLOBAL,
                effective_source=RegionSource.OA_AUXSEG_GLOBAL,
                status=RegionSelectionStatus.DIRECT,
                candidate_count=len(spatial.candidates),
            )
            mask = spatial.global_mask
            mask_reference = spatial.mask_reference
        elif request.task is UnifiedTask.REGION_INTERPRETATION:
            if request.region_source is RegionSource.USER_MASK:
                selection = RegionSelection(
                    requested_source=RegionSource.USER_MASK,
                    effective_source=RegionSource.USER_MASK,
                    status=RegionSelectionStatus.DIRECT,
                )
                mask = request.user_mask
                mask_reference = None if request.user_mask is None else str(request.user_mask)
                no_target = False
            else:
                assert spatial is not None
                selection, mask, mask_reference = self._candidate_selection(request, spatial)
                self._event(trace, "region_selected", **selection.to_dict())

        if plan.needs_region:
            assert selection is not None
            evidence = self._call(
                trace,
                provider="evidence",
                operation="build",
                task=request.task,
                function=lambda: self.evidence.build(
                    request,
                    optical_image=optical_image,
                    mask=mask,
                    selection=selection,
                    output_dir=writer.path("grounded"),
                    sample_id=sample_id,
                    source=source,
                    split=split,
                    no_target=bool(no_target),
                ),
            )
            self._metadata(trace, provider="evidence", value=evidence.metadata)
            mask_reference = evidence.mask_reference
            raw_observation = self._call(
                trace,
                provider="shared_mllm",
                operation="visual_observation",
                task=request.task,
                function=lambda: self.shared_mllm.generate_visual(evidence.messages),
            )
            metadata = getattr(self.shared_mllm, "runtime_metadata", lambda: {})()
            self._metadata(trace, provider="shared_mllm", value=metadata)
            observation = self._call(
                trace,
                provider="evidence",
                operation="parse_observation",
                task=request.task,
                function=lambda: self.evidence.parse_observation(raw_observation, evidence=evidence),
            )
            text = str(observation.get("short_summary", "")).strip() or raw_observation

        if plan.needs_rag:
            if observation is not None:
                self._release(trace, provider="shared_mllm", task=request.task)
            rag_result = self._call(
                trace,
                provider="text_rag",
                operation="retrieve",
                task=request.task,
                function=lambda: self.text_rag.retrieve(
                    request,
                    observation={} if observation is None else observation,
                    program_facts={} if evidence is None else evidence.program_facts,
                    target_status="not_applicable" if evidence is None else evidence.target_status,
                    available_modalities=(
                        ("general",)
                        if evidence is None
                        else (
                            ("optical",)
                            if spatial is None
                            else tuple(dict.fromkeys(("optical", *spatial.active_modalities)))
                        )
                    ),
                    candidate_count=0 if spatial is None else len(spatial.candidates),
                ),
            )
            self._metadata(trace, provider="text_rag", value=rag_result.metadata)
            self._release(trace, provider="text_rag", task=request.task)
            raw_knowledge = self._call(
                trace,
                provider="shared_mllm",
                operation="knowledge_generation",
                task=request.task,
                function=lambda: self.shared_mllm.generate_text(
                    rag_result.messages,
                    packet=rag_result.packet,
                ),
            )
            metadata = getattr(self.shared_mllm, "runtime_metadata", lambda: {})()
            self._metadata(trace, provider="shared_mllm", value=metadata)
            knowledge_output = self._call(
                trace,
                provider="text_rag",
                operation="parse_generation",
                task=request.task,
                function=lambda: self.text_rag.parse_generation(raw_knowledge, result=rag_result),
            )
            summary = knowledge_output.get("summary")
            if isinstance(summary, Mapping):
                text = str(summary.get("text", "")).strip() or raw_knowledge
            else:
                text = raw_knowledge

        limitations = tuple(dict.fromkeys((
            *self._fallback_limitation(selection),
            *(() if evidence is None else evidence.limitations),
            *self._observation_limitations(observation),
            *self._knowledge_limitations(knowledge_output),
            *(() if not no_target else ("OA_AUXSEG_NO_TARGET_OR_EMPTY_GLOBAL_MASK",)),
        )))
        return UnifiedResponse(
            request_id=request.request_id,
            task=request.task,
            response_kind=plan.response_kind,
            text=text,
            mask_reference=mask_reference,
            mask_probability_reference=None if spatial is None else spatial.mask_probability_reference,
            no_target=no_target,
            no_target_score=None if spatial is None else spatial.no_target_score,
            candidate_regions=() if spatial is None else tuple(candidate.public_dict() for candidate in spatial.candidates),
            region_selection=selection,
            region_observation=observation,
            knowledge_output=knowledge_output,
            citations=() if rag_result is None else rag_result.citations,
            limitations=limitations,
            execution_plan=plan if request.include_audit else None,
            trace=tuple(trace) if request.include_audit else (),
        )

    @staticmethod
    def _ledger_rows(root: Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.name in {"SHA256SUMS.jsonl", "manifest.json"}:
                continue
            relative = path.relative_to(root).as_posix()
            rows.append({
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
        return rows

    def _publish_failure(
        self,
        *,
        request: UnifiedRequest,
        output_root: Path,
        error: UnifiedInferenceError,
    ) -> None:
        if output_root.exists() or output_root.is_symlink():
            return
        with AtomicArtifactDirectory(output_root) as writer:
            writer.write_json("request.json", request.to_dict())
            writer.write_json("failure.json", error.to_dict())
            assert writer.staging is not None
            ledger = self._ledger_rows(writer.staging)
            writer.write_jsonl("SHA256SUMS.jsonl", ledger)
            writer.write_json("manifest.json", {
                "schema_version": "oa_groundrag.unified_manifest.v1",
                "request_id": request.request_id,
                "task": request.task.value,
                "status": "FAILED",
                "payload_count": len(ledger),
                "engineering_runtime": True,
                "formal_acceptance": False,
                "scientific_acceptance": False,
                "sealed_test_accessed": False,
            })
            writer.publish()

    def infer(self, request: UnifiedRequest, *, output_root: Path) -> UnifiedResponse:
        request.validate_paths()
        output_root = reject_test_or_sealed_path(output_root, label="output_root")
        plan = self.router.route(request)
        trace: list[dict[str, Any]] = []
        try:
            with AtomicArtifactDirectory(output_root) as writer:
                writer.write_json("request.json", request.to_dict())
                response = self._run(request, plan, writer=writer, trace=trace)
                self._release_remaining(trace, task=request.task, strict=True)
                if request.include_audit:
                    response = replace(response, trace=tuple(trace))
                writer.write_json("response.json", response.to_dict())
                assert writer.staging is not None
                ledger = self._ledger_rows(writer.staging)
                writer.write_jsonl("SHA256SUMS.jsonl", ledger)
                writer.write_json("manifest.json", {
                    "schema_version": "oa_groundrag.unified_manifest.v1",
                    "request_id": request.request_id,
                    "task": request.task.value,
                    "request_sha256": sha256_text(canonical_json(request.to_dict())),
                    "response_sha256": sha256_file(writer.path("response.json")),
                    "payload_count": len(ledger),
                    "engineering_runtime": True,
                    "formal_acceptance": False,
                    "scientific_acceptance": False,
                    "sealed_test_accessed": False,
                })
                writer.publish()
                return response
        except UnifiedInferenceError as error:
            self._release_remaining(trace, task=request.task, strict=False)
            enriched = error.with_trace(trace)
            self._publish_failure(
                request=request,
                output_root=output_root,
                error=enriched,
            )
            raise enriched
        except Exception as error:
            self._release_remaining(trace, task=request.task, strict=False)
            upstream = getattr(getattr(error, "code", None), "value", None)
            reason = {
                "OUTPUT_EXISTS": UnifiedReasonCode.OUTPUT_EXISTS,
                "OUTPUT_LINK": UnifiedReasonCode.OUTPUT_LINK,
            }.get(upstream, UnifiedReasonCode.ARTIFACT_WRITE_FAILED)
            enriched = UnifiedInferenceError(
                reason,
                str(error),
                task=request.task,
                provider="runtime",
                completed_trace=trace,
                cause_type=type(error).__name__,
                details={"upstream_reason_code": upstream},
            )
            if trace:
                self._publish_failure(
                    request=request,
                    output_root=output_root,
                    error=enriched,
                )
            raise enriched from error
