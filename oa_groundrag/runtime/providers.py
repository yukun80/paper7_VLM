"""Unified Runtime provider protocols 与轻量交换合同。"""

from __future__ import annotations

from dataclasses import dataclass
import gc
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from .contracts import RegionSelection, UnifiedRequest, reject_test_or_sealed_path


def _guard_regular(path: Path, *, label: str, directory: bool = False) -> None:
    from oa_groundrag.phase3.common import first_symlink_component

    reject_test_or_sealed_path(path, label=label)
    exists = path.is_dir() if directory else path.is_file()
    if first_symlink_component(path) is not None or not exists or path.is_symlink():
        kind = "目录" if directory else "文件"
        raise ValueError(f"{label} 必须是普通{kind}：{path}")


@dataclass(frozen=True)
class SpatialCandidate:
    region_id: int
    mask: Any
    bbox_xyxy: tuple[int, int, int, int]
    centroid_xy: tuple[float, float]
    area_pixels: int
    confidence: float
    mask_reference: str | None = None

    def public_dict(self) -> dict[str, Any]:
        return {
            "region_id": self.region_id,
            "bbox_xyxy": list(self.bbox_xyxy),
            "centroid_xy": list(self.centroid_xy),
            "area_pixels": self.area_pixels,
            "confidence": self.confidence,
            "mask_reference": self.mask_reference,
        }


@dataclass(frozen=True)
class SpatialResult:
    sample_id: str
    source: str
    split: str
    optical_image: Any
    global_mask: Any
    mask_probability: Any
    no_target: bool
    no_target_score: float
    candidates: tuple[SpatialCandidate, ...]
    active_modalities: tuple[str, ...]
    mask_reference: str | None = None
    mask_probability_reference: str | None = None
    identity: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class GroundedEvidenceResult:
    messages: tuple[Mapping[str, Any], ...]
    program_facts: Mapping[str, Any]
    target_status: str
    mask_reference: str
    limitations: tuple[str, ...]
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class TextRAGResult:
    packet: Mapping[str, Any]
    messages: tuple[Mapping[str, Any], ...]
    citations: tuple[Mapping[str, Any], ...]
    metadata: Mapping[str, Any]


class SpatialProvider(Protocol):
    def infer(self, request: UnifiedRequest, *, output_dir: Path) -> SpatialResult: ...
    def release(self) -> None: ...


class SharedMLLMProvider(Protocol):
    def describe(self, request: UnifiedRequest) -> str: ...
    def generate_visual(self, messages: Sequence[Mapping[str, Any]]) -> str: ...
    def generate_text(self, messages: Sequence[Mapping[str, Any]], *, packet: Mapping[str, Any]) -> str: ...
    def release(self) -> None: ...


class GroundedEvidenceProvider(Protocol):
    def build(
        self,
        request: UnifiedRequest,
        *,
        optical_image: Any,
        mask: Any,
        selection: RegionSelection,
        output_dir: Path,
        sample_id: str,
        source: str,
        split: str,
        no_target: bool,
    ) -> GroundedEvidenceResult: ...

    def parse_observation(
        self,
        raw_output: str,
        *,
        evidence: GroundedEvidenceResult,
    ) -> Mapping[str, Any]: ...


class TextRAGProvider(Protocol):
    def retrieve(
        self,
        request: UnifiedRequest,
        *,
        observation: Mapping[str, Any],
        program_facts: Mapping[str, Any],
        target_status: str,
        available_modalities: Sequence[str],
        candidate_count: int,
    ) -> TextRAGResult: ...

    def parse_generation(
        self,
        raw_output: str,
        *,
        result: TextRAGResult,
    ) -> Mapping[str, Any]: ...

    def release(self) -> None: ...


class OAAuxSegSpatialProvider:
    """现有 OA-AuxSeg single-request session 的 lazy adapter。"""

    def __init__(
        self,
        *,
        config_path: Path,
        checkpoint_path: Path,
        checkpoint_sha256: str,
        repo_root: Path,
    ) -> None:
        self.config_path = Path(config_path)
        self.checkpoint_path = Path(checkpoint_path)
        self.checkpoint_sha256 = checkpoint_sha256
        self.repo_root = Path(repo_root)
        self._session: Any | None = None

    def _load(self) -> Any:
        if self._session is None:
            from oa_groundrag.phase3.common import sha256_file
            from oa_groundrag.phase2.engine import load_runtime_config
            from oa_groundrag.phase2.inference_runtime import SpatialInferenceSession

            _guard_regular(self.repo_root, label="OA-AuxSeg repository_root", directory=True)
            _guard_regular(self.config_path, label="OA-AuxSeg config")
            _guard_regular(self.checkpoint_path, label="OA-AuxSeg checkpoint")
            config = load_runtime_config(self.config_path)
            benchmark_root = config.resolve_path(config.benchmark_root, self.repo_root)
            output_root = config.resolve_path(config.output_dir, self.repo_root)
            backbone = config.resolve_path(config.backbone_weights, self.repo_root)
            reject_test_or_sealed_path(benchmark_root, label="OA-AuxSeg benchmark_root")
            reject_test_or_sealed_path(output_root, label="OA-AuxSeg artifact_root")
            reject_test_or_sealed_path(backbone, label="OA-AuxSeg backbone")
            if sha256_file(self.checkpoint_path) != self.checkpoint_sha256:
                raise ValueError("OA-AuxSeg checkpoint SHA-256 漂移")
            self._session = SpatialInferenceSession(
                config,
                repo_root=self.repo_root,
                checkpoint_path=self.checkpoint_path,
            )
        return self._session

    def infer(self, request: UnifiedRequest, *, output_dir: Path) -> SpatialResult:
        import numpy as np
        from PIL import Image

        from oa_groundrag.phase2.inference_runtime import BatchInferenceResult

        from .contracts import BenchmarkSampleRef, InMemorySpatialInput

        session = self._load()
        spatial_input = request.spatial_input
        if isinstance(spatial_input, BenchmarkSampleRef):
            sample = session.infer_benchmark_sample(
                split=spatial_input.split,
                sample_id=spatial_input.sample_id,
            )
            result = sample.result
            sample_id = sample.sample_id
            source = sample.source
            split = sample.split
            optical_image = sample.optical_image
        elif isinstance(spatial_input, InMemorySpatialInput):
            result = session.infer_batch(spatial_input.batch)
            if result.output.mask_probability.shape[0] != 1:
                raise ValueError("Unified in-memory spatial input 首版只允许 batch_size=1")
            sample_id = spatial_input.sample_id
            source = spatial_input.source
            split = spatial_input.split
            optical_image = spatial_input.optical_image
        else:
            raise ValueError("spatial provider 缺少 spatial_input")
        if not isinstance(result, BatchInferenceResult):
            raise TypeError("OA-AuxSeg helper 返回类型非法")
        output = result.output
        probability = output.mask_probability[0, 0].detach().float().cpu().numpy()
        global_mask = probability >= float(session.config.region_threshold)
        output_dir = Path(output_dir)
        if output_dir.exists() or output_dir.is_symlink():
            raise FileExistsError(f"spatial output 已存在：{output_dir}")
        output_dir.mkdir(parents=True)
        with (output_dir / "mask_probability.npy").open("wb") as handle:
            np.save(handle, probability.astype(np.float32), allow_pickle=False)
        Image.fromarray(global_mask.astype(np.uint8) * 255, mode="L").save(
            output_dir / "global_mask.png",
            format="PNG",
        )
        raw_candidates = output.candidate_regions[0] if output.candidate_regions is not None else []
        candidates: list[SpatialCandidate] = []
        candidate_root = output_dir / "candidates"
        for region in raw_candidates:
            candidate_mask = region.mask.detach().cpu().numpy().astype(bool)
            candidate_root.mkdir(exist_ok=True)
            relative = f"spatial/candidates/region_{region.region_id}.png"
            Image.fromarray(candidate_mask.astype(np.uint8) * 255, mode="L").save(
                candidate_root / f"region_{region.region_id}.png",
                format="PNG",
            )
            candidates.append(SpatialCandidate(
                region_id=int(region.region_id),
                mask=candidate_mask,
                bbox_xyxy=tuple(int(value) for value in region.bbox_xyxy),
                centroid_xy=tuple(float(value) for value in region.centroid_xy),
                area_pixels=int(region.area_pixels),
                confidence=float(region.confidence),
                mask_reference=relative,
            ))
        return SpatialResult(
            sample_id=sample_id,
            source=source,
            split=split,
            optical_image=optical_image,
            global_mask=global_mask,
            mask_probability=probability,
            no_target=not bool(global_mask.any()),
            no_target_score=float(output.no_target_score[0].detach().float().item()),
            candidates=tuple(candidates),
            active_modalities=result.active_modalities[0],
            mask_reference="spatial/global_mask.png",
            mask_probability_reference="spatial/mask_probability.npy",
            identity=self.runtime_metadata(),
        )

    def release(self) -> None:
        if self._session is not None:
            self._session.release()
            self._session = None

    def runtime_metadata(self) -> Mapping[str, Any]:
        return (
            {}
            if self._session is None
            else {
                **self._session.identity,
                "checkpoint_sha256": self.checkpoint_sha256,
            }
        )


class Stage5SharedMLLMProvider:
    """Stage 5 best Qwen3-VL + LoRA 的 lazy shared semantic provider。"""

    def __init__(self, *, stage6_config_path: Path, device: str) -> None:
        self.stage6_config_path = Path(stage6_config_path)
        self.device_name = device
        self._bundle: Any | None = None
        self._device: Any | None = None

    def _load(self) -> Any:
        if self._bundle is None:
            import torch

            from oa_groundrag.phase4.stage5_runtime import load_stage5_best_generator
            from oa_groundrag.phase4.stage5_config import load_stage5_config
            from oa_groundrag.text_rag.contracts import load_stage6_config

            if self.device_name == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("Shared MLLM 配置要求 CUDA，但当前不可见")
            self._device = torch.device(self.device_name)
            _guard_regular(self.stage6_config_path, label="Stage 6 config")
            stage6 = load_stage6_config(self.stage6_config_path)
            for label, path in (
                ("Stage 6 source registry", stage6.source_registry_path),
                ("Stage 6 Bank", stage6.bank_root),
                ("Stage 6 dense model", stage6.dense.model_root),
                ("Stage 5 config", stage6.stage5.config_path),
            ):
                reject_test_or_sealed_path(path, label=label)
            _guard_regular(stage6.stage5.config_path, label="Stage 5 config")
            stage5 = load_stage5_config(stage6.stage5.config_path)
            for label, path in (
                ("Stage 5 workflow", stage5.workflow_root),
                ("Stage 5 checkpoint root", stage5.run.output_root),
                ("Stage 5 compact training", stage5.data_contract.compact_training_root),
                ("Stage 5 model", stage5.model.path),
                ("Stage 5 processor", stage5.model.processor_path),
            ):
                reject_test_or_sealed_path(path, label=label)
            self._bundle = load_stage5_best_generator(
                stage6.stage5,
                device=self._device,
            )
            self._bundle.model.eval()
        return self._bundle

    def _generate_visual(self, messages: Sequence[Mapping[str, Any]]) -> str:
        import torch
        from oa_groundrag.phase4.processing import single_inference_tensor_batch

        bundle = self._load()
        encoded = bundle.processor.encode_inference(messages)
        batch = {
            key: value.to(self._device)
            for key, value in single_inference_tensor_batch(encoded).items()
        }
        with torch.inference_mode():
            outputs = bundle.model.generate_text(
                batch,
                processor=bundle.processor.processor,
                max_new_tokens=bundle.config.generation.max_new_tokens,
                do_sample=bundle.config.generation.do_sample,
                temperature=bundle.config.generation.temperature,
                top_p=bundle.config.generation.top_p,
            )
        if len(outputs) != 1 or not outputs[0].strip():
            raise RuntimeError("Shared MLLM visual generation 必须返回一个非空文本")
        return outputs[0]

    def describe(self, request: UnifiedRequest) -> str:
        from oa_groundrag.phase3.common import first_symlink_component

        content: list[dict[str, Any]] = []
        for image in request.images:
            path = Path(image)
            if (
                first_symlink_component(path) is not None
                or not path.is_file()
                or path.is_symlink()
                or path.stat().st_nlink != 1
            ):
                raise ValueError(f"VLM image 必须是普通单链接文件：{path}")
            content.append({"type": "image", "image": str(path.resolve())})
        content.append({"type": "text", "text": request.normalized_instruction})
        return self._generate_visual(({"role": "user", "content": content},))

    def generate_visual(self, messages: Sequence[Mapping[str, Any]]) -> str:
        return self._generate_visual(messages)

    def generate_text(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        packet: Mapping[str, Any],
    ) -> str:
        import torch
        from transformers import LogitsProcessorList

        from oa_groundrag.text_rag.contracts import RagMode
        from oa_groundrag.text_rag.pass2 import (
            PASS2_ASSISTANT_PREFILL,
            build_pass2_logits_processor,
        )
        from oa_groundrag.phase4.processing import single_inference_tensor_batch

        bundle = self._load()
        encoded = bundle.processor.encode_text_inference(messages)
        batch = {
            key: value.to(self._device)
            for key, value in single_inference_tensor_batch(encoded).items()
        }
        constraint = build_pass2_logits_processor(
            tokenizer=bundle.processor.processor.tokenizer,
            prompt_length=encoded.input_token_count,
            mode=RagMode.TEXT_RAG,
            packet=packet,
        )
        with torch.inference_mode():
            outputs = bundle.model.generate_text(
                batch,
                processor=bundle.processor.processor,
                max_new_tokens=bundle.config.generation.max_new_tokens,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                logits_processor=LogitsProcessorList([constraint]),
            )
        if len(outputs) != 1:
            raise RuntimeError("Shared MLLM text generation 必须返回一个文本")
        return PASS2_ASSISTANT_PREFILL + outputs[0]

    def release(self) -> None:
        self._bundle = None
        self._device = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def runtime_metadata(self) -> Mapping[str, Any]:
        return {} if self._bundle is None else dict(self._bundle.identity)


class Phase4GroundedEvidenceProvider:
    """EvidenceBuilder + Stage 5 messages/parser 的薄 adapter。"""

    def __init__(self) -> None:
        from oa_groundrag.phase4.evidence import EvidenceBuilder

        self.builder = EvidenceBuilder(max_auxiliary_views=2)

    def build(
        self,
        request: UnifiedRequest,
        *,
        optical_image: Any,
        mask: Any,
        selection: RegionSelection,
        output_dir: Path,
        sample_id: str,
        source: str,
        split: str,
        no_target: bool,
    ) -> GroundedEvidenceResult:
        from oa_groundrag.phase4.messages import build_mask_grounded_region_messages

        mask_source = {
            "USER_MASK": "user_mask",
            "OA_AUXSEG_GLOBAL": "oa_auxseg_global",
            "OA_AUXSEG_CANDIDATE": "oa_auxseg_candidate",
        }[selection.effective_source.value]
        built = self.builder.build_runtime_region(
            optical_image=optical_image,
            mask=mask,
            output_root=output_dir,
            sample_id=sample_id,
            source=source,
            split=split,
            mask_source=mask_source,
            source_identity={
                "request_id": request.request_id,
                "requested_region_source": selection.requested_source.value,
                "effective_region_source": selection.effective_source.value,
                "region_selection_status": selection.status.value,
                "region_selection_reason": selection.reason.value,
                "requested_candidate_id": selection.requested_candidate_id,
                "selected_candidate_id": selection.selected_candidate_id,
            },
        )
        messages = build_mask_grounded_region_messages(
            built.record,
            asset_root=built.root,
            instruction=request.normalized_instruction,
        )
        limitations: list[str] = []
        if built.record["target_status"] == "no_target":
            limitations.append("NO_TARGET_NO_REGION_GEOMETRY")
        if selection.status.value == "FALLBACK_GLOBAL":
            limitations.append(
                f"CANDIDATE_FALLBACK_GLOBAL:{selection.reason.value}"
            )
        if request.auxiliary_views:
            limitations.append("P0_AUXILIARY_VIEWS_NOT_FORMAL_GROUNDED_INPUT")
        return GroundedEvidenceResult(
            messages=tuple(messages),
            program_facts=dict(built.record["program_facts"]),
            target_status=str(built.record["target_status"]),
            mask_reference="grounded/binary_mask.png",
            limitations=tuple(limitations),
            metadata={
                "record_id": built.record["record_id"],
                "formal_model_input_roles": list(built.record["formal_model_input_roles"]),
                "auxiliary_view_count": len(request.auxiliary_views),
                "auxiliary_views_formal_input": False,
            },
        )

    def parse_observation(
        self,
        raw_output: str,
        *,
        evidence: GroundedEvidenceResult,
    ) -> Mapping[str, Any]:
        from oa_groundrag.phase4.outputs import parse_region_model_output

        parsed = parse_region_model_output(raw_output)
        if parsed.target_status.value != evidence.target_status:
            raise ValueError("Pass-1 target_status 与程序 evidence 不一致")
        return parsed.to_dict()


class Stage6TextRAGProvider:
    """公共 runtime retriever + 既有 Pass-2 prompt/parser adapter。"""

    def __init__(self, *, stage6_config_path: Path) -> None:
        self.stage6_config_path = Path(stage6_config_path)
        self._retriever: Any | None = None

    def _load(self) -> Any:
        if self._retriever is None:
            from oa_groundrag.text_rag.runtime import RuntimeTextRetriever
            from oa_groundrag.text_rag.contracts import load_stage6_config

            _guard_regular(self.stage6_config_path, label="Stage 6 config")
            stage6 = load_stage6_config(self.stage6_config_path)
            for label, path in (
                ("Stage 6 source registry", stage6.source_registry_path),
                ("Stage 6 Bank", stage6.bank_root),
                ("Stage 6 dense model", stage6.dense.model_root),
                ("Stage 5 config", stage6.stage5.config_path),
            ):
                reject_test_or_sealed_path(path, label=label)
            self._retriever = RuntimeTextRetriever(stage6)
        return self._retriever

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
        from oa_groundrag.phase3.common import canonical_json
        from oa_groundrag.text_rag.contracts import TextRagTask
        from oa_groundrag.text_rag.pass2 import build_pass2_messages

        task = (
            TextRagTask.PROFESSIONAL_QA
            if request.task.value == "KNOWLEDGE_QA"
            else TextRagTask.CANDIDATE_INTERPRETATION
        )
        observation_before = canonical_json(dict(observation))
        facts_before = canonical_json(dict(program_facts))
        retrieved = self._load().retrieve(
            task=task,
            record_id=request.request_id,
            question=request.normalized_instruction or "",
            observation=observation,
            program_facts=program_facts,
            target_status=target_status,
            candidate_count=candidate_count,
            available_modalities=available_modalities,
        )
        if canonical_json(dict(observation)) != observation_before or canonical_json(dict(program_facts)) != facts_before:
            raise RuntimeError("Text RAG 修改了只读 observation/program facts")
        messages = build_pass2_messages(
            question=request.normalized_instruction or "",
            target_status=target_status,
            program_facts=program_facts,
            observation=observation,
            packet=retrieved.packet,
            task=task,
        )
        citations = tuple({
            "evidence_id": item["evidence_id"],
            "source_id": item["source_id"],
            "source_title": item["source_title"],
            "pdf_page": item["pdf_page"],
            "section": item["section"],
        } for item in retrieved.packet["items"])
        return TextRAGResult(
            packet=dict(retrieved.packet),
            messages=tuple(messages),
            citations=citations,
            metadata={
                "task": task.value,
                "bank_identity": dict(retrieved.bank_identity),
                "query_ids": [query["query_id"] for query in retrieved.queries],
            },
        )

    def parse_generation(
        self,
        raw_output: str,
        *,
        result: TextRAGResult,
    ) -> Mapping[str, Any]:
        from oa_groundrag.text_rag.contracts import RagMode
        from oa_groundrag.text_rag.pass2 import parse_pass2_output

        return parse_pass2_output(
            raw_output,
            mode=RagMode.TEXT_RAG,
            packet=result.packet,
        ).to_dict()

    def release(self) -> None:
        if self._retriever is not None:
            self._retriever.release()
            self._retriever = None
