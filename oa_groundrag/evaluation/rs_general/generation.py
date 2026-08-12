"""Gate B 的 Base/Adapter 配对确定性生成。"""

from __future__ import annotations

import os
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor

from oa_groundrag.phase3.common import (
    canonical_json,
    first_symlink_component,
    sha256_file,
    sha256_text,
)
from oa_groundrag.phase3.dataset import RSGeneralDescDataset
from oa_groundrag.phase3.errors import RSGeneralDescError

from .artifacts import AtomicArtifactDirectory
from .checkpoint import CheckpointManager
from .contracts import GATE_B_GENERATION_SCHEMA_VERSION
from .data import ExternalDescriptionDataset
from .errors import Phase4Error, PredictionError, ReasonCode
from .gate_b_contracts import (
    GATE_B_PROTOCOL_ID,
    GATE_B_SAMPLE_COUNT,
    GATE_B_SEED,
    GATE_B_TASK_ORDER,
    QWEN_TEMPLATE_VERSION,
    validate_frozen_training_root,
)
from .gate_b_selection import load_gate_b_selection, selection_locations
from .model import Qwen3VLModelAdapter
from .outputs import failure_row, generic_prediction_row
from .processing import DescriptionCollator, Qwen3VLProcessorAdapter
from .trainer import set_global_seed, training_layout_identity


@dataclass(frozen=True)
class GateBGenerationOutcome:
    root: Path
    prediction_count: int
    failure_count: int

    @property
    def valid_for_evaluation(self) -> bool:
        return (
            self.prediction_count == GATE_B_SAMPLE_COUNT
            and self.failure_count == 0
        )


def _move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: (
            value.to(device=device, non_blocking=False)
            if isinstance(value, Tensor)
            else value
        )
        for key, value in batch.items()
    }


def _verify_selected_dataset(context, *, derived_root: Path):
    source = context.protocol_source
    selection = context.selection
    canonical = RSGeneralDescDataset.from_locations(
        source.base_config.data.benchmark_root,
        selection_locations(selection),
        roles=("external_val",),
        task_families=GATE_B_TASK_ORDER,
        load_assets=False,
        seed=GATE_B_SEED,
        expected_manifest_sha256=(
            source.base_config.data.expected_manifest_sha256
        ),
        verifier=context.access.verifier,
    )
    if len(canonical.records) != GATE_B_SAMPLE_COUNT:
        raise PredictionError(
            ReasonCode.GATE_B_SELECTION_INVALID,
            "selection locations 未精确解析为 256 条 external_val records",
        )
    for ordinal, (item, record, location) in enumerate(
        zip(
            selection["items"],
            canonical.records,
            canonical.record_locations,
            strict=True,
        )
    ):
        expected = {
            "record_id": record["record_id"],
            "parent_id": record["parent_id"],
            "source": record["source"],
            "task_family": record["task_family"],
            "shard_path": location.shard_path,
            "line_index": location.line_index,
        }
        actual = {name: item[name] for name in expected}
        if actual != expected or item["ordinal"] != ordinal:
            raise PredictionError(
                ReasonCode.GATE_B_SELECTION_INVALID,
                "selection item identity 与实际 canonical record 不一致",
                details={"ordinal": ordinal, "expected": expected, "actual": actual},
            )
    return ExternalDescriptionDataset(
        canonical,
        derived_root=derived_root,
        seed=GATE_B_SEED,
        roles=("external_val",),
    )


def _configure_determinism() -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    set_global_seed(GATE_B_SEED)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


def _require_new_output_root(path: Path) -> None:
    linked = first_symlink_component(path)
    if linked is not None:
        raise PredictionError(
            ReasonCode.OUTPUT_LINK,
            "Gate B generation output_root 含链接组件",
            details={"path": str(linked)},
        )
    if path.exists() or path.is_symlink():
        raise PredictionError(
            ReasonCode.OUTPUT_EXISTS,
            "Gate B generation output_root 已存在",
            details={"path": str(path)},
        )


def generate_gate_b(
    protocol_path: Path | str,
    selection_path: Path | str,
    *,
    model_role: str,
    output_root: Path,
    training_root: Path | None = None,
) -> GateBGenerationOutcome:
    output_root = Path(output_root)
    _require_new_output_root(output_root)
    if model_role not in {"base", "adapter"}:
        raise PredictionError(
            ReasonCode.INVALID_ENUM,
            "Gate B model_role 只能是 base/adapter",
        )
    if model_role == "base" and training_root is not None:
        raise PredictionError(
            ReasonCode.GATE_B_PROTOCOL_INVALID,
            "Base generation 禁止 training_root",
        )
    if model_role == "adapter" and training_root is None:
        raise PredictionError(
            ReasonCode.CHECKPOINT_INCOMPATIBLE,
            "Adapter generation 必须通过 training_root 解析 final best checkpoint",
        )
    context = load_gate_b_selection(protocol_path, selection_path)
    frozen = context.frozen_protocol
    source = context.protocol_source
    if model_role == "adapter":
        assert training_root is not None
        validate_frozen_training_root(
            frozen,
            source,
            training_root=training_root,
            access=context.access,
        )
    if not torch.cuda.is_available():
        raise PredictionError(
            ReasonCode.CUDA_REQUIRED,
            "正式 Gate B generation 要求可用 CUDA",
        )
    _configure_determinism()
    device = torch.device("cuda")
    config = (
        source.base_config if model_role == "base" else source.adapter_config
    )
    processor = Qwen3VLProcessorAdapter(
        processor_path=config.model.processor_path,
        local_files_only=config.model.local_files_only,
        trust_remote_code=config.model.trust_remote_code,
        min_pixels=config.limits.min_pixels,
        max_pixels=config.limits.max_pixels,
        max_images=config.limits.max_images,
        max_input_tokens=config.limits.max_input_tokens,
    )
    processor_identity = processor.identity()
    if processor_identity != frozen["static_protocol"]["processor_identity"]:
        raise PredictionError(
            ReasonCode.MODEL_IDENTITY_MISMATCH,
            "Gate B processor identity 与 frozen protocol 不一致",
        )
    model = Qwen3VLModelAdapter.load(
        config.model,
        config.adaptation,
        device=device,
        gradient_checkpointing=False,
    )
    model_identity = model.identity.to_dict()
    if model_identity != frozen["static_protocol"]["model_identity"]:
        raise PredictionError(
            ReasonCode.MODEL_IDENTITY_MISMATCH,
            "Gate B model identity 与 frozen protocol 不一致",
        )
    checkpoint_identity: Mapping[str, Any] | None = None
    if model_role == "adapter":
        assert training_root is not None
        best = frozen["training_run"]["best_checkpoint"]
        checkpoint = Path(training_root).resolve() / best["relative_path"]
        payload = CheckpointManager().load(
            checkpoint,
            expected_config_semantic_sha256=config.semantic_sha256,
            expected_benchmark_identity=context.access.identity.training_identity_dict(),
            expected_validation_selection_identity=None,
            expected_model_identity=model_identity,
            expected_processor_identity=processor_identity,
            expected_training_layout=training_layout_identity(config),
            expected_trainable_names=model.trainable_names,
        )
        if payload.root.resolve() != checkpoint.resolve():
            raise PredictionError(
                ReasonCode.CHECKPOINT_INCOMPATIBLE,
                "CheckpointManager 未加载 frozen final best checkpoint",
            )
        model.load_trainable_state_dict(payload.trainable_state)
        checkpoint_identity = dict(best)

    predictions: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    token_total = 0
    image_total = 0
    selection = context.selection
    with tempfile.TemporaryDirectory(
        prefix=f"rs_generaldesc_gate_b_{model_role}_derived_"
    ) as temporary:
        dataset = _verify_selected_dataset(
            context,
            derived_root=Path(temporary) / "derived",
        )
        collator = DescriptionCollator(processor, training=False)
        for ordinal, item in enumerate(selection["items"]):
            try:
                sample = dataset[ordinal]
                batch = collator([sample])
                token_total += int(batch["input_token_counts"][0])
                image_total += int(batch["image_counts"][0])
                if token_total > config.limits.max_total_tokens:
                    raise PredictionError(
                        ReasonCode.TOKEN_LIMIT_EXCEEDED,
                        "Gate B input token total 超过 config 上限",
                    )
                generated = model.generate_text(
                    _move_batch(batch, device),
                    processor=processor.processor,
                    max_new_tokens=384,
                    do_sample=False,
                    temperature=0.0,
                    top_p=1.0,
                )
                if len(generated) != 1 or not generated[0].strip():
                    raise PredictionError(
                        ReasonCode.PREDICTION_INVALID,
                        "Gate B 单样本生成必须返回一个非空文本",
                    )
                provenance = {
                    "canonical_build_id": context.access.identity.build_id,
                    "canonical_payload_sha256": context.access.identity.payload_sha256,
                    "renderer": "phase3.render_canonical_messages",
                    "gate_b": {
                        "protocol_id": GATE_B_PROTOCOL_ID,
                        "protocol_sha256": frozen["protocol_sha256"],
                        "selection_sha256": selection["selection_sha256"],
                        "ordinal": ordinal,
                        "model_role": model_role,
                        "source": item["source"],
                        "shard_path": item["shard_path"],
                        "line_index": item["line_index"],
                        "template_version": QWEN_TEMPLATE_VERSION,
                    },
                }
                predictions.append(
                    generic_prediction_row(
                        record_id=sample.record_id,
                        parent_id=sample.parent_id,
                        logical_role=sample.logical_role,
                        task_family=sample.task_family,
                        generated_text=generated[0],
                        reference_responses=sample.reference_responses,
                        provenance=provenance,
                    )
                )
            except (Phase4Error, RSGeneralDescError) as error:
                failures.append(
                    failure_row(
                        record_id=str(item["record_id"]),
                        parent_id=str(item["parent_id"]),
                        stage="gate_b_generation",
                        code=error.code,
                        message=str(error),
                        details=error.details,
                    )
                )
    with AtomicArtifactDirectory(output_root) as writer:
        writer.write_jsonl("predictions.jsonl", predictions)
        writer.write_jsonl("failures.jsonl", failures)
        prediction_path = writer.path("predictions.jsonl")
        failure_path = writer.path("failures.jsonl")
        valid = (
            len(predictions) == GATE_B_SAMPLE_COUNT
            and len(failures) == 0
        )
        manifest = {
            "schema_version": GATE_B_GENERATION_SCHEMA_VERSION,
            "status": "completed" if valid else "invalid",
            "model_role": model_role,
            "protocol_id": GATE_B_PROTOCOL_ID,
            "protocol_sha256": frozen["protocol_sha256"],
            "selection_sha256": selection["selection_sha256"],
            "selection_file_sha256": sha256_file(Path(selection_path)),
            "benchmark_identity": context.access.identity.to_dict(),
            "config_identity": {
                "path": str(config.config_path.resolve()),
                "file_sha256": sha256_file(config.config_path),
                "semantic_sha256": config.semantic_sha256,
                "adaptation": config.adaptation.strategy,
            },
            "model_identity": model_identity,
            "processor_identity": processor_identity,
            "checkpoint_identity": checkpoint_identity,
            "generation": dict(frozen["static_protocol"]["generation"]),
            "ordered_record_ids_sha256": sha256_text(
                canonical_json([item["record_id"] for item in selection["items"]])
            ),
            "predictions": {
                "path": "predictions.jsonl",
                "count": len(predictions),
                "sha256": sha256_file(prediction_path),
            },
            "failures": {
                "path": "failures.jsonl",
                "count": len(failures),
                "sha256": sha256_file(failure_path),
            },
            "task_counts": dict(
                sorted(Counter(row["task_family"] for row in predictions).items())
            ),
            "input_token_count": token_total,
            "image_count": image_total,
            "valid_for_evaluation": valid,
        }
        writer.write_json("generation_manifest.json", manifest)
        target = writer.publish()
    return GateBGenerationOutcome(
        root=target,
        prediction_count=len(predictions),
        failure_count=len(failures),
    )
