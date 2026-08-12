"""Stage 5 best checkpoint 的公共只读 runtime loader。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from oa_groundrag.data.grounded.supervision.compact_training import (
    CompactTrainingMessageDataset,
)
from oa_groundrag.artifacts.identity import sha256_file
from oa_groundrag.artifacts.io import first_symlink_component
from oa_groundrag.data.rs_general.io import (
    portable_relative_path,
    read_json,
)

from .checkpoint import CheckpointManager
from .errors import ContractError, ReasonCode
from .model import Qwen3VLModelAdapter
from .processing import Qwen3VLProcessorAdapter
from oa_groundrag.training.grounding.config import (
    load_stage5_config,
    with_monitor_parent_count,
)
from oa_groundrag.training.grounding.data import (
    REGION_MONITOR_ROLE,
    RegionSubsetDataset,
    build_region_monitor_selection,
    split_compact_by_parent,
)
from oa_groundrag.training.grounding.workflow import (
    STAGE5_CUDA_CACHE_CLEANUP_INTERVAL_STEPS,
    _compact_benchmark_identity,
)
from oa_groundrag.training.vlm.trainer import training_layout_identity


class Stage5RuntimeBinding(Protocol):
    config_path: Path
    best_pointer_sha256: str
    checkpoint_manifest_sha256: str
    adapter_sha256: str
    workflow_state_sha256: str


@dataclass(frozen=True)
class Stage5BestReference:
    config: Any
    pointer: dict[str, Any]
    checkpoint: Path


@dataclass(frozen=True)
class Stage5GeneratorBundle:
    config: Any
    processor: Qwen3VLProcessorAdapter
    model: Qwen3VLModelAdapter
    checkpoint: Path
    identity: dict[str, Any]


def _regular_file(
    path: Path,
    *,
    label: str,
    expected_sha256: str | None = None,
) -> None:
    linked = first_symlink_component(path)
    if (
        linked is not None
        or not path.is_file()
        or path.is_symlink()
        or path.stat().st_nlink != 1
    ):
        raise ContractError(
            ReasonCode.OUTPUT_LINK,
            f"{label} 必须是普通单链接文件：{path}",
        )
    if expected_sha256 is not None and sha256_file(path) != expected_sha256:
        raise ContractError(
            ReasonCode.BENCHMARK_IDENTITY_MISMATCH,
            f"{label} SHA-256 漂移：{path}",
        )


def resolve_stage5_best(binding: Stage5RuntimeBinding) -> Stage5BestReference:
    """验证 workflow state、best pointer、manifest 和 Adapter identity。"""

    stage5 = load_stage5_config(binding.config_path)
    state_path = stage5.workflow_root / "workflow_state.json"
    _regular_file(
        state_path,
        label="Stage 5 workflow state",
        expected_sha256=binding.workflow_state_sha256,
    )
    state = read_json(state_path)
    if (
        state.get("stage") != "complete"
        or state.get("sealed_test_evaluated") is not False
        or state.get("formal_acceptance") is not False
    ):
        raise ContractError(
            ReasonCode.CHECKPOINT_INCOMPATIBLE,
            "Stage 5 workflow 未完成或科学边界非法",
        )
    pointer_path = stage5.run.output_root / "best_checkpoint.json"
    _regular_file(
        pointer_path,
        label="Stage 5 best pointer",
        expected_sha256=binding.best_pointer_sha256,
    )
    pointer = read_json(pointer_path)
    expected_pointer = {
        "schema_version", "selection_metric", "step", "macro_task_loss",
        "overall_loss", "checkpoint", "formal_acceptance",
    }
    if (
        set(pointer) != expected_pointer
        or pointer.get("selection_metric") != "region_monitor_loss"
        or pointer.get("formal_acceptance") is not False
        or isinstance(pointer.get("step"), bool)
        or not isinstance(pointer.get("step"), int)
        or pointer["step"] <= 0
    ):
        raise ContractError(
            ReasonCode.CHECKPOINT_CORRUPT,
            "Stage 5 best pointer 合同非法",
        )
    relative = portable_relative_path(
        str(pointer["checkpoint"]),
        location="best_checkpoint.checkpoint",
    )
    checkpoint = stage5.run.output_root.joinpath(*relative.parts)
    try:
        checkpoint.resolve().relative_to(stage5.run.output_root.resolve())
    except ValueError as error:
        raise ContractError(
            ReasonCode.PATH_ESCAPE,
            "Stage 5 best checkpoint 路径逃逸",
        ) from error
    if (
        checkpoint.is_symlink()
        or not checkpoint.is_dir()
        or first_symlink_component(checkpoint) is not None
    ):
        raise ContractError(
            ReasonCode.OUTPUT_LINK,
            "Stage 5 best checkpoint 不是普通目录",
        )
    manifest_path = checkpoint / "manifest.json"
    adapter_path = checkpoint / "adapter" / "adapter_model.safetensors"
    _regular_file(
        manifest_path,
        label="Stage 5 checkpoint manifest",
        expected_sha256=binding.checkpoint_manifest_sha256,
    )
    _regular_file(
        adapter_path,
        label="Stage 5 Adapter",
        expected_sha256=binding.adapter_sha256,
    )
    manifest = read_json(manifest_path)
    if manifest.get("cursor", {}).get("global_step") != pointer["step"]:
        raise ContractError(
            ReasonCode.CHECKPOINT_CORRUPT,
            "best pointer step 与 checkpoint 不一致",
        )
    return Stage5BestReference(stage5, pointer, checkpoint)


def load_stage5_best_generator(
    binding: Stage5RuntimeBinding,
    *,
    device: Any,
) -> Stage5GeneratorBundle:
    """只加载 Stage 5 best 的 trainable LoRA state。"""

    reference = resolve_stage5_best(binding)
    stage5 = reference.config
    compact = CompactTrainingMessageDataset(
        stage5.data_contract.compact_training_root
    )
    split = split_compact_by_parent(
        compact,
        seed=stage5.data_contract.split_seed,
    )
    stage5 = with_monitor_parent_count(stage5, len(split.monitor_parents))
    benchmark_identity = _compact_benchmark_identity(compact)
    monitor = RegionSubsetDataset(
        compact,
        split.monitor_indices,
        logical_role=REGION_MONITOR_ROLE,
    )
    selection = build_region_monitor_selection(
        monitor,
        benchmark_build_id=benchmark_identity.build_id,
        benchmark_payload_sha256=benchmark_identity.payload_sha256,
        seed=stage5.run.seed,
    )
    processor = Qwen3VLProcessorAdapter(
        processor_path=stage5.model.processor_path,
        local_files_only=stage5.model.local_files_only,
        trust_remote_code=stage5.model.trust_remote_code,
        min_pixels=stage5.limits.min_pixels,
        max_pixels=stage5.limits.max_pixels,
        max_images=stage5.limits.max_images,
        max_input_tokens=stage5.limits.max_input_tokens,
    )
    model = Qwen3VLModelAdapter.load(
        stage5.model,
        stage5.adaptation,
        device=device,
        gradient_checkpointing=False,
    )
    manifest, trainable = CheckpointManager().load_trainable(
        reference.checkpoint,
        expected_config_semantic_sha256=stage5.semantic_sha256,
        expected_benchmark_identity=benchmark_identity.training_identity_dict(),
        expected_validation_selection_identity=selection.identity_dict(),
        expected_model_identity=model.identity.to_dict(),
        expected_processor_identity=processor.identity(),
        expected_training_layout=training_layout_identity(
            stage5,
            cuda_cache_cleanup_interval_steps=(
                STAGE5_CUDA_CACHE_CLEANUP_INTERVAL_STEPS
            ),
        ),
        expected_trainable_names=model.trainable_names,
    )
    model.load_trainable_state_dict(trainable)
    identity = {
        "stage5_config_semantic_sha256": stage5.semantic_sha256,
        "best_pointer_sha256": binding.best_pointer_sha256,
        "best_step": reference.pointer["step"],
        "checkpoint": str(reference.checkpoint.resolve()),
        "checkpoint_manifest_sha256": binding.checkpoint_manifest_sha256,
        "adapter_sha256": binding.adapter_sha256,
        "model_identity": model.identity.to_dict(),
        "processor_identity": processor.identity(),
        "trainable_parameter_count": manifest["trainable_parameter_count"],
        "loaded_components": ["trainable_lora_state"],
        "excluded_components": ["optimizer", "scheduler", "rng", "sampler"],
    }
    return Stage5GeneratorBundle(
        config=stage5,
        processor=processor,
        model=model,
        checkpoint=reference.checkpoint,
        identity=identity,
    )
