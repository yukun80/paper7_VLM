"""Mask-Grounded train/monitor/replay/retention 的可恢复状态机。"""

from __future__ import annotations

from dataclasses import replace
import gc
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping

import torch
from safetensors.torch import load_file as load_safetensors

from oa_groundrag.data.grounded.supervision.compact_training import CompactTrainingMessageDataset
from oa_groundrag.artifacts.identity import (
    canonical_json,
    sha256_file,
    sha256_text,
)
from oa_groundrag.artifacts.io import atomic_write_json
from oa_groundrag.data.rs_general.io import read_json
from oa_groundrag.data.rs_general.dataset import RSGeneralDescDataset

from oa_groundrag.vlm.checkpoint import CheckpointManager
from oa_groundrag.vlm.backends import (
    VLMModelAdapter,
    VLMProcessorAdapter,
    build_model_adapter,
    build_processor_adapter,
)
from oa_groundrag.vlm.config import AdaptationSection
from oa_groundrag.vlm.data import ExternalDescriptionDataset
from oa_groundrag.vlm.errors import ModelError, ReasonCode
from oa_groundrag.vlm.preflight import BenchmarkIdentity, open_benchmark_access
from oa_groundrag.vlm.processing import DescriptionCollator
from .config import (
    STAGE5_CONFIG_SCHEMA_V3,
    STAGE5_CONFIG_SCHEMA_V4,
    Stage5Config,
    load_stage5_config,
    verify_retention_reference_files,
    verify_warm_start_files,
    with_monitor_parent_count,
)
from .resource_profile import verify_stage5_resource_profile
from .resource_gate import (
    run_worst_case_cuda_gate,
    verify_worst_case_cuda_gate,
)
from .loss_parity import verify_qwen35_loss_parity
from .data import (
    REGION_MONITOR_ROLE,
    REGION_TRAIN_ROLE,
    REPLAY_ROLE,
    RegionSubsetDataset,
    Stage5MixedDataset,
    Stage5MixedSampler,
    build_region_monitor_selection,
    evaluate_region_monitor_loss,
    parse_region_monitor_result,
    split_compact_by_parent,
)
from oa_groundrag.evaluation.grounding.adapter import (
    run_rs_general_retention_report,
)
from oa_groundrag.training.vlm.trainer import DescriptionTrainer, clear_cuda_cache, training_layout_identity
from oa_groundrag.training.vlm.cuda_telemetry import (
    CudaMicrobatchTelemetry,
    CudaTelemetryPolicy,
    allocator_environment_identity,
)
from oa_groundrag.training.vlm.qwen35_supervised import (
    wrap_qwen35_for_supervised_position_training,
)
from oa_groundrag.training.vlm.validation import evaluate_teacher_forced_loss, select_bounded_external_validation


STAGE5_WORKFLOW_SCHEMA = "rs_vlm.mask_grounded_train_workflow.v2"
STAGE5_WORKFLOW_SCHEMA_V3 = "rs_vlm.mask_grounded_train_workflow.v3"
STAGE5_WORKFLOW_SCHEMA_V4 = "rs_vlm.mask_grounded_train_workflow.v4"
STAGE5_CUDA_CACHE_CLEANUP_INTERVAL_STEPS = 1
_GATE_B_STATIC_PROTOCOL_FILENAME = "rs_generaldesc_gate_b_qwen3vl_2b.yaml"


def _emit(
    callback: Callable[[Mapping[str, Any]], None] | None,
    event: str,
    *,
    schema_version: str = STAGE5_WORKFLOW_SCHEMA,
    **details: Any,
) -> None:
    if callback is not None:
        callback({"schema_version": schema_version, "event": event, **details})


def _workflow_schema(config: Stage5Config) -> str:
    if config.schema_version == STAGE5_CONFIG_SCHEMA_V4:
        return STAGE5_WORKFLOW_SCHEMA_V4
    return (
        STAGE5_WORKFLOW_SCHEMA_V3
        if config.schema_version == STAGE5_CONFIG_SCHEMA_V3
        else STAGE5_WORKFLOW_SCHEMA
    )


def _prepare_stage5_training_memory(device: torch.device) -> None:
    """清理 retention probe 遗留的无用缓存，保留活跃模型权重。"""

    gc.collect()
    clear_cuda_cache(device)


def _stage5_cuda_policy(
    config: Stage5Config,
    resource_profile_identity: Mapping[str, Any] | None,
) -> CudaTelemetryPolicy | None:
    resource = config.resource_contract
    if resource is None:
        if resource_profile_identity is not None:
            raise AssertionError("非 v4 配置不能绑定 resource profile")
        return None
    if resource_profile_identity is None:
        raise AssertionError("M7 v4 缺少 resource profile identity")
    return CudaTelemetryPolicy(
        schema_version=resource.telemetry_schema,
        allocator_profile=resource.allocator_profile,
        microbatch_cache_policy=resource.microbatch_cache_policy,
        min_cuda_free_bytes=resource.min_cuda_free_bytes,
        max_microbatches=resource.telemetry_max_microbatches,
        synchronize_cuda=resource.synchronize_cuda,
        resource_profile_identity=dict(resource_profile_identity),
    )


def _stage5_gate_b_protocol_path(config: Stage5Config) -> Path:
    """返回 Gate B 静态 YAML；frozen JSON 仅由 selection validator 旁路读取。"""

    if config.retention_contract is not None:
        return config.retention_contract.gate_b_protocol_path
    return config.base.config_path.parent / _GATE_B_STATIC_PROTOCOL_FILENAME


def _legacy_gate_b_root(config: Stage5Config) -> Path:
    return (
        config.config_path.parents[3]
        / "outputs"
        / "phase4_rs_vlm"
        / "rs_generaldesc_gate_b_qwen3vl_2b_v1"
    )


def _stage5_gate_b_selection_path(config: Stage5Config) -> Path:
    if config.retention_contract is not None:
        return config.retention_contract.gate_b_selection_path
    return _legacy_gate_b_root(config) / "selection" / "gate_b_selection.json"


def _stage5_rs_general_predictions_path(config: Stage5Config) -> Path:
    if config.retention_contract is not None:
        return config.retention_contract.rs_general_predictions_path
    return _legacy_gate_b_root(config) / "adapter" / "predictions.jsonl"


def _stage5_retention_max_new_tokens(config: Stage5Config) -> int:
    if config.retention_contract is not None:
        return config.retention_contract.max_new_tokens
    return 384


def _compact_benchmark_identity(dataset: CompactTrainingMessageDataset) -> BenchmarkIdentity:
    manifest_sha = sha256_file(dataset.root / "manifest.json")
    messages_sha = sha256_file(dataset.root / "messages.jsonl")
    parent_count = len({str(row["parent_id"]) for row in dataset.records})
    return BenchmarkIdentity(
        root=dataset.root,
        manifest_schema=str(dataset.manifest["schema_version"]),
        canonical_schema=str(dataset.manifest["assistant_target_schema"]),
        build_id=str(dataset.manifest["compact_id"]),
        semantic_config_sha256=manifest_sha,
        payload_sha256=messages_sha,
        hash_manifest_sha256=sha256_file(dataset.root / "SHA256SUMS.jsonl"),
        benchmark_scope="mask_grounded_region_compact_train_only",
        source_roots_embedded=True,
        deep_validation_saved=True,
        formal_acceptance_eligible=False,
        formal_acceptance_blockers=("model_assisted_supervision", "no_gold", "development_only"),
        record_count=len(dataset),
        parent_count=parent_count,
    )


def _processor(config: Stage5Config) -> VLMProcessorAdapter:
    return build_processor_adapter(config.base)


def _prompt_only_adaptation(config: Stage5Config) -> AdaptationSection:
    return replace(
        config.adaptation,
        strategy="prompt_only",
        target_modules=(),
        rank=0,
        alpha=0,
        dropout=0.0,
    )


def _load_warm_start(
    model: VLMModelAdapter,
    processor_identity: Mapping[str, Any],
    config: Stage5Config,
) -> dict[str, Any]:
    """只加载旧 LoRA tensor；明确不加载其 optimizer/scheduler/RNG/sampler。"""

    identity = verify_warm_start_files(config)
    manifest = read_json(config.warm_start.checkpoint_root / "manifest.json")
    if (
        not isinstance(manifest, dict)
        or manifest.get("cursor", {}).get("global_step") != config.warm_start.checkpoint_step
        or manifest.get("model_identity") != model.identity.to_dict()
        or manifest.get("processor_identity") != dict(processor_identity)
        or manifest.get("trainable_parameter_names") != list(model.trainable_names)
    ):
        raise ModelError(ReasonCode.CHECKPOINT_INCOMPATIBLE, "RS-General warm-start topology/identity 不兼容")
    tensors = load_safetensors(
        str(config.warm_start.checkpoint_root / "adapter" / "adapter_model.safetensors"),
        device="cpu",
    )
    model.load_trainable_state_dict(tensors)
    return {
        **identity,
        "loaded_components": ["trainable_lora_state"],
        "fresh_components": ["optimizer", "scheduler", "rng", "sampler"],
        "source_checkpoint_read_only": True,
    }


def _latest_checkpoint(training_root: Path) -> Path | None:
    checkpoints = training_root / "checkpoints"
    if not checkpoints.is_dir() or checkpoints.is_symlink():
        return None
    values = sorted(
        (path for path in checkpoints.glob("step-*") if path.is_dir() and not path.is_symlink()),
        key=lambda path: path.name,
    )
    return None if not values else values[-1]


def _training_complete(training_root: Path) -> bool:
    report_path = training_root / "training_report.json"
    if not report_path.exists():
        return False
    report = read_json(report_path)
    status = report.get("status")
    step = report.get("cursor", {}).get("global_step")
    if status == "completed" and step == 1000:
        return True
    if (
        status == "paused"
        and isinstance(step, int)
        and not isinstance(step, bool)
        and 0 < step < 1000
    ):
        return False
    raise ModelError(
        ReasonCode.CHECKPOINT_CORRUPT,
        "Mask-Grounded training report 状态/step 非法",
    )


def _load_stage5_best(
    *,
    model: VLMModelAdapter,
    processor_identity: Mapping[str, Any],
    config: Stage5Config,
    benchmark_identity: BenchmarkIdentity,
    selection: Any,
    cuda_resource_identity: Mapping[str, Any] | None = None,
) -> Path:
    pointer = read_json(config.run.output_root / "best_checkpoint.json")
    if pointer.get("selection_metric") != "region_monitor_loss":
        raise ModelError(ReasonCode.CHECKPOINT_CORRUPT, "best checkpoint 未按 Region monitor loss 选择")
    checkpoint = config.run.output_root / str(pointer["checkpoint"])
    payload = CheckpointManager().load(
        checkpoint,
        expected_config_semantic_sha256=config.semantic_sha256,
        expected_benchmark_identity=benchmark_identity.training_identity_dict(),
        expected_validation_selection_identity=selection.identity_dict(),
        expected_model_identity=model.identity.to_dict(),
        expected_processor_identity=processor_identity,
        expected_training_layout=training_layout_identity(
            config,
            cuda_cache_cleanup_interval_steps=(
                STAGE5_CUDA_CACHE_CLEANUP_INTERVAL_STEPS
            ),
            cuda_resource_identity=cuda_resource_identity,
        ),
        expected_trainable_names=model.trainable_names,
    )
    model.load_trainable_state_dict(payload.trainable_state)
    return payload.root


def run_stage5_workflow(
    config_path: Path | str,
    *,
    stop_after_steps: int | None = None,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """运行固定状态机；已有合法阶段只读复用，非法/部分输出拒绝覆盖。"""

    config = load_stage5_config(config_path)
    allowed_stops = (
        {1, 20, 100}
        if config.schema_version == STAGE5_CONFIG_SCHEMA_V4
        else {1, 20}
    )
    if (
        isinstance(stop_after_steps, bool)
        or stop_after_steps not in {None, *allowed_stops}
    ):
        raise ModelError(
            ReasonCode.TYPE_MISMATCH,
            "Mask-Grounded bounded smoke stop_after_steps 与 schema 不匹配",
        )
    if config.resource_contract is not None:
        if (
            config.resource_contract.execution_mode == "resource_gate"
            and stop_after_steps is None
        ) or (
            config.resource_contract.execution_mode == "formal_training"
            and stop_after_steps is not None
        ):
            raise ModelError(
                ReasonCode.TYPE_MISMATCH,
                "M7 v4 resource_gate/formal_training 与 stop_after_steps 不匹配",
            )
        allocator_environment_identity(
            config.resource_contract.allocator_profile
        )
    smoke_mode = stop_after_steps is not None
    workflow_schema = _workflow_schema(config)

    def emit(event: str, **details: Any) -> None:
        _emit(
            progress_callback,
            event,
            schema_version=workflow_schema,
            **details,
        )

    compact = CompactTrainingMessageDataset(config.data_contract.compact_training_root)
    split = split_compact_by_parent(compact, seed=config.data_contract.split_seed)
    config = with_monitor_parent_count(config, len(split.monitor_parents))
    identity = _compact_benchmark_identity(compact)
    access = open_benchmark_access(config.base)
    warm_identity = verify_warm_start_files(config)
    retention_identity = verify_retention_reference_files(config)
    resource_profile_identity = (
        verify_stage5_resource_profile(config)
        if config.resource_contract is not None
        else None
    )
    loss_parity_identity = (
        verify_qwen35_loss_parity(config)
        if config.resource_contract is not None
        else None
    )
    resource_evidence_identity = (
        None
        if resource_profile_identity is None
        else {
            **resource_profile_identity,
            "loss_parity": loss_parity_identity,
        }
    )
    cuda_policy = _stage5_cuda_policy(config, resource_evidence_identity)
    workflow_root = config.workflow_root
    if workflow_root.exists() and (not workflow_root.is_dir() or workflow_root.is_symlink()):
        raise ModelError(ReasonCode.OUTPUT_LINK, "Stage 5 workflow root 必须是普通目录")
    if stop_after_steps == 20 and not workflow_root.is_dir():
        raise ModelError(
            ReasonCode.CHECKPOINT_INCOMPATIBLE,
            "20-step smoke 必须从已通过的同根 1-step checkpoint 恢复",
        )
    if stop_after_steps == 100 and not workflow_root.is_dir():
        raise ModelError(
            ReasonCode.CHECKPOINT_INCOMPATIBLE,
            "100-step resource gate 必须从已通过的同根 20-step checkpoint 恢复",
        )
    workflow_root.mkdir(parents=True, exist_ok=True)
    snapshot_path = workflow_root / "stage5_config_snapshot.json"
    split_path = workflow_root / "region_parent_split.json"
    if snapshot_path.exists():
        if read_json(snapshot_path) != config.snapshot_dict() or read_json(split_path) != split.to_dict():
            raise ModelError(ReasonCode.CHECKPOINT_INCOMPATIBLE, "Stage 5 config/split 与既有运行漂移")
    else:
        if any(workflow_root.iterdir()):
            raise ModelError(ReasonCode.OUTPUT_EXISTS, "Stage 5 root 非空但缺少 config snapshot")
        atomic_write_json(snapshot_path, config.snapshot_dict())
        atomic_write_json(split_path, split.to_dict())
        atomic_write_json(workflow_root / "workflow_state.json", {
            "schema_version": workflow_schema,
            "stage": "preflight",
            "compact_manifest_sha256": sha256_file(compact.root / "manifest.json"),
            "split_identity_sha256": split.identity_sha256,
            "warm_start": warm_identity,
            "retention_reference": retention_identity,
            "resource_profile": resource_profile_identity,
            "loss_parity": loss_parity_identity,
            "bounded_smoke": smoke_mode,
            "formal_acceptance": False,
        })
    emit(
        "preflight_complete",
        compact_records=len(compact),
        train_records=len(split.train_indices),
        monitor_records=len(split.monitor_indices),
        train_parents=len(split.train_parents),
        monitor_parents=len(split.monitor_parents),
        backend=config.model.backend,
        bounded_smoke=smoke_mode,
    )
    if not torch.cuda.is_available():
        raise ModelError(ReasonCode.CUDA_REQUIRED, "Mask-Grounded training/retention 要求 CUDA")
    device = torch.device("cuda")
    processor = _processor(config)
    processor_identity = processor.identity()
    monitor = RegionSubsetDataset(
        compact,
        split.monitor_indices,
        logical_role=REGION_MONITOR_ROLE,
    )
    selection = build_region_monitor_selection(
        monitor,
        benchmark_build_id=identity.build_id,
        benchmark_payload_sha256=identity.payload_sha256,
        seed=config.run.seed,
    )
    with tempfile.TemporaryDirectory(prefix="stage5_rs_general_derived_") as temporary:
        canonical = RSGeneralDescDataset(
            config.data.benchmark_root,
            roles=("external_train", "external_val"),
            task_families=config.data.task_families,
            load_assets=False,
            seed=config.run.seed,
            expected_manifest_sha256=config.data.expected_manifest_sha256,
            verifier=access.verifier,
        )
        replay = ExternalDescriptionDataset(
            canonical,
            derived_root=Path(temporary) / "replay",
            seed=config.run.seed,
            roles=("external_train",),
        )
        retention = ExternalDescriptionDataset(
            canonical,
            derived_root=Path(temporary) / "retention",
            seed=config.run.seed,
            roles=("external_val",),
        )
        retention_selection = select_bounded_external_validation(
            retention,
            benchmark_build_id=access.identity.build_id,
            benchmark_payload_sha256=access.identity.payload_sha256,
            seed=config.run.seed,
            max_parents=config.data_contract.replay_validation_parents,
        )
        loss_path = workflow_root / "retention_losses.json"
        losses = None if smoke_mode else (
            read_json(loss_path) if loss_path.exists() else {
                "schema_version": "rs_vlm.mask_grounded_train_retention_losses.v2",
                "selection": retention_selection.to_dict(),
                "base": None,
                "rs_general_adapter": None,
                "mask_grounded_region_adapter": None,
                "retention_gate_frozen": False,
                "formal_acceptance": False,
            }
        )
        if losses is not None and losses["base"] is None:
            emit("base_retention_probe_starting")
            base_config = replace(
                config.base,
                adaptation=_prompt_only_adaptation(config),
            )
            base_model = build_model_adapter(
                base_config,
                device=device,
                gradient_checkpointing=False,
            )
            if config.schema_version == STAGE5_CONFIG_SCHEMA_V4:
                base_model = wrap_qwen35_for_supervised_position_training(
                    base_model
                )
            losses["base"] = evaluate_teacher_forced_loss(
                model=base_model,
                collator=DescriptionCollator(processor, training=True),
                dataset=retention,
                selection=retention_selection,
                device=device,
                step=0,
            ).to_dict()
            atomic_write_json(loss_path, losses)
            del base_model
            gc.collect()
            torch.cuda.empty_cache()
            emit("base_retention_probe_complete")
        model = build_model_adapter(
            config.base,
            device=device,
            gradient_checkpointing=True,
        )
        if config.schema_version == STAGE5_CONFIG_SCHEMA_V4:
            model = wrap_qwen35_for_supervised_position_training(model)
        warm = _load_warm_start(model, processor_identity, config)
        if losses is not None and losses["rs_general_adapter"] is None:
            emit("rs_general_retention_probe_starting")
            losses["rs_general_adapter"] = evaluate_teacher_forced_loss(
                model=model,
                collator=DescriptionCollator(processor, training=True),
                dataset=retention,
                selection=retention_selection,
                device=device,
                step=0,
            ).to_dict()
            atomic_write_json(loss_path, losses)
            emit("rs_general_retention_probe_complete")
        region_train = RegionSubsetDataset(
            compact,
            split.train_indices,
            logical_role=REGION_TRAIN_ROLE,
        )
        mixed = Stage5MixedDataset(region_train, replay)
        training_collator = DescriptionCollator(processor, training=True)
        if cuda_policy is not None:
            assert config.resource_contract is not None
            worst_case_root = (
                config.resource_contract.resource_gate_root
                / "worst_case_cuda_gate"
            )
            if (
                config.resource_contract.execution_mode == "resource_gate"
                and stop_after_steps == 1
            ):
                emit("worst_case_cuda_gate_starting")
                run_worst_case_cuda_gate(
                    config=config,
                    dataset=mixed,
                    collator=training_collator,
                    model=model,
                    device=device,
                    policy=cuda_policy,
                    output_root=worst_case_root,
                )
                emit("worst_case_cuda_gate_complete")
            else:
                verify_worst_case_cuda_gate(
                    config=config,
                    policy=cuda_policy,
                    output_root=worst_case_root,
                )
        trainer = DescriptionTrainer(
            config=config,
            model=model,
            collator=training_collator,
            validation_dataset=monitor,
            validation_collator=DescriptionCollator(processor, training=True),
            validation_selection=selection,
            benchmark_identity=identity,
            processor_identity=processor_identity,
            device=device,
            allowed_training_roles=frozenset({REGION_TRAIN_ROLE, REPLAY_ROLE}),
            sampler_factory=lambda dataset: Stage5MixedSampler(dataset, seed=config.run.seed),
            validation_evaluator=evaluate_region_monitor_loss,
            validation_selection_metric="region_monitor_loss",
            validation_row_parser=parse_region_monitor_result,
            cuda_cache_cleanup_interval_steps=(
                STAGE5_CUDA_CACHE_CLEANUP_INTERVAL_STEPS
            ),
            cuda_resource_telemetry=(
                None
                if cuda_policy is None
                else CudaMicrobatchTelemetry(policy=cuda_policy, device=device)
            ),
        )
        training_complete = _training_complete(config.run.output_root)
        if training_complete and smoke_mode:
            raise ModelError(
                ReasonCode.CHECKPOINT_INCOMPATIBLE,
                "bounded smoke 根不得包含已完成的 1000-step training",
            )
        if not training_complete:
            resume = _latest_checkpoint(config.run.output_root)
            if stop_after_steps == 1 and resume is not None:
                raise ModelError(
                    ReasonCode.CHECKPOINT_INCOMPATIBLE,
                    "1-step smoke 必须写入全新 training root",
                )
            if stop_after_steps == 20 and (
                resume is None or resume.name != "step-00000001"
            ):
                raise ModelError(
                    ReasonCode.CHECKPOINT_INCOMPATIBLE,
                    "20-step smoke 只接受同根 step-00000001 checkpoint",
                )
            if stop_after_steps == 100 and (
                resume is None or resume.name != "step-00000020"
            ):
                raise ModelError(
                    ReasonCode.CHECKPOINT_INCOMPATIBLE,
                    "100-step resource gate 只接受同根 step-00000020 checkpoint",
                )
            emit(
                "training_starting" if resume is None else "training_resuming",
                resume_checkpoint=None if resume is None else str(resume),
                warm_start=warm,
                stop_after_steps=stop_after_steps,
            )
            # retention probe 产生了可变长度 forward cache；正式训练前只清理
            # Python 无用对象与 allocator 空闲块，不触碰模型权重或训练超参数。
            _prepare_stage5_training_memory(device)
            training_result = trainer.fit(
                mixed,
                resume_checkpoint=resume,
                stop_after_steps=stop_after_steps,
            )
            if training_result.status == "paused":
                state = {
                    "schema_version": workflow_schema,
                    "stage": "bounded_smoke_paused",
                    "config_semantic_sha256": config.semantic_sha256,
                    "compact_manifest_sha256": sha256_file(
                        compact.root / "manifest.json"
                    ),
                    "region_split_identity_sha256": split.identity_sha256,
                    "warm_start": warm,
                    "retention_reference": retention_identity,
                    "resource_profile": resource_profile_identity,
                    "loss_parity": loss_parity_identity,
                    "retention_probes_executed": False,
                    "execution_stop_step": stop_after_steps,
                    "cursor": training_result.cursor.to_dict(),
                    "checkpoint": str(training_result.checkpoint),
                    "cuda_peak_gib": training_result.cuda_peak_gib,
                    "formal_acceptance": False,
                    "scientific_acceptance": False,
                    "sealed_test_evaluated": False,
                }
                atomic_write_json(workflow_root / "workflow_state.json", state)
                emit(
                    "bounded_smoke_complete",
                    step=training_result.cursor.global_step,
                    checkpoint=str(training_result.checkpoint),
                    cuda_peak_gib=training_result.cuda_peak_gib,
                )
                return {"ok": True, "root": str(workflow_root), **state}
            emit("training_complete")
        best = _load_stage5_best(
            model=model,
            processor_identity=processor_identity,
            config=config,
            benchmark_identity=identity,
            selection=selection,
            cuda_resource_identity=(
                None if cuda_policy is None else cuda_policy.layout_identity()
            ),
        )
        if losses is None:
            raise AssertionError("正式训练完成后 retention losses 不得为空")
        if losses["mask_grounded_region_adapter"] is None:
            emit(
                "region_adapter_retention_probe_starting",
                checkpoint=str(best),
            )
            losses["mask_grounded_region_adapter"] = evaluate_teacher_forced_loss(
                model=model,
                collator=DescriptionCollator(processor, training=True),
                dataset=retention,
                selection=retention_selection,
                device=device,
                step=int(read_json(config.run.output_root / "best_checkpoint.json")["step"]),
            ).to_dict()
            atomic_write_json(loss_path, losses)
            emit("region_adapter_retention_probe_complete")
        retention_root = workflow_root / "rs_general_retention"
        if not retention_root.exists():
            emit("retention_report_starting")
            run_rs_general_retention_report(
                protocol_path=_stage5_gate_b_protocol_path(config),
                selection_path=_stage5_gate_b_selection_path(config),
                frozen_rs_predictions_path=(
                    _stage5_rs_general_predictions_path(config)
                ),
                model=model,
                processor_adapter=processor,
                config=config,
                device=device,
                max_new_tokens=_stage5_retention_max_new_tokens(config),
                output_root=retention_root,
            )
        state = {
            "schema_version": workflow_schema,
            "stage": "complete",
            "config_semantic_sha256": config.semantic_sha256,
            "compact_manifest_sha256": sha256_file(compact.root / "manifest.json"),
            "region_split_identity_sha256": split.identity_sha256,
            "region_train_records": len(split.train_indices),
            "region_monitor_records": len(split.monitor_indices),
            "region_micro_ratio": 0.9,
            "rs_general_replay_micro_ratio": 0.1,
            "warm_start": warm,
            "retention_reference": retention_identity,
            "resource_profile": resource_profile_identity,
            "loss_parity": loss_parity_identity,
            "cuda_resource_identity": (
                None if cuda_policy is None else cuda_policy.layout_identity()
            ),
            "best_checkpoint": str(best),
            "reference_authority": "automatic_contract_only",
            "expert_metrics_available": False,
            "retention_gate_frozen": False,
            "oa_grounded_eval_consumed": False,
            "formal_acceptance": False,
            "scientific_acceptance": False,
            "sealed_test_evaluated": False,
        }
        atomic_write_json(workflow_root / "workflow_state.json", state)
        emit("workflow_complete", best_checkpoint=str(best))
        return {"ok": True, "root": str(workflow_root), **state}
