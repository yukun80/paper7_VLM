"""Stage 5 Base→RS warm-start→训练→自动评价的可恢复状态机。"""

from __future__ import annotations

from dataclasses import replace
import gc
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping

import torch
from safetensors.torch import load_file as load_safetensors

from oa_groundrag.landslide_evidence.compact_training import CompactTrainingMessageDataset
from oa_groundrag.phase3.common import atomic_write_json, canonical_json, read_json, sha256_file, sha256_text
from oa_groundrag.phase3.dataset import RSGeneralDescDataset

from .checkpoint import CheckpointManager
from .config import AdaptationSection
from .data import ExternalDescriptionDataset
from .errors import ModelError, ReasonCode
from .model import Qwen3VLModelAdapter
from .preflight import BenchmarkIdentity, open_benchmark_access
from .processing import DescriptionCollator, Qwen3VLProcessorAdapter
from .stage5_config import (
    Stage5Config,
    load_stage5_config,
    verify_warm_start_files,
    with_monitor_parent_count,
)
from .stage5_data import (
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
from .stage5_evaluation import (
    evaluate_stage5_dev,
    load_stage5_eval_samples,
    run_rs_general_retention_report,
    run_stage5_region_inference,
)
from .trainer import DescriptionTrainer, clear_cuda_cache, training_layout_identity
from .validation import evaluate_teacher_forced_loss, select_bounded_external_validation


STAGE5_WORKFLOW_SCHEMA = "rs_vlm.mask_grounded_stage5_workflow.v1"
STAGE5_CUDA_CACHE_CLEANUP_INTERVAL_STEPS = 1
_GATE_B_STATIC_PROTOCOL_FILENAME = "rs_generaldesc_gate_b_qwen3vl_2b.yaml"


def _emit(
    callback: Callable[[Mapping[str, Any]], None] | None,
    event: str,
    **details: Any,
) -> None:
    if callback is not None:
        callback({"schema_version": STAGE5_WORKFLOW_SCHEMA, "event": event, **details})


def _prepare_stage5_training_memory(device: torch.device) -> None:
    """清理 baseline 遗留的无用 Python/CUDA 缓存，保留活跃模型权重。"""

    gc.collect()
    clear_cuda_cache(device)


def _stage5_gate_b_protocol_path(config: Stage5Config) -> Path:
    """返回 Gate B 静态 YAML；frozen JSON 仅由 selection validator 旁路读取。"""

    return config.config_path.parent / _GATE_B_STATIC_PROTOCOL_FILENAME


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


def _processor(config: Stage5Config) -> Qwen3VLProcessorAdapter:
    return Qwen3VLProcessorAdapter(
        processor_path=config.model.processor_path,
        local_files_only=config.model.local_files_only,
        trust_remote_code=config.model.trust_remote_code,
        min_pixels=config.limits.min_pixels,
        max_pixels=config.limits.max_pixels,
        max_images=config.limits.max_images,
        max_input_tokens=config.limits.max_input_tokens,
    )


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
    model: Qwen3VLModelAdapter,
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


def _phase_complete(root: Path, *, model_role: str) -> bool:
    prediction = root / "predictions"
    evaluation = root / "evaluation"
    existing = [path.exists() or path.is_symlink() for path in (prediction, evaluation)]
    if not any(existing):
        return False
    if not all(existing):
        raise ModelError(ReasonCode.OUTPUT_EXISTS, f"{model_role} baseline 部分发布，拒绝覆盖")
    prediction_manifest = read_json(prediction / "manifest.json")
    report = read_json(evaluation / "report.json")
    if (
        prediction_manifest.get("model_role") != model_role
        or prediction_manifest.get("input_count") != 340
        or report.get("model_role") != model_role
        or report.get("reference_authority") != "automatic_contract_only"
        or report.get("formal_acceptance") is not False
    ):
        raise ModelError(ReasonCode.PREDICTION_IDENTITY_MISMATCH, f"{model_role} baseline 已存在但无效")
    return True


def _run_dev_phase(
    *,
    root: Path,
    model_role: str,
    config: Stage5Config,
    samples: tuple[Any, ...],
    processor: Qwen3VLProcessorAdapter,
    model: Qwen3VLModelAdapter,
    device: torch.device,
) -> None:
    if _phase_complete(root, model_role=model_role):
        return
    prediction = run_stage5_region_inference(
        config=config,
        samples=samples,
        collator=DescriptionCollator(processor, training=False),
        model=model,
        processor=processor.processor,
        output_root=root / "predictions",
        model_role=model_role,
        device=device,
    )
    evaluate_stage5_dev(
        eval_root=config.data_contract.eval_dev_root,
        prediction_root=prediction,
        output_root=root / "evaluation",
        model_role=model_role,
    )


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
    if report.get("status") != "completed" or report.get("cursor", {}).get("global_step") != 1000:
        raise ModelError(ReasonCode.CHECKPOINT_CORRUPT, "Stage 5 training report 已存在但未完成/非法")
    return True


def _load_stage5_best(
    *,
    model: Qwen3VLModelAdapter,
    processor_identity: Mapping[str, Any],
    config: Stage5Config,
    benchmark_identity: BenchmarkIdentity,
    selection: Any,
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
        ),
        expected_trainable_names=model.trainable_names,
    )
    model.load_trainable_state_dict(payload.trainable_state)
    return payload.root


def run_stage5_workflow(
    config_path: Path | str,
    *,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """运行固定状态机；已有合法阶段只读复用，非法/部分输出拒绝覆盖。"""

    config = load_stage5_config(config_path)
    compact = CompactTrainingMessageDataset(config.data_contract.compact_training_root)
    split = split_compact_by_parent(compact, seed=config.data_contract.split_seed)
    config = with_monitor_parent_count(config, len(split.monitor_parents))
    identity = _compact_benchmark_identity(compact)
    access = open_benchmark_access(config.base)
    warm_identity = verify_warm_start_files(config)
    samples = load_stage5_eval_samples(config.data_contract.eval_dev_root)
    workflow_root = config.workflow_root
    if workflow_root.exists() and (not workflow_root.is_dir() or workflow_root.is_symlink()):
        raise ModelError(ReasonCode.OUTPUT_LINK, "Stage 5 workflow root 必须是普通目录")
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
            "schema_version": STAGE5_WORKFLOW_SCHEMA,
            "stage": "preflight",
            "compact_manifest_sha256": sha256_file(compact.root / "manifest.json"),
            "split_identity_sha256": split.identity_sha256,
            "warm_start": warm_identity,
            "formal_acceptance": False,
        })
    _emit(
        progress_callback,
        "preflight_complete",
        compact_records=len(compact),
        train_records=len(split.train_indices),
        monitor_records=len(split.monitor_indices),
        train_parents=len(split.train_parents),
        monitor_parents=len(split.monitor_parents),
    )
    if not torch.cuda.is_available():
        raise ModelError(ReasonCode.CUDA_REQUIRED, "run-stage5-workflow 的 baseline/training 要求 CUDA")
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
        losses = read_json(loss_path) if loss_path.exists() else {
            "schema_version": "rs_vlm.mask_grounded_stage5_retention_losses.v1",
            "selection": retention_selection.to_dict(),
            "base": None,
            "rs_general_adapter": None,
            "mask_grounded_region_adapter": None,
            "retention_gate_frozen": False,
            "formal_acceptance": False,
        }
        base_root = workflow_root / "base_gt_mask_baseline"
        if not _phase_complete(base_root, model_role="base") or losses["base"] is None:
            _emit(progress_callback, "base_baseline_starting")
            base_model = Qwen3VLModelAdapter.load(
                config.model,
                _prompt_only_adaptation(config),
                device=device,
                gradient_checkpointing=False,
            )
            _run_dev_phase(
                root=base_root,
                model_role="base",
                config=config,
                samples=samples,
                processor=processor,
                model=base_model,
                device=device,
            )
            if losses["base"] is None:
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
            _emit(progress_callback, "base_baseline_complete")
        model = Qwen3VLModelAdapter.load(
            config.model,
            config.adaptation,
            device=device,
            gradient_checkpointing=True,
        )
        warm = _load_warm_start(model, processor_identity, config)
        rs_root = workflow_root / "rs_general_adapter_gt_mask_baseline"
        if not _phase_complete(rs_root, model_role="rs_general_adapter") or losses["rs_general_adapter"] is None:
            _emit(progress_callback, "rs_general_baseline_starting")
            _run_dev_phase(
                root=rs_root,
                model_role="rs_general_adapter",
                config=config,
                samples=samples,
                processor=processor,
                model=model,
                device=device,
            )
            if losses["rs_general_adapter"] is None:
                losses["rs_general_adapter"] = evaluate_teacher_forced_loss(
                    model=model,
                    collator=DescriptionCollator(processor, training=True),
                    dataset=retention,
                    selection=retention_selection,
                    device=device,
                    step=0,
                ).to_dict()
                atomic_write_json(loss_path, losses)
            _emit(progress_callback, "rs_general_baseline_complete")
        region_train = RegionSubsetDataset(
            compact,
            split.train_indices,
            logical_role=REGION_TRAIN_ROLE,
        )
        mixed = Stage5MixedDataset(region_train, replay)
        trainer = DescriptionTrainer(
            config=config,
            model=model,
            collator=DescriptionCollator(processor, training=True),
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
        )
        if not _training_complete(config.run.output_root):
            resume = _latest_checkpoint(config.run.output_root)
            _emit(
                progress_callback,
                "training_starting" if resume is None else "training_resuming",
                resume_checkpoint=None if resume is None else str(resume),
                warm_start=warm,
            )
            # 两套 baseline 产生了大量可变长度 generation cache；正式训练前只清理
            # Python 无用对象与 allocator 空闲块，不触碰模型权重或训练超参数。
            _prepare_stage5_training_memory(device)
            trainer.fit(mixed, resume_checkpoint=resume)
            _emit(progress_callback, "training_complete")
        best = _load_stage5_best(
            model=model,
            processor_identity=processor_identity,
            config=config,
            benchmark_identity=identity,
            selection=selection,
        )
        region_root = workflow_root / "mask_grounded_region_adapter_gt_mask"
        _emit(progress_callback, "region_adapter_evaluation_starting", checkpoint=str(best))
        _run_dev_phase(
            root=region_root,
            model_role="mask_grounded_region_adapter",
            config=config,
            samples=samples,
            processor=processor,
            model=model,
            device=device,
        )
        if losses["mask_grounded_region_adapter"] is None:
            losses["mask_grounded_region_adapter"] = evaluate_teacher_forced_loss(
                model=model,
                collator=DescriptionCollator(processor, training=True),
                dataset=retention,
                selection=retention_selection,
                device=device,
                step=int(read_json(config.run.output_root / "best_checkpoint.json")["step"]),
            ).to_dict()
            atomic_write_json(loss_path, losses)
        retention_root = workflow_root / "rs_general_retention"
        if not retention_root.exists():
            _emit(progress_callback, "retention_report_starting")
            gate_b_root = (
                config.config_path.parents[2]
                / "outputs"
                / "phase4_rs_vlm"
                / "rs_generaldesc_gate_b_qwen3vl_2b_v1"
            )
            run_rs_general_retention_report(
                protocol_path=_stage5_gate_b_protocol_path(config),
                selection_path=gate_b_root / "selection" / "gate_b_selection.json",
                frozen_rs_predictions_path=gate_b_root / "adapter" / "predictions.jsonl",
                model=model,
                processor_adapter=processor,
                config=config,
                device=device,
                output_root=retention_root,
            )
        state = {
            "schema_version": STAGE5_WORKFLOW_SCHEMA,
            "stage": "complete",
            "config_semantic_sha256": config.semantic_sha256,
            "compact_manifest_sha256": sha256_file(compact.root / "manifest.json"),
            "region_split_identity_sha256": split.identity_sha256,
            "region_train_records": len(split.train_indices),
            "region_monitor_records": len(split.monitor_indices),
            "region_micro_ratio": 0.9,
            "rs_general_replay_micro_ratio": 0.1,
            "warm_start": warm,
            "best_checkpoint": str(best),
            "reference_authority": "automatic_contract_only",
            "expert_metrics_available": False,
            "retention_gate_frozen": False,
            "formal_acceptance": False,
            "scientific_acceptance": False,
            "sealed_test_evaluated": False,
        }
        atomic_write_json(workflow_root / "workflow_state.json", state)
        _emit(progress_callback, "workflow_complete", best_checkpoint=str(best))
        return {"ok": True, "root": str(workflow_root), **state}
