"""仅从 step 0 启动的 LoRA trainer、External 有界验证与严格 checkpoint。"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from oa_groundrag.artifacts.io import (
    atomic_write_json,
    atomic_write_jsonl,
    first_symlink_component,
)
from oa_groundrag.data.rs_general.dataset import ParentBalancedSampler

from oa_groundrag.vlm.checkpoint import CheckpointManager, TrainingCursor
from oa_groundrag.vlm.config import VLMConfig
from oa_groundrag.grounding.contracts import (
    RUN_MANIFEST_SCHEMA_VERSION,
    SAMPLE_TRACE_SCHEMA_VERSION,
)
from oa_groundrag.vlm.data import REQUIRED_EXTERNAL_TASKS, DescriptionSample
from oa_groundrag.vlm.errors import ModelError, ReasonCode
from .input_pipeline import (
    INPUT_PIPELINE_PREFETCH,
    INPUT_PIPELINE_SYNC,
    DeterministicBatchPlanner,
    OrderedBatchPrefetcher,
)
from .cuda_telemetry import CudaMicrobatchTelemetry
from oa_groundrag.vlm.preflight import BenchmarkIdentity
from oa_groundrag.vlm.processing import DescriptionCollator
from .progress import TrainingProgress
from .validation import (
    VALIDATION_RESULT_SCHEMA_VERSION,
    ValidationResult,
    ValidationSelection,
    evaluate_teacher_forced_loss,
    validation_is_better,
)


TRAIN_LOG_SCHEMA_VERSION = "rs_vlm.train_log.v1"
BEST_CHECKPOINT_SCHEMA_VERSION = (
    "rs_vlm.best_checkpoint.v1"
)
TRAINING_REPORT_SCHEMA_VERSION = (
    "rs_vlm.training_report.v1"
)


class TrainableAdapter(Protocol):
    model: nn.Module
    trainable_names: tuple[str, ...]
    trainable_parameter_count: int
    identity: Any

    def train(self) -> None:
        ...

    def eval(self) -> None:
        ...

    def forward(self, batch: Mapping[str, Any]) -> Any:
        ...

    def trainable_state_dict(self) -> Mapping[str, Tensor]:
        ...

    def load_trainable_state_dict(
        self,
        state: Mapping[str, Tensor],
    ) -> None:
        ...


class DescriptionTrainingDataset(Protocol):
    records: Sequence[Mapping[str, Any]]
    manifest: Mapping[str, Any]

    def __len__(self) -> int:
        ...

    def __getitem__(self, index: int) -> DescriptionSample:
        ...

    def set_epoch(self, epoch: int) -> None:
        ...


@dataclass(frozen=True)
class TrainingResult:
    cursor: TrainingCursor
    checkpoint: Path
    best_checkpoint: Path | None
    status: str
    max_steps: int
    last_loss: float
    run_elapsed_seconds: float
    cuda_peak_gib: float | None
    training_report: Path
    loss_history: tuple[float, ...]
    sample_trace: tuple[str, ...]

    def to_dict(self, *, include_trace: bool = False) -> dict[str, Any]:
        value: dict[str, Any] = {
            "cursor": self.cursor.to_dict(),
            "checkpoint": str(self.checkpoint),
            "best_checkpoint": (
                None
                if self.best_checkpoint is None
                else str(self.best_checkpoint)
            ),
            "status": self.status,
            "global_step": self.cursor.global_step,
            "max_steps": self.max_steps,
            "last_loss": self.last_loss,
            "run_elapsed_seconds": self.run_elapsed_seconds,
            "cuda_peak_gib": self.cuda_peak_gib,
            "training_report": str(self.training_report),
            "loss_history": list(self.loss_history),
        }
        if include_trace:
            value["sample_trace"] = list(self.sample_trace)
        return value


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def training_layout_identity(
    config: VLMConfig,
    *,
    cuda_cache_cleanup_interval_steps: int | None = None,
    cuda_resource_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """返回 checkpoint 身份所需的不可变物理训练布局。"""

    if (
        cuda_cache_cleanup_interval_steps is not None
        and (
            isinstance(cuda_cache_cleanup_interval_steps, bool)
            or not isinstance(cuda_cache_cleanup_interval_steps, int)
            or cuda_cache_cleanup_interval_steps <= 0
        )
    ):
        raise ModelError(
            ReasonCode.TYPE_MISMATCH,
            "CUDA cache 清理间隔必须是正整数或 null",
        )
    batch_size = config.training.batch_size
    accumulation_steps = config.training.gradient_accumulation_steps
    result = {
        "physical_batch_size": batch_size,
        "gradient_accumulation_steps": accumulation_steps,
        "effective_batch_size": batch_size * accumulation_steps,
        "input_pipeline_backend": (
            INPUT_PIPELINE_PREFETCH
            if config.training.num_workers > 0
            else INPUT_PIPELINE_SYNC
        ),
        "num_workers": config.training.num_workers,
        "prefetch_factor": config.training.prefetch_factor,
        "pin_memory": config.training.pin_memory,
        "sample_trace_schema_version": SAMPLE_TRACE_SCHEMA_VERSION,
    }
    # 仅 Stage 5 写入该扩展字段；默认省略以保持既有 Phase 3 checkpoint 身份不变。
    if cuda_cache_cleanup_interval_steps is not None:
        result["cuda_cache_cleanup_interval_steps"] = (
            cuda_cache_cleanup_interval_steps
        )
    if cuda_resource_identity is not None:
        if (
            not isinstance(cuda_resource_identity, Mapping)
            or not cuda_resource_identity
        ):
            raise ModelError(
                ReasonCode.TYPE_MISMATCH,
                "CUDA resource identity 必须是非空 mapping 或 null",
            )
        result["cuda_resource_identity"] = dict(cuda_resource_identity)
    return result


def clear_cuda_cache(device: torch.device) -> None:
    """仅在 CUDA 设备释放 allocator 未使用缓存，不改变活跃 tensor。"""

    if device.type == "cuda":
        torch.cuda.empty_cache()


def _scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    max_steps: int,
    warmup_ratio: float,
) -> LambdaLR:
    warmup_steps = int(math.ceil(max_steps * warmup_ratio))

    def scale(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return LambdaLR(optimizer, scale)


def _move_batch(
    batch: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    return {
        key: (
            value.to(device=device, non_blocking=device.type == "cuda")
            if isinstance(value, Tensor)
            else value
        )
        for key, value in batch.items()
    }


def _regular_file(path: Path, *, label: str) -> Path:
    linked = first_symlink_component(path)
    if (
        linked is not None
        or not path.is_file()
        or path.is_symlink()
        or path.stat().st_nlink != 1
    ):
        raise ModelError(
            ReasonCode.CHECKPOINT_INCOMPATIBLE,
            f"{label} 必须是普通单链接文件：{path}",
        )
    return path


def _artifact_int(
    value: Any,
    *,
    label: str,
    minimum: int = 0,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
    ):
        raise ModelError(
            ReasonCode.CHECKPOINT_INCOMPATIBLE,
            f"{label} 必须是 >= {minimum} 的整数",
        )
    return value


def _artifact_number(
    value: Any,
    *,
    label: str,
    minimum: float | None = None,
    allow_none: bool = False,
) -> float | None:
    if value is None and allow_none:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or (minimum is not None and float(value) < minimum)
    ):
        bound = "" if minimum is None else f"且 >= {minimum}"
        raise ModelError(
            ReasonCode.CHECKPOINT_INCOMPATIBLE,
            f"{label} 必须是有限数{bound}",
        )
    return float(value)


def _artifact_string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ModelError(
            ReasonCode.CHECKPOINT_INCOMPATIBLE,
            f"{label} 必须是非空字符串",
        )
    return value


def _cuda_memory(device: torch.device) -> dict[str, float | None]:
    if device.type != "cuda":
        return {
            "cuda_allocated_gib": None,
            "cuda_reserved_gib": None,
            "cuda_peak_gib": None,
        }
    divisor = float(1024**3)
    return {
        "cuda_allocated_gib": torch.cuda.memory_allocated(device) / divisor,
        "cuda_reserved_gib": torch.cuda.memory_reserved(device) / divisor,
        "cuda_peak_gib": torch.cuda.max_memory_allocated(device) / divisor,
    }


def _train_log_row(
    *,
    step: int,
    loss: float,
    ema_loss: float,
    learning_rate: float,
    gradient_norm: float,
    gradient_clipped: bool,
    samples: int,
    input_tokens: int,
    supervised_tokens: int,
    images: int,
    samples_per_second: float,
    tokens_per_second: float,
    step_duration_seconds: float,
    data_wait_seconds: float,
    data_wait_fraction: float,
    session_elapsed_seconds: float,
    run_elapsed_seconds: float,
    eta_seconds: float | None,
    memory: Mapping[str, float | None],
) -> dict[str, Any]:
    return {
        "schema_version": TRAIN_LOG_SCHEMA_VERSION,
        "step": step,
        "loss": loss,
        "ema_loss": ema_loss,
        "learning_rate": learning_rate,
        "gradient_norm": gradient_norm,
        "gradient_clipped": gradient_clipped,
        "samples": samples,
        "input_tokens": input_tokens,
        "supervised_tokens": supervised_tokens,
        "images": images,
        "samples_per_second": samples_per_second,
        "tokens_per_second": tokens_per_second,
        "step_duration_seconds": step_duration_seconds,
        "data_wait_seconds": data_wait_seconds,
        "data_wait_fraction": data_wait_fraction,
        "session_elapsed_seconds": session_elapsed_seconds,
        "run_elapsed_seconds": run_elapsed_seconds,
        "eta_seconds": eta_seconds,
        "cuda_allocated_gib": memory["cuda_allocated_gib"],
        "cuda_reserved_gib": memory["cuda_reserved_gib"],
        "cuda_peak_gib": memory["cuda_peak_gib"],
    }


def _validation_from_row(row: Mapping[str, Any]) -> ValidationResult:
    expected = {
        "schema_version",
        "step",
        "selection_metric",
        "macro_task_loss",
        "overall_loss",
        "task_losses",
        "task_counts",
        "sample_count",
        "parent_count",
        "input_tokens",
        "images",
        "duration_seconds",
        "formal_acceptance",
    }
    if (
        set(row) != expected
        or row.get("schema_version") != VALIDATION_RESULT_SCHEMA_VERSION
        or row.get("selection_metric") != "macro_task_loss"
        or row.get("formal_acceptance") is not False
        or not isinstance(row.get("task_losses"), dict)
        or not isinstance(row.get("task_counts"), dict)
    ):
        raise ModelError(
            ReasonCode.CHECKPOINT_INCOMPATIBLE,
            "validation_results schema 不兼容",
        )
    task_losses: dict[str, float] = {}
    for key, value in row["task_losses"].items():
        task = _artifact_string(
            key,
            label="validation.task_losses key",
        )
        loss = _artifact_number(
            value,
            label=f"validation.task_losses.{key}",
        )
        if loss is None:
            raise AssertionError("non-null artifact number returned None")
        task_losses[task] = loss
    task_counts: dict[str, int] = {}
    for key, value in row["task_counts"].items():
        task = _artifact_string(
            key,
            label="validation.task_counts key",
        )
        task_counts[task] = _artifact_int(
            value,
            label=f"validation.task_counts.{key}",
            minimum=1,
        )
    if (
        set(task_losses) != REQUIRED_EXTERNAL_TASKS
        or set(task_counts) != REQUIRED_EXTERNAL_TASKS
    ):
        raise ModelError(
            ReasonCode.CHECKPOINT_INCOMPATIBLE,
            "validation_results 未精确覆盖七类任务",
        )
    result = ValidationResult(
        step=_artifact_int(
            row["step"],
            label="validation.step",
            minimum=1,
        ),
        macro_task_loss=float(
            _artifact_number(
                row["macro_task_loss"],
                label="validation.macro_task_loss",
            )
        ),
        overall_loss=float(
            _artifact_number(
                row["overall_loss"],
                label="validation.overall_loss",
            )
        ),
        task_losses={
            key: float(value) for key, value in task_losses.items()
        },
        task_counts=task_counts,
        sample_count=_artifact_int(
            row["sample_count"],
            label="validation.sample_count",
            minimum=1,
        ),
        parent_count=_artifact_int(
            row["parent_count"],
            label="validation.parent_count",
            minimum=1,
        ),
        input_tokens=_artifact_int(
            row["input_tokens"],
            label="validation.input_tokens",
            minimum=1,
        ),
        images=_artifact_int(
            row["images"],
            label="validation.images",
            minimum=1,
        ),
        duration_seconds=float(
            _artifact_number(
                row["duration_seconds"],
                label="validation.duration_seconds",
                minimum=0.0,
            )
        ),
    )
    weighted = sum(
        result.task_losses[task] * result.task_counts[task]
        for task in REQUIRED_EXTERNAL_TASKS
    ) / sum(result.task_counts.values())
    macro = sum(result.task_losses.values()) / len(result.task_losses)
    if (
        result.sample_count != sum(result.task_counts.values())
        or result.parent_count != result.sample_count
        or not math.isclose(result.overall_loss, weighted, rel_tol=1e-12)
        or not math.isclose(
            result.macro_task_loss,
            macro,
            rel_tol=1e-12,
        )
    ):
        raise ModelError(
            ReasonCode.CHECKPOINT_INCOMPATIBLE,
            "validation_results 聚合值与逐任务值不一致",
        )
    return result


def _best_pointer(
    *,
    output_root: Path,
    result: ValidationResult,
    selection_metric: str = "macro_task_loss",
) -> dict[str, Any]:
    checkpoint = output_root / "checkpoints" / f"step-{result.step:08d}"
    if not checkpoint.is_dir() or checkpoint.is_symlink():
        raise ModelError(
            ReasonCode.CHECKPOINT_CORRUPT,
            "best checkpoint 对应目录不存在或是链接",
        )
    return {
        "schema_version": BEST_CHECKPOINT_SCHEMA_VERSION,
        "selection_metric": selection_metric,
        "step": result.step,
        "macro_task_loss": result.macro_task_loss,
        "overall_loss": result.overall_loss,
        "checkpoint": checkpoint.relative_to(output_root).as_posix(),
        "formal_acceptance": False,
    }


class DescriptionTrainer:
    def __init__(
        self,
        *,
        config: VLMConfig,
        model: TrainableAdapter,
        collator: DescriptionCollator,
        validation_dataset: DescriptionTrainingDataset,
        validation_collator: DescriptionCollator,
        validation_selection: ValidationSelection,
        benchmark_identity: BenchmarkIdentity,
        processor_identity: Mapping[str, Any],
        device: torch.device,
        progress: TrainingProgress | None = None,
        allowed_training_roles: frozenset[str] = frozenset({"external_train"}),
        sampler_factory: Callable[[DescriptionTrainingDataset], Any] | None = None,
        validation_evaluator: Callable[..., ValidationResult] = evaluate_teacher_forced_loss,
        validation_better: Callable[[ValidationResult, ValidationResult | None], bool] = validation_is_better,
        validation_selection_metric: str = "macro_task_loss",
        validation_row_parser: Callable[[Mapping[str, Any]], ValidationResult] = _validation_from_row,
        cuda_cache_cleanup_interval_steps: int | None = None,
        cuda_resource_telemetry: CudaMicrobatchTelemetry | None = None,
    ) -> None:
        self.config = config
        self.model = model
        self.collator = collator
        self.validation_dataset = validation_dataset
        self.validation_collator = validation_collator
        self.validation_selection = validation_selection
        self.benchmark_identity = benchmark_identity
        self.processor_identity = dict(processor_identity)
        self.device = device
        self.progress = progress
        self.allowed_training_roles = frozenset(allowed_training_roles)
        self.sampler_factory = sampler_factory
        self.validation_evaluator = validation_evaluator
        self.validation_better = validation_better
        self.validation_selection_metric = validation_selection_metric
        self.validation_row_parser = validation_row_parser
        self.cuda_cache_cleanup_interval_steps = (
            cuda_cache_cleanup_interval_steps
        )
        self.cuda_resource_telemetry = cuda_resource_telemetry
        if not self.allowed_training_roles or not validation_selection_metric:
            raise ModelError(
                ReasonCode.TYPE_MISMATCH,
                "trainer 角色集合和 validation selection metric 不能为空",
            )
        training_layout_identity(
            config,
            cuda_cache_cleanup_interval_steps=(
                self.cuda_cache_cleanup_interval_steps
            ),
            cuda_resource_identity=(
                None
                if self.cuda_resource_telemetry is None
                else self.cuda_resource_telemetry.layout_identity()
            ),
        )
        if model.trainable_parameter_count <= 0:
            raise ModelError(
                ReasonCode.MODEL_IDENTITY_MISMATCH,
                "prompt-only baseline 不进入 trainer；请直接 infer/evaluate",
            )
        if (
            validation_selection.seed != config.run.seed
            or validation_selection.max_parents
            != config.training.validation_max_parents
            or validation_selection.benchmark_build_id
            != benchmark_identity.build_id
            or validation_selection.benchmark_payload_sha256
            != benchmark_identity.payload_sha256
        ):
            raise ModelError(
                ReasonCode.VALIDATION_SELECTION_INVALID,
                "validation selection 与配置/Benchmark identity 不一致",
            )

    def fit(
        self,
        dataset: DescriptionTrainingDataset,
        *,
        stop_after_steps: int | None = None,
    ) -> TrainingResult:
        config = self.config
        output_root = config.run.output_root
        manager = CheckpointManager()
        linked = first_symlink_component(output_root)
        if linked is not None:
            raise ModelError(
                ReasonCode.OUTPUT_LINK,
                f"training output_root 含链接组件：{linked}",
            )
        actual_training_roles = {
            str(record.get("logical_role")) for record in dataset.records
        }
        if not actual_training_roles or actual_training_roles - self.allowed_training_roles:
            raise ModelError(
                ReasonCode.ROLE_FORBIDDEN,
                "训练 Dataset 含未授权 logical_role",
                details={
                    "allowed": sorted(self.allowed_training_roles),
                    "actual": sorted(actual_training_roles),
                },
            )
        sampler = (
            self.sampler_factory(dataset)
            if self.sampler_factory is not None
            else ParentBalancedSampler(
                dataset,
                seed=config.run.seed,
                source_weights=config.data.source_weights,
                task_weights=config.data.task_weights,
            )
        )

        def select_best_result(
            history: Sequence[ValidationResult],
        ) -> ValidationResult | None:
            best: ValidationResult | None = None
            for result in history:
                if self.validation_better(result, best):
                    best = result
            return best
        trainable_parameters = [
            parameter
            for parameter in self.model.model.parameters()
            if parameter.requires_grad
        ]
        optimizer = AdamW(
            trainable_parameters,
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
        scheduler = _scheduler(
            optimizer,
            max_steps=config.training.max_steps,
            warmup_ratio=config.training.warmup_ratio,
        )
        model_identity = self.model.identity.to_dict()
        benchmark_row = self.benchmark_identity.training_identity_dict()
        validation_identity = self.validation_selection.identity_dict()
        training_layout = training_layout_identity(
            config,
            cuda_cache_cleanup_interval_steps=(
                self.cuda_cache_cleanup_interval_steps
            ),
            cuda_resource_identity=(
                None
                if self.cuda_resource_telemetry is None
                else self.cuda_resource_telemetry.layout_identity()
            ),
        )
        worker_collators = tuple(
            self.collator.clone_for_worker()
            for _ in range(config.training.num_workers)
        )
        train_log_path = output_root / "train_log.jsonl"
        trace_path = output_root / "sample_trace.jsonl"
        validation_path = output_root / "validation_results.jsonl"
        selection_path = output_root / "validation_selection.json"
        best_path = output_root / "best_checkpoint.json"
        report_path = output_root / "training_report.json"
        log_rows: list[dict[str, Any]] = []
        trace_rows: list[dict[str, Any]] = []
        validation_history: list[ValidationResult] = []
        prior_elapsed = 0.0
        cumulative_input_tokens = 0
        cumulative_supervised_tokens = 0
        cumulative_images = 0
        ema_loss: float | None = None
        if output_root.exists() or output_root.is_symlink():
            raise ModelError(
                ReasonCode.OUTPUT_EXISTS,
                f"training output_root 必须是全新路径：{output_root}",
            )
        output_root.mkdir(parents=True)
        atomic_write_json(
            output_root / "config_snapshot.json",
            config.snapshot_dict(),
        )
        atomic_write_json(
            selection_path,
            self.validation_selection.to_dict(),
        )
        set_global_seed(config.run.seed)
        cursor = TrainingCursor(0, 0, 0, 0)
        target_steps = config.training.max_steps
        if stop_after_steps is not None:
            if (
                isinstance(stop_after_steps, bool)
                or not isinstance(stop_after_steps, int)
                or stop_after_steps <= 0
                or stop_after_steps > target_steps
            ):
                raise ModelError(
                    ReasonCode.TYPE_MISMATCH,
                    "stop_after_steps 必须位于 [1,max_steps]",
                )
            target_steps = stop_after_steps
        progress = self.progress or TrainingProgress(
            log_interval=config.training.log_interval
        )
        warmup_steps = int(
            math.ceil(config.training.max_steps * config.training.warmup_ratio)
        )
        gpu_name = (
            torch.cuda.get_device_name(self.device)
            if self.device.type == "cuda"
            else None
        )
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        progress.announce_setup(
            run_name=config.run.name,
            config_path=config.config_path,
            config_sha256=config.semantic_sha256,
            benchmark_build_id=self.benchmark_identity.build_id,
            benchmark_payload_sha256=self.benchmark_identity.payload_sha256,
            model_path=str(model_identity.get("model_path", "fixture")),
            processor_path=str(
                self.processor_identity.get("processor_path", "fixture")
            ),
            device=str(self.device),
            gpu_name=gpu_name,
            base_parameters=int(
                model_identity.get(
                    "base_parameter_count",
                    sum(
                        parameter.numel()
                        for parameter in self.model.model.parameters()
                    ),
                )
            ),
            trainable_parameters=self.model.trainable_parameter_count,
            train_samples=len(dataset),
            validation_samples=len(self.validation_selection.items),
            batch_size=config.training.batch_size,
            accumulation_steps=config.training.gradient_accumulation_steps,
            input_pipeline_backend=str(
                training_layout["input_pipeline_backend"]
            ),
            num_workers=config.training.num_workers,
            prefetch_factor=config.training.prefetch_factor,
            pin_memory=config.training.pin_memory,
            max_images=config.limits.max_images,
            max_input_tokens=config.limits.max_input_tokens,
            min_pixels=config.limits.min_pixels,
            max_pixels=config.limits.max_pixels,
            learning_rate=config.training.learning_rate,
            warmup_steps=warmup_steps,
            max_steps=config.training.max_steps,
            start_step=cursor.global_step,
            stop_step=target_steps,
            validation_interval=config.training.validation_interval,
            checkpoint_interval=config.training.checkpoint_interval,
            output_root=output_root,
        )
        self.model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch = cursor.epoch
        sample_offset = cursor.sample_offset
        global_step = cursor.global_step
        micro_step = cursor.micro_step
        accumulated_loss = 0.0
        loss_history: list[float] = []
        sample_trace: list[str] = []
        last_checkpoint: Path | None = None
        session_started = time.perf_counter()
        session_start_step = global_step
        session_start_samples = len(trace_rows)
        session_start_tokens = cumulative_input_tokens
        progress.start_training(
            start_step=global_step,
            stop_step=target_steps,
        )
        if self.cuda_resource_telemetry is not None:
            self.cuda_resource_telemetry.start(
                output_root,
                completed_microbatches=len(trace_rows),
            )
        input_pipeline: OrderedBatchPrefetcher | None = None
        try:
            remaining_micro_batches = (
                (target_steps - global_step)
                * config.training.gradient_accumulation_steps
                - micro_step
            )
            planner = DeterministicBatchPlanner(
                dataset=dataset,
                sampler=sampler,
                batch_size=config.training.batch_size,
                max_epochs=config.training.epochs,
                start_epoch=epoch,
                start_sample_offset=sample_offset,
                max_batches=remaining_micro_batches,
            )
            input_pipeline = OrderedBatchPrefetcher(
                planner=planner,
                collator=self.collator,
                num_workers=config.training.num_workers,
                prefetch_factor=config.training.prefetch_factor,
                pin_memory=(
                    config.training.pin_memory
                    and self.device.type == "cuda"
                ),
                worker_collators=worker_collators,
            )
            optimizer_step_started = time.perf_counter()
            data_wait_seconds = 0.0
            while global_step < target_steps:
                wait_started = time.perf_counter()
                try:
                    prepared = next(input_pipeline)
                except StopIteration as error:
                    raise ModelError(
                        ReasonCode.CHECKPOINT_CORRUPT,
                        "输入 pipeline 在目标 step 前意外耗尽",
                    ) from error
                data_wait_seconds += time.perf_counter() - wait_started
                samples = prepared.plan.samples
                sample_epochs = prepared.plan.sample_epochs
                epoch = prepared.plan.next_epoch
                sample_offset = prepared.plan.next_sample_offset
                resource_metadata = None
                if self.cuda_resource_telemetry is not None:
                    cpu_labels = prepared.batch.get("labels")
                    cpu_pixels = prepared.batch.get("pixel_values")
                    cpu_grid = prepared.batch.get("image_grid_thw")
                    if not all(
                        isinstance(value, Tensor)
                        for value in (cpu_labels, cpu_pixels, cpu_grid)
                    ):
                        raise ModelError(
                            ReasonCode.TYPE_MISMATCH,
                            "CUDA telemetry batch 缺少 labels/pixel/grid",
                        )
                    assert isinstance(cpu_labels, Tensor)
                    assert isinstance(cpu_pixels, Tensor)
                    assert isinstance(cpu_grid, Tensor)
                    resource_metadata = {
                        "optimizer_step": global_step + 1,
                        "micro_step": micro_step + 1,
                        "epoch": sample_epochs[0],
                        "record_id": samples[0].record_id,
                        "parent_id": samples[0].parent_id,
                        "logical_role": samples[0].logical_role,
                        "task_family": samples[0].task_family,
                        "input_tokens": int(
                            prepared.batch["input_token_counts"][0]
                        ),
                        "supervised_tokens": int(
                            cpu_labels[:, 1:].ne(-100).sum()
                        ),
                        "image_count": int(
                            prepared.batch["image_counts"][0]
                        ),
                        "pixel_shape": list(cpu_pixels.shape),
                        "pixel_dtype": str(cpu_pixels.dtype),
                        "pixel_numel": int(cpu_pixels.numel()),
                        "image_grid_thw": cpu_grid.tolist(),
                    }
                batch = _move_batch(prepared.batch, self.device)
                if self.cuda_resource_telemetry is not None:
                    assert resource_metadata is not None
                    self.cuda_resource_telemetry.begin(resource_metadata)
                result = self.model.forward(batch)
                if self.cuda_resource_telemetry is not None:
                    self.cuda_resource_telemetry.after_forward()
                loss = getattr(result, "loss", None)
                if (
                    not isinstance(loss, Tensor)
                    or loss.ndim != 0
                    or not bool(torch.isfinite(loss))
                ):
                    raise ModelError(
                        ReasonCode.NONFINITE_NUMBER,
                        "model adapter forward 必须返回有限标量 loss",
                    )
                scaled = loss / config.training.gradient_accumulation_steps
                scaled.backward()
                if self.cuda_resource_telemetry is not None:
                    self.cuda_resource_telemetry.after_backward()
                accumulated_loss += float(loss.detach().cpu())
                sample_trace.extend(sample.record_id for sample in samples)
                cumulative_input_tokens += sum(batch["input_token_counts"])
                cumulative_images += sum(batch["image_counts"])
                labels = batch.get("labels")
                if not isinstance(labels, Tensor):
                    raise ModelError(
                        ReasonCode.LOSS_MASK_INVALID,
                        "训练 batch 缺少 labels",
                    )
                cumulative_supervised_tokens += int((labels != -100).sum())
                if cumulative_input_tokens > config.limits.max_total_tokens:
                    raise ModelError(
                        ReasonCode.TOKEN_LIMIT_EXCEEDED,
                        "累计 input tokens 超过配置上限",
                    )
                for batch_slot, (sample, sample_epoch) in enumerate(
                    zip(samples, sample_epochs, strict=True)
                ):
                    trace_rows.append(
                        {
                            "schema_version": SAMPLE_TRACE_SCHEMA_VERSION,
                            "sequence_index": len(trace_rows),
                            "optimizer_step": global_step + 1,
                            "micro_step": micro_step + 1,
                            "batch_slot": batch_slot,
                            "epoch": sample_epoch,
                            "record_id": sample.record_id,
                            "parent_id": sample.parent_id,
                            "task_family": sample.task_family,
                        }
                    )
                micro_step += 1
                # backward 已完成后立即释放 logits/batch 等大对象，避免下一次 forward
                # 求值期间仍持有上一条可变形状样本的显存。
                del labels, scaled, loss, result, batch, prepared
                if self.cuda_resource_telemetry is not None:
                    self.cuda_resource_telemetry.complete_after_release()
                if micro_step < config.training.gradient_accumulation_steps:
                    continue
                gradient_norm_tensor = torch.nn.utils.clip_grad_norm_(
                    trainable_parameters,
                    config.training.max_grad_norm,
                )
                gradient_norm = float(gradient_norm_tensor.detach().cpu())
                if not math.isfinite(gradient_norm):
                    raise ModelError(
                        ReasonCode.NONFINITE_NUMBER,
                        "gradient norm 非有限",
                    )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                micro_step = 0
                del gradient_norm_tensor
                cleanup_interval = self.cuda_cache_cleanup_interval_steps
                if (
                    cleanup_interval is not None
                    and global_step % cleanup_interval == 0
                ):
                    clear_cuda_cache(self.device)
                if self.cuda_resource_telemetry is not None:
                    self.cuda_resource_telemetry.record_post_zero_grad(
                        optimizer_step=global_step
                    )
                step_loss = (
                    accumulated_loss
                    / config.training.gradient_accumulation_steps
                )
                accumulated_loss = 0.0
                loss_history.append(step_loss)
                ema_loss = (
                    step_loss
                    if ema_loss is None
                    else 0.9 * ema_loss + 0.1 * step_loss
                )
                cursor = TrainingCursor(
                    epoch=epoch,
                    sample_offset=sample_offset,
                    global_step=global_step,
                    micro_step=0,
                )
                now = time.perf_counter()
                step_duration = now - optimizer_step_started
                session_elapsed = max(now - session_started, 1e-12)
                run_elapsed = prior_elapsed + session_elapsed
                session_samples = len(trace_rows) - session_start_samples
                session_tokens = (
                    cumulative_input_tokens - session_start_tokens
                )
                samples_per_second = session_samples / session_elapsed
                tokens_per_second = session_tokens / session_elapsed
                completed_session_steps = global_step - session_start_step
                eta = (
                    None
                    if completed_session_steps <= 0
                    else (
                        (target_steps - global_step)
                        * session_elapsed
                        / completed_session_steps
                    )
                )
                memory = _cuda_memory(self.device)
                telemetry = _train_log_row(
                    step=global_step,
                    loss=step_loss,
                    ema_loss=ema_loss,
                    learning_rate=float(optimizer.param_groups[0]["lr"]),
                    gradient_norm=gradient_norm,
                    gradient_clipped=(
                        gradient_norm > config.training.max_grad_norm
                    ),
                    samples=len(trace_rows),
                    input_tokens=cumulative_input_tokens,
                    supervised_tokens=cumulative_supervised_tokens,
                    images=cumulative_images,
                    samples_per_second=samples_per_second,
                    tokens_per_second=tokens_per_second,
                    step_duration_seconds=step_duration,
                    data_wait_seconds=data_wait_seconds,
                    data_wait_fraction=min(
                        data_wait_seconds / max(step_duration, 1e-12),
                        1.0,
                    ),
                    session_elapsed_seconds=session_elapsed,
                    run_elapsed_seconds=run_elapsed,
                    eta_seconds=eta,
                    memory=memory,
                )
                progress.update_training(telemetry)
                validation_due = (
                    global_step % config.training.validation_interval == 0
                )
                checkpoint_due = (
                    global_step % config.training.checkpoint_interval == 0
                    or validation_due
                    or global_step == target_steps
                )
                log_due = (
                    global_step == 1
                    or global_step % config.training.log_interval == 0
                    or checkpoint_due
                )
                if log_due:
                    log_rows.append(telemetry)
                    atomic_write_jsonl(train_log_path, log_rows)
                if checkpoint_due:
                    checkpoint = (
                        output_root
                        / "checkpoints"
                        / f"step-{global_step:08d}"
                    )
                    if checkpoint.exists() or checkpoint.is_symlink():
                        raise ModelError(
                            ReasonCode.OUTPUT_EXISTS,
                            f"checkpoint 已存在，拒绝覆盖：{checkpoint}",
                        )
                    checkpoint_started = time.perf_counter()
                    atomic_write_jsonl(trace_path, trace_rows)
                    last_checkpoint = manager.save(
                        checkpoint,
                        trainable_state=self.model.trainable_state_dict(),
                        cursor=cursor,
                        config_semantic_sha256=config.semantic_sha256,
                        benchmark_identity=benchmark_row,
                        validation_selection_identity=validation_identity,
                        model_identity=model_identity,
                        processor_identity=self.processor_identity,
                        training_layout=training_layout,
                        trainable_names=self.model.trainable_names,
                        selection_metrics={
                            "training_loss": step_loss,
                            "ema_loss": ema_loss,
                        },
                    )
                    progress.checkpoint(
                        step=global_step,
                        path=last_checkpoint,
                        seconds=time.perf_counter() - checkpoint_started,
                    )
                if validation_due:
                    validation = self.validation_evaluator(
                        model=self.model,
                        collator=self.validation_collator,
                        dataset=self.validation_dataset,
                        selection=self.validation_selection,
                        device=self.device,
                        step=global_step,
                        progress=progress,
                    )
                    validation_history.append(validation)
                    atomic_write_jsonl(
                        validation_path,
                        (
                            item.to_dict()
                            for item in validation_history
                        ),
                    )
                    incumbent = select_best_result(validation_history[:-1])
                    if self.validation_better(validation, incumbent):
                        atomic_write_json(
                            best_path,
                            _best_pointer(
                                output_root=output_root,
                                result=validation,
                                selection_metric=self.validation_selection_metric,
                            ),
                        )
                if log_due and (checkpoint_due or validation_due):
                    post_event_elapsed = max(
                        time.perf_counter() - session_started,
                        1e-12,
                    )
                    updated = dict(log_rows[-1])
                    updated["session_elapsed_seconds"] = post_event_elapsed
                    updated["run_elapsed_seconds"] = (
                        prior_elapsed + post_event_elapsed
                    )
                    updated["samples_per_second"] = (
                        session_samples / post_event_elapsed
                    )
                    updated["tokens_per_second"] = (
                        session_tokens / post_event_elapsed
                    )
                    updated.update(_cuda_memory(self.device))
                    log_rows[-1] = updated
                    atomic_write_jsonl(train_log_path, log_rows)
                optimizer_step_started = time.perf_counter()
                data_wait_seconds = 0.0
            if last_checkpoint is None:
                raise ModelError(
                    ReasonCode.CHECKPOINT_CORRUPT,
                    "训练结束但未生成 checkpoint",
                )
            atomic_write_jsonl(trace_path, trace_rows)
            if not log_rows or log_rows[-1]["step"] != global_step:
                raise ModelError(
                    ReasonCode.CHECKPOINT_CORRUPT,
                    "训练结束但 train_log 未落到最后 checkpoint",
                )
            final_elapsed = prior_elapsed + (
                time.perf_counter() - session_started
            )
            memory = _cuda_memory(self.device)
            best_result = select_best_result(validation_history)
            best_checkpoint = (
                None
                if best_result is None
                else (
                    output_root
                    / "checkpoints"
                    / f"step-{best_result.step:08d}"
                )
            )
            status = (
                "completed"
                if global_step == config.training.max_steps
                else "bounded_complete"
            )
            report = {
                "schema_version": TRAINING_REPORT_SCHEMA_VERSION,
                "status": status,
                "formal_acceptance": False,
                "config_semantic_sha256": config.semantic_sha256,
                "benchmark": benchmark_row,
                "validation_selection": validation_identity,
                "training_layout": training_layout,
                "cursor": cursor.to_dict(),
                "max_steps": config.training.max_steps,
                "execution_stop_step": target_steps,
                "last_loss": loss_history[-1],
                "ema_loss": ema_loss,
                "session_loss_history": loss_history,
                "samples": len(trace_rows),
                "input_tokens": cumulative_input_tokens,
                "supervised_tokens": cumulative_supervised_tokens,
                "images": cumulative_images,
                "run_elapsed_seconds": final_elapsed,
                "cuda_peak_gib": memory["cuda_peak_gib"],
                "throughput": {
                    "samples_per_second": float(
                        log_rows[-1]["samples_per_second"]
                    ),
                    "tokens_per_second": float(
                        log_rows[-1]["tokens_per_second"]
                    ),
                    "step_duration_seconds": float(
                        log_rows[-1]["step_duration_seconds"]
                    ),
                    "data_wait_seconds": float(
                        log_rows[-1]["data_wait_seconds"]
                    ),
                    "data_wait_fraction": float(
                        log_rows[-1]["data_wait_fraction"]
                    ),
                },
                "last_gradient_norm": float(
                    log_rows[-1]["gradient_norm"]
                ),
                "checkpoint": str(last_checkpoint.resolve()),
                "best_checkpoint": (
                    None
                    if best_checkpoint is None
                    else str(best_checkpoint.resolve())
                ),
                "validation_history": [
                    item.to_dict() for item in validation_history
                ],
                "artifacts": {
                    "config_snapshot": "config_snapshot.json",
                    "train_log": "train_log.jsonl",
                    "sample_trace": "sample_trace.jsonl",
                    "validation_selection": "validation_selection.json",
                    "validation_results": (
                        "validation_results.jsonl"
                        if validation_history
                        else None
                    ),
                    "best_checkpoint": (
                        "best_checkpoint.json"
                        if best_checkpoint is not None
                        else None
                    ),
                },
            }
            if self.cuda_resource_telemetry is not None:
                report["artifacts"].update(
                    {
                        "cuda_resource_identity": "cuda_resource_identity.json",
                        "cuda_resource_telemetry": "cuda_resource_telemetry.jsonl",
                        "cuda_step_telemetry": "cuda_step_telemetry.jsonl",
                        "cuda_active_microbatch": "cuda_active_microbatch.json",
                    }
                )
            atomic_write_json(report_path, report)
            atomic_write_json(
                output_root / "manifest.json",
                {
                    "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
                    "run_kind": "training",
                    "config_semantic_sha256": config.semantic_sha256,
                    "benchmark": benchmark_row,
                    "validation_selection": validation_identity,
                    "model": model_identity,
                    "processor": self.processor_identity,
                    "adaptation": config.adaptation.strategy,
                    "training_layout": training_layout,
                    "trainable_parameter_count": (
                        self.model.trainable_parameter_count
                    ),
                    "cursor": cursor.to_dict(),
                    "checkpoint": str(last_checkpoint.resolve()),
                    "best_checkpoint": (
                        None
                        if best_checkpoint is None
                        else str(best_checkpoint.resolve())
                    ),
                    "training_report": str(report_path.resolve()),
                    "formal_acceptance": False,
                },
            )
            return TrainingResult(
                cursor=cursor,
                checkpoint=last_checkpoint,
                best_checkpoint=best_checkpoint,
                status=status,
                max_steps=config.training.max_steps,
                last_loss=loss_history[-1],
                run_elapsed_seconds=final_elapsed,
                cuda_peak_gib=memory["cuda_peak_gib"],
                training_report=report_path,
                loss_history=tuple(loss_history),
                sample_trace=tuple(sample_trace),
            )
        except BaseException as error:
            if self.cuda_resource_telemetry is not None:
                self.cuda_resource_telemetry.persist_failure(
                    error,
                    last_completed_optimizer_step=global_step,
                    last_completed_microbatches=(
                        self.cuda_resource_telemetry.completed_microbatches
                    ),
                )
            raise
        finally:
            if input_pipeline is not None:
                input_pipeline.close()
            progress.close()
