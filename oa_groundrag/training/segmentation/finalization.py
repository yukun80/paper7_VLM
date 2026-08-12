"""OA-AuxSeg checkpoint 的只读定版与证据复核。"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from oa_groundrag.data.oa_auxseg.dataset import (
    atomic_write_json,
    read_jsonl,
    sha256_file,
)
from oa_groundrag.evaluation.segmentation import evaluate_model
from oa_groundrag.segmentation.checkpoint import (
    model_from_checkpoint,
    read_checkpoint,
    save_training_checkpoint,
)
from oa_groundrag.segmentation.config import resolve_runtime
from oa_groundrag.segmentation.contracts import (
    CONFIG_SCHEMA_VERSION,
    RuntimeConfig,
    TRAINING_REPORT_SCHEMA_VERSION,
)
from oa_groundrag.segmentation.data import benchmark_contract_from_root, make_dataloader
from oa_groundrag.segmentation.policy import (
    validate_benchmark_registry as _validate_benchmark_registry,
    validate_inference_config as _validate_inference_config,
)
from .engine import (
    _acceptance_collated,
    _validation_is_better,
    checkpoint_reload_difference,
    make_optimizer,
    set_global_seed,
)
from .progress import TrainingProgress

def _finalization_candidate(
    row: Mapping[str, Any],
    *,
    location: str,
) -> dict[str, float | int]:
    validation = row.get("validation")
    if not isinstance(validation, Mapping):
        raise ValueError(f"{location}.validation 必须是对象")
    overall = validation.get("overall")
    if not isinstance(overall, Mapping):
        raise ValueError(f"{location}.validation.overall 必须是对象")
    step = row.get("step")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError(f"{location}.step 必须是非负整数")
    try:
        return {
            "step": step,
            "dice": float(overall["dice"]),
            "loss": float(validation["loss"]),
            "no_target_false_positive_rate": float(
                overall["no_target_false_positive_rate"]
            ),
        }
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{location} 缺少合法的 best selection 指标") from error
def _same_selection(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
) -> bool:
    required = (
        "step",
        "dice",
        "loss",
        "no_target_false_positive_rate",
    )
    try:
        if int(first["step"]) != int(second["step"]):
            return False
        return all(
            math.isclose(
                float(first[name]),
                float(second[name]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for name in required[1:]
        )
    except (KeyError, TypeError, ValueError):
        return False
def _read_finalization_log(path: Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"训练日志不存在：{path}")
    rows = [dict(row) for row in read_jsonl(path)]
    if not rows:
        raise ValueError("训练日志不能为空")
    previous_step = -1
    for index, row in enumerate(rows):
        step = row.get("step")
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError(f"train_log[{index}].step 必须是非负整数")
        if step < previous_step:
            raise ValueError(
                "训练日志 step 必须单调不下降："
                f"train_log[{index - 1}]={previous_step}, "
                f"train_log[{index}]={step}"
            )
        previous_step = step
    return rows
def _checkpoint_training_diagnostics(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    step = int(payload["step"])
    state = payload["training_state"]
    if not isinstance(state, Mapping):
        raise ValueError("checkpoint training_state 必须是对象")
    loss_history = state.get("loss_history")
    if not isinstance(loss_history, list) or not loss_history:
        raise ValueError("checkpoint loss_history 必须是非空数组")
    if len(loss_history) != step:
        raise ValueError(
            "checkpoint loss_history 长度与 step 不一致："
            f"{len(loss_history)} != {step}"
        )
    losses = [float(value) for value in loss_history]
    window = min(10, len(losses))
    initial_window = sum(losses[:window]) / window
    final_window = sum(losses[-window:]) / window
    training_seconds = float(state.get("training_step_seconds", 0.0))
    timed_steps = int(state.get("training_steps_timed", 0))
    samples_seen = int(state.get("training_samples_seen", 0))
    clipped_steps = int(state.get("clipped_steps", 0))
    return {
        "step": step,
        "loss_history_length": len(losses),
        "initial_loss_window_mean": initial_window,
        "final_loss_window_mean": final_window,
        "loss_drop_fraction": (
            1.0 - final_window / max(initial_window, 1e-12)
        ),
        "initial_ema_loss": state.get("initial_ema_loss"),
        "final_ema_loss": state.get("ema_loss"),
        "gradient_max_norms": dict(state.get("gradient_max", {})),
        "parameter_max_changes": dict(
            state.get("parameter_change_max", {})
        ),
        "training_step_seconds": training_seconds,
        "training_steps_timed": timed_steps,
        "training_samples_seen": samples_seen,
        "average_steps_per_second": (
            timed_steps / training_seconds if training_seconds > 0 else None
        ),
        "average_samples_per_second": (
            samples_seen / training_seconds if training_seconds > 0 else None
        ),
        "gradient_clipped_steps": clipped_steps,
        "gradient_clipped_fraction": (
            clipped_steps / max(timed_steps, 1)
        ),
        "gradient_clip_scale_mean": (
            float(state.get("clip_scale_sum", 0.0))
            / max(timed_steps, 1)
        ),
        "gradient_clip_scale_min": float(
            state.get("clip_scale_min", 1.0)
        ),
        "training_subset_counts": dict(
            state.get("training_subset_counts", {})
        ),
        "training_subset_reason_counts": dict(
            state.get("training_subset_reason_counts", {})
        ),
        "training_native_auxiliary_samples": int(
            state.get("training_native_auxiliary_samples", 0)
        ),
        "training_active_auxiliary_samples": int(
            state.get("training_active_auxiliary_samples", 0)
        ),
        "validation_count": len(state.get("validation_trajectory", [])),
    }
def _validate_finalization_evidence(
    *,
    config: RuntimeConfig,
    rows: Sequence[Mapping[str, Any]],
    selected_payload: Mapping[str, Any],
    last_payload: Mapping[str, Any],
) -> dict[str, Any]:
    if not rows:
        raise ValueError("训练日志不能为空")
    selected_step = int(selected_payload["step"])
    last_checkpoint_step = int(last_payload["step"])
    last_logged_step = int(rows[-1]["step"])
    if not (
        0 <= selected_step <= last_checkpoint_step <= last_logged_step
        <= config.max_steps
    ):
        raise ValueError(
            "训练步数关系不合法："
            f"selected={selected_step}, last_checkpoint={last_checkpoint_step}, "
            f"last_logged={last_logged_step}, max={config.max_steps}"
        )

    selected_state = selected_payload["training_state"]
    last_state = last_payload["training_state"]
    selected_config = RuntimeConfig.from_dict(
        selected_state["runtime_config"]
    )
    last_config = RuntimeConfig.from_dict(last_state["runtime_config"])
    if selected_config.to_dict() != config.to_dict():
        raise ValueError("final checkpoint runtime config 与当前配置不一致")
    if last_config.to_dict() != config.to_dict():
        raise ValueError("last checkpoint runtime config 与当前配置不一致")
    if selected_payload["model_contract"] != last_payload["model_contract"]:
        raise ValueError("best/last checkpoint model contract 不一致")

    selected_best = selected_state.get("best_selection")
    last_best = last_state.get("best_selection")
    if not isinstance(selected_best, Mapping):
        raise ValueError("final checkpoint 缺少 best_selection")
    if not isinstance(last_best, Mapping):
        raise ValueError("last checkpoint 缺少 best_selection")
    if int(selected_best.get("step", -1)) != selected_step:
        raise ValueError("final checkpoint step 与 best_selection.step 不一致")
    if not _same_selection(selected_best, last_best):
        raise ValueError("best/last checkpoint 的 best_selection 不一致")

    computed_best: dict[str, float | int] | None = None
    selected_rows: list[Mapping[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if "validation" not in row:
            continue
        candidate = _finalization_candidate(
            row, location=f"train_log[{index}]"
        )
        expected_is_new_best = _validation_is_better(
            candidate, computed_best
        )
        if row.get("is_new_best") is not expected_is_new_best:
            raise ValueError(
                f"train_log[{index}].is_new_best 与既有选择规则不一致"
            )
        if expected_is_new_best:
            computed_best = candidate
        if int(candidate["step"]) == selected_step:
            selected_rows.append(row)
        validation_rows.append(dict(row))
    if computed_best is None:
        raise ValueError("训练日志中没有 validation 记录")
    if not _same_selection(computed_best, selected_best):
        raise ValueError("训练日志计算得到的 best selection 与 checkpoint 不一致")
    if len(selected_rows) != 1:
        raise ValueError(
            "训练日志中 final checkpoint step 必须且只能有一条 validation："
            f"step={selected_step}, count={len(selected_rows)}"
        )
    selected_row = selected_rows[0]
    if selected_row.get("is_new_best") is not True:
        raise ValueError("final checkpoint 对应日志没有 is_new_best=true")
    log_selection = _finalization_candidate(
        selected_row, location=f"train_log@step{selected_step}"
    )
    if not _same_selection(log_selection, selected_best):
        raise ValueError("final checkpoint 与同一步日志指标不一致")

    selected_trajectory = selected_state.get("validation_trajectory")
    if not isinstance(selected_trajectory, list):
        raise ValueError("final checkpoint validation_trajectory 必须是数组")
    trajectory_matches = [
        item
        for item in selected_trajectory
        if isinstance(item, Mapping)
        and int(item.get("step", -1)) == selected_step
    ]
    if len(trajectory_matches) != 1:
        raise ValueError("final checkpoint trajectory 缺少唯一的 best step")
    trajectory_item = trajectory_matches[0]
    if trajectory_item.get("is_new_best") is not True:
        raise ValueError("final checkpoint trajectory 的 best step 标记错误")
    trajectory_candidate = _finalization_candidate(
        trajectory_item,
        location=f"checkpoint.validation_trajectory@step{selected_step}",
    )
    if not _same_selection(trajectory_candidate, selected_best):
        raise ValueError("checkpoint trajectory 与 best_selection 指标不一致")
    if trajectory_item.get("validation") != selected_row.get("validation"):
        raise ValueError("checkpoint trajectory 与 train log validation 不一致")

    return {
        "selected_checkpoint_step": selected_step,
        "last_checkpoint_step": last_checkpoint_step,
        "last_logged_step": last_logged_step,
        "best_selection": dict(selected_best),
        "selection_validation_snapshot": dict(selected_row["validation"]),
        "validation_trajectory": validation_rows,
        "validation_count": len(validation_rows),
        "logged_but_not_checkpointed": {
            "after_step": last_checkpoint_step,
            "through_step": last_logged_step,
            "step_count": last_logged_step - last_checkpoint_step,
            "included_in_final_weights": False,
        },
    }
def _metric_replay_delta(
    stored: Any,
    fresh: Any,
    *,
    location: str = "$",
) -> Any:
    if isinstance(stored, Mapping) and isinstance(fresh, Mapping):
        if set(stored) != set(fresh):
            raise ValueError(
                f"{location} fresh/stored metric keys 不一致："
                f"stored={sorted(stored)} fresh={sorted(fresh)}"
            )
        return {
            str(key): _metric_replay_delta(
                stored[key],
                fresh[key],
                location=f"{location}.{key}",
            )
            for key in stored
        }
    if (
        isinstance(stored, (int, float))
        and not isinstance(stored, bool)
        and isinstance(fresh, (int, float))
        and not isinstance(fresh, bool)
    ):
        return float(fresh) - float(stored)
    if stored != fresh:
        raise ValueError(
            f"{location} fresh/stored metric structure 不一致："
            f"{stored!r} != {fresh!r}"
        )
    return None
def finalize_training_run(
    config: RuntimeConfig,
    *,
    repo_root: Path,
    checkpoint_path: Path,
    termination_reason: str,
    progress: TrainingProgress | None = None,
) -> dict[str, Any]:
    """只读冻结人工终止训练，并用 final best 生成 train/val 工程报告。"""

    terminal = (
        progress
        if progress is not None
        else TrainingProgress(log_interval=config.log_interval)
    )
    try:
        return _finalize_training_run_impl(
            config,
            repo_root=repo_root,
            checkpoint_path=checkpoint_path,
            termination_reason=termination_reason,
            progress=terminal,
        )
    finally:
        terminal.close()
def _asset_stat(path: Path) -> dict[str, int]:
    value = path.stat()
    return {
        "size_bytes": int(value.st_size),
        "mtime_ns": int(value.st_mtime_ns),
    }
def _finalize_training_run_impl(
    config: RuntimeConfig,
    *,
    repo_root: Path,
    checkpoint_path: Path,
    termination_reason: str,
    progress: TrainingProgress,
) -> dict[str, Any]:
    if termination_reason != "project_owner_manual_stop":
        raise ValueError(
            "finalize 仅接受 "
            "termination_reason=project_owner_manual_stop"
        )
    wall_started = time.perf_counter()
    benchmark_root, output_dir, _, device = resolve_runtime(config, repo_root)
    output_dir = output_dir.resolve()
    selected_checkpoint_path = Path(checkpoint_path).resolve()
    expected_best_path = (output_dir / "checkpoint_best.pt").resolve()
    last_checkpoint_path = (output_dir / "checkpoint_last.pt").resolve()
    train_log_path = (output_dir / "train_log.jsonl").resolve()
    training_report_path = (output_dir / "training_report.json").resolve()
    if selected_checkpoint_path != expected_best_path:
        raise ValueError(
            "finalize 只允许冻结配置输出根中的 checkpoint_best.pt："
            f"{selected_checkpoint_path} != {expected_best_path}"
        )
    for label, path in (
        ("checkpoint_best.pt", selected_checkpoint_path),
        ("checkpoint_last.pt", last_checkpoint_path),
        ("train_log.jsonl", train_log_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} 不存在：{path}")
    if training_report_path.exists():
        raise FileExistsError(
            f"训练报告已存在，拒绝覆盖：{training_report_path}"
        )

    input_paths = {
        "checkpoint_best": selected_checkpoint_path,
        "checkpoint_last": last_checkpoint_path,
        "train_log": train_log_path,
    }
    input_stats_before = {
        name: _asset_stat(path) for name, path in input_paths.items()
    }
    progress.phase(
        "[finalize] loading checkpoint/log evidence without optimizer state "
        f"output={output_dir}"
    )
    benchmark_contract = benchmark_contract_from_root(benchmark_root)
    rows = _read_finalization_log(train_log_path)
    selected_payload = read_checkpoint(
        selected_checkpoint_path,
        expected_benchmark_contract=benchmark_contract,
    )
    last_payload = read_checkpoint(
        last_checkpoint_path,
        expected_benchmark_contract=benchmark_contract,
    )
    evidence = _validate_finalization_evidence(
        config=config,
        rows=rows,
        selected_payload=selected_payload,
        last_payload=last_payload,
    )
    selected_checkpoint_sha256 = sha256_file(selected_checkpoint_path)
    train_log_sha256 = sha256_file(train_log_path)

    model = model_from_checkpoint(selected_payload, device=device)
    _validate_inference_config(selected_payload, model, config)
    _validate_benchmark_registry(model, benchmark_root)
    train_loader = make_dataloader(
        benchmark_root,
        split="train",
        batch_size=config.batch_size,
        normalization=config.normalization,
        shuffle=False,
        num_workers=config.num_workers,
    )
    validation_loader = make_dataloader(
        benchmark_root,
        split="val",
        batch_size=config.batch_size,
        normalization=config.normalization,
        shuffle=False,
        num_workers=config.num_workers,
    )
    set_global_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    progress.phase(
        "[finalize] evaluating final checkpoint on train/val only; "
        "test remains sealed"
    )
    train_metrics = evaluate_model(
        model,
        train_loader,
        device=device,
        config=config,
        progress=progress,
        progress_label="finalize-train",
    )
    validation_metrics = evaluate_model(
        model,
        validation_loader,
        device=device,
        config=config,
        progress=progress,
        progress_label="finalize-val",
    )
    selection_snapshot = evidence["selection_validation_snapshot"]
    validation_replay_delta = _metric_replay_delta(
        selection_snapshot,
        validation_metrics,
        location="$.validation_metrics",
    )

    progress.phase(
        "[finalize] checking strict checkpoint reload output consistency"
    )
    reload_started = time.perf_counter()
    reload_difference = checkpoint_reload_difference(
        checkpoint_path=selected_checkpoint_path,
        model=model,
        collated=_acceptance_collated(
            benchmark_root,
            normalization=config.normalization,
            batch_size=config.batch_size,
        ),
        benchmark_contract=benchmark_contract,
        device=device,
    )
    reload_seconds = time.perf_counter() - reload_started
    input_stats_after = {
        name: _asset_stat(path) for name, path in input_paths.items()
    }
    if input_stats_after != input_stats_before:
        raise RuntimeError(
            "finalize 期间 checkpoint 或 train log 的 size/mtime 发生变化，"
            "拒绝生成报告"
        )

    selected_step = int(evidence["selected_checkpoint_step"])
    last_checkpoint_step = int(evidence["last_checkpoint_step"])
    last_logged_step = int(evidence["last_logged_step"])
    train_sample_count = int(train_metrics["overall"]["sample_count"])
    validation_sample_count = int(
        validation_metrics["overall"]["sample_count"]
    )
    peak_memory = (
        int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else None
    )
    engineering_checks = {
        "config_checkpoint_benchmark_identity_consistent": True,
        "best_last_log_selection_evidence_consistent": True,
        "train_sample_count_matches_loader": (
            train_sample_count == len(train_loader.dataset)
        ),
        "validation_sample_count_matches_loader": (
            validation_sample_count == len(validation_loader.dataset)
        ),
        "checkpoint_reload_within_1e_6": (
            max(reload_difference.values()) <= 1e-6
        ),
        "input_checkpoints_and_log_unchanged": True,
        "test_split_not_evaluated": True,
    }
    engineering_checks_passed = all(engineering_checks.values())
    selected_diagnostics = _checkpoint_training_diagnostics(
        selected_payload
    )
    last_diagnostics = _checkpoint_training_diagnostics(last_payload)
    report: dict[str, Any] = {
        "schema_version": TRAINING_REPORT_SCHEMA_VERSION,
        "runtime_config_schema_version": CONFIG_SCHEMA_VERSION,
        "command": "finalize",
        "status": "completed",
        "completion_mode": termination_reason,
        "training_declared_complete": True,
        "automatic_early_stopping": False,
        "scheduled_max_steps_reached": (
            last_logged_step >= config.max_steps
        ),
        "stopped_early": last_logged_step < config.max_steps,
        "resume_required": False,
        "configured_max_steps": config.max_steps,
        "maximum_steps": config.max_steps,
        "steps": last_logged_step,
        "last_logged_step": last_logged_step,
        "last_checkpoint_step": last_checkpoint_step,
        "selected_checkpoint_step": selected_step,
        "weights_include_training_through_step": selected_step,
        "logged_but_not_checkpointed": evidence[
            "logged_but_not_checkpointed"
        ],
        "selected_checkpoint_role": "final",
        "checkpoint": str(selected_checkpoint_path),
        "checkpoint_best": str(selected_checkpoint_path),
        "checkpoint_last": str(last_checkpoint_path),
        "selected_checkpoint_sha256": selected_checkpoint_sha256,
        "selected_checkpoint_size_bytes": input_stats_before[
            "checkpoint_best"
        ]["size_bytes"],
        "train_log": str(train_log_path),
        "train_log_sha256": train_log_sha256,
        "train_log_row_count": len(rows),
        "selection_rule": [
            "maximize_validation_dice",
            "minimize_validation_loss_on_tie",
            "minimize_no_target_false_positive_rate_on_tie",
        ],
        "best_step": selected_step,
        "best_selection": evidence["best_selection"],
        "selection_validation_snapshot": selection_snapshot,
        "validation_trajectory": evidence["validation_trajectory"],
        "validation_replay_delta": validation_replay_delta,
        "benchmark_index_sha256": str(
            benchmark_contract["index_sha256"]
        ),
        "benchmark_contract": dict(benchmark_contract),
        "model_contract": dict(selected_payload["model_contract"]),
        "backbone_sha256": model.backbone_sha256,
        "variant": config.variant,
        "backbone": config.backbone,
        "architecture": model.model_contract()["architecture"],
        "modality_weight_order": list(model.modality_weight_order),
        "region_feature_dim": model.region_feature_dim,
        "training_diagnostics": {
            "sources": {
                "complete_run_tail": "train_log.jsonl",
                "last_recoverable_state": "checkpoint_last.pt",
                "final_weight_state": "checkpoint_best.pt",
            },
            "selected_checkpoint": selected_diagnostics,
            "last_checkpoint": last_diagnostics,
            "run_log": {
                "last_logged_step": last_logged_step,
                "row_count": len(rows),
                "validation_count": int(evidence["validation_count"]),
            },
        },
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "checkpoint_reload_max_abs_difference": reload_difference,
        "checkpoint_reload_seconds": reload_seconds,
        "evaluation_scope": "train_val_engineering",
        "test_evaluated": False,
        "engineering_checks": engineering_checks,
        "engineering_checks_passed": engineering_checks_passed,
        "acceptance_scope": "engineering_runtime_only",
        "acceptance_enforced": False,
        "acceptance_passed": None,
        "gate_a_evaluated": False,
        "formal_acceptance": False,
        "formal_acceptance_reason": "gate_a_not_evaluated",
        "input_asset_stats_before": input_stats_before,
        "input_asset_stats_after": input_stats_after,
        "evaluation_duration_seconds": {
            "train": train_metrics["duration_seconds"],
            "validation": validation_metrics["duration_seconds"],
        },
        "peak_cuda_memory_bytes": peak_memory,
        "peak_cuda_memory_gib": (
            peak_memory / (1024**3) if peak_memory is not None else None
        ),
        "wall_elapsed_seconds": time.perf_counter() - wall_started,
        "training_report": str(training_report_path),
    }
    atomic_write_json(training_report_path, report)
    if not engineering_checks_passed:
        failed = [
            name
            for name, passed in engineering_checks.items()
            if not passed
        ]
        raise RuntimeError(
            "人工定版报告已写入，但工程检查未通过：" + ", ".join(failed)
        )
    progress.phase(
        "[finalize] completed "
        f"selected_step={selected_step} last_logged_step={last_logged_step} "
        f"report={training_report_path}"
    )
    return report

__all__ = ["finalize_training_run"]
