"""Phase 2 训练与评价的低噪声终端进度显示。"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping, TextIO

from tqdm import tqdm


def format_duration(seconds: float | None) -> str:
    """将秒数格式化为紧凑且稳定的终端时间。"""

    if seconds is None or seconds < 0:
        return "--:--"
    total_seconds = int(round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds_value = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds_value:02d}"
    return f"{minutes:02d}:{seconds_value:02d}"


def _float(value: Any, default: float = 0.0) -> float:
    return default if value is None else float(value)


class TrainingProgress:
    """TTY 使用 tqdm，非 TTY 使用按 log_interval 限流的固定文本。"""

    def __init__(
        self,
        *,
        log_interval: int,
        stream: TextIO | None = None,
        force_tty: bool | None = None,
    ) -> None:
        if log_interval <= 0:
            raise ValueError("log_interval 必须大于 0")
        self.log_interval = int(log_interval)
        self.stream = stream if stream is not None else sys.stderr
        self.is_tty = (
            bool(self.stream.isatty()) if force_tty is None else bool(force_tty)
        )
        self._training_bar: tqdm[Any] | None = None
        self._evaluation_bar: tqdm[Any] | None = None
        self._training_last_step = 0
        self._evaluation_last_batch = 0
        self._training_total = 0
        self.closed = False

    def __enter__(self) -> "TrainingProgress":
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: Any,
    ) -> None:
        self.close()

    def _emit(self, message: str) -> None:
        if self.is_tty and self._training_bar is not None:
            tqdm.write(message, file=self.stream)
        else:
            print(message, file=self.stream, flush=True)

    def phase(self, message: str) -> None:
        self._emit(message)

    def announce_setup(
        self,
        *,
        command: str,
        variant: str,
        device: str,
        gpu_name: str | None,
        train_samples: int,
        validation_samples: int,
        batch_size: int,
        total_steps: int,
        start_step: int,
        eval_interval: int,
        checkpoint_interval: int,
        output_dir: Path,
    ) -> None:
        device_label = device if gpu_name is None else f"{device} ({gpu_name})"
        resume_label = (
            f" resume_step={start_step}" if start_step > 0 else ""
        )
        self._emit(
            "[setup] "
            f"command={command} variant={variant} device={device_label} "
            f"train={train_samples} val={validation_samples} "
            f"batch={batch_size} steps={total_steps}{resume_label} "
            f"log/eval/ckpt={self.log_interval}/{eval_interval}/"
            f"{checkpoint_interval} output={output_dir}"
        )

    def start_training(
        self, *, variant: str, total_steps: int, start_step: int
    ) -> None:
        self.finish_training()
        self.closed = False
        self._training_total = int(total_steps)
        self._training_last_step = int(start_step)
        if self.is_tty:
            self._training_bar = tqdm(
                total=total_steps,
                initial=start_step,
                desc=f"train {variant}",
                unit="step",
                dynamic_ncols=True,
                mininterval=1.0,
                leave=True,
                file=self.stream,
                position=0,
            )

    def update_training(
        self,
        *,
        step: int,
        loss: float,
        ema_loss: float,
        bce: float,
        dice_loss: float,
        learning_rates: tuple[float, float],
        samples_per_second: float,
        estimated_remaining_seconds: float | None,
        peak_cuda_memory_gib: float | None,
    ) -> None:
        peak_label = (
            "-"
            if peak_cuda_memory_gib is None
            else f"{peak_cuda_memory_gib:.2f}G"
        )
        if self.is_tty and self._training_bar is not None:
            self._training_bar.set_postfix(
                {
                    "loss": f"{loss:.4f}",
                    "ema": f"{ema_loss:.4f}",
                    "bce": f"{bce:.4f}",
                    "diceL": f"{dice_loss:.4f}",
                    "lr": (
                        f"{learning_rates[0]:.2e}/"
                        f"{learning_rates[1]:.2e}"
                    ),
                    "sample/s": f"{samples_per_second:.2f}",
                    "peak": peak_label,
                },
                refresh=False,
            )
            increment = max(int(step) - self._training_last_step, 0)
            if increment:
                self._training_bar.update(increment)
            self._training_last_step = max(self._training_last_step, int(step))
            return
        if (
            step == 1
            or step % self.log_interval == 0
            or step == self._training_total
        ):
            print(
                "[train] "
                f"step={step}/{self._training_total} "
                f"loss={loss:.4f} ema={ema_loss:.4f} "
                f"bce={bce:.4f} diceL={dice_loss:.4f} "
                f"lr={learning_rates[0]:.2e}/{learning_rates[1]:.2e} "
                f"sample/s={samples_per_second:.2f} "
                f"eta={format_duration(estimated_remaining_seconds)} "
                f"peak={peak_label}",
                file=self.stream,
                flush=True,
            )

    def finish_training(self) -> None:
        if self._training_bar is not None:
            self._training_bar.close()
            self._training_bar = None

    def start_evaluation(self, *, label: str, total_batches: int) -> None:
        self.abort_evaluation()
        self.closed = False
        self._evaluation_last_batch = 0
        if self.is_tty:
            self._evaluation_bar = tqdm(
                total=total_batches,
                desc=label,
                unit="batch",
                dynamic_ncols=True,
                mininterval=1.0,
                leave=False,
                file=self.stream,
                position=1 if self._training_bar is not None else 0,
            )
        else:
            self._emit(f"[eval] phase={label} batches={total_batches} start")

    def update_evaluation(
        self,
        *,
        batch: int,
        running_loss: float,
        metrics: Mapping[str, Any],
    ) -> None:
        if self._evaluation_bar is None:
            return
        self._evaluation_bar.set_postfix(
            {
                "loss": f"{running_loss:.4f}",
                "dice": f"{_float(metrics.get('dice')):.4f}",
                "iou": f"{_float(metrics.get('iou')):.4f}",
            },
            refresh=False,
        )
        increment = max(int(batch) - self._evaluation_last_batch, 0)
        if increment:
            self._evaluation_bar.update(increment)
        self._evaluation_last_batch = max(
            self._evaluation_last_batch, int(batch)
        )

    def finish_evaluation(
        self,
        *,
        label: str,
        result: Mapping[str, Any],
        duration_seconds: float,
    ) -> None:
        self.abort_evaluation()
        overall = result["overall"]
        self._emit(
            "[eval] "
            f"phase={label} time={format_duration(duration_seconds)} "
            f"loss={_float(result.get('loss')):.4f} "
            f"iou={_float(overall.get('iou')):.4f} "
            f"dice={_float(overall.get('dice')):.4f} "
            f"precision={_float(overall.get('precision')):.4f} "
            f"recall={_float(overall.get('recall')):.4f} "
            f"f1={_float(overall.get('f1')):.4f} "
            "positive_dice="
            f"{_float(overall.get('positive_only_dice')):.4f} "
            "empty_fpr="
            f"{_float(overall.get('no_target_false_positive_rate')):.4f}"
        )

    def abort_evaluation(self) -> None:
        if self._evaluation_bar is not None:
            self._evaluation_bar.close()
            self._evaluation_bar = None

    def close(self) -> None:
        self.abort_evaluation()
        self.finish_training()
        self.closed = True


def format_compact_training_report(report: Mapping[str, Any]) -> str:
    """训练 CLI 默认的人类可读终态摘要。"""

    train = report["train_metrics"]["overall"]
    validation = report["validation_metrics"]["overall"]
    acceptance_enforced = bool(report.get("acceptance_enforced", False))
    if acceptance_enforced:
        acceptance_label = (
            "PASS" if bool(report.get("acceptance_passed")) else "FAIL"
        )
    else:
        acceptance_label = "NOT_ENFORCED"
    peak_memory = report.get("peak_cuda_memory_gib")
    peak_label = "-" if peak_memory is None else f"{float(peak_memory):.2f} GiB"
    return "\n".join(
        (
            "[done] "
            f"command={report['command']} variant={report['variant']} "
            f"steps={report['steps']} "
            f"wall={format_duration(_float(report.get('wall_elapsed_seconds')))} "
            f"peak_cuda={peak_label}",
            "[done] "
            f"loss_drop={100.0 * _float(report.get('loss_drop_fraction')):.2f}% "
            f"ema_drop={100.0 * _float(report.get('ema_drop_fraction')):.2f}% "
            f"train_dice={_float(train.get('dice')):.4f} "
            f"train_iou={_float(train.get('iou')):.4f}",
            "[done] "
            f"val_loss={_float(report['validation_metrics'].get('loss')):.4f} "
            f"val_dice={_float(validation.get('dice')):.4f} "
            f"val_iou={_float(validation.get('iou')):.4f} "
            f"precision={_float(validation.get('precision')):.4f} "
            f"recall={_float(validation.get('recall')):.4f} "
            f"f1={_float(validation.get('f1')):.4f} "
            "empty_fpr="
            f"{_float(validation.get('no_target_false_positive_rate')):.4f}",
            "[done] "
            f"acceptance={acceptance_label} "
            f"checkpoint={report['checkpoint']} "
            f"report={report['training_report']}",
        )
    )


def format_compact_finalization_report(report: Mapping[str, Any]) -> str:
    """人工定版 CLI 默认的人类可读终态摘要。"""

    train = report["train_metrics"]["overall"]
    validation = report["validation_metrics"]["overall"]
    peak_memory = report.get("peak_cuda_memory_gib")
    peak_label = "-" if peak_memory is None else f"{float(peak_memory):.2f} GiB"
    engineering_label = (
        "PASS" if bool(report.get("engineering_checks_passed")) else "FAIL"
    )
    return "\n".join(
        (
            "[done] "
            f"command=finalize status={report['status']} "
            f"completion={report['completion_mode']} "
            f"selected_step={report['selected_checkpoint_step']} "
            f"last_checkpoint_step={report['last_checkpoint_step']} "
            f"last_logged_step={report['last_logged_step']}",
            "[done] "
            f"train_dice={_float(train.get('dice')):.4f} "
            f"train_iou={_float(train.get('iou')):.4f} "
            f"val_dice={_float(validation.get('dice')):.4f} "
            f"val_iou={_float(validation.get('iou')):.4f} "
            "empty_fpr="
            f"{_float(validation.get('no_target_false_positive_rate')):.4f} "
            f"peak_cuda={peak_label}",
            "[done] "
            f"engineering_checks={engineering_label} "
            "gate_a=NOT_EVALUATED formal_acceptance=false "
            f"checkpoint={report['checkpoint']} "
            f"report={report['training_report']}",
        )
    )
