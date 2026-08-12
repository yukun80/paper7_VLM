"""Phase4 训练/验证的 TTY 进度和低噪声固定日志。"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping, TextIO

from tqdm import tqdm


def format_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "--:--"
    total = int(round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, seconds_value = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{seconds_value:02d}"
    return f"{minutes:02d}:{seconds_value:02d}"


def _memory_label(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}G"


class TrainingProgress:
    """交互终端使用 tqdm，重定向时输出可解析的固定行。"""

    def __init__(
        self,
        *,
        log_interval: int,
        stream: TextIO | None = None,
        force_tty: bool | None = None,
    ) -> None:
        if log_interval <= 0:
            raise ValueError("log_interval 必须 > 0")
        self.log_interval = int(log_interval)
        self.stream = stream if stream is not None else sys.stderr
        self.is_tty = (
            bool(self.stream.isatty()) if force_tty is None else bool(force_tty)
        )
        self._train_bar: tqdm[Any] | None = None
        self._validation_bar: tqdm[Any] | None = None
        self._last_step = 0
        self._validation_completed = 0
        self._stop_step = 0

    def __enter__(self) -> "TrainingProgress":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def _emit(self, message: str) -> None:
        if self.is_tty and self._train_bar is not None:
            tqdm.write(message, file=self.stream)
        else:
            print(message, file=self.stream, flush=True)

    def phase(self, message: str) -> None:
        self._emit(message)

    def announce_setup(
        self,
        *,
        run_name: str,
        config_path: Path,
        config_sha256: str,
        benchmark_build_id: str,
        benchmark_payload_sha256: str,
        model_path: str,
        processor_path: str,
        device: str,
        gpu_name: str | None,
        base_parameters: int,
        trainable_parameters: int,
        train_samples: int,
        validation_samples: int,
        batch_size: int,
        accumulation_steps: int,
        input_pipeline_backend: str,
        num_workers: int,
        prefetch_factor: int,
        pin_memory: bool,
        max_images: int,
        max_input_tokens: int,
        min_pixels: int,
        max_pixels: int,
        learning_rate: float,
        warmup_steps: int,
        max_steps: int,
        start_step: int,
        stop_step: int,
        validation_interval: int,
        checkpoint_interval: int,
        output_root: Path,
    ) -> None:
        device_label = device if gpu_name is None else f"{device}:{gpu_name}"
        self._emit(
            "[setup] "
            f"run={run_name} config={config_path} "
            f"config_sha256={config_sha256}"
        )
        self._emit(
            "[setup] "
            f"benchmark_build={benchmark_build_id} "
            f"payload_sha256={benchmark_payload_sha256} "
            f"train={train_samples} bounded_val={validation_samples}"
        )
        self._emit(
            "[setup] "
            f"model={model_path} processor={processor_path} "
            f"device={device_label} base_params={base_parameters} "
            f"trainable_params={trainable_parameters}"
        )
        self._emit(
            "[setup] "
            f"batch={batch_size} accumulation={accumulation_steps} "
            f"effective_batch={batch_size * accumulation_steps} "
            f"images<={max_images} tokens<={max_input_tokens} "
            f"pixels={min_pixels}..{max_pixels}"
        )
        self._emit(
            "[setup] "
            f"input_pipeline={input_pipeline_backend} workers={num_workers} "
            f"prefetch_factor={prefetch_factor} "
            f"pin_memory={str(pin_memory).lower()}"
        )
        self._emit(
            "[setup] "
            f"lr={learning_rate:.3e} warmup_steps={warmup_steps} "
            f"steps={start_step}->{stop_step}/{max_steps} "
            f"log/val/ckpt={self.log_interval}/{validation_interval}/"
            f"{checkpoint_interval} output={output_root}"
        )
        self._emit(
            "[setup] validation=bounded_external_val_teacher_forced_loss "
            "metric=macro_task_loss tie_break=overall_loss,earlier_step "
            "generation=false"
        )

    def start_training(
        self,
        *,
        start_step: int,
        stop_step: int,
    ) -> None:
        self._stop_step = stop_step
        self._last_step = start_step
        if self.is_tty:
            self._train_bar = tqdm(
                total=stop_step,
                initial=start_step,
                desc="phase4 train",
                unit="step",
                dynamic_ncols=True,
                mininterval=1.0,
                leave=True,
                file=self.stream,
            )

    def update_training(self, telemetry: Mapping[str, Any]) -> None:
        step = int(telemetry["step"])
        peak = _memory_label(telemetry.get("cuda_peak_gib"))
        if self._train_bar is not None:
            memory = (
                f"{_memory_label(telemetry.get('cuda_allocated_gib'))}/"
                f"{_memory_label(telemetry.get('cuda_reserved_gib'))}/"
                f"{peak}"
            )
            self._train_bar.set_postfix(
                {
                    "loss": f"{float(telemetry['loss']):.4f}",
                    "ema": f"{float(telemetry['ema_loss']):.4f}",
                    "lr": f"{float(telemetry['learning_rate']):.2e}",
                    "grad": f"{float(telemetry['gradient_norm']):.3f}",
                    "clip": str(
                        bool(telemetry["gradient_clipped"])
                    ).lower(),
                    "n": int(telemetry["samples"]),
                    "tok": int(telemetry["input_tokens"]),
                    "img": int(telemetry["images"]),
                    "dt": format_duration(
                        float(telemetry["step_duration_seconds"])
                    ),
                    "wait": (
                        f"{float(telemetry['data_wait_seconds']):.2f}s/"
                        f"{100.0 * float(telemetry['data_wait_fraction']):.0f}%"
                    ),
                    "sample/s": f"{float(telemetry['samples_per_second']):.2f}",
                    "tok/s": f"{float(telemetry['tokens_per_second']):.1f}",
                    "mem": memory,
                },
                refresh=False,
            )
            increment = max(step - self._last_step, 0)
            if increment:
                self._train_bar.update(increment)
            self._last_step = max(self._last_step, step)
            return
        if (
            step == 1
            or step % self.log_interval == 0
            or step == self._stop_step
        ):
            self._emit(
                "[train] "
                f"step={step}/{self._stop_step} "
                f"loss={float(telemetry['loss']):.6f} "
                f"ema={float(telemetry['ema_loss']):.6f} "
                f"lr={float(telemetry['learning_rate']):.3e} "
                f"grad={float(telemetry['gradient_norm']):.4f} "
                f"clipped={str(bool(telemetry['gradient_clipped'])).lower()} "
                f"samples={int(telemetry['samples'])} "
                f"tokens={int(telemetry['input_tokens'])} "
                f"images={int(telemetry['images'])} "
                f"step_time={format_duration(float(telemetry['step_duration_seconds']))} "
                f"data_wait={float(telemetry['data_wait_seconds']):.3f}s "
                f"wait_ratio={100.0 * float(telemetry['data_wait_fraction']):.1f}% "
                f"sample/s={float(telemetry['samples_per_second']):.3f} "
                f"tok/s={float(telemetry['tokens_per_second']):.1f} "
                f"elapsed={format_duration(float(telemetry['run_elapsed_seconds']))} "
                f"eta={format_duration(telemetry.get('eta_seconds'))} "
                f"cuda={_memory_label(telemetry.get('cuda_allocated_gib'))}/"
                f"{_memory_label(telemetry.get('cuda_reserved_gib'))}/"
                f"{peak}"
            )

    def start_validation(self, *, step: int, total_samples: int) -> None:
        self.abort_validation()
        self._validation_completed = 0
        if self.is_tty:
            self._validation_bar = tqdm(
                total=total_samples,
                desc=f"external_val step {step}",
                unit="sample",
                dynamic_ncols=True,
                mininterval=1.0,
                leave=False,
                file=self.stream,
                position=1 if self._train_bar is not None else 0,
            )
        else:
            self._emit(
                f"[val] step={step} samples={total_samples} start"
            )

    def update_validation(
        self,
        *,
        completed: int,
        total_samples: int,
        running_loss: float,
    ) -> None:
        if self._validation_bar is None:
            return
        self._validation_bar.set_postfix(
            {"loss": f"{running_loss:.4f}"},
            refresh=False,
        )
        increment = max(completed - self._validation_completed, 0)
        if increment:
            self._validation_bar.update(increment)
        self._validation_completed = completed

    def finish_validation(
        self,
        *,
        step: int,
        macro_task_loss: float,
        overall_loss: float,
        duration_seconds: float,
    ) -> None:
        self.abort_validation()
        self._emit(
            "[val] "
            f"step={step} macro_task_loss={macro_task_loss:.6f} "
            f"overall_loss={overall_loss:.6f} "
            f"time={format_duration(duration_seconds)}"
        )

    def checkpoint(self, *, step: int, path: Path, seconds: float) -> None:
        self._emit(
            "[checkpoint] "
            f"step={step} time={format_duration(seconds)} path={path}"
        )

    def close(self) -> None:
        self.abort_validation()
        if self._train_bar is not None:
            self._train_bar.close()
            self._train_bar = None

    def abort_validation(self) -> None:
        if self._validation_bar is not None:
            self._validation_bar.close()
            self._validation_bar = None


def compact_training_result(result: Mapping[str, Any]) -> str:
    peak = _memory_label(result.get("cuda_peak_gib"))
    best = result.get("best_checkpoint")
    best_label = "-" if best is None else str(best)
    return "\n".join(
        (
            "[done] "
            f"status={result['status']} "
            f"step={result['global_step']}/{result['max_steps']} "
            f"last_loss={float(result['last_loss']):.6f} "
            f"wall={format_duration(float(result['run_elapsed_seconds']))} "
            f"peak_cuda={peak}",
            "[done] "
            f"checkpoint={result['checkpoint']} "
            f"best={best_label} report={result['training_report']}",
        )
    )
