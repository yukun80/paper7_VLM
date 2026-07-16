#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Training-run ownership, resume reconciliation and terminal artifact bindings."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping

from qpsalm_seg.paths import resolve_project_path

from .common import write_json
from .json_protocol import strict_json_loads


RESUME_RECONCILIATION_PROTOCOL = (
    "qpsalm_segdesc_resume_reconciliation_v1_checkpoint_cursor_bound"
)
FAILURE_HISTORY_PROTOCOL = "qpsalm_segdesc_failure_history_v1"
DESCRIPTION_TRAINING_COMPLETION_PROTOCOL = (
    "qpsalm_description_training_completion_v3_checkpoint_replayed"
)
JOINT_TRAINING_COMPLETION_PROTOCOL = (
    "qpsalm_segdesc_joint_training_completion_v3_checkpoint_replayed"
)
TERMINAL_CHECKPOINT_AUDIT_PROTOCOL = (
    "qpsalm_segdesc_terminal_checkpoint_audit_v2_role_progress_replayed"
)
CHECKPOINT_RUN_COMPLETION_PROTOCOL = (
    "qpsalm_segdesc_checkpoint_run_completion_v1_selection_role_bound"
)


def sha256_file(path: str | Path) -> str:
    resolved = resolve_project_path(path) or Path(path)
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_binding(path: str | Path) -> dict[str, Any]:
    resolved = resolve_project_path(path) or Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"run artifact 不存在: {resolved}")
    return {
        "path": str(resolved.resolve(strict=False)),
        "sha256": sha256_file(resolved),
        "bytes": int(resolved.stat().st_size),
    }


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_bytes(payload)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _jsonl_rows(path: Path) -> tuple[bytes, list[dict[str, Any]]]:
    payload = path.read_bytes()
    rows: list[dict[str, Any]] = []
    try:
        for line_number, raw in enumerate(payload.splitlines(), start=1):
            if not raw.strip():
                continue
            value = strict_json_loads(raw)
            if not isinstance(value, dict):
                raise ValueError(f"line={line_number} 顶层不是 object")
            rows.append(value)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"resume history 不是合法 JSONL: {path}: {exc}") from exc
    return payload, rows


def reconcile_resume_history(
    path: str | Path,
    *,
    checkpoint_step: int,
    required: bool,
) -> dict[str, Any]:
    """Keep the append-only history on the checkpoint timeline before resuming.

    A process can log steps after the most recent periodic checkpoint and then
    terminate. Those rows describe model states that cannot be restored. They
    are archived byte-for-byte and removed atomically from the active history.
    """

    resolved = resolve_project_path(path) or Path(path)
    if not resolved.is_file():
        if required:
            raise FileNotFoundError(
                f"resume checkpoint step={checkpoint_step} 但缺少 history: {resolved}"
            )
        return {
            "path": str(resolved.resolve(strict=False)),
            "exists": False,
            "required": False,
            "checkpoint_step": int(checkpoint_step),
            "rows_before": 0,
            "rows_retained": 0,
            "rows_archived": 0,
            "archive": None,
        }

    original, rows = _jsonl_rows(resolved)
    steps: list[int] = []
    for index, row in enumerate(rows):
        step = row.get("step")
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError(
                f"resume history step 非法: path={resolved} row={index} step={step!r}"
            )
        steps.append(int(step))
    if any(current <= previous for previous, current in zip(steps, steps[1:])):
        raise ValueError(f"resume history step 必须严格递增: {resolved}")

    retained = [row for row in rows if int(row["step"]) <= int(checkpoint_step)]
    discarded = rows[len(retained):]
    archive_binding: dict[str, Any] | None = None
    if discarded:
        original_sha = hashlib.sha256(original).hexdigest()
        archive = resolved.with_name(
            f"{resolved.stem}.pre_resume_{original_sha[:12]}{resolved.suffix}"
        )
        if archive.exists() and archive.read_bytes() != original:
            raise RuntimeError(f"resume history archive 名称冲突: {archive}")
        if not archive.exists():
            _atomic_bytes(archive, original)
        archive_binding = file_binding(archive)
        rewritten = b"".join(
            (
                json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
            ).encode("utf-8")
            for row in retained
        )
        _atomic_bytes(resolved, rewritten)

    active_binding = file_binding(resolved)
    return {
        "path": str(resolved.resolve(strict=False)),
        "exists": True,
        "required": bool(required),
        "checkpoint_step": int(checkpoint_step),
        "rows_before": len(rows),
        "rows_retained": len(retained),
        "rows_archived": len(discarded),
        "maximum_step_before": max(steps) if steps else None,
        "maximum_step_retained": (
            int(retained[-1]["step"]) if retained else None
        ),
        "active": active_binding,
        "archive": archive_binding,
    }


def reconcile_resume_run(
    output_dir: str | Path,
    *,
    resume_checkpoint: str | Path,
    checkpoint_step: int,
    histories: Mapping[str, bool],
    checkpoint_step_reader: Callable[[Path], int],
) -> dict[str, Any]:
    """Reject an older sibling checkpoint and reconcile all active histories."""

    output = resolve_project_path(output_dir) or Path(output_dir)
    output = output.resolve(strict=False)
    source = resolve_project_path(resume_checkpoint) or Path(resume_checkpoint)
    source = source.resolve(strict=False)
    if source.parent != output:
        raise ValueError("resume checkpoint 必须位于同一 output-dir")
    sibling_steps: dict[str, int] = {}
    for name in ("checkpoint_best.pt", "checkpoint_last.pt"):
        sibling = output / name
        if sibling.is_file():
            sibling_steps[name] = int(checkpoint_step_reader(sibling))
    newer = {
        name: step for name, step in sibling_steps.items()
        if int(step) > int(checkpoint_step)
    }
    if newer:
        raise RuntimeError(
            "resume checkpoint 不是该 run 最新可恢复状态；"
            f"source={source.name}@{checkpoint_step} newer={newer}"
        )

    history_audits = {
        name: reconcile_resume_history(
            output / name,
            checkpoint_step=int(checkpoint_step),
            required=bool(required),
        )
        for name, required in histories.items()
    }
    entry = {
        "resume_checkpoint": file_binding(source),
        "checkpoint_step": int(checkpoint_step),
        "sibling_checkpoint_steps": sibling_steps,
        "histories": history_audits,
    }
    report_path = output / "resume_reconciliation.json"
    previous_entries: list[dict[str, Any]] = []
    if report_path.is_file():
        try:
            previous = strict_json_loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("既有 resume_reconciliation.json 已损坏") from exc
        if (
            not isinstance(previous, dict)
            or previous.get("protocol") != RESUME_RECONCILIATION_PROTOCOL
            or not isinstance(previous.get("attempts"), list)
        ):
            raise ValueError("既有 resume reconciliation protocol 不兼容")
        previous_entries = list(previous["attempts"])
    report = {
        "protocol": RESUME_RECONCILIATION_PROTOCOL,
        "attempts": [*previous_entries, entry],
        "latest": entry,
    }
    write_json(report_path, report)
    return entry


def prepare_training_attempt(output_dir: str | Path, *, resume: bool) -> dict[str, Any]:
    """Enforce mutually exclusive terminal state while retaining old failures."""

    output = resolve_project_path(output_dir) or Path(output_dir)
    completion = output / "training_report.json"
    if resume and completion.is_file():
        raise RuntimeError(
            "该 output-dir 已有 training_report.json；完成的单 run 不允许再次 resume"
        )
    failure = output / "failure_report.json"
    archived: dict[str, Any] | None = None
    if failure.is_file():
        try:
            payload = strict_json_loads(failure.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("既有 failure_report.json 已损坏，拒绝覆盖") from exc
        if not isinstance(payload, dict):
            raise ValueError("既有 failure_report.json 顶层必须为 object")
        binding = file_binding(failure)
        history_path = output / "failure_history.json"
        entries: list[dict[str, Any]] = []
        if history_path.is_file():
            history = strict_json_loads(history_path.read_text(encoding="utf-8"))
            if (
                not isinstance(history, dict)
                or history.get("protocol") != FAILURE_HISTORY_PROTOCOL
                or not isinstance(history.get("entries"), list)
            ):
                raise ValueError("failure_history.json protocol 不兼容")
            entries = list(history["entries"])
        archived = {"binding": binding, "report": payload}
        if not any(
            isinstance(value, dict)
            and (value.get("binding") or {}).get("sha256") == binding["sha256"]
            for value in entries
        ):
            entries.append(archived)
        write_json(history_path, {
            "protocol": FAILURE_HISTORY_PROTOCOL,
            "entries": entries,
        })
        failure.unlink()
    return {"resume": bool(resume), "archived_failure": archived}


def build_training_completion_report(
    *,
    protocol: str,
    report: Mapping[str, Any],
    required_artifacts: Mapping[str, str | Path],
    optional_artifacts: Mapping[str, str | Path] | None = None,
) -> dict[str, Any]:
    """Bind a successful terminal report to the files that prove completion."""

    reserved = {"protocol", "terminal_status", "artifacts"}
    collisions = sorted(reserved & set(report))
    if collisions:
        raise ValueError(
            f"training completion report 不能覆盖保留字段: {collisions}"
        )
    steps = report.get("steps")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
        raise ValueError(f"training completion steps 必须为正整数: {steps!r}")

    artifacts = {
        name: file_binding(path) for name, path in required_artifacts.items()
    }
    for name, path in (optional_artifacts or {}).items():
        resolved = resolve_project_path(path) or Path(path)
        artifacts[name] = file_binding(resolved) if resolved.is_file() else None
    terminal_checkpoint = report.get("terminal_checkpoint_audit")
    if terminal_checkpoint is not None:
        checkpoint_binding = artifacts.get("checkpoint_last")
        if (
            not isinstance(terminal_checkpoint, Mapping)
            or terminal_checkpoint.get("protocol")
            != TERMINAL_CHECKPOINT_AUDIT_PROTOCOL
            or not isinstance(checkpoint_binding, dict)
            or terminal_checkpoint.get("checkpoint_sha256")
            != checkpoint_binding.get("sha256")
            or Path(str(terminal_checkpoint.get("checkpoint") or "")).resolve(
                strict=False
            )
            != Path(str(checkpoint_binding.get("path") or "")).resolve(
                strict=False
            )
        ):
            raise RuntimeError(
                "terminal checkpoint audit 与 completion artifact binding 不一致"
            )
        for role in ("progress", "history"):
            artifact_name = terminal_checkpoint.get(f"{role}_artifact_name")
            audited_binding = terminal_checkpoint.get(f"{role}_artifact")
            completion_binding = artifacts.get(str(artifact_name or ""))
            if (
                not isinstance(artifact_name, str)
                or not artifact_name
                or not isinstance(audited_binding, Mapping)
                or not isinstance(completion_binding, dict)
                or dict(audited_binding) != completion_binding
            ):
                raise RuntimeError(
                    "terminal checkpoint audit 的 progress/history binding "
                    f"不一致: role={role} name={artifact_name!r}"
                )
    return {
        **dict(report),
        "protocol": str(protocol),
        "terminal_status": "completed",
        "artifacts": artifacts,
    }


def validate_terminal_checkpoint_provenance(
    provenance: Mapping[str, Any],
    *,
    checkpoint: str | Path,
    expected_step: int,
    expected_stage: str,
    progress_key: str,
    expected_progress_protocol: str,
    progress_artifact: str | Path,
    progress_artifact_name: str,
    history_artifact: str | Path,
    history_artifact_name: str,
) -> dict[str, Any]:
    """Bind terminal success to the exact saved checkpoint progress cursor."""
    resolved = (resolve_project_path(checkpoint) or Path(checkpoint)).resolve(
        strict=False
    )
    observed = Path(str(provenance.get("checkpoint") or "")).resolve(
        strict=False
    )
    if observed != resolved:
        raise RuntimeError(
            "terminal checkpoint provenance path 不一致: "
            f"expected={resolved} observed={observed}"
        )
    checkpoint_step = provenance.get("checkpoint_step")
    if (
        isinstance(checkpoint_step, bool)
        or not isinstance(checkpoint_step, int)
        or int(checkpoint_step) != int(expected_step)
    ):
        raise RuntimeError(
            "terminal checkpoint step 与训练返回值不一致: "
            f"expected={expected_step} observed={checkpoint_step!r}"
        )
    metadata_report = provenance.get("checkpoint_metadata")
    metadata = (
        metadata_report.get("metadata")
        if isinstance(metadata_report, dict)
        else None
    )
    observed_stage = metadata.get("stage") if isinstance(metadata, dict) else None
    if not isinstance(metadata, dict) or observed_stage != expected_stage:
        raise RuntimeError(
            "terminal checkpoint stage 不一致: "
            f"expected={expected_stage!r} observed="
            f"{observed_stage!r}"
        )
    if metadata.get("checkpoint_role") != "terminal_last":
        raise RuntimeError(
            "terminal checkpoint 必须声明 checkpoint_role=terminal_last: "
            f"observed={metadata.get('checkpoint_role')!r}"
        )
    progress = metadata.get(progress_key)
    observed_progress_step = (
        progress.get("step") if isinstance(progress, dict) else None
    )
    if (
        not isinstance(progress, dict)
        or isinstance(observed_progress_step, bool)
        or not isinstance(observed_progress_step, int)
        or observed_progress_step != int(expected_step)
    ):
        raise RuntimeError(
            "terminal checkpoint progress 与 checkpoint step 不一致: "
            f"key={progress_key} expected={expected_step} "
            f"observed={observed_progress_step!r}"
        )
    if progress.get("protocol") != expected_progress_protocol:
        raise RuntimeError(
            "terminal checkpoint progress protocol 不一致: "
            f"expected={expected_progress_protocol!r} "
            f"observed={progress.get('protocol')!r}"
        )
    progress_path = resolve_project_path(progress_artifact) or Path(
        progress_artifact
    )
    try:
        published_progress = strict_json_loads(
            progress_path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"terminal progress artifact 不是合法 JSON: {progress_path}"
        ) from exc
    if published_progress != progress:
        raise RuntimeError(
            "terminal progress artifact 与 checkpoint metadata 不一致: "
            f"path={progress_path}"
        )
    history_path = resolve_project_path(history_artifact) or Path(
        history_artifact
    )
    try:
        _history_bytes, history_rows = _jsonl_rows(history_path)
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            f"terminal history artifact 不是合法 JSONL: {history_path}"
        ) from exc
    history_steps = [value.get("step") for value in history_rows]
    if (
        not history_steps
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in history_steps
        )
        or any(
            current <= previous
            for previous, current in zip(history_steps, history_steps[1:])
        )
        or history_steps[-1] != int(expected_step)
    ):
        raise RuntimeError(
            "terminal history 必须严格递增并结束于 checkpoint step: "
            f"expected={expected_step} observed={history_steps[-8:]}"
        )
    sha256 = provenance.get("checkpoint_sha256")
    if (
        not isinstance(sha256, str)
        or len(sha256) != 64
        or any(value not in "0123456789abcdef" for value in sha256.casefold())
    ):
        raise RuntimeError("terminal checkpoint provenance 缺少 SHA-256")
    return {
        "protocol": TERMINAL_CHECKPOINT_AUDIT_PROTOCOL,
        "checkpoint": str(resolved),
        "checkpoint_sha256": sha256,
        "checkpoint_step": int(expected_step),
        "stage": str(expected_stage),
        "checkpoint_role": "terminal_last",
        "progress_key": str(progress_key),
        "progress_protocol": str(expected_progress_protocol),
        "progress_artifact_name": str(progress_artifact_name),
        "progress_artifact": file_binding(progress_path),
        "history_artifact_name": str(history_artifact_name),
        "history_artifact": file_binding(history_path),
        "checkpoint_provenance": dict(provenance),
    }


def validate_training_completion_report(
    path: str | Path,
    *,
    expected_protocol: str,
) -> dict[str, Any]:
    """Replay a completed training publication and every terminal binding."""
    report_path = resolve_project_path(path) or Path(path)
    if not report_path.is_file():
        raise FileNotFoundError(
            f"training completion report 不存在: {report_path}"
        )
    try:
        report = strict_json_loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"training completion report 不是合法 JSON: {report_path}"
        ) from exc
    if (
        not isinstance(report, dict)
        or report.get("protocol") != expected_protocol
        or report.get("terminal_status") != "completed"
        or not isinstance(report.get("artifacts"), dict)
    ):
        raise ValueError(
            "training completion protocol/status/artifacts 不兼容: "
            f"{report_path}"
        )
    artifacts = dict(report["artifacts"])
    for name, binding in artifacts.items():
        if binding is None:
            continue
        if (
            not isinstance(binding, dict)
            or not isinstance(binding.get("path"), str)
            or not binding["path"]
        ):
            raise ValueError(
                f"training completion artifact binding 非法: {name}"
            )
        try:
            current = file_binding(str(binding["path"]))
        except (FileNotFoundError, OSError) as exc:
            raise ValueError(
                f"training completion artifact 已缺失: {name}"
            ) from exc
        if current != binding:
            raise ValueError(
                f"training completion artifact 已漂移: {name}"
            )

    terminal = report.get("terminal_checkpoint_audit")
    if (
        not isinstance(terminal, dict)
        or terminal.get("protocol") != TERMINAL_CHECKPOINT_AUDIT_PROTOCOL
    ):
        raise ValueError("training completion 缺少当前 terminal checkpoint audit")
    checkpoint_binding = artifacts.get("checkpoint_last")
    progress_name = terminal.get("progress_artifact_name")
    history_name = terminal.get("history_artifact_name")
    progress_binding = artifacts.get(str(progress_name or ""))
    history_binding = artifacts.get(str(history_name or ""))
    report_steps = report.get("steps")
    terminal_step = terminal.get("checkpoint_step")
    if (
        not isinstance(checkpoint_binding, dict)
        or not isinstance(progress_binding, dict)
        or not isinstance(history_binding, dict)
        or terminal.get("checkpoint_role") != "terminal_last"
        or isinstance(report_steps, bool)
        or not isinstance(report_steps, int)
        or report_steps <= 0
        or isinstance(terminal_step, bool)
        or not isinstance(terminal_step, int)
        or terminal_step != report_steps
        or not isinstance(progress_name, str)
        or not progress_name
        or not isinstance(history_name, str)
        or not history_name
    ):
        raise ValueError("training completion terminal artifact/step/role 绑定非法")

    from .checkpoint import inspect_segdesc_checkpoint

    try:
        provenance = inspect_segdesc_checkpoint(checkpoint_binding["path"])
        rebuilt_terminal = validate_terminal_checkpoint_provenance(
            provenance,
            checkpoint=checkpoint_binding["path"],
            expected_step=terminal_step,
            expected_stage=str(terminal.get("stage") or ""),
            progress_key=str(terminal.get("progress_key") or ""),
            expected_progress_protocol=str(
                terminal.get("progress_protocol") or ""
            ),
            progress_artifact=progress_binding["path"],
            progress_artifact_name=str(progress_name),
            history_artifact=history_binding["path"],
            history_artifact_name=str(history_name),
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        raise ValueError(
            "training completion terminal checkpoint 无法重放"
        ) from exc
    if rebuilt_terminal != terminal:
        raise ValueError("training completion terminal checkpoint audit 已漂移")
    return report


def validate_checkpoint_run_completion(
    checkpoint: str | Path,
    *,
    expected_completion_protocol: str,
    expected_stage: str,
    expected_role: str,
) -> dict[str, Any]:
    """Bind a selected best/last checkpoint to one successfully completed run."""
    if expected_role not in {"validation_best", "terminal_last"}:
        raise ValueError(
            f"checkpoint run completion role 非法: {expected_role!r}"
        )
    checkpoint_path = (
        resolve_project_path(checkpoint) or Path(checkpoint)
    ).resolve(strict=False)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"checkpoint run completion 源 checkpoint 不存在: {checkpoint_path}"
        )
    run_dir = checkpoint_path.parent
    completion_path = run_dir / "training_report.json"
    if (run_dir / "failure_report.json").exists():
        raise ValueError(
            "已发布 training completion 的 run 不得同时残留 failure_report.json"
        )
    completion = validate_training_completion_report(
        completion_path,
        expected_protocol=expected_completion_protocol,
    )
    if completion.get("stage") != expected_stage:
        raise ValueError(
            "checkpoint run completion stage 不一致: "
            f"expected={expected_stage!r} observed={completion.get('stage')!r}"
        )
    artifacts = dict(completion.get("artifacts") or {})
    selected_artifact_name = (
        "checkpoint_best"
        if expected_role == "validation_best"
        else "checkpoint_last"
    )
    selected_binding = artifacts.get(selected_artifact_name)
    current_binding = file_binding(checkpoint_path)
    if not isinstance(selected_binding, dict) or selected_binding != current_binding:
        raise ValueError(
            "selected checkpoint 未由成功 training completion 精确绑定: "
            f"artifact={selected_artifact_name}"
        )

    from .checkpoint import inspect_segdesc_checkpoint

    try:
        selected_provenance = inspect_segdesc_checkpoint(checkpoint_path)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        raise ValueError("selected checkpoint payload 无法重放") from exc
    metadata = dict(
        (selected_provenance.get("checkpoint_metadata") or {}).get(
            "metadata"
        ) or {}
    )
    if (
        metadata.get("stage") != expected_stage
        or metadata.get("checkpoint_role") != expected_role
    ):
        raise ValueError(
            "selected checkpoint payload stage/role 与完成 run 不一致"
        )

    selection_binding: dict[str, Any] | None = None
    if expected_role == "validation_best":
        selection_binding = artifacts.get("validation_best")
        if not isinstance(selection_binding, dict):
            raise ValueError(
                "validation_best checkpoint 缺少 selection report binding"
            )
        try:
            selection = strict_json_loads(
                Path(selection_binding["path"]).read_text(encoding="utf-8")
            )
        except (KeyError, OSError, json.JSONDecodeError, TypeError) as exc:
            raise ValueError("validation_best selection report 无法重放") from exc
        if (
            not isinstance(selection, dict)
            or isinstance(selection.get("step"), bool)
            or not isinstance(selection.get("step"), int)
            or selection.get("step")
            != selected_provenance.get("checkpoint_step")
        ):
            raise ValueError(
                "validation_best selection step 与 checkpoint 不一致"
            )
    else:
        terminal = dict(completion.get("terminal_checkpoint_audit") or {})
        if terminal.get("checkpoint_sha256") != current_binding["sha256"]:
            raise ValueError(
                "terminal_last checkpoint 与 completion terminal audit 不一致"
            )

    return {
        "protocol": CHECKPOINT_RUN_COMPLETION_PROTOCOL,
        "passed": True,
        "training_report": file_binding(completion_path),
        "completion_protocol": expected_completion_protocol,
        "stage": expected_stage,
        "checkpoint_role": expected_role,
        "checkpoint_step": int(selected_provenance["checkpoint_step"]),
        "selected_artifact_name": selected_artifact_name,
        "selected_checkpoint": current_binding,
        "selection_report": selection_binding,
    }
