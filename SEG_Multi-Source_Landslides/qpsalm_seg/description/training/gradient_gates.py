"""Task-path-aware accumulation-window gradient gate lifecycle."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable

import torch

from ..protocols.versions import DESCRIPTION_GRADIENT_GATE_PROTOCOL
from ..protocols.gates import d_minus_one_gradient_gate_passed
from .engineering_gates import description_step_gradient_gate


class DescriptionGradientGateTracker:
    """Track proof for every training stream and every required task path."""

    def __init__(self, stream_names: Iterable[str], *, run_stage: str) -> None:
        self.run_stage = str(run_stage)
        names = tuple(str(value) for value in stream_names)
        required = (
            {"global_caption", "region_description"}
            if self.run_stage == "overfit" else {"stage"}
        )
        self.required = {name: set(required) for name in names}
        self.observed = {name: set() for name in names}
        self.reports: dict[str, dict[str, Any]] = {
            name: {
                "required_task_paths": sorted(self.required[name]),
                "observed_task_paths": [],
                "path_reports": {},
                "passed": False,
            }
            for name in names
        }

    @staticmethod
    def task_paths(use_region_tokens: Iterable[bool]) -> set[str]:
        return {
            "region_description" if bool(value) else "global_caption"
            for value in use_region_tokens
        }

    def stream_complete(self, stream_name: str) -> bool:
        return bool(self.reports[str(stream_name)]["passed"])

    @property
    def complete(self) -> bool:
        return bool(self.reports) and all(
            value["passed"] is True for value in self.reports.values()
        )

    def audit_window(
        self,
        model: Any,
        optimizer: torch.optim.Optimizer,
        *,
        stream_name: str,
        stream_stage: str,
        observed_task_paths: set[str],
    ) -> dict[str, Any]:
        """Validate one real window and record only evidence it executed."""
        name = str(stream_name)
        paths = (
            set(observed_task_paths)
            if self.run_stage == "overfit" else {"stage"}
        )
        if self.run_stage == "overfit" and len(paths) != 1:
            raise ValueError(
                "D-1 gradient proof 要求 accumulation window 为单一 task path；"
                f"observed={sorted(paths)}"
            )
        gate = description_step_gradient_gate(
            model,
            optimizer,
            run_stage=self.run_stage,
            stream_name=name,
            stream_stage=str(stream_stage),
            observed_task_paths=(paths if self.run_stage == "overfit" else None),
        )
        if gate["passed"] is not True:
            return gate
        self.observed[name].update(paths)
        stream_report = self.reports[name]
        task_path = next(iter(paths))
        stream_report["path_reports"].setdefault(task_path, gate)
        stream_report["observed_task_paths"] = sorted(self.observed[name])
        stream_report["passed"] = (
            self.observed[name] >= self.required[name]
            and all(
                value.get("passed") is True
                for value in stream_report["path_reports"].values()
            )
        )
        return gate

    def payload(self) -> dict[str, Any]:
        checked = sorted(
            name for name in self.reports if self.stream_complete(name)
        )
        return {
            "protocol": DESCRIPTION_GRADIENT_GATE_PROTOCOL,
            "run_stage": self.run_stage,
            "required_streams": sorted(self.reports),
            "checked_streams": checked,
            "all_required_streams_checked": self.complete,
            "streams": self.reports,
            "passed": self.complete,
        }

    def restore_completed(self, payload: Any) -> None:
        """Restore a terminal proof only for report publication after step N."""
        if self.run_stage != "overfit":
            raise RuntimeError(
                "completed gradient replay 当前只用于 D-1 terminal recovery"
            )
        if not isinstance(payload, dict) or not d_minus_one_gradient_gate_passed(
            payload
        ):
            raise RuntimeError(
                "D-1 terminal checkpoint 缺少完整 task-path gradient proof"
            )
        if set(payload["streams"]) != set(self.reports):
            raise RuntimeError("D-1 checkpoint gradient stream inventory 漂移")
        self.reports = deepcopy(payload["streams"])
        self.observed = {
            name: set(report["observed_task_paths"])
            for name, report in self.reports.items()
        }
        if not self.complete:
            raise RuntimeError("D-1 checkpoint gradient proof 未完成")
