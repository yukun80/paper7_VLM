#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""M4, D4 and M6 gate publication workflows.

用途：集中重放科学门禁、候选自验证与原子发布，保持 CLI 仅负责参数解析。
推荐调用：由 ``qpsalm-segdesc validate m4|d4|m6`` 薄入口调用。
输入：已存在的评价、专家报告、Bridge gate 和 checkpoint-bound 报告路径。
输出：M4 suite、D4 curriculum 或 M6 acceptance JSON gate。
写入行为：只写显式 output；不运行模型、训练、benchmark 或 CUDA。
工作流阶段：M4/M6 scientific gate orchestration。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from qpsalm_seg.paths import resolve_project_path

from ..evaluation.comparison import (
    M4_BASELINE_REGION_ENCODERS,
    build_m4_region_encoder_suite,
    validate_m4_region_encoder_suite_gate,
)
from ..evaluation.d4_curriculum import (
    build_d4_curriculum_gate,
    validate_d4_curriculum_gate,
)
from ..evaluation.m6_acceptance import (
    build_m6_acceptance_gate,
    validate_m6_acceptance_gate,
)
from ..protocols.io import atomic_write_json


def parse_m4_gate_bindings(values: list[str]) -> dict[str, str]:
    """Parse and require exactly one gate for every non-MGRR baseline."""
    result: dict[str, str] = {}
    for value in values:
        encoder, separator, path = value.partition("=")
        if not separator or not encoder or not path:
            raise ValueError(f"--gate 必须是 ENCODER=PATH: {value!r}")
        if encoder not in M4_BASELINE_REGION_ENCODERS:
            raise ValueError(f"未知 M4 baseline encoder: {encoder!r}")
        if encoder in result:
            raise ValueError(f"M4 baseline gate 重复: {encoder}")
        result[encoder] = path
    if set(result) != M4_BASELINE_REGION_ENCODERS:
        raise ValueError(
            "M4 suite gate 集合不完整: "
            f"expected={sorted(M4_BASELINE_REGION_ENCODERS)} "
            f"observed={sorted(result)}"
        )
    return result


def _publish_revalidated_gate(
    report: dict[str, Any],
    *,
    output: str,
    validator: Callable[[Path], Any],
) -> dict[str, Any]:
    destination = resolve_project_path(output) or Path(output)
    candidate = destination.with_name(f".{destination.name}.candidate")
    try:
        atomic_write_json(candidate, report)
        validator(candidate)
        candidate.replace(destination)
    finally:
        candidate.unlink(missing_ok=True)
    return report


def run_m4_suite_gate(
    gate_bindings: list[str], *, output: str
) -> dict[str, Any]:
    report = build_m4_region_encoder_suite(
        parse_m4_gate_bindings(gate_bindings)
    )
    return _publish_revalidated_gate(
        report,
        output=output,
        validator=validate_m4_region_encoder_suite_gate,
    )


def run_d4_curriculum_gate(
    *,
    evaluation_dir: str,
    expert_report: str,
    bridge_benchmark: str,
    current_fraction: float,
    next_fraction: float | None,
    seed: int,
    m4_suite_gate: str | None,
    output: str,
) -> dict[str, Any]:
    report = build_d4_curriculum_gate(
        evaluation_dir=evaluation_dir,
        expert_report=expert_report,
        bridge_benchmark=bridge_benchmark,
        current_fraction=current_fraction,
        next_fraction=next_fraction,
        seed=seed,
        m4_suite_gate=m4_suite_gate,
    )
    return _publish_revalidated_gate(
        report,
        output=output,
        validator=validate_d4_curriculum_gate,
    )


def run_m6_acceptance_gate(
    *,
    gt_evaluation_dir: str,
    gt_expert_report: str,
    fixed_evaluation_dir: str,
    fixed_expert_report: str,
    end_to_end_evaluation_dir: str,
    end_to_end_expert_report: str,
    bridge_benchmark: str,
    d4_final_gate: str,
    seed: int,
    output: str,
) -> dict[str, Any]:
    report = build_m6_acceptance_gate(
        gt_evaluation_dir=gt_evaluation_dir,
        gt_expert_report=gt_expert_report,
        fixed_evaluation_dir=fixed_evaluation_dir,
        fixed_expert_report=fixed_expert_report,
        end_to_end_evaluation_dir=end_to_end_evaluation_dir,
        end_to_end_expert_report=end_to_end_expert_report,
        bridge_benchmark=bridge_benchmark,
        d4_final_gate=d4_final_gate,
        seed=seed,
    )
    return _publish_revalidated_gate(
        report,
        output=output,
        validator=validate_m6_acceptance_gate,
    )
