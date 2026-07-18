#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""M6 language metrics and blind-review workflows.

用途：集中 RSIEval caption metrics、caption 双审与 expert ERFS 的输出发布。
推荐调用：由 score_caption_metrics/score_caption_human_review/
score_expert_factuality 薄入口调用。
输入：冻结 raw generations、本地 BERTScore model 或独立人工 review JSONL。
输出：source-bound metric/review templates/aggregate reports。
写入行为：只写显式 output，不修改 generation、review 或 benchmark。
工作流阶段：M6 secondary caption evaluation and primary ERFS orchestration。
"""

from __future__ import annotations

from pathlib import Path
import traceback
from typing import Any

from qpsalm_seg.paths import resolve_project_path

from ..evaluation.caption_human_review import (
    aggregate_caption_human_reviews,
    build_caption_human_review_template,
    write_caption_review_jsonl,
)
from ..evaluation.caption_metrics import (
    score_caption_metrics,
    write_caption_metric_report,
)
from ..evaluation.expert_factuality import (
    aggregate_expert_factuality,
    build_expert_review_template,
)
from ..protocols.io import atomic_write_json, atomic_write_jsonl


class ReviewLaunchError(ValueError):
    """The requested review output ownership or reviewer set is invalid."""


def _output(path_ref: str, *, overwrite: bool) -> Path:
    path = resolve_project_path(path_ref) or Path(path_ref)
    if path.exists():
        if path.is_dir():
            raise ReviewLaunchError(f"review/metric output 不能是目录: {path}")
        if not overwrite:
            raise ReviewLaunchError(
                f"review/metric output 已存在；请改路径或显式覆盖: {path}"
            )
        path.unlink()
    return path


def run_caption_metric_scoring(
    *,
    eval_dir: str,
    bertscore_model: str,
    bertscore_num_layers: int,
    bertscore_batch_size: int,
    device: str,
    seed: int,
    output: str,
    overwrite_output: bool,
) -> dict[str, Any]:
    destination = _output(output, overwrite=overwrite_output)
    try:
        report = score_caption_metrics(
            eval_dir,
            bertscore_model=bertscore_model,
            bertscore_num_layers=bertscore_num_layers,
            bertscore_batch_size=bertscore_batch_size,
            device=device,
            seed=seed,
        )
    except BaseException as exc:
        write_caption_metric_report(
            destination.with_name(destination.stem + ".failure.json"),
            {
                "protocol": "qpsalm_rsieval_caption_metrics_failure_v1",
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        raise
    write_caption_metric_report(destination, report)
    return report


def run_caption_human_review(
    *,
    eval_dir: str,
    reviews: list[str],
    minimum_reviewers: int,
    seed: int,
    write_template: bool,
    output: str,
    overwrite_output: bool,
) -> dict[str, Any]:
    if write_template:
        if reviews:
            raise ReviewLaunchError(
                "--write-template 与 --review 不能同时使用"
            )
        destination = _output(output, overwrite=overwrite_output)
        rows = build_caption_human_review_template(eval_dir)
        write_caption_review_jsonl(destination, rows)
        return {
            "template": str(destination),
            "num_samples": len(rows),
            "reference_target_hidden": True,
        }
    if len(reviews) < max(2, int(minimum_reviewers)):
        raise ReviewLaunchError(
            "正式 caption human review 至少提供两份独立审核文件"
        )
    destination = _output(output, overwrite=overwrite_output)
    report = aggregate_caption_human_reviews(
        eval_dir,
        reviews,
        seed=seed,
        minimum_reviewers=minimum_reviewers,
    )
    atomic_write_json(destination, report)
    return report


def run_expert_factuality_review(
    *,
    eval_dir: str,
    reviews: list[str],
    minimum_reviewers: int,
    seed: int,
    write_template: bool,
    output: str,
    overwrite_output: bool,
) -> dict[str, Any]:
    if write_template:
        if reviews:
            raise ReviewLaunchError(
                "--write-template 与 --review 不能同时使用"
            )
        destination = _output(output, overwrite=overwrite_output)
        rows = build_expert_review_template(eval_dir)
        atomic_write_jsonl(destination, rows)
        return {"template": str(destination), "num_samples": len(rows)}
    if len(reviews) < minimum_reviewers:
        raise ReviewLaunchError(
            "正式 ERFS 至少提供 minimum-reviewers 份独立审核文件"
        )
    destination = _output(output, overwrite=overwrite_output)
    report = aggregate_expert_factuality(
        eval_dir,
        reviews,
        seed=seed,
        minimum_reviewers=minimum_reviewers,
    )
    atomic_write_json(destination, report)
    return report
