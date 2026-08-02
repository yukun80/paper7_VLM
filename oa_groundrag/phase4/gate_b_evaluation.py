"""Gate B 配对指标、bootstrap、判据和正式报告。"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from oa_groundrag.phase3.common import (
    canonical_json,
    first_symlink_component,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_text,
)
from oa_groundrag.phase3.dataset import RSGeneralDescDataset
from oa_groundrag.phase3.errors import RSGeneralDescError

from .artifacts import AtomicArtifactDirectory
from .contracts import (
    GATE_B_GENERATION_SCHEMA_VERSION,
    GATE_B_REPORT_SCHEMA_VERSION,
    PREDICTION_SCHEMA_VERSION,
)
from .errors import EvaluationError, Phase4Error, ReasonCode
from .gate_b_contracts import (
    GATE_B_OPEN_TASKS,
    GATE_B_PROTOCOL_ID,
    GATE_B_SAMPLE_COUNT,
    GATE_B_SEED,
    GATE_B_SHORT_TASKS,
    GATE_B_TASK_ORDER,
    QWEN_TEMPLATE_VERSION,
)
from .gate_b_selection import (
    GateBSelectionContext,
    load_gate_b_selection,
    selection_locations,
)


_PREDICTION_FIELDS = {
    "schema_version",
    "record_id",
    "parent_id",
    "logical_role",
    "task_family",
    "mask_mode",
    "generated_text",
    "model_output",
    "reference_responses",
    "evidence_ids",
    "provenance",
    "counterfactual",
}
_GENERATION_MANIFEST_FIELDS = {
    "schema_version",
    "status",
    "model_role",
    "protocol_id",
    "protocol_sha256",
    "selection_sha256",
    "selection_file_sha256",
    "benchmark_identity",
    "config_identity",
    "model_identity",
    "processor_identity",
    "checkpoint_identity",
    "generation",
    "ordered_record_ids_sha256",
    "predictions",
    "failures",
    "task_counts",
    "input_token_count",
    "image_count",
    "valid_for_evaluation",
}
_FILE_IDENTITY_FIELDS = {"path", "count", "sha256"}
_GATE_PROVENANCE_FIELDS = {
    "protocol_id",
    "protocol_sha256",
    "selection_sha256",
    "ordinal",
    "model_role",
    "source",
    "shard_path",
    "line_index",
    "template_version",
}


@dataclass(frozen=True)
class GateBEvaluationOutcome:
    root: Path
    status: str
    gate_b_passed: bool


def normalize_gate_b_text(value: str) -> str:
    return " ".join(re.findall(r"[\w]+", value.lower(), flags=re.UNICODE))


def gate_b_token_f1(prediction: str, reference: str) -> float:
    predicted = normalize_gate_b_text(prediction).split()
    expected = normalize_gate_b_text(reference).split()
    if not predicted or not expected:
        return float(predicted == expected)
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 2.0 * precision * recall / (precision + recall)


def gate_b_rouge_l_f1(prediction: str, reference: str) -> float:
    predicted = normalize_gate_b_text(prediction).split()
    expected = normalize_gate_b_text(reference).split()
    if not predicted or not expected:
        return float(predicted == expected)
    previous = [0] * (len(expected) + 1)
    for predicted_token in predicted:
        current = [0]
        for index, expected_token in enumerate(expected, 1):
            if predicted_token == expected_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    overlap = previous[-1]
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 2.0 * precision * recall / (precision + recall)


def score_gate_b_text(
    prediction: str,
    references: Sequence[str],
    *,
    task_family: str,
) -> dict[str, float]:
    if (
        not isinstance(prediction, str)
        or not prediction.strip()
        or not references
        or not all(isinstance(value, str) and value.strip() for value in references)
    ):
        raise EvaluationError(
            ReasonCode.PREDICTION_INVALID,
            "Gate B prediction/references 不能为空",
        )
    normalized_prediction = normalize_gate_b_text(prediction)
    normalized_references = {
        normalize_gate_b_text(reference) for reference in references
    }
    exact = float(normalized_prediction in normalized_references)
    token_f1 = max(
        gate_b_token_f1(prediction, reference) for reference in references
    )
    rouge_l = max(
        gate_b_rouge_l_f1(prediction, reference) for reference in references
    )
    if task_family in GATE_B_OPEN_TASKS:
        primary = (token_f1 + rouge_l) / 2.0
    elif task_family in GATE_B_SHORT_TASKS:
        primary = (exact + token_f1) / 2.0
    else:
        raise EvaluationError(
            ReasonCode.GATE_B_RUN_INVALID,
            f"Gate B 未注册 task={task_family}",
        )
    return {
        "primary": primary,
        "normalized_exact_match": exact,
        "best_reference_token_f1": token_f1,
        "best_reference_rouge_l_f1": rouge_l,
    }


def task_stratified_paired_bootstrap(
    deltas_by_task: Mapping[str, Sequence[float]],
    *,
    iterations: int = 10_000,
    seed: int = GATE_B_SEED,
) -> dict[str, float | int]:
    if (
        set(deltas_by_task) != set(GATE_B_TASK_ORDER)
        or isinstance(iterations, bool)
        or not isinstance(iterations, int)
        or iterations <= 0
        or isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed < 0
    ):
        raise EvaluationError(
            ReasonCode.GATE_B_RUN_INVALID,
            "bootstrap task/iterations/seed 合同非法",
        )
    arrays: dict[str, np.ndarray] = {}
    for task in GATE_B_TASK_ORDER:
        values = np.asarray(deltas_by_task[task], dtype=np.float64)
        if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
            raise EvaluationError(
                ReasonCode.NONFINITE_NUMBER,
                f"bootstrap task={task} 数据非法",
            )
        arrays[task] = values
    rng = np.random.Generator(np.random.PCG64(seed))
    distribution = np.empty(iterations, dtype=np.float64)
    for iteration in range(iterations):
        task_means = []
        for task in GATE_B_TASK_ORDER:
            values = arrays[task]
            indices = rng.integers(0, values.size, size=values.size)
            task_means.append(float(values[indices].mean()))
        distribution[iteration] = float(np.mean(task_means))
    lower, upper = np.quantile(
        distribution,
        [0.025, 0.975],
        method="linear",
    )
    return {
        "iterations": iterations,
        "seed": seed,
        "confidence_level": 0.95,
        "lower": float(lower),
        "upper": float(upper),
    }


def evaluate_gate_b_criteria(
    *,
    bootstrap_lower: float,
    per_task_deltas: Mapping[str, float],
    per_source_deltas: Mapping[str, float],
    open_rouge_l_delta: float,
    short_exact_delta: float,
) -> tuple[list[dict[str, Any]], bool]:
    if set(per_task_deltas) != set(GATE_B_TASK_ORDER) or not per_source_deltas:
        raise EvaluationError(
            ReasonCode.GATE_B_RUN_INVALID,
            "Gate B criteria 输入缺少 task/source",
        )
    values = [
        bootstrap_lower,
        *per_task_deltas.values(),
        *per_source_deltas.values(),
        open_rouge_l_delta,
        short_exact_delta,
    ]
    if any(not math.isfinite(float(value)) for value in values):
        raise EvaluationError(
            ReasonCode.NONFINITE_NUMBER,
            "Gate B criteria 不接受非有限数",
        )
    improved_tasks = sum(value > 0.0 for value in per_task_deltas.values())
    minimum_task_delta = min(per_task_deltas.values())
    minimum_source_delta = min(per_source_deltas.values())
    specifications = (
        (
            "primary_macro_bootstrap_ci",
            bootstrap_lower > 0.0,
            bootstrap_lower,
            ">",
            0.0,
        ),
        (
            "minimum_improved_tasks",
            improved_tasks >= 4,
            improved_tasks,
            ">=",
            4,
        ),
        (
            "task_primary_delta_floor",
            minimum_task_delta >= -0.02,
            minimum_task_delta,
            ">=",
            -0.02,
        ),
        (
            "source_primary_delta_floor",
            minimum_source_delta >= -0.02,
            minimum_source_delta,
            ">=",
            -0.02,
        ),
        (
            "open_generation_rouge_l_delta",
            open_rouge_l_delta >= 0.0,
            open_rouge_l_delta,
            ">=",
            0.0,
        ),
        (
            "short_answer_exact_match_delta",
            short_exact_delta >= 0.0,
            short_exact_delta,
            ">=",
            0.0,
        ),
    )
    criteria = [
        {
            "criterion": name,
            "observed": observed,
            "operator": operator,
            "threshold": threshold,
            "status": "PASS" if passed else "FAIL",
        }
        for name, passed, observed, operator, threshold in specifications
    ]
    return criteria, all(item[1] for item in specifications)


def _regular_file(path: Path, *, location: str) -> Path:
    linked = first_symlink_component(path)
    if linked is not None:
        raise EvaluationError(
            ReasonCode.OUTPUT_LINK,
            f"{location}: 路径含 symlink",
            details={"path": str(linked)},
        )
    if not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1:
        raise EvaluationError(
            ReasonCode.OUTPUT_LINK,
            f"{location}: 必须是普通单链接文件",
            details={"path": str(path)},
        )
    return path


def _read_mapping(path: Path, *, location: str) -> dict[str, Any]:
    try:
        value = read_json(_regular_file(path, location=location))
    except RSGeneralDescError as error:
        raise EvaluationError(
            ReasonCode.GATE_B_RUN_INVALID,
            f"{location}: 无法严格读取 JSON",
            details={"error": str(error)},
        ) from error
    if not isinstance(value, dict):
        raise EvaluationError(
            ReasonCode.GATE_B_RUN_INVALID,
            f"{location}: 必须是对象",
        )
    return value


def _load_generation_run(
    root: Path,
    *,
    model_role: str,
    context: GateBSelectionContext,
) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    root = Path(root)
    linked = first_symlink_component(root)
    if linked is not None or root.is_symlink() or not root.is_dir():
        raise EvaluationError(
            ReasonCode.OUTPUT_LINK,
            f"{model_role} generation root 不存在或含链接",
        )
    manifest_path = _regular_file(
        root / "generation_manifest.json",
        location=f"{model_role}.generation_manifest",
    )
    manifest = _read_mapping(
        manifest_path,
        location=f"{model_role}.generation_manifest",
    )
    if set(manifest) != _GENERATION_MANIFEST_FIELDS:
        raise EvaluationError(
            ReasonCode.GATE_B_RUN_INVALID,
            f"{model_role} generation manifest 字段不匹配",
        )
    selection = context.selection
    frozen = context.frozen_protocol
    expected_config = (
        context.protocol_source.base_config
        if model_role == "base"
        else context.protocol_source.adapter_config
    )
    expected_checkpoint = (
        None
        if model_role == "base"
        else frozen["training_run"]["best_checkpoint"]
    )
    expected_record_sha = sha256_text(
        canonical_json([item["record_id"] for item in selection["items"]])
    )
    if (
        manifest.get("schema_version") != GATE_B_GENERATION_SCHEMA_VERSION
        or manifest.get("model_role") != model_role
        or manifest.get("protocol_id") != GATE_B_PROTOCOL_ID
        or manifest.get("protocol_sha256") != frozen["protocol_sha256"]
        or manifest.get("selection_sha256") != selection["selection_sha256"]
    ):
        raise EvaluationError(
            ReasonCode.GATE_B_RUN_INVALID,
            f"{model_role} generation protocol/selection role identity 不匹配",
        )
    if not isinstance(manifest.get("selection_file_sha256"), str):
        raise EvaluationError(
            ReasonCode.GATE_B_RUN_INVALID,
            f"{model_role} generation 缺少 selection file SHA",
        )
    if (
        manifest.get("benchmark_identity") != context.access.identity.to_dict()
        or manifest.get("model_identity")
        != frozen["static_protocol"]["model_identity"]
        or manifest.get("processor_identity")
        != frozen["static_protocol"]["processor_identity"]
        or manifest.get("checkpoint_identity") != expected_checkpoint
        or manifest.get("generation") != frozen["static_protocol"]["generation"]
        or manifest.get("ordered_record_ids_sha256") != expected_record_sha
    ):
        raise EvaluationError(
            ReasonCode.GATE_B_RUN_INVALID,
            f"{model_role} generation Benchmark/model/checkpoint identity 不匹配",
        )
    config_identity = manifest.get("config_identity")
    if (
        not isinstance(config_identity, dict)
        or set(config_identity)
        != {"path", "file_sha256", "semantic_sha256", "adaptation"}
        or config_identity.get("path") != str(expected_config.config_path.resolve())
        or config_identity.get("file_sha256") != sha256_file(expected_config.config_path)
        or config_identity.get("semantic_sha256") != expected_config.semantic_sha256
        or config_identity.get("adaptation") != expected_config.adaptation.strategy
    ):
        raise EvaluationError(
            ReasonCode.GATE_B_RUN_INVALID,
            f"{model_role} generation config identity 不匹配",
        )
    prediction_contract = manifest.get("predictions")
    failure_contract = manifest.get("failures")
    if (
        not isinstance(prediction_contract, dict)
        or set(prediction_contract) != _FILE_IDENTITY_FIELDS
        or not isinstance(failure_contract, dict)
        or set(failure_contract) != _FILE_IDENTITY_FIELDS
    ):
        raise EvaluationError(
            ReasonCode.GATE_B_RUN_INVALID,
            f"{model_role} generation file identity 字段非法",
        )
    prediction_path = _regular_file(
        root / str(prediction_contract["path"]),
        location=f"{model_role}.predictions",
    )
    failure_path = _regular_file(
        root / str(failure_contract["path"]),
        location=f"{model_role}.failures",
    )
    if (
        prediction_contract["path"] != "predictions.jsonl"
        or failure_contract["path"] != "failures.jsonl"
        or sha256_file(prediction_path) != prediction_contract["sha256"]
        or sha256_file(failure_path) != failure_contract["sha256"]
    ):
        raise EvaluationError(
            ReasonCode.GATE_B_RUN_INVALID,
            f"{model_role} prediction/failure SHA 不匹配",
        )
    try:
        predictions = read_jsonl(prediction_path)
        failures = read_jsonl(failure_path)
    except RSGeneralDescError as error:
        raise EvaluationError(
            ReasonCode.GATE_B_RUN_INVALID,
            f"{model_role} prediction/failure JSONL 非法",
            details={"error": str(error)},
        ) from error
    if (
        prediction_contract["count"] != len(predictions)
        or failure_contract["count"] != len(failures)
        or failures
        or len(predictions) != GATE_B_SAMPLE_COUNT
        or manifest.get("status") != "completed"
        or manifest.get("valid_for_evaluation") is not True
    ):
        raise EvaluationError(
            ReasonCode.GATE_B_RUN_INVALID,
            f"{model_role} generation 数量/failure/status 使 Gate B 无效",
            details={
                "prediction_count": len(predictions),
                "failure_count": len(failures),
                "status": manifest.get("status"),
            },
        )
    if (
        isinstance(manifest.get("input_token_count"), bool)
        or not isinstance(manifest.get("input_token_count"), int)
        or manifest["input_token_count"] < 0
        or isinstance(manifest.get("image_count"), bool)
        or not isinstance(manifest.get("image_count"), int)
        or manifest["image_count"] < 0
        or manifest.get("task_counts") != selection["task_counts"]
    ):
        raise EvaluationError(
            ReasonCode.GATE_B_RUN_INVALID,
            f"{model_role} generation 派生计数与 selection 不一致",
        )
    return manifest, predictions, sha256_file(manifest_path)


def _validate_prediction_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    model_role: str,
    context: GateBSelectionContext,
    canonical_records: Sequence[Mapping[str, Any]],
) -> None:
    expected_count = len(context.selection["items"])
    if len(rows) != expected_count or len(canonical_records) != expected_count:
        raise EvaluationError(
            ReasonCode.GATE_B_RUN_INVALID,
            f"{model_role} prediction/selection/Benchmark 数量不一致",
            details={
                "predictions": len(rows),
                "selection": expected_count,
                "canonical_records": len(canonical_records),
            },
        )
    for ordinal, (row, item, record) in enumerate(
        zip(rows, context.selection["items"], canonical_records, strict=True)
    ):
        if set(row) != _PREDICTION_FIELDS:
            raise EvaluationError(
                ReasonCode.PREDICTION_INVALID,
                f"{model_role} prediction 字段不匹配",
                details={"ordinal": ordinal},
            )
        references = row.get("reference_responses")
        if (
            row.get("schema_version") != PREDICTION_SCHEMA_VERSION
            or row.get("record_id") != item["record_id"]
            or row.get("parent_id") != item["parent_id"]
            or row.get("logical_role") != "external_val"
            or row.get("task_family") != item["task_family"]
            or row.get("mask_mode") != "external_generic"
            or not isinstance(row.get("generated_text"), str)
            or not row["generated_text"].strip()
            or row.get("model_output") is not None
            or row.get("evidence_ids") != []
            or row.get("counterfactual") is not None
            or references != record["reference_responses"]
        ):
            raise EvaluationError(
                ReasonCode.PREDICTION_INVALID,
                f"{model_role} prediction 与 selection/Benchmark 不一致",
                details={"ordinal": ordinal},
            )
        provenance = row.get("provenance")
        if (
            not isinstance(provenance, dict)
            or set(provenance)
            != {
                "canonical_build_id",
                "canonical_payload_sha256",
                "renderer",
                "gate_b",
            }
            or provenance["canonical_build_id"] != context.access.identity.build_id
            or provenance["canonical_payload_sha256"]
            != context.access.identity.payload_sha256
            or provenance["renderer"] != "phase3.render_canonical_messages"
        ):
            raise EvaluationError(
                ReasonCode.PREDICTION_INVALID,
                f"{model_role} prediction provenance 非法",
                details={"ordinal": ordinal},
            )
        gate = provenance["gate_b"]
        expected_gate = {
            "protocol_id": GATE_B_PROTOCOL_ID,
            "protocol_sha256": context.frozen_protocol["protocol_sha256"],
            "selection_sha256": context.selection["selection_sha256"],
            "ordinal": ordinal,
            "model_role": model_role,
            "source": item["source"],
            "shard_path": item["shard_path"],
            "line_index": item["line_index"],
            "template_version": QWEN_TEMPLATE_VERSION,
        }
        if not isinstance(gate, dict) or set(gate) != _GATE_PROVENANCE_FIELDS or gate != expected_gate:
            raise EvaluationError(
                ReasonCode.PREDICTION_INVALID,
                f"{model_role} prediction Gate B provenance 不匹配",
                details={"ordinal": ordinal},
            )


def _mean_metrics(values: Sequence[Mapping[str, float]]) -> dict[str, float]:
    if not values:
        raise EvaluationError(
            ReasonCode.GATE_B_RUN_INVALID,
            "指标分组不能为空",
        )
    names = {
        "primary",
        "normalized_exact_match",
        "best_reference_token_f1",
        "best_reference_rouge_l_f1",
    }
    return {
        name: sum(value[name] for value in values) / len(values)
        for name in sorted(names)
    }


def _comparison(base: Mapping[str, float], adapter: Mapping[str, float]) -> dict[str, Any]:
    return {
        "base": dict(base),
        "adapter": dict(adapter),
        "delta": {
            name: adapter[name] - base[name]
            for name in sorted(base)
        },
    }


def _completed_metrics(
    context: GateBSelectionContext,
    base_rows: Sequence[Mapping[str, Any]],
    adapter_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], list[dict[str, Any]], bool]:
    paired: list[dict[str, Any]] = []
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_cell: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    deltas_by_task: dict[str, list[float]] = defaultdict(list)
    for item, base_row, adapter_row in zip(
        context.selection["items"],
        base_rows,
        adapter_rows,
        strict=True,
    ):
        base_score = score_gate_b_text(
            str(base_row["generated_text"]),
            tuple(base_row["reference_responses"]),
            task_family=item["task_family"],
        )
        adapter_score = score_gate_b_text(
            str(adapter_row["generated_text"]),
            tuple(adapter_row["reference_responses"]),
            task_family=item["task_family"],
        )
        delta = {
            name: adapter_score[name] - base_score[name]
            for name in sorted(base_score)
        }
        row = {
            "schema_version": GATE_B_REPORT_SCHEMA_VERSION,
            "artifact_kind": "paired_score",
            "ordinal": item["ordinal"],
            "record_id": item["record_id"],
            "parent_id": item["parent_id"],
            "source": item["source"],
            "task_family": item["task_family"],
            "base": base_score,
            "adapter": adapter_score,
            "delta": delta,
        }
        paired.append(row)
        by_task[item["task_family"]].append(row)
        by_cell[(item["source"], item["task_family"])].append(row)
        deltas_by_task[item["task_family"]].append(delta["primary"])

    per_task: dict[str, Any] = {}
    for task in GATE_B_TASK_ORDER:
        values = by_task[task]
        base = _mean_metrics([value["base"] for value in values])
        adapter = _mean_metrics([value["adapter"] for value in values])
        per_task[task] = {"count": len(values), **_comparison(base, adapter)}
    per_source_task: dict[str, dict[str, Any]] = defaultdict(dict)
    for (source, task), values in sorted(by_cell.items()):
        base = _mean_metrics([value["base"] for value in values])
        adapter = _mean_metrics([value["adapter"] for value in values])
        per_source_task[source][task] = {
            "count": len(values),
            **_comparison(base, adapter),
        }
    per_source: dict[str, Any] = {}
    for source, cells in sorted(per_source_task.items()):
        base = _mean_metrics([value["base"] for value in cells.values()])
        adapter = _mean_metrics([value["adapter"] for value in cells.values()])
        per_source[source] = {
            "task_count": len(cells),
            "record_count": sum(value["count"] for value in cells.values()),
            **_comparison(base, adapter),
        }
    overall_base = _mean_metrics([per_task[task]["base"] for task in GATE_B_TASK_ORDER])
    overall_adapter = _mean_metrics([per_task[task]["adapter"] for task in GATE_B_TASK_ORDER])
    overall = _comparison(overall_base, overall_adapter)
    open_base = sum(
        per_task[task]["base"]["best_reference_rouge_l_f1"]
        for task in GATE_B_OPEN_TASKS
    ) / len(GATE_B_OPEN_TASKS)
    open_adapter = sum(
        per_task[task]["adapter"]["best_reference_rouge_l_f1"]
        for task in GATE_B_OPEN_TASKS
    ) / len(GATE_B_OPEN_TASKS)
    short_base = sum(
        per_task[task]["base"]["normalized_exact_match"]
        for task in GATE_B_SHORT_TASKS
    ) / len(GATE_B_SHORT_TASKS)
    short_adapter = sum(
        per_task[task]["adapter"]["normalized_exact_match"]
        for task in GATE_B_SHORT_TASKS
    ) / len(GATE_B_SHORT_TASKS)
    bootstrap = task_stratified_paired_bootstrap(deltas_by_task)
    criteria, passed = evaluate_gate_b_criteria(
        bootstrap_lower=float(bootstrap["lower"]),
        per_task_deltas={
            task: per_task[task]["delta"]["primary"]
            for task in GATE_B_TASK_ORDER
        },
        per_source_deltas={
            source: value["delta"]["primary"]
            for source, value in per_source.items()
        },
        open_rouge_l_delta=open_adapter - open_base,
        short_exact_delta=short_adapter - short_base,
    )
    metrics = {
        "overall_task_macro": overall,
        "open_generation_macro_rouge_l": {
            "base": open_base,
            "adapter": open_adapter,
            "delta": open_adapter - open_base,
        },
        "short_answer_macro_normalized_exact_match": {
            "base": short_base,
            "adapter": short_adapter,
            "delta": short_adapter - short_base,
        },
        "per_task": per_task,
        "per_source": per_source,
        "per_source_task": dict(per_source_task),
    }
    return paired, metrics, bootstrap, criteria, passed


def _not_evaluated_criteria() -> list[dict[str, Any]]:
    return [
        {
            "criterion": name,
            "observed": None,
            "operator": operator,
            "threshold": threshold,
            "status": "NOT_EVALUATED",
        }
        for name, operator, threshold in (
            ("primary_macro_bootstrap_ci", ">", 0.0),
            ("minimum_improved_tasks", ">=", 4),
            ("task_primary_delta_floor", ">=", -0.02),
            ("source_primary_delta_floor", ">=", -0.02),
            ("open_generation_rouge_l_delta", ">=", 0.0),
            ("short_answer_exact_match_delta", ">=", 0.0),
        )
    ]


def evaluate_gate_b(
    protocol_path: Path | str,
    selection_path: Path | str,
    *,
    base_run: Path,
    adapter_run: Path,
    output_root: Path,
) -> GateBEvaluationOutcome:
    with AtomicArtifactDirectory(Path(output_root)) as writer:
        paired: list[dict[str, Any]] = []
        context: GateBSelectionContext | None = None
        base_manifest: Mapping[str, Any] | None = None
        adapter_manifest: Mapping[str, Any] | None = None
        base_manifest_sha: str | None = None
        adapter_manifest_sha: str | None = None
        invalid_reasons: list[dict[str, Any]] = []
        try:
            context = load_gate_b_selection(protocol_path, selection_path)
            expected_selection_file_sha = sha256_file(Path(selection_path))
            base_manifest, base_rows, base_manifest_sha = _load_generation_run(
                Path(base_run),
                model_role="base",
                context=context,
            )
            adapter_manifest, adapter_rows, adapter_manifest_sha = _load_generation_run(
                Path(adapter_run),
                model_role="adapter",
                context=context,
            )
            if (
                base_manifest["selection_file_sha256"] != expected_selection_file_sha
                or adapter_manifest["selection_file_sha256"] != expected_selection_file_sha
            ):
                raise EvaluationError(
                    ReasonCode.GATE_B_RUN_INVALID,
                    "Base/Adapter generation selection file SHA 不匹配",
                )
            canonical = RSGeneralDescDataset.from_locations(
                context.protocol_source.base_config.data.benchmark_root,
                selection_locations(context.selection),
                roles=("external_val",),
                task_families=GATE_B_TASK_ORDER,
                load_assets=False,
                seed=GATE_B_SEED,
                expected_manifest_sha256=(
                    context.protocol_source.base_config.data.expected_manifest_sha256
                ),
                verifier=context.access.verifier,
            )
            _validate_prediction_rows(
                base_rows,
                model_role="base",
                context=context,
                canonical_records=canonical.records,
            )
            _validate_prediction_rows(
                adapter_rows,
                model_role="adapter",
                context=context,
                canonical_records=canonical.records,
            )
            paired, metrics, bootstrap, criteria, passed = _completed_metrics(
                context,
                base_rows,
                adapter_rows,
            )
            status = "completed"
            evaluated = True
        except (Phase4Error, RSGeneralDescError) as error:
            invalid_reasons.append(
                {
                    "reason_code": error.code.value,
                    "message": str(error),
                    "details": dict(error.details),
                }
            )
            metrics = None
            bootstrap = {
                "iterations": 10_000,
                "seed": GATE_B_SEED,
                "confidence_level": 0.95,
                "lower": None,
                "upper": None,
            }
            criteria = _not_evaluated_criteria()
            passed = False
            status = "invalid"
            evaluated = False
        writer.write_jsonl("paired_scores.jsonl", paired)
        paired_path = writer.path("paired_scores.jsonl")
        report = {
            "schema_version": GATE_B_REPORT_SCHEMA_VERSION,
            "status": status,
            "gate_b_evaluated": evaluated,
            "gate_b_passed": bool(evaluated and passed),
            "formal_acceptance": bool(evaluated and passed),
            "adapter_status": (
                "accepted"
                if evaluated and passed
                else ("not_accepted" if evaluated else "not_evaluated")
            ),
            "protocol_identity": (
                None
                if context is None
                else {
                    "protocol_id": GATE_B_PROTOCOL_ID,
                    "protocol_sha256": context.frozen_protocol["protocol_sha256"],
                }
            ),
            "selection_identity": (
                None
                if context is None
                else {
                    "schema_version": context.selection["schema_version"],
                    "selection_sha256": context.selection["selection_sha256"],
                    "selection_file_sha256": sha256_file(Path(selection_path)),
                    "sample_count": context.selection["sample_count"],
                    "parent_count": context.selection["parent_count"],
                    "monitoring_exclusion": context.selection["monitoring_exclusion"],
                }
            ),
            "benchmark_identity": (
                None if context is None else context.access.identity.to_dict()
            ),
            "base_generation": (
                None
                if base_manifest is None
                else {
                    "manifest_sha256": base_manifest_sha,
                    "prediction_sha256": base_manifest["predictions"]["sha256"],
                    "prediction_count": base_manifest["predictions"]["count"],
                    "failure_count": base_manifest["failures"]["count"],
                    "config_identity": base_manifest["config_identity"],
                    "model_identity": base_manifest["model_identity"],
                    "processor_identity": base_manifest["processor_identity"],
                }
            ),
            "adapter_generation": (
                None
                if adapter_manifest is None
                else {
                    "manifest_sha256": adapter_manifest_sha,
                    "prediction_sha256": adapter_manifest["predictions"]["sha256"],
                    "prediction_count": adapter_manifest["predictions"]["count"],
                    "failure_count": adapter_manifest["failures"]["count"],
                    "config_identity": adapter_manifest["config_identity"],
                    "model_identity": adapter_manifest["model_identity"],
                    "processor_identity": adapter_manifest["processor_identity"],
                    "checkpoint_identity": adapter_manifest["checkpoint_identity"],
                }
            ),
            "pairing": {
                "complete": evaluated,
                "paired_count": len(paired),
                "expected_count": GATE_B_SAMPLE_COUNT,
                "order": "frozen_selection_order",
                "paired_scores_path": "paired_scores.jsonl",
                "paired_scores_sha256": sha256_file(paired_path),
            },
            "metrics": metrics,
            "bootstrap": bootstrap,
            "criteria": criteria,
            "invalid_reasons": invalid_reasons,
        }
        writer.write_json("gate_b_report.json", report)
        target = writer.publish()
    return GateBEvaluationOutcome(
        root=target,
        status=status,
        gate_b_passed=bool(evaluated and passed),
    )
