"""Gate B 协议、训练闭环与冻结身份合同。"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from oa_groundrag.phase3.common import (
    canonical_json,
    first_symlink_component,
    portable_relative_path,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_text,
)
from oa_groundrag.phase3.errors import RSGeneralDescError

from .checkpoint import CheckpointManager
from .config import Phase4Config, _load_yaml, load_config
from .contracts import (
    GATE_B_PROTOCOL_SCHEMA_VERSION,
)
from .errors import ContractError, Phase4Error, ReasonCode
from .model import (
    EXPECTED_ATTENTION_LORA_R8_PARAMETERS,
    local_model_identity,
)
from .preflight import BenchmarkAccess, open_benchmark_access
from .processing import Qwen3VLProcessorAdapter
from .trainer import (
    BEST_CHECKPOINT_SCHEMA_VERSION,
    TRAINING_REPORT_SCHEMA_VERSION,
    _validation_from_row,
    training_layout_identity,
    validation_is_better,
)
from .validation import VALIDATION_SELECTION_SCHEMA_VERSION


GATE_B_PROTOCOL_ID = "rs_generaldesc_gate_b_qwen3vl_2b_v1"
GATE_B_SELECTION_ALGORITHM = (
    "external_val_parent_source_task_waterfill.v1"
)
GATE_B_SEED = 20260802
GATE_B_SAMPLE_COUNT = 256
GATE_B_TASK_ORDER = (
    "bbox_region_caption",
    "global_caption",
    "object_count",
    "scene_understanding",
    "spatial_relation",
    "visible_change_report",
    "visual_qa",
)
GATE_B_OPEN_TASKS = (
    "bbox_region_caption",
    "global_caption",
    "visible_change_report",
)
GATE_B_SHORT_TASKS = (
    "visual_qa",
    "object_count",
    "scene_understanding",
    "spatial_relation",
)
QWEN_TEMPLATE_VERSION = "qwen3vl_messages.v2"


_PROTOCOL_FIELDS = {
    "schema_version",
    "protocol_id",
    "benchmark",
    "base",
    "adapter",
    "training",
    "model_identity",
    "processor_identity",
    "selection",
    "generation",
    "evaluation",
}
_BENCHMARK_FIELDS = {
    "root",
    "manifest_schema",
    "canonical_schema",
    "manifest_sha256",
    "validation_sha256",
    "build_id",
    "payload_sha256",
    "hash_manifest_sha256",
}
_CONFIG_FIELDS = {
    "config",
    "config_file_sha256",
    "config_semantic_sha256",
}
_TRAINING_FIELDS = {
    "training_report_sha256",
    "training_manifest_sha256",
    "config_snapshot_sha256",
    "best_checkpoint_sha256",
    "monitoring_selection_file_sha256",
    "monitoring_selection_sha256",
    "validation_results_sha256",
    "checkpoint_step",
    "checkpoint_manifest_sha256",
    "adapter_size_bytes",
    "adapter_sha256",
}
_MODEL_IDENTITY_FIELDS = {
    "model_path",
    "model_type",
    "config_sha256",
    "weights_metadata_sha256",
    "processor_config_sha256",
    "base_parameter_count",
}
_PROCESSOR_IDENTITY_FIELDS = {
    "processor_path",
    "files",
    "processor_class",
    "tokenizer_class",
    "pad_token_id",
}
_SELECTION_FIELDS = {
    "algorithm",
    "role",
    "sample_count",
    "seed",
    "task_order",
}
_GENERATION_FIELDS = {
    "seed",
    "max_new_tokens",
    "do_sample",
    "temperature",
    "top_p",
    "template_version",
}
_EVALUATION_FIELDS = {
    "open_tasks",
    "short_tasks",
    "text_normalization",
    "token_f1",
    "rouge_l",
    "task_macro_weighting",
    "source_macro_weighting",
    "bootstrap",
    "criteria",
}
_BOOTSTRAP_FIELDS = {
    "iterations",
    "seed",
    "confidence_level",
    "rng",
    "quantile_method",
}
_CRITERIA_FIELDS = {
    "primary_ci_lower_strictly_greater_than",
    "minimum_improved_tasks",
    "task_delta_floor",
    "source_delta_floor",
    "open_rouge_l_delta_floor",
    "short_exact_match_delta_floor",
}
_TRAINING_REPORT_FIELDS = {
    "schema_version",
    "status",
    "formal_acceptance",
    "config_semantic_sha256",
    "benchmark",
    "validation_selection",
    "training_layout",
    "cursor",
    "max_steps",
    "execution_stop_step",
    "last_loss",
    "ema_loss",
    "session_loss_history",
    "samples",
    "input_tokens",
    "supervised_tokens",
    "images",
    "run_elapsed_seconds",
    "cuda_peak_gib",
    "throughput",
    "last_gradient_norm",
    "checkpoint",
    "best_checkpoint",
    "validation_history",
    "artifacts",
}
_TRAINING_MANIFEST_FIELDS = {
    "schema_version",
    "run_kind",
    "config_semantic_sha256",
    "benchmark",
    "validation_selection",
    "model",
    "processor",
    "adaptation",
    "training_layout",
    "trainable_parameter_count",
    "cursor",
    "checkpoint",
    "best_checkpoint",
    "training_report",
    "resume_recoveries",
    "formal_acceptance",
}
_MONITORING_SELECTION_FIELDS = {
    "schema_version",
    "benchmark_build_id",
    "benchmark_payload_sha256",
    "seed",
    "max_parents",
    "selected_records",
    "selected_parents",
    "selection_sha256",
    "role",
    "algorithm",
    "source_counts",
    "task_counts",
    "items",
    "formal_acceptance",
}
_MONITORING_ITEM_FIELDS = {
    "dataset_index",
    "record_id",
    "parent_id",
    "source",
    "task_family",
}
_BEST_FIELDS = {
    "schema_version",
    "selection_metric",
    "step",
    "macro_task_loss",
    "overall_loss",
    "checkpoint",
    "formal_acceptance",
}
_FROZEN_PROTOCOL_FIELDS = {
    "schema_version",
    "protocol_id",
    "static_protocol",
    "training_run",
    "protocol_sha256",
}


@dataclass(frozen=True)
class GateBProtocolSource:
    path: Path
    raw: Mapping[str, Any]
    base_config: Phase4Config
    adapter_config: Phase4Config


def _fail(message: str, *, details: Mapping[str, Any] | None = None) -> None:
    raise ContractError(
        ReasonCode.GATE_B_PROTOCOL_INVALID,
        message,
        details=details,
    )


def _mapping(value: Any, *, location: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(
        isinstance(key, str) for key in value
    ):
        _fail(f"{location}: 必须是字符串键对象")
    return dict(value)


def _exact_fields(
    value: Mapping[str, Any],
    expected: set[str],
    *,
    location: str,
) -> None:
    if set(value) != expected:
        _fail(
            f"{location}: 字段不匹配",
            details={
                "unknown": sorted(set(value) - expected),
                "missing": sorted(expected - set(value)),
            },
        )


def _string(value: Any, *, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(f"{location}: 必须是非空字符串")
    return value.strip()


def _integer(value: Any, *, location: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _fail(f"{location}: 必须是 >= {minimum} 的整数")
    return value


def _number(value: Any, *, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{location}: 必须是有限数值")
    result = float(value)
    if not math.isfinite(result):
        _fail(f"{location}: 必须是有限数值")
    return result


def _sha(value: Any, *, location: str) -> str:
    result = _string(value, location=location)
    if len(result) != 64 or any(
        character not in "0123456789abcdef" for character in result
    ):
        _fail(f"{location}: 必须是 lowercase SHA-256")
    return result


def _string_list(
    value: Any,
    *,
    location: str,
    expected: Sequence[str] | None = None,
) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, str) and item for item in value)
        or len(value) != len(set(value))
    ):
        _fail(f"{location}: 必须是非空唯一字符串列表")
    result = tuple(value)
    if expected is not None and result != tuple(expected):
        _fail(
            f"{location}: 固定顺序不匹配",
            details={"expected": list(expected), "actual": list(result)},
        )
    return result


def _regular_file(path: Path, *, location: str) -> Path:
    path = Path(os.path.abspath(path))
    linked = first_symlink_component(path)
    if linked is not None:
        _fail(f"{location}: 路径含 symlink", details={"path": str(linked)})
    if not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1:
        _fail(f"{location}: 必须是普通单链接文件", details={"path": str(path)})
    return path


def _regular_directory(path: Path, *, location: str) -> Path:
    path = Path(os.path.abspath(path))
    linked = first_symlink_component(path)
    if linked is not None:
        _fail(f"{location}: 路径含 symlink", details={"path": str(linked)})
    if path.is_symlink() or not path.is_dir():
        _fail(f"{location}: 必须是普通目录", details={"path": str(path)})
    return path


def _read_mapping(path: Path, *, location: str) -> dict[str, Any]:
    try:
        value = read_json(_regular_file(path, location=location))
    except RSGeneralDescError as error:
        _fail(
            f"{location}: 无法严格读取 JSON",
            details={"error": str(error)},
        )
    return _mapping(value, location=location)


def _resolved_protocol_path(
    base: Path,
    value: Any,
    *,
    location: str,
) -> Path:
    text = _string(value, location=location)
    candidate = Path(text)
    if not candidate.is_absolute():
        candidate = base / candidate
    return _regular_file(candidate, location=location)


def _validated_config_section(
    value: Any,
    *,
    base: Path,
    location: str,
) -> tuple[dict[str, Any], Phase4Config]:
    row = _mapping(value, location=location)
    _exact_fields(row, _CONFIG_FIELDS, location=location)
    path = _resolved_protocol_path(
        base,
        row["config"],
        location=f"{location}.config",
    )
    file_sha = _sha(
        row["config_file_sha256"],
        location=f"{location}.config_file_sha256",
    )
    if sha256_file(path) != file_sha:
        _fail(
            f"{location}: config 文件 SHA-256 不匹配",
            details={"path": str(path)},
        )
    config = load_config(path)
    semantic_sha = _sha(
        row["config_semantic_sha256"],
        location=f"{location}.config_semantic_sha256",
    )
    if config.semantic_sha256 != semantic_sha:
        _fail(
            f"{location}: config semantic SHA-256 不匹配",
            details={
                "expected": semantic_sha,
                "actual": config.semantic_sha256,
            },
        )
    return (
        {
            "config_path": str(path.resolve()),
            "config_file_sha256": file_sha,
            "config_semantic_sha256": semantic_sha,
        },
        config,
    )


def _validate_protocol_values(row: dict[str, Any]) -> None:
    if row["schema_version"] != GATE_B_PROTOCOL_SCHEMA_VERSION:
        _fail("protocol schema_version 不匹配")
    if row["protocol_id"] != GATE_B_PROTOCOL_ID:
        _fail("protocol_id 不匹配")

    benchmark = _mapping(row["benchmark"], location="$.benchmark")
    _exact_fields(benchmark, _BENCHMARK_FIELDS, location="$.benchmark")
    for name in (
        "root",
        "manifest_schema",
        "canonical_schema",
        "build_id",
    ):
        _string(benchmark[name], location=f"$.benchmark.{name}")
    for name in (
        "manifest_sha256",
        "validation_sha256",
        "payload_sha256",
        "hash_manifest_sha256",
    ):
        _sha(benchmark[name], location=f"$.benchmark.{name}")

    training = _mapping(row["training"], location="$.training")
    _exact_fields(training, _TRAINING_FIELDS, location="$.training")
    for name in _TRAINING_FIELDS - {"checkpoint_step", "adapter_size_bytes"}:
        _sha(training[name], location=f"$.training.{name}")
    _integer(training["checkpoint_step"], location="$.training.checkpoint_step", minimum=1)
    _integer(training["adapter_size_bytes"], location="$.training.adapter_size_bytes", minimum=1)

    model = _mapping(row["model_identity"], location="$.model_identity")
    _exact_fields(model, _MODEL_IDENTITY_FIELDS, location="$.model_identity")
    for name in ("model_path", "model_type"):
        _string(model[name], location=f"$.model_identity.{name}")
    for name in (
        "config_sha256",
        "weights_metadata_sha256",
        "processor_config_sha256",
    ):
        _sha(model[name], location=f"$.model_identity.{name}")
    _integer(
        model["base_parameter_count"],
        location="$.model_identity.base_parameter_count",
        minimum=1,
    )

    processor = _mapping(
        row["processor_identity"],
        location="$.processor_identity",
    )
    _exact_fields(
        processor,
        _PROCESSOR_IDENTITY_FIELDS,
        location="$.processor_identity",
    )
    _string(
        processor["processor_path"],
        location="$.processor_identity.processor_path",
    )
    files = _mapping(
        processor["files"],
        location="$.processor_identity.files",
    )
    if set(files) != {
        "preprocessor_config.json",
        "tokenizer_config.json",
        "chat_template.json",
    }:
        _fail("$.processor_identity.files: 文件集合不匹配")
    for name, value in files.items():
        _sha(value, location=f"$.processor_identity.files.{name}")
    for name in ("processor_class", "tokenizer_class"):
        _string(processor[name], location=f"$.processor_identity.{name}")
    _integer(
        processor["pad_token_id"],
        location="$.processor_identity.pad_token_id",
    )

    selection = _mapping(row["selection"], location="$.selection")
    _exact_fields(selection, _SELECTION_FIELDS, location="$.selection")
    if (
        selection["algorithm"] != GATE_B_SELECTION_ALGORITHM
        or selection["role"] != "external_val"
        or selection["sample_count"] != GATE_B_SAMPLE_COUNT
        or selection["seed"] != GATE_B_SEED
    ):
        _fail("$.selection: 固定算法/role/count/seed 不匹配")
    _string_list(
        selection["task_order"],
        location="$.selection.task_order",
        expected=GATE_B_TASK_ORDER,
    )

    generation = _mapping(row["generation"], location="$.generation")
    _exact_fields(generation, _GENERATION_FIELDS, location="$.generation")
    if (
        generation["seed"] != GATE_B_SEED
        or generation["max_new_tokens"] != 384
        or generation["do_sample"] is not False
        or _number(generation["temperature"], location="$.generation.temperature") != 0.0
        or _number(generation["top_p"], location="$.generation.top_p") != 1.0
        or generation["template_version"] != QWEN_TEMPLATE_VERSION
    ):
        _fail("$.generation: 固定生成参数不匹配")

    evaluation = _mapping(row["evaluation"], location="$.evaluation")
    _exact_fields(evaluation, _EVALUATION_FIELDS, location="$.evaluation")
    _string_list(
        evaluation["open_tasks"],
        location="$.evaluation.open_tasks",
        expected=GATE_B_OPEN_TASKS,
    )
    _string_list(
        evaluation["short_tasks"],
        location="$.evaluation.short_tasks",
        expected=GATE_B_SHORT_TASKS,
    )
    expected_literals = {
        "text_normalization": "unicode_word_lower.v1",
        "token_f1": "multiset_token_f1.v1",
        "rouge_l": "token_lcs_f1.v1",
        "task_macro_weighting": "equal_task.v1",
        "source_macro_weighting": "equal_present_source_task_cell.v1",
    }
    for name, expected in expected_literals.items():
        if evaluation[name] != expected:
            _fail(f"$.evaluation.{name}: 固定值不匹配")
    bootstrap = _mapping(
        evaluation["bootstrap"],
        location="$.evaluation.bootstrap",
    )
    _exact_fields(bootstrap, _BOOTSTRAP_FIELDS, location="$.evaluation.bootstrap")
    if (
        bootstrap["iterations"] != 10_000
        or bootstrap["seed"] != GATE_B_SEED
        or _number(bootstrap["confidence_level"], location="$.evaluation.bootstrap.confidence_level") != 0.95
        or bootstrap["rng"] != "numpy.PCG64"
        or bootstrap["quantile_method"] != "linear"
    ):
        _fail("$.evaluation.bootstrap: 固定 bootstrap 合同不匹配")
    criteria = _mapping(
        evaluation["criteria"],
        location="$.evaluation.criteria",
    )
    _exact_fields(criteria, _CRITERIA_FIELDS, location="$.evaluation.criteria")
    expected_criteria = {
        "primary_ci_lower_strictly_greater_than": 0.0,
        "minimum_improved_tasks": 4,
        "task_delta_floor": -0.02,
        "source_delta_floor": -0.02,
        "open_rouge_l_delta_floor": 0.0,
        "short_exact_match_delta_floor": 0.0,
    }
    for name, expected in expected_criteria.items():
        actual = criteria[name]
        if isinstance(expected, int):
            if actual != expected or isinstance(actual, bool):
                _fail(f"$.evaluation.criteria.{name}: 固定值不匹配")
        elif _number(actual, location=f"$.evaluation.criteria.{name}") != expected:
            _fail(f"$.evaluation.criteria.{name}: 固定值不匹配")


def load_gate_b_protocol(path: Path | str) -> GateBProtocolSource:
    protocol_path = _regular_file(Path(path), location="protocol")
    try:
        row = _load_yaml(protocol_path)
    except Phase4Error as error:
        _fail("无法严格读取 Gate B protocol YAML", details={"error": str(error)})
    _exact_fields(row, _PROTOCOL_FIELDS, location="$")
    _validate_protocol_values(row)
    base_row, base_config = _validated_config_section(
        row["base"],
        base=protocol_path.parent,
        location="$.base",
    )
    adapter_row, adapter_config = _validated_config_section(
        row["adapter"],
        base=protocol_path.parent,
        location="$.adapter",
    )
    if base_config.adaptation.strategy != "prompt_only":
        _fail("Base config 必须是 prompt_only")
    if adapter_config.adaptation.strategy != "lora":
        _fail("Adapter config 必须是 lora")
    if base_config.data.roles != ("external_val",):
        _fail("Base config 必须只登记 external_val")
    if adapter_config.data.roles != ("external_train",):
        _fail("Adapter config 必须只登记 external_train")

    benchmark = row["benchmark"]
    expected_data = {
        "root": str(base_config.data.benchmark_root),
        "manifest_schema": base_config.data.expected_manifest_schema,
        "canonical_schema": base_config.data.expected_canonical_schema,
        "manifest_sha256": base_config.data.expected_manifest_sha256,
        "validation_sha256": base_config.data.expected_validation_sha256,
        "build_id": base_config.data.expected_build_id,
        "payload_sha256": base_config.data.expected_payload_sha256,
        "hash_manifest_sha256": base_config.data.expected_hash_manifest_sha256,
    }
    if benchmark != expected_data:
        _fail(
            "protocol Benchmark identity 与 Base config 不一致",
            details={"expected": expected_data, "actual": benchmark},
        )
    adapter_data = {
        "root": str(adapter_config.data.benchmark_root),
        "manifest_schema": adapter_config.data.expected_manifest_schema,
        "canonical_schema": adapter_config.data.expected_canonical_schema,
        "manifest_sha256": adapter_config.data.expected_manifest_sha256,
        "validation_sha256": adapter_config.data.expected_validation_sha256,
        "build_id": adapter_config.data.expected_build_id,
        "payload_sha256": adapter_config.data.expected_payload_sha256,
        "hash_manifest_sha256": adapter_config.data.expected_hash_manifest_sha256,
    }
    if adapter_data != expected_data:
        _fail("Base/Adapter config 的 Benchmark identity 不一致")

    parity = {
        "model": base_config.model == adapter_config.model,
        "generation": base_config.generation == adapter_config.generation,
        "task_families": (
            base_config.data.task_families
            == adapter_config.data.task_families
            and set(base_config.data.task_families) == set(GATE_B_TASK_ORDER)
        ),
        "image_limits": all(
            getattr(base_config.limits, name)
            == getattr(adapter_config.limits, name)
            for name in (
                "max_images",
                "max_input_tokens",
                "min_pixels",
                "max_pixels",
                "max_new_tokens",
            )
        ),
    }
    if not all(parity.values()):
        _fail("Base/Adapter 的模型、模板任务或生成限制不一致", details=parity)
    protocol_generation = row["generation"]
    if (
        base_config.generation.max_new_tokens
        != protocol_generation["max_new_tokens"]
        or base_config.generation.do_sample
        is not protocol_generation["do_sample"]
        or base_config.generation.temperature
        != float(protocol_generation["temperature"])
        or base_config.generation.top_p
        != float(protocol_generation["top_p"])
    ):
        _fail("protocol 生成参数与 Base/Adapter config 不一致")
    normalized = dict(row)
    normalized["base"] = base_row
    normalized["adapter"] = adapter_row
    return GateBProtocolSource(
        path=protocol_path,
        raw=normalized,
        base_config=base_config,
        adapter_config=adapter_config,
    )


def _implementation_identity() -> dict[str, str]:
    phase4_root = Path(__file__).resolve().parent
    package_root = phase4_root.parent
    paths = {
        "phase3/exporter.py": package_root / "phase3" / "exporter.py",
        "phase4/checkpoint.py": phase4_root / "checkpoint.py",
        "phase4/data.py": phase4_root / "data.py",
        "phase4/gate_b_contracts.py": phase4_root / "gate_b_contracts.py",
        "phase4/gate_b_evaluation.py": phase4_root / "gate_b_evaluation.py",
        "phase4/gate_b_generation.py": phase4_root / "gate_b_generation.py",
        "phase4/gate_b_selection.py": phase4_root / "gate_b_selection.py",
        "phase4/model.py": phase4_root / "model.py",
        "phase4/processing.py": phase4_root / "processing.py",
    }
    return {
        name: sha256_file(_regular_file(path, location=f"implementation.{name}"))
        for name, path in sorted(paths.items())
    }


def static_protocol_snapshot(source: GateBProtocolSource) -> dict[str, Any]:
    return {
        "benchmark": dict(source.raw["benchmark"]),
        "base": dict(source.raw["base"]),
        "adapter": dict(source.raw["adapter"]),
        "training_expectations": dict(source.raw["training"]),
        "model_identity": dict(source.raw["model_identity"]),
        "processor_identity": dict(source.raw["processor_identity"]),
        "selection": dict(source.raw["selection"]),
        "generation": dict(source.raw["generation"]),
        "evaluation": dict(source.raw["evaluation"]),
        "implementation_files": _implementation_identity(),
    }


def _fingerprint(path: Path) -> tuple[int, int, int, int, str]:
    stat = path.stat()
    return (
        stat.st_ino,
        stat.st_size,
        stat.st_mtime_ns,
        stat.st_nlink,
        sha256_file(path),
    )


def _assert_expected_hash(path: Path, expected: str, *, location: str) -> Path:
    regular = _regular_file(path, location=location)
    actual = sha256_file(regular)
    if actual != expected:
        _fail(
            f"{location}: SHA-256 不匹配",
            details={"expected": expected, "actual": actual, "path": str(path)},
        )
    return regular


def _monitoring_selection_identity(
    row: Mapping[str, Any],
) -> tuple[str, tuple[str, ...]]:
    _exact_fields(row, _MONITORING_SELECTION_FIELDS, location="monitoring_selection")
    if (
        row.get("schema_version") != VALIDATION_SELECTION_SCHEMA_VERSION
        or row.get("role") != "external_val"
        or row.get("formal_acceptance") is not False
        or row.get("selected_records") != 128
        or row.get("selected_parents") != 128
    ):
        _fail("monitoring selection 固定合同不匹配")
    items = row.get("items")
    if not isinstance(items, list) or len(items) != 128:
        _fail("monitoring selection items 必须恰好为 128")
    parents: list[str] = []
    records: list[str] = []
    for index, value in enumerate(items):
        item = _mapping(value, location=f"monitoring_selection.items[{index}]")
        _exact_fields(
            item,
            _MONITORING_ITEM_FIELDS,
            location=f"monitoring_selection.items[{index}]",
        )
        _integer(
            item["dataset_index"],
            location=f"monitoring_selection.items[{index}].dataset_index",
        )
        for name in ("record_id", "parent_id", "source", "task_family"):
            _string(
                item[name],
                location=f"monitoring_selection.items[{index}].{name}",
            )
        parents.append(item["parent_id"])
        records.append(item["record_id"])
    if len(set(parents)) != 128 or len(set(records)) != 128:
        _fail("monitoring selection parent/record 必须唯一")
    payload = {
        "schema_version": row["schema_version"],
        "benchmark_build_id": row["benchmark_build_id"],
        "benchmark_payload_sha256": row["benchmark_payload_sha256"],
        "seed": row["seed"],
        "max_parents": row["max_parents"],
        "items": items,
    }
    identity = sha256_text(canonical_json(payload))
    if identity != row.get("selection_sha256"):
        _fail("monitoring selection canonical SHA-256 不匹配")
    return identity, tuple(parents)


def _training_run_identity(
    source: GateBProtocolSource,
    training_root: Path,
    access: BenchmarkAccess,
) -> dict[str, Any]:
    root = _regular_directory(training_root, location="training_root")
    expected = source.raw["training"]
    paths = {
        "training_report": root / "training_report.json",
        "training_manifest": root / "manifest.json",
        "config_snapshot": root / "config_snapshot.json",
        "best_checkpoint": root / "best_checkpoint.json",
        "monitoring_selection": root / "validation_selection.json",
        "validation_results": root / "validation_results.jsonl",
    }
    expected_keys = {
        "training_report": "training_report_sha256",
        "training_manifest": "training_manifest_sha256",
        "config_snapshot": "config_snapshot_sha256",
        "best_checkpoint": "best_checkpoint_sha256",
        "monitoring_selection": "monitoring_selection_file_sha256",
        "validation_results": "validation_results_sha256",
    }
    for name, key in expected_keys.items():
        _assert_expected_hash(
            paths[name],
            expected[key],
            location=f"training.{name}",
        )
    before = {name: _fingerprint(path) for name, path in paths.items()}

    report = _read_mapping(paths["training_report"], location="training_report")
    _exact_fields(report, _TRAINING_REPORT_FIELDS, location="training_report")
    training_manifest = _read_mapping(
        paths["training_manifest"],
        location="training_manifest",
    )
    _exact_fields(
        training_manifest,
        _TRAINING_MANIFEST_FIELDS,
        location="training_manifest",
    )
    config_snapshot = _read_mapping(
        paths["config_snapshot"],
        location="training config_snapshot",
    )
    if config_snapshot != source.adapter_config.snapshot_dict():
        _fail("training config_snapshot 与当前 Adapter config 不一致")

    monitoring = _read_mapping(
        paths["monitoring_selection"],
        location="monitoring_selection",
    )
    monitoring_sha, monitoring_parents = _monitoring_selection_identity(monitoring)
    if monitoring_sha != expected["monitoring_selection_sha256"]:
        _fail("monitoring selection identity 与 protocol 不一致")

    try:
        validation_rows = read_jsonl(paths["validation_results"])
    except RSGeneralDescError as error:
        _fail("validation_results 无法严格读取", details={"error": str(error)})
    validations = [_validation_from_row(row) for row in validation_rows]
    if not validations:
        _fail("completed training 缺少 validation_results")
    computed_best = None
    for validation in validations:
        if validation_is_better(validation, computed_best):
            computed_best = validation
    assert computed_best is not None

    best = _read_mapping(paths["best_checkpoint"], location="best_checkpoint")
    _exact_fields(best, _BEST_FIELDS, location="best_checkpoint")
    checkpoint_step = expected["checkpoint_step"]
    if (
        best.get("schema_version") != BEST_CHECKPOINT_SCHEMA_VERSION
        or best.get("selection_metric") != "macro_task_loss"
        or best.get("formal_acceptance") is not False
        or best.get("step") != checkpoint_step
        or best.get("step") != computed_best.step
        or best.get("macro_task_loss") != computed_best.macro_task_loss
        or best.get("overall_loss") != computed_best.overall_loss
    ):
        _fail("best_checkpoint 与重算 validation best 不一致")
    checkpoint_relative = _string(
        best["checkpoint"],
        location="best_checkpoint.checkpoint",
    )
    try:
        pure_checkpoint = portable_relative_path(
            checkpoint_relative,
            location="best_checkpoint.checkpoint",
        )
    except RSGeneralDescError as error:
        _fail("best checkpoint 相对路径非法", details={"error": str(error)})
    checkpoint = _regular_directory(
        root.joinpath(*pure_checkpoint.parts),
        location="resolved best checkpoint",
    )
    if checkpoint_relative != f"checkpoints/step-{checkpoint_step:08d}":
        _fail("best checkpoint 不是固定 step 路径")
    checkpoint_manifest_path = _assert_expected_hash(
        checkpoint / "manifest.json",
        expected["checkpoint_manifest_sha256"],
        location="checkpoint manifest",
    )
    adapter_path = _assert_expected_hash(
        checkpoint / "adapter" / "adapter_model.safetensors",
        expected["adapter_sha256"],
        location="checkpoint adapter",
    )
    if adapter_path.stat().st_size != expected["adapter_size_bytes"]:
        _fail("checkpoint adapter size 与 protocol 不一致")
    checkpoint_manifest = _read_mapping(
        checkpoint_manifest_path,
        location="checkpoint manifest",
    )
    trainable_names = checkpoint_manifest.get("trainable_parameter_names")
    if (
        not isinstance(trainable_names, list)
        or not trainable_names
        or not all(isinstance(name, str) and name for name in trainable_names)
        or len(trainable_names) != len(set(trainable_names))
        or checkpoint_manifest.get("trainable_parameter_count")
        != EXPECTED_ATTENTION_LORA_R8_PARAMETERS
    ):
        _fail("checkpoint LoRA parameter identity 不满足固定合同")

    actual_model = local_model_identity(
        source.adapter_config.model.path,
        base_parameter_count=source.raw["model_identity"]["base_parameter_count"],
        model_type=source.raw["model_identity"]["model_type"],
    ).to_dict()
    if actual_model != source.raw["model_identity"]:
        _fail("本地 Base model identity 与 protocol 不一致")
    processor = Qwen3VLProcessorAdapter(
        processor_path=source.adapter_config.model.processor_path,
        local_files_only=source.adapter_config.model.local_files_only,
        trust_remote_code=source.adapter_config.model.trust_remote_code,
        min_pixels=source.adapter_config.limits.min_pixels,
        max_pixels=source.adapter_config.limits.max_pixels,
        max_images=source.adapter_config.limits.max_images,
        max_input_tokens=source.adapter_config.limits.max_input_tokens,
    )
    actual_processor = processor.identity()
    if actual_processor != source.raw["processor_identity"]:
        _fail("本地 processor identity 与 protocol 不一致")

    benchmark_identity = access.identity.training_identity_dict()
    validation_identity = dict(monitoring)
    for key in (
        "role",
        "algorithm",
        "source_counts",
        "task_counts",
        "items",
        "formal_acceptance",
    ):
        validation_identity.pop(key)
    checkpoint_payload = CheckpointManager().load(
        checkpoint,
        expected_config_semantic_sha256=source.adapter_config.semantic_sha256,
        expected_benchmark_identity=benchmark_identity,
        expected_validation_selection_identity=validation_identity,
        expected_model_identity=actual_model,
        expected_processor_identity=actual_processor,
        expected_training_layout=training_layout_identity(source.adapter_config),
        expected_trainable_names=tuple(trainable_names),
    )
    if checkpoint_payload.config_snapshot != config_snapshot:
        _fail("training root/checkpoint config_snapshot 不一致")

    if (
        report.get("schema_version") != TRAINING_REPORT_SCHEMA_VERSION
        or report.get("status") != "completed"
        or report.get("formal_acceptance") is not False
        or report.get("config_semantic_sha256")
        != source.adapter_config.semantic_sha256
        or report.get("benchmark") != benchmark_identity
        or report.get("validation_selection") != validation_identity
        or report.get("training_layout")
        != training_layout_identity(source.adapter_config)
        or report.get("cursor") != checkpoint_payload.cursor.to_dict()
        or report.get("max_steps") != checkpoint_step
        or report.get("execution_stop_step") != checkpoint_step
        or report.get("cursor", {}).get("global_step") != checkpoint_step
        or report.get("validation_history") != validation_rows
        or report.get("checkpoint") != str(checkpoint.resolve())
        or report.get("best_checkpoint") != str(checkpoint.resolve())
    ):
        _fail("training_report completed/best/identity 合同不一致")
    if (
        training_manifest.get("run_kind") != "training"
        or training_manifest.get("formal_acceptance") is not False
        or training_manifest.get("config_semantic_sha256")
        != source.adapter_config.semantic_sha256
        or training_manifest.get("benchmark") != benchmark_identity
        or training_manifest.get("validation_selection") != validation_identity
        or training_manifest.get("model") != actual_model
        or training_manifest.get("processor") != actual_processor
        or training_manifest.get("adaptation") != "lora"
        or training_manifest.get("cursor") != checkpoint_payload.cursor.to_dict()
        or training_manifest.get("checkpoint") != str(checkpoint.resolve())
        or training_manifest.get("best_checkpoint") != str(checkpoint.resolve())
        or training_manifest.get("training_report")
        != str(paths["training_report"].resolve())
    ):
        _fail("training manifest 与 completed report/checkpoint 不一致")

    after = {name: _fingerprint(path) for name, path in paths.items()}
    if before != after:
        _fail("训练目录在闭环检查期间发生变化，疑似仍有 writer")
    return {
        "training_root": str(root.resolve()),
        "training_report": {
            "schema_version": report["schema_version"],
            "sha256": expected["training_report_sha256"],
            "status": "completed",
            "formal_acceptance": False,
            "cursor": dict(report["cursor"]),
            "max_steps": report["max_steps"],
        },
        "training_manifest_sha256": expected["training_manifest_sha256"],
        "config_snapshot_sha256": expected["config_snapshot_sha256"],
        "monitoring_selection": {
            "schema_version": monitoring["schema_version"],
            "file_sha256": expected["monitoring_selection_file_sha256"],
            "selection_sha256": monitoring_sha,
            "parent_count": len(monitoring_parents),
            "parent_ids_sha256": sha256_text(
                canonical_json(sorted(monitoring_parents))
            ),
        },
        "validation_results": {
            "file_sha256": expected["validation_results_sha256"],
            "result_count": len(validations),
            "best_step": computed_best.step,
            "best_macro_task_loss": computed_best.macro_task_loss,
            "best_overall_loss": computed_best.overall_loss,
        },
        "best_checkpoint": {
            "pointer_sha256": expected["best_checkpoint_sha256"],
            "relative_path": checkpoint_relative,
            "step": checkpoint_step,
            "manifest_sha256": expected["checkpoint_manifest_sha256"],
            "adapter_size_bytes": expected["adapter_size_bytes"],
            "adapter_sha256": expected["adapter_sha256"],
            "trainable_parameter_count": EXPECTED_ATTENTION_LORA_R8_PARAMETERS,
        },
        "benchmark_identity": benchmark_identity,
        "model_identity": actual_model,
        "processor_identity": actual_processor,
        "training_layout": training_layout_identity(source.adapter_config),
    }


def build_frozen_protocol(
    protocol_path: Path | str,
    *,
    training_root: Path,
) -> tuple[dict[str, Any], GateBProtocolSource, BenchmarkAccess]:
    source = load_gate_b_protocol(protocol_path)
    access = open_benchmark_access(source.base_config)
    training_run = _training_run_identity(source, training_root, access)
    body = {
        "schema_version": GATE_B_PROTOCOL_SCHEMA_VERSION,
        "protocol_id": GATE_B_PROTOCOL_ID,
        "static_protocol": static_protocol_snapshot(source),
        "training_run": training_run,
    }
    return (
        {**body, "protocol_sha256": sha256_text(canonical_json(body))},
        source,
        access,
    )


def read_frozen_protocol(path: Path | str) -> dict[str, Any]:
    row = _read_mapping(Path(path), location="frozen protocol")
    _exact_fields(row, _FROZEN_PROTOCOL_FIELDS, location="frozen protocol")
    if (
        row.get("schema_version") != GATE_B_PROTOCOL_SCHEMA_VERSION
        or row.get("protocol_id") != GATE_B_PROTOCOL_ID
    ):
        _fail("frozen protocol schema/id 不匹配")
    expected = _sha(
        row["protocol_sha256"],
        location="frozen protocol.protocol_sha256",
    )
    body = {key: value for key, value in row.items() if key != "protocol_sha256"}
    actual = sha256_text(canonical_json(body))
    if actual != expected:
        _fail(
            "frozen protocol canonical SHA-256 不匹配",
            details={"expected": expected, "actual": actual},
        )
    _mapping(row["static_protocol"], location="frozen protocol.static_protocol")
    _mapping(row["training_run"], location="frozen protocol.training_run")
    return row


def validate_frozen_protocol(
    protocol_path: Path | str,
    frozen_path: Path | str,
) -> tuple[dict[str, Any], GateBProtocolSource]:
    frozen = read_frozen_protocol(frozen_path)
    source = load_gate_b_protocol(protocol_path)
    current_static = static_protocol_snapshot(source)
    if current_static != frozen["static_protocol"]:
        _fail("protocol 或 Gate B implementation 在冻结后发生变化")
    return frozen, source


def validate_frozen_training_root(
    frozen: Mapping[str, Any],
    source: GateBProtocolSource,
    *,
    training_root: Path,
    access: BenchmarkAccess,
) -> dict[str, Any]:
    """重新核验 Adapter training root，禁止冻结后漂移或改用其他 checkpoint。"""

    current = _training_run_identity(source, training_root, access)
    if current != frozen.get("training_run"):
        _fail("Adapter training root identity 与 frozen protocol 不一致")
    return current
