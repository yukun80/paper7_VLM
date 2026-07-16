#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Paired Small-seed comparison for crop-only, masked pooling and MGRR."""

from __future__ import annotations

import json
from pathlib import Path
import hashlib
import math
from collections import Counter, defaultdict
from typing import Any

from qpsalm_seg.paths import resolve_project_path

from .data import REGION_TRAINING_DATA_PROTOCOL, load_frozen_scientific_gate
from .json_protocol import strict_json_loads
from .checkpoint import (
    description_protocol_assets_spec,
    inspect_segdesc_checkpoint,
    validate_description_stage_lineage,
)
from .evaluator import (
    COUNTERFACTUAL_INPUT_AUDIT_PROTOCOL,
    DESCRIPTION_EVALUATION_PROTOCOL,
    EVALUATION_CHECKPOINT_BINDING_PROTOCOL,
    SAME_IMAGE_RETRIEVAL_PROTOCOL,
    evaluation_population_sha256,
    revalidate_evaluation_mask_artifacts,
    revalidate_evaluation_publication,
)
from .metrics import (
    bootstrap_mean_ci,
    caption_token_f1,
    paired_bootstrap_delta_ci,
    structured_disagreement,
    unsupported_claim_counts,
)
from .output_protocol import parse_description_output
from .expert_factuality import (
    EXPERT_FACTUALITY_PROTOCOL,
    revalidate_expert_factuality_report,
)
from .vision_cache import revalidate_description_cache_artifact
from .run_artifacts import (
    DESCRIPTION_TRAINING_COMPLETION_PROTOCOL,
    validate_checkpoint_run_completion,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_sha256(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


M4_BASELINE_REGION_ENCODERS = {
    "crop_only",
    "full_image_box",
    "masked_pooling",
    "roi_replay_only",
    "mgrr_no_context",
}
M4_SEED_GATE_PROTOCOL = (
    "qpsalm_description_seed_gate_v12_strict_json_finite"
)
M4_SUITE_GATE_PROTOCOL = (
    "qpsalm_m4_region_encoder_suite_v8_strict_json_finite"
)
M4_CROSS_SEED_TRAINING_POPULATION_PROTOCOL = (
    "qpsalm_m4_cross_seed_training_population_v1"
)
M4_VARIANT_CONFIG_FIELDS = {"region_encoder", "output_dir"}


def _m4_dataset_population_audit(
    value: Any,
    *,
    label: str,
    expected_stage: str,
    expected_split: str,
) -> dict[str, Any]:
    """Validate one seed-independent description dataset identity."""
    if not isinstance(value, dict):
        raise ValueError(f"M4 {label} dataset audit 缺失")
    audit = dict(value)
    if audit.get("protocol") != "qpsalm_description_dataset_population_v1":
        raise ValueError(f"M4 {label} dataset population protocol 不兼容")
    if (
        str(audit.get("stage") or "") != expected_stage
        or str(audit.get("split") or "") != expected_split
    ):
        raise ValueError(f"M4 {label} dataset stage/split 不匹配")
    try:
        num_samples = int(audit.get("num_samples"))
        num_parents = int(audit.get("num_parents"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"M4 {label} dataset population counts 非法") from exc
    population_sha256 = str(audit.get("population_sha256") or "")
    if (
        num_samples <= 0
        or num_parents <= 0
        or num_parents > num_samples
        or not _is_sha256(population_sha256)
        or not isinstance(audit.get("tasks"), dict)
        or not isinstance(audit.get("sources"), dict)
        or sum(int(value) for value in audit["tasks"].values()) != num_samples
        or sum(int(value) for value in audit["sources"].values()) != num_samples
    ):
        raise ValueError(f"M4 {label} dataset population 不完整")
    return audit


def _m4_stream_loader_contract(
    value: Any,
    *,
    label: str,
    stream: str,
    expected_stage: str,
    expected_seed: int,
    dataset_audit: dict[str, Any],
) -> dict[str, Any]:
    """Verify a run-local loader seed, then remove it for cross-seed comparison."""
    if not isinstance(value, dict):
        raise ValueError(f"M4 {label} loader binding 缺失")
    binding = dict(value)
    if binding.get("protocol") != "qpsalm_description_stream_binding_v1":
        raise ValueError(f"M4 {label} loader binding protocol 不兼容")
    try:
        dataset_samples = int(binding.get("dataset_samples"))
        epoch_zero_batches = int(binding.get("epoch_zero_batches"))
        num_workers = int(binding.get("num_workers"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"M4 {label} loader counts 非法") from exc
    if (
        str(binding.get("stream") or "") != stream
        or str(binding.get("stage") or "") != expected_stage
        or dataset_samples != int(dataset_audit["num_samples"])
        or epoch_zero_batches <= 0
        or num_workers < 0
        or binding.get("persistent_workers") is not False
        or binding.get("dataset_audit_sha256")
        != _canonical_sha256(dataset_audit)
    ):
        raise ValueError(f"M4 {label} loader/dataset binding 不一致")
    if binding.get("binding_sha256") != _canonical_sha256({
        key: item for key, item in binding.items() if key != "binding_sha256"
    }):
        raise ValueError(f"M4 {label} loader binding hash 不一致")
    sampler = binding.get("batch_sampler")
    if not isinstance(sampler, dict):
        raise ValueError(f"M4 {label} batch sampler binding 缺失")
    try:
        loader_seed = int(binding.get("loader_seed"))
        sampler_seed = int(sampler.get("seed"))
        batch_size = int(sampler.get("batch_size"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"M4 {label} loader seed 非法") from exc
    if (
        loader_seed != expected_seed
        or sampler_seed != expected_seed
        or batch_size <= 0
        or not str(sampler.get("class") or "")
        or not str(sampler.get("protocol") or "")
        or not isinstance(sampler.get("drop_last"), bool)
    ):
        raise ValueError(
            f"M4 {label} loader seed 未按 run seed 的预注册偏移生成"
        )

    # 运行 seed 只控制 shuffle 顺序；批预算和 sampler 类型仍属于科学控制变量。
    normalized = {
        key: item
        for key, item in binding.items()
        if key not in {"loader_seed", "binding_sha256"}
    }
    normalized["batch_sampler"] = {
        key: item for key, item in sampler.items() if key != "seed"
    }
    return normalized


def _m4_cross_seed_training_population_contract(
    data_audit: Any,
    region_data_audit: Any,
    *,
    expected_seed: int,
) -> dict[str, Any]:
    """Build the strict seed-neutral D3b population used by the M4 gate."""
    if not isinstance(data_audit, dict):
        raise ValueError("M4 checkpoint 缺少 training data audit")
    if (
        data_audit.get("protocol")
        != "qpsalm_description_training_data_binding_v2_loader_bound"
    ):
        raise ValueError("M4 training data binding protocol 不兼容")
    expected_stages = {
        "bridge": "bridge_expert",
        "dior": "dior_alignment",
        "global_caption": "rsicap_caption",
    }
    expected_loader_offsets = {
        "bridge": 11_003,
        "dior": 21_013,
        "global_caption": 31_019,
    }
    streams = data_audit.get("training_streams")
    loaders = data_audit.get("stream_loader_bindings")
    if (
        not isinstance(streams, dict)
        or not isinstance(loaders, dict)
        or set(streams) != set(expected_stages)
        or set(loaders) != set(expected_stages)
    ):
        raise ValueError("M4 D3b 必须绑定 bridge/dior/global_caption 三个训练流")
    stream_pattern = data_audit.get("stream_pattern")
    if (
        not isinstance(stream_pattern, list)
        or not stream_pattern
        or set(stream_pattern) != set(expected_stages)
    ):
        raise ValueError("M4 D3b task stream pattern 不完整")

    dataset_contracts: dict[str, dict[str, Any]] = {}
    loader_contracts: dict[str, dict[str, Any]] = {}
    for name, stage in expected_stages.items():
        dataset_contracts[name] = _m4_dataset_population_audit(
            streams[name],
            label=f"stream={name}",
            expected_stage=stage,
            expected_split="train",
        )
        loader_contracts[name] = _m4_stream_loader_contract(
            loaders[name],
            label=f"stream={name}",
            stream=name,
            expected_stage=stage,
            expected_seed=int(expected_seed) + expected_loader_offsets[name],
            dataset_audit=dataset_contracts[name],
        )
    validation = _m4_dataset_population_audit(
        data_audit.get("validation"),
        label="validation",
        expected_stage="bridge_expert",
        expected_split="val",
    )

    if not isinstance(region_data_audit, dict):
        raise ValueError("M4 checkpoint 缺少 region data audit")
    region = dict(region_data_audit)
    bridge_dataset = dataset_contracts["bridge"]
    expected_region_population = {
        key: bridge_dataset[key]
        for key in (
            "protocol", "stage", "split", "num_samples", "num_parents",
            "population_sha256",
        )
    }
    if (
        region.get("protocol") != REGION_TRAINING_DATA_PROTOCOL
        or region.get("stage") != "bridge_expert"
        or region.get("population") != expected_region_population
        or region.get("expert_gate_audit")
        != bridge_dataset.get("expert_gate_audit")
        or region.get("bridge_engineering_audit")
        != bridge_dataset.get("bridge_engineering_audit")
        or region.get("predicted_index_audit")
        != bridge_dataset.get("predicted_index_audit")
        or region.get("curriculum_audit")
        != bridge_dataset.get("curriculum_audit")
    ):
        raise ValueError("M4 region audit 与 D3b bridge population 不一致")

    return {
        "protocol": M4_CROSS_SEED_TRAINING_POPULATION_PROTOCOL,
        "training_data_protocol": data_audit["protocol"],
        "training_streams": dataset_contracts,
        "stream_loader_contracts": loader_contracts,
        "validation": validation,
        "stream_pattern": list(stream_pattern),
        "region_data_audit": region,
    }


def _m4_training_control_audit(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    expected_seed: int,
) -> dict[str, Any]:
    """Prove the paired encoder runs share data, budgets and a D1 ancestor."""
    baseline_metadata = dict(
        (baseline.get("checkpoint_metadata") or {}).get("metadata") or {}
    )
    candidate_metadata = dict(
        (candidate.get("checkpoint_metadata") or {}).get("metadata") or {}
    )
    baseline_config = dict(baseline_metadata.get("config") or {})
    candidate_config = dict(candidate_metadata.get("config") or {})
    baseline_encoder = baseline_config.get("region_encoder")
    candidate_encoder = candidate_config.get("region_encoder")
    if (
        baseline_encoder not in M4_BASELINE_REGION_ENCODERS
        or candidate_encoder != "mgrr"
    ):
        raise ValueError(
            "正式 M4 配对必须是预注册 baseline region encoder 对 full MGRR"
        )
    baseline_controlled = {
        key: value
        for key, value in baseline_config.items()
        if key not in M4_VARIANT_CONFIG_FIELDS
    }
    candidate_controlled = {
        key: value
        for key, value in candidate_config.items()
        if key not in M4_VARIANT_CONFIG_FIELDS
    }
    if baseline_controlled != candidate_controlled:
        raise ValueError("M4 baseline/candidate 除 region encoder 外训练配置不一致")
    try:
        checkpoint_seed = int(baseline_controlled.get("seed"))
    except (TypeError, ValueError) as exc:
        raise ValueError("M4 checkpoint config seed 非法") from exc
    if checkpoint_seed != int(expected_seed):
        raise ValueError("M4 checkpoint config seed 与配对 seed 不一致")
    if (
        baseline_metadata.get("data_audit") != candidate_metadata.get("data_audit")
        or baseline_metadata.get("region_data_audit")
        != candidate_metadata.get("region_data_audit")
    ):
        raise ValueError("M4 baseline/candidate training population 不一致")

    def lineage(metadata: dict[str, Any], label: str) -> list[dict[str, Any]]:
        try:
            value = validate_description_stage_lineage(
                metadata.get("stage_lineage"),
                expected_target_stage=str(metadata.get("stage") or ""),
            )
        except RuntimeError as exc:
            raise ValueError(
                f"M4 {label} checkpoint 缺少合法 stage lineage"
            ) from exc
        return [dict(item) for item in value["entries"]]

    baseline_lineage = lineage(baseline_metadata, "baseline")
    candidate_lineage = lineage(candidate_metadata, "candidate")
    baseline_stages = [str(item.get("stage") or "") for item in baseline_lineage]
    candidate_stages = [str(item.get("stage") or "") for item in candidate_lineage]
    if baseline_stages != candidate_stages:
        raise ValueError("M4 baseline/candidate D0-D3 stage lineage 不同")
    controlled_fields = (
        "stage",
        "seed",
        "controlled_config_sha256",
        "data_audit_sha256",
        "region_data_audit_sha256",
        "d_minus_one_acceptance_sha256",
    )
    for left, right in zip(baseline_lineage, candidate_lineage, strict=True):
        if any(left.get(field) != right.get(field) for field in controlled_fields):
            raise ValueError("M4 baseline/candidate 上游训练预算或 population 不一致")
        if int(left.get("seed", -1)) != int(expected_seed):
            raise ValueError("M4 stage lineage seed 与配对 seed 不一致")
    baseline_d1 = [
        item for item in baseline_lineage if item.get("stage") == "rsicap_caption"
    ]
    candidate_d1 = [
        item for item in candidate_lineage if item.get("stage") == "rsicap_caption"
    ]
    if (
        len(baseline_d1) != 1
        or len(candidate_d1) != 1
        or baseline_d1[0].get("checkpoint_sha256")
        != candidate_d1[0].get("checkpoint_sha256")
    ):
        raise ValueError("M4 baseline/candidate 必须共享同一个 D1 upstream checkpoint")
    cross_seed_config = {
        key: value
        for key, value in baseline_controlled.items()
        if key != "seed"
    }
    cross_seed_population = _m4_cross_seed_training_population_contract(
        baseline_metadata.get("data_audit"),
        baseline_metadata.get("region_data_audit"),
        expected_seed=expected_seed,
    )
    return {
        "baseline_region_encoder": baseline_encoder,
        "candidate_region_encoder": candidate_encoder,
        "shared_d1_checkpoint_sha256": baseline_d1[0]["checkpoint_sha256"],
        "controlled_config_sha256": _canonical_sha256(baseline_controlled),
        "cross_seed_scientific_config_sha256": _canonical_sha256(
            cross_seed_config
        ),
        "training_data_audit_sha256": _canonical_sha256(
            baseline_metadata.get("data_audit")
        ),
        "region_data_audit_sha256": _canonical_sha256(
            baseline_metadata.get("region_data_audit")
        ),
        "cross_seed_training_population": cross_seed_population,
        "cross_seed_training_population_sha256": _canonical_sha256(
            cross_seed_population
        ),
        "lineage_stages": baseline_stages,
    }


def _validate_expert_binding(
    expert: dict[str, Any],
    evaluation_dir: str | Path,
    *,
    expert_report_path: str | Path,
    label: str,
) -> None:
    root = (resolve_project_path(evaluation_dir) or Path(evaluation_dir)).resolve(
        strict=False
    )
    observed_root = (resolve_project_path(str(expert.get("eval_dir") or "")) or Path(
        str(expert.get("eval_dir") or "")
    )).resolve(strict=False)
    if observed_root != root:
        raise ValueError(f"{label} expert report 与 eval directory 不匹配")
    generation = root / "raw_generations.jsonl"
    report = root / "eval_report.json"
    if (
        str(expert.get("raw_generations_sha256") or "") != _sha256(generation)
        or str(expert.get("eval_report_sha256") or "") != _sha256(report)
    ):
        raise ValueError(f"{label} expert report 的 frozen eval 指纹已失效")
    rebuilt = revalidate_expert_factuality_report(
        expert_report_path,
        evaluation_dir=evaluation_dir,
    )
    if rebuilt != expert:
        raise ValueError(f"{label} expert report source revalidation 不一致")


def _validate_evaluation_checkpoint_provenance(
    root: Path, report: dict[str, Any],
) -> dict[str, Any]:
    """Reject reports detached from the exact model and lineage they evaluated."""
    checkpoint = resolve_project_path(str(report.get("checkpoint") or ""))
    checkpoint_sha256 = str(report.get("checkpoint_sha256") or "")
    if (
        checkpoint is None
        or not checkpoint.is_file()
        or not checkpoint_sha256
        or _sha256(checkpoint) != checkpoint_sha256
    ):
        raise ValueError(f"正式 evaluation checkpoint path/hash 已失效: {root}")
    binding = dict(report.get("checkpoint_binding") or {})
    checkpoint_report = dict(report.get("checkpoint_metadata") or {})
    try:
        checkpoint_provenance = inspect_segdesc_checkpoint(checkpoint)
    except (FileNotFoundError, RuntimeError) as exc:
        raise ValueError(
            f"正式 evaluation checkpoint payload 无法重放: {root}"
        ) from exc
    if (
        checkpoint_report != checkpoint_provenance["checkpoint_metadata"]
        or report.get("checkpoint_step") != checkpoint_provenance["checkpoint_step"]
        or checkpoint_sha256 != checkpoint_provenance["checkpoint_sha256"]
    ):
        raise ValueError(
            f"正式 evaluation checkpoint step/metadata 与 payload 不一致: {root}"
        )
    architecture = dict(
        checkpoint_report.get("description_architecture_spec") or {}
    )
    try:
        cache_artifact_provenance = revalidate_description_cache_artifact(
            architecture.get("description_cache_artifact_binding")
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"正式 evaluation Description Vision Cache artifact 无法重放: {root}"
        ) from exc
    metadata = dict(checkpoint_report.get("metadata") or {})
    checkpoint_stage = str(metadata.get("stage") or "")
    expected_checkpoint_role = (
        "terminal_last"
        if checkpoint_stage in {"overfit", "bridge_auto"}
        else "validation_best"
    )
    try:
        run_completion = validate_checkpoint_run_completion(
            checkpoint,
            expected_completion_protocol=(
                DESCRIPTION_TRAINING_COMPLETION_PROTOCOL
            ),
            expected_stage=checkpoint_stage,
            expected_role=expected_checkpoint_role,
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        raise ValueError(
            f"正式 evaluation checkpoint 训练 run 未成功完成: {root}"
        ) from exc
    if binding.get("protocol") != EVALUATION_CHECKPOINT_BINDING_PROTOCOL:
        raise ValueError(f"正式 evaluation 缺少 checkpoint binding: {root}")
    if (
        str(binding.get("checkpoint_stage") or "")
        != checkpoint_stage
        or metadata.get("checkpoint_role") != expected_checkpoint_role
        or binding.get("checkpoint_role") != expected_checkpoint_role
        or binding.get("expected_checkpoint_role") != expected_checkpoint_role
        or binding.get("saved_segmentation_migration")
        != checkpoint_report.get("segmentation_migration")
        or str(binding.get("evaluation_data_stage") or "")
        != str(report.get("stage") or "")
        or str(binding.get("evaluation_mode") or "")
        != str(report.get("evaluation_mode") or "")
        or binding.get("segmentation_source_sha256_match") is not True
        or binding.get("run_completion") != run_completion
    ):
        raise ValueError(f"正式 evaluation checkpoint binding 与报告不一致: {root}")
    seed_binding = _formal_seed_binding(
        report,
        expected_seed=binding.get("evaluation_seed"),
        label=str(root),
    )
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_stage": binding["checkpoint_stage"],
        "checkpoint_role": expected_checkpoint_role,
        "run_completion": run_completion,
        "checkpoint_payload_provenance": checkpoint_provenance,
        "description_cache_artifact_provenance": cache_artifact_provenance,
        "seed_binding": seed_binding,
    }


def _formal_seed_binding(
    report: dict[str, Any],
    *,
    expected_seed: int | None,
    label: str,
) -> dict[str, Any]:
    """Bind a formal artifact slot to its saved training and runtime seed."""
    binding = dict(report.get("checkpoint_binding") or {})
    checkpoint_metadata = dict(report.get("checkpoint_metadata") or {})
    metadata = dict(checkpoint_metadata.get("metadata") or {})
    protocol_assets = checkpoint_metadata.get("description_protocol_assets")
    current_protocol_assets = description_protocol_assets_spec()
    if protocol_assets != current_protocol_assets:
        raise ValueError(
            f"正式 artifact ontology/schema binding 已漂移: {label}"
        )
    saved_config = dict(metadata.get("config") or {})
    statistics = dict(report.get("statistics_protocol") or {})
    values = {
        "expected_seed": expected_seed,
        "checkpoint_training_seed": binding.get("checkpoint_training_seed"),
        "checkpoint_config_seed": saved_config.get("seed"),
        "evaluation_seed": binding.get("evaluation_seed"),
        "runtime_seed": statistics.get("runtime_seed"),
    }
    if expected_seed is None:
        raise ValueError(f"正式 artifact 缺少 expected seed: {label}")
    try:
        normalized = {key: int(value) for key, value in values.items()}
    except (TypeError, ValueError) as exc:
        raise ValueError(f"正式 artifact seed binding 不完整: {label}:{values}") from exc
    if len(set(normalized.values())) != 1 or binding.get("seed_match") is not True:
        raise ValueError(f"正式 artifact seed binding 不一致: {label}:{values}")
    checkpoint_sha256 = str(report.get("checkpoint_sha256") or "")
    if len(checkpoint_sha256) != 64:
        raise ValueError(f"正式 artifact 缺少 checkpoint SHA-256: {label}")
    return {
        **normalized,
        "checkpoint_sha256": checkpoint_sha256,
        "description_protocol_assets_sha256": _canonical_sha256(
            current_protocol_assets
        ),
    }


def _validate_formal_evaluation_limit(
    report: dict[str, Any],
    *,
    observed_count: int,
    label: str,
) -> dict[str, Any]:
    limit = dict(report.get("evaluation_limit_audit") or {})
    if (
        limit.get("protocol") != "qpsalm_description_evaluation_limit_v1"
        or int(limit.get("requested_max_samples", -1)) != 0
        or limit.get("full_population_requested") is not True
        or int(limit.get("dataset_rows_evaluated", -1)) != int(observed_count)
    ):
        raise ValueError(
            f"正式 {label} 要求 --max-val-samples 0 的完整 population"
        )
    return limit


def _rows(
    directory: str | Path,
    *,
    require_complete_generation: bool = False,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    root = resolve_project_path(directory) or Path(directory)
    ordered = [
        strict_json_loads(line)
        for line in (root / "raw_generations.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    sample_ids = [str(row.get("sample_id") or "") for row in ordered]
    duplicates = sorted(
        value for value, count in Counter(sample_ids).items() if not value or count > 1
    )
    if duplicates:
        raise ValueError(f"正式 paired generation sample_id 非空且唯一: {duplicates[:8]}")
    rows = {str(row["sample_id"]): row for row in ordered}
    report = strict_json_loads(
        (root / "eval_report.json").read_text(encoding="utf-8")
    )
    if require_complete_generation and report.get("protocol") != DESCRIPTION_EVALUATION_PROTOCOL:
        raise ValueError(
            f"正式 paired gate 需要 {DESCRIPTION_EVALUATION_PROTOCOL}：{root}；"
            "请用当前 evaluator 重跑评估"
        )
    if require_complete_generation and not bool(
        (report.get("generation_coverage") or {}).get("complete")
    ):
        raise ValueError(
            f"正式 paired gate 需要全量 generation：{root}；"
            "请用 --max-generate-samples 0 重跑评估"
        )
    if require_complete_generation:
        _validate_formal_evaluation_limit(
            report, observed_count=len(ordered), label=f"paired gate: {root}"
        )
        revalidate_evaluation_publication(root, report)
        _validate_evaluation_checkpoint_provenance(root, report)
        revalidate_evaluation_mask_artifacts(root, ordered, report)
        coverage = report.get("generation_coverage") or {}
        observed_population = evaluation_population_sha256(ordered)
        if (
            int(report.get("num_samples", -1)) != len(ordered)
            or int(report.get("num_generated", -1)) != len(ordered)
            or coverage.get("population_sha256") != observed_population
        ):
            raise ValueError(f"正式 paired gate 的 generation population 指纹失效: {root}")
    return rows, report


def _score(row: dict[str, Any]) -> float:
    metrics = row.get("raw_metrics") or {}
    if metrics.get("raw_field_accuracy") is not None:
        return float(metrics["raw_field_accuracy"])
    return float(metrics.get("caption_token_f1") or 0.0)


def _claim_rate(rows: list[dict[str, Any]]) -> float:
    unsupported = sum(int((row.get("raw_metrics") or {}).get("unsupported_claims") or 0) for row in rows)
    claims = sum(int((row.get("raw_metrics") or {}).get("factual_claims") or 0) for row in rows)
    return unsupported / max(claims, 1)


def _load_counterfactual_rows(directory: str | Path) -> list[dict[str, Any]]:
    root = resolve_project_path(directory) or Path(directory)
    path = root / "counterfactual_generations.jsonl"
    rows = [
        strict_json_loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    keys = [
        (str(row.get("sample_id") or ""), str(row.get("mode") or ""))
        for row in rows
    ]
    duplicates = sorted(value for value, count in Counter(keys).items() if count > 1)
    if duplicates:
        raise ValueError(f"counterfactual generations 存在重复 sample/mode: {duplicates[:8]}")
    return rows


def _parent_counterfactual_values(
    rows: list[dict[str, Any]], mode: str, field: str,
) -> list[float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if str(row.get("mode") or "") == mode:
            grouped[str(row["parent_sample_id"])].append(float(row[field]))
    return [sum(values) / len(values) for _parent, values in sorted(grouped.items())]


def _validate_counterfactual_row_bindings(
    rows: list[dict[str, Any]],
    generation_rows: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Replay deltas and prove every counted perturbation changed model input."""

    mask_modes = {
        "full_mask", "zero_mask", "shuffled_mask", "region_swap",
        "cross_parent_region_swap",
    }
    state_modes = {"modality_removal", "cross_parent_modality_swap"}
    allowed_modes = mask_modes | state_modes
    for index, row in enumerate(rows):
        sample_id = str(row.get("sample_id") or "")
        mode = str(row.get("mode") or "")
        base = generation_rows.get(sample_id)
        if base is None or mode not in allowed_modes:
            raise ValueError(
                f"counterfactual row 未绑定正式 generation/mode: row={index} "
                f"sample={sample_id!r} mode={mode!r}"
            )
        if (
            str(row.get("parent_sample_id") or "")
            != str(base.get("parent_sample_id") or "")
            or str(row.get("baseline_generation") or "")
            != str(base.get("raw_generation") or "")
        ):
            raise ValueError("counterfactual baseline generation/parent 已与正式行漂移")

        change = row.get("input_change_audit")
        if (
            not isinstance(change, dict)
            or change.get("protocol") != COUNTERFACTUAL_INPUT_AUDIT_PROTOCOL
            or change.get("mode") != mode
            or change.get("changed") is not True
        ):
            raise ValueError("counterfactual row 缺少可重放 input-change audit")
        dimensions = change.get("changed_dimensions")
        if (
            not isinstance(dimensions, list)
            or len(dimensions) != len(set(dimensions))
            or not set(dimensions).issubset({"region_mask", "backbone_state"})
            or (mode in mask_modes and "region_mask" not in dimensions)
            or (mode in state_modes and "backbone_state" not in dimensions)
        ):
            raise ValueError("counterfactual input-change dimension 与 mode 不一致")
        for prefix in ("region_mask", "backbone_state"):
            left = str(change.get(f"baseline_{prefix}_sha256") or "")
            right = str(change.get(f"counterfactual_{prefix}_sha256") or "")
            if len(left) != 64 or len(right) != 64:
                raise ValueError("counterfactual input fingerprint 不完整")
            expected_changed = prefix in dimensions
            if (left != right) != expected_changed:
                raise ValueError("counterfactual input fingerprint 与 changed_dimensions 冲突")

        donor = row.get("counterfactual_input")
        parent = str(base.get("parent_sample_id") or "")
        if mode == "region_swap" and (
            not isinstance(donor, dict)
            or donor.get("protocol") != "qpsalm_same_parent_region_swap_v1"
            or str(donor.get("parent_sample_id") or "") != parent
        ):
            raise ValueError("same-parent region swap donor audit 非法")
        if mode == "cross_parent_region_swap" and (
            not isinstance(donor, dict)
            or donor.get("protocol") != "qpsalm_cross_parent_region_swap_v1"
            or str(donor.get("target_parent_sample_id") or "") != parent
            or not str(donor.get("donor_parent_sample_id") or "")
            or str(donor.get("donor_parent_sample_id")) == parent
        ):
            raise ValueError("cross-parent region swap donor audit 非法")
        if mode == "cross_parent_modality_swap" and (
            not isinstance(donor, dict)
            or donor.get("protocol") != "qpsalm_cross_parent_modality_donor_v1"
            or str(donor.get("target_parent_sample_id") or "") != parent
            or not str(donor.get("donor_parent_sample_id") or "")
            or str(donor.get("donor_parent_sample_id")) == parent
            or not isinstance(donor.get("applied_swap"), dict)
            or (donor.get("applied_swap") or {}).get("protocol")
            != "qpsalm_cross_parent_modality_swap_v1"
        ):
            raise ValueError("cross-parent modality swap donor audit 非法")

        baseline = str(row.get("baseline_generation") or "")
        changed = str(row.get("counterfactual_generation") or "")
        target = str(base.get("target_text") or "")
        structured = str(base.get("task_family") or "") in {
            "region_description_auto", "region_description_expert",
        }
        if structured:
            baseline_parsed = parse_description_output(baseline).parsed
            changed_parsed = parse_description_output(changed).parsed
            target_parsed = parse_description_output(target).parsed
            expected_sensitivity = structured_disagreement(
                baseline_parsed, changed_parsed
            )
            baseline_score = 1.0 - structured_disagreement(
                baseline_parsed, target_parsed
            )
            changed_score = 1.0 - structured_disagreement(
                changed_parsed, target_parsed
            )
            baseline_claims = unsupported_claim_counts(
                baseline_parsed, target_parsed
            )[1]
            changed_claims = unsupported_claim_counts(
                changed_parsed, target_parsed
            )[1]
        else:
            references = list(base.get("reference_texts") or [])
            expected_sensitivity = 1.0 - caption_token_f1(changed, [baseline])
            baseline_score = caption_token_f1(baseline, references)
            changed_score = caption_token_f1(changed, references)
            baseline_claims = changed_claims = 0
        expected = {
            "sensitivity": expected_sensitivity,
            "baseline_target_score": baseline_score,
            "counterfactual_target_score": changed_score,
            "target_score_delta": changed_score - baseline_score,
            "factual_claim_count_delta": float(changed_claims - baseline_claims),
        }
        for field, value in expected.items():
            try:
                observed = float(row[field])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"counterfactual row 缺少派生字段: {field}") from exc
            if not math.isclose(observed, float(value), rel_tol=0.0, abs_tol=1.0e-12):
                raise ValueError(f"反事实派生指标无法重新计算: {field}")
    return {
        "protocol": "qpsalm_counterfactual_row_binding_v1_generation_replayed",
        "num_rows": len(rows),
        "num_generation_rows": len(generation_rows),
        "rows_sha256": _canonical_sha256(rows),
        "passed": True,
    }


def _counterfactual_gate(
    report: dict[str, Any],
    rows: list[dict[str, Any]] | None = None,
    scientific_protocol: dict[str, Any] | None = None,
    generation_rows: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    values = report.get("counterfactual_sensitivity") or {}

    def upper(mode: str, metric: str) -> float | None:
        value = (((values.get(mode) or {}).get(metric) or {}).get("high"))
        return float(value) if value is not None else None

    required_modes = (
        "shuffled_mask", "region_swap", "cross_parent_region_swap",
        "cross_parent_modality_swap", "modality_removal",
    )
    frozen_statistics: dict[str, Any] | None = None
    row_binding = (
        _validate_counterfactual_row_bindings(rows, generation_rows)
        if rows is not None and generation_rows is not None else None
    )
    if rows is None or scientific_protocol is None:
        coverage = {
            mode: bool(
                int((values.get(mode) or {}).get("requested") or 0) > 0
                and int((values.get(mode) or {}).get("n") or 0)
                >= int((values.get(mode) or {}).get("requested") or 0)
                and (values.get(mode) or {}).get("coverage_complete") is True
            )
            for mode in required_modes
        }
    else:
        bootstrap = scientific_protocol["bootstrap"]
        minimums = scientific_protocol["counterfactual_minimum_effective_parents"]
        frozen_statistics = {}
        coverage = {}
        for index, mode in enumerate(required_modes):
            score_values = _parent_counterfactual_values(
                rows, mode, "target_score_delta"
            )
            claim_values = _parent_counterfactual_values(
                rows, mode, "factual_claim_count_delta"
            )
            row_count = sum(str(row.get("mode") or "") == mode for row in rows)
            report_mode = values.get(mode) or {}
            coverage[mode] = bool(
                len(score_values) >= int(minimums[mode])
                and len(score_values) == len(claim_values)
                and int(report_mode.get("n", -1)) == row_count
                and int(report_mode.get("num_effective_parents", -1)) == len(score_values)
                and report_mode.get("aggregation_unit") == "parent"
            )
            frozen_statistics[mode] = {
                "minimum_effective_parents": int(minimums[mode]),
                "num_effective_parents": len(score_values),
                "num_effective_rows": row_count,
                "paired_target_score_delta_ci": bootstrap_mean_ci(
                    score_values,
                    seed=int(bootstrap["seed"]) + 104729 * (index + 1),
                    samples=int(bootstrap["samples"]),
                    confidence=float(bootstrap["confidence"]),
                ),
                "paired_factual_claim_count_delta_ci": bootstrap_mean_ci(
                    claim_values,
                    seed=int(bootstrap["seed"]) + 130363 * (index + 1),
                    samples=int(bootstrap["samples"]),
                    confidence=float(bootstrap["confidence"]),
                ),
            }

    def frozen_upper(mode: str, metric: str) -> float | None:
        if frozen_statistics is None:
            return upper(mode, metric)
        value = ((frozen_statistics.get(mode) or {}).get(metric) or {}).get("high")
        return float(value) if value is not None else None

    checks = {
        "counterfactual_coverage_complete": all(coverage.values()),
        "shuffled_mask_degrades_target_score": (
            frozen_upper("shuffled_mask", "paired_target_score_delta_ci") is not None
            and frozen_upper("shuffled_mask", "paired_target_score_delta_ci") < 0
        ),
        "region_swap_degrades_target_score": (
            frozen_upper("region_swap", "paired_target_score_delta_ci") is not None
            and frozen_upper("region_swap", "paired_target_score_delta_ci") < 0
        ),
        "cross_parent_region_swap_degrades_target_score": (
            frozen_upper(
                "cross_parent_region_swap", "paired_target_score_delta_ci"
            ) is not None
            and frozen_upper(
                "cross_parent_region_swap", "paired_target_score_delta_ci"
            ) < 0
        ),
        "cross_parent_swap_degrades_target_score": (
            frozen_upper("cross_parent_modality_swap", "paired_target_score_delta_ci") is not None
            and frozen_upper("cross_parent_modality_swap", "paired_target_score_delta_ci") < 0
        ),
        "modality_removal_reduces_factual_claims": (
            frozen_upper("modality_removal", "paired_factual_claim_count_delta_ci") is not None
            and frozen_upper("modality_removal", "paired_factual_claim_count_delta_ci") < 0
        ),
    }
    return {
        "checks": checks,
        "coverage_by_mode": coverage,
        "passed": all(checks.values()),
        "counterfactual_sensitivity": values,
        "frozen_parent_statistics": frozen_statistics,
        "row_binding": row_binding,
    }


def _validate_paired_evaluation_reports(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    expert_gate_audit: dict[str, Any],
    expected_seed: int | None = None,
) -> dict[str, Any]:
    paired_fields = (
        "protocol", "stage", "split", "evaluation_mode", "region_protocol",
        "num_samples",
    )
    mismatches = {
        key: (baseline.get(key), candidate.get(key))
        for key in paired_fields
        if baseline.get(key) != candidate.get(key)
    }
    if mismatches:
        raise ValueError(f"paired description eval protocol/population 不一致: {mismatches}")
    if baseline.get("protocol") != DESCRIPTION_EVALUATION_PROTOCOL:
        raise ValueError("paired description eval 不是当前 gate-bound protocol")
    _validate_formal_evaluation_limit(
        baseline,
        observed_count=int(baseline.get("num_samples", -1)),
        label="baseline paired evaluation",
    )
    _validate_formal_evaluation_limit(
        candidate,
        observed_count=int(candidate.get("num_samples", -1)),
        label="candidate paired evaluation",
    )
    if baseline.get("stage") != "bridge_expert":
        raise ValueError("正式 MGRR gate 只接受 expert Bridge stage")
    if baseline.get("evaluation_mode") != "gt_mask":
        raise ValueError("正式 MGRR 主模型准入只接受 GT-mask 配对评价")
    if baseline.get("region_protocol") != "vision_only":
        raise ValueError("正式 MGRR 视觉理解准入只接受 Vision-only 配对评价")
    if (
        baseline.get("expert_gate_audit") != expert_gate_audit
        or candidate.get("expert_gate_audit") != expert_gate_audit
    ):
        raise ValueError("paired description eval 未绑定当前 frozen expert gate")
    baseline_population = (baseline.get("generation_coverage") or {}).get(
        "population_sha256"
    )
    candidate_population = (candidate.get("generation_coverage") or {}).get(
        "population_sha256"
    )
    if not baseline_population or baseline_population != candidate_population:
        raise ValueError("paired description eval 的精确 generation population 不一致")
    baseline_binding = dict(baseline.get("checkpoint_binding") or {})
    candidate_binding = dict(candidate.get("checkpoint_binding") or {})
    baseline_segmentation_source = str(
        (baseline_binding.get("saved_segmentation_migration") or {}).get(
            "source_sha256"
        ) or ""
    )
    candidate_segmentation_source = str(
        (candidate_binding.get("saved_segmentation_migration") or {}).get(
            "source_sha256"
        ) or ""
    )
    if (
        not baseline_segmentation_source
        or baseline_segmentation_source != candidate_segmentation_source
    ):
        raise ValueError(
            "paired description baseline/candidate 必须共享同一 segmentation source"
        )
    baseline_seed = _formal_seed_binding(
        baseline, expected_seed=expected_seed, label="baseline main evaluation",
    ) if expected_seed is not None else None
    candidate_seed = _formal_seed_binding(
        candidate, expected_seed=expected_seed, label="candidate main evaluation",
    ) if expected_seed is not None else None
    training_control = (
        _m4_training_control_audit(
            baseline, candidate, expected_seed=int(expected_seed)
        )
        if expected_seed is not None else None
    )
    return {
        "stage": baseline["stage"],
        "split": baseline["split"],
        "evaluation_mode": baseline["evaluation_mode"],
        "region_protocol": baseline["region_protocol"],
        "population_sha256": baseline_population,
        "num_samples": int(baseline["num_samples"]),
        "segmentation_source_sha256": baseline_segmentation_source,
        "seed_binding": {
            "baseline": baseline_seed,
            "candidate": candidate_seed,
        } if expected_seed is not None else None,
        "training_control": training_control,
    }


def _formal_retrieval_report(
    directory: str | Path,
    *,
    expected_seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = resolve_project_path(directory) or Path(directory)
    report = strict_json_loads(
        (root / "eval_report.json").read_text(encoding="utf-8")
    )
    _validate_formal_evaluation_limit(
        report,
        observed_count=int(report.get("num_samples", -1)),
        label=f"retrieval evaluation: {root}",
    )
    revalidate_evaluation_publication(root, report)
    _validate_evaluation_checkpoint_provenance(root, report)
    retrieval = report.get("same_image_retrieval") or {}
    if (
        report.get("protocol") != DESCRIPTION_EVALUATION_PROTOCOL
        or report.get("stage") != "dior_alignment"
        or report.get("region_protocol") != "vision_only"
        or retrieval.get("protocol") != SAME_IMAGE_RETRIEVAL_PROTOCOL
        or retrieval.get("population_identity_complete") is not True
        or not retrieval.get("population_sha256")
    ):
        raise ValueError(
            f"正式 retrieval gate 需要当前 evaluator 的完整 sample identity: {root}"
        )
    return report, _formal_seed_binding(
        report, expected_seed=expected_seed, label=f"retrieval:{root}",
    )


def _absolute_candidate_gate(
    candidate_report: dict[str, Any],
    candidate_expert: dict[str, Any],
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    generation = candidate_report.get("generation_metrics") or {}
    status = generation.get("target_status") or {}
    per_label = status.get("per_label") or {}
    present_recall = ((per_label.get("present") or {}).get("recall"))
    absent_recall = ((per_label.get("absent") or {}).get("recall"))
    unavailable_rate = candidate_expert.get(
        "unavailable_modality_unsupported_claim_rate"
    )
    values = {
        "expert_fact_score": candidate_expert.get("expert_region_factuality_score"),
        "unsupported_claim_rate": candidate_expert.get("expert_unsupported_claim_rate"),
        "unavailable_unsupported_claim_rate": unavailable_rate,
        "target_status_macro_f1": status.get("macro_f1"),
        "present_recall": present_recall,
        "absent_recall": absent_recall,
        "no_target_rejection": absent_recall,
        "false_description_rate": status.get("false_description_rate"),
        "false_rejection_rate": status.get("positive_false_rejection_rate"),
    }
    minimum_fields = {
        "expert_fact_score", "target_status_macro_f1", "present_recall",
        "absent_recall", "no_target_rejection",
    }
    checks = {}
    for key, value in values.items():
        if value is None:
            checks[key] = False
        elif key in minimum_fields:
            checks[key] = float(value) >= float(thresholds[key])
        else:
            checks[key] = float(value) <= float(thresholds[key])
    if int(candidate_expert.get("unavailable_modality_num_samples") or 0) <= 0:
        checks["unavailable_subset_nonempty"] = False
    else:
        checks["unavailable_subset_nonempty"] = True
    return {
        "values": values,
        "thresholds": {
            key: thresholds[key]
            for key in values
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def compare_description_run_pair(
    baseline_dir: str | Path,
    candidate_dir: str | Path,
    *,
    seed: int,
    frozen_gate: dict[str, Any],
    baseline_retrieval_dir: str | Path,
    candidate_retrieval_dir: str | Path,
    baseline_expert_report: str | Path,
    candidate_expert_report: str | Path,
) -> dict[str, Any]:
    baseline, baseline_report = _rows(
        baseline_dir, require_complete_generation=True
    )
    candidate, candidate_report = _rows(
        candidate_dir, require_complete_generation=True
    )
    paired_evaluation = _validate_paired_evaluation_reports(
        baseline_report,
        candidate_report,
        expert_gate_audit=frozen_gate["audit"],
        expected_seed=seed,
    )
    shared = sorted(set(baseline) & set(candidate))
    if set(baseline) != set(candidate):
        raise ValueError(
            f"paired description samples 不一致: baseline={len(baseline)} "
            f"candidate={len(candidate)} shared={len(shared)}"
        )
    baseline_rows = [baseline[value] for value in shared]
    candidate_rows = [candidate[value] for value in shared]
    baseline_expert_path = resolve_project_path(baseline_expert_report) or Path(baseline_expert_report)
    candidate_expert_path = resolve_project_path(candidate_expert_report) or Path(candidate_expert_report)
    baseline_expert = strict_json_loads(
        baseline_expert_path.read_text(encoding="utf-8")
    )
    candidate_expert = strict_json_loads(
        candidate_expert_path.read_text(encoding="utf-8")
    )
    if baseline_expert.get("protocol") != EXPERT_FACTUALITY_PROTOCOL or candidate_expert.get("protocol") != EXPERT_FACTUALITY_PROTOCOL:
        raise ValueError("正式 MGRR gate 需要当前 source-revalidated ERFS 报告")
    _validate_expert_binding(
        baseline_expert,
        baseline_dir,
        expert_report_path=baseline_expert_path,
        label="baseline",
    )
    _validate_expert_binding(
        candidate_expert,
        candidate_dir,
        expert_report_path=candidate_expert_path,
        label="candidate",
    )
    baseline_parent = baseline_expert.get("per_parent_scores") or {}
    candidate_parent = candidate_expert.get("per_parent_scores") or {}
    if set(baseline_parent) != set(candidate_parent):
        raise ValueError("baseline/candidate ERFS parent 集合不一致")
    expert_parents = sorted(baseline_parent)
    bootstrap = frozen_gate["scientific_protocol"]["bootstrap"]
    ci = paired_bootstrap_delta_ci(
        [float(baseline_parent[value]) for value in expert_parents],
        [float(candidate_parent[value]) for value in expert_parents],
        seed=int(bootstrap["seed"]),
        samples=int(bootstrap["samples"]),
    )
    automatic_baseline_ufcr = _claim_rate(baseline_rows)
    automatic_candidate_ufcr = _claim_rate(candidate_rows)
    baseline_ufcr = float(baseline_expert.get("expert_unsupported_claim_rate") or 0.0)
    candidate_ufcr = float(candidate_expert.get("expert_unsupported_claim_rate") or 0.0)
    baseline_retrieval, baseline_retrieval_seed = _formal_retrieval_report(
        baseline_retrieval_dir, expected_seed=seed,
    )
    candidate_retrieval, candidate_retrieval_seed = _formal_retrieval_report(
        candidate_retrieval_dir, expected_seed=seed,
    )
    retrieval_paired_fields = ("split", "evaluation_mode", "region_protocol")
    retrieval_mismatches = {
        key: (baseline_retrieval.get(key), candidate_retrieval.get(key))
        for key in retrieval_paired_fields
        if baseline_retrieval.get(key) != candidate_retrieval.get(key)
    }
    baseline_retrieval_payload = baseline_retrieval.get("same_image_retrieval") or {}
    candidate_retrieval_payload = candidate_retrieval.get("same_image_retrieval") or {}
    baseline_retrieval_source = str(
        ((baseline_retrieval.get("checkpoint_binding") or {}).get(
            "saved_segmentation_migration"
        ) or {}).get("source_sha256") or ""
    )
    candidate_retrieval_source = str(
        ((candidate_retrieval.get("checkpoint_binding") or {}).get(
            "saved_segmentation_migration"
        ) or {}).get("source_sha256") or ""
    )
    if (
        retrieval_mismatches
        or not baseline_retrieval_source
        or baseline_retrieval_source != candidate_retrieval_source
        or baseline_retrieval_payload.get("population_sha256")
        != candidate_retrieval_payload.get("population_sha256")
    ):
        raise ValueError(
            f"paired retrieval protocol/population 不一致: {retrieval_mismatches}"
        )
    baseline_r1 = baseline_retrieval_payload.get("mean_r1")
    candidate_r1 = candidate_retrieval_payload.get("mean_r1")
    baseline_retrieval_parent = (
        baseline_retrieval_payload.get("per_parent_mean_r1") or {}
    )
    candidate_retrieval_parent = (
        candidate_retrieval_payload.get("per_parent_mean_r1") or {}
    )
    if not baseline_retrieval_parent or set(baseline_retrieval_parent) != set(candidate_retrieval_parent):
        raise ValueError(
            "正式 MGRR gate 需要相同 parent 的 per_parent_mean_r1；请用当前 evaluator 重跑 DIOR"
        )
    retrieval_parents = sorted(baseline_retrieval_parent)
    retrieval_ci = paired_bootstrap_delta_ci(
        [float(baseline_retrieval_parent[value]) for value in retrieval_parents],
        [float(candidate_retrieval_parent[value]) for value in retrieval_parents],
        seed=int(bootstrap["seed"]) + 15485863,
        samples=int(bootstrap["samples"]),
    )
    retrieval_improved = (
        baseline_r1 is not None and candidate_r1 is not None
        and float(candidate_r1) > float(baseline_r1)
        and retrieval_ci["low"] is not None
        and float(retrieval_ci["low"]) > 0
    )
    counterfactual_gate = _counterfactual_gate(
        candidate_report,
        _load_counterfactual_rows(candidate_dir),
        frozen_gate["scientific_protocol"],
        candidate,
    )
    absolute_gate = _absolute_candidate_gate(
        candidate_report, candidate_expert, frozen_gate["thresholds"]
    )
    unsupported_noninferiority = float(
        frozen_gate["thresholds"]["unsupported_claim_rate_noninferiority"]
    )
    passed = (
        ci["low"] is not None
        and float(ci["low"]) > 0
        and retrieval_improved
        and candidate_ufcr <= baseline_ufcr + float(unsupported_noninferiority)
        and counterfactual_gate["passed"]
        and absolute_gate["passed"]
    )
    return {
        "seed": seed,
        "frozen_gate_audit": frozen_gate["audit"],
        "paired_evaluation": paired_evaluation,
        "num_paired_samples": len(shared),
        "num_expert_parents": len(expert_parents),
        "expert_region_factuality_delta_ci": ci,
        "baseline_unsupported_claim_rate": baseline_ufcr,
        "candidate_unsupported_claim_rate": candidate_ufcr,
        "automatic_proxy_baseline_unsupported_claim_rate": automatic_baseline_ufcr,
        "automatic_proxy_candidate_unsupported_claim_rate": automatic_candidate_ufcr,
        "unsupported_noninferiority": unsupported_noninferiority,
        "baseline_same_image_r1": baseline_r1,
        "candidate_same_image_r1": candidate_r1,
        "num_retrieval_parents": len(retrieval_parents),
        "retrieval_population_sha256": baseline_retrieval_payload[
            "population_sha256"
        ],
        "retrieval_segmentation_source_sha256": baseline_retrieval_source,
        "same_image_r1_delta_ci": retrieval_ci,
        "retrieval_improved": retrieval_improved,
        "artifact_seed_binding": {
            "expected_seed": int(seed),
            "main_evaluation": paired_evaluation["seed_binding"],
            "retrieval_evaluation": {
                "baseline": baseline_retrieval_seed,
                "candidate": candidate_retrieval_seed,
            },
        },
        "counterfactual_gate": counterfactual_gate,
        "absolute_candidate_gate": absolute_gate,
        "passed": passed,
    }


def _validate_three_seed_artifact_uniqueness(
    pairs: list[dict[str, Any]],
) -> dict[str, list[str]]:
    """Reject the same trained artifact being relabelled into multiple seed slots."""
    roles = {
        "baseline_main": ("main_evaluation", "baseline"),
        "candidate_main": ("main_evaluation", "candidate"),
        "baseline_retrieval": ("retrieval_evaluation", "baseline"),
        "candidate_retrieval": ("retrieval_evaluation", "candidate"),
    }
    fingerprints: dict[str, list[str]] = {}
    for role, (family, side) in roles.items():
        values = [
            str(
                (((pair.get("artifact_seed_binding") or {}).get(family) or {}).get(side) or {}).get(
                    "checkpoint_sha256"
                )
                or ""
            )
            for pair in pairs
        ]
        if any(len(value) != 64 for value in values):
            raise ValueError(f"三 seed gate 缺少 {role} checkpoint fingerprint")
        if len(set(values)) != len(values):
            raise ValueError(f"三 seed gate 检测到重复 {role} checkpoint，禁止重复 run 改标签")
        fingerprints[role] = values
    return fingerprints


def compare_description_seeds(
    baseline_dirs: list[str],
    candidate_dirs: list[str],
    *,
    seeds: list[int],
    bridge_benchmark: str | Path,
    baseline_retrieval_dirs: list[str],
    candidate_retrieval_dirs: list[str],
    baseline_expert_reports: list[str],
    candidate_expert_reports: list[str],
) -> dict[str, Any]:
    if not (
        len(baseline_dirs) == len(candidate_dirs) == len(seeds)
        == len(baseline_retrieval_dirs) == len(candidate_retrieval_dirs)
        == len(baseline_expert_reports) == len(candidate_expert_reports)
    ):
        raise ValueError("description/retrieval baseline/candidate/seeds 数量必须一致")
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError("正式 MGRR gate 必须提供三个不同 seed 的配对运行")
    bridge_dir = resolve_project_path(bridge_benchmark) or Path(bridge_benchmark)
    frozen_gate = load_frozen_scientific_gate(bridge_dir)
    pairs = [
        compare_description_run_pair(
            baseline, candidate, seed=seed,
            frozen_gate=frozen_gate,
            baseline_retrieval_dir=baseline_retrieval,
            candidate_retrieval_dir=candidate_retrieval,
            baseline_expert_report=baseline_expert,
            candidate_expert_report=candidate_expert,
        )
        for baseline, candidate, seed, baseline_retrieval, candidate_retrieval, baseline_expert, candidate_expert in zip(
            baseline_dirs, candidate_dirs, seeds,
            baseline_retrieval_dirs, candidate_retrieval_dirs,
            baseline_expert_reports, candidate_expert_reports,
        )
    ]
    artifact_fingerprints = _validate_three_seed_artifact_uniqueness(pairs)
    main_populations = {
        pair["paired_evaluation"]["population_sha256"] for pair in pairs
    }
    retrieval_populations = {
        pair["retrieval_population_sha256"] for pair in pairs
    }
    scientific_configs = {
        pair["paired_evaluation"]["training_control"][
            "cross_seed_scientific_config_sha256"
        ]
        for pair in pairs
    }
    training_populations = {
        pair["paired_evaluation"]["training_control"][
            "cross_seed_training_population_sha256"
        ]
        for pair in pairs
    }
    if len(main_populations) != 1 or len(retrieval_populations) != 1:
        raise ValueError("M4 三 seed evaluation/retrieval population 不一致")
    if len(scientific_configs) != 1 or len(training_populations) != 1:
        raise ValueError("M4 三 seed scientific config/training population 不一致")
    required = 2
    passed = sum(int(value["passed"]) for value in pairs)
    def resolved(value: str | Path) -> str:
        return str((resolve_project_path(value) or Path(value)).resolve(strict=False))

    return {
        "protocol": M4_SEED_GATE_PROTOCOL,
        "inputs": {
            "baseline_dirs": [resolved(value) for value in baseline_dirs],
            "candidate_dirs": [resolved(value) for value in candidate_dirs],
            "seeds": [int(value) for value in seeds],
            "bridge_benchmark": resolved(bridge_dir),
            "baseline_retrieval_dirs": [
                resolved(value) for value in baseline_retrieval_dirs
            ],
            "candidate_retrieval_dirs": [
                resolved(value) for value in candidate_retrieval_dirs
            ],
            "baseline_expert_reports": [
                resolved(value) for value in baseline_expert_reports
            ],
            "candidate_expert_reports": [
                resolved(value) for value in candidate_expert_reports
            ],
        },
        "frozen_gate_audit": frozen_gate["audit"],
        "scientific_protocol": frozen_gate["scientific_protocol"],
        "thresholds": frozen_gate["thresholds"],
        "pairs": pairs,
        "artifact_checkpoint_fingerprints": artifact_fingerprints,
        "same_evaluation_population_across_seeds": True,
        "same_retrieval_population_across_seeds": True,
        "same_scientific_config_across_seeds": True,
        "same_training_population_across_seeds": True,
        "cross_seed_training_population_sha256": next(
            iter(training_populations)
        ),
        "num_passed": passed,
        "required_passed": required,
        "passed_2_of_3_gate": passed >= 2,
        "passed_fraction_gate": passed >= required,
    }


def validate_m4_seed_gate(path_ref: str | Path) -> tuple[Path, dict[str, Any]]:
    """Recompute one baseline-vs-MGRR three-seed gate from bound raw inputs."""
    path = resolve_project_path(path_ref) or Path(path_ref)
    if not path.is_file():
        raise FileNotFoundError(f"M4 seed gate 不存在: {path}")
    payload = strict_json_loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("protocol") != M4_SEED_GATE_PROTOCOL:
        raise ValueError("M4 seed gate protocol 不兼容")
    inputs = payload.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("M4 seed gate 缺少可重算 input bindings")
    rebuilt = compare_description_seeds(
        list(inputs.get("baseline_dirs") or []),
        list(inputs.get("candidate_dirs") or []),
        seeds=[int(value) for value in (inputs.get("seeds") or [])],
        bridge_benchmark=str(inputs.get("bridge_benchmark") or ""),
        baseline_retrieval_dirs=list(
            inputs.get("baseline_retrieval_dirs") or []
        ),
        candidate_retrieval_dirs=list(
            inputs.get("candidate_retrieval_dirs") or []
        ),
        baseline_expert_reports=list(
            inputs.get("baseline_expert_reports") or []
        ),
        candidate_expert_reports=list(
            inputs.get("candidate_expert_reports") or []
        ),
    )
    if rebuilt != payload:
        raise ValueError("M4 seed gate 与绑定原始评估的重新计算结果不一致")
    return path.resolve(strict=False), rebuilt


def aggregate_m4_region_encoder_reports(
    reports: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Require all five preregistered baselines against one MGRR candidate set."""
    if set(reports) != M4_BASELINE_REGION_ENCODERS:
        raise ValueError(
            "M4 suite 必须恰好覆盖五种 baseline: "
            f"expected={sorted(M4_BASELINE_REGION_ENCODERS)} "
            f"observed={sorted(reports)}"
        )
    frozen_audits = set()
    candidate_fingerprints = set()
    candidate_retrieval_fingerprints = set()
    seeds = set()
    training_populations = set()
    failures = []
    for encoder, report in sorted(reports.items()):
        if report.get("protocol") != M4_SEED_GATE_PROTOCOL:
            raise ValueError(f"M4 suite {encoder} gate protocol 不兼容")
        pairs = report.get("pairs")
        if not isinstance(pairs, list) or len(pairs) != 3:
            raise ValueError(f"M4 suite {encoder} 必须包含三个 seed pair")
        observed_baselines = {
            ((pair.get("paired_evaluation") or {}).get("training_control") or {}).get(
                "baseline_region_encoder"
            )
            for pair in pairs
        }
        observed_candidates = {
            ((pair.get("paired_evaluation") or {}).get("training_control") or {}).get(
                "candidate_region_encoder"
            )
            for pair in pairs
        }
        if observed_baselines != {encoder} or observed_candidates != {"mgrr"}:
            raise ValueError(f"M4 suite {encoder} encoder identity 与 gate 不一致")
        if not all(
            report.get(name) is True
            for name in (
                "same_evaluation_population_across_seeds",
                "same_retrieval_population_across_seeds",
                "same_scientific_config_across_seeds",
                "same_training_population_across_seeds",
            )
        ):
            raise ValueError(f"M4 suite {encoder} 跨 seed 可比性未通过")
        frozen_audits.add(_canonical_sha256(report.get("frozen_gate_audit")))
        fingerprints = report.get("artifact_checkpoint_fingerprints") or {}
        candidate_fingerprints.add(tuple(fingerprints.get("candidate_main") or []))
        candidate_retrieval_fingerprints.add(tuple(
            fingerprints.get("candidate_retrieval") or []
        ))
        seeds.add(tuple(int(pair.get("seed", -1)) for pair in pairs))
        training_population_sha256 = str(
            report.get("cross_seed_training_population_sha256") or ""
        )
        if len(training_population_sha256) != 64:
            raise ValueError(
                f"M4 suite {encoder} 缺少跨 seed training population hash"
            )
        training_populations.add(training_population_sha256)
        if report.get("passed_fraction_gate") is not True:
            failures.append(encoder)
    if len(frozen_audits) != 1:
        raise ValueError("M4 suite baseline gates 未绑定同一个 frozen Bridge")
    if (
        len(candidate_fingerprints) != 1
        or len(candidate_retrieval_fingerprints) != 1
        or len(seeds) != 1
        or len(training_populations) != 1
    ):
        raise ValueError(
            "M4 suite 未复用同一组三 seed full-MGRR candidate artifacts/population"
        )
    return {
        "protocol": M4_SUITE_GATE_PROTOCOL,
        "required_baselines": sorted(M4_BASELINE_REGION_ENCODERS),
        "candidate_region_encoder": "mgrr",
        "seeds": list(next(iter(seeds))),
        "same_frozen_bridge": True,
        "same_candidate_artifacts": True,
        "same_training_population": True,
        "cross_seed_training_population_sha256": next(
            iter(training_populations)
        ),
        "frozen_gate_audit": next(iter(reports.values()))[
            "frozen_gate_audit"
        ],
        "candidate_main_checkpoint_sha256": list(
            next(iter(candidate_fingerprints))
        ),
        "candidate_retrieval_checkpoint_sha256": list(
            next(iter(candidate_retrieval_fingerprints))
        ),
        "num_baselines": len(reports),
        "num_passed": len(reports) - len(failures),
        "failed_baselines": failures,
        "passed": not failures,
    }


def build_m4_region_encoder_suite(
    gate_paths: dict[str, str | Path],
) -> dict[str, Any]:
    validated: dict[str, dict[str, Any]] = {}
    bindings: dict[str, dict[str, str]] = {}
    for encoder, path_ref in sorted(gate_paths.items()):
        path, report = validate_m4_seed_gate(path_ref)
        validated[encoder] = report
        bindings[encoder] = {
            "gate": str(path),
            "gate_sha256": _sha256(path),
        }
    result = aggregate_m4_region_encoder_reports(validated)
    result["comparison_gate_bindings"] = bindings
    return result


def validate_m4_region_encoder_suite_gate(
    path_ref: str | Path,
) -> tuple[Path, dict[str, Any]]:
    """Deep-recompute a published five-baseline suite gate."""
    path = resolve_project_path(path_ref) or Path(path_ref)
    if not path.is_file():
        raise FileNotFoundError(f"M4 suite gate 不存在: {path}")
    payload = strict_json_loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("protocol") != M4_SUITE_GATE_PROTOCOL:
        raise ValueError("M4 suite gate protocol 不兼容")
    bindings = payload.get("comparison_gate_bindings")
    if not isinstance(bindings, dict) or set(bindings) != M4_BASELINE_REGION_ENCODERS:
        raise ValueError("M4 suite gate 的五 baseline bindings 不完整")
    rebuilt = build_m4_region_encoder_suite({
        encoder: str((binding or {}).get("gate") or "")
        for encoder, binding in bindings.items()
    })
    if rebuilt != payload:
        raise ValueError("M4 suite gate 与绑定 comparison gates 的重新计算结果不一致")
    return path.resolve(strict=False), rebuilt
