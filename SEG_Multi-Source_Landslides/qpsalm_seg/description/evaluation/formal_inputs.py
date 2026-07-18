"""Formal M4 input, checkpoint, seed and population bindings."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

from qpsalm_seg.paths import resolve_project_path

from ..protocols.config import (
    require_serialized_segdesc_config,
    serialized_segdesc_config_value,
    serialized_segdesc_config_without,
)
from ..data.engineering_contracts import REGION_TRAINING_DATA_PROTOCOL
from ..protocols.io import (
    canonical_sha256 as _canonical_sha256,
    sha256_file as _sha256,
    strict_json_loads,
)
from ..protocols.stages import DESCRIPTION_STREAM_SEED_OFFSETS
from ..protocols.versions import DESCRIPTION_TRAINING_COMPLETION_PROTOCOL
from ..training.checkpoint import (
    description_protocol_assets_spec,
    inspect_segdesc_checkpoint,
    validate_description_stage_lineage,
)
from ..training.run_artifacts import validate_checkpoint_run_completion
from ..data.vision_cache import revalidate_description_cache_artifact
from .artifacts import revalidate_evaluation_mask_artifacts
from .contracts import (
    DESCRIPTION_EVALUATION_PROTOCOL,
    EVALUATION_CHECKPOINT_BINDING_PROTOCOL,
)
from .expert_factuality import revalidate_expert_factuality_report
from .publication import (
    evaluation_population_sha256,
    revalidate_evaluation_publication,
)

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


def m4_dataset_population_audit(
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


def m4_stream_loader_contract(
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


def m4_cross_seed_training_population_contract(
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
        name: DESCRIPTION_STREAM_SEED_OFFSETS[name]
        for name in expected_stages
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
        dataset_contracts[name] = m4_dataset_population_audit(
            streams[name],
            label=f"stream={name}",
            expected_stage=stage,
            expected_split="train",
        )
        loader_contracts[name] = m4_stream_loader_contract(
            loaders[name],
            label=f"stream={name}",
            stream=name,
            expected_stage=stage,
            expected_seed=int(expected_seed) + expected_loader_offsets[name],
            dataset_audit=dataset_contracts[name],
        )
    validation = m4_dataset_population_audit(
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


def m4_training_control_audit(
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
    baseline_config = require_serialized_segdesc_config(
        baseline_metadata.get("config"), label="M4 baseline checkpoint config"
    )
    candidate_config = require_serialized_segdesc_config(
        candidate_metadata.get("config"), label="M4 candidate checkpoint config"
    )
    baseline_encoder = serialized_segdesc_config_value(
        baseline_config, "region_encoder"
    )
    candidate_encoder = serialized_segdesc_config_value(
        candidate_config, "region_encoder"
    )
    if (
        baseline_encoder not in M4_BASELINE_REGION_ENCODERS
        or candidate_encoder != "mgrr"
    ):
        raise ValueError(
            "正式 M4 配对必须是预注册 baseline region encoder 对 full MGRR"
        )
    baseline_controlled = serialized_segdesc_config_without(
        baseline_config,
        M4_VARIANT_CONFIG_FIELDS,
        label="M4 baseline checkpoint config",
    )
    candidate_controlled = serialized_segdesc_config_without(
        candidate_config,
        M4_VARIANT_CONFIG_FIELDS,
        label="M4 candidate checkpoint config",
    )
    if baseline_controlled != candidate_controlled:
        raise ValueError("M4 baseline/candidate 除 region encoder 外训练配置不一致")
    try:
        checkpoint_seed = int(serialized_segdesc_config_value(
            baseline_config, "seed"
        ))
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
    cross_seed_population = m4_cross_seed_training_population_contract(
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


def validate_expert_binding(
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


def validate_evaluation_checkpoint_provenance(
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
    seed_binding = formal_seed_binding(
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


def formal_seed_binding(
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
    saved_config = require_serialized_segdesc_config(
        metadata.get("config"), label=f"formal artifact {label} checkpoint config"
    )
    statistics = dict(report.get("statistics_protocol") or {})
    values = {
        "expected_seed": expected_seed,
        "checkpoint_training_seed": binding.get("checkpoint_training_seed"),
        "checkpoint_config_seed": serialized_segdesc_config_value(
            saved_config, "seed"
        ),
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


def validate_formal_evaluation_limit(
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


def load_evaluation_rows(
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
        validate_formal_evaluation_limit(
            report, observed_count=len(ordered), label=f"paired gate: {root}"
        )
        revalidate_evaluation_publication(root, report)
        validate_evaluation_checkpoint_provenance(root, report)
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
