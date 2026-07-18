"""Atomic description-evaluation publication and checkpoint binding."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

import numpy as np

from qpsalm_seg.paths import resolve_project_path

from ..protocols.config import (
    SegDescConfig,
    require_serialized_segdesc_config,
    serialized_segdesc_config_value,
)
from ..protocols.io import canonical_sha256, sha256_file, strict_json_loads
from ..protocols.gates import structured_generation_audits_current
from ..protocols.versions import STRUCTURED_GENERATION_PROTOCOL
from ..training.checkpoint import validate_description_stage_lineage
from .contracts import (
    DESCRIPTION_EVALUATION_PROTOCOL,
    EVALUATION_CHECKPOINT_BINDING_PROTOCOL,
    EVALUATION_POPULATION_FIELDS,
    EVALUATION_PUBLICATION_PROTOCOL,
)
from .d_minus_one import revalidate_saved_d_minus_one_acceptance


def evaluation_population_sha256(rows: list[dict[str, Any]]) -> str:
    """Hash the exact generated sample/target/region population, not model text."""
    identities = [
        {key: row.get(key) for key in EVALUATION_POPULATION_FIELDS}
        for row in rows
    ]
    sample_ids = [str(value.get("sample_id") or "") for value in identities]
    if any(not value for value in sample_ids):
        raise ValueError("description evaluation population 存在空 sample_id")
    if len(sample_ids) != len(set(sample_ids)):
        duplicates = sorted(
            value for value, count in Counter(sample_ids).items() if count > 1
        )
        raise ValueError(f"description evaluation population 存在重复 sample_id: {duplicates[:8]}")
    return canonical_sha256(sorted(identities, key=lambda value: str(value["sample_id"])))


def _publication_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"description evaluation publication 缺少 {label}: {path}")
    try:
        rows = [
            strict_json_loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"description evaluation publication 的 {label} 不是严格 JSONL"
        ) from exc
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(
            f"description evaluation publication 的 {label} 每行必须是 object"
        )
    return rows


def _publication_count(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"description evaluation publication {label} 必须是非负整数")
    return int(value)


def _publication_file_binding(
    root: Path,
    relative_path: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    path = (root / relative_path).resolve(strict=False)
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("description evaluation publication artifact 逃逸输出目录") from exc
    return {
        "path": relative_path,
        "sha256": sha256_file(path),
        "bytes": int(path.stat().st_size),
        "records": len(rows),
    }


def build_evaluation_publication_audit(
    output_dir: str | Path,
    report: dict[str, Any],
) -> dict[str, Any]:
    """Reopen every standalone-evaluation artifact before publishing the report.

    The report hash deliberately excludes ``publication_audit`` so the audit can
    live inside the atomically written final report without a self-hash cycle.
    """
    root = (resolve_project_path(output_dir) or Path(output_dir)).resolve(
        strict=False
    )
    if report.get("protocol") != DESCRIPTION_EVALUATION_PROTOCOL:
        raise ValueError("description evaluation publication protocol 不兼容")
    payload = {
        key: value for key, value in report.items()
        if key != "publication_audit"
    }
    raw_path = root / "raw_generations.jsonl"
    counterfactual_path = root / "counterfactual_generations.jsonl"
    raw_rows = _publication_jsonl(raw_path, label="raw_generations")
    counterfactual_rows = _publication_jsonl(
        counterfactual_path, label="counterfactual_generations"
    )

    num_samples = _publication_count(report.get("num_samples"), label="num_samples")
    num_generated = _publication_count(
        report.get("num_generated"), label="num_generated"
    )
    coverage = report.get("generation_coverage")
    if not isinstance(coverage, dict):
        raise ValueError("description evaluation publication 缺少 generation_coverage")
    eligible = _publication_count(
        coverage.get("eligible_samples"), label="eligible_samples"
    )
    generated = _publication_count(
        coverage.get("generated_samples"), label="generated_samples"
    )
    sample_ids = [str(row.get("sample_id") or "") for row in raw_rows]
    if any(not value for value in sample_ids) or len(sample_ids) != len(set(sample_ids)):
        raise ValueError(
            "description evaluation publication raw sample_id 必须非空且唯一"
        )
    expected_fraction = len(raw_rows) / max(num_samples, 1)
    fraction = coverage.get("fraction")
    if (
        isinstance(fraction, bool)
        or not isinstance(fraction, (int, float))
        or not np.isfinite(float(fraction))
        or not np.isclose(
            float(fraction), expected_fraction, rtol=0.0, atol=1.0e-12
        )
        or num_samples != eligible
        or num_generated != len(raw_rows)
        or generated != len(raw_rows)
        or coverage.get("complete") is not (len(raw_rows) == num_samples)
        or coverage.get("population_sha256")
        != evaluation_population_sha256(raw_rows)
        or coverage.get("population_identity_fields")
        != list(EVALUATION_POPULATION_FIELDS)
    ):
        raise ValueError(
            "description evaluation publication generation population/count 绑定不一致"
        )

    raw_by_sample = {str(row["sample_id"]): row for row in raw_rows}
    structured_rows = [
        row for row in raw_rows if row.get("structured_output") is True
    ]
    if structured_rows and not structured_generation_audits_current(
        structured_rows
    ):
        raise ValueError(
            "description evaluation publication structured generation audit 非法"
        )
    counterfactual_keys: set[tuple[str, str]] = set()
    observed_counterfactual_counts: Counter[str] = Counter()
    for row in counterfactual_rows:
        sample_id = str(row.get("sample_id") or "")
        mode = str(row.get("mode") or "")
        key = (sample_id, mode)
        if (
            sample_id not in raw_by_sample
            or not mode
            or key in counterfactual_keys
            or (
                row.get("parent_sample_id") is not None
                and str(row.get("parent_sample_id"))
                != str(raw_by_sample[sample_id].get("parent_sample_id"))
            )
        ):
            raise ValueError(
                "description evaluation publication counterfactual identity 非法"
            )
        counterfactual_keys.add(key)
        observed_counterfactual_counts[mode] += 1
    sensitivity = report.get("counterfactual_sensitivity") or {}
    if not isinstance(sensitivity, dict):
        raise ValueError(
            "description evaluation publication counterfactual summary 必须是 object"
        )
    expected_counterfactual_counts: dict[str, int] = {}
    for mode, summary in sensitivity.items():
        if not isinstance(summary, dict):
            raise ValueError(
                "description evaluation publication counterfactual mode summary 非法"
            )
        expected_counterfactual_counts[str(mode)] = _publication_count(
            summary.get("n"), label=f"counterfactual_sensitivity.{mode}.n"
        )
    if (
        set(observed_counterfactual_counts) - set(expected_counterfactual_counts)
        or any(
            observed_counterfactual_counts.get(mode, 0) != count
            for mode, count in expected_counterfactual_counts.items()
        )
    ):
        raise ValueError(
            "description evaluation publication counterfactual 行数与报告不一致"
        )

    artifacts = {
        "raw_generations": _publication_file_binding(
            root, "raw_generations.jsonl", raw_rows
        ),
        "counterfactual_generations": _publication_file_binding(
            root, "counterfactual_generations.jsonl", counterfactual_rows
        ),
    }
    optional_specs = (
        (
            "end_to_end_target_audit",
            "end_to_end_target_audit.jsonl",
            report.get("end_to_end_coverage") is not None,
        ),
        (
            "cycle_localization",
            "cycle_localization.jsonl",
            report.get("cycle_localization") is not None,
        ),
    )
    for name, relative_path, required in optional_specs:
        path = root / relative_path
        if required:
            rows = _publication_jsonl(path, label=name)
            ids = [str(row.get("sample_id") or row.get("bridge_sample_id") or "") for row in rows]
            if any(not value for value in ids) or len(ids) != len(set(ids)):
                raise ValueError(
                    f"description evaluation publication {name} sample identity 非法"
                )
            if set(ids) - set(raw_by_sample):
                raise ValueError(
                    f"description evaluation publication {name} 超出 generation population"
                )
            if name == "end_to_end_target_audit" and len(rows) != len(raw_rows):
                raise ValueError(
                    "description evaluation publication end-to-end audit 未覆盖 generation"
                )
            if name == "cycle_localization" and len(rows) != _publication_count(
                (report.get("cycle_localization") or {}).get("evaluated_samples"),
                label="cycle_localization.evaluated_samples",
            ):
                raise ValueError(
                    "description evaluation publication cycle 行数与报告不一致"
                )
            artifacts[name] = _publication_file_binding(root, relative_path, rows)
        elif path.exists():
            raise ValueError(
                f"description evaluation publication 存在未声明的 {relative_path}"
            )

    checkpoint_ref = report.get("checkpoint")
    checkpoint = resolve_project_path(str(checkpoint_ref or ""))
    if checkpoint is None or not checkpoint.is_file():
        raise FileNotFoundError(
            "description evaluation publication checkpoint 不存在"
        )
    checkpoint_sha256 = str(report.get("checkpoint_sha256") or "")
    checkpoint_step = report.get("checkpoint_step")
    if checkpoint_sha256 != sha256_file(checkpoint):
        raise ValueError(
            "description evaluation publication checkpoint path/hash 已漂移"
        )
    if (
        isinstance(checkpoint_step, bool)
        or not isinstance(checkpoint_step, int)
        or checkpoint_step < 0
        or not isinstance(report.get("checkpoint_metadata"), dict)
        or not isinstance(report.get("checkpoint_binding"), dict)
        or (report.get("checkpoint_binding") or {}).get("protocol")
        != EVALUATION_CHECKPOINT_BINDING_PROTOCOL
    ):
        raise ValueError(
            "description evaluation publication checkpoint/metadata 绑定不一致"
        )
    if (root / "failure_report.json").exists():
        raise ValueError(
            "description evaluation publication 目录同时存在 failure_report"
        )
    temporary_files = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.suffix in {".tmp", ".part"}
    )
    if temporary_files:
        raise ValueError(
            "description evaluation publication 残留临时文件: "
            f"{temporary_files[:8]}"
        )
    return {
        "protocol": EVALUATION_PUBLICATION_PROTOCOL,
        "terminal_status": "published",
        "report_payload_sha256": canonical_sha256(payload),
        "checkpoint": str(checkpoint_ref),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_step": int(checkpoint_step),
        "population_sha256": coverage["population_sha256"],
        "num_samples": num_samples,
        "num_generated": num_generated,
        "structured_generation_protocol": (
            STRUCTURED_GENERATION_PROTOCOL if structured_rows else None
        ),
        "num_structured_generations": len(structured_rows),
        "artifacts": artifacts,
    }


def revalidate_evaluation_publication(
    output_dir: str | Path,
    report: dict[str, Any],
) -> dict[str, Any]:
    """Rebuild and compare the terminal publication audit from disk."""
    observed = report.get("publication_audit")
    if not isinstance(observed, dict):
        raise ValueError(
            "formal description evaluation 缺少 terminal publication audit"
        )
    root = (resolve_project_path(output_dir) or Path(output_dir)).resolve(
        strict=False
    )
    artifacts = observed.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError(
            "formal description evaluation publication artifact/report 已漂移"
        )
    # 先重放已发布的文件绑定，使文件被篡改与报告内部自相矛盾保持可区分。
    for binding in artifacts.values():
        if not isinstance(binding, dict):
            raise ValueError(
                "formal description evaluation publication artifact/report 已漂移"
            )
        relative = str(binding.get("path") or "")
        path = (root / relative).resolve(strict=False)
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                "formal description evaluation publication artifact/report 已漂移"
            ) from exc
        if (
            not relative
            or not path.is_file()
            or str(binding.get("sha256") or "") != sha256_file(path)
            or binding.get("bytes") != int(path.stat().st_size)
        ):
            raise ValueError(
                "formal description evaluation publication artifact/report 已漂移"
            )
    rebuilt = build_evaluation_publication_audit(output_dir, report)
    if observed != rebuilt:
        raise ValueError(
            "formal description evaluation publication artifact/report 已漂移"
        )
    return rebuilt


def validate_evaluation_checkpoint_binding(
    config: SegDescConfig,
    checkpoint_report: dict[str, Any],
    runtime_segmentation_migration: dict[str, Any],
    predicted_index_audit: dict[str, Any] | None,
    *,
    checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    """Bind evaluation data mode to the intended trained stage and segmenter."""
    metadata = dict(checkpoint_report.get("metadata") or {})
    checkpoint_stage = str(metadata.get("stage") or "")
    saved_config = require_serialized_segdesc_config(
        metadata.get("config"), label="description evaluation checkpoint config"
    )
    saved_seed = serialized_segdesc_config_value(saved_config, "seed")
    if saved_seed is None or int(saved_seed) != int(config.training.seed):
        raise RuntimeError(
            "description evaluation seed 与 checkpoint 训练 seed 不一致: "
            f"evaluation={int(config.training.seed)} checkpoint={saved_seed!r}"
        )
    expected_checkpoint_stage = (
        "predicted_mask" if config.evaluation.evaluation_mode == "end_to_end" else config.training.stage
    )
    if checkpoint_stage != expected_checkpoint_stage:
        raise RuntimeError(
            "description evaluation checkpoint stage 非法: "
            f"mode={config.evaluation.evaluation_mode} data_stage={config.training.stage} "
            f"expected={expected_checkpoint_stage} observed={checkpoint_stage}"
        )
    expected_checkpoint_role = (
        "terminal_last"
        if checkpoint_stage in {"overfit", "bridge_auto"}
        else "validation_best"
    )
    observed_checkpoint_role = metadata.get("checkpoint_role")
    if observed_checkpoint_role != expected_checkpoint_role:
        raise RuntimeError(
            "description evaluation checkpoint role 非法: "
            f"stage={checkpoint_stage!r} expected={expected_checkpoint_role!r} "
            f"observed={observed_checkpoint_role!r}"
        )
    run_completion = None
    if checkpoint is not None:
        from ..protocols.versions import DESCRIPTION_TRAINING_COMPLETION_PROTOCOL
        from ..training.run_artifacts import validate_checkpoint_run_completion
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
            raise RuntimeError(
                "description evaluation checkpoint 所属训练 run 未成功完成"
            ) from exc
    d_minus_one_acceptance = None
    stage_lineage = None
    if checkpoint_stage != "overfit":
        d_minus_one_acceptance = revalidate_saved_d_minus_one_acceptance(
            metadata.get("d_minus_one_acceptance"),
            expected_description_benchmark=config.data.description_benchmark,
            expected_bridge_benchmark=getattr(
                config.data, "bridge_benchmark", None
            ),
            expected_unified_benchmark=getattr(
                config.data, "unified_benchmark", None
            ),
            expected_description_cache=getattr(
                getattr(config, "model", None),
                "description_vision_cache",
                None,
            ),
        )
        if checkpoint_stage != "mmrs_caption":
            stage_lineage = validate_description_stage_lineage(
                metadata.get("stage_lineage"),
                expected_target_stage=checkpoint_stage,
            )
    saved_migration = dict(checkpoint_report.get("segmentation_migration") or {})
    saved_source_sha = str(saved_migration.get("source_sha256") or "")
    runtime_source_sha = str(runtime_segmentation_migration.get("source_sha256") or "")
    if not saved_source_sha or saved_source_sha != runtime_source_sha:
        raise RuntimeError(
            "description evaluation 当前 segmentation source 与 checkpoint lineage 不一致"
        )
    fixed_prediction_match: bool | None = None
    if config.evaluation.evaluation_mode == "fixed_prediction":
        audit = dict(predicted_index_audit or {})
        predicted_source_sha = str(
            audit.get("segmentation_checkpoint_sha256") or ""
        )
        fixed_prediction_match = bool(
            predicted_source_sha and predicted_source_sha == saved_source_sha
        )
        if not fixed_prediction_match:
            raise RuntimeError(
                "fixed prediction masks 必须由 description checkpoint 绑定的同一 segmentation "
                "checkpoint 生成"
            )
    return {
        "protocol": EVALUATION_CHECKPOINT_BINDING_PROTOCOL,
        "evaluation_mode": config.evaluation.evaluation_mode,
        "evaluation_data_stage": config.training.stage,
        "checkpoint_stage": checkpoint_stage,
        "checkpoint_role": observed_checkpoint_role,
        "expected_checkpoint_role": expected_checkpoint_role,
        "expected_checkpoint_stage": expected_checkpoint_stage,
        "run_completion": run_completion,
        "saved_segmentation_migration": saved_migration,
        "runtime_segmentation_migration": dict(runtime_segmentation_migration),
        "segmentation_source_sha256_match": True,
        "fixed_prediction_segmentation_source_match": fixed_prediction_match,
        "checkpoint_training_seed": int(saved_seed),
        "evaluation_seed": int(config.training.seed),
        "seed_match": True,
        "d_minus_one_acceptance": d_minus_one_acceptance,
        "stage_lineage": stage_lineage,
    }
