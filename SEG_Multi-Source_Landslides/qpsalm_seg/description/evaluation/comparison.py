#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Paired Small-seed comparison for crop-only, masked pooling and MGRR."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from qpsalm_seg.paths import resolve_project_path

from ..data.expert_contracts import load_frozen_scientific_gate
from ..protocols.io import strict_json_loads
from .contracts import (
    DESCRIPTION_EVALUATION_PROTOCOL,
    SAME_IMAGE_RETRIEVAL_PROTOCOL,
)
from .publication import revalidate_evaluation_publication
from .metrics import (
    paired_bootstrap_delta_ci,
)
from .expert_factuality import EXPERT_FACTUALITY_PROTOCOL
from .formal_inputs import (
    M4_BASELINE_REGION_ENCODERS,
    M4_SEED_GATE_PROTOCOL,
    M4_SUITE_GATE_PROTOCOL,
    formal_seed_binding,
    load_evaluation_rows,
    m4_training_control_audit,
    validate_evaluation_checkpoint_provenance,
    validate_expert_binding,
    validate_formal_evaluation_limit,
)
from .counterfactual_gate import (
    claim_rate,
    counterfactual_gate,
    load_counterfactual_rows,
)
from ..protocols.io import canonical_sha256 as _canonical_sha256
from ..protocols.io import sha256_file as _sha256


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
    validate_formal_evaluation_limit(
        baseline,
        observed_count=int(baseline.get("num_samples", -1)),
        label="baseline paired evaluation",
    )
    validate_formal_evaluation_limit(
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
    baseline_seed = formal_seed_binding(
        baseline, expected_seed=expected_seed, label="baseline main evaluation",
    ) if expected_seed is not None else None
    candidate_seed = formal_seed_binding(
        candidate, expected_seed=expected_seed, label="candidate main evaluation",
    ) if expected_seed is not None else None
    training_control = (
        m4_training_control_audit(
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
    validate_formal_evaluation_limit(
        report,
        observed_count=int(report.get("num_samples", -1)),
        label=f"retrieval evaluation: {root}",
    )
    revalidate_evaluation_publication(root, report)
    validate_evaluation_checkpoint_provenance(root, report)
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
    return report, formal_seed_binding(
        report, expected_seed=expected_seed, label=f"retrieval:{root}",
    )


def absolute_candidate_gate(
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
    baseline, baseline_report = load_evaluation_rows(
        baseline_dir, require_complete_generation=True
    )
    candidate, candidate_report = load_evaluation_rows(
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
    baseline_expert_path = (
        resolve_project_path(baseline_expert_report)
        or Path(baseline_expert_report)
    )
    candidate_expert_path = (
        resolve_project_path(candidate_expert_report)
        or Path(candidate_expert_report)
    )
    baseline_expert = strict_json_loads(
        baseline_expert_path.read_text(encoding="utf-8")
    )
    candidate_expert = strict_json_loads(
        candidate_expert_path.read_text(encoding="utf-8")
    )
    if (
        baseline_expert.get("protocol") != EXPERT_FACTUALITY_PROTOCOL
        or candidate_expert.get("protocol") != EXPERT_FACTUALITY_PROTOCOL
    ):
        raise ValueError("正式 MGRR gate 需要当前 source-revalidated ERFS 报告")
    validate_expert_binding(
        baseline_expert,
        baseline_dir,
        expert_report_path=baseline_expert_path,
        label="baseline",
    )
    validate_expert_binding(
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
    automatic_baseline_ufcr = claim_rate(baseline_rows)
    automatic_candidate_ufcr = claim_rate(candidate_rows)
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
    counterfactual_audit = counterfactual_gate(
        candidate_report,
        load_counterfactual_rows(candidate_dir),
        frozen_gate["scientific_protocol"],
        candidate,
    )
    absolute_gate = absolute_candidate_gate(
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
        and counterfactual_audit["passed"]
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
        "counterfactual_gate": counterfactual_audit,
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
        for (
            baseline,
            candidate,
            seed,
            baseline_retrieval,
            candidate_retrieval,
            baseline_expert,
            candidate_expert,
        ) in zip(
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
