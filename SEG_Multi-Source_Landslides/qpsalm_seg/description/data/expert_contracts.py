"""Frozen expert Bridge contracts shared by datasets and prediction workflows."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

from qpsalm_seg.paths import resolve_project_path

from ..protocols.io import read_jsonl, sha256_file, strict_json_loads


BRIDGE_EXPERT_STATUS = "expert_pilot_frozen"
BRIDGE_GATE_PROTOCOL = "landslide_bridge_evaluation_gate_v2"
BRIDGE_BUILDER_VERSION = "landslide_bridge_m2_v7_expert_review_replay_bound"
BRIDGE_EXPERT_ARTIFACT_PROTOCOL = (
    "landslide_bridge_expert_artifact_binding_v1_review_sources_and_outputs"
)
BRIDGE_EXPERT_REPLAY_PROTOCOL = (
    "landslide_bridge_expert_review_replay_v1_exact_semantic_projection"
)
FROZEN_GATE_THRESHOLD_KEYS = (
    "no_target_rejection",
    "unsupported_claim_rate",
    "unavailable_unsupported_claim_rate",
    "unsupported_claim_rate_noninferiority",
    "expert_fact_score",
    "target_status_macro_f1",
    "present_recall",
    "absent_recall",
    "false_description_rate",
    "false_rejection_rate",
)
FROZEN_GATE_COUNTERFACTUAL_MODES = (
    "shuffled_mask",
    "region_swap",
    "cross_parent_region_swap",
    "cross_parent_modality_swap",
    "modality_removal",
)
FROZEN_GATE_SCIENTIFIC_PROTOCOLS = {
    "erfs_rubric": "qpsalm_erfs_eight_family_parent_macro_v1",
    "claim_inventory": "qpsalm_structured_claim_inventory_v1",
    "retrieval_scorer": "qpsalm_same_image_region_retrieval_v2_parent_ranked",
    "region_protocol_reporting": "separate_assisted_vision_only_v1",
}


def _read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path, label="frozen Bridge JSONL")


def _bound_record_count(path: Path) -> int:
    if path.suffix.casefold() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return sum(1 for _ in csv.DictReader(handle))
    return len(_read_jsonl_rows(path))


def _revalidate_file_artifact(
    binding: Any,
    *,
    label: str,
    expected_path: Path | None = None,
    expected_records: int | None = None,
) -> Path:
    if not isinstance(binding, dict):
        raise RuntimeError(f"frozen Bridge {label} artifact binding 必须是 object")
    path_ref = binding.get("path")
    if not isinstance(path_ref, str) or not path_ref.strip():
        raise RuntimeError(f"frozen Bridge {label} artifact binding 缺少 path")
    path = resolve_project_path(path_ref)
    assert path is not None
    if expected_path is not None and path != expected_path.resolve(strict=False):
        raise RuntimeError(
            f"frozen Bridge {label} artifact path 不匹配: "
            f"expected={expected_path} observed={path}"
        )
    if not path.is_file():
        raise FileNotFoundError(f"frozen Bridge {label} artifact 缺失: {path_ref} -> {path}")
    observed_sha = sha256_file(path)
    if binding.get("sha256") != observed_sha:
        raise RuntimeError(f"frozen Bridge {label} artifact hash 漂移: {path_ref}")
    observed_bytes = path.stat().st_size
    if binding.get("bytes") != observed_bytes:
        raise RuntimeError(f"frozen Bridge {label} artifact bytes 漂移: {path_ref}")
    if expected_records is not None and binding.get("records") != expected_records:
        raise RuntimeError(
            f"frozen Bridge {label} artifact records 不匹配: "
            f"expected={expected_records} observed={binding.get('records')!r}"
        )
    return path


def _revalidate_expert_artifacts(
    bridge_dir: Path,
    report: dict[str, Any],
    gate_path: Path,
) -> dict[str, Any]:
    """Replay review sources, split projections and validation-level binding."""
    validation_binding = report.get("expert_artifact_binding")
    if not isinstance(validation_binding, dict):
        raise RuntimeError("frozen Bridge validation report 缺少 expert_artifact_binding")
    if (
        validation_binding.get("protocol") != BRIDGE_EXPERT_ARTIFACT_PROTOCOL
        or validation_binding.get("builder_version") != BRIDGE_BUILDER_VERSION
    ):
        raise RuntimeError("frozen Bridge validation-level expert artifact binding 过期")
    review_report_path = bridge_dir / "reports/expert_review_report.json"
    _revalidate_file_artifact(
        validation_binding.get("review_report"),
        label="expert_review_report",
        expected_path=review_report_path,
    )
    review_report = strict_json_loads(
        review_report_path.read_text(encoding="utf-8")
    )
    if (
        review_report.get("builder_version") != BRIDGE_BUILDER_VERSION
        or review_report.get("status") != "complete"
        or review_report.get("frozen_evaluation_gate") is not True
        or (review_report.get("errors") or [])
    ):
        raise RuntimeError("frozen Bridge expert_review_report 不是当前完整 v7 merge report")
    merge_binding = review_report.get("expert_artifact_binding")
    if validation_binding.get("merge_artifacts") != merge_binding:
        raise RuntimeError(
            "frozen Bridge validation report 与 expert_review_report artifact binding 不一致"
        )
    if not isinstance(merge_binding, dict) or (
        merge_binding.get("protocol") != BRIDGE_EXPERT_ARTIFACT_PROTOCOL
        or merge_binding.get("builder_version") != BRIDGE_BUILDER_VERSION
    ):
        raise RuntimeError("frozen Bridge merge-level expert artifact binding 过期")
    semantic_replay = validation_binding.get("semantic_replay")
    if not isinstance(semantic_replay, dict) or (
        semantic_replay.get("protocol") != BRIDGE_EXPERT_REPLAY_PROTOCOL
    ):
        raise RuntimeError("frozen Bridge validation report 缺少精确 expert 语义重放")
    sources = merge_binding.get("sources")
    outputs = merge_binding.get("outputs")
    expected_source_keys = {
        "reviewer_1", "reviewer_2", "arbitration", "evaluation_gate_source",
    }
    expected_output_paths = {
        "expert_all": bridge_dir / "indexes/expert_all.jsonl",
        "expert_train": bridge_dir / "indexes/expert_train.jsonl",
        "expert_val": bridge_dir / "indexes/expert_val.jsonl",
        "expert_test": bridge_dir / "indexes/expert_test.jsonl",
        "pending_arbitration": bridge_dir / "indexes/pending_arbitration.jsonl",
        "evaluation_gate": gate_path,
    }
    if not isinstance(sources, dict) or set(sources) != expected_source_keys:
        raise RuntimeError("frozen Bridge expert artifact sources 集合不完整")
    if not isinstance(outputs, dict) or set(outputs) != set(expected_output_paths):
        raise RuntimeError("frozen Bridge expert artifact outputs 集合不完整")
    review_items = int(review_report.get("review_items", -1))
    review_selection_path = bridge_dir / "manifests/review_selection.jsonl"
    selection_count = len(_read_jsonl_rows(review_selection_path))
    if review_items != selection_count:
        raise RuntimeError(
            "frozen Bridge expert review 未精确覆盖完整 review_selection: "
            f"expected={selection_count} observed={review_items}"
        )
    final_decisions = review_report.get("final_decisions")
    if not isinstance(final_decisions, dict):
        raise RuntimeError("frozen Bridge expert_review_report 缺少 final_decisions")
    try:
        final_count = sum(int(value) for value in final_decisions.values())
    except (TypeError, ValueError) as exc:
        raise RuntimeError("frozen Bridge final_decisions 计数非法") from exc
    try:
        pending_count = int(review_report.get("pending_arbitration", -1))
    except (TypeError, ValueError) as exc:
        raise RuntimeError("frozen Bridge pending_arbitration 计数非法") from exc
    if final_count != review_items or pending_count != 0:
        raise RuntimeError(
            "frozen Bridge final_decisions/pending 未精确覆盖 review_selection"
        )
    candidate_path = bridge_dir / "indexes/candidate_all.jsonl"
    candidate_count = len(_read_jsonl_rows(candidate_path))
    _revalidate_file_artifact(
        semantic_replay.get("candidate_index"),
        label="semantic_replay.candidate_index",
        expected_path=candidate_path,
        expected_records=candidate_count,
    )
    _revalidate_file_artifact(
        semantic_replay.get("review_selection"),
        label="semantic_replay.review_selection",
        expected_path=review_selection_path,
        expected_records=selection_count,
    )
    if (
        semantic_replay.get("review_items") != review_items
        or semantic_replay.get("pending_arbitration") != pending_count
        or semantic_replay.get("final_decisions") != final_decisions
        or semantic_replay.get("review_report_statistics_verified") is not True
    ):
        raise RuntimeError(
            "frozen Bridge semantic replay 计数/决策/审核统计与 "
            "expert_review_report 不一致"
        )
    source_paths: dict[str, str | None] = {}
    for name in ("reviewer_1", "reviewer_2"):
        source_path = _revalidate_file_artifact(sources[name], label=name)
        source_count = _bound_record_count(source_path)
        _revalidate_file_artifact(
            sources[name], label=name, expected_records=source_count
        )
        if source_count != review_items:
            raise RuntimeError(
                f"frozen Bridge {name} 未精确覆盖 review_items: "
                f"expected={review_items} observed={source_count}"
            )
        source_paths[name] = str(source_path)
    arbitration_binding = sources["arbitration"]
    if arbitration_binding is None:
        source_paths["arbitration"] = None
    else:
        arbitration_path = _revalidate_file_artifact(
            arbitration_binding, label="arbitration"
        )
        arbitration_count = _bound_record_count(arbitration_path)
        _revalidate_file_artifact(
            arbitration_binding,
            label="arbitration",
            expected_records=arbitration_count,
        )
        source_paths["arbitration"] = str(arbitration_path)
    gate_source_path = _revalidate_file_artifact(
        sources["evaluation_gate_source"], label="evaluation_gate_source"
    )
    source_paths["evaluation_gate_source"] = str(gate_source_path)
    source_gate = strict_json_loads(gate_source_path.read_text(encoding="utf-8"))
    expected_published_gate = dict(source_gate)
    expected_published_gate["source_file"] = sources[
        "evaluation_gate_source"
    ]["path"]
    published_gate = strict_json_loads(gate_path.read_text(encoding="utf-8"))
    if published_gate != expected_published_gate:
        raise RuntimeError(
            "frozen Bridge published gate 不是 frozen gate source 的精确带来源副本"
        )

    expert_all_path = expected_output_paths["expert_all"]
    expert_rows = _read_jsonl_rows(expert_all_path)
    split_rows = {
        split: _read_jsonl_rows(expected_output_paths[f"expert_{split}"])
        for split in ("train", "val", "test")
    }
    pending_rows = _read_jsonl_rows(expected_output_paths["pending_arbitration"])
    if pending_rows:
        raise RuntimeError("frozen Bridge 仍含 pending arbitration")
    if int(review_report.get("expert_records", -1)) != len(expert_rows):
        raise RuntimeError("frozen Bridge expert_review_report 记录数与 expert_all 不一致")
    if semantic_replay.get("expert_records") != len(expert_rows):
        raise RuntimeError("frozen Bridge semantic replay 记录数与 expert_all 不一致")
    if sum(int(final_decisions.get(key, 0)) for key in ("accept", "revise")) != len(
        expert_rows
    ):
        raise RuntimeError("frozen Bridge accept/revise 计数与 expert_all 不一致")
    ids = [str(row.get("bridge_record_id") or "") for row in expert_rows]
    if not all(ids) or len(ids) != len(set(ids)):
        raise RuntimeError("frozen Bridge expert_all bridge_record_id 缺失或重复")
    for split, rows in split_rows.items():
        expected_rows = [row for row in expert_rows if row.get("split") == split]
        if not rows or rows != expected_rows:
            raise RuntimeError(
                f"frozen Bridge expert_{split} 不是 expert_all 的非空精确 split 投影"
            )
        validate_expert_rows(rows, stage="bridge_expert", split=split)
    output_counts = {
        "expert_all": len(expert_rows),
        "expert_train": len(split_rows["train"]),
        "expert_val": len(split_rows["val"]),
        "expert_test": len(split_rows["test"]),
        "pending_arbitration": 0,
    }
    for name, path in expected_output_paths.items():
        _revalidate_file_artifact(
            outputs[name],
            label=name,
            expected_path=path,
            expected_records=output_counts.get(name),
        )
    return {
        "protocol": BRIDGE_EXPERT_ARTIFACT_PROTOCOL,
        "semantic_replay_protocol": BRIDGE_EXPERT_REPLAY_PROTOCOL,
        "review_report": str(review_report_path),
        "review_report_sha256": sha256_file(review_report_path),
        "source_paths": source_paths,
        "expert_index_sha256": {
            split: sha256_file(expected_output_paths[f"expert_{split}"])
            for split in ("train", "val", "test")
        },
    }


def _validate_frozen_scientific_gate(gate: dict[str, Any]) -> None:
    """Require every Pilot-frozen threshold used by M4/M6 formal gates."""
    thresholds = gate.get("thresholds")
    if not isinstance(thresholds, dict):
        raise RuntimeError("frozen Bridge gate 缺少 thresholds object")
    invalid_thresholds = []
    for key in FROZEN_GATE_THRESHOLD_KEYS:
        value = thresholds.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
        ):
            invalid_thresholds.append(f"{key}={value!r}")
    if invalid_thresholds:
        raise RuntimeError(
            "frozen Bridge gate 的科学阈值未完整冻结: " + ", ".join(invalid_thresholds)
        )
    scientific = gate.get("scientific_protocol")
    if not isinstance(scientific, dict):
        raise RuntimeError("frozen Bridge gate 缺少 scientific_protocol")
    for key, expected in FROZEN_GATE_SCIENTIFIC_PROTOCOLS.items():
        if scientific.get(key) != expected:
            raise RuntimeError(
                f"frozen Bridge gate scientific_protocol.{key} 非法: "
                f"expected={expected!r} observed={scientific.get(key)!r}"
            )
    bootstrap = scientific.get("bootstrap")
    if not isinstance(bootstrap, dict):
        raise RuntimeError("frozen Bridge gate 缺少 bootstrap protocol")
    seed = bootstrap.get("seed")
    confidence = bootstrap.get("confidence")
    if (
        bootstrap.get("unit") != "parent"
        or bootstrap.get("samples") != 10000
        or isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isclose(float(confidence), 0.95, abs_tol=1.0e-12)
        or isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed < 0
    ):
        raise RuntimeError("frozen Bridge gate bootstrap 必须为 parent/10000/0.95/非负 seed")
    minimums = scientific.get("counterfactual_minimum_effective_parents")
    if not isinstance(minimums, dict) or set(minimums) != set(
        FROZEN_GATE_COUNTERFACTUAL_MODES
    ):
        raise RuntimeError("frozen Bridge gate 反事实模式集合不完整")
    invalid_minimums = {
        mode: minimums.get(mode)
        for mode in FROZEN_GATE_COUNTERFACTUAL_MODES
        if (
            isinstance(minimums.get(mode), bool)
            or not isinstance(minimums.get(mode), int)
            or int(minimums.get(mode)) <= 0
        )
    }
    if invalid_minimums:
        raise RuntimeError(f"frozen Bridge gate 反事实 parent 门槛非法: {invalid_minimums}")


def require_frozen_expert_bridge(bridge_dir: Path) -> dict[str, Any]:
    """Reject stale expert indexes unless the current Pilot gate is frozen."""
    report_path = bridge_dir / "reports/validation_report.json"
    gate_path = bridge_dir / "manifests/evaluation_gate_manifest.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"Bridge 缺少 validation report: {report_path}")
    report = strict_json_loads(report_path.read_text(encoding="utf-8"))
    if (
        report.get("builder_version") != BRIDGE_BUILDER_VERSION
        or
        report.get("status") != BRIDGE_EXPERT_STATUS
        or report.get("require_expert_complete") is not True
        or (report.get("errors") or [])
    ):
        raise RuntimeError(
            "D3b/D4/M7 expert 数据要求 Bridge status=expert_pilot_frozen、"
            "require_expert_complete=true 且 errors=[]；"
            f"当前 status={report.get('status')!r}"
        )
    if not gate_path.is_file():
        raise FileNotFoundError(f"frozen Bridge 缺少 evaluation gate: {gate_path}")
    gate = strict_json_loads(gate_path.read_text(encoding="utf-8"))
    if (
        gate.get("protocol") != BRIDGE_GATE_PROTOCOL
        or gate.get("builder_version") != BRIDGE_BUILDER_VERSION
        or gate.get("frozen") is not True
        or gate.get("status") != "frozen_after_pilot"
    ):
        raise RuntimeError("Bridge evaluation gate 不是当前 builder 的人工冻结 v2 Pilot gate")
    _validate_frozen_scientific_gate(gate)
    binding_paths = {
        "pilot_parent_manifest_sha256": bridge_dir / "manifests/pilot_parent_manifest.jsonl",
        "review_selection_sha256": bridge_dir / "manifests/review_selection.jsonl",
        "candidate_index_sha256": bridge_dir / "indexes/candidate_all.jsonl",
    }
    missing = [str(path) for path in binding_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Bridge evaluation gate 绑定文件缺失: {missing}")
    expected = {name: sha256_file(path) for name, path in binding_paths.items()}
    if gate.get("bindings") != expected:
        raise RuntimeError("Bridge evaluation gate 与当前 Pilot/selection/candidate hash 不一致")
    expert_artifact_audit = _revalidate_expert_artifacts(
        bridge_dir.resolve(strict=False), report, gate_path.resolve(strict=False)
    )
    return {
        "status": BRIDGE_EXPERT_STATUS,
        "validation_report": str(report_path),
        "validation_report_sha256": sha256_file(report_path),
        "evaluation_gate": str(gate_path),
        "evaluation_gate_sha256": sha256_file(gate_path),
        "candidate_index": str(
            binding_paths["candidate_index_sha256"].resolve(strict=False)
        ),
        "candidate_index_sha256": expected["candidate_index_sha256"],
        "expert_artifact_audit": expert_artifact_audit,
    }


def load_frozen_scientific_gate(bridge_dir: Path) -> dict[str, Any]:
    """Load the exact Pilot-frozen thresholds used by formal M4/M6 gates."""
    audit = require_frozen_expert_bridge(bridge_dir)
    gate_path = Path(audit["evaluation_gate"])
    gate = strict_json_loads(gate_path.read_text(encoding="utf-8"))
    return {
        "audit": audit,
        "thresholds": dict(gate["thresholds"]),
        "scientific_protocol": dict(gate["scientific_protocol"]),
    }


def validate_expert_rows(
    rows: list[dict[str, Any]], *, stage: str, split: str,
) -> None:
    if not rows:
        raise ValueError(f"expert stage={stage} split={split} 不能为空")
    for row in rows:
        sample_id = str(row.get("bridge_record_id") or row.get("sample_id") or "unknown")
        target = row.get("expert_target")
        if not isinstance(target, dict) or not isinstance(
            target.get("structured_output"), dict
        ) or not str(target.get("summary") or "").strip():
            raise ValueError(f"expert row 缺少人工审核 target: {sample_id}")
        review_status = str((row.get("review") or {}).get("status") or "")
        if review_status not in {"accepted", "revised", "arbitrated"}:
            raise ValueError(f"expert row review status 非法: {sample_id}={review_status!r}")
        predicted = (
            str(row.get("region_source") or "") == "predicted_proposal"
            or str(row.get("schema_version") or "").startswith("qpsalm_predicted_region")
        )
        provenance = row.get("prediction_provenance")
        if (
            stage == "predicted_mask"
            and split == "train"
            and predicted
            and (
                not isinstance(provenance, dict)
                or provenance.get("out_of_fold_verified") is not True
            )
        ):
            raise ValueError(f"D4 train predicted row 未通过 OOF 审计: {sample_id}")
