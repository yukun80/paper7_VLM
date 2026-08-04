"""Stage 4 OA-GroundedEval-dev 自动检查与专家 annotation 聚合。"""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from oa_groundrag.phase3.common import canonical_json, read_json, read_jsonl, sha256_file, sha256_text
from oa_groundrag.phase4.artifacts import AtomicArtifactDirectory

from oa_groundrag.landslide_evidence.annotation import SCORE_FIELDS
from oa_groundrag.landslide_evidence.contracts import fail
from oa_groundrag.landslide_evidence.region_pipeline import ledger_rows
from oa_groundrag.landslide_evidence.region_validation import validate_eval_dev

from .outputs import (
    REGION_PREDICTION_SCHEMA_VERSION,
    REGION_PROVENANCE_SCHEMA_VERSION,
    detect_forbidden_region_claims,
    parse_region_model_output,
)


GROUNDED_EVAL_REPORT_SCHEMA = "oa_groundrag.oa_grounded_eval_dev.report.v1"


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else float(numerator / denominator)


def _asset_identities(queue: list[dict[str, Any]]) -> dict[str, str]:
    result = {}
    for row in queue:
        record_id, identity = row.get("record_id"), row.get("asset_identity_sha256")
        if not isinstance(record_id, str) or not isinstance(identity, str) or record_id in result:
            fail("PREDICTION_IDENTITY_MISMATCH", "annotation queue identity 非法")
        result[record_id] = identity
    return result


def _human_metrics(annotations_root: Path | None, *, eval_root: Path) -> dict[str, Any] | None:
    if annotations_root is None:
        return None
    manifest = read_json(annotations_root / "manifest.json")
    if not isinstance(manifest, dict) or manifest.get("source_manifest_sha256") != sha256_file(eval_root / "manifest.json"):
        fail("ANNOTATION_INVALID", "annotation package 未绑定当前 Eval-dev manifest")
    rows = read_jsonl(annotations_root / "annotations.jsonl")
    weights = {"correct": 1.0, "partially_correct": 0.5, "incorrect": 0.0}
    aggregates: dict[str, Any] = {}
    for field in SCORE_FIELDS:
        values = [weights[row["scores"][field]] for row in rows if row["scores"][field] in weights]
        aggregates[field] = None if not values else float(sum(values) / len(values))
        aggregates[f"{field}_count"] = len(values)
    unsupported = sum(bool(row["unsupported_claims"]) for row in rows)
    aggregates["unsupported_claim_rate"] = _ratio(unsupported, len(rows))
    aggregates["annotation_count"] = len(rows)
    aggregates["expert_review_completed"] = bool(manifest.get("expert_review_completed", False))
    return aggregates


def evaluate_dev(
    *,
    eval_root: Path | str,
    predictions_path: Path | str,
    output_root: Path | str,
    annotations_root: Path | str | None = None,
) -> dict[str, Any]:
    eval_root = Path(os.path.abspath(Path(eval_root)))
    manifest = read_json(eval_root / "manifest.json")
    if not isinstance(manifest, dict):
        fail("PREDICTION_INVALID", "Eval manifest 非法")
    train_root = Path(str(manifest["train_corpus"]["root"]))
    validation = validate_eval_dev(eval_root, train_corpus_root=train_root, verify_source=True)
    predictions_path = Path(os.path.abspath(Path(predictions_path)))
    predictions = read_jsonl(predictions_path)
    records = read_jsonl(eval_root / "records.jsonl")
    groups = read_jsonl(eval_root / "counterfactual_groups.jsonl")
    queue = read_jsonl(eval_root / "annotation_queue.jsonl")
    records_by_id = {record["record_id"]: record for record in records}
    asset_ids = _asset_identities(queue)
    eval_manifest_sha = sha256_file(eval_root / "manifest.json")
    seen: set[str] = set()
    valid_outputs: dict[str, Mapping[str, Any]] = {}
    counts: Counter[str] = Counter()
    counts["prediction_rows"] = len(predictions)
    for index, prediction in enumerate(predictions):
        if set(prediction) != {
            "schema_version", "record_id", "parent_id", "model_output", "generated_text",
            "provenance", "counterfactual_group_id",
        }:
            counts["schema_invalid"] += 1
            continue
        record_id = prediction.get("record_id")
        if prediction.get("schema_version") != REGION_PREDICTION_SCHEMA_VERSION or not isinstance(record_id, str):
            counts["schema_invalid"] += 1
            continue
        record = records_by_id.get(record_id)
        if record is None or record_id in seen or prediction.get("parent_id") != record["parent_id"]:
            counts["identity_invalid"] += 1
            continue
        seen.add(record_id)
        provenance = prediction.get("provenance")
        if not isinstance(provenance, dict) or provenance.get("schema_version") != REGION_PROVENANCE_SCHEMA_VERSION:
            counts["identity_invalid"] += 1
            continue
        identity_ok = (
            provenance.get("record_id") == record_id
            and provenance.get("asset_manifest_sha256") == eval_manifest_sha
            and provenance.get("asset_identity_sha256") == asset_ids[record_id]
            and provenance.get("representation_mode") == record["representation_mode"]
            and provenance.get("formal_model_input_roles") == record["formal_model_input_roles"]
        )
        if not identity_ok:
            counts["identity_invalid"] += 1
            continue
        counts["identity_valid"] += 1
        if "audit_overlay" in provenance.get("formal_model_input_roles", []):
            counts["overlay_leakage"] += 1
        try:
            parsed = parse_region_model_output(prediction["model_output"])
            if canonical_json(parsed.to_dict()) != prediction.get("generated_text"):
                counts["schema_invalid"] += 1
                continue
        except Exception:
            counts["schema_invalid"] += 1
            if detect_forbidden_region_claims(prediction.get("model_output")):
                counts["forbidden_claim"] += 1
            continue
        counts["schema_valid"] += 1
        if parsed.target_status.value == record["target_status"]:
            counts["target_status_correct"] += 1
        if detect_forbidden_region_claims(parsed.to_dict()):
            counts["forbidden_claim"] += 1
        if record["target_status"] == "no_target":
            counts["no_target_total"] += 1
            if parsed.target_status.value == "no_target":
                counts["no_target_refusal"] += 1
        valid_outputs[record_id] = parsed.to_dict()
    group_complete = 0
    shift_changed = 0
    empty_refusal = 0
    context_changed = 0
    region_sensitive = 0
    for group in groups:
        variants = group["variants"]
        required_ids = [
            variants["baseline_correct_mask"], variants["empty_mask"],
            variants["deterministic_mask_shift"], variants["context_removal"],
        ]
        if not all(record_id in valid_outputs for record_id in required_ids):
            continue
        group_complete += 1
        baseline, empty, shifted, context = (valid_outputs[record_id] for record_id in required_ids)
        if shifted != baseline:
            shift_changed += 1
            region_sensitive += 1
        if empty["target_status"] == "no_target":
            empty_refusal += 1
        if context != baseline:
            context_changed += 1
    automatic = {
        "expected_prediction_count": len(records),
        "prediction_count": len(predictions),
        "schema_validity": _ratio(counts["schema_valid"], len(predictions)),
        "target_status_correctness": _ratio(counts["target_status_correct"], counts["schema_valid"]),
        "no_target_hallucination_rate": None if counts["no_target_total"] == 0 else 1.0 - counts["no_target_refusal"] / counts["no_target_total"],
        "forbidden_claim_rate": _ratio(counts["forbidden_claim"], len(predictions)),
        "binary_mask_input_identity_rate": _ratio(counts["identity_valid"], len(predictions)),
        "overlay_leakage_rate": _ratio(counts["overlay_leakage"], len(predictions)),
        "mask_region_response_sensitivity": _ratio(region_sensitive, len(groups)),
        "empty_mask_refusal_rate": _ratio(empty_refusal, len(groups)),
        "shifted_mask_response_change_rate": _ratio(shift_changed, len(groups)),
        "context_removal_sensitivity": _ratio(context_changed, len(groups)),
        "prediction_evidence_identity_rate": _ratio(counts["identity_valid"], len(predictions)),
        "counterfactual_group_completeness": _ratio(group_complete, len(groups)),
        "complete_prediction_set": len(seen) == len(records),
    }
    annotation_path = None if annotations_root is None else Path(os.path.abspath(Path(annotations_root)))
    report = {
        "schema_version": GROUNDED_EVAL_REPORT_SCHEMA,
        "eval_root": str(eval_root),
        "eval_manifest_sha256": eval_manifest_sha,
        "predictions_path": str(predictions_path),
        "predictions_sha256": sha256_file(predictions_path),
        "automatic_metrics": automatic,
        "human_metrics": _human_metrics(annotation_path, eval_root=eval_root),
        "validation_identity": validation,
        "development_only": True,
        "sealed_test_accessed": False,
        "thresholds_frozen": False,
        "formal_acceptance": False,
    }
    output = Path(os.path.abspath(Path(output_root)))
    if output.exists() or output.is_symlink():
        fail("OUTPUT_EXISTS", f"evaluation output 已存在：{output}")
    with AtomicArtifactDirectory(output) as writer:
        assert writer.staging is not None
        writer.write_json("report.json", report)
        ledger = ledger_rows(writer.staging, ["report.json"])
        writer.write_jsonl("SHA256SUMS.jsonl", ledger)
        writer.write_json("manifest.json", {
            "schema_version": "oa_groundrag.oa_grounded_eval_dev.report_artifact.v1",
            "report_schema_version": GROUNDED_EVAL_REPORT_SCHEMA,
            "report_sha256": sha256_file(writer.path("report.json")),
            "ledger_sha256": sha256_file(writer.path("SHA256SUMS.jsonl")),
            "formal_acceptance": False,
        })
        root = writer.publish()
    return {
        "ok": True,
        "root": str(root),
        "report_sha256": sha256_file(root / "report.json"),
        "automatic_metrics": automatic,
        "formal_acceptance": False,
    }
