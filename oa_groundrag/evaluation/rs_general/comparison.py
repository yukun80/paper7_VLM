"""2B/4B Gate B 同记录配对 CI；仅报告，不参与任何升级判据。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from oa_groundrag.artifacts.directory import AtomicArtifactDirectory
from oa_groundrag.artifacts.identity import canonical_json, sha256_file, sha256_text
from oa_groundrag.artifacts.io import first_symlink_component
from oa_groundrag.data.rs_general.dataset import RSGeneralDescDataset
from oa_groundrag.data.rs_general.errors import RSGeneralDescError
from oa_groundrag.data.rs_general.io import read_json, read_jsonl
from oa_groundrag.vlm.errors import EvaluationError, ReasonCode

from .contracts import GATE_B_SAMPLE_COUNT, GATE_B_SEED, GATE_B_TASK_ORDER
from .metrics import (
    _completed_metrics,
    _load_generation_run,
    _validate_prediction_rows,
)
from .selection import (
    _strict_selection,
    load_gate_b_selection,
    load_gate_b_selection_for_stage5_retention,
    selection_locations,
)


FAMILY_COMPARISON_SCHEMA_VERSION = "rs_vlm.gate_b_family_comparison.v1"


@dataclass(frozen=True)
class GateBFamilyComparisonOutcome:
    root: Path
    paired_count: int
    report_only: bool = True


def _fail(message: str, **details: Any) -> None:
    raise EvaluationError(
        ReasonCode.GATE_B_RUN_INVALID,
        message,
        details=details,
    )


def _sha(value: str, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(f"{label} 必须是 lowercase SHA-256")
    return value


def _regular_file(path: Path, *, label: str) -> Path:
    path = Path(path)
    linked = first_symlink_component(path)
    if (
        linked is not None
        or not path.is_file()
        or path.is_symlink()
        or path.stat().st_nlink != 1
    ):
        _fail(f"{label} 必须是普通单链接文件", path=str(path))
    return path


def _regular_directory(path: Path, *, label: str) -> Path:
    path = Path(path)
    linked = first_symlink_component(path)
    if linked is not None or path.is_symlink() or not path.is_dir():
        _fail(f"{label} 必须是普通目录", path=str(path))
    return path


def _read_mapping(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = read_json(_regular_file(path, label=label))
    except RSGeneralDescError as error:
        _fail(f"{label} 无法严格读取", error=str(error))
    if not isinstance(value, dict):
        _fail(f"{label} 必须是对象")
    return value


def _read_rows(path: Path, *, label: str) -> list[dict[str, Any]]:
    try:
        return read_jsonl(_regular_file(path, label=label))
    except RSGeneralDescError as error:
        _fail(f"{label} 无法严格读取", error=str(error))
    raise AssertionError("unreachable")


def _rename_metric_roles(value: Any) -> Any:
    if isinstance(value, list):
        return [_rename_metric_roles(item) for item in value]
    if not isinstance(value, dict):
        return value
    renamed: dict[str, Any] = {}
    for key, item in value.items():
        target = {
            "base": "reference_2b",
            "adapter": "candidate_4b",
        }.get(key, key)
        renamed[target] = _rename_metric_roles(item)
    return renamed


def compare_gate_b_families(
    *,
    reference_protocol: Path,
    reference_selection: Path,
    reference_adapter_run: Path,
    reference_evaluation_root: Path,
    expected_reference_report_sha256: str,
    expected_reference_predictions_sha256: str,
    candidate_protocol: Path,
    candidate_selection: Path,
    candidate_adapter_run: Path,
    output_root: Path,
) -> GateBFamilyComparisonOutcome:
    """生成同 256 条记录的 paired CI；输出永远不具有 Gate 或升级语义。"""

    expected_report_sha = _sha(
        expected_reference_report_sha256,
        label="expected_reference_report_sha256",
    )
    expected_predictions_sha = _sha(
        expected_reference_predictions_sha256,
        label="expected_reference_predictions_sha256",
    )
    reference_context = load_gate_b_selection_for_stage5_retention(
        reference_protocol,
        reference_selection,
    )
    candidate_context = load_gate_b_selection(
        candidate_protocol,
        candidate_selection,
    )
    _strict_selection(reference_context.selection, reference_context.protocol_source.profile)
    _strict_selection(candidate_context.selection, candidate_context.protocol_source.profile)
    reference_items = reference_context.selection["items"]
    candidate_items = candidate_context.selection["items"]
    if reference_items != candidate_items:
        _fail("2B/4B selection 未精确复用同一 256 条 record identity")

    reference_root = _regular_directory(
        reference_adapter_run,
        label="reference adapter generation root",
    )
    reference_manifest = _read_mapping(
        reference_root / "generation_manifest.json",
        label="reference generation manifest",
    )
    reference_predictions_path = _regular_file(
        reference_root / "predictions.jsonl",
        label="reference predictions",
    )
    if sha256_file(reference_predictions_path) != expected_predictions_sha:
        _fail("reference predictions SHA 不匹配")
    reference_report_path = _regular_file(
        _regular_directory(
            reference_evaluation_root,
            label="reference evaluation root",
        )
        / "gate_b_report.json",
        label="reference Gate B report",
    )
    if sha256_file(reference_report_path) != expected_report_sha:
        _fail("reference Gate B report SHA 不匹配")
    reference_report = _read_mapping(
        reference_report_path,
        label="reference Gate B report",
    )
    reference_rows = _read_rows(
        reference_predictions_path,
        label="reference predictions",
    )
    if (
        reference_report.get("status") != "completed"
        or reference_report.get("gate_b_evaluated") is not True
        or reference_report.get("gate_b_passed") is not True
        or reference_report.get("formal_acceptance") is not True
        or reference_report.get("adapter_generation", {}).get(
            "prediction_sha256"
        )
        != expected_predictions_sha
        or reference_report.get("adapter_generation", {}).get(
            "manifest_sha256"
        )
        != sha256_file(reference_root / "generation_manifest.json")
        or reference_manifest.get("status") != "completed"
        or reference_manifest.get("model_role") != "adapter"
        or reference_manifest.get("valid_for_evaluation") is not True
        or reference_manifest.get("predictions", {}).get("sha256")
        != expected_predictions_sha
    ):
        _fail("reference 2B Adapter/Gate B 已接受身份不完整")

    candidate_manifest, candidate_rows, candidate_manifest_sha = (
        _load_generation_run(
            candidate_adapter_run,
            model_role="adapter",
            context=candidate_context,
        )
    )
    canonical = RSGeneralDescDataset.from_locations(
        candidate_context.protocol_source.base_config.data.benchmark_root,
        selection_locations(candidate_context.selection),
        roles=("external_val",),
        task_families=GATE_B_TASK_ORDER,
        load_assets=False,
        seed=GATE_B_SEED,
        expected_manifest_sha256=(
            candidate_context.protocol_source.base_config.data.expected_manifest_sha256
        ),
        verifier=candidate_context.access.verifier,
    )
    _validate_prediction_rows(
        reference_rows,
        model_role="adapter",
        context=reference_context,
        canonical_records=canonical.records,
    )
    _validate_prediction_rows(
        candidate_rows,
        model_role="adapter",
        context=candidate_context,
        canonical_records=canonical.records,
    )

    paired, metrics, bootstrap, _criteria, _would_pass = _completed_metrics(
        candidate_context,
        reference_rows,
        candidate_rows,
    )
    report_pairs = [
        {
            **row,
            "schema_version": FAMILY_COMPARISON_SCHEMA_VERSION,
            "artifact_kind": "family_paired_score",
            "reference_2b": row["base"],
            "candidate_4b": row["adapter"],
        }
        for row in paired
    ]
    for row in report_pairs:
        del row["base"]
        del row["adapter"]

    ordered_record_ids_sha = sha256_text(
        canonical_json([item["record_id"] for item in candidate_items])
    )
    with AtomicArtifactDirectory(Path(output_root)) as writer:
        writer.write_jsonl("paired_scores.jsonl", report_pairs)
        paired_path = writer.path("paired_scores.jsonl")
        report = {
            "schema_version": FAMILY_COMPARISON_SCHEMA_VERSION,
            "status": "completed",
            "report_only": True,
            "promotion_criterion_used": False,
            "gate_b_effect": "none",
            "scientific_superiority_claim_supported": False,
            "paired_count": len(report_pairs),
            "ordered_record_ids_sha256": ordered_record_ids_sha,
            "exact_record_identity_match": True,
            "reference_2b": {
                "protocol_id": reference_context.frozen_protocol["protocol_id"],
                "protocol_sha256": reference_context.frozen_protocol[
                    "protocol_sha256"
                ],
                "selection_sha256": reference_context.selection[
                    "selection_sha256"
                ],
                "gate_b_report_sha256": expected_report_sha,
                "generation_manifest_sha256": sha256_file(
                    reference_root / "generation_manifest.json"
                ),
                "predictions_sha256": expected_predictions_sha,
            },
            "candidate_4b": {
                "protocol_id": candidate_context.frozen_protocol["protocol_id"],
                "protocol_sha256": candidate_context.frozen_protocol[
                    "protocol_sha256"
                ],
                "selection_sha256": candidate_context.selection[
                    "selection_sha256"
                ],
                "generation_manifest_sha256": candidate_manifest_sha,
                "predictions_sha256": candidate_manifest["predictions"]["sha256"],
            },
            "metrics": _rename_metric_roles(metrics),
            "paired_bootstrap_ci": bootstrap,
            "gate_criteria_evaluated": False,
            "paired_scores": {
                "path": "paired_scores.jsonl",
                "count": len(report_pairs),
                "sha256": sha256_file(paired_path),
            },
        }
        writer.write_json("family_comparison_report.json", report)
        target = writer.publish()
    if len(report_pairs) != GATE_B_SAMPLE_COUNT:
        _fail("family comparison 未生成 256 对")
    return GateBFamilyComparisonOutcome(
        root=target,
        paired_count=len(report_pairs),
    )
