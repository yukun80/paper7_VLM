"""Gate B 已发布结果的只读证据链复核。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from oa_groundrag.artifacts.identity import sha256_file
from oa_groundrag.artifacts.io import first_symlink_component
from oa_groundrag.data.rs_general.io import (
    read_json,
    read_jsonl,
)
from oa_groundrag.data.rs_general.dataset import RSGeneralDescDataset
from oa_groundrag.data.rs_general.errors import RSGeneralDescError

from oa_groundrag.grounding.contracts import GATE_B_REPORT_SCHEMA_VERSION
from oa_groundrag.vlm.errors import EvaluationError, ReasonCode
from .contracts import (
    GATE_B_PROTOCOL_ID,
    GATE_B_SEED,
    GATE_B_TASK_ORDER,
    validate_frozen_training_root,
)
from .metrics import (
    _completed_metrics,
    _load_generation_run,
    _validate_prediction_rows,
)
from .selection import (
    _scan_candidates,
    _select_candidates,
    _selection_document,
    load_gate_b_selection,
    selection_locations,
)


_REPORT_FIELDS = {
    "schema_version",
    "status",
    "gate_b_evaluated",
    "gate_b_passed",
    "formal_acceptance",
    "adapter_status",
    "protocol_identity",
    "selection_identity",
    "benchmark_identity",
    "base_generation",
    "adapter_generation",
    "pairing",
    "metrics",
    "bootstrap",
    "criteria",
    "invalid_reasons",
}
_SHA_FIELDS = (
    "protocol_file",
    "selection_file",
    "base_manifest",
    "base_predictions",
    "adapter_manifest",
    "adapter_predictions",
    "paired_scores",
    "report",
)


@dataclass(frozen=True)
class GateBAcceptanceVerification:
    """只读复核成功后返回的最小接受身份。"""

    status: str
    gate_b_evaluated: bool
    gate_b_passed: bool
    formal_acceptance: bool
    paired_count: int
    protocol_sha256: str
    selection_sha256: str
    artifact_sha256: Mapping[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "gate_b_evaluated": self.gate_b_evaluated,
            "gate_b_passed": self.gate_b_passed,
            "formal_acceptance": self.formal_acceptance,
            "paired_count": self.paired_count,
            "protocol_sha256": self.protocol_sha256,
            "selection_sha256": self.selection_sha256,
            "artifact_sha256": {
                name: self.artifact_sha256[name] for name in _SHA_FIELDS
            },
            "full_asset_rescan_performed": False,
        }


def _fail(message: str, *, details: Mapping[str, Any] | None = None) -> None:
    raise EvaluationError(
        ReasonCode.GATE_B_RUN_INVALID,
        message,
        details=details,
    )


def _sha(value: str, *, location: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(f"{location}: SHA-256 非法")
    return value


def _regular_file(path: Path, *, location: str) -> Path:
    path = Path(path)
    linked = first_symlink_component(path)
    if linked is not None:
        _fail(f"{location}: 路径含 symlink", details={"path": str(linked)})
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_nlink != 1
    ):
        _fail(f"{location}: 必须是普通单链接文件", details={"path": str(path)})
    return path


def _regular_directory(path: Path, *, location: str) -> Path:
    path = Path(path)
    linked = first_symlink_component(path)
    if linked is not None:
        _fail(f"{location}: 路径含 symlink", details={"path": str(linked)})
    if not path.is_dir() or path.is_symlink():
        _fail(f"{location}: 必须是普通目录", details={"path": str(path)})
    return path


def _read_mapping(path: Path, *, location: str) -> dict[str, Any]:
    try:
        value = read_json(_regular_file(path, location=location))
    except RSGeneralDescError as error:
        _fail(f"{location}: 无法严格读取 JSON", details={"error": str(error)})
    if not isinstance(value, dict):
        _fail(f"{location}: 必须是对象")
    return value


def _read_rows(path: Path, *, location: str) -> list[dict[str, Any]]:
    try:
        return read_jsonl(_regular_file(path, location=location))
    except RSGeneralDescError as error:
        _fail(f"{location}: 无法严格读取 JSONL", details={"error": str(error)})
    raise AssertionError("unreachable")


def _recompute_selection(context: Any, training_root: Path) -> None:
    frozen = context.frozen_protocol
    candidates, probed_shards = _scan_candidates(context.access)
    monitoring_path = _regular_file(
        training_root / "validation_selection.json",
        location="training monitoring selection",
    )
    monitoring_identity = frozen["training_run"]["monitoring_selection"]
    actual_monitoring_sha = sha256_file(monitoring_path)
    if actual_monitoring_sha != monitoring_identity["file_sha256"]:
        _fail(
            "training monitoring selection SHA 与 frozen protocol 不一致",
            details={
                "expected": monitoring_identity["file_sha256"],
                "actual": actual_monitoring_sha,
            },
        )
    monitoring = _read_mapping(
        monitoring_path,
        location="training monitoring selection",
    )
    items = monitoring.get("items")
    if not isinstance(items, list):
        _fail("training monitoring selection items 非法")
    monitoring_parents: list[str] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            _fail(
                "training monitoring selection item 非法",
                details={"index": index},
            )
        parent_id = item.get("parent_id")
        if not isinstance(parent_id, str) or not parent_id:
            _fail(
                "training monitoring selection parent_id 非法",
                details={"index": index},
            )
        monitoring_parents.append(parent_id)
    selected, quotas, capacities, cells = _select_candidates(
        candidates,
        excluded_parents=set(monitoring_parents),
    )
    rebuilt = _selection_document(
        frozen_protocol=frozen,
        access=context.access,
        selected=selected,
        quotas=quotas,
        capacities=capacities,
        cells=cells,
        probed_shards=probed_shards,
        monitoring_file_sha256=actual_monitoring_sha,
        monitoring_selection_sha256=monitoring_identity["selection_sha256"],
        monitoring_parents=monitoring_parents,
    )
    if rebuilt != context.selection:
        first_difference = next(
            (
                index
                for index, (actual, expected) in enumerate(
                    zip(
                        context.selection.get("items", []),
                        rebuilt["items"],
                        strict=False,
                    )
                )
                if actual != expected
            ),
            None,
        )
        _fail(
            "frozen selection 不是预注册确定性算法的唯一输出",
            details={"first_item_difference": first_difference},
        )


def verify_gate_b_acceptance(
    protocol_path: Path | str,
    selection_path: Path | str,
    *,
    training_root: Path | str,
    base_run: Path | str,
    adapter_run: Path | str,
    evaluation_root: Path | str,
    expected_protocol_file_sha256: str,
    expected_report_sha256: str,
) -> GateBAcceptanceVerification:
    """严格只读复核已发布 Gate B v1 接受证据，不读取图像资产。"""

    expected_protocol_file_sha256 = _sha(
        expected_protocol_file_sha256,
        location="expected_protocol_file_sha256",
    )
    expected_report_sha256 = _sha(
        expected_report_sha256,
        location="expected_report_sha256",
    )
    selection_path = _regular_file(
        Path(selection_path),
        location="Gate B selection",
    )
    protocol_file = _regular_file(
        selection_path.parent / "gate_b_protocol.json",
        location="Gate B frozen protocol",
    )
    actual_protocol_file_sha = sha256_file(protocol_file)
    if actual_protocol_file_sha != expected_protocol_file_sha256:
        _fail(
            "Gate B frozen protocol file SHA 不匹配",
            details={
                "expected": expected_protocol_file_sha256,
                "actual": actual_protocol_file_sha,
            },
        )

    evaluation_root = _regular_directory(
        Path(evaluation_root),
        location="Gate B evaluation root",
    )
    report_path = _regular_file(
        evaluation_root / "gate_b_report.json",
        location="Gate B report",
    )
    actual_report_sha = sha256_file(report_path)
    if actual_report_sha != expected_report_sha256:
        _fail(
            "Gate B report file SHA 不匹配",
            details={
                "expected": expected_report_sha256,
                "actual": actual_report_sha,
            },
        )

    context = load_gate_b_selection(protocol_path, selection_path)
    training_root = _regular_directory(
        Path(training_root),
        location="Gate B training root",
    )
    validate_frozen_training_root(
        context.frozen_protocol,
        context.protocol_source,
        training_root=training_root,
        access=context.access,
    )
    _recompute_selection(context, training_root)

    base_run = _regular_directory(Path(base_run), location="Base generation root")
    adapter_run = _regular_directory(
        Path(adapter_run),
        location="Adapter generation root",
    )
    base_manifest, base_rows, base_manifest_sha = _load_generation_run(
        base_run,
        model_role="base",
        context=context,
    )
    adapter_manifest, adapter_rows, adapter_manifest_sha = _load_generation_run(
        adapter_run,
        model_role="adapter",
        context=context,
    )
    selection_file_sha = sha256_file(selection_path)
    if (
        base_manifest["selection_file_sha256"] != selection_file_sha
        or adapter_manifest["selection_file_sha256"] != selection_file_sha
    ):
        _fail("Base/Adapter generation selection file SHA 不匹配")
    if (
        base_manifest["input_token_count"]
        != adapter_manifest["input_token_count"]
        or base_manifest["image_count"] != adapter_manifest["image_count"]
    ):
        _fail("Base/Adapter 实际输入 token/image 总数不一致")

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
    if not passed:
        _fail("Gate B 指标重算未通过预注册六项判据")

    paired_path = evaluation_root / "paired_scores.jsonl"
    stored_paired = _read_rows(paired_path, location="Gate B paired scores")
    if stored_paired != paired:
        _fail("Gate B paired scores 与 predictions 重算结果不一致")
    paired_sha = sha256_file(paired_path)
    report = _read_mapping(report_path, location="Gate B report")
    if set(report) != _REPORT_FIELDS:
        _fail(
            "Gate B report 字段不匹配",
            details={
                "unknown": sorted(set(report) - _REPORT_FIELDS),
                "missing": sorted(_REPORT_FIELDS - set(report)),
            },
        )
    expected_report = {
        "schema_version": GATE_B_REPORT_SCHEMA_VERSION,
        "status": "completed",
        "gate_b_evaluated": True,
        "gate_b_passed": True,
        "formal_acceptance": True,
        "adapter_status": "accepted",
        "protocol_identity": {
            "protocol_id": GATE_B_PROTOCOL_ID,
            "protocol_sha256": context.frozen_protocol["protocol_sha256"],
        },
        "selection_identity": {
            "schema_version": context.selection["schema_version"],
            "selection_sha256": context.selection["selection_sha256"],
            "selection_file_sha256": selection_file_sha,
            "sample_count": context.selection["sample_count"],
            "parent_count": context.selection["parent_count"],
            "monitoring_exclusion": context.selection["monitoring_exclusion"],
        },
        "benchmark_identity": context.access.identity.to_dict(),
        "base_generation": {
            "manifest_sha256": base_manifest_sha,
            "prediction_sha256": base_manifest["predictions"]["sha256"],
            "prediction_count": base_manifest["predictions"]["count"],
            "failure_count": base_manifest["failures"]["count"],
            "config_identity": base_manifest["config_identity"],
            "model_identity": base_manifest["model_identity"],
            "processor_identity": base_manifest["processor_identity"],
        },
        "adapter_generation": {
            "manifest_sha256": adapter_manifest_sha,
            "prediction_sha256": adapter_manifest["predictions"]["sha256"],
            "prediction_count": adapter_manifest["predictions"]["count"],
            "failure_count": adapter_manifest["failures"]["count"],
            "config_identity": adapter_manifest["config_identity"],
            "model_identity": adapter_manifest["model_identity"],
            "processor_identity": adapter_manifest["processor_identity"],
            "checkpoint_identity": adapter_manifest["checkpoint_identity"],
        },
        "pairing": {
            "complete": True,
            "paired_count": len(paired),
            "expected_count": len(context.selection["items"]),
            "order": "frozen_selection_order",
            "paired_scores_path": "paired_scores.jsonl",
            "paired_scores_sha256": paired_sha,
        },
        "metrics": metrics,
        "bootstrap": bootstrap,
        "criteria": criteria,
        "invalid_reasons": [],
    }
    if report != expected_report:
        differing = sorted(
            key
            for key in _REPORT_FIELDS
            if report.get(key) != expected_report.get(key)
        )
        _fail(
            "Gate B report 与严格重算结果不一致",
            details={"differing_fields": differing},
        )

    artifacts = {
        "protocol_file": actual_protocol_file_sha,
        "selection_file": selection_file_sha,
        "base_manifest": base_manifest_sha,
        "base_predictions": sha256_file(base_run / "predictions.jsonl"),
        "adapter_manifest": adapter_manifest_sha,
        "adapter_predictions": sha256_file(adapter_run / "predictions.jsonl"),
        "paired_scores": paired_sha,
        "report": actual_report_sha,
    }
    return GateBAcceptanceVerification(
        status="accepted",
        gate_b_evaluated=True,
        gate_b_passed=True,
        formal_acceptance=True,
        paired_count=len(paired),
        protocol_sha256=context.frozen_protocol["protocol_sha256"],
        selection_sha256=context.selection["selection_sha256"],
        artifact_sha256=artifacts,
    )
