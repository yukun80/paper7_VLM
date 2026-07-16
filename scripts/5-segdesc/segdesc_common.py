#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Segmentation-description unified-index helpers; not a standalone entrypoint."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
BUILDER_VERSION = "qpsalm_segdesc_index_builder_v3_component_contract_bound"
INDEX_SCHEMA = "qpsalm_segdesc_index_v1"
VALIDATION_PROTOCOL = "qpsalm_segdesc_index_validation_v3_component_contract_bound"
STATISTICS_PROTOCOL = "qpsalm_segdesc_index_statistics_v3_component_contract_bound"
BRIDGE_AWAITING_STATUS = "awaiting_expert_review"
BRIDGE_FROZEN_STATUS = "expert_pilot_frozen"
DESCRIPTION_BUILDER_VERSION = "description_benchmark_m1_v4_answer_trace"
BRIDGE_BUILDER_VERSION = "landslide_bridge_m2_v7_expert_review_replay_bound"
BRIDGE_EXPERT_ARTIFACT_PROTOCOL = (
    "landslide_bridge_expert_artifact_binding_v1_review_sources_and_outputs"
)
BRIDGE_EXPERT_REPLAY_PROTOCOL = (
    "landslide_bridge_expert_review_replay_v1_exact_semantic_projection"
)
SEGMENTATION_INSTRUCTION_REPORT = "reports/instruction_validation_report.json"
TASK_WEIGHTS = {
    "segmentation": 1.0,
    "global_caption": 1.0,
    "region_alignment": 1.0,
    "region_description_auto": 0.5,
    "region_description_expert": 1.0,
}
TASK_COMPONENTS = {
    "segmentation": "landslide_segmentation_v2",
    "global_caption": "description_v2",
    "region_alignment": "description_v2",
    "region_description_auto": "landslide_bridge_v1",
    "region_description_expert": "landslide_bridge_v1",
}
TASK_INDEX_NAMES = {
    "segmentation": {
        "instruction_train.jsonl", "instruction_val.jsonl", "instruction_test.jsonl",
    },
    "global_caption": {"train_eligible.jsonl", "dev.jsonl", "test.jsonl"},
    "region_alignment": {"train_eligible.jsonl", "dev.jsonl", "test.jsonl"},
    "region_description_auto": {"auto_train.jsonl"},
    "region_description_expert": {"expert_all.jsonl"},
}


def component_validation_contract(
    name: str,
    *,
    mode: str,
    root: Path,
) -> dict[str, Any]:
    """Return the exact current validation-report contract for one component."""
    if name == "segmentation":
        # Landslide V2 的 final validator 早于统一 builder version 字段；用稳定的
        # stage 与 portable benchmark root 明确排除 source/referring/V1 报告。
        return {
            "stage": "final",
            "benchmark_dir": project_ref(root),
        }
    if name == "description":
        return {
            "builder_version": DESCRIPTION_BUILDER_VERSION,
            "mode": mode,
        }
    if name == "bridge":
        return {
            "builder_version": BRIDGE_BUILDER_VERSION,
            "mode": mode,
        }
    raise KeyError(f"未知 unified component: {name!r}")


def component_contract_errors(
    name: str,
    report: dict[str, Any],
    *,
    mode: str,
    root: Path,
) -> list[str]:
    expected = component_validation_contract(name, mode=mode, root=root)
    return [
        f"{field} expected={expected_value!r} actual={nested_report_value(report, field)!r}"
        for field, expected_value in expected.items()
        if nested_report_value(report, field) != expected_value
    ]


def nested_report_value(report: dict[str, Any], field: str) -> Any:
    value: Any = report
    for part in field.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def segmentation_instruction_validation_contract(root: Path) -> dict[str, Any]:
    return {
        "benchmark_dir": project_ref(root),
        "template_config": "configs/instruction_templates/multisource_landslide_v2.yaml",
        "num_errors": 0,
        "parent_split_isolation.num_leaking": 0,
    }


def segmentation_instruction_contract_errors(
    report: dict[str, Any],
    *,
    root: Path,
) -> list[str]:
    expected = segmentation_instruction_validation_contract(root)
    return [
        f"{field} expected={expected_value!r} actual={nested_report_value(report, field)!r}"
        for field, expected_value in expected.items()
        if nested_report_value(report, field) != expected_value
    ]


def bridge_publication_policy(
    bridge_status: str,
    *,
    expert_index_present: bool,
    gate_present: bool,
) -> dict[str, bool]:
    """Resolve expert publication without inferring supervision from stale files."""
    if bridge_status == BRIDGE_FROZEN_STATUS:
        if not expert_index_present:
            raise ValueError("Bridge 已冻结但缺少 indexes/expert_all.jsonl")
        if not gate_present:
            raise ValueError("Bridge 已冻结但缺少 evaluation_gate_manifest.json")
        return {
            "expert_index_published": True,
            "bridge_gate_published": True,
            "stale_expert_index_ignored": False,
            "stale_bridge_gate_ignored": False,
        }
    if bridge_status == BRIDGE_AWAITING_STATUS:
        return {
            "expert_index_published": False,
            "bridge_gate_published": False,
            "stale_expert_index_ignored": bool(expert_index_present),
            "stale_bridge_gate_ignored": bool(gate_present),
        }
    raise ValueError(
        "统一索引只接受正式 Bridge prepare 或 frozen expert Pilot，"
        f"当前 status={bridge_status!r}"
    )


def benchmark_root() -> Path:
    configured = os.environ.get("PAPER7_BENCHMARK_ROOT") or os.environ.get("BENCHMARK_PREFIX")
    if configured:
        path = Path(configured).expanduser()
        return path if path.is_absolute() else (REPO_ROOT / path).resolve(strict=False)
    sibling = REPO_ROOT.parent / "benchmark"
    return sibling if sibling.exists() else REPO_ROOT / "benchmark"


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    parts = path.parts
    if parts and parts[0] == "benchmark":
        return benchmark_root().joinpath(*parts[1:])
    return REPO_ROOT / path


def project_ref(path: str | Path) -> str:
    resolved = Path(path).resolve(strict=False)
    root = benchmark_root().resolve(strict=False)
    try:
        return (Path("benchmark") / resolved.relative_to(root)).as_posix()
    except ValueError:
        try:
            return resolved.relative_to(REPO_ROOT.resolve(strict=False)).as_posix()
        except ValueError:
            return str(resolved)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: record must be an object")
            rows.append(value)
    return rows


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".part", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path: Path, payload: Any) -> None:
    _atomic_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    _atomic_text(path, "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_errors(
    binding: Any,
    *,
    label: str,
    expected_path: Path | None = None,
    expected_records: int | None = None,
) -> tuple[list[str], Path | None]:
    errors: list[str] = []
    if not isinstance(binding, dict):
        return [f"{label} artifact binding 必须是 object"], None
    path_ref = binding.get("path")
    if not isinstance(path_ref, str) or not path_ref.strip():
        return [f"{label} artifact binding 缺少 path"], None
    path = resolve_path(path_ref).resolve(strict=False)
    if expected_path is not None and path != expected_path.resolve(strict=False):
        errors.append(f"{label} artifact path 越出 frozen Bridge")
    if not path.is_file():
        errors.append(f"{label} artifact 缺失: {path}")
        return errors, path
    if binding.get("sha256") != sha256_file(path):
        errors.append(f"{label} artifact hash 漂移")
    if binding.get("bytes") != path.stat().st_size:
        errors.append(f"{label} artifact bytes 漂移")
    if expected_records is not None and binding.get("records") != expected_records:
        errors.append(
            f"{label} artifact records 不匹配: "
            f"expected={expected_records} observed={binding.get('records')!r}"
        )
    return errors, path


def _artifact_record_count(path: Path) -> int:
    if path.suffix.casefold() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return sum(1 for _ in csv.DictReader(handle))
    return len(read_jsonl(path))


def bridge_expert_artifact_errors(
    bridge_root: Path,
    validation_report: dict[str, Any],
) -> list[str]:
    """Deeply replay v7 expert publication before exposing unified expert rows."""
    errors: list[str] = []
    validation_binding = validation_report.get("expert_artifact_binding")
    if not isinstance(validation_binding, dict) or (
        validation_binding.get("protocol") != BRIDGE_EXPERT_ARTIFACT_PROTOCOL
        or validation_binding.get("builder_version") != BRIDGE_BUILDER_VERSION
    ):
        return ["frozen Bridge validation report 缺少当前 expert artifact binding"]
    review_report_path = bridge_root / "reports/expert_review_report.json"
    binding_errors, _ = _artifact_errors(
        validation_binding.get("review_report"),
        label="expert_review_report",
        expected_path=review_report_path,
    )
    errors.extend(binding_errors)
    if not review_report_path.is_file():
        return errors
    review_report = read_json(review_report_path)
    if (
        review_report.get("builder_version") != BRIDGE_BUILDER_VERSION
        or review_report.get("status") != "complete"
        or review_report.get("frozen_evaluation_gate") is not True
        or (review_report.get("errors") or [])
    ):
        errors.append("expert_review_report 不是当前完整 v7 merge report")
    merge_binding = review_report.get("expert_artifact_binding")
    if validation_binding.get("merge_artifacts") != merge_binding:
        errors.append("validation 与 merge expert artifact binding 不一致")
    if not isinstance(merge_binding, dict) or (
        merge_binding.get("protocol") != BRIDGE_EXPERT_ARTIFACT_PROTOCOL
        or merge_binding.get("builder_version") != BRIDGE_BUILDER_VERSION
    ):
        errors.append("merge expert artifact binding 过期")
        return errors
    semantic_replay = validation_binding.get("semantic_replay")
    if not isinstance(semantic_replay, dict) or (
        semantic_replay.get("protocol") != BRIDGE_EXPERT_REPLAY_PROTOCOL
    ):
        errors.append("validation report 缺少精确 expert 语义重放")
        semantic_replay = {}
    sources = merge_binding.get("sources")
    outputs = merge_binding.get("outputs")
    expected_sources = {
        "reviewer_1", "reviewer_2", "arbitration", "evaluation_gate_source",
    }
    output_paths = {
        "expert_all": bridge_root / "indexes/expert_all.jsonl",
        "expert_train": bridge_root / "indexes/expert_train.jsonl",
        "expert_val": bridge_root / "indexes/expert_val.jsonl",
        "expert_test": bridge_root / "indexes/expert_test.jsonl",
        "pending_arbitration": bridge_root / "indexes/pending_arbitration.jsonl",
        "evaluation_gate": bridge_root / "manifests/evaluation_gate_manifest.json",
    }
    if not isinstance(sources, dict) or set(sources) != expected_sources:
        errors.append("expert artifact sources 集合不完整")
        sources = sources if isinstance(sources, dict) else {}
    if not isinstance(outputs, dict) or set(outputs) != set(output_paths):
        errors.append("expert artifact outputs 集合不完整")
        outputs = outputs if isinstance(outputs, dict) else {}
    review_items = int(review_report.get("review_items", -1))
    review_selection_path = bridge_root / "manifests/review_selection.jsonl"
    if not review_selection_path.is_file():
        errors.append("frozen Bridge 缺少 review_selection.jsonl")
        selection_count = -1
    else:
        selection_count = len(read_jsonl(review_selection_path))
    if review_items != selection_count:
        errors.append("expert review 未精确覆盖完整 review_selection")
    final_decisions = review_report.get("final_decisions")
    try:
        final_count = (
            sum(int(value) for value in final_decisions.values())
            if isinstance(final_decisions, dict) else -1
        )
    except (TypeError, ValueError):
        final_count = -1
    try:
        pending_count = int(review_report.get("pending_arbitration", -1))
    except (TypeError, ValueError):
        pending_count = -1
    if final_count != review_items or pending_count != 0:
        errors.append("final_decisions/pending 未精确覆盖 review_selection")
    candidate_path = bridge_root / "indexes/candidate_all.jsonl"
    candidate_count = len(read_jsonl(candidate_path)) if candidate_path.is_file() else -1
    semantic_errors, _ = _artifact_errors(
        semantic_replay.get("candidate_index"),
        label="semantic_replay.candidate_index",
        expected_path=candidate_path,
        expected_records=candidate_count,
    )
    errors.extend(semantic_errors)
    semantic_errors, _ = _artifact_errors(
        semantic_replay.get("review_selection"),
        label="semantic_replay.review_selection",
        expected_path=review_selection_path,
        expected_records=selection_count,
    )
    errors.extend(semantic_errors)
    if (
        semantic_replay.get("review_items") != review_items
        or semantic_replay.get("pending_arbitration") != pending_count
        or semantic_replay.get("final_decisions") != final_decisions
        or semantic_replay.get("review_report_statistics_verified") is not True
    ):
        errors.append(
            "semantic replay 计数/决策/审核统计与 expert_review_report 不一致"
        )
    for name in ("reviewer_1", "reviewer_2"):
        source_errors, path = _artifact_errors(sources.get(name), label=name)
        errors.extend(source_errors)
        if path is not None and path.is_file():
            count = _artifact_record_count(path)
            count_errors, _ = _artifact_errors(
                sources.get(name), label=name, expected_records=count,
            )
            errors.extend(count_errors)
            if count != review_items:
                errors.append(f"{name} 未精确覆盖 review_items")
    arbitration = sources.get("arbitration")
    if arbitration is not None:
        source_errors, path = _artifact_errors(arbitration, label="arbitration")
        errors.extend(source_errors)
        if path is not None and path.is_file():
            count_errors, _ = _artifact_errors(
                arbitration,
                label="arbitration",
                expected_records=_artifact_record_count(path),
            )
            errors.extend(count_errors)
    source_errors, gate_source_path = _artifact_errors(
        sources.get("evaluation_gate_source"), label="evaluation_gate_source"
    )
    errors.extend(source_errors)
    gate_output_path = output_paths["evaluation_gate"]
    if (
        gate_source_path is not None and gate_source_path.is_file()
        and gate_output_path.is_file()
        and isinstance(sources.get("evaluation_gate_source"), dict)
    ):
        source_gate = read_json(gate_source_path)
        expected_gate = dict(source_gate)
        expected_gate["source_file"] = sources["evaluation_gate_source"].get("path")
        if read_json(gate_output_path) != expected_gate:
            errors.append("published gate 不是 frozen gate source 的精确带来源副本")

    rows_by_name: dict[str, list[dict[str, Any]]] = {}
    for name in ("expert_all", "expert_train", "expert_val", "expert_test", "pending_arbitration"):
        path = output_paths[name]
        if path.is_file():
            rows_by_name[name] = read_jsonl(path)
        else:
            rows_by_name[name] = []
    expert_all = rows_by_name["expert_all"]
    if rows_by_name["pending_arbitration"]:
        errors.append("frozen Bridge 仍含 pending arbitration")
    if int(review_report.get("expert_records", -1)) != len(expert_all):
        errors.append("expert_review_report 与 expert_all 记录数不一致")
    if semantic_replay.get("expert_records") != len(expert_all):
        errors.append("semantic replay 记录数与 expert_all 不一致")
    if isinstance(final_decisions, dict):
        try:
            accepted_count = sum(
                int(final_decisions.get(key, 0)) for key in ("accept", "revise")
            )
        except (TypeError, ValueError):
            accepted_count = -1
        if accepted_count != len(expert_all):
            errors.append("accept/revise 计数与 expert_all 不一致")
    record_ids = [str(row.get("bridge_record_id") or "") for row in expert_all]
    if not all(record_ids) or len(record_ids) != len(set(record_ids)):
        errors.append("expert_all bridge_record_id 缺失或重复")
    for split in ("train", "val", "test"):
        expected = [row for row in expert_all if row.get("split") == split]
        if not rows_by_name[f"expert_{split}"] or rows_by_name[f"expert_{split}"] != expected:
            errors.append(f"expert_{split} 不是 expert_all 的非空精确 split 投影")
    output_counts = {
        "expert_all": len(expert_all),
        "expert_train": len(rows_by_name["expert_train"]),
        "expert_val": len(rows_by_name["expert_val"]),
        "expert_test": len(rows_by_name["expert_test"]),
        "pending_arbitration": len(rows_by_name["pending_arbitration"]),
    }
    for name, path in output_paths.items():
        output_errors, _ = _artifact_errors(
            outputs.get(name),
            label=name,
            expected_path=path,
            expected_records=output_counts.get(name),
        )
        errors.extend(output_errors)
    return errors


def ensure_output(output_dir: Path, overwrite: bool, dry_run: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"输出目录非空；请显式使用 --overwrite: {output_dir}")
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
