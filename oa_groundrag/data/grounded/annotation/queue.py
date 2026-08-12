"""Stage 4 人工 annotation queue 导出、导入与严格身份验证。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

from oa_groundrag.artifacts.identity import (
    canonical_json,
    sha256_file,
    sha256_text,
)
from oa_groundrag.artifacts.io import first_symlink_component
from oa_groundrag.data.rs_general.io import (
    read_json,
    read_jsonl,
)
from oa_groundrag.artifacts.directory import AtomicArtifactDirectory

from ..contracts import fail
from ..region_contracts import (
    ANNOTATION_PACKAGE_SCHEMA,
    ANNOTATION_SCHEMA,
    AdjudicationStatus,
    AnnotationStatus,
)
from ..region import ledger_rows


SCORE_FIELDS = (
    "target_appearance_accuracy",
    "target_morphology_accuracy",
    "surrounding_environment_accuracy",
    "region_context_relation_accuracy",
    "confuser_recognition",
    "evidence_sufficiency_accuracy",
    "expert_factuality",
)
SCORE_VALUES = {"correct", "partially_correct", "incorrect", "not_applicable"}


def _mapping(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        fail("ANNOTATION_INVALID", f"{location} 必须是对象")
    return dict(value)


def _exact(value: Mapping[str, Any], fields: tuple[str, ...], location: str) -> None:
    if set(value) != set(fields):
        fail("ANNOTATION_INVALID", f"{location} 字段不匹配", details={
            "missing": sorted(set(fields) - set(value)),
            "unknown": sorted(set(value) - set(fields)),
        })


def _nonempty(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        fail("ANNOTATION_INVALID", f"{location} 必须是非空字符串")
    return value.strip()


def _asset_root(root: Path | str) -> tuple[Path, dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    root = Path(os.path.abspath(Path(root)))
    if first_symlink_component(root) is not None or not root.is_dir():
        fail("ANNOTATION_INVALID", f"asset root 非普通目录：{root}")
    manifest = read_json(root / "manifest.json")
    records = read_jsonl(root / "records.jsonl")
    queue = read_jsonl(root / "annotation_queue.jsonl")
    if not isinstance(manifest, dict) or len(records) != len(queue):
        fail("ANNOTATION_INVALID", "asset root manifest/record/queue 非法")
    return root, manifest, records, queue


def export_annotation_queue(asset_root: Path | str, output_root: Path | str) -> dict[str, Any]:
    source_root, source_manifest, _, queue = _asset_root(asset_root)
    output = Path(os.path.abspath(Path(output_root)))
    if output.exists() or output.is_symlink():
        fail("OUTPUT_EXISTS", f"annotation export 已存在：{output}")
    guideline = read_json(source_root / "annotation_guideline.json")
    with AtomicArtifactDirectory(output) as writer:
        assert writer.staging is not None
        writer.write_jsonl("annotation_queue.jsonl", queue)
        writer.write_json("annotation_guideline.json", guideline)
        ledger = ledger_rows(writer.staging, ["annotation_queue.jsonl", "annotation_guideline.json"])
        writer.write_jsonl("SHA256SUMS.jsonl", ledger)
        manifest = {
            "schema_version": "oa_groundrag.mask_grounded_region.annotation_export.v1",
            "source_root": str(source_root),
            "source_manifest_sha256": sha256_file(source_root / "manifest.json"),
            "source_schema_version": source_manifest["schema_version"],
            "record_count": len(queue),
            "ledger": {
                "path": "SHA256SUMS.jsonl",
                "entry_count": len(ledger),
                "file_sha256": sha256_file(writer.path("SHA256SUMS.jsonl")),
                "root_sha256": sha256_text(canonical_json(ledger)),
            },
            "contains_expert_answers": False,
        }
        writer.write_json("manifest.json", manifest)
        root = writer.publish()
    return {
        "ok": True,
        "root": str(root),
        "manifest_sha256": sha256_file(root / "manifest.json"),
        "record_count": len(queue),
    }


ANNOTATION_FIELDS = (
    "schema_version", "record_id", "asset_identity_sha256", "annotation_version",
    "annotation_status", "annotator_id", "reviewer_id", "adjudicator_id",
    "adjudication_status", "description", "scores", "unsupported_claims",
)


def validate_annotation_row(
    value: Any,
    *,
    queue_row: Mapping[str, Any],
    location: str,
) -> dict[str, Any]:
    row = _mapping(value, location)
    _exact(row, ANNOTATION_FIELDS, location)
    if row["schema_version"] != ANNOTATION_SCHEMA or row["annotation_version"] != "region_annotation.v1":
        fail("ANNOTATION_INVALID", f"{location} schema/version 非法")
    if row["record_id"] != queue_row["record_id"] or row["asset_identity_sha256"] != queue_row["asset_identity_sha256"]:
        fail("ANNOTATION_INVALID", f"{location} record/asset identity 不匹配")
    try:
        status = AnnotationStatus(row["annotation_status"])
        adjudication = AdjudicationStatus(row["adjudication_status"])
    except (TypeError, ValueError) as error:
        fail("ANNOTATION_INVALID", f"{location} status enum 非法", details={"error": str(error)})
    if status is AnnotationStatus.QUEUED:
        fail("ANNOTATION_INVALID", f"{location} 导入记录不能仍为 queued")
    annotator = _nonempty(row["annotator_id"], f"{location}.annotator_id")
    reviewer = row["reviewer_id"]
    adjudicator = row["adjudicator_id"]
    if status in {AnnotationStatus.REVIEWED, AnnotationStatus.ADJUDICATED}:
        reviewer = _nonempty(reviewer, f"{location}.reviewer_id")
        if reviewer == annotator:
            fail("ANNOTATION_INVALID", f"{location} reviewer 必须不同于 annotator")
    elif reviewer is not None:
        fail("ANNOTATION_INVALID", f"{location} annotated 状态 reviewer 必须为 null")
    if status is AnnotationStatus.ADJUDICATED:
        adjudicator = _nonempty(adjudicator, f"{location}.adjudicator_id")
        if adjudication is not AdjudicationStatus.RESOLVED or adjudicator in {annotator, reviewer}:
            fail("ANNOTATION_INVALID", f"{location} 仲裁身份/状态非法")
    elif adjudicator is not None or adjudication is AdjudicationStatus.RESOLVED:
        fail("ANNOTATION_INVALID", f"{location} 非 adjudicated 不得提供 resolved 仲裁")
    from oa_groundrag.grounding.outputs import parse_region_model_output

    parsed = parse_region_model_output(row["description"])
    if parsed.target_status.value != queue_row["target_status"]:
        fail("ANNOTATION_INVALID", f"{location} target_status 与 queue 不一致")
    scores = _mapping(row["scores"], f"{location}.scores")
    _exact(scores, SCORE_FIELDS, f"{location}.scores")
    for field, score in scores.items():
        if score not in SCORE_VALUES:
            fail("ANNOTATION_INVALID", f"{location}.scores.{field} enum 非法")
    unsupported = row["unsupported_claims"]
    if not isinstance(unsupported, list) or not all(isinstance(item, str) and item.strip() for item in unsupported):
        fail("ANNOTATION_INVALID", f"{location}.unsupported_claims 必须是字符串列表")
    if len(unsupported) != len(set(unsupported)):
        fail("ANNOTATION_INVALID", f"{location}.unsupported_claims 不得重复")
    return row


def validate_annotations(
    asset_root: Path | str,
    annotations_path: Path | str,
    output_root: Path | str,
) -> dict[str, Any]:
    source_root, source_manifest, _, queue = _asset_root(asset_root)
    annotations_path = Path(os.path.abspath(Path(annotations_path)))
    if not annotations_path.is_file() or annotations_path.is_symlink():
        fail("ANNOTATION_INVALID", f"annotations 必须是普通 JSONL：{annotations_path}")
    annotations = read_jsonl(annotations_path)
    queue_by_id = {row["record_id"]: row for row in queue}
    if len(queue_by_id) != len(queue):
        fail("ANNOTATION_INVALID", "annotation queue record_id 重复")
    seen: set[str] = set()
    validated = []
    for index, value in enumerate(annotations):
        record_id = value.get("record_id")
        if not isinstance(record_id, str) or record_id not in queue_by_id or record_id in seen:
            fail("ANNOTATION_INVALID", f"annotations[{index}] record_id 未知或重复")
        seen.add(record_id)
        validated.append(validate_annotation_row(
            value,
            queue_row=queue_by_id[record_id],
            location=f"annotations[{index}]",
        ))
    output = Path(os.path.abspath(Path(output_root)))
    if output.exists() or output.is_symlink():
        fail("OUTPUT_EXISTS", f"annotation package 已存在：{output}")
    with AtomicArtifactDirectory(output) as writer:
        assert writer.staging is not None
        writer.write_jsonl("annotations.jsonl", validated)
        ledger = ledger_rows(writer.staging, ["annotations.jsonl"])
        writer.write_jsonl("SHA256SUMS.jsonl", ledger)
        manifest = {
            "schema_version": ANNOTATION_PACKAGE_SCHEMA,
            "source_root": str(source_root),
            "source_manifest_sha256": sha256_file(source_root / "manifest.json"),
            "source_schema_version": source_manifest["schema_version"],
            "annotation_source_sha256": sha256_file(annotations_path),
            "annotation_count": len(validated),
            "record_ids_sha256": sha256_text(canonical_json([row["record_id"] for row in validated])),
            "ledger": {
                "path": "SHA256SUMS.jsonl",
                "entry_count": len(ledger),
                "file_sha256": sha256_file(writer.path("SHA256SUMS.jsonl")),
                "root_sha256": sha256_text(canonical_json(ledger)),
            },
            "expert_review_completed": len(validated) == len(queue) and all(
                row["annotation_status"] in {"reviewed", "adjudicated"} for row in validated
            ),
            "formal_acceptance": False,
        }
        writer.write_json("manifest.json", manifest)
        root = writer.publish()
    return {
        "ok": True,
        "root": str(root),
        "manifest_sha256": sha256_file(root / "manifest.json"),
        "annotation_count": len(validated),
        "queue_count": len(queue),
        "expert_review_completed": manifest["expert_review_completed"],
        "formal_acceptance": False,
    }
