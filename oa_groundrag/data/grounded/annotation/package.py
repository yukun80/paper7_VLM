"""Stage 4 单专家 annotation package 的不可变发布与独立验证。"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from oa_groundrag.artifacts.identity import (
    canonical_json,
    sha256_file,
    sha256_text,
)
from oa_groundrag.data.rs_general.io import (
    read_json,
    read_jsonl,
    safe_join,
)
from oa_groundrag.artifacts.directory import AtomicArtifactDirectory

from ..contracts import fail
from ..region import ledger_rows
from ..region_validation import validate_region_asset_files
from .project import (
    ANNOTATION_ASSIGNMENT_SCHEMA,
    AnnotationAssetContext,
    AnnotationIntendedUse,
    TRAIN_ANNOTATION_COUNT,
    _exact,
    _mapping,
    _ordinary_file,
    _ordinary_root,
    _selected_records,
    _text,
    _validate_draft_message_identities,
    load_annotation_asset,
    load_annotation_project,
    load_draft_runs,
    load_model_drafts,
    load_verified_work,
    validate_draft_run_row,
    validate_model_draft_row,
    validate_verified_annotation_row,
)


VERIFIED_PACKAGE_SCHEMA = (
    "oa_groundrag.mask_grounded_region.expert_verified_package.v1"
)
PACKAGE_MANIFEST_FIELDS = (
    "schema_version",
    "package_id",
    "source_root",
    "source_manifest_sha256",
    "source_schema_version",
    "split",
    "intended_use",
    "reference_authority",
    "expert_consensus",
    "annotation_count",
    "ordered_record_ids_sha256",
    "annotations",
    "model_drafts",
    "draft_runs",
    "ledger",
    "training_eligible",
    "thresholds_frozen",
    "formal_acceptance",
    "scientific_acceptance",
    "sealed_test_evaluated",
)


@dataclass(frozen=True)
class VerifiedPackage:
    root: Path
    manifest: Mapping[str, Any]
    annotations: tuple[Mapping[str, Any], ...]
    drafts: tuple[Mapping[str, Any], ...]
    draft_runs: tuple[Mapping[str, Any], ...]


def _package_asset_rows(
    project: Mapping[str, Any],
    assignments: Sequence[Mapping[str, Any]],
) -> AnnotationAssetContext:
    validate_region_asset_files(project["asset_root"])
    context = load_annotation_asset(project["asset_root"])
    if context.manifest_sha256 != project["asset_manifest_sha256"]:
        fail("ANNOTATION_INVALID", "project source manifest 漂移")
    queue_by_id = {row["record_id"]: row for row in context.queue}
    for assignment in assignments:
        queue = queue_by_id.get(assignment["record_id"])
        if queue is None or queue["asset_identity_sha256"] != assignment["asset_identity_sha256"]:
            fail("ANNOTATION_INVALID", "assignment asset identity 漂移")
    return context


def export_verified_annotations(
    *,
    project_root: Path | str,
    output_root: Path | str,
) -> dict[str, Any]:
    """将 mutable 单专家工作目录发布为新的严格、不可覆盖 annotation package。"""

    work_root = _ordinary_root(project_root, location="project_root")
    project, assignments = load_annotation_project(work_root)
    context = _package_asset_rows(project, assignments)
    drafts = load_model_drafts(work_root, assignments=assignments)
    verified = load_verified_work(work_root, assignments=assignments, drafts=drafts)
    if len(drafts) != len(assignments) or len(verified) != len(assignments):
        fail(
            "ANNOTATION_INVALID",
            "只有全部记录已生成一次草稿并完成专家核验时才能发布",
            details={"records": len(assignments), "drafts": len(drafts), "verified": len(verified)},
        )
    annotations = [verified[row["record_id"]] for row in assignments]
    draft_rows = [drafts[row["record_id"]] for row in assignments]
    run_rows = list(load_draft_runs(work_root))
    run_by_id = {row["draft_run_id"]: row for row in run_rows}
    if not run_by_id or any(
        row["draft_run_id"] not in run_by_id
        or row["record_id"] not in run_by_id[row["draft_run_id"]]["record_ids"]
        for row in draft_rows
    ):
        fail("ANNOTATION_INVALID", "model drafts 未绑定完整 draft run provenance")
    _validate_draft_message_identities(context, draft_rows, run_by_id)
    training_eligible = (
        project["split"] == "train"
        and project["intended_use"] == AnnotationIntendedUse.TRAIN_SUPERVISION.value
        and len(annotations) == TRAIN_ANNOTATION_COUNT
    )
    output = Path(os.path.abspath(Path(output_root)))
    with AtomicArtifactDirectory(output) as writer:
        assert writer.staging is not None
        writer.write_jsonl("annotations.jsonl", annotations)
        writer.write_jsonl("model_drafts.jsonl", draft_rows)
        writer.write_jsonl("draft_runs.jsonl", run_rows)
        ledger = ledger_rows(
            writer.staging,
            ("annotations.jsonl", "model_drafts.jsonl", "draft_runs.jsonl"),
        )
        writer.write_jsonl("SHA256SUMS.jsonl", ledger)
        annotation_sha = sha256_file(writer.path("annotations.jsonl"))
        drafts_sha = sha256_file(writer.path("model_drafts.jsonl"))
        runs_sha = sha256_file(writer.path("draft_runs.jsonl"))
        ordered_ids_sha = sha256_text(canonical_json([row["record_id"] for row in annotations]))
        package_id = sha256_text(canonical_json({
            "source_manifest_sha256": context.manifest_sha256,
            "annotations_sha256": annotation_sha,
            "model_drafts_sha256": drafts_sha,
            "draft_runs_sha256": runs_sha,
            "intended_use": project["intended_use"],
        }))
        manifest = {
            "schema_version": VERIFIED_PACKAGE_SCHEMA,
            "package_id": package_id,
            "source_root": str(context.root),
            "source_manifest_sha256": context.manifest_sha256,
            "source_schema_version": context.manifest["schema_version"],
            "split": project["split"],
            "intended_use": project["intended_use"],
            "reference_authority": "single_expert",
            "expert_consensus": False,
            "annotation_count": len(annotations),
            "ordered_record_ids_sha256": ordered_ids_sha,
            "annotations": {"path": "annotations.jsonl", "sha256": annotation_sha},
            "model_drafts": {"path": "model_drafts.jsonl", "sha256": drafts_sha},
            "draft_runs": {"path": "draft_runs.jsonl", "sha256": runs_sha},
            "ledger": {
                "path": "SHA256SUMS.jsonl",
                "entry_count": len(ledger),
                "file_sha256": sha256_file(writer.path("SHA256SUMS.jsonl")),
                "root_sha256": sha256_text(canonical_json(ledger)),
            },
            "training_eligible": training_eligible,
            "thresholds_frozen": False,
            "formal_acceptance": False,
            "scientific_acceptance": False,
            "sealed_test_evaluated": False,
        }
        writer.write_json("manifest.json", manifest)
        root = writer.publish()
    validate_verified_annotation_package(asset_root=context.root, package_root=root)
    return {
        "ok": True,
        "root": str(root),
        "package_id": package_id,
        "manifest_sha256": sha256_file(root / "manifest.json"),
        "annotation_count": len(annotations),
        "training_eligible": training_eligible,
        "formal_acceptance": False,
    }


def _validate_ledger(root: Path, manifest: Mapping[str, Any]) -> None:
    contract = _mapping(manifest["ledger"], "manifest.ledger")
    path = safe_join(root, _text(contract.get("path"), "manifest.ledger.path"), location="ledger.path")
    _ordinary_file(path, location="ledger")
    rows = read_jsonl(path)
    if contract.get("entry_count") != len(rows) or contract.get("file_sha256") != sha256_file(path):
        fail("LEDGER_INVALID", "annotation package ledger identity 不一致")
    expected_paths = {"annotations.jsonl", "model_drafts.jsonl", "draft_runs.jsonl"}
    if {row.get("path") for row in rows} != expected_paths:
        fail("LEDGER_INVALID", "annotation package ledger 路径集合不一致")
    for row in rows:
        child = safe_join(root, str(row["path"]), location="ledger.row.path")
        _ordinary_file(child, location="ledger asset")
        if row.get("size_bytes") != child.stat().st_size or row.get("sha256") != sha256_file(child):
            fail("LEDGER_INVALID", f"annotation package 文件篡改：{row.get('path')}")
    if contract.get("root_sha256") != sha256_text(canonical_json(rows)):
        fail("LEDGER_INVALID", "annotation package ledger root 不一致")


def validate_verified_annotation_package(
    *,
    asset_root: Path | str,
    package_root: Path | str,
) -> VerifiedPackage:
    """独立验证单专家 package；不因其通过而升级为 Gold 或科学验收。"""

    validate_region_asset_files(asset_root)
    context = load_annotation_asset(asset_root)
    root = _ordinary_root(package_root, location="package_root")
    manifest_path = root / "manifest.json"
    _ordinary_file(manifest_path, location="package manifest")
    manifest = _mapping(read_json(manifest_path), "package manifest")
    _exact(manifest, PACKAGE_MANIFEST_FIELDS, "package manifest")
    if manifest["schema_version"] != VERIFIED_PACKAGE_SCHEMA:
        fail("ANNOTATION_INVALID", "verified package schema 非法")
    if (
        manifest["source_root"] != str(context.root)
        or manifest["source_manifest_sha256"] != context.manifest_sha256
        or manifest["source_schema_version"] != context.manifest["schema_version"]
    ):
        fail("ANNOTATION_INVALID", "verified package source identity 不匹配")
    if (
        manifest["reference_authority"] != "single_expert"
        or manifest["expert_consensus"] is not False
        or manifest["thresholds_frozen"] is not False
        or manifest["formal_acceptance"] is not False
        or manifest["scientific_acceptance"] is not False
        or manifest["sealed_test_evaluated"] is not False
    ):
        fail("FORMAL_EVALUATION_FORBIDDEN", "单专家 package 不得声明 Gold/共识/科学验收")
    _validate_ledger(root, manifest)
    paths: dict[str, Path] = {}
    for key in ("annotations", "model_drafts", "draft_runs"):
        contract = _mapping(manifest[key], f"manifest.{key}")
        _exact(contract, ("path", "sha256"), f"manifest.{key}")
        path = safe_join(root, _text(contract["path"], f"manifest.{key}.path"), location=f"manifest.{key}.path")
        _ordinary_file(path, location=f"package.{key}")
        if contract["sha256"] != sha256_file(path):
            fail("ANNOTATION_INVALID", f"package {key} SHA 漂移")
        paths[key] = path
    try:
        use = AnnotationIntendedUse(str(manifest["intended_use"]))
    except ValueError as error:
        fail("ANNOTATION_INVALID", "package intended_use 非法")
        raise AssertionError from error
    selected, _ = _selected_records(context, use)
    expected_split = "train" if use is AnnotationIntendedUse.TRAIN_SUPERVISION else "val"
    if manifest["split"] != expected_split:
        fail("SPLIT_FORBIDDEN", "package split 与 intended_use 不一致")
    queue_by_id = {row["record_id"]: row for row in context.queue}
    assignments = []
    for ordinal, record in enumerate(selected):
        queue = queue_by_id[record["record_id"]]
        assignments.append({
            "schema_version": ANNOTATION_ASSIGNMENT_SCHEMA,
            "ordinal": ordinal,
            "record_id": record["record_id"],
            "source": record["source"],
            "split": record["split"],
            "target_status": record["target_status"],
            "asset_identity_sha256": queue["asset_identity_sha256"],
            "partition": "all",
        })
    assignment_by_id = {row["record_id"]: row for row in assignments}
    draft_rows = tuple(_mapping(row, f"drafts[{index}]") for index, row in enumerate(read_jsonl(paths["model_drafts"])))
    if len(draft_rows) != len(assignments):
        fail("ANNOTATION_INVALID", "package model draft 数量不完整")
    drafts: dict[str, dict[str, Any]] = {}
    draft_ids: set[str] = set()
    for index, value in enumerate(draft_rows):
        assignment = assignment_by_id.get(value.get("record_id"))
        if assignment is None or value.get("record_id") in drafts:
            fail("ANNOTATION_INVALID", f"drafts[{index}] record_id 未知或重复")
        validated = validate_model_draft_row(
            value, assignment=assignment, location=f"drafts[{index}]"
        )
        if validated["draft_id"] in draft_ids:
            fail("ANNOTATION_INVALID", f"drafts[{index}] draft_id 重复")
        draft_ids.add(validated["draft_id"])
        drafts[str(value["record_id"])] = validated
    run_rows = tuple(
        validate_draft_run_row(row, location=f"draft_runs[{index}]")
        for index, row in enumerate(read_jsonl(paths["draft_runs"]))
    )
    run_by_id = {row["draft_run_id"]: row for row in run_rows}
    if len(run_by_id) != len(run_rows) or any(
        row["draft_run_id"] not in run_by_id
        or row["record_id"] not in run_by_id[row["draft_run_id"]]["record_ids"]
        for row in draft_rows
    ):
        fail("ANNOTATION_INVALID", "package draft run provenance 不完整")
    _validate_draft_message_identities(context, draft_rows, run_by_id)
    annotation_values = read_jsonl(paths["annotations"])
    if len(annotation_values) != len(assignments):
        fail("ANNOTATION_INVALID", "package annotation 数量不完整")
    annotations = []
    for index, (assignment, value) in enumerate(zip(assignments, annotation_values, strict=True)):
        if value.get("record_id") != assignment["record_id"]:
            fail("ANNOTATION_INVALID", "package annotation 顺序与源 records 不一致")
        annotations.append(validate_verified_annotation_row(
            value,
            assignment=assignment,
            draft=drafts[assignment["record_id"]],
            location=f"annotations[{index}]",
        ))
    ordered_sha = sha256_text(canonical_json([row["record_id"] for row in annotations]))
    expected_training = use is AnnotationIntendedUse.TRAIN_SUPERVISION and len(annotations) == TRAIN_ANNOTATION_COUNT
    if (
        manifest["annotation_count"] != len(annotations)
        or manifest["ordered_record_ids_sha256"] != ordered_sha
        or manifest["training_eligible"] is not expected_training
    ):
        fail("ANNOTATION_INVALID", "package count/order/training_eligible 不一致")
    expected_package_id = sha256_text(canonical_json({
        "source_manifest_sha256": context.manifest_sha256,
        "annotations_sha256": manifest["annotations"]["sha256"],
        "model_drafts_sha256": manifest["model_drafts"]["sha256"],
        "draft_runs_sha256": manifest["draft_runs"]["sha256"],
        "intended_use": use.value,
    }))
    if manifest["package_id"] != expected_package_id:
        fail("ANNOTATION_INVALID", "package_id 重算不一致")
    return VerifiedPackage(
        root=root,
        manifest=manifest,
        annotations=tuple(annotations),
        drafts=draft_rows,
        draft_runs=run_rows,
    )

