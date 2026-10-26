"""Stage 4 单专家 train-only message 导出、重载与 Dataset 接口。"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Mapping

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
from oa_groundrag.grounding.messages import build_mask_grounded_region_messages
from oa_groundrag.grounding.outputs import (
    REGION_OUTPUT_SCHEMA_VERSION,
    parse_region_model_output,
)

from ..contracts import fail
from ..region import ledger_rows, region_asset_identity
from ..region_validation import validate_region_asset_files
from .project import (
    AnnotationIntendedUse,
    TRAIN_ANNOTATION_COUNT,
    _exact,
    _mapping,
    _ordinary_file,
    _ordinary_root,
    _text,
    load_annotation_asset,
)
from .package import validate_verified_annotation_package


TRAINING_MESSAGE_SCHEMA = "oa_groundrag.mask_grounded_region.training_message.v1"
TRAINING_MANIFEST_SCHEMA = "oa_groundrag.mask_grounded_region.training_messages.v1"
TRAINING_ROW_FIELDS = (
    "schema_version", "record_id", "parent_id", "source", "logical_role",
    "task_family", "messages", "annotation_identity_sha256", "asset_identity_sha256",
)
TRAINING_MANIFEST_FIELDS = (
    "schema_version", "source_root", "source_manifest_sha256",
    "annotation_package_root", "annotation_package_manifest_sha256", "split",
    "record_count", "ordered_record_ids_sha256", "messages",
    "assistant_target_schema", "ledger", "training_eligible", "reference_authority",
    "formal_acceptance",
)


@dataclass(frozen=True)
class TrainingMessageArtifact:
    root: Path
    manifest: Mapping[str, Any]
    rows: tuple[Mapping[str, Any], ...]


def export_training_messages(
    *,
    asset_root: Path | str,
    annotations_root: Path | str,
    output_root: Path | str,
) -> dict[str, Any]:
    """导出 assistant-only 监督消息；val/test 在代码级拒绝。"""

    validate_region_asset_files(asset_root)
    context = load_annotation_asset(asset_root)
    package = validate_verified_annotation_package(
        asset_root=context.root,
        package_root=annotations_root,
    )
    if (
        package.manifest["split"] != "train"
        or package.manifest["intended_use"] != AnnotationIntendedUse.TRAIN_SUPERVISION.value
        or package.manifest["training_eligible"] is not True
        or len(package.annotations) != TRAIN_ANNOTATION_COUNT
    ):
        fail("SPLIT_FORBIDDEN", "training messages 只允许完整 500 条 train 单专家 package")
    records = {row["record_id"]: row for row in context.records}
    queue = {row["record_id"]: row for row in context.queue}
    rows = []
    for annotation in package.annotations:
        record = records[annotation["record_id"]]
        if record["split"] != "train":
            fail("SPLIT_FORBIDDEN", "val/test annotation 不得进入训练")
        actual_asset_id = region_asset_identity(context.root, record["assets"])
        if actual_asset_id != annotation["asset_identity_sha256"] or actual_asset_id != queue[record["record_id"]]["asset_identity_sha256"]:
            fail("ANNOTATION_INVALID", "training export asset identity 漂移")
        parsed = parse_region_model_output(annotation["description"])
        assistant_target = canonical_json(parsed.to_dict())
        messages = build_mask_grounded_region_messages(
            record,
            asset_root=context.root,
            assistant_target=assistant_target,
        )
        rows.append({
            "schema_version": TRAINING_MESSAGE_SCHEMA,
            "record_id": record["record_id"],
            "parent_id": record["parent_id"],
            "source": record["source"],
            "logical_role": "train",
            "task_family": "mask_grounded_region_description",
            "messages": messages,
            "annotation_identity_sha256": sha256_text(canonical_json(annotation)),
            "asset_identity_sha256": actual_asset_id,
        })
    output = Path(os.path.abspath(Path(output_root)))
    with AtomicArtifactDirectory(output) as writer:
        assert writer.staging is not None
        writer.write_jsonl("messages.jsonl", rows)
        ledger = ledger_rows(writer.staging, ("messages.jsonl",))
        writer.write_jsonl("SHA256SUMS.jsonl", ledger)
        messages_sha = sha256_file(writer.path("messages.jsonl"))
        manifest = {
            "schema_version": TRAINING_MANIFEST_SCHEMA,
            "source_root": str(context.root),
            "source_manifest_sha256": context.manifest_sha256,
            "annotation_package_root": str(package.root),
            "annotation_package_manifest_sha256": sha256_file(package.root / "manifest.json"),
            "split": "train",
            "record_count": len(rows),
            "ordered_record_ids_sha256": sha256_text(canonical_json([row["record_id"] for row in rows])),
            "messages": {"path": "messages.jsonl", "sha256": messages_sha},
            "assistant_target_schema": REGION_OUTPUT_SCHEMA_VERSION,
            "ledger": {
                "path": "SHA256SUMS.jsonl",
                "entry_count": len(ledger),
                "file_sha256": sha256_file(writer.path("SHA256SUMS.jsonl")),
                "root_sha256": sha256_text(canonical_json(ledger)),
            },
            "training_eligible": True,
            "reference_authority": "single_expert",
            "formal_acceptance": False,
        }
        writer.write_json("manifest.json", manifest)
        root = writer.publish()
    return {
        "ok": True,
        "root": str(root),
        "manifest_sha256": sha256_file(root / "manifest.json"),
        "record_count": len(rows),
        "training_eligible": True,
        "formal_acceptance": False,
    }


def load_training_message_artifact(
    training_root: Path | str,
) -> TrainingMessageArtifact:
    """严格重载 train-only messages，供 Dataset/Collator 在训练前再次绑定身份。"""

    root = _ordinary_root(training_root, location="training_root")
    manifest_path = root / "manifest.json"
    _ordinary_file(manifest_path, location="training manifest")
    manifest = _mapping(read_json(manifest_path), "training manifest")
    _exact(manifest, TRAINING_MANIFEST_FIELDS, "training manifest")
    if (
        manifest["schema_version"] != TRAINING_MANIFEST_SCHEMA
        or manifest["split"] != "train"
        or manifest["record_count"] != TRAIN_ANNOTATION_COUNT
        or manifest["assistant_target_schema"] != REGION_OUTPUT_SCHEMA_VERSION
        or manifest["training_eligible"] is not True
        or manifest["reference_authority"] != "single_expert"
        or manifest["formal_acceptance"] is not False
    ):
        fail("SPLIT_FORBIDDEN", "training message manifest 不是完整 train-only 单专家监督")
    source_root = _text(manifest["source_root"], "manifest.source_root")
    validate_region_asset_files(source_root)
    context = load_annotation_asset(source_root)
    if context.manifest_sha256 != manifest["source_manifest_sha256"]:
        fail("ANNOTATION_INVALID", "training message source manifest 漂移")
    package_root = _ordinary_root(
        _text(manifest["annotation_package_root"], "manifest.annotation_package_root"),
        location="annotation_package_root",
    )
    if sha256_file(package_root / "manifest.json") != manifest["annotation_package_manifest_sha256"]:
        fail("ANNOTATION_INVALID", "training message annotation package 漂移")
    package = validate_verified_annotation_package(
        asset_root=context.root,
        package_root=package_root,
    )
    if package.manifest["training_eligible"] is not True:
        fail("SPLIT_FORBIDDEN", "training message 只允许完整 train annotation package")

    message_contract = _mapping(manifest["messages"], "manifest.messages")
    _exact(message_contract, ("path", "sha256"), "manifest.messages")
    messages_path = safe_join(
        root,
        _text(message_contract["path"], "manifest.messages.path"),
        location="manifest.messages.path",
    )
    _ordinary_file(messages_path, location="messages")
    if message_contract["sha256"] != sha256_file(messages_path):
        fail("LEDGER_INVALID", "training messages SHA 漂移")
    ledger_contract = _mapping(manifest["ledger"], "manifest.ledger")
    _exact(
        ledger_contract,
        ("path", "entry_count", "file_sha256", "root_sha256"),
        "manifest.ledger",
    )
    ledger_path = safe_join(
        root,
        _text(ledger_contract["path"], "manifest.ledger.path"),
        location="manifest.ledger.path",
    )
    _ordinary_file(ledger_path, location="training ledger")
    ledger = read_jsonl(ledger_path)
    if (
        len(ledger) != 1
        or ledger_contract["entry_count"] != 1
        or ledger_contract["file_sha256"] != sha256_file(ledger_path)
        or ledger_contract["root_sha256"] != sha256_text(canonical_json(ledger))
        or ledger[0].get("path") != message_contract["path"]
        or ledger[0].get("size_bytes") != messages_path.stat().st_size
        or ledger[0].get("sha256") != sha256_file(messages_path)
    ):
        fail("LEDGER_INVALID", "training message ledger identity 非法")

    values = read_jsonl(messages_path)
    if len(values) != TRAIN_ANNOTATION_COUNT:
        fail("ANNOTATION_INVALID", "training messages 必须精确为 500 条")
    records = {row["record_id"]: row for row in context.records}
    annotations = {row["record_id"]: row for row in package.annotations}
    rows = []
    seen: set[str] = set()
    for index, value in enumerate(values):
        row = _mapping(value, f"messages[{index}]")
        _exact(row, TRAINING_ROW_FIELDS, f"messages[{index}]")
        record_id = _text(row["record_id"], f"messages[{index}].record_id")
        record = records.get(record_id)
        annotation = annotations.get(record_id)
        if record is None or annotation is None or record_id in seen:
            fail("ANNOTATION_INVALID", f"messages[{index}] record 未知或重复")
        seen.add(record_id)
        actual_asset_id = region_asset_identity(context.root, record["assets"])
        expected_messages = build_mask_grounded_region_messages(
            record,
            asset_root=context.root,
            assistant_target=canonical_json(
                parse_region_model_output(annotation["description"]).to_dict()
            ),
        )
        if (
            row["schema_version"] != TRAINING_MESSAGE_SCHEMA
            or row["parent_id"] != record["parent_id"]
            or row["source"] != record["source"]
            or row["logical_role"] != "train"
            or row["task_family"] != "mask_grounded_region_description"
            or row["messages"] != expected_messages
            or row["annotation_identity_sha256"] != sha256_text(canonical_json(annotation))
            or row["asset_identity_sha256"] != actual_asset_id
        ):
            fail("ANNOTATION_INVALID", f"messages[{index}] 监督或 identity 漂移")
        rows.append(row)
    ordered_sha = sha256_text(canonical_json([row["record_id"] for row in rows]))
    if manifest["ordered_record_ids_sha256"] != ordered_sha:
        fail("ANNOTATION_INVALID", "training messages ordered IDs 漂移")
    return TrainingMessageArtifact(root=root, manifest=manifest, rows=tuple(rows))


class MaskGroundedTrainingMessageDataset:
    """把已验证训练 artifact 暴露为现有 DescriptionCollator 可消费的样本。"""

    def __init__(self, training_root: Path | str) -> None:
        artifact = load_training_message_artifact(training_root)
        self.root = artifact.root
        self.manifest = artifact.manifest
        self.records = artifact.rows

    def __len__(self) -> int:
        return len(self.records)

    def set_epoch(self, epoch: int) -> None:
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            fail("ANNOTATION_INVALID", "training message epoch 必须是非负整数")

    def __getitem__(self, index: int) -> Any:
        from oa_groundrag.grounding.contracts import MaskMode
        from oa_groundrag.vlm.data import DescriptionSample

        if index < 0:
            index += len(self.records)
        if index < 0 or index >= len(self.records):
            raise IndexError(index)
        row = self.records[index]
        messages = tuple(row["messages"])
        assistant = messages[-1]["content"][0]["text"]
        return DescriptionSample(
            record_id=str(row["record_id"]),
            parent_id=str(row["parent_id"]),
            logical_role="train",
            task_family="mask_grounded_region_description",
            messages=messages,
            reference_responses=(str(assistant),),
            mask_mode=MaskMode.GT_MASK,
            evidence_ids=(str(row["asset_identity_sha256"]),),
            provenance={
                "training_manifest_sha256": sha256_file(self.root / "manifest.json"),
                "annotation_package_manifest_sha256": self.manifest[
                    "annotation_package_manifest_sha256"
                ],
                "reference_authority": "single_expert",
                "formal_acceptance": False,
            },
            counterfactual=None,
        )
