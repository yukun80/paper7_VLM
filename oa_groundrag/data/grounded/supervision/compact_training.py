"""Stage 5 独立 compact 训练消息合同。

发布阶段一次性读取 Stage 4 raw supervision；发布后的 loader 只依赖保留的
Region collection 和本目录自身。这样可以删除可恢复工作目录与模型草稿，同时仍能
逐条重算 GT-mask 资产、消息顺序和 assistant 监督身份。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Mapping

from oa_groundrag.phase3.common import (
    canonical_json,
    read_json,
    read_jsonl,
    safe_join,
    sha256_file,
    sha256_text,
)
from oa_groundrag.phase4.artifacts import AtomicArtifactDirectory
from oa_groundrag.phase4.messages import build_mask_grounded_region_messages
from oa_groundrag.phase4.outputs import REGION_OUTPUT_SCHEMA_VERSION, parse_region_model_output

from .contracts import fail
from .model_assisted import (
    EXPERT_AUTHORITY,
    MODEL_AUTHORITY,
    REFERENCE_AUTHORITY,
    PACKAGE_MANIFEST_FIELDS,
    TRAINING_MANIFEST_FIELDS,
    TRAINING_ROW_FIELDS,
    TRAINING_MANIFEST_SCHEMA,
    ModelAssistedTrainingArtifact,
    _file_contract,
    _ledger_contract,
    _load_collection_context,
    _manifest_payload,
    _mapping,
    _ordinary_file,
    _ordinary_root,
    _sha256,
    _text,
    _validate_ledger,
)
from .region_pipeline import ledger_rows, region_asset_identity


EXPECTED_COMPACT_COUNT = 6_974
COMPACT_ROW_SCHEMA = "oa_groundrag.mask_grounded_region.compact_training_message.v3"
COMPACT_MANIFEST_SCHEMA = "oa_groundrag.mask_grounded_region.compact_training_messages.v3"

COMPACT_ROW_FIELDS = (
    "schema_version",
    "record_id",
    "parent_id",
    "source",
    "logical_role",
    "task_family",
    "messages",
    "assistant_target_sha256",
    "source_supervision_identity_sha256",
    "asset_identity_sha256",
    "supervision_authority",
)
COMPACT_MANIFEST_FIELDS = (
    "schema_version",
    "compact_id",
    "split",
    "record_count",
    "ordered_record_ids_sha256",
    "authority_counts",
    "source_collection",
    "historical_source",
    "draft_provenance",
    "messages",
    "assistant_target_schema",
    "ledger",
    "training_eligible",
    "reference_authority",
    "expert_consensus",
    "gold",
    "thresholds_frozen",
    "formal_acceptance",
    "scientific_acceptance",
    "sealed_test_evaluated",
)


@dataclass(frozen=True)
class CompactTrainingArtifact:
    """已严格验证、可脱离 raw supervision 使用的训练消息。"""

    root: Path
    manifest: Mapping[str, Any]
    rows: tuple[Mapping[str, Any], ...]


def _exact(value: Mapping[str, Any], fields: tuple[str, ...], location: str) -> None:
    expected = set(fields)
    if set(value) != expected:
        fail(
            "SCHEMA_MISMATCH",
            f"{location} 字段不匹配",
            details={
                "missing": sorted(expected - set(value)),
                "unknown": sorted(set(value) - expected),
            },
        )


def _assistant_target(messages: Any, *, location: str) -> tuple[str, dict[str, Any]]:
    if (
        not isinstance(messages, list)
        or len(messages) != 2
        or messages[0].get("role") != "user"
        or messages[1].get("role") != "assistant"
    ):
        fail("ANNOTATION_INVALID", f"{location} 必须是单轮 user/assistant")
    content = messages[1].get("content")
    if (
        not isinstance(content, list)
        or len(content) != 1
        or not isinstance(content[0], dict)
        or set(content[0]) != {"type", "text"}
        or content[0].get("type") != "text"
        or not isinstance(content[0].get("text"), str)
    ):
        fail("ANNOTATION_INVALID", f"{location} assistant target 合同非法")
    raw = str(content[0]["text"])
    parsed = parse_region_model_output(raw).to_dict()
    if raw != canonical_json(parsed):
        fail("ANNOTATION_INVALID", f"{location} assistant target 必须 canonical 序列化")
    return raw, parsed


def _historical_source(source_root: Path, source_manifest: Mapping[str, Any]) -> dict[str, Any]:
    package_root = _ordinary_root(
        source_manifest["supervision_package_root"],
        location="source supervision package",
    )
    package_manifest = _mapping(read_json(package_root / "manifest.json"), "package manifest")
    draft_runs_path = safe_join(
        package_root,
        _text(package_manifest["draft_runs"]["path"], "package.draft_runs.path"),
        location="package.draft_runs.path",
    )
    _ordinary_file(draft_runs_path, location="draft runs")
    draft_runs = read_jsonl(draft_runs_path)
    provenance = []
    for index, row in enumerate(draft_runs):
        if not isinstance(row, dict):
            fail("ANNOTATION_INVALID", f"draft_runs[{index}] 必须是对象")
        provenance.append({
            "draft_run_id": row.get("draft_run_id"),
            "model_repository": row.get("model_repository"),
            "model_revision": row.get("model_revision"),
            "model_identity": row.get("model_identity"),
            "processor_identity": row.get("processor_identity"),
            "prompt_sha256": row.get("prompt_sha256"),
            "config_semantic_sha256": row.get("config_semantic_sha256"),
            "generation": row.get("generation"),
        })
    return {
        "training_manifest_sha256": sha256_file(source_root / "manifest.json"),
        "messages_sha256": sha256_file(source_root / "messages.jsonl"),
        "supervision_package_manifest_sha256": sha256_file(package_root / "manifest.json"),
        "source_record_count": package_manifest.get("record_count"),
        "eligible_count": package_manifest.get("eligible_count"),
        "excluded_count": package_manifest.get("excluded_count"),
        "authority_counts": package_manifest.get("authority_counts"),
        "exclusion_counts": package_manifest.get("exclusion_counts"),
        "draft_provenance": provenance,
    }


def _load_v2_source_lightweight(source_training_root: Path | str) -> ModelAssistedTrainingArtifact:
    """流式验证 raw 字节身份，只装载最终 6,974 条消息。

    旧完整 loader 会把 8,450 条 raw draft 与解析结果同时常驻内存。compact 发布不需要
    重新持有这些大对象，但仍通过 package ledger 对其逐文件 SHA/size 做完整验证。
    """

    root = _ordinary_root(source_training_root, location="source v2 training root")
    manifest_path = root / "manifest.json"
    _ordinary_file(manifest_path, location="source v2 manifest")
    manifest = _mapping(read_json(manifest_path), "source v2 manifest")
    _exact(manifest, TRAINING_MANIFEST_FIELDS, "source v2 manifest")
    if (
        manifest.get("schema_version") != TRAINING_MANIFEST_SCHEMA
        or manifest.get("split") != "train"
        or manifest.get("record_count") != EXPECTED_COMPACT_COUNT
        or manifest.get("assistant_target_schema") != REGION_OUTPUT_SCHEMA_VERSION
        or manifest.get("training_eligible") is not True
        or manifest.get("reference_authority") != REFERENCE_AUTHORITY
        or manifest.get("gold") is not False
        or manifest.get("formal_acceptance") is not False
        or manifest.get("scientific_acceptance") is not False
    ):
        fail("SPLIT_FORBIDDEN", "source v2 training manifest 身份非法")
    # collection 索引本身仍完整验证；compact 随后会对 6,974 条实际训练记录逐资产
    # 重算 identity，因此无需先额外扫描 8,450 条成员资产造成重复内存/IO 压力。
    collection = _load_collection_context(
        manifest["source_collection_root"],
        verify_members=False,
    )
    if manifest["source_collection_manifest_sha256"] != collection.manifest_sha256:
        fail("ANNOTATION_INVALID", "source v2 collection identity 漂移")
    package_root = _ordinary_root(
        manifest["supervision_package_root"],
        location="source v2 supervision package",
    )
    package_manifest_path = package_root / "manifest.json"
    _ordinary_file(package_manifest_path, location="source package manifest")
    if sha256_file(package_manifest_path) != manifest["supervision_package_manifest_sha256"]:
        fail("ANNOTATION_INVALID", "source v2 package manifest SHA 漂移")
    package_manifest = _mapping(read_json(package_manifest_path), "source package manifest")
    _exact(package_manifest, PACKAGE_MANIFEST_FIELDS, "source package manifest")
    payload_keys = (
        "supervision", "exclusions", "model_drafts", "draft_runs",
        "expert_annotations", "import_provenance",
    )
    payload_paths = {
        key: _manifest_payload(package_root, package_manifest, key)
        for key in payload_keys
    }
    _validate_ledger(package_root, package_manifest, {path.name for path in payload_paths.values()})
    if (
        package_manifest.get("record_count") != 8_450
        or package_manifest.get("eligible_count") != EXPECTED_COMPACT_COUNT
        or package_manifest.get("eligible_count") + package_manifest.get("excluded_count") != 8_450
    ):
        fail("ANNOTATION_INVALID", "source package 8,450/6,974 统计漂移")
    messages_path = _manifest_payload(root, manifest, "messages")
    _validate_ledger(root, manifest, {messages_path.name})
    values = read_jsonl(messages_path)
    if len(values) != EXPECTED_COMPACT_COUNT:
        fail("ANNOTATION_INVALID", "source v2 messages 数量漂移")
    seen: set[str] = set()
    for index, value in enumerate(values):
        row = _mapping(value, f"source messages[{index}]")
        _exact(row, TRAINING_ROW_FIELDS, f"source messages[{index}]")
        record_id = _text(row.get("record_id"), f"source messages[{index}].record_id")
        if record_id in seen:
            fail("ANNOTATION_INVALID", f"source v2 record 重复：{record_id}")
        seen.add(record_id)
    if manifest["ordered_record_ids_sha256"] != sha256_text(canonical_json([
        row["record_id"] for row in values
    ])):
        fail("ANNOTATION_INVALID", "source v2 ordered IDs 漂移")
    return ModelAssistedTrainingArtifact(root=root, manifest=manifest, rows=tuple(values))


def publish_compact_training_messages(
    *,
    source_training_root: Path | str,
    output_root: Path | str,
) -> dict[str, Any]:
    """原子发布 6,974 条 compact 消息；发布后不再依赖 raw package。"""

    source = _load_v2_source_lightweight(source_training_root)
    if len(source.rows) != EXPECTED_COMPACT_COUNT:
        fail(
            "ANNOTATION_INVALID",
            f"compact source 必须精确为 {EXPECTED_COMPACT_COUNT} 条",
        )
    collection = _load_collection_context(
        source.manifest["source_collection_root"],
        verify_members=False,
    )
    history = _historical_source(source.root, source.manifest)
    entries = {str(entry.record["record_id"]): entry for entry in collection.entries}
    rows: list[dict[str, Any]] = []
    for index, source_row in enumerate(source.rows):
        record_id = _text(source_row.get("record_id"), f"source[{index}].record_id")
        entry = entries.get(record_id)
        if entry is None or entry.record.get("split") != "train":
            fail("SPLIT_FORBIDDEN", f"compact source 引用非 train record：{record_id}")
        target, parsed = _assistant_target(source_row.get("messages"), location=f"source[{index}]")
        actual_asset = region_asset_identity(entry.member_root, entry.record["assets"])
        if source_row.get("asset_identity_sha256") != actual_asset:
            fail("ANNOTATION_INVALID", f"compact source asset identity 漂移：{record_id}")
        expected_messages = build_mask_grounded_region_messages(
            entry.record,
            asset_root=entry.member_root,
            assistant_target=canonical_json(parsed),
        )
        if source_row.get("messages") != expected_messages:
            fail("ANNOTATION_INVALID", f"compact source message 漂移：{record_id}")
        authority = source_row.get("supervision_authority")
        if authority not in {EXPERT_AUTHORITY, MODEL_AUTHORITY}:
            fail("ANNOTATION_INVALID", f"compact supervision authority 非法：{record_id}")
        rows.append({
            "schema_version": COMPACT_ROW_SCHEMA,
            "record_id": record_id,
            "parent_id": entry.record["parent_id"],
            "source": entry.record["source"],
            "logical_role": "mask_grounded_train",
            "task_family": "mask_grounded_region_description",
            "messages": expected_messages,
            "assistant_target_sha256": sha256_text(target),
            "source_supervision_identity_sha256": _sha256(
                source_row.get("supervision_identity_sha256"),
                f"source[{index}].supervision_identity_sha256",
            ),
            "asset_identity_sha256": actual_asset,
            "supervision_authority": authority,
        })
    authority_counts = dict(sorted(Counter(row["supervision_authority"] for row in rows).items()))
    ordered_sha = sha256_text(canonical_json([row["record_id"] for row in rows]))
    output = Path(os.path.abspath(Path(output_root)))
    with AtomicArtifactDirectory(output) as writer:
        assert writer.staging is not None
        writer.write_jsonl("messages.jsonl", rows)
        ledger = ledger_rows(writer.staging, ("messages.jsonl",))
        writer.write_jsonl("SHA256SUMS.jsonl", ledger)
        source_collection = {
            "root": str(collection.root),
            "manifest_sha256": collection.manifest_sha256,
        }
        historical = {key: value for key, value in history.items() if key != "draft_provenance"}
        compact_id = "compact_" + sha256_text(canonical_json({
            "source_collection": source_collection,
            "historical_source": historical,
            "messages_sha256": sha256_file(writer.path("messages.jsonl")),
            "ordered_record_ids_sha256": ordered_sha,
        }))
        manifest = {
            "schema_version": COMPACT_MANIFEST_SCHEMA,
            "compact_id": compact_id,
            "split": "train",
            "record_count": len(rows),
            "ordered_record_ids_sha256": ordered_sha,
            "authority_counts": authority_counts,
            "source_collection": source_collection,
            "historical_source": historical,
            "draft_provenance": history["draft_provenance"],
            "messages": _file_contract(writer.path("messages.jsonl")),
            "assistant_target_schema": REGION_OUTPUT_SCHEMA_VERSION,
            "ledger": _ledger_contract(writer.path("SHA256SUMS.jsonl"), ledger),
            "training_eligible": True,
            "reference_authority": REFERENCE_AUTHORITY,
            "expert_consensus": False,
            "gold": False,
            "thresholds_frozen": False,
            "formal_acceptance": False,
            "scientific_acceptance": False,
            "sealed_test_evaluated": False,
        }
        writer.write_json("manifest.json", manifest)
        root = writer.publish()
    artifact = load_compact_training_messages(root)
    return {
        "ok": True,
        "root": str(root),
        "compact_id": artifact.manifest["compact_id"],
        "manifest_sha256": sha256_file(root / "manifest.json"),
        "messages_sha256": sha256_file(root / "messages.jsonl"),
        "record_count": len(artifact.rows),
        "authority_counts": artifact.manifest["authority_counts"],
        "formal_acceptance": False,
    }


def load_compact_training_messages(training_root: Path | str) -> CompactTrainingArtifact:
    """严格加载 compact；不会读取旧 work、annotation package 或 v2 messages。"""

    root = _ordinary_root(training_root, location="compact_training_root")
    manifest_path = root / "manifest.json"
    _ordinary_file(manifest_path, location="compact manifest")
    manifest = _mapping(read_json(manifest_path), "compact manifest")
    _exact(manifest, COMPACT_MANIFEST_FIELDS, "compact manifest")
    false_fields = (
        "expert_consensus", "gold", "thresholds_frozen", "formal_acceptance",
        "scientific_acceptance", "sealed_test_evaluated",
    )
    if (
        manifest.get("schema_version") != COMPACT_MANIFEST_SCHEMA
        or manifest.get("split") != "train"
        or manifest.get("record_count") != EXPECTED_COMPACT_COUNT
        or manifest.get("assistant_target_schema") != REGION_OUTPUT_SCHEMA_VERSION
        or manifest.get("training_eligible") is not True
        or manifest.get("reference_authority") != REFERENCE_AUTHORITY
        or any(manifest.get(field) is not False for field in false_fields)
    ):
        fail("SPLIT_FORBIDDEN", "compact manifest 身份或科学边界非法")
    source_collection = _mapping(manifest["source_collection"], "manifest.source_collection")
    _exact(source_collection, ("root", "manifest_sha256"), "manifest.source_collection")
    collection = _load_collection_context(source_collection["root"], verify_members=False)
    if source_collection["manifest_sha256"] != collection.manifest_sha256:
        fail("ANNOTATION_INVALID", "compact collection identity 漂移")
    historical = _mapping(manifest["historical_source"], "manifest.historical_source")
    _exact(
        historical,
        (
            "training_manifest_sha256", "messages_sha256",
            "supervision_package_manifest_sha256", "source_record_count",
            "eligible_count", "excluded_count", "authority_counts", "exclusion_counts",
        ),
        "manifest.historical_source",
    )
    for key in ("training_manifest_sha256", "messages_sha256", "supervision_package_manifest_sha256"):
        _sha256(historical[key], f"manifest.historical_source.{key}")
    if historical["eligible_count"] != EXPECTED_COMPACT_COUNT:
        fail("ANNOTATION_INVALID", "historical eligible_count 漂移")
    provenance = manifest["draft_provenance"]
    if not isinstance(provenance, list) or not provenance:
        fail("ANNOTATION_INVALID", "compact 必须冻结至少一个 draft provenance")
    messages_path = _manifest_payload(root, manifest, "messages")
    _validate_ledger(root, manifest, {messages_path.name})
    values = read_jsonl(messages_path)
    if len(values) != EXPECTED_COMPACT_COUNT:
        fail("ANNOTATION_INVALID", "compact messages 数量漂移")
    entries = {str(entry.record["record_id"]): entry for entry in collection.entries}
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    authority_counts: Counter[str] = Counter()
    for index, value in enumerate(values):
        row = _mapping(value, f"messages[{index}]")
        _exact(row, COMPACT_ROW_FIELDS, f"messages[{index}]")
        record_id = _text(row["record_id"], f"messages[{index}].record_id")
        if record_id in seen:
            fail("ANNOTATION_INVALID", f"compact record 重复：{record_id}")
        seen.add(record_id)
        entry = entries.get(record_id)
        if entry is None or entry.record.get("split") != "train":
            fail("SPLIT_FORBIDDEN", f"compact 引用非 train record：{record_id}")
        target, parsed = _assistant_target(row["messages"], location=f"messages[{index}]")
        actual_asset = region_asset_identity(entry.member_root, entry.record["assets"])
        expected = {
            "schema_version": COMPACT_ROW_SCHEMA,
            "record_id": record_id,
            "parent_id": entry.record["parent_id"],
            "source": entry.record["source"],
            "logical_role": "mask_grounded_train",
            "task_family": "mask_grounded_region_description",
            "messages": build_mask_grounded_region_messages(
                entry.record,
                asset_root=entry.member_root,
                assistant_target=canonical_json(parsed),
            ),
            "assistant_target_sha256": sha256_text(target),
            "source_supervision_identity_sha256": _sha256(
                row["source_supervision_identity_sha256"],
                f"messages[{index}].source_supervision_identity_sha256",
            ),
            "asset_identity_sha256": actual_asset,
            "supervision_authority": row["supervision_authority"],
        }
        if row["supervision_authority"] not in {EXPERT_AUTHORITY, MODEL_AUTHORITY} or row != expected:
            fail("ANNOTATION_INVALID", f"compact row identity 漂移：{record_id}")
        authority_counts[str(row["supervision_authority"])] += 1
        rows.append(row)
    ordered_sha = sha256_text(canonical_json([row["record_id"] for row in rows]))
    expected_compact_id = "compact_" + sha256_text(canonical_json({
        "source_collection": source_collection,
        "historical_source": historical,
        "messages_sha256": sha256_file(messages_path),
        "ordered_record_ids_sha256": ordered_sha,
    }))
    if (
        manifest["ordered_record_ids_sha256"] != ordered_sha
        or manifest["authority_counts"] != dict(sorted(authority_counts.items()))
        or manifest["compact_id"] != expected_compact_id
    ):
        fail("ANNOTATION_INVALID", "compact manifest 统计或 identity 漂移")
    return CompactTrainingArtifact(root=root, manifest=manifest, rows=tuple(rows))


class CompactTrainingMessageDataset:
    """把 compact 训练行暴露给现有 Qwen DescriptionCollator。"""

    def __init__(self, training_root: Path | str) -> None:
        artifact = load_compact_training_messages(training_root)
        self.root = artifact.root
        self.manifest = artifact.manifest
        self.records = artifact.rows

    def __len__(self) -> int:
        return len(self.records)

    def set_epoch(self, epoch: int) -> None:
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            fail("TYPE_MISMATCH", "compact epoch 必须是非负整数")

    def __getitem__(self, index: int) -> Any:
        from oa_groundrag.phase4.contracts import MaskMode
        from oa_groundrag.phase4.data import DescriptionSample

        if index < 0:
            index += len(self.records)
        if index < 0 or index >= len(self.records):
            raise IndexError(index)
        row = self.records[index]
        assistant = row["messages"][-1]["content"][0]["text"]
        return DescriptionSample(
            record_id=str(row["record_id"]),
            parent_id=str(row["parent_id"]),
            logical_role="mask_grounded_train",
            task_family="mask_grounded_region_description",
            messages=tuple(row["messages"]),
            reference_responses=(str(assistant),),
            mask_mode=MaskMode.GT_MASK,
            evidence_ids=(str(row["asset_identity_sha256"]),),
            provenance={
                "compact_manifest_sha256": sha256_file(self.root / "manifest.json"),
                "source_supervision_identity_sha256": row[
                    "source_supervision_identity_sha256"
                ],
                "supervision_authority": row["supervision_authority"],
                "reference_authority": REFERENCE_AUTHORITY,
                "gold": False,
                "formal_acceptance": False,
            },
            counterfactual=None,
        )
