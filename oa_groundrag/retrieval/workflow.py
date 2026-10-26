"""Stage 6 dev 选择、检索、Stage 5 best 文本生成与原子发布。"""

from __future__ import annotations

from collections import Counter
import gc
import importlib.metadata
import os
from pathlib import Path
import platform
import time
from typing import Any, Mapping, Sequence

import numpy as np

from oa_groundrag.artifacts.identity import (
    canonical_json,
    sha256_file,
    sha256_text,
)
from oa_groundrag.artifacts.io import first_symlink_component
from oa_groundrag.data.rs_general.io import (
    portable_relative_path,
    read_json,
    read_jsonl,
)
from oa_groundrag.artifacts.directory import AtomicArtifactDirectory
from oa_groundrag.vlm.errors import (
    ContractError,
    ModelError,
    PredictionError,
    ReasonCode,
    SelectionError,
)
from oa_groundrag.grounding.outputs import parse_region_model_output
from oa_groundrag.training.grounding.config import (
    load_stage5_config,
)
from oa_groundrag.vlm.grounded_adapter import (
    load_stage5_best_generator,
    resolve_stage5_best,
)

from .bank import validate_bank
from .contracts import (
    PASS2_FAILURE_SCHEMA,
    PASS2_PREDICTION_SCHEMA,
    PASS2_RUN_SCHEMA,
    RETRIEVAL_REPORT_SCHEMA,
    SELECTION_SCHEMA,
    KnowledgeType,
    QueryIntent,
    RagMode,
    Stage6Config,
    load_source_registry,
    load_stage6_config,
    packet_identity,
    query_identity,
    route_text_rag,
)
from .pass2 import (
    PASS2_ASSISTANT_PREFILL,
    PASS2_CONSTRAINT_SCHEMA,
    build_pass2_messages,
    build_pass2_logits_processor,
    parse_pass2_output,
    pass2_constraint_identity,
    prompt_sha256,
    validate_prompt_fairness,
)
from .search import (
    BGEM3DenseEmbedder,
    HybridRetriever,
    build_balanced_packet,
    build_counter_limitation_query,
    build_interpretation_query,
    dense_model_identity,
    make_query_row,
    quota_counts,
)
from .runtime import load_runtime_bank_payload


_RETRIEVAL_PAYLOAD_FILES = frozenset({
    "selection.json",
    "queries.jsonl",
    "packets.jsonl",
    "query_embeddings.npy",
    "query_dense_index.json",
    "report.json",
    "environment.json",
})
_RETRIEVAL_ALL_FILES = _RETRIEVAL_PAYLOAD_FILES | {
    "SHA256SUMS.jsonl",
    "manifest.json",
}
_RUN_PAYLOAD_FILES = frozenset({
    "predictions.jsonl",
    "failures.jsonl",
    "report.json",
    "environment.json",
})
_RUN_ALL_FILES = _RUN_PAYLOAD_FILES | {"SHA256SUMS.jsonl", "manifest.json"}


def _regular_file(path: Path, *, label: str, expected_sha256: str | None = None) -> None:
    linked = first_symlink_component(path)
    if (
        linked is not None
        or not path.is_file()
        or path.is_symlink()
        or path.stat().st_nlink != 1
    ):
        raise ContractError(ReasonCode.OUTPUT_LINK, f"{label} 必须是普通单链接文件：{path}")
    if expected_sha256 is not None and sha256_file(path) != expected_sha256:
        raise ContractError(ReasonCode.BENCHMARK_IDENTITY_MISMATCH, f"{label} SHA-256 漂移：{path}")


def _reject_sealed_path(path: Path, *, label: str) -> None:
    for part in path.resolve(strict=False).parts:
        normalized = part.lower().replace("-", "_")
        tokens = tuple(value for value in normalized.split("_") if value)
        if "sealed" in tokens or "test" in tokens:
            raise ContractError(
                ReasonCode.SPLIT_FORBIDDEN,
                f"{label} 指向 sealed/test 路径，Stage 6 拒绝读取：{path}",
            )


def _ledger_rows(root: Path, files: Sequence[str]) -> list[dict[str, Any]]:
    return [
        {
            "path": relative,
            "size_bytes": (root / relative).stat().st_size,
            "sha256": sha256_file(root / relative),
        }
        for relative in sorted(files)
    ]


def _validate_artifact_files(root: Path, *, expected: set[str] | frozenset[str]) -> None:
    linked = first_symlink_component(root)
    if linked is not None or root.is_symlink() or not root.is_dir():
        raise ContractError(ReasonCode.OUTPUT_LINK, f"artifact root 必须是普通目录：{root}")
    links = [path for path in root.rglob("*") if path.is_symlink()]
    if links:
        raise ContractError(ReasonCode.OUTPUT_LINK, f"artifact 含链接：{links[0]}")
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    if actual != set(expected):
        raise ContractError(
            ReasonCode.BENCHMARK_IDENTITY_MISMATCH,
            "artifact 文件集漂移",
            details={"missing": sorted(set(expected) - actual), "unexpected": sorted(actual - set(expected))},
        )


def _validate_ledger(root: Path, payload_files: set[str] | frozenset[str]) -> str:
    rows = read_jsonl(root / "SHA256SUMS.jsonl")
    if len(rows) != len(payload_files):
        raise ContractError(ReasonCode.BENCHMARK_IDENTITY_MISMATCH, "artifact ledger 行数不匹配")
    observed: set[str] = set()
    for row in rows:
        if set(row) != {"path", "size_bytes", "sha256"} or row.get("path") in observed:
            raise ContractError(ReasonCode.UNKNOWN_FIELD, "artifact ledger 字段或 path 重复")
        relative = str(row["path"])
        portable_relative_path(relative, location="ledger.path")
        if relative not in payload_files:
            raise ContractError(ReasonCode.PATH_ESCAPE, f"ledger 未登记 payload：{relative}")
        observed.add(relative)
        path = root / relative
        _regular_file(path, label="ledger payload")
        if path.stat().st_size != row.get("size_bytes") or sha256_file(path) != row.get("sha256"):
            raise ContractError(ReasonCode.BENCHMARK_IDENTITY_MISMATCH, f"ledger hash/size 漂移：{relative}")
    if observed != set(payload_files):
        raise ContractError(ReasonCode.BENCHMARK_IDENTITY_MISMATCH, "ledger payload 集合不完整")
    return sha256_file(root / "SHA256SUMS.jsonl")


def preflight(config_path: Path | str, *, require_dense_model: bool = True) -> dict[str, Any]:
    """只读核验 Stage 6 来源、dev、Stage 5 best 与本地权重边界。"""

    config = load_stage6_config(config_path)
    registry = load_source_registry(config.source_registry_path)
    _reject_sealed_path(config.dev.eval_root, label="OA-GroundedEval-dev")
    _reject_sealed_path(config.dev.prediction_root, label="Pass-1 predictions")
    reference = resolve_stage5_best(config.stage5)
    stage5, pointer, checkpoint = (
        reference.config,
        reference.pointer,
        reference.checkpoint,
    )
    eval_manifest_path = config.dev.eval_root / "manifest.json"
    prediction_manifest_path = config.dev.prediction_root / "manifest.json"
    predictions_path = config.dev.prediction_root / "predictions.jsonl"
    _regular_file(eval_manifest_path, label="OA-GroundedEval-dev manifest", expected_sha256=config.dev.eval_manifest_sha256)
    _regular_file(prediction_manifest_path, label="Pass-1 manifest", expected_sha256=config.dev.prediction_manifest_sha256)
    _regular_file(predictions_path, label="Pass-1 predictions", expected_sha256=config.dev.predictions_sha256)
    eval_manifest = read_json(eval_manifest_path)
    prediction_manifest = read_json(prediction_manifest_path)
    if eval_manifest.get("sealed_test_accessed") is not False:
        raise ContractError(ReasonCode.SPLIT_FORBIDDEN, "Eval manifest 未证明 sealed test 未访问")
    if prediction_manifest.get("sealed_test_evaluated") is not False:
        raise ContractError(ReasonCode.SPLIT_FORBIDDEN, "Pass-1 manifest 未证明 sealed test 未访问")
    if prediction_manifest.get("model_role") != "mask_grounded_region_adapter":
        raise ContractError(ReasonCode.PREDICTION_IDENTITY_MISMATCH, "Pass-1 不是 Stage 5 Adapter 输出")
    dense_identity: Mapping[str, Any] | None = None
    if config.dense.model_root.exists() or config.dense.model_root.is_symlink():
        dense_identity = dense_model_identity(
            config.dense.model_root,
            repo_id=config.dense.repo_id,
            revision=config.dense.revision,
        )
    elif require_dense_model:
        raise ContractError(ReasonCode.ASSET_MISSING, f"BGE-M3 尚未下载：{config.dense.model_root}")
    return {
        "ok": True,
        "config_semantic_sha256": config.semantic_sha256,
        "source_count": len(registry.sources),
        "source_registry_semantic_sha256": registry.semantic_sha256,
        "stage5_config_semantic_sha256": stage5.semantic_sha256,
        "stage5_best_step": pointer["step"],
        "stage5_checkpoint": str(checkpoint),
        "stage5_best_pointer_sha256": config.stage5.best_pointer_sha256,
        "stage5_checkpoint_manifest_sha256": config.stage5.checkpoint_manifest_sha256,
        "stage5_adapter_sha256": config.stage5.adapter_sha256,
        "eval_manifest_sha256": config.dev.eval_manifest_sha256,
        "prediction_manifest_sha256": config.dev.prediction_manifest_sha256,
        "predictions_sha256": config.dev.predictions_sha256,
        "dense_model_available": dense_identity is not None,
        "dense_model_files_identity_sha256": None if dense_identity is None else dense_identity["files_identity_sha256"],
        "sealed_test_accessed": False,
    }


def _selection_identity(row: Mapping[str, Any]) -> str:
    return sha256_text(canonical_json({key: value for key, value in row.items() if key != "selection_id"}))


def prepare_dev_selection(config: Stage6Config) -> dict[str, Any]:
    """冻结 target-present baseline/correct-mask 记录；不依据 RAG 输出重新选样。"""

    preflight(config.config_path, require_dense_model=False)
    eval_records = read_jsonl(config.dev.eval_root / "records.jsonl")
    predictions = read_jsonl(config.dev.prediction_root / "predictions.jsonl")
    by_record: dict[str, Mapping[str, Any]] = {}
    for prediction in predictions:
        record_id = prediction.get("record_id")
        if not isinstance(record_id, str) or not record_id or record_id in by_record:
            raise SelectionError(ReasonCode.PREDICTION_INVALID, "Pass-1 record_id 非法或重复")
        by_record[record_id] = prediction
    buckets: dict[str, list[dict[str, Any]]] = {}
    for record in eval_records:
        counterfactual = record.get("program_facts", {}).get("counterfactual")
        if (
            record.get("split") != "val"
            or record.get("target_status") != "target_present"
            or not isinstance(counterfactual, dict)
            or counterfactual.get("kind") != "baseline_correct_mask"
        ):
            continue
        record_id = str(record.get("record_id", ""))
        prediction = by_record.get(record_id)
        if prediction is None:
            continue
        parsed = parse_region_model_output(prediction.get("model_output"))
        observation = parsed.to_dict()
        if observation.get("target_status") != "target_present":
            continue
        roles = record.get("formal_model_input_roles")
        if not isinstance(roles, list) or set(roles) != {"optical_full", "binary_mask", "context_crop"}:
            raise SelectionError(ReasonCode.ASSET_ROLE_LEAKAGE, f"{record_id}: formal roles 非法")
        source = str(record.get("source", ""))
        if not source:
            raise SelectionError(ReasonCode.TYPE_MISMATCH, f"{record_id}: source 为空")
        program_facts = record.get("program_facts")
        if not isinstance(program_facts, dict):
            raise SelectionError(ReasonCode.TYPE_MISMATCH, f"{record_id}: program_facts 非法")
        child = {
            "record_id": record_id,
            "parent_id": str(record.get("parent_id")),
            "sample_id": str(record.get("sample_id")),
            "source": source,
            "split": "val",
            "target_status": "target_present",
            "formal_model_input_roles": list(roles),
            "available_modalities": ["optical"],
            "program_facts": program_facts,
            "program_facts_sha256": sha256_text(canonical_json(program_facts)),
            "observation": observation,
            "observation_sha256": sha256_text(canonical_json(observation)),
            "pass1_prediction_sha256": sha256_text(canonical_json(prediction)),
        }
        buckets.setdefault(source, []).append(child)
    for values in buckets.values():
        values.sort(key=lambda value: value["record_id"])
    source_counts = {source: len(values) for source, values in sorted(buckets.items())}
    if sum(source_counts.values()) != config.dev.expected_records or len(source_counts) != 5 or len(set(source_counts.values())) != 1:
        raise SelectionError(
            ReasonCode.VALIDATION_SELECTION_INVALID,
            "Stage 6 dev pool 不是冻结的 5-source balanced 80 records",
            details={"source_counts": source_counts},
        )
    ordered: list[dict[str, Any]] = []
    for offset in range(max(source_counts.values())):
        for source in sorted(buckets):
            ordered.append(buckets[source][offset])
    row = {
        "schema_version": SELECTION_SCHEMA,
        "selection_id": "",
        "algorithm": "source_sorted_round_robin_target_present_baseline_correct_mask.v1",
        "config_semantic_sha256": config.semantic_sha256,
        "eval_manifest_sha256": config.dev.eval_manifest_sha256,
        "prediction_manifest_sha256": config.dev.prediction_manifest_sha256,
        "predictions_sha256": config.dev.predictions_sha256,
        "question": config.dev.question,
        "question_sha256": sha256_text(config.dev.question),
        "task": config.dev.task.value,
        "record_count": len(ordered),
        "source_counts": source_counts,
        "records": ordered,
        "development_only": True,
        "sealed_test_accessed": False,
    }
    row["selection_id"] = _selection_identity(row)
    return row


def _environment() -> dict[str, Any]:
    packages = {}
    for name in ("torch", "transformers", "peft", "numpy", "PyMuPDF", "rapidocr", "onnxruntime"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
    }


def retrieve_dev(config_path: Path | str) -> dict[str, Any]:
    config = load_stage6_config(config_path)
    validate_bank(config.bank_root, config=config, verify_sources=True)
    if config.retrieval_root.exists() or config.retrieval_root.is_symlink():
        raise ContractError(ReasonCode.OUTPUT_EXISTS, f"retrieval root 已存在：{config.retrieval_root}")
    selection = prepare_dev_selection(config)
    bank_manifest, units, embeddings, dense_ids = load_runtime_bank_payload(config.bank_root)
    queries: list[dict[str, Any]] = []
    for selected in selection["records"]:
        shared = {
            "record_id": selected["record_id"],
            "available_modalities": selected["available_modalities"],
            "observation_sha256": selected["observation_sha256"],
            "bank_id": bank_manifest["bank_id"],
        }
        queries.append(make_query_row(
            **shared,
            intent=QueryIntent.INTERPRETATION,
            text=build_interpretation_query(
                question=config.dev.question,
                observation=selected["observation"],
                available_modalities=selected["available_modalities"],
            ),
            knowledge_types=(KnowledgeType.INTERPRETATION,),
        ))
        queries.append(make_query_row(
            **shared,
            intent=QueryIntent.COUNTER_LIMITATION,
            text=build_counter_limitation_query(
                question=config.dev.question,
                observation=selected["observation"],
                available_modalities=selected["available_modalities"],
            ),
            knowledge_types=(KnowledgeType.CONFOUNDER, KnowledgeType.LIMITATION),
        ))
    embedder = BGEM3DenseEmbedder(
        config.dense.model_root,
        repo_id=config.dense.repo_id,
        revision=config.dense.revision,
        device=config.dense.device,
        batch_size=config.dense.batch_size,
        max_tokens=config.dense.max_tokens,
    )
    query_embeddings = embedder.encode([row["text"] for row in queries])
    retriever = HybridRetriever(
        sqlite_path=config.bank_root / "lexical.sqlite3",
        units=units,
        embeddings=embeddings,
        dense_unit_ids=dense_ids,
        config=config.retrieval,
    )
    packets: list[dict[str, Any]] = []
    try:
        for offset, selected in enumerate(selection["records"]):
            interpretation_query = queries[offset * 2]
            counter_query = queries[offset * 2 + 1]
            packet = build_balanced_packet(
                record_id=selected["record_id"],
                bank_id=bank_manifest["bank_id"],
                interpretation_query_id=interpretation_query["query_id"],
                counter_query_id=counter_query["query_id"],
                interpretation_candidates=retriever.retrieve(interpretation_query, query_embeddings[offset * 2]),
                counter_candidates=retriever.retrieve(counter_query, query_embeddings[offset * 2 + 1]),
                config=config.retrieval,
            )
            packets.append(packet)
    finally:
        retriever.close()
    utilization = Counter(item["knowledge_type"] for packet in packets for item in packet["items"])
    duplicate_packets = sum(
        len({(item["source_id"], item["pdf_page"]) for item in packet["items"]}) != len(packet["items"])
        for packet in packets
    )
    report = {
        "schema_version": RETRIEVAL_REPORT_SCHEMA,
        "selection_id": selection["selection_id"],
        "bank_id": bank_manifest["bank_id"],
        "record_count": len(selection["records"]),
        "query_count": len(queries),
        "packet_count": len(packets),
        "knowledge_type_item_counts": dict(sorted(utilization.items())),
        "packet_same_source_page_duplicate_count": duplicate_packets,
        "traceable_item_count": sum(len(packet["items"]) for packet in packets),
        "query_reproducible": True,
        "rank_reproducible": True,
        "recall_at_k": None,
        "mrr": None,
        "ndcg": None,
        "retrieval_gold_available": False,
        "formal_acceptance": False,
        "scientific_acceptance": False,
        "sealed_test_accessed": False,
    }
    with AtomicArtifactDirectory(config.retrieval_root) as writer:
        writer.write_json("selection.json", selection)
        writer.write_jsonl("queries.jsonl", queries)
        writer.write_jsonl("packets.jsonl", packets)
        with writer.path("query_embeddings.npy").open("wb") as handle:
            np.save(handle, query_embeddings, allow_pickle=False)
        writer.write_json("query_dense_index.json", {
            "schema_version": "oa_groundrag.text_rag.query_dense_index.v1",
            "model_identity": dict(embedder.identity),
            "embedding_dimension": int(query_embeddings.shape[1]),
            "dtype": str(query_embeddings.dtype),
            "query_ids": [row["query_id"] for row in queries],
            "query_order_sha256": sha256_text(canonical_json([row["query_id"] for row in queries])),
        })
        writer.write_json("report.json", report)
        writer.write_json("environment.json", _environment())
        assert writer.staging is not None
        payload_hashes = {relative: sha256_file(writer.path(relative)) for relative in sorted(_RETRIEVAL_PAYLOAD_FILES)}
        retrieval_id = sha256_text(canonical_json({
            "config_semantic_sha256": config.semantic_sha256,
            "bank_id": bank_manifest["bank_id"],
            "selection_id": selection["selection_id"],
            "payload_hashes": payload_hashes,
        }))
        writer.write_jsonl("SHA256SUMS.jsonl", _ledger_rows(writer.staging, _RETRIEVAL_PAYLOAD_FILES))
        writer.write_json("manifest.json", {
            "schema_version": "oa_groundrag.text_rag.retrieval_manifest.v1",
            "retrieval_id": retrieval_id,
            "config_semantic_sha256": config.semantic_sha256,
            "bank_id": bank_manifest["bank_id"],
            "selection_id": selection["selection_id"],
            "record_count": len(selection["records"]),
            "query_count": len(queries),
            "packet_count": len(packets),
            "dense_model_identity_sha256": sha256_text(canonical_json(embedder.identity)),
            "payload_hashes": payload_hashes,
            "ledger_sha256": sha256_file(writer.path("SHA256SUMS.jsonl")),
            "development_only": True,
            "formal_acceptance": False,
            "scientific_acceptance": False,
            "sealed_test_accessed": False,
        })
        writer.publish()
    del embedder
    gc.collect()
    return validate_retrieval(config.retrieval_root, config=config)


def validate_retrieval(root: Path | str, *, config: Stage6Config) -> dict[str, Any]:
    root = Path(root)
    _validate_artifact_files(root, expected=_RETRIEVAL_ALL_FILES)
    ledger_sha = _validate_ledger(root, _RETRIEVAL_PAYLOAD_FILES)
    manifest = read_json(root / "manifest.json")
    selection = read_json(root / "selection.json")
    queries = read_jsonl(root / "queries.jsonl")
    packets = read_jsonl(root / "packets.jsonl")
    report = read_json(root / "report.json")
    if selection.get("selection_id") != _selection_identity(selection):
        raise ContractError(ReasonCode.VALIDATION_SELECTION_INVALID, "selection identity 无法重算")
    expected_selection = prepare_dev_selection(config)
    if selection != expected_selection:
        raise ContractError(ReasonCode.VALIDATION_SELECTION_INVALID, "selection 与 dev 现场重算不一致")
    bank_validation = validate_bank(config.bank_root, config=config, verify_sources=True)
    bank_manifest, units, embeddings, dense_ids = load_runtime_bank_payload(config.bank_root)
    units_by_id = {unit["unit_id"]: unit for unit in units if unit.get("indexed")}
    if manifest.get("bank_id") != bank_validation["bank_id"] or manifest.get("selection_id") != selection["selection_id"]:
        raise ContractError(ReasonCode.BENCHMARK_IDENTITY_MISMATCH, "retrieval manifest 上游身份不一致")
    if len(queries) != len(selection["records"]) * 2 or len(packets) != len(selection["records"]):
        raise ContractError(ReasonCode.BENCHMARK_IDENTITY_MISMATCH, "query/packet 数量不匹配")
    expected_queries: list[dict[str, Any]] = []
    for selected in selection["records"]:
        shared = {
            "record_id": selected["record_id"],
            "available_modalities": selected["available_modalities"],
            "observation_sha256": selected["observation_sha256"],
            "bank_id": bank_manifest["bank_id"],
        }
        expected_queries.extend((
            make_query_row(
                **shared,
                intent=QueryIntent.INTERPRETATION,
                text=build_interpretation_query(question=config.dev.question, observation=selected["observation"], available_modalities=selected["available_modalities"]),
                knowledge_types=(KnowledgeType.INTERPRETATION,),
            ),
            make_query_row(
                **shared,
                intent=QueryIntent.COUNTER_LIMITATION,
                text=build_counter_limitation_query(question=config.dev.question, observation=selected["observation"], available_modalities=selected["available_modalities"]),
                knowledge_types=(KnowledgeType.CONFOUNDER, KnowledgeType.LIMITATION),
            ),
        ))
    if queries != expected_queries or any(row.get("query_id") != query_identity(row) for row in queries):
        raise ContractError(ReasonCode.BENCHMARK_IDENTITY_MISMATCH, "query 无法确定性重算")
    dense_index = read_json(root / "query_dense_index.json")
    with (root / "query_embeddings.npy").open("rb") as handle:
        query_embeddings = np.load(handle, allow_pickle=False)
    if (
        query_embeddings.dtype != np.float32
        or query_embeddings.shape != (len(queries), dense_index.get("embedding_dimension"))
        or dense_index.get("query_ids") != [row["query_id"] for row in queries]
        or not np.isfinite(query_embeddings).all()
        or not np.allclose(np.linalg.norm(query_embeddings, axis=1), 1.0, atol=1e-4)
    ):
        raise ContractError(ReasonCode.BENCHMARK_IDENTITY_MISMATCH, "query dense matrix/order 非法")
    retriever = HybridRetriever(
        sqlite_path=config.bank_root / "lexical.sqlite3",
        units=units,
        embeddings=embeddings,
        dense_unit_ids=dense_ids,
        config=config.retrieval,
    )
    try:
        for offset, (selected, packet) in enumerate(zip(selection["records"], packets, strict=True)):
            if packet.get("packet_id") != packet_identity(packet) or packet.get("record_id") != selected["record_id"]:
                raise ContractError(ReasonCode.BENCHMARK_IDENTITY_MISMATCH, "packet identity/record 不一致")
            expected_packet = build_balanced_packet(
                record_id=selected["record_id"],
                bank_id=bank_manifest["bank_id"],
                interpretation_query_id=queries[offset * 2]["query_id"],
                counter_query_id=queries[offset * 2 + 1]["query_id"],
                interpretation_candidates=retriever.retrieve(queries[offset * 2], query_embeddings[offset * 2]),
                counter_candidates=retriever.retrieve(queries[offset * 2 + 1], query_embeddings[offset * 2 + 1]),
                config=config.retrieval,
            )
            if packet != expected_packet:
                raise ContractError(ReasonCode.BENCHMARK_IDENTITY_MISMATCH, "packet/rank 无法重算")
            pages: set[tuple[str, int]] = set()
            for item in packet["items"]:
                unit = units_by_id.get(item.get("evidence_id"))
                if unit is None or item["unit_id"] != unit["unit_id"] or item["content"] != unit["content"]:
                    raise ContractError(ReasonCode.EVIDENCE_REFERENCE_INVALID, "packet 引用未知或漂移 Evidence Unit")
                if "general" not in item["applicable_modalities"] and not set(item["applicable_modalities"]) & set(selected["available_modalities"]):
                    raise ContractError(ReasonCode.EVIDENCE_REFERENCE_INVALID, "packet evidence modality 不适用")
                page_key = (item["source_id"], item["pdf_page"])
                if page_key in pages:
                    raise ContractError(ReasonCode.EVIDENCE_REFERENCE_INVALID, "packet 同 source/page 重复")
                pages.add(page_key)
            counts = quota_counts(packet)
            quotas = packet["quotas"]
            if any(counts[key] > quotas[key] for key in quotas):
                raise ContractError(ReasonCode.EVIDENCE_REFERENCE_INVALID, "packet 超过 knowledge-type quota")
    finally:
        retriever.close()
    payload_hashes = {relative: sha256_file(root / relative) for relative in sorted(_RETRIEVAL_PAYLOAD_FILES)}
    retrieval_id = sha256_text(canonical_json({
        "config_semantic_sha256": config.semantic_sha256,
        "bank_id": bank_manifest["bank_id"],
        "selection_id": selection["selection_id"],
        "payload_hashes": payload_hashes,
    }))
    if (
        manifest.get("retrieval_id") != retrieval_id
        or manifest.get("payload_hashes") != payload_hashes
        or manifest.get("ledger_sha256") != ledger_sha
        or report.get("query_reproducible") is not True
        or report.get("rank_reproducible") is not True
        or report.get("retrieval_gold_available") is not False
        or manifest.get("sealed_test_accessed") is not False
    ):
        raise ContractError(ReasonCode.BENCHMARK_IDENTITY_MISMATCH, "retrieval manifest/report identity 非法")
    return {
        "ok": True,
        "root": str(root),
        "retrieval_id": retrieval_id,
        "manifest_sha256": sha256_file(root / "manifest.json"),
        "ledger_sha256": ledger_sha,
        "selection_id": selection["selection_id"],
        "record_count": len(selection["records"]),
        "query_count": len(queries),
        "packet_count": len(packets),
        "sealed_test_accessed": False,
    }


def generate_paired(
    config_path: Path | str,
    *,
    limit: int,
    output_root: Path | None = None,
) -> dict[str, Any]:
    """在同一冻结 selection 上运行 no-RAG/text-RAG；任何非法输出均不发布。"""

    config = load_stage6_config(config_path)
    root = config.generation_root if output_root is None else Path(os.path.abspath(output_root))
    if root == config.generation_root and limit != config.dev.smoke_limit:
        raise ContractError(ReasonCode.TYPE_MISMATCH, "正式 paired root 必须使用配置冻结的 smoke_limit")
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1 or limit > config.dev.smoke_limit:
        raise ContractError(ReasonCode.TYPE_MISMATCH, "paired limit 超出冻结范围")
    if root.exists() or root.is_symlink():
        raise ContractError(ReasonCode.OUTPUT_EXISTS, f"paired output root 已存在：{root}")
    validate_retrieval(config.retrieval_root, config=config)
    selection = read_json(config.retrieval_root / "selection.json")
    packets = read_jsonl(config.retrieval_root / "packets.jsonl")
    retrieval_manifest = read_json(config.retrieval_root / "manifest.json")
    selected = selection["records"][:limit]
    if len({row["source"] for row in selected}) != limit:
        raise SelectionError(ReasonCode.VALIDATION_SELECTION_INVALID, "GPU smoke selection 未保持 source balance")
    _generate_selected_pairs(
        config=config,
        selected=selected,
        selection_id=selection["selection_id"],
        retrieval_id=retrieval_manifest["retrieval_id"],
        packets=packets,
        root=root,
        run_schema=PASS2_RUN_SCHEMA,
        report_schema="oa_groundrag.text_rag.pass2_report.v1",
        run_context=None,
        minimum_total_memory_bytes=0,
        minimum_free_memory_bytes=0,
    )
    return validate_run(root, config=config)


def _generate_selected_pairs(
    *,
    config: Stage6Config,
    selected: Sequence[Mapping[str, Any]],
    selection_id: str,
    retrieval_id: str,
    packets: Sequence[Mapping[str, Any]],
    root: Path,
    run_schema: str,
    report_schema: str,
    run_context: Mapping[str, Any] | None,
    minimum_total_memory_bytes: int,
    minimum_free_memory_bytes: int,
) -> None:
    """对显式冻结记录生成 paired 输出；调用方负责选择合同与最终 validator。"""

    import torch
    from transformers import LogitsProcessorList

    if not selected:
        raise SelectionError(ReasonCode.VALIDATION_SELECTION_INVALID, "paired selected records 为空")
    record_ids = [row.get("record_id") for row in selected]
    if not all(isinstance(value, str) and value for value in record_ids) or len(record_ids) != len(set(record_ids)):
        raise SelectionError(ReasonCode.VALIDATION_SELECTION_INVALID, "paired selected record_id 非法或重复")
    if root.exists() or root.is_symlink():
        raise ContractError(ReasonCode.OUTPUT_EXISTS, f"paired output root 已存在：{root}")
    if not torch.cuda.is_available():
        raise ModelError(ReasonCode.CUDA_REQUIRED, "Stage 6 paired generation 需要 CUDA")
    device = torch.device("cuda")
    free_memory, total_memory = torch.cuda.mem_get_info(device)
    if total_memory < minimum_total_memory_bytes or free_memory < minimum_free_memory_bytes:
        raise ModelError(
            ReasonCode.CUDA_REQUIRED,
            "Gate D GPU 显存不满足冻结 preflight",
            details={
                "free_memory_bytes": int(free_memory),
                "total_memory_bytes": int(total_memory),
                "minimum_free_memory_bytes": minimum_free_memory_bytes,
                "minimum_total_memory_bytes": minimum_total_memory_bytes,
            },
        )
    torch.cuda.empty_cache()
    generator = load_stage5_best_generator(config.stage5, device=device)
    stage5 = generator.config
    processor = generator.processor
    model = generator.model
    generator_identity = generator.identity
    model.eval()
    predictions: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    started = time.perf_counter()
    packet_by_record = {packet["record_id"]: packet for packet in packets}
    missing_packets = sorted(set(record_ids) - set(packet_by_record))
    if missing_packets:
        raise SelectionError(
            ReasonCode.VALIDATION_SELECTION_INVALID,
            "paired selected record 缺少 packet",
            details={"record_ids": missing_packets},
        )
    try:
        for selected_row in selected:
            packet = packet_by_record[selected_row["record_id"]]
            no_messages = build_pass2_messages(
                question=config.dev.question,
                target_status=selected_row["target_status"],
                program_facts=selected_row["program_facts"],
                observation=selected_row["observation"],
                packet=None,
            )
            rag_messages = build_pass2_messages(
                question=config.dev.question,
                target_status=selected_row["target_status"],
                program_facts=selected_row["program_facts"],
                observation=selected_row["observation"],
                packet=packet,
            )
            fairness_sha = validate_prompt_fairness(no_messages, rag_messages)
            for mode, messages, active_packet in (
                (RagMode.NO_RAG, no_messages, None),
                (RagMode.TEXT_RAG, rag_messages, packet),
            ):
                generation_started = time.perf_counter()
                generated: str | None = None
                encoded = None
                batch = None
                constraint = None
                continuation = None
                parsed = None
                try:
                    encoded = processor.encode_text_inference(messages)
                    batch = {key: value.unsqueeze(0).to(device) for key, value in encoded.tensors.items()}
                    constraint = build_pass2_logits_processor(
                        tokenizer=processor.processor.tokenizer,
                        prompt_length=encoded.input_token_count,
                        mode=mode,
                        packet=active_packet,
                    )
                    torch.cuda.reset_peak_memory_stats(device)
                    continuation = model.generate_text(
                        batch,
                        processor=processor.processor,
                        max_new_tokens=stage5.generation.max_new_tokens,
                        do_sample=False,
                        temperature=1.0,
                        top_p=1.0,
                        logits_processor=LogitsProcessorList([constraint]),
                    )[0]
                    generated = PASS2_ASSISTANT_PREFILL + continuation
                    parsed = parse_pass2_output(
                        generated,
                        mode=mode,
                        packet={"items": []} if active_packet is None else active_packet,
                    )
                    duration = time.perf_counter() - generation_started
                    row = {
                        "schema_version": PASS2_PREDICTION_SCHEMA,
                        "prediction_id": "",
                        "record_id": selected_row["record_id"],
                        "source": selected_row["source"],
                        "mode": mode.value,
                        "packet_id": None if active_packet is None else active_packet["packet_id"],
                        "selection_id": selection_id,
                        "retrieval_id": retrieval_id,
                        "observation_sha256": selected_row["observation_sha256"],
                        "program_facts_sha256": selected_row["program_facts_sha256"],
                        "prompt_sha256": prompt_sha256(messages),
                        "fair_prompt_body_sha256": fairness_sha,
                        "generator_identity_sha256": sha256_text(canonical_json(generator_identity)),
                        "decoding": {
                            "batch_size": 1,
                            "do_sample": False,
                            "max_new_tokens": stage5.generation.max_new_tokens,
                            "constraint_schema": PASS2_CONSTRAINT_SCHEMA,
                            "constraint_identity_sha256": sha256_text(canonical_json(constraint.identity)),
                        },
                        "input_token_count": encoded.input_token_count,
                        "duration_seconds": round(duration, 6),
                        "peak_vram_bytes": int(torch.cuda.max_memory_allocated(device)),
                        "generated_text": generated,
                        "output": parsed.to_dict(),
                    }
                    row["prediction_id"] = sha256_text(canonical_json({key: value for key, value in row.items() if key not in {"prediction_id", "duration_seconds", "peak_vram_bytes"}}))
                    predictions.append(row)
                except Exception as error:  # 输出无效时保留内存诊断，但不发布失败 root。
                    failures.append({
                        "schema_version": PASS2_FAILURE_SCHEMA,
                        "record_id": selected_row["record_id"],
                        "source": selected_row["source"],
                        "mode": mode.value,
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "generated_preview": None if generated is None else generated[:2000],
                    })
                finally:
                    encoded = None
                    batch = None
                    constraint = None
                    continuation = None
                    parsed = None
            torch.cuda.empty_cache()
        if failures or len(predictions) != len(selected) * 2:
            raise PredictionError(
                ReasonCode.PREDICTION_INVALID,
                "paired generation 未达到全量 strict contract，未发布输出",
                details={"prediction_count": len(predictions), "failure_count": len(failures), "failures": failures[:10]},
            )
        elapsed = time.perf_counter() - started
        field_utilization = {
            field: sum(bool(prediction["output"][field]) for prediction in predictions if prediction["mode"] == RagMode.TEXT_RAG.value)
            for field in ("supporting_interpretations", "alternative_explanations", "limitations", "recommended_verification")
        }
        report = {
            "schema_version": "oa_groundrag.text_rag.pass2_report.v1",
            "record_count": len(selected),
            "prediction_count": len(predictions),
            "failure_count": 0,
            "schema_valid_count": len(predictions),
            "evidence_id_valid_count": len(predictions),
            "citation_valid_count": len(predictions),
            "forbidden_claim_count": 0,
            "evidence_binding_rate": 1.0,
            "text_rag_field_utilization": field_utilization,
            "duration_seconds": round(elapsed, 6),
            "peak_vram_bytes": max(row["peak_vram_bytes"] for row in predictions),
            "generator_identity": generator_identity,
            "reference_authority": "automatic_contract_only",
            "expert_reference_available": False,
            "formal_acceptance": False,
            "scientific_acceptance": False,
            "sealed_test_accessed": False,
        }
        report["schema_version"] = report_schema
        if run_context is not None:
            report["run_context"] = dict(run_context)
        with AtomicArtifactDirectory(root) as writer:
            writer.write_jsonl("predictions.jsonl", predictions)
            writer.write_jsonl("failures.jsonl", [])
            writer.write_json("report.json", report)
            environment = {
                **_environment(),
                "cuda_device": torch.cuda.get_device_name(device),
                "cuda_total_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
            }
            if run_context is not None:
                environment["cuda_free_memory_bytes_preflight"] = int(free_memory)
            writer.write_json("environment.json", environment)
            assert writer.staging is not None
            payload_hashes = {relative: sha256_file(writer.path(relative)) for relative in sorted(_RUN_PAYLOAD_FILES)}
            run_identity = {
                "config_semantic_sha256": config.semantic_sha256,
                "selection_id": selection_id,
                "retrieval_id": retrieval_id,
                "generator_identity_sha256": sha256_text(canonical_json(generator_identity)),
                "payload_hashes": payload_hashes,
            }
            if run_context is not None:
                run_identity["run_context"] = dict(run_context)
            run_id = sha256_text(canonical_json(run_identity))
            writer.write_jsonl("SHA256SUMS.jsonl", _ledger_rows(writer.staging, _RUN_PAYLOAD_FILES))
            run_manifest = {
                "schema_version": run_schema,
                "run_id": run_id,
                "config_semantic_sha256": config.semantic_sha256,
                "selection_id": selection_id,
                "retrieval_id": retrieval_id,
                "generator_identity_sha256": sha256_text(canonical_json(generator_identity)),
                "record_count": len(selected),
                "prediction_count": len(predictions),
                "failure_count": 0,
                "payload_hashes": payload_hashes,
                "ledger_sha256": sha256_file(writer.path("SHA256SUMS.jsonl")),
                "development_only": True,
                "formal_acceptance": False,
                "scientific_acceptance": False,
                "sealed_test_accessed": False,
            }
            if run_context is not None:
                run_manifest["run_context"] = dict(run_context)
            writer.write_json("manifest.json", run_manifest)
            writer.publish()
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


def _validate_selected_run(
    root: Path | str,
    *,
    config: Stage6Config,
    selected: Sequence[Mapping[str, Any]],
    expected_selection_id: str,
    expected_retrieval_id: str,
    expected_run_schema: str,
    expected_report_schema: str,
    expected_run_context: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """从持久化文件重算显式 selected-record paired run。"""

    root = Path(root)
    _validate_artifact_files(root, expected=_RUN_ALL_FILES)
    ledger_sha = _validate_ledger(root, _RUN_PAYLOAD_FILES)
    manifest = read_json(root / "manifest.json")
    report = read_json(root / "report.json")
    predictions = read_jsonl(root / "predictions.jsonl")
    failures = read_jsonl(root / "failures.jsonl")
    validate_retrieval(config.retrieval_root, config=config)
    retrieval_manifest = read_json(config.retrieval_root / "manifest.json")
    packets = {row["record_id"]: row for row in read_jsonl(config.retrieval_root / "packets.jsonl")}
    stage5_config = load_stage5_config(config.stage5.config_path)
    selected_by_id = {row["record_id"]: row for row in selected}
    if len(selected_by_id) != len(selected):
        raise ContractError(ReasonCode.VALIDATION_SELECTION_INVALID, "paired selected records 重复")
    if failures or manifest.get("failure_count") != 0 or len(predictions) != len(selected) * 2:
        raise ContractError(ReasonCode.PREDICTION_INVALID, "paired run 含失败或数量不匹配")
    pair_messages: dict[str, dict[str, Sequence[Mapping[str, Any]]]] = {}
    seen: set[tuple[str, str]] = set()
    generator_hashes: set[str] = set()
    for prediction in predictions:
        record_id = prediction.get("record_id")
        mode = prediction.get("mode")
        key = (record_id, mode)
        if record_id not in selected_by_id or mode not in {value.value for value in RagMode} or key in seen:
            raise ContractError(ReasonCode.PREDICTION_INVALID, "paired prediction record/mode 非法或重复")
        seen.add(key)
        selected_row = selected_by_id[record_id]
        if prediction.get("source") != selected_row.get("source"):
            raise ContractError(ReasonCode.PREDICTION_IDENTITY_MISMATCH, "paired prediction source 漂移")
        active_packet = None if mode == RagMode.NO_RAG.value else packets[record_id]
        expected_constraint = pass2_constraint_identity(mode=RagMode(mode), packet=active_packet)
        expected_decoding = {
            "batch_size": 1,
            "do_sample": False,
            "max_new_tokens": stage5_config.generation.max_new_tokens,
            "constraint_schema": PASS2_CONSTRAINT_SCHEMA,
            "constraint_identity_sha256": sha256_text(canonical_json(expected_constraint)),
        }
        messages = build_pass2_messages(
            question=config.dev.question,
            target_status=selected_row["target_status"],
            program_facts=selected_row["program_facts"],
            observation=selected_row["observation"],
            packet=active_packet,
        )
        if (
            prediction.get("prompt_sha256") != prompt_sha256(messages)
            or prediction.get("selection_id") != expected_selection_id
            or prediction.get("retrieval_id") != expected_retrieval_id
            or prediction.get("observation_sha256") != selected_row["observation_sha256"]
            or prediction.get("program_facts_sha256") != selected_row["program_facts_sha256"]
            or prediction.get("packet_id") != (None if active_packet is None else active_packet["packet_id"])
            or prediction.get("decoding") != expected_decoding
        ):
            raise ContractError(ReasonCode.PREDICTION_IDENTITY_MISMATCH, "paired prediction 上游/prompt 身份漂移")
        parsed = parse_pass2_output(
            prediction.get("generated_text"),
            mode=mode,
            packet={"items": []} if active_packet is None else active_packet,
        )
        if prediction.get("output") != parsed.to_dict():
            raise ContractError(ReasonCode.PREDICTION_IDENTITY_MISMATCH, "paired parsed output 漂移")
        identity_payload = {
            key: value for key, value in prediction.items()
            if key not in {"prediction_id", "duration_seconds", "peak_vram_bytes"}
        }
        if prediction.get("prediction_id") != sha256_text(canonical_json(identity_payload)):
            raise ContractError(ReasonCode.PREDICTION_IDENTITY_MISMATCH, "prediction ID 无法重算")
        pair_messages.setdefault(record_id, {})[mode] = messages
        generator_hashes.add(str(prediction.get("generator_identity_sha256")))
    expected_seen = {(row["record_id"], mode.value) for row in selected for mode in RagMode}
    if seen != expected_seen or len(generator_hashes) != 1:
        raise ContractError(ReasonCode.PREDICTION_IDENTITY_MISMATCH, "paired mode/generator 不公平")
    for record_id, values in pair_messages.items():
        fairness = validate_prompt_fairness(values[RagMode.NO_RAG.value], values[RagMode.TEXT_RAG.value])
        if any(row["fair_prompt_body_sha256"] != fairness for row in predictions if row["record_id"] == record_id):
            raise ContractError(ReasonCode.PREDICTION_IDENTITY_MISMATCH, "paired prompt fairness SHA 漂移")
    generator_hash = next(iter(generator_hashes))
    payload_hashes = {relative: sha256_file(root / relative) for relative in sorted(_RUN_PAYLOAD_FILES)}
    run_identity = {
        "config_semantic_sha256": config.semantic_sha256,
        "selection_id": expected_selection_id,
        "retrieval_id": expected_retrieval_id,
        "generator_identity_sha256": generator_hash,
        "payload_hashes": payload_hashes,
    }
    if expected_run_context is not None:
        run_identity["run_context"] = dict(expected_run_context)
    run_id = sha256_text(canonical_json(run_identity))
    expected_context = None if expected_run_context is None else dict(expected_run_context)
    if (
        manifest.get("schema_version") != expected_run_schema
        or manifest.get("run_id") != run_id
        or manifest.get("config_semantic_sha256") != config.semantic_sha256
        or manifest.get("selection_id") != expected_selection_id
        or manifest.get("retrieval_id") != expected_retrieval_id
        or manifest.get("generator_identity_sha256") != generator_hash
        or manifest.get("record_count") != len(selected)
        or manifest.get("prediction_count") != len(predictions)
        or manifest.get("payload_hashes") != payload_hashes
        or manifest.get("ledger_sha256") != ledger_sha
        or manifest.get("sealed_test_accessed") is not False
        or report.get("schema_version") != expected_report_schema
        or report.get("record_count") != len(selected)
        or report.get("prediction_count") != len(predictions)
        or report.get("schema_valid_count") != len(predictions)
        or report.get("evidence_id_valid_count") != len(predictions)
        or report.get("citation_valid_count") != len(predictions)
        or report.get("forbidden_claim_count") != 0
        or report.get("evidence_binding_rate") != 1.0
        or report.get("generator_identity") is None
        or sha256_text(canonical_json(report["generator_identity"])) != generator_hash
        or report.get("reference_authority") != "automatic_contract_only"
        or report.get("expert_reference_available") is not False
        or report.get("formal_acceptance") is not False
        or report.get("scientific_acceptance") is not False
        or report.get("sealed_test_accessed") is not False
        or retrieval_manifest.get("retrieval_id") != expected_retrieval_id
    ):
        raise ContractError(ReasonCode.PREDICTION_IDENTITY_MISMATCH, "paired manifest/report identity 非法")
    if expected_context is None:
        if "run_context" in manifest or "run_context" in report:
            raise ContractError(ReasonCode.UNKNOWN_FIELD, "v1 paired run 不得含 run_context")
    elif manifest.get("run_context") != expected_context or report.get("run_context") != expected_context:
        raise ContractError(ReasonCode.PREDICTION_IDENTITY_MISMATCH, "paired run_context 漂移")
    return {
        "ok": True,
        "root": str(root),
        "run_id": run_id,
        "manifest_sha256": sha256_file(root / "manifest.json"),
        "ledger_sha256": ledger_sha,
        "record_count": len(selected),
        "prediction_count": len(predictions),
        "failure_count": 0,
        "sealed_test_accessed": False,
    }


def validate_run(root: Path | str, *, config: Stage6Config) -> dict[str, Any]:
    manifest = read_json(Path(root) / "manifest.json")
    selection = read_json(config.retrieval_root / "selection.json")
    retrieval_manifest = read_json(config.retrieval_root / "manifest.json")
    selected = selection["records"][: manifest.get("record_count", 0)]
    return _validate_selected_run(
        root,
        config=config,
        selected=selected,
        expected_selection_id=selection["selection_id"],
        expected_retrieval_id=retrieval_manifest["retrieval_id"],
        expected_run_schema=PASS2_RUN_SCHEMA,
        expected_report_schema="oa_groundrag.text_rag.pass2_report.v1",
        expected_run_context=None,
    )
