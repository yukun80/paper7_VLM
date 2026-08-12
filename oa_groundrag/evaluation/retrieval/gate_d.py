"""Gate D automatic-only 开发协议、显式 paired run 与可重算描述性评价。

本模块不产生专家标签、retrieval Gold 或 Gate D 通过结论。当前五条工程 smoke
只用于冻结排除集；所有评价记录在看到新输出前由确定性规则选定。
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import os
from pathlib import Path
import re
import statistics
from typing import Any, Mapping, Sequence

from oa_groundrag.artifacts.identity import (
    canonical_json,
    sha256_file,
    sha256_text,
)
from oa_groundrag.artifacts.io import first_symlink_component
from oa_groundrag.data.rs_general.io import (
    read_json,
    read_jsonl,
    require_exact_keys,
    require_int,
    require_mapping,
    require_string,
    resolve_config_path,
)
from oa_groundrag.artifacts.directory import AtomicArtifactDirectory
from oa_groundrag.vlm.config import _load_yaml
from oa_groundrag.vlm.errors import (
    ConfigError,
    ContractError,
    ReasonCode,
    SelectionError,
)
from oa_groundrag.vlm.processing import Qwen3VLProcessorAdapter
from oa_groundrag.training.grounding.config import load_stage5_config

from oa_groundrag.retrieval.bank import validate_bank
from oa_groundrag.retrieval.contracts import RagMode, Stage6Config, load_stage6_config
from oa_groundrag.retrieval.pass2 import build_pass2_messages, prompt_sha256, validate_prompt_fairness
from oa_groundrag.retrieval.workflow import (
    _environment,
    _generate_selected_pairs,
    _ledger_rows,
    _reject_sealed_path,
    _validate_artifact_files,
    _validate_ledger,
    _validate_selected_run,
    validate_retrieval,
    validate_run,
)


GATE_D_CONFIG_SCHEMA = "oa_groundrag.text_rag.gate_d_dev_config.v1"
GATE_D_PROTOCOL_SCHEMA = "oa_groundrag.text_rag.gate_d_dev_protocol.v1"
GATE_D_PROMPT_AUDIT_SCHEMA = "oa_groundrag.text_rag.gate_d_prompt_audit.v1"
GATE_D_PROTOCOL_MANIFEST_SCHEMA = "oa_groundrag.text_rag.gate_d_protocol_manifest.v1"
GATE_D_RUN_SCHEMA = "oa_groundrag.text_rag.gate_d_dev_run.v1"
GATE_D_RUN_REPORT_SCHEMA = "oa_groundrag.text_rag.gate_d_run_report.v1"
GATE_D_RUN_CONTEXT_SCHEMA = "oa_groundrag.text_rag.gate_d_run_context.v1"
GATE_D_CASE_SCHEMA = "oa_groundrag.text_rag.gate_d_paired_case.v1"
GATE_D_AUTO_REPORT_SCHEMA = "oa_groundrag.text_rag.gate_d_auto_report.v1"
GATE_D_EVAL_MANIFEST_SCHEMA = "oa_groundrag.text_rag.gate_d_eval_manifest.v1"

_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_PROTOCOL_PAYLOAD_FILES = frozenset({"protocol.json", "prompt_token_audit.jsonl", "environment.json"})
_PROTOCOL_ALL_FILES = _PROTOCOL_PAYLOAD_FILES | {"SHA256SUMS.jsonl", "manifest.json"}
_EVALUATION_PAYLOAD_FILES = frozenset({"paired_cases.jsonl", "report.json", "environment.json"})
_EVALUATION_ALL_FILES = _EVALUATION_PAYLOAD_FILES | {"SHA256SUMS.jsonl", "manifest.json"}
_OUTPUT_FIELDS = (
    "supporting_interpretations",
    "alternative_explanations",
    "limitations",
    "recommended_verification",
    "summary",
)


def _sha(value: Any, *, location: str) -> str:
    text = require_string(value, location=location)
    if not _SHA_RE.fullmatch(text):
        raise ConfigError(ReasonCode.TYPE_MISMATCH, f"{location}: 必须是小写 SHA-256")
    return text


def _strict_string_list(value: Any, *, location: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ConfigError(ReasonCode.TYPE_MISMATCH, f"{location}: 必须是非空列表")
    result = tuple(require_string(item, location=f"{location}[]") for item in value)
    if len(result) != len(set(result)):
        raise ConfigError(ReasonCode.TYPE_MISMATCH, f"{location}: 不允许重复")
    return result


@dataclass(frozen=True)
class GateDDevConfig:
    config_path: Path
    stage6_config_path: Path
    expected_stage6_semantic_sha256: str
    expected_selection_id: str
    expected_retrieval_id: str
    expected_retrieval_manifest_sha256: str
    smoke_run_root: Path
    expected_smoke_run_id: str
    expected_smoke_manifest_sha256: str
    source_order: tuple[str, ...]
    per_source: int
    protocol_root: Path
    run_root: Path
    evaluation_root: Path
    minimum_total_memory_bytes: int
    minimum_free_memory_bytes: int
    reference_authority: str
    semantic_sha256: str

    def semantic_dict(self) -> dict[str, Any]:
        return {
            "schema_version": GATE_D_CONFIG_SCHEMA,
            "stage6": {
                "config": str(self.stage6_config_path),
                "expected_semantic_sha256": self.expected_stage6_semantic_sha256,
            },
            "upstream": {
                "selection_id": self.expected_selection_id,
                "retrieval_id": self.expected_retrieval_id,
                "retrieval_manifest_sha256": self.expected_retrieval_manifest_sha256,
            },
            "smoke_run": {
                "root": str(self.smoke_run_root),
                "run_id": self.expected_smoke_run_id,
                "manifest_sha256": self.expected_smoke_manifest_sha256,
            },
            "selection": {
                "source_order": list(self.source_order),
                "per_source": self.per_source,
            },
            "outputs": {
                "protocol_root": str(self.protocol_root),
                "run_root": str(self.run_root),
                "evaluation_root": str(self.evaluation_root),
            },
            "gpu": {
                "minimum_total_memory_bytes": self.minimum_total_memory_bytes,
                "minimum_free_memory_bytes": self.minimum_free_memory_bytes,
            },
            "reference_authority": self.reference_authority,
        }


def load_gate_d_config(path: Path | str) -> GateDDevConfig:
    config_path = Path(os.path.abspath(Path(path)))
    if (
        first_symlink_component(config_path) is not None
        or not config_path.is_file()
        or config_path.is_symlink()
        or config_path.stat().st_nlink != 1
    ):
        raise ConfigError(ReasonCode.OUTPUT_LINK, f"Gate D 配置必须是普通单链接文件：{config_path}")
    row = _load_yaml(config_path)
    require_exact_keys(
        row,
        required=(
            "schema_version", "stage6", "upstream", "smoke_run", "selection",
            "outputs", "gpu", "reference_authority",
        ),
        location="$",
    )
    if row["schema_version"] != GATE_D_CONFIG_SCHEMA:
        raise ConfigError(ReasonCode.INVALID_ENUM, f"仅支持 {GATE_D_CONFIG_SCHEMA}")
    base = config_path.parent
    stage6_row = require_mapping(row["stage6"], location="$.stage6")
    require_exact_keys(stage6_row, required=("config", "expected_semantic_sha256"), location="$.stage6")
    upstream = require_mapping(row["upstream"], location="$.upstream")
    require_exact_keys(
        upstream,
        required=("selection_id", "retrieval_id", "retrieval_manifest_sha256"),
        location="$.upstream",
    )
    smoke = require_mapping(row["smoke_run"], location="$.smoke_run")
    require_exact_keys(smoke, required=("root", "run_id", "manifest_sha256"), location="$.smoke_run")
    selection = require_mapping(row["selection"], location="$.selection")
    require_exact_keys(selection, required=("source_order", "per_source"), location="$.selection")
    source_order = _strict_string_list(selection["source_order"], location="$.selection.source_order")
    if len(source_order) != 5:
        raise ConfigError(ReasonCode.VALIDATION_SELECTION_INVALID, "Gate D source_order 必须恰好五个来源")
    outputs = require_mapping(row["outputs"], location="$.outputs")
    require_exact_keys(
        outputs,
        required=("protocol_root", "run_root", "evaluation_root"),
        location="$.outputs",
    )
    gpu = require_mapping(row["gpu"], location="$.gpu")
    require_exact_keys(
        gpu,
        required=("minimum_total_memory_bytes", "minimum_free_memory_bytes"),
        location="$.gpu",
    )
    reference_authority = require_string(row["reference_authority"], location="$.reference_authority")
    if reference_authority != "automatic_contract_only":
        raise ConfigError(ReasonCode.INVALID_ENUM, "Gate D v1 只允许 automatic_contract_only")
    provisional = GateDDevConfig(
        config_path=config_path,
        stage6_config_path=resolve_config_path(base, stage6_row["config"], location="$.stage6.config"),
        expected_stage6_semantic_sha256=_sha(
            stage6_row["expected_semantic_sha256"], location="$.stage6.expected_semantic_sha256"
        ),
        expected_selection_id=_sha(upstream["selection_id"], location="$.upstream.selection_id"),
        expected_retrieval_id=_sha(upstream["retrieval_id"], location="$.upstream.retrieval_id"),
        expected_retrieval_manifest_sha256=_sha(
            upstream["retrieval_manifest_sha256"], location="$.upstream.retrieval_manifest_sha256"
        ),
        smoke_run_root=resolve_config_path(base, smoke["root"], location="$.smoke_run.root"),
        expected_smoke_run_id=_sha(smoke["run_id"], location="$.smoke_run.run_id"),
        expected_smoke_manifest_sha256=_sha(
            smoke["manifest_sha256"], location="$.smoke_run.manifest_sha256"
        ),
        source_order=source_order,
        per_source=require_int(selection["per_source"], location="$.selection.per_source", minimum=1),
        protocol_root=resolve_config_path(base, outputs["protocol_root"], location="$.outputs.protocol_root"),
        run_root=resolve_config_path(base, outputs["run_root"], location="$.outputs.run_root"),
        evaluation_root=resolve_config_path(base, outputs["evaluation_root"], location="$.outputs.evaluation_root"),
        minimum_total_memory_bytes=require_int(
            gpu["minimum_total_memory_bytes"], location="$.gpu.minimum_total_memory_bytes", minimum=1
        ),
        minimum_free_memory_bytes=require_int(
            gpu["minimum_free_memory_bytes"], location="$.gpu.minimum_free_memory_bytes", minimum=1
        ),
        reference_authority=reference_authority,
        semantic_sha256="",
    )
    roots = (provisional.protocol_root, provisional.run_root, provisional.evaluation_root)
    if len(set(roots)) != len(roots):
        raise ConfigError(ReasonCode.TYPE_MISMATCH, "Gate D 三个输出根必须不同")
    for label, root in zip(("protocol", "run", "evaluation"), roots, strict=True):
        _reject_sealed_path(root, label=f"Gate D {label} root")
    semantic = sha256_text(canonical_json(provisional.semantic_dict()))
    return GateDDevConfig(**{**provisional.__dict__, "semantic_sha256": semantic})


def select_gate_d_records(
    records: Sequence[Mapping[str, Any]],
    *,
    smoke_record_ids: Sequence[str],
    source_order: Sequence[str],
    per_source: int,
) -> list[dict[str, Any]]:
    """排除已调试 smoke，并按来源内原顺序取样后 round-robin。"""

    if len({row.get("record_id") for row in records}) != len(records):
        raise SelectionError(ReasonCode.VALIDATION_SELECTION_INVALID, "上游 selection record_id 重复")
    record_by_id = {str(row["record_id"]): row for row in records}
    smoke_ids = tuple(smoke_record_ids)
    if len(smoke_ids) != len(source_order) or len(smoke_ids) != len(set(smoke_ids)):
        raise SelectionError(ReasonCode.VALIDATION_SELECTION_INVALID, "工程 smoke 必须为五来源各一条")
    if not set(smoke_ids) <= set(record_by_id):
        raise SelectionError(ReasonCode.VALIDATION_SELECTION_INVALID, "工程 smoke 不属于冻结 selection")
    smoke_sources = [record_by_id[record_id].get("source") for record_id in smoke_ids]
    if set(smoke_sources) != set(source_order) or len(smoke_sources) != len(set(smoke_sources)):
        raise SelectionError(ReasonCode.VALIDATION_SELECTION_INVALID, "工程 smoke 来源不平衡")
    buckets: dict[str, list[dict[str, Any]]] = {source: [] for source in source_order}
    smoke_set = set(smoke_ids)
    for index, row in enumerate(records):
        source = row.get("source")
        if source not in buckets or row.get("record_id") in smoke_set:
            continue
        if row.get("split") != "val" or row.get("target_status") != "target_present":
            raise SelectionError(ReasonCode.SPLIT_FORBIDDEN, "Gate D 只允许 val target-present records")
        child = dict(row)
        child["selection_index"] = index
        buckets[str(source)].append(child)
    insufficient = {source: len(values) for source, values in buckets.items() if len(values) < per_source}
    if insufficient:
        raise SelectionError(
            ReasonCode.VALIDATION_SELECTION_INVALID,
            "Gate D 来源剩余记录不足",
            details={"source_counts": insufficient},
        )
    selected: list[dict[str, Any]] = []
    for source_slot in range(per_source):
        for source in source_order:
            child = dict(buckets[source][source_slot])
            child["source_slot"] = source_slot
            child["evaluation_ordinal"] = len(selected)
            selected.append(child)
    if len(selected) != len(source_order) * per_source or set(smoke_ids) & {
        row["record_id"] for row in selected
    }:
        raise SelectionError(ReasonCode.VALIDATION_SELECTION_INVALID, "Gate D selection 数量或排除集非法")
    return selected


def _make_processor(stage6: Stage6Config) -> tuple[Any, Qwen3VLProcessorAdapter]:
    stage5 = load_stage5_config(stage6.stage5.config_path)
    processor = Qwen3VLProcessorAdapter(
        processor_path=stage5.model.processor_path,
        local_files_only=stage5.model.local_files_only,
        trust_remote_code=stage5.model.trust_remote_code,
        min_pixels=stage5.limits.min_pixels,
        max_pixels=stage5.limits.max_pixels,
        max_images=stage5.limits.max_images,
        max_input_tokens=stage5.limits.max_input_tokens,
    )
    return stage5, processor


def audit_selected_prompts(
    stage6: Stage6Config,
    selected: Sequence[Mapping[str, Any]],
    packets: Mapping[str, Mapping[str, Any]],
    *,
    processor: Any | None = None,
    max_input_tokens: int | None = None,
) -> tuple[list[dict[str, Any]], str]:
    """使用真实或测试 processor 重算 2×N 个 text-only prompt。"""

    if processor is None:
        stage5, processor = _make_processor(stage6)
        limit = stage5.limits.max_input_tokens
    else:
        if max_input_tokens is None:
            raise ContractError(ReasonCode.TYPE_MISMATCH, "测试 processor 必须显式提供 max_input_tokens")
        limit = max_input_tokens
    processor_identity = processor.identity()
    rows: list[dict[str, Any]] = []
    for selected_row in selected:
        record_id = selected_row["record_id"]
        if record_id not in packets:
            raise SelectionError(ReasonCode.VALIDATION_SELECTION_INVALID, f"{record_id}: 缺少 packet")
        packet = packets[record_id]
        no_messages = build_pass2_messages(
            question=stage6.dev.question,
            target_status=selected_row["target_status"],
            program_facts=selected_row["program_facts"],
            observation=selected_row["observation"],
            packet=None,
        )
        rag_messages = build_pass2_messages(
            question=stage6.dev.question,
            target_status=selected_row["target_status"],
            program_facts=selected_row["program_facts"],
            observation=selected_row["observation"],
            packet=packet,
        )
        fairness_sha = validate_prompt_fairness(no_messages, rag_messages)
        for mode, messages, packet_id in (
            (RagMode.NO_RAG, no_messages, None),
            (RagMode.TEXT_RAG, rag_messages, packet["packet_id"]),
        ):
            encoded = processor.encode_text_inference(messages)
            if encoded.image_count != 0:
                raise ContractError(ReasonCode.ASSET_ROLE_LEAKAGE, f"{record_id}/{mode.value}: Pass-2 含图")
            if encoded.input_token_count > limit:
                raise ContractError(
                    ReasonCode.TOKEN_LIMIT_EXCEEDED,
                    f"{record_id}/{mode.value}: prompt 超过 {limit} tokens",
                )
            rows.append({
                "schema_version": GATE_D_PROMPT_AUDIT_SCHEMA,
                "record_id": record_id,
                "source": selected_row["source"],
                "mode": mode.value,
                "packet_id": packet_id,
                "prompt_sha256": prompt_sha256(messages),
                "fair_prompt_body_sha256": fairness_sha,
                "input_token_count": encoded.input_token_count,
                "image_count": encoded.image_count,
                "max_input_tokens": limit,
            })
    return rows, sha256_text(canonical_json(processor_identity))


def _validate_upstream(config: GateDDevConfig) -> tuple[Stage6Config, dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    stage6 = load_stage6_config(config.stage6_config_path)
    if stage6.semantic_sha256 != config.expected_stage6_semantic_sha256:
        raise ContractError(ReasonCode.BENCHMARK_IDENTITY_MISMATCH, "Stage 6 config semantic SHA 漂移")
    validate_bank(stage6.bank_root, config=stage6, verify_sources=True)
    retrieval_result = validate_retrieval(stage6.retrieval_root, config=stage6)
    if (
        retrieval_result["retrieval_id"] != config.expected_retrieval_id
        or retrieval_result["selection_id"] != config.expected_selection_id
        or retrieval_result["manifest_sha256"] != config.expected_retrieval_manifest_sha256
    ):
        raise ContractError(ReasonCode.BENCHMARK_IDENTITY_MISMATCH, "Gate D retrieval/selection 身份漂移")
    smoke_result = validate_run(config.smoke_run_root, config=stage6)
    if (
        smoke_result["run_id"] != config.expected_smoke_run_id
        or smoke_result["manifest_sha256"] != config.expected_smoke_manifest_sha256
        or smoke_result["record_count"] != len(config.source_order)
    ):
        raise ContractError(ReasonCode.PREDICTION_IDENTITY_MISMATCH, "Gate D 工程 smoke 身份漂移")
    selection = read_json(stage6.retrieval_root / "selection.json")
    records = selection.get("records")
    if selection.get("selection_id") != config.expected_selection_id or not isinstance(records, list):
        raise ContractError(ReasonCode.VALIDATION_SELECTION_INVALID, "Gate D 上游 selection 非法")
    packets = read_jsonl(stage6.retrieval_root / "packets.jsonl")
    smoke_predictions = read_jsonl(config.smoke_run_root / "predictions.jsonl")
    return stage6, selection, packets, smoke_predictions


def _build_protocol(config: GateDDevConfig) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    stage6, selection, packets, smoke_predictions = _validate_upstream(config)
    smoke_modes: dict[str, set[str]] = defaultdict(set)
    for prediction in smoke_predictions:
        smoke_modes[str(prediction.get("record_id"))].add(str(prediction.get("mode")))
    if any(modes != {mode.value for mode in RagMode} for modes in smoke_modes.values()):
        raise ContractError(ReasonCode.PREDICTION_INVALID, "工程 smoke 不含完整 paired modes")
    selection_order = {row["record_id"]: index for index, row in enumerate(selection["records"])}
    smoke_ids = sorted(smoke_modes, key=lambda record_id: selection_order.get(record_id, 10**9))
    selected = select_gate_d_records(
        selection["records"],
        smoke_record_ids=smoke_ids,
        source_order=config.source_order,
        per_source=config.per_source,
    )
    packet_by_record = {row["record_id"]: row for row in packets}
    audit, processor_identity_sha256 = audit_selected_prompts(stage6, selected, packet_by_record)
    selected_bindings = [{
        "evaluation_ordinal": row["evaluation_ordinal"],
        "source_slot": row["source_slot"],
        "selection_index": row["selection_index"],
        "record_id": row["record_id"],
        "source": row["source"],
        "selection_record_sha256": sha256_text(canonical_json({
            key: value for key, value in row.items()
            if key not in {"evaluation_ordinal", "source_slot", "selection_index"}
        })),
        "observation_sha256": row["observation_sha256"],
        "program_facts_sha256": row["program_facts_sha256"],
        "packet_id": packet_by_record[row["record_id"]]["packet_id"],
    } for row in selected]
    ordered_record_ids = [row["record_id"] for row in selected_bindings]
    protocol = {
        "schema_version": GATE_D_PROTOCOL_SCHEMA,
        "protocol_id": "",
        "gate_d_config_semantic_sha256": config.semantic_sha256,
        "stage6_config_semantic_sha256": stage6.semantic_sha256,
        "selection_id": selection["selection_id"],
        "retrieval_id": config.expected_retrieval_id,
        "retrieval_manifest_sha256": config.expected_retrieval_manifest_sha256,
        "smoke_run_id": config.expected_smoke_run_id,
        "smoke_manifest_sha256": config.expected_smoke_manifest_sha256,
        "smoke_record_ids": smoke_ids,
        "selection_algorithm": "exclude_validated_smoke_then_source_order_first_n_round_robin.v1",
        "source_order": list(config.source_order),
        "per_source": config.per_source,
        "record_count": len(selected_bindings),
        "source_counts": dict(Counter(row["source"] for row in selected_bindings)),
        "ordered_record_ids_sha256": sha256_text(canonical_json(ordered_record_ids)),
        "processor_identity_sha256": processor_identity_sha256,
        "prompt_count": len(audit),
        "prompt_token_min": min(row["input_token_count"] for row in audit),
        "prompt_token_max": max(row["input_token_count"] for row in audit),
        "prompt_over_limit_count": 0,
        "records": selected_bindings,
        "reference_authority": config.reference_authority,
        "expert_reference_available": False,
        "retrieval_gold_available": False,
        "development_only": True,
        "formal_acceptance": False,
        "scientific_acceptance": False,
        "sealed_test_accessed": False,
    }
    protocol["protocol_id"] = sha256_text(canonical_json({
        key: value for key, value in protocol.items() if key != "protocol_id"
    }))
    return protocol, audit, _environment()


def prepare_gate_d_protocol(config_path: Path | str) -> dict[str, Any]:
    config = load_gate_d_config(config_path)
    if config.protocol_root.exists() or config.protocol_root.is_symlink():
        raise ContractError(ReasonCode.OUTPUT_EXISTS, f"Gate D protocol root 已存在：{config.protocol_root}")
    protocol, audit, environment = _build_protocol(config)
    with AtomicArtifactDirectory(config.protocol_root) as writer:
        writer.write_json("protocol.json", protocol)
        writer.write_jsonl("prompt_token_audit.jsonl", audit)
        writer.write_json("environment.json", environment)
        assert writer.staging is not None
        payload_hashes = {
            relative: sha256_file(writer.path(relative)) for relative in sorted(_PROTOCOL_PAYLOAD_FILES)
        }
        writer.write_jsonl("SHA256SUMS.jsonl", _ledger_rows(writer.staging, _PROTOCOL_PAYLOAD_FILES))
        writer.write_json("manifest.json", {
            "schema_version": GATE_D_PROTOCOL_MANIFEST_SCHEMA,
            "protocol_id": protocol["protocol_id"],
            "gate_d_config_semantic_sha256": config.semantic_sha256,
            "stage6_config_semantic_sha256": protocol["stage6_config_semantic_sha256"],
            "record_count": protocol["record_count"],
            "prompt_count": protocol["prompt_count"],
            "payload_hashes": payload_hashes,
            "ledger_sha256": sha256_file(writer.path("SHA256SUMS.jsonl")),
            "reference_authority": config.reference_authority,
            "formal_acceptance": False,
            "scientific_acceptance": False,
            "sealed_test_accessed": False,
        })
        writer.publish()
    return validate_gate_d_protocol(config.config_path)


def validate_gate_d_protocol(config_path: Path | str) -> dict[str, Any]:
    config = load_gate_d_config(config_path)
    root = config.protocol_root
    _validate_artifact_files(root, expected=_PROTOCOL_ALL_FILES)
    ledger_sha = _validate_ledger(root, _PROTOCOL_PAYLOAD_FILES)
    observed_protocol = read_json(root / "protocol.json")
    observed_audit = read_jsonl(root / "prompt_token_audit.jsonl")
    observed_environment = read_json(root / "environment.json")
    expected_protocol, expected_audit, expected_environment = _build_protocol(config)
    if (
        observed_protocol != expected_protocol
        or observed_audit != expected_audit
        or observed_environment != expected_environment
    ):
        raise ContractError(ReasonCode.BENCHMARK_IDENTITY_MISMATCH, "Gate D protocol 无法确定性重算")
    payload_hashes = {relative: sha256_file(root / relative) for relative in sorted(_PROTOCOL_PAYLOAD_FILES)}
    manifest = read_json(root / "manifest.json")
    expected_manifest = {
        "schema_version": GATE_D_PROTOCOL_MANIFEST_SCHEMA,
        "protocol_id": expected_protocol["protocol_id"],
        "gate_d_config_semantic_sha256": config.semantic_sha256,
        "stage6_config_semantic_sha256": expected_protocol["stage6_config_semantic_sha256"],
        "record_count": expected_protocol["record_count"],
        "prompt_count": expected_protocol["prompt_count"],
        "payload_hashes": payload_hashes,
        "ledger_sha256": ledger_sha,
        "reference_authority": config.reference_authority,
        "formal_acceptance": False,
        "scientific_acceptance": False,
        "sealed_test_accessed": False,
    }
    if manifest != expected_manifest:
        raise ContractError(ReasonCode.BENCHMARK_IDENTITY_MISMATCH, "Gate D protocol manifest 漂移")
    return {
        "ok": True,
        "root": str(root),
        "protocol_id": expected_protocol["protocol_id"],
        "manifest_sha256": sha256_file(root / "manifest.json"),
        "ledger_sha256": ledger_sha,
        "record_count": expected_protocol["record_count"],
        "prompt_count": expected_protocol["prompt_count"],
        "prompt_token_min": expected_protocol["prompt_token_min"],
        "prompt_token_max": expected_protocol["prompt_token_max"],
        "sealed_test_accessed": False,
    }


def _selected_from_protocol(stage6: Stage6Config, protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    selection = read_json(stage6.retrieval_root / "selection.json")
    by_id = {row["record_id"]: row for row in selection["records"]}
    output: list[dict[str, Any]] = []
    for binding in protocol["records"]:
        record_id = binding["record_id"]
        row = by_id.get(record_id)
        if row is None:
            raise SelectionError(ReasonCode.VALIDATION_SELECTION_INVALID, f"Gate D record 不在 selection：{record_id}")
        if (
            sha256_text(canonical_json(row)) != binding["selection_record_sha256"]
            or row["observation_sha256"] != binding["observation_sha256"]
            or row["program_facts_sha256"] != binding["program_facts_sha256"]
        ):
            raise ContractError(ReasonCode.BENCHMARK_IDENTITY_MISMATCH, f"{record_id}: selection binding 漂移")
        output.append(dict(row))
    return output


def _run_context(config: GateDDevConfig, protocol: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": GATE_D_RUN_CONTEXT_SCHEMA,
        "gate_d_config_semantic_sha256": config.semantic_sha256,
        "protocol_id": protocol["protocol_id"],
        "protocol_manifest_sha256": sha256_file(config.protocol_root / "manifest.json"),
        "ordered_record_ids_sha256": protocol["ordered_record_ids_sha256"],
        "selection_authority": "frozen_gate_d_dev_protocol.v1",
    }


def generate_gate_d_pairs(config_path: Path | str) -> dict[str, Any]:
    config = load_gate_d_config(config_path)
    validate_gate_d_protocol(config.config_path)
    stage6 = load_stage6_config(config.stage6_config_path)
    protocol = read_json(config.protocol_root / "protocol.json")
    selected = _selected_from_protocol(stage6, protocol)
    packets = read_jsonl(stage6.retrieval_root / "packets.jsonl")
    _generate_selected_pairs(
        config=stage6,
        selected=selected,
        selection_id=config.expected_selection_id,
        retrieval_id=config.expected_retrieval_id,
        packets=packets,
        root=config.run_root,
        run_schema=GATE_D_RUN_SCHEMA,
        report_schema=GATE_D_RUN_REPORT_SCHEMA,
        run_context=_run_context(config, protocol),
        minimum_total_memory_bytes=config.minimum_total_memory_bytes,
        minimum_free_memory_bytes=config.minimum_free_memory_bytes,
    )
    return validate_gate_d_run(config.config_path)


def validate_gate_d_run(config_path: Path | str) -> dict[str, Any]:
    config = load_gate_d_config(config_path)
    validate_gate_d_protocol(config.config_path)
    stage6 = load_stage6_config(config.stage6_config_path)
    protocol = read_json(config.protocol_root / "protocol.json")
    selected = _selected_from_protocol(stage6, protocol)
    result = _validate_selected_run(
        config.run_root,
        config=stage6,
        selected=selected,
        expected_selection_id=config.expected_selection_id,
        expected_retrieval_id=config.expected_retrieval_id,
        expected_run_schema=GATE_D_RUN_SCHEMA,
        expected_report_schema=GATE_D_RUN_REPORT_SCHEMA,
        expected_run_context=_run_context(config, protocol),
    )
    expected_audit = {
        (row["record_id"], row["mode"]): row
        for row in read_jsonl(config.protocol_root / "prompt_token_audit.jsonl")
    }
    predictions = read_jsonl(config.run_root / "predictions.jsonl")
    for prediction in predictions:
        audit = expected_audit.get((prediction["record_id"], prediction["mode"]))
        if (
            audit is None
            or prediction["input_token_count"] != audit["input_token_count"]
            or prediction["prompt_sha256"] != audit["prompt_sha256"]
            or prediction["fair_prompt_body_sha256"] != audit["fair_prompt_body_sha256"]
            or prediction["packet_id"] != audit["packet_id"]
        ):
            raise ContractError(ReasonCode.PREDICTION_IDENTITY_MISMATCH, "Gate D prompt audit 与 run 漂移")
    return {**result, "protocol_id": protocol["protocol_id"]}


def _field_text(output: Mapping[str, Any], field: str) -> str:
    value = output[field]
    if field == "summary":
        return str(value["text"])
    return "\n".join(str(item["text"]) for item in value)


def _field_ids(output: Mapping[str, Any], field: str) -> list[str]:
    value = output[field]
    if field == "summary":
        return list(value["evidence_ids"])
    return [evidence_id for item in value for evidence_id in item["evidence_ids"]]


def build_automatic_evaluation(
    *,
    protocol: Mapping[str, Any],
    run_manifest: Mapping[str, Any],
    predictions: Sequence[Mapping[str, Any]],
    packets: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """构建不带质量胜负判断的 deterministic paired 描述性报告。"""

    by_pair: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for prediction in predictions:
        record_id = str(prediction.get("record_id"))
        mode = str(prediction.get("mode"))
        if mode in by_pair[record_id]:
            raise ContractError(ReasonCode.PREDICTION_INVALID, "Gate D evaluation mode 重复")
        by_pair[record_id][mode] = prediction
    cases: list[dict[str, Any]] = []
    field_nonempty = {mode.value: Counter() for mode in RagMode}
    field_changed = Counter()
    character_counts = {mode.value: [] for mode in RagMode}
    character_deltas: list[int] = []
    any_changed = 0
    limitation_preserved = 0
    citation_type_counts: Counter[str] = Counter()
    citation_reference_count = 0
    unique_evidence: set[str] = set()
    for binding in protocol["records"]:
        record_id = binding["record_id"]
        values = by_pair.get(record_id, {})
        if set(values) != {mode.value for mode in RagMode}:
            raise ContractError(ReasonCode.PREDICTION_INVALID, f"{record_id}: Gate D pair 不完整")
        no_row = values[RagMode.NO_RAG.value]
        rag_row = values[RagMode.TEXT_RAG.value]
        if no_row["generator_identity_sha256"] != rag_row["generator_identity_sha256"]:
            raise ContractError(ReasonCode.PREDICTION_IDENTITY_MISMATCH, f"{record_id}: generator 不公平")
        packet = packets.get(record_id)
        if packet is None or rag_row["packet_id"] != packet["packet_id"]:
            raise ContractError(ReasonCode.EVIDENCE_REFERENCE_INVALID, f"{record_id}: packet 漂移")
        packet_by_evidence = {item["evidence_id"]: item for item in packet["items"]}
        comparisons: dict[str, Any] = {}
        cited_ids: list[str] = []
        pair_changed = False
        for field in _OUTPUT_FIELDS:
            no_text = _field_text(no_row["output"], field)
            rag_text = _field_text(rag_row["output"], field)
            no_ids = _field_ids(no_row["output"], field)
            rag_ids = _field_ids(rag_row["output"], field)
            if no_ids:
                raise ContractError(ReasonCode.EVIDENCE_REFERENCE_INVALID, f"{record_id}: no_rag 含 citation")
            unknown = sorted(set(rag_ids) - set(packet_by_evidence))
            if unknown:
                raise ContractError(ReasonCode.EVIDENCE_REFERENCE_INVALID, f"{record_id}: unknown citation {unknown}")
            cited_ids.extend(rag_ids)
            no_nonempty = bool(no_text.strip())
            rag_nonempty = bool(rag_text.strip())
            field_nonempty[RagMode.NO_RAG.value][field] += int(no_nonempty)
            field_nonempty[RagMode.TEXT_RAG.value][field] += int(rag_nonempty)
            changed = no_text != rag_text
            field_changed[field] += int(changed)
            pair_changed = pair_changed or changed
            comparisons[field] = {
                "no_rag_nonempty": no_nonempty,
                "text_rag_nonempty": rag_nonempty,
                "text_changed": changed,
                "no_rag_characters": len(no_text),
                "text_rag_characters": len(rag_text),
                "character_delta": len(rag_text) - len(no_text),
            }
        any_changed += int(pair_changed)
        limitation_preserved += int(
            bool(_field_text(no_row["output"], "limitations").strip())
            and bool(_field_text(rag_row["output"], "limitations").strip())
        )
        no_total = sum(len(_field_text(no_row["output"], field)) for field in _OUTPUT_FIELDS)
        rag_total = sum(len(_field_text(rag_row["output"], field)) for field in _OUTPUT_FIELDS)
        character_counts[RagMode.NO_RAG.value].append(no_total)
        character_counts[RagMode.TEXT_RAG.value].append(rag_total)
        character_deltas.append(rag_total - no_total)
        trace = []
        for evidence_id in dict.fromkeys(cited_ids):
            item = packet_by_evidence[evidence_id]
            unique_evidence.add(evidence_id)
            trace.append({
                "evidence_id": evidence_id,
                "knowledge_type": item["knowledge_type"],
                "source_id": item["source_id"],
                "pdf_page": item["pdf_page"],
                "section": item["section"],
            })
        for evidence_id in cited_ids:
            citation_type_counts[packet_by_evidence[evidence_id]["knowledge_type"]] += 1
        citation_reference_count += len(cited_ids)
        case = {
            "schema_version": GATE_D_CASE_SCHEMA,
            "case_id": "",
            "protocol_id": protocol["protocol_id"],
            "run_id": run_manifest["run_id"],
            "evaluation_ordinal": binding["evaluation_ordinal"],
            "record_id": record_id,
            "source": binding["source"],
            "observation_sha256": binding["observation_sha256"],
            "program_facts_sha256": binding["program_facts_sha256"],
            "packet_id": packet["packet_id"],
            "no_rag_prediction_id": no_row["prediction_id"],
            "text_rag_prediction_id": rag_row["prediction_id"],
            "no_rag_output": no_row["output"],
            "text_rag_output": rag_row["output"],
            "field_comparison": comparisons,
            "cited_evidence_trace": trace,
        }
        case["case_id"] = sha256_text(canonical_json({
            key: value for key, value in case.items() if key != "case_id"
        }))
        cases.append(case)
    record_count = len(cases)

    def stats(values: Sequence[int]) -> dict[str, Any]:
        return {
            "total": sum(values),
            "mean": round(sum(values) / len(values), 6),
            "median": float(statistics.median(values)),
            "minimum": min(values),
            "maximum": max(values),
        }

    report = {
        "schema_version": GATE_D_AUTO_REPORT_SCHEMA,
        "protocol_id": protocol["protocol_id"],
        "run_id": run_manifest["run_id"],
        "record_count": record_count,
        "prediction_count": len(predictions),
        "pair_complete_count": record_count,
        "source_counts": dict(Counter(case["source"] for case in cases)),
        "schema_valid_count": len(predictions),
        "prompt_fairness_valid_count": record_count,
        "no_rag_empty_citation_count": record_count,
        "text_rag_citation_valid_count": record_count,
        "forbidden_claim_count": 0,
        "candidate_upgrade_count": 0,
        "field_nonempty_counts": {
            mode: {field: counts[field] for field in _OUTPUT_FIELDS}
            for mode, counts in field_nonempty.items()
        },
        "field_text_changed_counts": {field: field_changed[field] for field in _OUTPUT_FIELDS},
        "pair_any_text_changed_count": any_changed,
        "character_counts": {
            mode: stats(values) for mode, values in character_counts.items()
        },
        "character_delta": stats(character_deltas),
        "limitation_preserved_pair_count": limitation_preserved,
        "citation_reference_count": citation_reference_count,
        "traceable_citation_count": citation_reference_count,
        "unique_cited_evidence_count": len(unique_evidence),
        "citation_knowledge_type_counts": dict(sorted(citation_type_counts.items())),
        "unsupported_claim_rate": None,
        "expert_relevance": None,
        "recall_at_k": None,
        "mrr": None,
        "ndcg": None,
        "gate_d_pass": None,
        "reference_authority": "automatic_contract_only",
        "expert_reference_available": False,
        "retrieval_gold_available": False,
        "formal_acceptance": False,
        "scientific_acceptance": False,
        "sealed_test_accessed": False,
    }
    return cases, report


def _expected_evaluation(config: GateDDevConfig) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    validate_gate_d_run(config.config_path)
    protocol = read_json(config.protocol_root / "protocol.json")
    run_manifest = read_json(config.run_root / "manifest.json")
    predictions = read_jsonl(config.run_root / "predictions.jsonl")
    stage6 = load_stage6_config(config.stage6_config_path)
    packets = {row["record_id"]: row for row in read_jsonl(stage6.retrieval_root / "packets.jsonl")}
    return build_automatic_evaluation(
        protocol=protocol,
        run_manifest=run_manifest,
        predictions=predictions,
        packets=packets,
    )


def evaluate_gate_d(config_path: Path | str) -> dict[str, Any]:
    config = load_gate_d_config(config_path)
    if config.evaluation_root.exists() or config.evaluation_root.is_symlink():
        raise ContractError(ReasonCode.OUTPUT_EXISTS, f"Gate D evaluation root 已存在：{config.evaluation_root}")
    cases, report = _expected_evaluation(config)
    with AtomicArtifactDirectory(config.evaluation_root) as writer:
        writer.write_jsonl("paired_cases.jsonl", cases)
        writer.write_json("report.json", report)
        writer.write_json("environment.json", _environment())
        assert writer.staging is not None
        payload_hashes = {
            relative: sha256_file(writer.path(relative)) for relative in sorted(_EVALUATION_PAYLOAD_FILES)
        }
        evaluation_id = sha256_text(canonical_json({
            "gate_d_config_semantic_sha256": config.semantic_sha256,
            "protocol_id": report["protocol_id"],
            "run_id": report["run_id"],
            "payload_hashes": payload_hashes,
        }))
        writer.write_jsonl("SHA256SUMS.jsonl", _ledger_rows(writer.staging, _EVALUATION_PAYLOAD_FILES))
        writer.write_json("manifest.json", {
            "schema_version": GATE_D_EVAL_MANIFEST_SCHEMA,
            "evaluation_id": evaluation_id,
            "gate_d_config_semantic_sha256": config.semantic_sha256,
            "protocol_id": report["protocol_id"],
            "run_id": report["run_id"],
            "record_count": report["record_count"],
            "payload_hashes": payload_hashes,
            "ledger_sha256": sha256_file(writer.path("SHA256SUMS.jsonl")),
            "reference_authority": "automatic_contract_only",
            "formal_acceptance": False,
            "scientific_acceptance": False,
            "sealed_test_accessed": False,
        })
        writer.publish()
    return validate_gate_d_evaluation(config.config_path)


def validate_gate_d_evaluation(config_path: Path | str) -> dict[str, Any]:
    config = load_gate_d_config(config_path)
    root = config.evaluation_root
    _validate_artifact_files(root, expected=_EVALUATION_ALL_FILES)
    ledger_sha = _validate_ledger(root, _EVALUATION_PAYLOAD_FILES)
    expected_cases, expected_report = _expected_evaluation(config)
    if (
        read_jsonl(root / "paired_cases.jsonl") != expected_cases
        or read_json(root / "report.json") != expected_report
        or read_json(root / "environment.json") != _environment()
    ):
        raise ContractError(ReasonCode.BENCHMARK_IDENTITY_MISMATCH, "Gate D automatic evaluation 无法重算")
    payload_hashes = {relative: sha256_file(root / relative) for relative in sorted(_EVALUATION_PAYLOAD_FILES)}
    evaluation_id = sha256_text(canonical_json({
        "gate_d_config_semantic_sha256": config.semantic_sha256,
        "protocol_id": expected_report["protocol_id"],
        "run_id": expected_report["run_id"],
        "payload_hashes": payload_hashes,
    }))
    manifest = read_json(root / "manifest.json")
    expected_manifest = {
        "schema_version": GATE_D_EVAL_MANIFEST_SCHEMA,
        "evaluation_id": evaluation_id,
        "gate_d_config_semantic_sha256": config.semantic_sha256,
        "protocol_id": expected_report["protocol_id"],
        "run_id": expected_report["run_id"],
        "record_count": expected_report["record_count"],
        "payload_hashes": payload_hashes,
        "ledger_sha256": ledger_sha,
        "reference_authority": "automatic_contract_only",
        "formal_acceptance": False,
        "scientific_acceptance": False,
        "sealed_test_accessed": False,
    }
    if manifest != expected_manifest:
        raise ContractError(ReasonCode.BENCHMARK_IDENTITY_MISMATCH, "Gate D evaluation manifest 漂移")
    return {
        "ok": True,
        "root": str(root),
        "evaluation_id": evaluation_id,
        "manifest_sha256": sha256_file(root / "manifest.json"),
        "ledger_sha256": ledger_sha,
        "record_count": expected_report["record_count"],
        "gate_d_pass": None,
        "scientific_acceptance": False,
        "sealed_test_accessed": False,
    }


def validate_gate_d(config_path: Path | str) -> dict[str, Any]:
    config = load_gate_d_config(config_path)
    protocol = validate_gate_d_protocol(config.config_path)
    run = validate_gate_d_run(config.config_path)
    evaluation = validate_gate_d_evaluation(config.config_path)
    return {
        "ok": True,
        "protocol": protocol,
        "run": run,
        "evaluation": evaluation,
        "gate_d_pass": None,
        "formal_acceptance": False,
        "scientific_acceptance": False,
        "sealed_test_accessed": False,
    }
