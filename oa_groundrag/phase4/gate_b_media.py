"""由 Gate B prediction 的持久化 provenance 定位 canonical 图片。"""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from oa_groundrag.phase3.common import (
    first_symlink_component,
    read_json,
    read_jsonl_indices,
)
from oa_groundrag.phase3.contracts import (
    CANONICAL_SCHEMA_VERSION,
    MANIFEST_VERSION,
    RS_GENERALDESC_SOURCES,
    RS_GENERALDESC_TASK_FAMILIES,
)
from oa_groundrag.phase3.dataset import (
    CanonicalRecordLocation,
    RSGeneralDescDataset,
)
from oa_groundrag.phase3.errors import RSGeneralDescError

from .contracts import PREDICTION_SCHEMA_VERSION
from .errors import PredictionError, ReasonCode
from .gate_b_contracts import (
    GATE_B_PROTOCOL_ID,
    QWEN_TEMPLATE_VERSION,
)


DEFAULT_GATE_B_BENCHMARK_ROOT = (
    Path(__file__).resolve().parents[2].parent
    / "benchmark"
    / "rs_generaldesc_v1"
)

_PREDICTION_FIELDS = frozenset(
    {
        "schema_version",
        "record_id",
        "parent_id",
        "logical_role",
        "task_family",
        "mask_mode",
        "generated_text",
        "model_output",
        "reference_responses",
        "evidence_ids",
        "provenance",
        "counterfactual",
    }
)
_PROVENANCE_FIELDS = frozenset(
    {
        "canonical_build_id",
        "canonical_payload_sha256",
        "renderer",
        "gate_b",
    }
)
_GATE_B_FIELDS = frozenset(
    {
        "protocol_id",
        "protocol_sha256",
        "selection_sha256",
        "ordinal",
        "model_role",
        "source",
        "shard_path",
        "line_index",
        "template_version",
    }
)


@dataclass(frozen=True)
class GateBMediaPath:
    """一张仍由 canonical Benchmark 持久化保存的图片。"""

    role: str
    path: Path


@dataclass(frozen=True)
class _PredictionLocation:
    record_id: str
    parent_id: str
    logical_role: str
    task_family: str
    reference_responses: tuple[str, ...]
    build_id: str
    payload_sha256: str
    source: str
    shard_path: str
    line_index: int


def _prediction_error(
    message: str,
    *,
    code: ReasonCode = ReasonCode.PREDICTION_INVALID,
    **details: object,
) -> PredictionError:
    return PredictionError(code, message, details=details)


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _ordinary_prediction_file(path: Path | str) -> Path:
    absolute = Path(os.path.abspath(path))
    linked = first_symlink_component(absolute)
    if linked is not None:
        raise _prediction_error(
            "predictions 路径含 symlink 组件",
            code=ReasonCode.OUTPUT_LINK,
            path=str(absolute),
            linked_component=str(linked),
        )
    try:
        file_stat = absolute.stat(follow_symlinks=False)
    except (FileNotFoundError, NotADirectoryError) as error:
        raise _prediction_error(
            "predictions 文件不存在",
            code=ReasonCode.ASSET_MISSING,
            path=str(absolute),
        ) from error
    if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_nlink != 1:
        raise _prediction_error(
            "predictions 必须是普通单链接文件",
            code=ReasonCode.OUTPUT_LINK,
            path=str(absolute),
        )
    return absolute


def _prediction_row(path: Path, *, line_number: int) -> dict[str, Any]:
    if (
        isinstance(line_number, bool)
        or not isinstance(line_number, int)
        or line_number <= 0
    ):
        raise _prediction_error(
            "--line-number 必须是从 1 开始的正整数",
            code=ReasonCode.TYPE_MISMATCH,
            line_number=line_number,
        )
    index = line_number - 1
    try:
        return read_jsonl_indices(path, (index,))[index]
    except RSGeneralDescError as error:
        raise _prediction_error(
            "无法读取指定 prediction 行",
            path=str(path),
            line_number=line_number,
            source_reason_code=error.code.value,
            source_error=str(error),
        ) from error


def _nonempty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _parse_prediction(
    row: Mapping[str, Any],
    *,
    line_number: int,
) -> _PredictionLocation:
    if set(row) != _PREDICTION_FIELDS:
        raise _prediction_error(
            "指定行不是严格 Gate B prediction",
            line_number=line_number,
            missing=sorted(_PREDICTION_FIELDS - set(row)),
            unknown=sorted(set(row) - _PREDICTION_FIELDS),
        )
    references = row.get("reference_responses")
    task_family = row.get("task_family")
    task_values = {value.value for value in RS_GENERALDESC_TASK_FAMILIES}
    if (
        row.get("schema_version") != PREDICTION_SCHEMA_VERSION
        or not _nonempty_string(row.get("record_id"))
        or not _nonempty_string(row.get("parent_id"))
        or row.get("logical_role") != "external_val"
        or not isinstance(task_family, str)
        or task_family not in task_values
        or row.get("mask_mode") != "external_generic"
        or not _nonempty_string(row.get("generated_text"))
        or row.get("model_output") is not None
        or not isinstance(references, list)
        or not references
        or any(not _nonempty_string(value) for value in references)
        or len(references) != len(set(references))
        or row.get("evidence_ids") != []
        or row.get("counterfactual") is not None
    ):
        raise _prediction_error(
            "指定行不满足 Gate B prediction 合同",
            line_number=line_number,
        )

    provenance = row.get("provenance")
    if not isinstance(provenance, dict) or set(provenance) != _PROVENANCE_FIELDS:
        raise _prediction_error(
            "prediction provenance 不是严格 Gate B provenance",
            line_number=line_number,
        )
    build_id = provenance.get("canonical_build_id")
    payload_sha256 = provenance.get("canonical_payload_sha256")
    if (
        not isinstance(build_id, str)
        or not build_id.startswith("build_")
        or not _is_sha256(build_id.removeprefix("build_"))
        or not _is_sha256(payload_sha256)
        or provenance.get("renderer")
        != "phase3.render_canonical_messages"
    ):
        raise _prediction_error(
            "prediction canonical identity/renderer 非法",
            line_number=line_number,
        )

    gate = provenance.get("gate_b")
    if not isinstance(gate, dict) or set(gate) != _GATE_B_FIELDS:
        raise _prediction_error(
            "prediction 缺少严格 Gate B 定位信息",
            line_number=line_number,
        )
    line_index = gate.get("line_index")
    model_role = gate.get("model_role")
    ordinal = gate.get("ordinal")
    shard_path = gate.get("shard_path")
    source = gate.get("source")
    if (
        gate.get("protocol_id") != GATE_B_PROTOCOL_ID
        or not _is_sha256(gate.get("protocol_sha256"))
        or not _is_sha256(gate.get("selection_sha256"))
        or isinstance(ordinal, bool)
        or not isinstance(ordinal, int)
        or ordinal != line_number - 1
        or not isinstance(model_role, str)
        or model_role not in {"base", "adapter"}
        or not isinstance(source, str)
        or source not in RS_GENERALDESC_SOURCES
        or not _nonempty_string(shard_path)
        or isinstance(line_index, bool)
        or not isinstance(line_index, int)
        or line_index < 0
        or gate.get("template_version") != QWEN_TEMPLATE_VERSION
    ):
        raise _prediction_error(
            "prediction Gate B 定位字段非法或与行号不一致",
            line_number=line_number,
        )

    assert isinstance(references, list)
    assert isinstance(build_id, str)
    assert isinstance(payload_sha256, str)
    assert isinstance(source, str)
    assert isinstance(shard_path, str)
    assert isinstance(line_index, int)
    return _PredictionLocation(
        record_id=str(row["record_id"]),
        parent_id=str(row["parent_id"]),
        logical_role=str(row["logical_role"]),
        task_family=str(row["task_family"]),
        reference_responses=tuple(str(value) for value in references),
        build_id=build_id,
        payload_sha256=payload_sha256,
        source=source,
        shard_path=shard_path,
        line_index=line_index,
    )


def _benchmark_root_and_manifest(
    root: Path | str,
    *,
    expected_build_id: str,
    expected_payload_sha256: str,
) -> Path:
    absolute = Path(os.path.abspath(root))
    linked = first_symlink_component(absolute)
    if linked is not None:
        raise _prediction_error(
            "Benchmark root 含 symlink 组件",
            code=ReasonCode.OUTPUT_LINK,
            path=str(absolute),
            linked_component=str(linked),
        )
    try:
        root_stat = absolute.stat(follow_symlinks=False)
    except (FileNotFoundError, NotADirectoryError) as error:
        raise _prediction_error(
            "Benchmark root 不存在",
            code=ReasonCode.ASSET_MISSING,
            path=str(absolute),
        ) from error
    if not stat.S_ISDIR(root_stat.st_mode):
        raise _prediction_error(
            "Benchmark root 必须是普通目录",
            code=ReasonCode.OUTPUT_LINK,
            path=str(absolute),
        )

    manifest_path = absolute / "manifest.json"
    try:
        manifest_stat = manifest_path.stat(follow_symlinks=False)
    except (FileNotFoundError, NotADirectoryError) as error:
        raise _prediction_error(
            "Benchmark manifest 不存在",
            code=ReasonCode.ASSET_MISSING,
            path=str(manifest_path),
        ) from error
    if (
        not stat.S_ISREG(manifest_stat.st_mode)
        or manifest_path.is_symlink()
        or manifest_stat.st_nlink != 1
    ):
        raise _prediction_error(
            "Benchmark manifest 必须是普通单链接文件",
            code=ReasonCode.OUTPUT_LINK,
            path=str(manifest_path),
        )
    try:
        manifest = read_json(manifest_path)
    except RSGeneralDescError as error:
        raise _prediction_error(
            "Benchmark manifest 不是严格 JSON",
            code=ReasonCode.BENCHMARK_IDENTITY_MISMATCH,
            path=str(manifest_path),
            source_reason_code=error.code.value,
        ) from error
    if not isinstance(manifest, dict):
        raise _prediction_error(
            "Benchmark manifest 必须是对象",
            code=ReasonCode.BENCHMARK_IDENTITY_MISMATCH,
            path=str(manifest_path),
        )
    actual_identity = {
        "schema_version": manifest.get("schema_version"),
        "canonical_schema_version": manifest.get(
            "canonical_schema_version"
        ),
        "build_id": manifest.get("build_id"),
        "payload_root_sha256": manifest.get("payload_root_sha256"),
    }
    expected_identity = {
        "schema_version": MANIFEST_VERSION,
        "canonical_schema_version": CANONICAL_SCHEMA_VERSION,
        "build_id": expected_build_id,
        "payload_root_sha256": expected_payload_sha256,
    }
    mismatches = {
        key: {"expected": expected, "actual": actual_identity[key]}
        for key, expected in expected_identity.items()
        if actual_identity[key] != expected
    }
    if mismatches:
        raise _prediction_error(
            "prediction 与 Benchmark manifest identity 不一致",
            code=ReasonCode.BENCHMARK_IDENTITY_MISMATCH,
            mismatches=mismatches,
        )
    return absolute


def locate_gate_b_media(
    predictions: Path | str,
    *,
    line_number: int,
    benchmark_root: Path | str = DEFAULT_GATE_B_BENCHMARK_ROOT,
) -> tuple[GateBMediaPath, ...]:
    """返回指定 Gate B prediction 对应的持久化 canonical 图片路径。"""

    predictions_path = _ordinary_prediction_file(predictions)
    row = _prediction_row(predictions_path, line_number=line_number)
    location = _parse_prediction(row, line_number=line_number)
    root = _benchmark_root_and_manifest(
        benchmark_root,
        expected_build_id=location.build_id,
        expected_payload_sha256=location.payload_sha256,
    )
    dataset = RSGeneralDescDataset.from_locations(
        root,
        (
            CanonicalRecordLocation(
                location.shard_path,
                location.line_index,
            ),
        ),
        load_assets=False,
    )
    manifest = dataset.manifest
    if (
        manifest.get("build_id") != location.build_id
        or manifest.get("payload_root_sha256") != location.payload_sha256
    ):
        raise _prediction_error(
            "Benchmark manifest identity 在查询期间发生变化",
            code=ReasonCode.BENCHMARK_IDENTITY_MISMATCH,
        )
    if len(dataset.records) != 1:
        raise _prediction_error(
            "Gate B 定位未解析到唯一 canonical record",
            shard_path=location.shard_path,
            line_index=location.line_index,
        )
    record = dataset.records[0]
    canonical_identity = {
        "record_id": record.get("record_id"),
        "parent_id": record.get("parent_id"),
        "logical_role": record.get("logical_role"),
        "task_family": record.get("task_family"),
        "source": record.get("source"),
        "reference_responses": tuple(record.get("reference_responses", ())),
    }
    expected_identity = {
        "record_id": location.record_id,
        "parent_id": location.parent_id,
        "logical_role": location.logical_role,
        "task_family": location.task_family,
        "source": location.source,
        "reference_responses": location.reference_responses,
    }
    mismatches = {
        key: {"expected": expected, "actual": canonical_identity[key]}
        for key, expected in expected_identity.items()
        if canonical_identity[key] != expected
    }
    if mismatches:
        raise _prediction_error(
            "prediction 与定位到的 canonical record 不一致",
            shard_path=location.shard_path,
            line_index=location.line_index,
            mismatches=mismatches,
        )

    output: list[GateBMediaPath] = []
    for index, media in enumerate(record["media"]):
        role = str(media["role"])
        if any(character in role for character in "\t\r\n"):
            raise _prediction_error(
                "canonical media role 不能包含 TSV 控制字符",
                media_index=index,
                role=role,
            )
        verified = dataset.resolve_verified_asset_path(str(media["path"]))
        output.append(GateBMediaPath(role=role, path=verified))
    if not output:
        raise _prediction_error("canonical record 没有持久化图片")
    return tuple(output)
