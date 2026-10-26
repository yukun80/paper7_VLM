"""统一 Dataset 接入、bounded locator 与 mask-grounded 核心。"""

from __future__ import annotations

import json
import math
import copy
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
from torch.utils.data import Dataset

from oa_groundrag.artifacts.identity import (
    sha256_file,
    stable_hash,
)
from oa_groundrag.artifacts.io import first_symlink_component
from oa_groundrag.data.rs_general.io import (
    portable_relative_path,
    read_json,
    read_jsonl_indices,
    reject_nonfinite,
)
from oa_groundrag.data.rs_general.contracts import validate_canonical_record
from oa_groundrag.data.rs_general.dataset import (
    CanonicalRecordLocation,
    RSGeneralDescDataset,
)
from oa_groundrag.data.rs_general.exporter import render_canonical_messages

from .config import VLMConfig
from oa_groundrag.grounding.contracts import (
    SUPPORTED_AUXSEG_INFERENCE_SCHEMA,
    DataMode,
    MaskMode,
    RegionCandidate,
    RegionInventory,
)
from .errors import ContractError, PreflightError, ReasonCode
from .preflight import BenchmarkAccess, open_benchmark_access


REQUIRED_EXTERNAL_ROLES = frozenset({"external_train", "external_val"})
REQUIRED_EXTERNAL_SOURCES = frozenset({"rsgpt", "mmrs1m", "disasterm3"})
REQUIRED_EXTERNAL_TASKS = frozenset(
    {
        "global_caption",
        "visual_qa",
        "bbox_region_caption",
        "scene_understanding",
        "object_count",
        "spatial_relation",
        "visible_change_report",
    }
)
AUXSEG_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "checkpoint",
        "checkpoint_step",
        "benchmark_index_sha256",
        "benchmark_contract",
        "backbone_sha256",
        "backbone",
        "architecture",
        "split",
        "source",
        "sample_count",
        "modality_weight_order",
        "modality_weight_map_strides",
        "modality_weight_map_summary",
        "region_feature_dim",
    }
)
AUXSEG_PREDICTION_FIELDS = frozenset(
    {
        "sample_id",
        "source",
        "split",
        "available_modalities",
        "active_modalities",
        "no_target_score",
        "modality_names",
        "modality_weights",
        "array_keys",
        "regions",
    }
)
AUXSEG_ARRAY_KEY_FIELDS = frozenset(
    {
        "mask_probability",
        "global_mask",
        "region_masks",
        "region_features",
        "modality_weights",
        "modality_weight_maps",
    }
)


@dataclass(frozen=True)
class BoundedCanonicalSelection:
    locations: tuple[CanonicalRecordLocation, ...]
    records: tuple[Mapping[str, Any], ...]
    probed_shards: tuple[str, ...]
    probed_record_count: int
    selected_parent_count: int
    selected_asset_count: int
    selected_asset_bytes: int
    copied_asset_bytes: int
    role_counts: Mapping[str, int]
    source_counts: Mapping[str, int]
    task_counts: Mapping[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "probed_shards": list(self.probed_shards),
            "probed_shard_count": len(self.probed_shards),
            "probed_record_count": self.probed_record_count,
            "selected_record_count": len(self.records),
            "selected_parent_count": self.selected_parent_count,
            "selected_asset_count": self.selected_asset_count,
            "selected_asset_bytes": self.selected_asset_bytes,
            "copied_asset_bytes": self.copied_asset_bytes,
            "role_counts": dict(sorted(self.role_counts.items())),
            "source_counts": dict(sorted(self.source_counts.items())),
            "task_counts": dict(sorted(self.task_counts.items())),
            "record_ids": [str(row["record_id"]) for row in self.records],
            "locations": [
                {
                    "shard_path": location.shard_path,
                    "line_index": location.line_index,
                }
                for location in self.locations
            ],
        }


def _evenly_spaced(first: int, last: int, count: int) -> tuple[int, ...]:
    if last < first or count <= 0:
        return ()
    length = last - first + 1
    if length <= count:
        return tuple(range(first, last + 1))
    return tuple(
        sorted(
            {
                int(round(first + step * (length - 1) / (count - 1)))
                for step in range(count)
            }
        )
    )


def _probe_shard_indices(
    statistics: Mapping[str, Any],
    *,
    records_per_shard: int,
) -> tuple[int, ...]:
    source_role_counts = statistics.get("source_role_counts")
    if not isinstance(source_role_counts, dict):
        raise PreflightError(
            ReasonCode.BENCHMARK_IDENTITY_MISMATCH,
            "statistics 缺少 source_role_counts",
        )
    offset = 0
    selected: set[int] = set()
    for role in ("external_train", "external_val"):
        for source in ("disasterm3", "mmrs1m", "rsgpt"):
            source_counts = source_role_counts.get(source)
            if (
                not isinstance(source_counts, dict)
                or isinstance(source_counts.get(role), bool)
                or not isinstance(source_counts.get(role), int)
                or source_counts[role] <= 0
            ):
                raise PreflightError(
                    ReasonCode.BENCHMARK_IDENTITY_MISMATCH,
                    f"statistics 缺少 {role}/{source} count",
                )
            count = int(source_counts[role])
            first_shard = offset // records_per_shard
            last_shard = (offset + count - 1) // records_per_shard
            selected.update(_evenly_spaced(first_shard, last_shard, 5))
            offset += count
    if offset != statistics.get("record_count"):
        raise PreflightError(
            ReasonCode.BENCHMARK_IDENTITY_MISMATCH,
            "role/source segment count 与 record_count 不一致",
        )
    return tuple(sorted(selected))


def _asset_increment(
    record: Mapping[str, Any],
    *,
    seen: set[str],
) -> tuple[set[str], int]:
    added: set[str] = set()
    size = 0
    media_value = record.get("media")
    if not isinstance(media_value, list):
        raise PreflightError(
            ReasonCode.BENCHMARK_IDENTITY_MISMATCH,
            "canonical media 合同非法",
        )
    for media in media_value:
        if not isinstance(media, dict):
            raise PreflightError(
                ReasonCode.BENCHMARK_IDENTITY_MISMATCH,
                "canonical media item 合同非法",
            )
        asset_id = media.get("asset_id")
        size_bytes = media.get("size_bytes")
        if (
            not isinstance(asset_id, str)
            or isinstance(size_bytes, bool)
            or not isinstance(size_bytes, int)
            or size_bytes <= 0
        ):
            raise PreflightError(
                ReasonCode.BENCHMARK_IDENTITY_MISMATCH,
                "canonical asset identity/size 合同非法",
            )
        if asset_id not in seen and asset_id not in added:
            added.add(asset_id)
            size += size_bytes
    return added, size


def locate_bounded_external_records(
    config: VLMConfig,
    *,
    access: BenchmarkAccess | None = None,
) -> BoundedCanonicalSelection:
    """在固定探测上限内覆盖 2 roles、3 sources、7 tasks。"""

    if config.run.mode is not DataMode.EXTERNAL_GENERIC:
        raise PreflightError(
            ReasonCode.EXTERNAL_MASK_FORBIDDEN,
            "bounded External locator 只接受 external_generic",
        )
    opened = access or open_benchmark_access(config)
    if opened.root != config.data.benchmark_root.resolve():
        raise PreflightError(
            ReasonCode.BENCHMARK_IDENTITY_MISMATCH,
            "BenchmarkAccess root 与 locator 配置不一致",
        )
    limits = config.limits
    if (
        limits.max_probe_shards > 19
        or limits.max_records_per_shard > 256
        or limits.max_probe_records > 4864
        or limits.max_records > 16
        or limits.max_parents > 16
        or limits.max_assets > 32
        or limits.max_asset_bytes > 64 * 1024 * 1024
        or limits.max_copied_asset_bytes != 0
        or limits.max_total_tokens > 32768
    ):
        raise PreflightError(
            ReasonCode.ASSET_LIMIT_EXCEEDED,
            "bounded smoke 配置超过允许的探测/资产/token 上限",
        )
    shards = opened.record_shards
    counts: list[int] = []
    for index, shard in enumerate(shards):
        if (
            not isinstance(shard, dict)
            or set(shard) != {"path", "record_count"}
            or not isinstance(shard["path"], str)
            or isinstance(shard["record_count"], bool)
            or not isinstance(shard["record_count"], int)
            or shard["record_count"] <= 0
        ):
            raise PreflightError(
                ReasonCode.BENCHMARK_IDENTITY_MISMATCH,
                f"record_shards[{index}] 非法",
            )
        counts.append(int(shard["record_count"]))
    records_per_shard = max(counts)
    if any(value != records_per_shard for value in counts[:-1]):
        raise PreflightError(
            ReasonCode.BENCHMARK_IDENTITY_MISMATCH,
            "仅支持除尾分片外等长 canonical shards",
        )
    shard_indices = _probe_shard_indices(
        opened.statistics,
        records_per_shard=records_per_shard,
    )
    if len(shard_indices) > limits.max_probe_shards:
        raise PreflightError(
            ReasonCode.ASSET_LIMIT_EXCEEDED,
            f"需要探测 {len(shard_indices)} shards，超过配置上限",
        )
    candidates: list[
        tuple[CanonicalRecordLocation, Mapping[str, Any]]
    ] = []
    probed_records = 0
    for shard_index in shard_indices:
        if shard_index >= len(shards):
            raise PreflightError(
                ReasonCode.BENCHMARK_IDENTITY_MISMATCH,
                f"计算出的 shard index 越界：{shard_index}",
            )
        shard = shards[shard_index]
        count = min(
            int(shard["record_count"]),
            limits.max_records_per_shard,
        )
        if probed_records + count > limits.max_probe_records:
            raise PreflightError(
                ReasonCode.ASSET_LIMIT_EXCEEDED,
                "probe records 将超过配置上限，拒绝扩大窗口",
            )
        path = opened.verify_file(str(shard["path"]))
        rows = read_jsonl_indices(path, tuple(range(count)))
        for line_index, row in rows.items():
            validate_canonical_record(row)
            for media in row["media"]:
                opened.assert_declared_identity(
                    str(media["path"]),
                    size_bytes=media["size_bytes"],
                    sha256=media["sha256"],
                )
            candidates.append(
                (
                    CanonicalRecordLocation(shard["path"], line_index),
                    row,
                )
            )
        probed_records += count
    missing_roles = set(REQUIRED_EXTERNAL_ROLES)
    missing_sources = set(REQUIRED_EXTERNAL_SOURCES)
    missing_tasks = set(REQUIRED_EXTERNAL_TASKS)
    remaining = list(candidates)
    selected: list[
        tuple[CanonicalRecordLocation, Mapping[str, Any]]
    ] = []
    seen_parents: set[str] = set()
    seen_assets: set[str] = set()
    asset_bytes = 0
    while missing_roles or missing_sources or missing_tasks:
        feasible: list[
            tuple[
                tuple[int, int, int, str],
                CanonicalRecordLocation,
                Mapping[str, Any],
                set[str],
                int,
            ]
        ] = []
        for location, record in remaining:
            parent = str(record["parent_id"])
            if parent in seen_parents:
                continue
            added_assets, added_bytes = _asset_increment(
                record,
                seen=seen_assets,
            )
            if (
                len(seen_assets) + len(added_assets) > limits.max_assets
                or asset_bytes + added_bytes > limits.max_asset_bytes
            ):
                continue
            new_role = int(record["logical_role"] in missing_roles)
            new_source = int(record["source"] in missing_sources)
            new_task = int(record["task_family"] in missing_tasks)
            score = new_role + new_source + new_task
            if score == 0:
                continue
            feasible.append(
                (
                    (
                        -score,
                        -new_task,
                        -new_source,
                        stable_hash(
                            config.run.seed,
                            record["record_id"],
                            "bounded_external_selection",
                        ),
                    ),
                    location,
                    record,
                    added_assets,
                    added_bytes,
                )
            )
        if (
            not feasible
            or len(selected) >= limits.max_records
            or len(seen_parents) >= limits.max_parents
        ):
            raise PreflightError(
                ReasonCode.ASSET_LIMIT_EXCEEDED,
                "固定 bounded probe 内无法完成 role/source/task 覆盖",
                details={
                    "missing_roles": sorted(missing_roles),
                    "missing_sources": sorted(missing_sources),
                    "missing_tasks": sorted(missing_tasks),
                    "selected": len(selected),
                },
            )
        _, location, record, added_assets, added_bytes = min(
            feasible,
            key=lambda item: item[0],
        )
        selected.append((location, record))
        remaining = [
            item for item in remaining if item[0] != location
        ]
        seen_parents.add(str(record["parent_id"]))
        seen_assets.update(added_assets)
        asset_bytes += added_bytes
        missing_roles.discard(str(record["logical_role"]))
        missing_sources.discard(str(record["source"]))
        missing_tasks.discard(str(record["task_family"]))
    selected.sort(
        key=lambda item: (
            str(item[1]["logical_role"]),
            str(item[1]["source"]),
            str(item[1]["parent_id"]),
            str(item[1]["task_family"]),
            str(item[1]["record_id"]),
        )
    )
    records = tuple(record for _, record in selected)
    return BoundedCanonicalSelection(
        locations=tuple(location for location, _ in selected),
        records=records,
        probed_shards=tuple(
            str(shards[index]["path"]) for index in shard_indices
        ),
        probed_record_count=probed_records,
        selected_parent_count=len(seen_parents),
        selected_asset_count=len(seen_assets),
        selected_asset_bytes=asset_bytes,
        copied_asset_bytes=0,
        role_counts=dict(Counter(str(row["logical_role"]) for row in records)),
        source_counts=dict(Counter(str(row["source"]) for row in records)),
        task_counts=dict(Counter(str(row["task_family"]) for row in records)),
    )


@dataclass(frozen=True)
class DescriptionSample:
    record_id: str
    parent_id: str
    logical_role: str
    task_family: str
    messages: tuple[Mapping[str, Any], ...]
    reference_responses: tuple[str, ...]
    mask_mode: MaskMode
    evidence_ids: tuple[str, ...]
    provenance: Mapping[str, Any]
    counterfactual: Mapping[str, Any] | None = None


class ExternalDescriptionDataset(Dataset[DescriptionSample]):
    """Phase 2 canonical Dataset + 唯一 task-aware renderer。"""

    def __init__(
        self,
        canonical: RSGeneralDescDataset,
        *,
        derived_root: Path,
        seed: int,
        roles: Sequence[str] | None = None,
    ) -> None:
        self.canonical = canonical
        role_set = set(roles or REQUIRED_EXTERNAL_ROLES)
        if not role_set or role_set - REQUIRED_EXTERNAL_ROLES:
            raise ContractError(
                ReasonCode.ROLE_FORBIDDEN,
                "ExternalDescriptionDataset roles 只能是非空 "
                "external_train/external_val 子集",
            )
        self.records = [
            record
            for record in canonical.records
            if record["logical_role"] in role_set
        ]
        if not self.records:
            raise ContractError(
                ReasonCode.ASSET_MISSING,
                f"canonical Dataset 中没有请求的 roles：{sorted(role_set)}",
            )
        self.manifest = canonical.manifest
        self.derived_root = Path(derived_root)
        linked = first_symlink_component(self.derived_root)
        if linked is not None:
            raise ContractError(
                ReasonCode.OUTPUT_LINK,
                f"derived_root 含链接组件：{linked}",
            )
        if self.derived_root.exists() or self.derived_root.is_symlink():
            raise ContractError(
                ReasonCode.OUTPUT_EXISTS,
                f"derived_root 已存在：{self.derived_root}",
            )
        self.derived_root.mkdir(parents=True)
        self.seed = seed
        self._samples: dict[int, DescriptionSample] = {}
        for record in canonical.records:
            if record["logical_role"] not in REQUIRED_EXTERNAL_ROLES:
                raise ContractError(
                    ReasonCode.EXTERNAL_MASK_FORBIDDEN,
                    "ExternalDescriptionDataset 只接受 external_train/val",
                )

    def set_epoch(self, epoch: int) -> None:
        self.canonical.set_epoch(epoch)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> DescriptionSample:
        if index < 0:
            index += len(self.records)
        if index < 0 or index >= len(self.records):
            raise IndexError(index)
        sample = self._samples.get(index)
        if sample is None:
            sample = self._build_sample(index)
            self._samples[index] = sample
        if sample.logical_role != "external_train":
            return sample
        response = self.canonical.training_response(self.records[index])
        if response is None:
            raise ContractError(
                ReasonCode.TYPE_MISMATCH,
                "external_train 缺少 epoch-aware training response",
            )
        messages = copy.deepcopy(list(sample.messages))
        messages[-1] = {
            "role": "assistant",
            "content": [{"type": "text", "text": response}],
        }
        return replace(sample, messages=tuple(messages))

    def _build_sample(self, index: int) -> DescriptionSample:
        record = self.records[index]
        purpose = (
            "training"
            if record["logical_role"] == "external_train"
            else "validation"
        )
        rendered, _ = render_canonical_messages(
            record,
            purpose=purpose,
            seed=self.seed,
            dataset=self.canonical,
            derived_root=self.derived_root,
        )
        messages = self._resolve_assets(rendered["messages"])
        return DescriptionSample(
            record_id=str(record["record_id"]),
            parent_id=str(record["parent_id"]),
            logical_role=str(record["logical_role"]),
            task_family=str(record["task_family"]),
            messages=tuple(messages),
            reference_responses=tuple(
                str(value) for value in record["reference_responses"]
            ),
            mask_mode=MaskMode.EXTERNAL_GENERIC,
            evidence_ids=(),
            provenance={
                "canonical_build_id": self.manifest["build_id"],
                "canonical_payload_sha256": self.manifest[
                    "payload_root_sha256"
                ],
                "renderer": "phase3.render_canonical_messages",
            },
            counterfactual=None,
        )

    def _resolve_assets(
        self,
        messages: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                raise ContractError(
                    ReasonCode.TYPE_MISMATCH,
                    "renderer message content 必须是列表",
                )
            resolved: list[dict[str, Any]] = []
            for item in content:
                if not isinstance(item, dict):
                    raise ContractError(
                        ReasonCode.TYPE_MISMATCH,
                        "renderer content item 必须是对象",
                    )
                if item.get("type") != "image":
                    resolved.append(dict(item))
                    continue
                if "benchmark_asset" in item:
                    path = self.canonical.resolve_verified_asset_path(
                        str(item["benchmark_asset"]),
                    )
                elif "export_asset" in item:
                    pure = portable_relative_path(
                        str(item["export_asset"]),
                        location="renderer.export_asset",
                    )
                    path = self.derived_root.joinpath(*pure.parts)
                    try:
                        path.resolve().relative_to(self.derived_root.resolve())
                    except ValueError as error:
                        raise ContractError(
                            ReasonCode.PATH_ESCAPE,
                            "renderer export_asset 路径逃逸",
                        ) from error
                    if (
                        not path.is_file()
                        or path.is_symlink()
                        or path.stat().st_nlink != 1
                    ):
                        raise ContractError(
                            ReasonCode.OUTPUT_LINK,
                            "renderer export_asset 必须是普通单链接文件",
                        )
                else:
                    raise ContractError(
                        ReasonCode.ASSET_MISSING,
                        "renderer image 缺少 benchmark/export asset",
                    )
                resolved.append(
                    {
                        "type": "image",
                        "image": str(path.resolve()),
                        "asset_role": str(item.get("asset_role", "image")),
                    }
                )
            output.append({"role": str(message["role"]), "content": resolved})
        return output


def collate_description_samples(
    samples: Sequence[DescriptionSample],
) -> dict[str, Any]:
    if not samples:
        raise ContractError(
            ReasonCode.TYPE_MISMATCH,
            "不能 collate 空 description batch",
        )
    return {
        "samples": list(samples),
        "record_id": [sample.record_id for sample in samples],
        "parent_id": [sample.parent_id for sample in samples],
        "logical_role": [sample.logical_role for sample in samples],
        "task_family": [sample.task_family for sample in samples],
        "messages": [list(sample.messages) for sample in samples],
        "reference_responses": [
            sample.reference_responses for sample in samples
        ],
        "mask_mode": [sample.mask_mode.value for sample in samples],
        "evidence_ids": [sample.evidence_ids for sample in samples],
        "provenance": [sample.provenance for sample in samples],
        "counterfactual": [sample.counterfactual for sample in samples],
    }


def _unique_json(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate key: {key}")
        output[key] = value
    return output


def _find_prediction_row(
    path: Path,
    *,
    sample_id: str,
    max_rows: int,
) -> tuple[Mapping[str, Any], int]:
    if isinstance(max_rows, bool) or not isinstance(max_rows, int) or max_rows <= 0:
        raise ContractError(
            ReasonCode.TYPE_MISMATCH,
            "max_rows 必须是正整数",
        )
    if not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1:
        raise ContractError(
            ReasonCode.OUTPUT_LINK,
            "predictions.jsonl 必须是普通单链接文件",
        )
    found: Mapping[str, Any] | None = None
    row_count = 0
    try:
        with path.open(encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if index >= max_rows:
                    raise ContractError(
                        ReasonCode.ASSET_LIMIT_EXCEEDED,
                        "AuxSeg predictions 超过 max_rows，无法证明唯一性",
                    )
                row_count = index + 1
                if not line.strip():
                    raise ValueError("blank JSONL line")
                row = json.loads(line, object_pairs_hook=_unique_json)
                reject_nonfinite(row, location=f"{path}:{index + 1}")
                if not isinstance(row, dict) or set(row) != AUXSEG_PREDICTION_FIELDS:
                    raise ContractError(
                        ReasonCode.UNKNOWN_FIELD,
                        f"AuxSeg prediction[{index}] 字段合同非法",
                    )
                if (
                    not isinstance(row["sample_id"], str)
                    or not row["sample_id"]
                ):
                    raise ContractError(
                        ReasonCode.TYPE_MISMATCH,
                        f"AuxSeg prediction[{index}].sample_id 非法",
                    )
                if row.get("sample_id") == sample_id:
                    if found is not None:
                        raise ValueError("duplicate sample_id")
                    found = row
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise ContractError(
            ReasonCode.TYPE_MISMATCH,
            "无法严格读取 AuxSeg prediction",
            details={"error": str(error)},
        ) from error
    if found is None:
        raise ContractError(
            ReasonCode.REGION_NOT_FOUND,
            f"AuxSeg inference 中不存在 sample_id={sample_id}",
        )
    return found, row_count


def inventory_from_auxseg_inference(
    root: Path,
    *,
    sample_id: str,
    mask_mode: MaskMode,
    max_rows: int = 100_000,
) -> RegionInventory:
    if mask_mode not in {
        MaskMode.FIXED_PREDICTED_MASK,
        MaskMode.END_TO_END_PREDICTED_MASK,
    }:
        raise ContractError(
            ReasonCode.INVALID_ENUM,
            "AuxSeg inference 只对应 predicted mask mode",
        )
    root = Path(root)
    linked = first_symlink_component(root)
    if linked is not None:
        raise ContractError(
            ReasonCode.OUTPUT_LINK,
            f"AuxSeg inference root 含链接组件：{linked}",
        )
    if root.is_symlink() or not root.is_dir():
        raise ContractError(
            ReasonCode.ASSET_MISSING,
            f"AuxSeg inference root 不存在：{root}",
        )
    manifest_path = root / "manifest.json"
    if (
        not manifest_path.is_file()
        or manifest_path.is_symlink()
        or manifest_path.stat().st_nlink != 1
    ):
        raise ContractError(
            ReasonCode.OUTPUT_LINK,
            "AuxSeg manifest.json 必须是普通单链接文件",
        )
    manifest = read_json(manifest_path)
    if (
        not isinstance(manifest, dict)
        or set(manifest) != AUXSEG_MANIFEST_FIELDS
        or manifest.get("schema_version") != SUPPORTED_AUXSEG_INFERENCE_SCHEMA
    ):
        raise ContractError(
            ReasonCode.BENCHMARK_IDENTITY_MISMATCH,
            "AuxSeg inference manifest/schema 合同不受支持",
        )
    sample_count = manifest.get("sample_count")
    checkpoint_step = manifest.get("checkpoint_step")
    benchmark_sha = manifest.get("benchmark_index_sha256")
    if (
        isinstance(sample_count, bool)
        or not isinstance(sample_count, int)
        or sample_count <= 0
        or sample_count > max_rows
        or isinstance(checkpoint_step, bool)
        or not isinstance(checkpoint_step, int)
        or checkpoint_step < 0
        or not isinstance(benchmark_sha, str)
        or len(benchmark_sha) != 64
    ):
        raise ContractError(
            ReasonCode.TYPE_MISMATCH,
            "AuxSeg manifest sample_count/checkpoint identity 非法",
        )
    row, row_count = _find_prediction_row(
        root / "predictions.jsonl",
        sample_id=sample_id,
        max_rows=max_rows,
    )
    if row_count != sample_count:
        raise ContractError(
            ReasonCode.BENCHMARK_IDENTITY_MISMATCH,
            "AuxSeg manifest.sample_count 与 predictions.jsonl 不一致",
        )
    array_keys = row.get("array_keys")
    regions = row.get("regions")
    if (
        not isinstance(array_keys, dict)
        or set(array_keys) != AUXSEG_ARRAY_KEY_FIELDS
        or not isinstance(array_keys.get("global_mask"), str)
        or not isinstance(array_keys.get("region_masks"), str)
        or not isinstance(array_keys.get("modality_weight_maps"), dict)
        or not isinstance(regions, list)
    ):
        raise ContractError(
            ReasonCode.TYPE_MISMATCH,
            "AuxSeg prediction array_keys/regions 合同非法",
        )
    archive_path = root / "predictions.npz"
    if (
        not archive_path.is_file()
        or archive_path.is_symlink()
        or archive_path.stat().st_nlink != 1
    ):
        raise ContractError(
            ReasonCode.OUTPUT_LINK,
            "predictions.npz 必须是普通单链接文件",
        )
    try:
        with np.load(archive_path, allow_pickle=False) as archive:
            if (
                array_keys["global_mask"] not in archive
                or array_keys["region_masks"] not in archive
            ):
                raise ContractError(
                    ReasonCode.ASSET_MISSING,
                    "AuxSeg mask array key 缺失",
                )
            global_mask = np.asarray(
                archive[array_keys["global_mask"]]
            ).copy()
            region_masks = np.asarray(
                archive[array_keys["region_masks"]]
            ).copy()
    except (OSError, ValueError) as error:
        raise ContractError(
            ReasonCode.TYPE_MISMATCH,
            "无法严格读取 AuxSeg predictions.npz",
            details={"error": str(error)},
        ) from error
    if region_masks.ndim != 3 or region_masks.shape[0] != len(regions):
        raise ContractError(
            ReasonCode.MASK_SHAPE_MISMATCH,
            "AuxSeg region metadata/masks 数量不一致",
        )
    candidates: list[RegionCandidate] = []
    for index, region in enumerate(regions):
        if (
            not isinstance(region, dict)
            or set(region)
            != {
                "region_id",
                "bbox_xyxy",
                "centroid_xy",
                "area_pixels",
                "confidence",
            }
        ):
            raise ContractError(
                ReasonCode.UNKNOWN_FIELD,
                f"AuxSeg regions[{index}] 合同非法",
            )
        region_id = region["region_id"]
        bbox = region["bbox_xyxy"]
        centroid = region["centroid_xy"]
        area = region["area_pixels"]
        confidence = region["confidence"]
        if (
            isinstance(region_id, bool)
            or not isinstance(region_id, (str, int))
            or not str(region_id).strip()
            or not isinstance(bbox, list)
            or len(bbox) != 4
            or not isinstance(centroid, list)
            or len(centroid) != 2
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in (*bbox, *centroid)
            )
            or isinstance(area, bool)
            or not isinstance(area, int)
            or area <= 0
            or isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not math.isfinite(float(confidence))
            or not 0.0 <= float(confidence) <= 1.0
        ):
            raise ContractError(
                ReasonCode.TYPE_MISMATCH,
                f"AuxSeg regions[{index}] 值类型非法",
            )
        candidates.append(
            RegionCandidate(
                region_id=str(region_id),
                mask=region_masks[index],
                confidence=float(confidence),
                provenance="oa_auxseg_inference_v6",
            )
        )
    return RegionInventory(
        sample_id=sample_id,
        mask_mode=mask_mode,
        global_mask=global_mask,
        candidates=tuple(candidates),
        source_identity=(
            "oa_auxseg_inference_v6:"
            f"{sha256_file(manifest_path)}:{sample_id}"
        ),
    )
