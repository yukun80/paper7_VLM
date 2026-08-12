"""manifest 驱动、完全 source-hidden 的 canonical Dataset API。"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler

from .common import (
    first_symlink_component,
    portable_relative_path,
    read_json,
    read_jsonl,
    read_jsonl_indices,
    sha256_file,
    stable_hash,
)
from .contracts import (
    BENCHMARK_SCOPE,
    CANONICAL_SCHEMA_VERSION,
    MANIFEST_VERSION,
    AnnotationLayer,
    LogicalRole,
    RS_GENERALDESC_ROLES,
    RS_GENERALDESC_SOURCES,
    RS_GENERALDESC_TASK_FAMILIES,
    TaskFamily,
    validate_canonical_record,
)
from .errors import ReasonCode, SchemaError
from .hash_ledger import HashLedgerVerifier


@dataclass(frozen=True)
class CanonicalRecordLocation:
    """manifest 所列 canonical 分片中的零基记录位置。"""

    shard_path: str
    line_index: int

    def __post_init__(self) -> None:
        portable_relative_path(
            self.shard_path,
            location="CanonicalRecordLocation.shard_path",
        )
        if (
            isinstance(self.line_index, bool)
            or not isinstance(self.line_index, int)
            or self.line_index < 0
        ):
            raise SchemaError(
                ReasonCode.TYPE_MISMATCH,
                "CanonicalRecordLocation.line_index 必须是 >= 0 的整数",
            )


class RSGeneralDescDataset(Dataset[dict[str, Any]]):
    """读取原生 v1 RS-GeneralDesc train/val records。"""

    def __init__(
        self,
        root: Path | str,
        *,
        roles: Sequence[str] | None = None,
        sources: Sequence[str] | None = None,
        task_families: Sequence[str] | None = None,
        load_assets: bool = True,
        seed: int = 0,
        epoch: int = 0,
        record_locations: Sequence[CanonicalRecordLocation] | None = None,
        expected_manifest_sha256: str | None = None,
        verifier: HashLedgerVerifier | None = None,
    ) -> None:
        input_root = Path(root).absolute()
        linked_root = first_symlink_component(input_root)
        if linked_root is not None:
            raise SchemaError(
                ReasonCode.OUTPUT_LINK,
                f"Benchmark root 含 symlink 组件：{linked_root}",
            )
        self.root = input_root
        if not self.root.is_dir():
            raise SchemaError(
                ReasonCode.OUTPUT_LINK,
                "Benchmark root 必须是普通目录",
            )
        manifest_path = self.root / "manifest.json"
        if (
            not manifest_path.is_file()
            or manifest_path.is_symlink()
            or manifest_path.stat().st_nlink != 1
        ):
            raise SchemaError(
                ReasonCode.OUTPUT_LINK,
                "Benchmark manifest 必须是普通单链接文件",
            )
        if (
            expected_manifest_sha256 is not None
            and sha256_file(manifest_path) != expected_manifest_sha256
        ):
            raise SchemaError(
                ReasonCode.HASH_MISMATCH,
                "Benchmark manifest SHA-256 与运行配置不一致",
                details={"path": "manifest.json"},
            )
        self.manifest = read_json(manifest_path)
        if (
            self.manifest.get("schema_version") != MANIFEST_VERSION
            or self.manifest.get("canonical_schema_version")
            != CANONICAL_SCHEMA_VERSION
            or self.manifest.get("benchmark_scope") != BENCHMARK_SCOPE
        ):
            raise SchemaError(
                ReasonCode.SCHEMA_MISMATCH,
                "仅支持原生 rs_generaldesc v1 Benchmark",
            )
        layout = self.manifest.get("layout")
        if not isinstance(layout, dict) or not isinstance(
            layout.get("record_shards"), list
        ):
            raise SchemaError(
                ReasonCode.SCHEMA_MISMATCH,
                "manifest 缺少 record_shards layout",
            )
        hashes_relative = layout.get("hashes")
        payload_sha256 = self.manifest.get("payload_root_sha256")
        hash_manifest_sha256 = self.manifest.get("hash_manifest_sha256")
        if not all(
            isinstance(value, str)
            for value in (
                hashes_relative,
                payload_sha256,
                hash_manifest_sha256,
            )
        ):
            raise SchemaError(
                ReasonCode.SCHEMA_MISMATCH,
                "manifest 缺少 hash ledger identity",
            )
        if verifier is None:
            verifier = HashLedgerVerifier(
                self.root,
                ledger_path=hashes_relative,
                expected_ledger_sha256=hash_manifest_sha256,
                expected_root_sha256=payload_sha256,
            )
        elif (
            verifier.root.resolve() != self.root.resolve()
            or verifier.ledger_relative != hashes_relative
            or verifier.ledger_sha256 != hash_manifest_sha256
            or verifier.root_sha256 != payload_sha256
        ):
            raise SchemaError(
                ReasonCode.HASH_MISMATCH,
                "共享 hash ledger verifier 与 manifest identity 不一致",
                details={"path": str(hashes_relative)},
            )
        self.verifier = verifier
        shard_contracts: dict[str, tuple[Path, int]] = {}
        for shard in layout["record_shards"]:
            if not isinstance(shard, dict) or not isinstance(shard.get("path"), str):
                raise SchemaError(
                    ReasonCode.SCHEMA_MISMATCH,
                    "manifest record shard 合同非法",
                )
            if set(shard) != {"path", "record_count"}:
                raise SchemaError(
                    ReasonCode.SCHEMA_MISMATCH,
                    "manifest record shard 字段非法",
                )
            record_count = shard.get("record_count")
            if (
                isinstance(record_count, bool)
                or not isinstance(record_count, int)
                or record_count <= 0
            ):
                raise SchemaError(
                    ReasonCode.SCHEMA_MISMATCH,
                    f"manifest record shard count 非法：{shard['path']}",
                )
            shard_path = self.resolve_contract_path(
                shard["path"],
                location="manifest.record_shards.path",
            )
            relative = str(shard["path"])
            self.verifier.entry(relative)
            if relative in shard_contracts:
                raise SchemaError(
                    ReasonCode.SCHEMA_MISMATCH,
                    f"manifest record shard 重复：{relative}",
                )
            shard_contracts[relative] = (shard_path, record_count)
        located_rows: list[
            tuple[CanonicalRecordLocation | None, dict[str, Any]]
        ] = []
        if record_locations is None:
            for relative, (shard_path, record_count) in shard_contracts.items():
                self.verifier.verify_file(relative)
                shard_rows = read_jsonl(shard_path)
                if record_count != len(shard_rows):
                    raise SchemaError(
                        ReasonCode.SCHEMA_MISMATCH,
                        f"record shard count 不一致：{relative}",
                    )
                located_rows.extend((None, row) for row in shard_rows)
        else:
            locations = tuple(record_locations)
            if not locations:
                raise SchemaError(
                    ReasonCode.SCHEMA_MISMATCH,
                    "record_locations 不能为空",
                )
            if not all(
                isinstance(location, CanonicalRecordLocation)
                for location in locations
            ):
                raise SchemaError(
                    ReasonCode.TYPE_MISMATCH,
                    "record_locations 必须全部是 CanonicalRecordLocation",
                )
            if len(set(locations)) != len(locations):
                raise SchemaError(
                    ReasonCode.SCHEMA_MISMATCH,
                    "record_locations 不能重复",
                )
            grouped: dict[str, list[int]] = defaultdict(list)
            for location in locations:
                contract = shard_contracts.get(location.shard_path)
                if contract is None:
                    raise SchemaError(
                        ReasonCode.PATH_ESCAPE,
                        f"location shard 不在 manifest：{location.shard_path}",
                    )
                if location.line_index >= contract[1]:
                    raise SchemaError(
                        ReasonCode.SCHEMA_MISMATCH,
                        f"location 超出 manifest record_count：{location}",
                    )
                grouped[location.shard_path].append(location.line_index)
            selected: dict[CanonicalRecordLocation, dict[str, Any]] = {}
            for relative, indices in grouped.items():
                shard_path = shard_contracts[relative][0]
                self.verifier.verify_file(relative)
                for index, row in read_jsonl_indices(
                    shard_path,
                    indices,
                ).items():
                    selected[
                        CanonicalRecordLocation(relative, index)
                    ] = row
            located_rows = [
                (location, selected[location]) for location in locations
            ]
        for _, row in located_rows:
            validate_canonical_record(row)
            if (
                LogicalRole(row["logical_role"]) not in RS_GENERALDESC_ROLES
                or TaskFamily(row["task_family"])
                not in RS_GENERALDESC_TASK_FAMILIES
                or row["annotation"]["layer"]
                != AnnotationLayer.EXTERNAL_SOURCE.value
            ):
                raise SchemaError(
                    ReasonCode.ROLE_CONTAMINATION,
                    "RS-GeneralDesc Dataset 只消费原生 train/val 合同",
                )
            for media in row["media"]:
                self.verifier.assert_declared_identity(
                    str(media["path"]),
                    size_bytes=media["size_bytes"],
                    sha256=media["sha256"],
                )
        role_set = set(roles or ())
        source_set = set(sources or ())
        task_set = set(task_families or ())
        if role_set:
            for value in role_set:
                try:
                    valid = LogicalRole(value) in RS_GENERALDESC_ROLES
                except ValueError:
                    valid = False
                if not valid:
                    raise SchemaError(
                        ReasonCode.ROLE_CONTAMINATION,
                        f"RS-GeneralDesc Dataset 不接受 role={value}",
                    )
        if task_set:
            for value in task_set:
                try:
                    valid = TaskFamily(value) in RS_GENERALDESC_TASK_FAMILIES
                except ValueError:
                    valid = False
                if not valid:
                    raise SchemaError(
                        ReasonCode.ROLE_CONTAMINATION,
                        f"RS-GeneralDesc Dataset 不接受 task={value}",
                    )
        unknown_sources = source_set - RS_GENERALDESC_SOURCES
        if unknown_sources:
            raise SchemaError(
                ReasonCode.ROLE_CONTAMINATION,
                f"RS-GeneralDesc Dataset 不接受 source={sorted(unknown_sources)}",
            )
        selected_rows = [
            (location, row)
            for location, row in located_rows
            if (not role_set or row["logical_role"] in role_set)
            and (not source_set or row["source"] in source_set)
            and (not task_set or row["task_family"] in task_set)
        ]
        self.record_locations = tuple(location for location, _ in selected_rows)
        self.records = [row for _, row in selected_rows]
        self.load_assets = load_assets
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise SchemaError(
                ReasonCode.TYPE_MISMATCH,
                "Dataset seed 必须是 >= 0 的整数",
            )
        self.seed = seed
        self.set_epoch(epoch)

    @classmethod
    def from_locations(
        cls,
        root: Path | str,
        locations: Sequence[CanonicalRecordLocation],
        **kwargs: Any,
    ) -> "RSGeneralDescDataset":
        """由 manifest 内已知位置构造严格有界 Dataset。"""

        if "record_locations" in kwargs:
            raise SchemaError(
                ReasonCode.TYPE_MISMATCH,
                "from_locations 不接受重复的 record_locations 参数",
            )
        return cls(root, record_locations=locations, **kwargs)

    def resolve_contract_path(self, relative: str, *, location: str) -> Path:
        """解析 manifest 路径；拒绝路径逃逸、链接组件和多硬链接。"""

        pure = portable_relative_path(relative, location=location)
        path = self.root.joinpath(*pure.parts)
        linked = first_symlink_component(path)
        if linked is not None:
            raise SchemaError(
                ReasonCode.OUTPUT_LINK,
                f"{location}: 路径含 symlink 组件 {linked}",
            )
        try:
            path.resolve().relative_to(self.root.resolve())
        except ValueError as error:
            raise SchemaError(
                ReasonCode.PATH_ESCAPE,
                f"{location}: 路径逃逸",
            ) from error
        if not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1:
            raise SchemaError(
                ReasonCode.OUTPUT_LINK,
                f"{location}: 必须是普通单链接文件",
            )
        return path

    def __len__(self) -> int:
        return len(self.records)

    def set_epoch(self, epoch: int) -> None:
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise SchemaError(
                ReasonCode.TYPE_MISMATCH,
                "Dataset epoch 必须是 >= 0 的整数",
            )
        self.epoch = epoch

    def training_response(self, record: Mapping[str, Any]) -> str | None:
        """按 seed/epoch 稳定选择 train reference；val 返回 null。"""

        if record["logical_role"] != LogicalRole.EXTERNAL_TRAIN.value:
            return None
        values = record["training_responses"]
        if not isinstance(values, list) or not values:
            raise SchemaError(
                ReasonCode.SCHEMA_MISMATCH,
                f"{record['record_id']}: 训练记录缺少 training_responses",
            )
        rank = int(
            stable_hash(
                self.seed,
                self.epoch,
                record["record_id"],
                "training_reference",
            ),
            16,
        )
        return str(values[rank % len(values)])

    def _resolve_asset(self, media: Mapping[str, Any]) -> Path:
        relative = str(media["path"])
        self.verifier.assert_declared_identity(
            relative,
            size_bytes=media.get("size_bytes"),
            sha256=media.get("sha256"),
        )
        return self.verifier.verify_file(relative)

    def resolve_verified_asset_path(self, relative: str) -> Path:
        """在返回运行时路径前验证其实际 bytes。"""

        return self.verifier.verify_file(relative)

    def load_image(self, media: Mapping[str, Any]) -> torch.Tensor:
        path = self._resolve_asset(media)
        with Image.open(path) as image:
            array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        return torch.from_numpy(array).permute(2, 0, 1).contiguous()

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        item: dict[str, Any] = {
            "record": record,
            "record_id": record["record_id"],
            "parent_id": record["parent_id"],
            "logical_role": record["logical_role"],
            "task_family": record["task_family"],
            "supervision_kind": record["supervision_kind"],
            "input_layout": record["input_layout"],
            "output_modality": record["output_modality"],
            "training_response": self.training_response(record),
            "reference_responses": tuple(record["reference_responses"]),
            "bboxes": (
                list(record["target"].get("boxes", []))
                if record["target"]["type"] == "bbox"
                else []
            ),
        }
        if not self.load_assets:
            item["images"] = {}
            return item
        images: dict[str, torch.Tensor] = {}
        for media in record["media"]:
            if media["media_type"] != "image":
                raise SchemaError(
                    ReasonCode.INVALID_ENUM,
                    f"未知 media_type：{media['media_type']}",
                )
            images[str(media["role"])] = self.load_image(media)
        item["images"] = images
        return item


def collate_canonical_samples(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("不能 collate 空 canonical batch")
    return {
        "record_id": [sample["record_id"] for sample in samples],
        "parent_id": [sample["parent_id"] for sample in samples],
        "logical_role": [sample["logical_role"] for sample in samples],
        "task_family": [sample["task_family"] for sample in samples],
        "supervision_kind": [sample["supervision_kind"] for sample in samples],
        "input_layout": [sample["input_layout"] for sample in samples],
        "output_modality": [sample["output_modality"] for sample in samples],
        "training_response": [sample["training_response"] for sample in samples],
        "reference_responses": [
            sample["reference_responses"] for sample in samples
        ],
        "images": [sample["images"] for sample in samples],
        "bboxes": [sample["bboxes"] for sample in samples],
        "records": [sample["record"] for sample in samples],
    }


class ParentBalancedSampler(Sampler[int]):
    """先按 task 做平滑加权轮转，再在 task 内平衡 parent 与记录视图。"""

    def __init__(
        self,
        dataset: RSGeneralDescDataset,
        *,
        seed: int,
        num_samples: int | None = None,
        source_weights: Mapping[str, float] | None = None,
        task_weights: Mapping[str, float] | None = None,
    ) -> None:
        if len(dataset) == 0:
            raise ValueError("ParentBalancedSampler 不接受空 Dataset")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed 必须是整数")
        self.dataset = dataset
        self.seed = seed
        if num_samples is not None and (
            isinstance(num_samples, bool) or not isinstance(num_samples, int)
        ):
            raise ValueError("num_samples 必须是整数或 null")
        self.num_samples = len(dataset) if num_samples is None else num_samples
        if self.num_samples <= 0:
            raise ValueError("num_samples 必须大于 0")
        self.source_weights = dict(source_weights or {})
        self.task_weights = dict(task_weights or {})
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0
            for value in (*self.source_weights.values(), *self.task_weights.values())
        ):
            raise ValueError("sampler 权重必须为有限正数")
        available_sources = {str(row["source"]) for row in dataset.records}
        available_tasks = {str(row["task_family"]) for row in dataset.records}
        if set(self.source_weights) - available_sources:
            raise ValueError("source_weights 含 Dataset 中不存在的 source")
        if set(self.task_weights) - available_tasks:
            raise ValueError("task_weights 含 Dataset 中不存在的 task family")
        by_task_parent: dict[str, dict[str, list[int]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for index, record in enumerate(dataset.records):
            by_task_parent[str(record["task_family"])][
                str(record["parent_id"])
            ].append(index)
        self.by_task_parent = {
            task: {
                parent: tuple(indices)
                for parent, indices in sorted(parent_rows.items())
            }
            for task, parent_rows in sorted(by_task_parent.items())
        }
        self.set_epoch(0)

    def set_epoch(self, epoch: int) -> None:
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("epoch 必须是 >= 0 的整数")
        self.epoch = epoch
        self.dataset.set_epoch(epoch)

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "rs_generaldesc.task_parent_sampler.v1",
            "benchmark_payload_root_sha256": self.dataset.manifest[
                "payload_root_sha256"
            ],
            "seed": self.seed,
            "epoch": self.epoch,
            "num_samples": self.num_samples,
            "source_weights": dict(sorted(self.source_weights.items())),
            "task_weights": dict(sorted(self.task_weights.items())),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        required = {
            "schema_version",
            "benchmark_payload_root_sha256",
            "seed",
            "epoch",
            "num_samples",
            "source_weights",
            "task_weights",
        }
        expected = self.state_dict()
        if set(state) != required or any(
            state[key] != expected[key] for key in required - {"epoch"}
        ):
            raise ValueError("sampler state 与当前 Benchmark/配置不兼容")
        self.set_epoch(state["epoch"])

    def _parent_weight(self, task: str, parent: str) -> float:
        indices = self.by_task_parent[task][parent]
        values = []
        for index in indices:
            row = self.dataset.records[index]
            values.append(float(self.source_weights.get(str(row["source"]), 1.0)))
        return sum(values) / len(values)

    def _parent_order(self, task: str, cycle: int) -> list[str]:
        random_state = random.Random(
            int(stable_hash(self.seed, self.epoch, task, cycle, "parents"), 16)
        )
        return sorted(
            self.by_task_parent[task],
            key=lambda parent: (
                -math.log(max(random_state.random(), 1e-15))
                / self._parent_weight(task, parent),
                stable_hash(self.seed, self.epoch, task, cycle, parent),
            ),
        )

    def __iter__(self) -> Iterator[int]:
        tasks = tuple(sorted(self.by_task_parent))
        task_weights = {
            task: float(self.task_weights.get(task, 1.0)) for task in tasks
        }
        total_task_weight = sum(task_weights.values())
        deficits = {task: 0.0 for task in tasks}
        parent_cycles = {task: 0 for task in tasks}
        parent_orders = {
            task: self._parent_order(task, 0) for task in tasks
        }
        parent_positions = {task: 0 for task in tasks}
        record_positions: dict[tuple[str, str], int] = defaultdict(int)
        output: list[int] = []
        for step in range(self.num_samples):
            for task in tasks:
                deficits[task] += task_weights[task]
            task = min(
                tasks,
                key=lambda name: (
                    -deficits[name],
                    stable_hash(self.seed, self.epoch, step, name),
                ),
            )
            deficits[task] -= total_task_weight
            if parent_positions[task] >= len(parent_orders[task]):
                parent_cycles[task] += 1
                parent_orders[task] = self._parent_order(
                    task, parent_cycles[task]
                )
                parent_positions[task] = 0
            parent = parent_orders[task][parent_positions[task]]
            parent_positions[task] += 1
            indices = self.by_task_parent[task][parent]
            key = (task, parent)
            offset = int(
                stable_hash(
                    self.seed,
                    self.epoch,
                    task,
                    parent,
                    "record_view",
                ),
                16,
            ) % len(indices)
            position = (offset + record_positions[key]) % len(indices)
            output.append(indices[position])
            record_positions[key] += 1
        return iter(output)

    def __len__(self) -> int:
        return self.num_samples
