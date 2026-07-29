"""manifest 驱动、完全 source-hidden 的 canonical Dataset API。"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler

from .common import portable_relative_path, read_json, read_jsonl, stable_hash
from .contracts import (
    CANONICAL_SCHEMA_VERSION,
    MANIFEST_VERSION,
    AnnotationLayer,
    LogicalRole,
    TaskFamily,
    validate_canonical_record,
)
from .errors import ReasonCode, SchemaError


class OALandslideDescDataset(Dataset[dict[str, Any]]):
    """只依赖构建产物；不接受也不保存任何 source-root 配置。"""

    def __init__(
        self,
        root: Path | str,
        *,
        roles: Sequence[str] | None = None,
        sources: Sequence[str] | None = None,
        task_families: Sequence[str] | None = None,
        annotation_layers: Sequence[str] | None = None,
        load_assets: bool = True,
        seed: int = 0,
        epoch: int = 0,
    ) -> None:
        input_root = Path(root)
        if input_root.is_symlink():
            raise SchemaError(
                ReasonCode.OUTPUT_LINK,
                "Benchmark root 不能是 symlink",
            )
        self.root = input_root.resolve()
        if not self.root.is_dir():
            raise SchemaError(
                ReasonCode.OUTPUT_LINK,
                "Benchmark root 必须是普通目录",
            )
        self.manifest = read_json(self.root / "manifest.json")
        if (
            self.manifest.get("schema_version") != MANIFEST_VERSION
            or self.manifest.get("canonical_schema_version")
            != CANONICAL_SCHEMA_VERSION
        ):
            raise SchemaError(
                ReasonCode.SCHEMA_MISMATCH,
                "Benchmark manifest/canonical schema version 不受支持",
            )
        layout = self.manifest.get("layout")
        if not isinstance(layout, dict) or not isinstance(
            layout.get("record_shards"), list
        ):
            raise SchemaError(
                ReasonCode.SCHEMA_MISMATCH,
                "manifest 缺少 record_shards layout",
            )
        rows: list[dict[str, Any]] = []
        for shard in layout["record_shards"]:
            if not isinstance(shard, dict) or not isinstance(shard.get("path"), str):
                raise SchemaError(
                    ReasonCode.SCHEMA_MISMATCH,
                    "manifest record shard 合同非法",
                )
            shard_path = self._resolve_contract_path(
                shard["path"],
                location="manifest.record_shards.path",
            )
            shard_rows = read_jsonl(shard_path)
            if (
                isinstance(shard.get("record_count"), bool)
                or not isinstance(shard.get("record_count"), int)
                or shard["record_count"] != len(shard_rows)
            ):
                raise SchemaError(
                    ReasonCode.SCHEMA_MISMATCH,
                    f"record shard count 不一致：{shard['path']}",
                )
            for row in shard_rows:
                validate_canonical_record(row)
            rows.extend(shard_rows)
        role_set = set(roles or ())
        source_set = set(sources or ())
        task_set = set(task_families or ())
        layer_set = set(annotation_layers or ())
        if role_set:
            for value in role_set:
                LogicalRole(value)
        if task_set:
            for value in task_set:
                TaskFamily(value)
        if layer_set:
            for value in layer_set:
                AnnotationLayer(value)
        self.records = [
            row
            for row in rows
            if (not role_set or row["logical_role"] in role_set)
            and (not source_set or row["source"] in source_set)
            and (not task_set or row["task_family"] in task_set)
            and (not layer_set or row["annotation"]["layer"] in layer_set)
        ]
        self.load_assets = load_assets
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("Dataset seed 必须是 >= 0 的整数")
        self.seed = seed
        self.set_epoch(epoch)

    def _resolve_contract_path(self, relative: str, *, location: str) -> Path:
        pure = portable_relative_path(relative, location=location)
        path = self.root.joinpath(*pure.parts)
        try:
            path.resolve().relative_to(self.root)
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
            raise ValueError("Dataset epoch 必须是 >= 0 的整数")
        self.epoch = epoch

    def _training_response(self, record: Mapping[str, Any]) -> str | None:
        if record["logical_role"] not in {
            LogicalRole.EXTERNAL_TRAIN.value,
            LogicalRole.OA_TRAIN.value,
        }:
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
        return self._resolve_contract_path(
            str(media["path"]),
            location=f"asset[{media.get('asset_id', 'unknown')}].path",
        )

    def load_image(self, media: Mapping[str, Any]) -> torch.Tensor:
        path = self._resolve_asset(media)
        with Image.open(path) as image:
            array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        return torch.from_numpy(array).permute(2, 0, 1).contiguous()

    def load_mask(self, media: Mapping[str, Any]) -> torch.Tensor:
        path = self._resolve_asset(media)
        with Image.open(path) as image:
            array = (np.asarray(image.convert("L"), dtype=np.uint8) > 0).astype(
                np.uint8
            )
        return torch.from_numpy(array.copy())[None, ...]

    def load_array(self, media: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        path = self._resolve_asset(media)
        arrays: dict[str, torch.Tensor] = {}
        with np.load(path, allow_pickle=False) as archive:
            for name in sorted(archive.files):
                value = np.asarray(archive[name]).copy()
                arrays[name] = torch.from_numpy(value)
        return arrays

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
            "training_response": self._training_response(record),
            "reference_responses": tuple(record["reference_responses"]),
            "bboxes": (
                list(record["target"].get("boxes", []))
                if record["target"]["type"] == "bbox"
                else []
            ),
        }
        if not self.load_assets:
            item.update({"images": {}, "masks": {}, "arrays": {}})
            return item
        images: dict[str, torch.Tensor] = {}
        masks: dict[str, torch.Tensor] = {}
        arrays: dict[str, dict[str, torch.Tensor]] = {}
        for media in record["media"]:
            media_type = media["media_type"]
            if media_type == "image":
                images[str(media["role"])] = self.load_image(media)
            elif media_type == "mask":
                masks[str(media["role"])] = self.load_mask(media)
            elif media_type == "array":
                arrays[str(media["role"])] = self.load_array(media)
            else:
                raise SchemaError(
                    ReasonCode.INVALID_ENUM,
                    f"未知 media_type：{media_type}",
                )
        item.update({"images": images, "masks": masks, "arrays": arrays})
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
        "masks": [sample["masks"] for sample in samples],
        "arrays": [sample["arrays"] for sample in samples],
        "bboxes": [sample["bboxes"] for sample in samples],
        "records": [sample["record"] for sample in samples],
    }


class ParentBalancedSampler(Sampler[int]):
    """先按 task 做平滑加权轮转，再在 task 内平衡 parent 与记录视图。"""

    def __init__(
        self,
        dataset: OALandslideDescDataset,
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
            "schema_version": "oa_landslidedesc.task_parent_sampler.v2",
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
