"""只读 Benchmark Browser 与 Frozen Evaluation catalog。"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import random
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from PIL import Image

from oa_groundrag.artifacts.identity import canonical_json, sha256_file, sha256_text
from oa_groundrag.artifacts.io import first_symlink_component
from oa_groundrag.data.grounded.pilot import render_optical
from oa_groundrag.data.grounded.region_contracts import size_bin
from oa_groundrag.data.oa_auxseg.dataset import BenchmarkDataset, load_index

from .access import DemoTestAccessController, DemoTestAccessReceipt
from .config import BenchmarkBinding, FrozenEvaluationBinding


class DemoCatalogError(RuntimeError):
    """只读目录的过滤、身份或 payload 访问合同失败。"""


@dataclass(frozen=True)
class BenchmarkFilter:
    split: str
    source: str | None = None
    sample_id_query: str | None = None
    target_status: str = "all"
    size: str = "all"
    modalities: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.split not in {"train", "val", "test"}:
            raise DemoCatalogError(f"未知 split：{self.split}")
        if self.target_status not in {"all", "target", "no_target"}:
            raise DemoCatalogError(f"未知 target filter：{self.target_status}")
        if self.size not in {"all", "empty", "small", "medium", "large"}:
            raise DemoCatalogError(f"未知 size filter：{self.size}")
        if len(self.modalities) != len(set(self.modalities)):
            raise DemoCatalogError("modality filter 不允许重复")


@dataclass(frozen=True)
class BenchmarkRecord:
    canonical_index: int
    sample_id: str
    split: str
    source: str
    foreground_ratio: float
    target_status: str
    size: str
    available_modalities: tuple[str, ...]
    optical_channel_names: tuple[str, ...]
    auxiliary_channel_names: Mapping[str, tuple[str, ...]]
    row: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical_index": self.canonical_index,
            "sample_id": self.sample_id,
            "split": self.split,
            "source": self.source,
            "foreground_ratio": self.foreground_ratio,
            "target_status": self.target_status,
            "size": self.size,
            "available_modalities": list(self.available_modalities),
            "optical_channel_names": list(self.optical_channel_names),
            "auxiliary_channel_names": {
                key: list(value) for key, value in self.auxiliary_channel_names.items()
            },
        }


@dataclass(frozen=True)
class LoadedBenchmarkSample:
    record: BenchmarkRecord
    sample: Mapping[str, Any]
    model_sample: Mapping[str, Any]
    optical_image: Image.Image
    reference_mask: Image.Image
    test_receipt: DemoTestAccessReceipt | None = None


DatasetFactory = Callable[..., BenchmarkDataset]


class BenchmarkCatalog:
    """index-only 筛选；只有 ``load`` 才通过现有 BenchmarkDataset 打开 shard。"""

    def __init__(
        self,
        binding: BenchmarkBinding,
        *,
        access_controller: DemoTestAccessController | None = None,
        dataset_factory: DatasetFactory = BenchmarkDataset,
    ) -> None:
        self.binding = binding
        self.root = binding.root
        self.access_controller = access_controller
        self.dataset_factory = dataset_factory
        self._datasets: dict[tuple[str, str], BenchmarkDataset] = {}
        self._dataset_indices: dict[tuple[str, str], dict[str, int]] = {}
        manifest_path = self.root / "manifest.json"
        index_path = self.root / "index.jsonl"
        if (
            first_symlink_component(self.root) is not None
            or not self.root.is_dir()
            or self.root.is_symlink()
            or not manifest_path.is_file()
            or not index_path.is_file()
        ):
            raise DemoCatalogError(f"Benchmark root/identity 文件非法：{self.root}")
        if sha256_file(manifest_path) != binding.manifest_sha256:
            raise DemoCatalogError("Benchmark manifest SHA-256 漂移")
        if sha256_file(index_path) != binding.index_sha256:
            raise DemoCatalogError("Benchmark index SHA-256 漂移")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            self.manifest.get("schema_version") != binding.schema_version
            or self.manifest.get("index_sha256") != binding.index_sha256
        ):
            raise DemoCatalogError("Benchmark manifest schema/index identity 不匹配")
        rows = load_index(index_path)
        records: list[BenchmarkRecord] = []
        seen: set[tuple[str, str]] = set()
        for index, row in enumerate(rows):
            split = str(row.get("split", ""))
            sample_id = str(row.get("sample_id", ""))
            source = str(row.get("source", ""))
            if split not in {"train", "val", "test"} or not sample_id or not source:
                raise DemoCatalogError(f"Benchmark index[{index}] split/sample/source 非法")
            key = (split, sample_id)
            if key in seen:
                raise DemoCatalogError(f"Benchmark 重复 sample：{key}")
            seen.add(key)
            ratio = float(row.get("foreground_ratio", -1.0))
            if not 0.0 <= ratio <= 1.0:
                raise DemoCatalogError(f"Benchmark foreground_ratio 非法：{key}")
            optical = row.get("optical")
            auxiliaries = row.get("auxiliaries")
            if not isinstance(optical, Mapping) or not isinstance(auxiliaries, Mapping):
                raise DemoCatalogError(f"Benchmark modality contract 非法：{key}")
            optical_channels = tuple(str(value) for value in optical.get("channel_names", ()))
            auxiliary_channels = {
                str(name): tuple(str(value) for value in contract.get("channel_names", ()))
                for name, contract in auxiliaries.items()
                if isinstance(name, str) and isinstance(contract, Mapping)
            }
            records.append(BenchmarkRecord(
                canonical_index=index,
                sample_id=sample_id,
                split=split,
                source=source,
                foreground_ratio=ratio,
                target_status="target" if ratio > 0.0 else "no_target",
                size=size_bin(ratio),
                available_modalities=("optical", *sorted(auxiliary_channels)),
                optical_channel_names=optical_channels,
                auxiliary_channel_names=auxiliary_channels,
                row=dict(row),
            ))
        expected_count = self.manifest.get("sample_count")
        if expected_count != len(records):
            raise DemoCatalogError(
                f"Benchmark sample_count 不一致：manifest={expected_count}, index={len(records)}"
            )
        self.records = tuple(records)
        self._by_key = {(row.split, row.sample_id): row for row in self.records}
        self._by_id: dict[str, list[BenchmarkRecord]] = {}
        for record in self.records:
            self._by_id.setdefault(record.sample_id, []).append(record)

    @property
    def identity(self) -> dict[str, str]:
        return self.binding.identity

    @property
    def sources(self) -> tuple[str, ...]:
        return tuple(sorted({row.source for row in self.records}))

    @property
    def modalities(self) -> tuple[str, ...]:
        return tuple(sorted({value for row in self.records for value in row.available_modalities}))

    def sources_for_split(self, split: str) -> tuple[str, ...]:
        return tuple(sorted({row.source for row in self.records if row.split == split}))

    @staticmethod
    def _matches(record: BenchmarkRecord, filters: BenchmarkFilter) -> bool:
        if record.split != filters.split:
            return False
        if filters.source is not None and record.source != filters.source:
            return False
        if filters.sample_id_query and filters.sample_id_query.strip() not in record.sample_id:
            return False
        if filters.target_status != "all" and record.target_status != filters.target_status:
            return False
        if filters.size != "all" and record.size != filters.size:
            return False
        return set(filters.modalities).issubset(record.available_modalities)

    def filtered(self, filters: BenchmarkFilter) -> tuple[BenchmarkRecord, ...]:
        """保持 canonical index 顺序；不读取模型输出或分数。"""

        return tuple(row for row in self.records if self._matches(row, filters))

    def locate(self, sample_id: str, *, split: str | None = None) -> BenchmarkRecord:
        matches = (
            [self._by_key[(split, sample_id)]]
            if split is not None and (split, sample_id) in self._by_key
            else [row for row in self._by_id.get(sample_id, ()) if split is None or row.split == split]
        )
        if len(matches) != 1:
            raise DemoCatalogError(
                f"sample_id 必须唯一定位，实际 {len(matches)}：split={split}, sample_id={sample_id}"
            )
        return matches[0]

    def navigate(
        self,
        filters: BenchmarkFilter,
        *,
        current_sample_id: str | None,
        delta: int,
    ) -> tuple[BenchmarkRecord, int, int]:
        rows = self.filtered(filters)
        if not rows:
            raise DemoCatalogError("当前筛选没有样本")
        current = next(
            (index for index, row in enumerate(rows) if row.sample_id == current_sample_id),
            0,
        )
        target = min(max(current + int(delta), 0), len(rows) - 1)
        return rows[target], target, len(rows)

    def random_record(
        self,
        filters: BenchmarkFilter,
        *,
        chooser: random.Random | random.SystemRandom | None = None,
    ) -> BenchmarkRecord:
        rows = self.filtered(filters)
        if not rows:
            raise DemoCatalogError("当前筛选没有可随机选择的样本")
        return (chooser or random.SystemRandom()).choice(rows)

    def _dataset(self, *, split: str, normalization: str) -> tuple[BenchmarkDataset, Mapping[str, int]]:
        key = (split, normalization)
        if key not in self._datasets:
            dataset = self.dataset_factory(
                self.root,
                split=split,
                auxiliary_policy="all",
                normalization=normalization,
            )
            indices = {str(row["sample_id"]): index for index, row in enumerate(dataset.rows)}
            if len(indices) != len(dataset.rows):
                raise DemoCatalogError(f"BenchmarkDataset {split} sample_id 重复")
            self._datasets[key] = dataset
            self._dataset_indices[key] = indices
        return self._datasets[key], self._dataset_indices[key]

    def load(
        self,
        record: BenchmarkRecord,
        *,
        action: str = "BROWSE",
        model_normalization: str = "none",
    ) -> LoadedBenchmarkSample:
        """test 时先发布 receipt；其后才允许 Dataset ``__getitem__`` 打开 HDF5。"""

        canonical = self.locate(record.sample_id, split=record.split)
        receipt: DemoTestAccessReceipt | None = None
        if canonical.split == "test":
            if self.access_controller is None:
                raise DemoCatalogError("test split 未配置独立 Demo 授权控制器")
            receipt = self.access_controller.issue(
                sample_id=canonical.sample_id,
                action=action,
            )
        raw_dataset, raw_indices = self._dataset(split=canonical.split, normalization="none")
        model_dataset, model_indices = self._dataset(
            split=canonical.split,
            normalization=model_normalization,
        )
        if canonical.sample_id not in raw_indices or canonical.sample_id not in model_indices:
            raise DemoCatalogError(f"BenchmarkDataset 缺少样本：{canonical.sample_id}")
        raw_sample = raw_dataset[raw_indices[canonical.sample_id]]
        model_sample = model_dataset[model_indices[canonical.sample_id]]
        if raw_sample["sample_id"] != model_sample["sample_id"]:
            raise DemoCatalogError("raw/model BenchmarkDataset 顺序或身份漂移")
        mask = np.asarray(raw_sample["mask"].cpu().numpy(), dtype=np.uint8)
        if mask.ndim != 3 or mask.shape[0] != 1 or not set(np.unique(mask)).issubset({0, 1}):
            raise DemoCatalogError("Reference mask 合同非法")
        return LoadedBenchmarkSample(
            record=canonical,
            sample=raw_sample,
            model_sample=model_sample,
            optical_image=render_optical(raw_sample),
            reference_mask=Image.fromarray(mask[0] * 255),
            test_receipt=receipt,
        )


@dataclass(frozen=True)
class FrozenEvaluationItem:
    ordinal: int
    evaluation_name: str
    baseline_record: Mapping[str, Any]
    counterfactual_records: Mapping[str, Mapping[str, Any]]
    root: Path

    @property
    def sample_id(self) -> str:
        return str(self.baseline_record["sample_id"])

    def asset_path(self, role: str, *, variant: str = "baseline_correct_mask") -> Path | None:
        record = self.counterfactual_records.get(variant)
        if record is None:
            return None
        value = record.get("assets", {}).get(role)
        if value is None:
            return None
        path = Path(os.path.abspath(self.root / str(value)))
        try:
            path.relative_to(self.root)
        except ValueError as error:
            raise DemoCatalogError("Frozen asset path 逃逸") from error
        if not path.is_file() or path.is_symlink() or first_symlink_component(path) is not None:
            raise DemoCatalogError(f"Frozen asset 非普通文件：{path}")
        return path

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": "Frozen Evaluation / Read Only",
            "evaluation_name": self.evaluation_name,
            "ordinal": self.ordinal,
            "sample_id": self.sample_id,
            "source": self.baseline_record.get("source"),
            "split": self.baseline_record.get("split"),
            "baseline_record_id": self.baseline_record.get("record_id"),
            "target_status": self.baseline_record.get("target_status"),
            "counterfactual_variants": list(self.counterfactual_records),
            "selection_unchanged": True,
            "formal_evaluation": False,
            "gallery_mutation_allowed": False,
        }


class FrozenEvaluationCatalog:
    """校验并只读呈现固定 100 baseline 与其反事实变体。"""

    def __init__(self, binding: FrozenEvaluationBinding, *, verify_payloads: bool = True) -> None:
        self.binding = binding
        self.name = binding.name
        self.root = binding.root
        manifest_path = self.root / "manifest.json"
        ledger_path = self.root / "SHA256SUMS.jsonl"
        if (
            first_symlink_component(self.root) is not None
            or not self.root.is_dir()
            or self.root.is_symlink()
            or not manifest_path.is_file()
            or not ledger_path.is_file()
        ):
            raise DemoCatalogError(f"Frozen Evaluation root 非法：{self.root}")
        if sha256_file(manifest_path) != binding.manifest_sha256:
            raise DemoCatalogError(f"{self.name} manifest SHA-256 漂移")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            self.manifest.get("schema_version")
            != "oa_groundrag.oa_grounded_eval_dev.manifest.v1"
            or self.manifest.get("development_only") is not True
            or self.manifest.get("sealed_test_accessed") is not False
            or self.manifest.get("formal_acceptance") is not False
        ):
            raise DemoCatalogError(f"{self.name} 科学状态合同不匹配")
        ledger = [
            json.loads(line)
            for line in ledger_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        ledger_manifest = self.manifest.get("ledger", {})
        if (
            sha256_file(ledger_path) != ledger_manifest.get("file_sha256")
            or len(ledger) != ledger_manifest.get("entry_count")
            or sha256_text(canonical_json(ledger)) != ledger_manifest.get("root_sha256")
        ):
            raise DemoCatalogError(f"{self.name} ledger identity 不匹配")
        if verify_payloads:
            for entry in ledger:
                relative = str(entry.get("path", ""))
                path = Path(os.path.abspath(self.root / relative))
                try:
                    path.relative_to(self.root)
                except ValueError as error:
                    raise DemoCatalogError("Frozen ledger path 逃逸") from error
                if (
                    not path.is_file()
                    or path.is_symlink()
                    or first_symlink_component(path) is not None
                    or path.stat().st_size != entry.get("size_bytes")
                    or sha256_file(path) != entry.get("sha256")
                ):
                    raise DemoCatalogError(f"Frozen ledger payload 漂移：{relative}")
        records_path = self.root / str(self.manifest["records"]["path"])
        if (
            sha256_file(records_path) != self.manifest["records"]["sha256"]
            or records_path.stat().st_size != self.manifest["records"]["size_bytes"]
        ):
            raise DemoCatalogError("Frozen records identity 不匹配")
        records = [
            json.loads(line)
            for line in records_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(records) != self.manifest["records"]["count"]:
            raise DemoCatalogError("Frozen records count 不匹配")
        by_id = {str(row.get("record_id")): row for row in records}
        if len(by_id) != len(records):
            raise DemoCatalogError("Frozen record_id 重复")
        selection = self.manifest.get("selection", {})
        baseline_ids = selection.get("baseline_record_ids")
        sample_ids = selection.get("ordered_sample_ids")
        if (
            selection.get("split") != "val"
            or selection.get("sample_count") != 100
            or not isinstance(baseline_ids, list)
            or not isinstance(sample_ids, list)
            or len(baseline_ids) != 100
            or len(sample_ids) != 100
            or len(set(baseline_ids)) != 100
            or len(set(sample_ids)) != 100
        ):
            raise DemoCatalogError("Frozen selection 必须是唯一有序的 100 条 val baseline")
        baselines: list[Mapping[str, Any]] = []
        for record_id, sample_id in zip(baseline_ids, sample_ids, strict=True):
            record = by_id.get(str(record_id))
            counterfactual = (
                None
                if record is None
                else record.get("program_facts", {}).get("counterfactual")
            )
            counterfactual_ok = (
                record is not None
                and (
                    (
                        record.get("target_status") == "target_present"
                        and isinstance(counterfactual, Mapping)
                        and counterfactual.get("kind") == "baseline_correct_mask"
                    )
                    or (
                        record.get("target_status") == "no_target"
                        and counterfactual is None
                    )
                )
            )
            if (
                record is None
                or record.get("sample_id") != sample_id
                or record.get("split") != "val"
                or not counterfactual_ok
            ):
                raise DemoCatalogError(f"Frozen baseline 身份或顺序漂移：{record_id}")
            baselines.append(record)
        groups_path = self.root / "counterfactual_groups.jsonl"
        groups = [
            json.loads(line)
            for line in groups_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(groups) != self.manifest.get("counterfactuals", {}).get("group_count"):
            raise DemoCatalogError("Frozen counterfactual group count 不匹配")
        group_by_baseline = {str(row["baseline_record_id"]): row for row in groups}
        variants: dict[str, dict[str, Mapping[str, Any]]] = {}
        for baseline in baselines:
            baseline_id = str(baseline["record_id"])
            current = {"baseline_correct_mask": baseline}
            group = group_by_baseline.get(baseline_id)
            if group is not None:
                for kind, record_id in group.get("variants", {}).items():
                    record = by_id.get(str(record_id))
                    if record is None:
                        raise DemoCatalogError(f"Frozen counterfactual record 缺失：{record_id}")
                    current[str(kind)] = record
            variants[baseline_id] = current
        self._baselines = tuple(baselines)
        self._variants = variants

    def __len__(self) -> int:
        return len(self._baselines)

    def item(self, ordinal: int) -> FrozenEvaluationItem:
        if isinstance(ordinal, bool) or not isinstance(ordinal, int):
            raise DemoCatalogError("Frozen ordinal 必须是整数")
        index = min(max(ordinal, 0), len(self._baselines) - 1)
        baseline = self._baselines[index]
        return FrozenEvaluationItem(
            ordinal=index,
            evaluation_name=self.name,
            baseline_record=baseline,
            counterfactual_records=self._variants[str(baseline["record_id"])],
            root=self.root,
        )
