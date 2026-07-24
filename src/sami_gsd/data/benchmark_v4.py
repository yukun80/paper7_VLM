"""P1 HDF5 Canonical Benchmark v4 的自包含物化构建主逻辑。

用途：从已审计 HDF5 source index 构建不可覆盖的 Small benchmark。
推荐入口：由 ``sami-gsd benchmark build`` 调用本模块的 ``build_benchmark_v4``。
输入：严格 v4 config、只读 datasets root、schema/authority binding。
输出：HDF5 byte copies、source/parent/split indexes、channel catalog 与总 manifest。
写行为：仅写全新 staging，逐文件校验后原子发布；source 始终只读。
阶段：P1；本模块不包含模型、语言、bbox 或未来候选逻辑。
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

from sami_gsd.utilities.artifacts import (
    atomic_copy_file,
    atomic_output_directory,
    atomic_write_json,
    canonical_json_bytes,
    reject_non_finite,
    sha256_bytes,
    sha256_file,
)


BUILDER_PROTOCOL = "sami_hdf5_benchmark_v4_builder_v1"
NORMALIZATION_PROTOCOL = "zscore_canonical_train_valid_pixels_v1"
SPLIT_PROJECTION_PROTOCOL = "sami_native_split_projection_v1"
BENCHMARK_RELATIVE_PATH = "sami_landslide_hdf5_v4/small"

_SOURCE_INDEX_PATH = "indexes/source_records.jsonl"
_PARENT_INDEX_PATH = "indexes/canonical_parents.jsonl"
_SPLIT_PATHS = {
    "train": "indexes/train.jsonl",
    "val": "indexes/val.jsonl",
    "test": "indexes/test.jsonl",
}
_BUILD_CONFIG_PATH = "manifests/build_config.json"
_SOURCE_REGISTRY_PATH = "manifests/source_registry.json"
_CHANNEL_CATALOG_PATH = "manifests/channel_catalog.json"
_MATERIALIZATION_INDEX_PATH = "manifests/materialized_assets.jsonl"
_NORMALIZATION_PATH = "manifests/normalization.json"
_STATISTICS_PATH = "reports/statistics.json"
_DUPLICATE_RISK_PATH = "reports/duplicate_risk.json"
_ELIGIBILITY_PATH = "reports/evaluation_eligibility.json"
_MANIFEST_PATH = "manifests/benchmark_manifest.json"
_SPLIT_KEYS = ("train", "val", "test")
_ASSURANCE_KEYS = (
    "verified_group_isolated",
    "source_declared_unverified",
    "train_only",
)
_ELIGIBILITY_KEYS = ("strict", "exploratory", "train_only")


class BenchmarkV4BuildError(ValueError):
    """Benchmark v4 输入、合同或 lineage 不满足冻结规则。"""


def _complete_counts(
    values: Mapping[str, int],
    keys: Sequence[str],
) -> dict[str, int]:
    """显式保留零人口，避免把 unavailable 误读为未统计。"""

    return {key: int(values.get(key, 0)) for key in keys}


def _assert_source_population(
    source: Mapping[str, Any],
    *,
    observed_pair_count: int,
    observed_split_counts: Mapping[str, int],
    observed_positive_count: int,
    observed_no_target_count: int,
) -> None:
    """将现场人口与绑定 inventory 比较，不把审计数字硬编码进算法。"""

    source_key = str(source["source_key"])
    expected = {
        "pair_count": int(source["expected_pair_count"]),
        "split_counts": dict(source["expected_source_split_counts"]),
        "positive_count": int(source["expected_positive_count"]),
        "no_target_count": int(source["expected_no_target_count"]),
    }
    observed = {
        "pair_count": observed_pair_count,
        "split_counts": dict(sorted(observed_split_counts.items())),
        "positive_count": observed_positive_count,
        "no_target_count": observed_no_target_count,
    }
    if observed != expected:
        raise BenchmarkV4BuildError(
            f"live source population drifts from bound inventory for "
            f"{source_key}: expected={expected}, observed={observed}"
        )


def _as_mapping(value: Any, *, context: str) -> dict[str, Any]:
    """将合同对象投影为纯 JSON mapping，不接受隐式字符串化。"""

    if isinstance(value, Mapping):
        payload = dict(value)
    elif callable(getattr(value, "to_mapping", None)):
        payload = value.to_mapping()
    elif callable(getattr(value, "model_dump", None)):
        payload = value.model_dump(mode="json")
    else:
        raise TypeError(f"{context} must expose to_mapping() or model_dump(mode='json')")
    if not isinstance(payload, Mapping):
        raise TypeError(f"{context} projection must be a mapping")
    result = dict(payload)
    reject_non_finite(result)
    return result


def _portable_path(value: str, *, prefix: str | None = None) -> str:
    """验证 benchmark 内部或 datasets 命名空间中的 POSIX 逻辑路径。"""

    if not value or "\\" in value:
        raise BenchmarkV4BuildError("logical path must be a non-empty POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in value.split("/")):
        raise BenchmarkV4BuildError(f"non-portable logical path: {value!r}")
    if prefix is not None and not value.startswith(prefix):
        raise BenchmarkV4BuildError(f"logical path must start with {prefix!r}: {value!r}")
    return value


def _reject_machine_paths(value: Any, *, location: str = "$") -> None:
    """拒绝规范 payload 中泄漏的机器路径；HTTPS provenance 不受影响。"""

    if isinstance(value, str):
        field_name = location.rsplit(".", maxsplit=1)[-1]
        is_hdf5_key = field_name in {
            "dataset_key",
            "valid_mask_key",
            "pixel_valid_key",
            "channel_valid_key",
        }
        if (
            (value.startswith("/") and not is_hdf5_key)
            or value.startswith("file://")
            or "/home/" in value
            or "\\home\\" in value
        ):
            raise BenchmarkV4BuildError(f"machine path leaked into canonical payload at {location}")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_machine_paths(item, location=f"{location}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            _reject_machine_paths(item, location=f"{location}[{index}]")


def _record_without_hash(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result.pop("record_sha256", None)
    return result


def _verify_record_hash(payload: Mapping[str, Any], *, context: str) -> None:
    expected = payload.get("record_sha256")
    if not isinstance(expected, str):
        raise BenchmarkV4BuildError(f"{context} is missing record_sha256")
    actual = sha256_bytes(canonical_json_bytes(_record_without_hash(payload)))
    if actual != expected:
        raise BenchmarkV4BuildError(
            f"{context} record hash mismatch: expected {expected}, recomputed {actual}"
        )


def _source_record_payload(observation: Any) -> dict[str, Any]:
    """从 reader 的单样本 observation 获取唯一规范 source payload。"""

    projector = getattr(observation, "to_source_record_payload", None)
    if not callable(projector):
        raise TypeError("SourceObservation must expose to_source_record_payload()")
    payload = _contract_from_mapping(
        "Hdf5SourceRecordV1",
        _as_mapping(projector(), context="source record"),
    )
    _reject_machine_paths(payload)
    _verify_record_hash(payload, context="source record")
    return payload


def _array(observation: Any, name: str) -> Any:
    if not hasattr(observation, name):
        raise TypeError(f"SourceObservation is missing required array {name!r}")
    return getattr(observation, name)


def _config_sources(config: Any, payload: Mapping[str, Any]) -> tuple[Any, ...]:
    values = getattr(config, "sources", None)
    if values is None:
        values = payload.get("sources")
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        raise BenchmarkV4BuildError("v4 config sources must be an ordered sequence")
    return tuple(values)


def _benchmark_relative_path(config: Any, payload: Mapping[str, Any]) -> str:
    value = getattr(config, "benchmark_relative_path", None)
    if value is None:
        value = payload.get("benchmark_relative_path")
    if value is None and isinstance(payload.get("benchmark"), Mapping):
        value = payload["benchmark"].get("relative_path")
    if value != BENCHMARK_RELATIVE_PATH:
        raise BenchmarkV4BuildError(
            f"benchmark_relative_path must be exactly {BENCHMARK_RELATIVE_PATH!r}"
        )
    return _portable_path(str(value))


def _contract_from_mapping(class_name: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    """用 v4 合同验证 mapping；保留 duck-typed 合同实现的窄边界。"""

    from sami_gsd.contracts import benchmark_v4 as contracts

    contract_class = getattr(contracts, class_name)
    if callable(getattr(contract_class, "from_mapping", None)):
        instance = contract_class.from_mapping(payload)
    elif callable(getattr(contract_class, "model_validate", None)):
        instance = contract_class.model_validate(payload)
    else:
        instance = contract_class(**dict(payload))
    return _as_mapping(instance, context=class_name)


def _seal_parent(payload: Mapping[str, Any]) -> dict[str, Any]:
    from sami_gsd.contracts.benchmark_v4 import seal_canonical_parent

    return _as_mapping(seal_canonical_parent(payload), context="CanonicalParentV4")


def _parent_id(source_key: str, source_sample_id: str) -> str:
    identity = {
        "schema_version": "sami_canonical_parent_identity_v4",
        "source_key": source_key,
        "source_sample_id": source_sample_id,
    }
    return f"p4-{sha256_bytes(canonical_json_bytes(identity))}"


@dataclass
class _Moments:
    """可组合的 float64 population moments；每次只保留一个 channel 的标量。"""

    count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    minimum: float | None = None
    maximum: float | None = None

    def update(self, values: Any) -> None:
        import numpy as np

        flat = np.asarray(values, dtype=np.float64).reshape(-1)
        if flat.size == 0:
            return
        if not bool(np.isfinite(flat).all()):
            raise BenchmarkV4BuildError("non-finite value found in a valid normalization pixel")
        batch_count = int(flat.size)
        batch_mean = float(flat.mean(dtype=np.float64))
        centered = flat - batch_mean
        batch_m2 = float(np.dot(centered, centered))
        batch_min = float(flat.min())
        batch_max = float(flat.max())
        if self.count == 0:
            self.count = batch_count
            self.mean = batch_mean
            self.m2 = batch_m2
            self.minimum = batch_min
            self.maximum = batch_max
            return
        total = self.count + batch_count
        delta = batch_mean - self.mean
        self.m2 += batch_m2 + delta * delta * self.count * batch_count / total
        self.mean += delta * batch_count / total
        self.count = total
        assert self.minimum is not None and self.maximum is not None
        self.minimum = min(self.minimum, batch_min)
        self.maximum = max(self.maximum, batch_max)

    def payload(
        self,
        *,
        source_key: str,
        channel_key: str,
        channel_index: int,
    ) -> dict[str, Any]:
        if self.count <= 0 or self.minimum is None or self.maximum is None:
            raise BenchmarkV4BuildError(
                f"normalization population is empty for {source_key}:{channel_key}"
            )
        variance = max(0.0, self.m2 / self.count)
        return {
            "source_key": source_key,
            "channel_key": channel_key,
            "channel_index": channel_index,
            "valid_pixel_count": self.count,
            "mean": self.mean,
            "std": math.sqrt(variance),
        }


class _NormalizationAccumulator:
    def __init__(self) -> None:
        self._moments: dict[tuple[str, str], _Moments] = {}
        self._channel_indices: dict[tuple[str, str], int] = {}
        self._population = hashlib.sha256()
        self.parent_count = 0

    def update(self, observation: Any, source_record: Mapping[str, Any]) -> None:
        import numpy as np

        if source_record["canonical_split"] != "train":
            return
        image_values = np.asarray(_array(observation, "image_values"))
        channel_valid = np.asarray(_array(observation, "channel_valid"), dtype=bool)
        pixel_valid = np.asarray(_array(observation, "pixel_valid"), dtype=bool)
        channels = source_record["channels"]
        if image_values.ndim != 3:
            raise BenchmarkV4BuildError("image_values must have CHW rank 3")
        if channel_valid.shape != (image_values.shape[0],):
            raise BenchmarkV4BuildError("channel_valid shape does not match image channel count")
        if pixel_valid.shape != image_values.shape:
            raise BenchmarkV4BuildError("pixel_valid must have the same CHW shape as image_values")
        if len(channels) != image_values.shape[0]:
            raise BenchmarkV4BuildError("channel descriptor count does not match image channel count")

        population_row = {
            "source_key": source_record["source_key"],
            "source_sample_id": source_record["source_sample_id"],
            "source_record_sha256": source_record["record_sha256"],
        }
        self._population.update(canonical_json_bytes(population_row))
        self.parent_count += 1

        for index, descriptor in enumerate(channels):
            if int(descriptor["index"]) != index:
                raise BenchmarkV4BuildError("channel descriptors must use contiguous source indices")
            if not bool(channel_valid[index]):
                continue
            valid_values = image_values[index][pixel_valid[index]]
            key = (str(source_record["source_key"]), str(descriptor["channel_key"]))
            previous_index = self._channel_indices.setdefault(key, index)
            if previous_index != index:
                raise BenchmarkV4BuildError(
                    f"channel index drift for {key[0]}:{key[1]}"
                )
            self._moments.setdefault(key, _Moments()).update(valid_values)

    def finalize(self, *, source_index_sha256: str) -> dict[str, Any]:
        from sami_gsd.contracts.benchmark_v4 import (
            NormalizationChannelStatisticV1,
            NormalizationManifestV1,
        )

        rows = [
            moments.payload(
                source_key=source_key,
                channel_key=channel_key,
                channel_index=self._channel_indices[(source_key, channel_key)],
            )
            for (source_key, channel_key), moments in sorted(self._moments.items())
        ]
        aggregate_sha256 = sha256_bytes(canonical_json_bytes(rows))
        manifest = NormalizationManifestV1(
            population_split="train",
            source_index_sha256=source_index_sha256,
            population_sha256=self._population.hexdigest(),
            statistics=tuple(
                NormalizationChannelStatisticV1(**row) for row in rows
            ),
            aggregate_sha256=aggregate_sha256,
        )
        return _as_mapping(manifest, context="NormalizationManifestV1")


class _StatisticsAccumulator:
    def __init__(self) -> None:
        self.source_counts: Counter[str] = Counter()
        self.split_counts: Counter[str] = Counter()
        self.assurance_counts: Counter[str] = Counter()
        self.eligibility_counts: Counter[str] = Counter()
        self.positive_counts: Counter[str] = Counter()
        self.no_target_counts: Counter[str] = Counter()
        self.valid_pixel_counts: Counter[str] = Counter()
        self.positive_pixel_counts: Counter[str] = Counter()
        self.group_completeness_counts: Counter[str] = Counter()
        self.duplicate_evidence_counts: Counter[str] = Counter()

    def update(self, observation: Any, source_record: Mapping[str, Any]) -> None:
        import numpy as np

        source_key = str(source_record["source_key"])
        target = np.asarray(_array(observation, "mask_values"))
        target_valid = np.asarray(_array(observation, "target_valid"), dtype=bool)
        if target.shape != target_valid.shape or target.ndim != 2:
            raise BenchmarkV4BuildError("mask_values and target_valid must be matching HW arrays")
        valid_count = int(np.count_nonzero(target_valid))
        if valid_count <= 0:
            raise BenchmarkV4BuildError(
                f"target-valid population is empty for {source_key}:{source_record['source_sample_id']}"
            )
        valid_target = target[target_valid]
        if not bool(np.isfinite(valid_target).all()):
            raise BenchmarkV4BuildError("non-finite value found in target-valid mask pixels")
        if not bool(np.isin(valid_target, (0, 1)).all()):
            raise BenchmarkV4BuildError("target-valid mask pixels must be binary")
        positive_pixels = int(np.count_nonzero(valid_target))

        self.source_counts[source_key] += 1
        self.split_counts[str(source_record["canonical_split"])] += 1
        self.assurance_counts[str(source_record["split_assurance"])] += 1
        self.eligibility_counts[str(source_record["evaluation_eligibility"])] += 1
        self.valid_pixel_counts[source_key] += valid_count
        self.positive_pixel_counts[source_key] += positive_pixels
        if positive_pixels:
            self.positive_counts[source_key] += 1
        else:
            self.no_target_counts[source_key] += 1
        group = source_record.get("group") or {}
        duplicate = source_record.get("duplicate_component") or {}
        self.group_completeness_counts[str(group.get("completeness", "unavailable"))] += 1
        self.duplicate_evidence_counts[
            str(duplicate.get("evidence_level", "unavailable"))
        ] += 1

    def statistics_payload(
        self,
        *,
        normalization_binding_sha256: str,
    ) -> dict[str, Any]:
        source_rows: dict[str, dict[str, int]] = {}
        for source_key in sorted(self.source_counts):
            source_rows[source_key] = {
                "parent_count": self.source_counts[source_key],
                "positive_count": self.positive_counts[source_key],
                "no_target_count": self.no_target_counts[source_key],
                "target_valid_pixel_count": self.valid_pixel_counts[source_key],
                "positive_target_pixel_count": self.positive_pixel_counts[source_key],
            }
        parent_count = sum(self.source_counts.values())
        core = {
            "source_counts": source_rows,
            "source_record_count": parent_count,
            "parent_count": parent_count,
            "positive_count": sum(self.positive_counts.values()),
            "no_target_count": sum(self.no_target_counts.values()),
            "split_counts": _complete_counts(self.split_counts, _SPLIT_KEYS),
            "assurance_counts": _complete_counts(
                self.assurance_counts,
                _ASSURANCE_KEYS,
            ),
            "eligibility_counts": _complete_counts(
                self.eligibility_counts,
                _ELIGIBILITY_KEYS,
            ),
            "strict_population": self.eligibility_counts.get("strict", 0),
            "strict_generalization_status": (
                "unavailable" if self.eligibility_counts.get("strict", 0) == 0 else "available"
            ),
            "normalization_binding_sha256": normalization_binding_sha256,
        }
        core["aggregate_sha256"] = sha256_bytes(canonical_json_bytes(core))
        return _contract_from_mapping("BenchmarkStatisticsV4", core)

    def risk_payload(self, source_configs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        source_risks = []
        for source in source_configs:
            source_risks.append(
                {
                    "source_key": source["source_key"],
                    "ingestion_status": source["ingestion_status"],
                    "risks": list(source.get("risks", ())),
                    "known_location_cross_split_conflict_count": source.get(
                        "known_location_cross_split_conflict_count"
                    ),
                }
            )
        return {
            "schema_version": "sami_benchmark_v4_duplicate_risk_report_v1",
            "protocol": "reported_incomplete_group_and_duplicate_evidence_v1",
            "group_completeness_counts": dict(sorted(self.group_completeness_counts.items())),
            "duplicate_evidence_counts": dict(sorted(self.duplicate_evidence_counts.items())),
            "training_blocked_by_incomplete_group_or_duplicate_evidence": False,
            "sources": source_risks,
        }

    def eligibility_payload(self) -> dict[str, Any]:
        strict_count = self.eligibility_counts.get("strict", 0)
        return {
            "schema_version": "sami_benchmark_v4_evaluation_eligibility_report_v1",
            "protocol": "disjoint_strict_exploratory_train_only_v1",
            "populations": _complete_counts(
                self.eligibility_counts,
                _ELIGIBILITY_KEYS,
            ),
            "strict_generalization_status": "unavailable" if strict_count == 0 else "available",
            "strict_population": strict_count,
            "overlap_count": 0,
        }


class _JsonlWriter:
    """staging 内的确定性 JSONL writer；失败由外层原子目录整体回收。"""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._handle: BinaryIO = path.open("wb")
        self.line_count = 0

    def write(self, payload: Mapping[str, Any]) -> None:
        self._handle.write(canonical_json_bytes(payload))
        self.line_count += 1

    def close(self) -> None:
        if self._handle.closed:
            return
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._handle.close()

    def __enter__(self) -> "_JsonlWriter":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value!r}")

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                payload = json.loads(line, parse_constant=reject_constant)
            except (json.JSONDecodeError, ValueError) as error:
                raise BenchmarkV4BuildError(f"invalid JSONL at {path}:{line_number}: {error}") from error
            if not isinstance(payload, dict):
                raise BenchmarkV4BuildError(f"JSONL row must be an object at {path}:{line_number}")
            yield payload


def _registered_rgb(
    source_config: Mapping[str, Any],
    *,
    normalization_binding: str,
) -> list[dict[str, Any]]:
    registered = source_config.get("registered_rgb")
    if registered is None:
        return []
    if not isinstance(registered, Mapping):
        raise BenchmarkV4BuildError("registered_rgb must be null or a mapping")
    payload = {
        "schema_version": "sami_registered_rgb_view_v1",
        "view_id": "registered_rgb",
        "role": "rgb",
        "source_indices": list(registered["source_indices"]),
        "channel_keys": list(registered["channel_keys"]),
        "normalization_binding": normalization_binding,
        "mapping_evidence": registered["mapping_evidence"],
    }
    return [_contract_from_mapping("RegisteredRGBViewV1", payload)]


def _parent_from_source(
    source_record: Mapping[str, Any],
    source_config: Mapping[str, Any],
    *,
    normalization_binding: str,
) -> dict[str, Any]:
    payload = {
        "schema_version": "sami_canonical_parent_v4",
        "parent_id": _parent_id(
            str(source_record["source_key"]),
            str(source_record["source_sample_id"]),
        ),
        "source_key": source_record["source_key"],
        "source_sample_id": source_record["source_sample_id"],
        "canonical_split": source_record["canonical_split"],
        "split_assurance": source_record["split_assurance"],
        "evaluation_eligibility": source_record["evaluation_eligibility"],
        "channels": source_record["channels"],
        "image_ref": source_record["image"],
        "mask_ref": source_record["mask"],
        "validity_ref": source_record["validity"],
        "registered_views": _registered_rgb(
            source_config,
            normalization_binding=normalization_binding,
        ),
        "group": source_record["group"],
        "source_record_sha256": source_record["record_sha256"],
    }
    parent = _seal_parent(payload)
    _reject_machine_paths(parent)
    _verify_record_hash(parent, context="canonical parent")
    return parent


def _split_projection(parent: Mapping[str, Any], *, all_line_number: int) -> dict[str, Any]:
    payload = {
        "schema_version": "sami_benchmark_v4_split_projection_v1",
        "protocol": SPLIT_PROJECTION_PROTOCOL,
        "parent_id": parent["parent_id"],
        "canonical_split": parent["canonical_split"],
        "all_line_number": all_line_number,
        "parent_record_sha256": parent["record_sha256"],
    }
    payload["record_sha256"] = sha256_bytes(canonical_json_bytes(payload))
    return payload


def _artifact_binding(staging: Path, relative_path: str) -> dict[str, Any]:
    relative_path = _portable_path(relative_path)
    path = staging / relative_path
    if not path.is_file():
        raise BenchmarkV4BuildError(f"missing staged artifact: {relative_path}")
    return {
        "path": relative_path,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _local_benchmark_asset_path(benchmark_logical_path: str) -> str:
    """Convert the portable global Benchmark path into a package-relative path."""

    prefix = PurePosixPath("benchmark") / BENCHMARK_RELATIVE_PATH
    logical = PurePosixPath(_portable_path(benchmark_logical_path, prefix="benchmark/"))
    try:
        relative = logical.relative_to(prefix)
    except ValueError as error:
        raise BenchmarkV4BuildError(
            f"asset path is outside Benchmark v4 Small: {benchmark_logical_path}"
        ) from error
    if not relative.parts or relative.parts[0] != "assets":
        raise BenchmarkV4BuildError("materialized HDF5 must be stored below assets/")
    return _portable_path(relative.as_posix(), prefix="assets/")


def _materialization_preflight(
    source_configs: Sequence[Any],
    *,
    datasets_root: Path,
    output_dir: Path,
) -> tuple[int, int]:
    """Stat every selected source file and reject insufficient staging capacity."""

    from sami_gsd.data.hdf5_sources_v4 import iter_source_asset_paths

    unique: dict[tuple[str, str], tuple[Path, int]] = {}
    for source in source_configs:
        source_mapping = _as_mapping(source, context="source preflight")
        if source_mapping["ingestion_status"] != "ready":
            continue
        source_key = str(source_mapping["source_key"])
        for _role, logical, physical in iter_source_asset_paths(
            source_mapping,
            datasets_root,
        ):
            resolved = physical.resolve()
            size = resolved.stat().st_size
            key = (source_key, logical)
            previous = unique.setdefault(key, (resolved, size))
            if previous != (resolved, size):
                raise BenchmarkV4BuildError(f"source size drift during preflight: {resolved}")
    required = sum(size for _path, size in unique.values())
    if not unique or required <= 0:
        raise BenchmarkV4BuildError("materialization preflight found no HDF5 assets")
    capacity_root = output_dir.parent
    while not capacity_root.exists():
        capacity_root = capacity_root.parent
    free = shutil.disk_usage(capacity_root).free
    margin = max(64 * 1024 * 1024, min(1024 * 1024 * 1024, required // 20))
    if free < required + margin:
        raise BenchmarkV4BuildError(
            "insufficient free space for atomic Benchmark materialization: "
            f"required_bytes={required}, margin_bytes={margin}, free_bytes={free}"
        )
    return len(unique), required


def _materialize_observation(
    observation: Any,
    source_record: Mapping[str, Any],
    *,
    staging: Path,
    writer: _JsonlWriter,
    materialized: dict[str, dict[str, Any]],
) -> None:
    """Copy one image/mask pair exactly once and append its strict ledger rows."""

    for role, source_path_attribute in (
        ("image", "image_source_path"),
        ("mask", "mask_source_path"),
    ):
        reference = source_record[role]
        logical = str(reference["benchmark_logical_path"])
        relative = _local_benchmark_asset_path(logical)
        row = {
            "schema_version": "sami_materialized_hdf5_asset_v1",
            "role": role,
            "source_key": source_record["source_key"],
            "source_logical_path": reference["source_logical_path"],
            "benchmark_logical_path": logical,
            "sha256": reference["sha256"],
            "size_bytes": reference["size_bytes"],
        }
        existing = materialized.get(relative)
        if existing is not None:
            if existing != row:
                raise BenchmarkV4BuildError(f"materialized asset collision: {relative}")
            continue
        source_path = getattr(observation, source_path_attribute, None)
        if not isinstance(source_path, Path):
            raise TypeError(f"SourceObservation lacks {source_path_attribute}")
        atomic_copy_file(
            source_path,
            staging / relative,
            expected_sha256=str(reference["sha256"]),
            expected_size_bytes=int(reference["size_bytes"]),
        )
        materialized[relative] = row
        writer.write(row)


def _channel_catalog(source_configs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Build the explicit channel vocabulary consumed by later dense loaders."""

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    invariants: dict[str, tuple[Any, ...]] = {}
    for source in source_configs:
        if source["ingestion_status"] != "ready":
            continue
        for descriptor in source["channels"]:
            channel_key = str(descriptor["channel_key"])
            invariant = (
                descriptor["modality_family"],
                descriptor["wavelength_nm"],
                descriptor["wavelength_known"],
                descriptor["gsd_m"],
                descriptor["gsd_known"],
            )
            previous = invariants.setdefault(channel_key, invariant)
            if previous != invariant:
                raise BenchmarkV4BuildError(
                    f"global channel meaning drifts for {channel_key}"
                )
            grouped[channel_key].append(
                {
                    "source_key": source["source_key"],
                    "source_index": descriptor["index"],
                    "display_name": descriptor["display_name"],
                    "physical_unit": descriptor["physical_unit"],
                    "normalization": descriptor["normalization"],
                    "validity_source": descriptor["validity_source"],
                    "descriptor_sha256": sha256_bytes(
                        canonical_json_bytes(descriptor)
                    ),
                }
            )
    entries = []
    for token_index, channel_key in enumerate(sorted(grouped)):
        invariant = invariants[channel_key]
        entries.append(
            {
                "channel_token": token_index,
                "channel_key": channel_key,
                "modality_family": invariant[0],
                "wavelength_nm": invariant[1],
                "wavelength_known": invariant[2],
                "gsd_m": invariant[3],
                "gsd_known": invariant[4],
                "source_bindings": sorted(
                    grouped[channel_key],
                    key=lambda row: (str(row["source_key"]), int(row["source_index"])),
                ),
            }
        )
    if not entries:
        raise BenchmarkV4BuildError("channel catalog cannot be empty")
    payload = {
        "schema_version": "sami_channel_catalog_v1",
        "protocol": "explicit_channel_semantics_v1",
        "ordering_rule": "channel_token_is_lexicographic_channel_key_not_tensor_order",
        "entries": entries,
        "aggregate_sha256": sha256_bytes(canonical_json_bytes(entries)),
    }
    _reject_machine_paths(payload)
    return payload


def _schema_bindings(schemas_root: Path) -> list[dict[str, Any]]:
    names = (
        "hdf5_source_record_v1.schema.json",
        "canonical_parent_v4.schema.json",
        "benchmark_manifest_v4.schema.json",
        "benchmark_statistics_v4.schema.json",
        "benchmark_validation_report_v4.schema.json",
        "channel_catalog_v1.schema.json",
        "materialized_hdf5_asset_v1.schema.json",
    )
    bindings = []
    for name in names:
        path = schemas_root / name
        if not path.is_file():
            raise FileNotFoundError(f"required Benchmark v4 schema is missing: {path}")
        bindings.append(
            {
                "path": f"schemas/{name}",
                "sha256": sha256_file(path),
            }
        )
    return bindings


def _authority_binding(path: Path, *, repository_root: Path) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        logical = resolved.relative_to(repository_root.resolve()).as_posix()
    except ValueError as error:
        raise BenchmarkV4BuildError(f"authority binding must be inside repository: {path}") from error
    return {"path": _portable_path(logical), "sha256": sha256_file(resolved)}


def _build_source_registry(
    *,
    source_configs: Sequence[Mapping[str, Any]],
    config_binding: Mapping[str, Any],
    config_payload_sha256: str,
    source_contract_binding: Mapping[str, Any],
    source_inventory_binding: Mapping[str, Any],
    schema_bindings: Sequence[Mapping[str, Any]],
    channel_catalog_binding: Mapping[str, Any],
) -> dict[str, Any]:
    payload = {
        "schema_version": "sami_benchmark_v4_source_registry_v1",
        "protocol": BUILDER_PROTOCOL,
        "config": dict(config_binding),
        "config_payload_sha256": config_payload_sha256,
        "source_contract": dict(source_contract_binding),
        "source_inventory": dict(source_inventory_binding),
        "schemas": [dict(binding) for binding in schema_bindings],
        "channel_catalog": dict(channel_catalog_binding),
        "sources": list(source_configs),
    }
    _reject_machine_paths(payload)
    payload["record_sha256"] = sha256_bytes(canonical_json_bytes(payload))
    return payload


def _manifest_payload(
    *,
    config_sha256: str,
    artifact_bindings: Sequence[Mapping[str, Any]],
    source_contract_sha256: str,
    source_inventory_sha256: str,
    source_registry_sha256: str,
    statistics: Mapping[str, Any],
    normalization_binding: str,
    channel_catalog_sha256: str,
    materialization_index_sha256: str,
    materialized_asset_count: int,
    materialized_size_bytes: int,
) -> dict[str, Any]:
    binding_rows = sorted(
        (dict(binding) for binding in artifact_bindings),
        key=lambda row: str(row["path"]),
    )
    binding_map = {
        str(binding["path"]): str(binding["sha256"]) for binding in binding_rows
    }
    aggregate_sha256 = sha256_bytes(canonical_json_bytes(binding_map))
    payload = {
        "schema_version": "sami_benchmark_manifest_v4",
        "protocol": "sami_hdf5_materialized_copy_v1",
        "benchmark_name": "SAMI Landslide HDF5 Benchmark v4",
        "mode": "small",
        "benchmark_relative_path": BENCHMARK_RELATIVE_PATH,
        "config_sha256": config_sha256,
        "source_contract_sha256": source_contract_sha256,
        "source_inventory_sha256": source_inventory_sha256,
        "source_registry_sha256": source_registry_sha256,
        "artifact_bindings": binding_map,
        "normalization_binding_sha256": normalization_binding,
        "channel_catalog_sha256": channel_catalog_sha256,
        "materialization_index_sha256": materialization_index_sha256,
        "materialized_asset_count": materialized_asset_count,
        "materialized_size_bytes": materialized_size_bytes,
        "source_record_count": statistics["source_record_count"],
        "parent_count": statistics["parent_count"],
        "split_counts": statistics["split_counts"],
        "assurance_counts": statistics["assurance_counts"],
        "eligibility_counts": statistics["eligibility_counts"],
        "strict_generalization_status": statistics["strict_generalization_status"],
        "aggregate_sha256": aggregate_sha256,
    }
    return _contract_from_mapping("BenchmarkManifestV4", payload)


def build_benchmark_v4(
    config: Any,
    *,
    config_path: Path,
    datasets_root: Path,
    benchmark_root: Path,
    schemas_root: Path,
    source_contract_path: Path | None = None,
    source_inventory_path: Path | None = None,
) -> dict[str, Any]:
    """构建全新、自包含的 Benchmark v4 Small。

    本函数会读取全部 owner 指定 source，因此只能由项目负责人运行。每次 observation
    处理完成后即释放数组；source HDF5 会逐字节复制并校验到 benchmark。
    """

    from sami_gsd.data.hdf5_sources_v4 import iter_source_observations

    config_payload = _as_mapping(config, context="BenchmarkV4Config")
    _reject_machine_paths(config_payload)
    relative_path = _benchmark_relative_path(config, config_payload)
    datasets_root = datasets_root.resolve()
    benchmark_root = benchmark_root.resolve()
    schemas_root = schemas_root.resolve()
    if not datasets_root.is_dir():
        raise FileNotFoundError(f"datasets root does not exist: {datasets_root}")
    if not schemas_root.is_dir():
        raise FileNotFoundError(f"schemas root does not exist: {schemas_root}")

    repository_root = schemas_root.parent
    source_contract_path = (
        repository_root / "docs/audits/hdf5_source_contract.yaml"
        if source_contract_path is None
        else source_contract_path
    )
    source_inventory_path = (
        repository_root / "docs/audits/hdf5_source_inventory.json"
        if source_inventory_path is None
        else source_inventory_path
    )
    expected_contract_path = (
        repository_root / str(config_payload["source_contract_path"])
    ).resolve()
    expected_inventory_path = (
        repository_root / str(config_payload["source_inventory_path"])
    ).resolve()
    if source_contract_path.resolve() != expected_contract_path:
        raise BenchmarkV4BuildError(
            "source_contract_path differs from the config binding"
        )
    if source_inventory_path.resolve() != expected_inventory_path:
        raise BenchmarkV4BuildError(
            "source_inventory_path differs from the config binding"
        )
    config_path = config_path.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config_binding = _authority_binding(
        config_path,
        repository_root=repository_root,
    )
    config_sha256 = str(config_binding["sha256"])
    config_payload_sha256 = sha256_bytes(canonical_json_bytes(config_payload))
    output_dir = benchmark_root / relative_path
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output_dir}")

    source_objects = _config_sources(config, config_payload)
    source_configs = tuple(
        _as_mapping(source, context=f"source config {index}")
        for index, source in enumerate(source_objects)
    )
    source_keys = [str(source["source_key"]) for source in source_configs]
    if len(source_keys) != len(set(source_keys)):
        raise BenchmarkV4BuildError("source configs must use unique source_key values")
    expected_asset_count, expected_asset_bytes = _materialization_preflight(
        source_objects,
        datasets_root=datasets_root,
        output_dir=output_dir,
    )

    source_contract_binding = _authority_binding(
        source_contract_path,
        repository_root=repository_root,
    )
    source_inventory_binding = _authority_binding(
        source_inventory_path,
        repository_root=repository_root,
    )
    schema_bindings = _schema_bindings(schemas_root)

    with atomic_output_directory(output_dir) as staging:
        for directory in ("assets", "indexes", "manifests", "derived", "reports"):
            (staging / directory).mkdir(parents=True, exist_ok=False)

        normalization = _NormalizationAccumulator()
        statistics = _StatisticsAccumulator()
        seen_ids: set[tuple[str, str]] = set()
        ready_source_count = 0
        materialized: dict[str, dict[str, Any]] = {}

        with (
            _JsonlWriter(staging / _SOURCE_INDEX_PATH) as source_writer,
            _JsonlWriter(staging / _MATERIALIZATION_INDEX_PATH) as materialization_writer,
        ):
            for source_config in source_configs:
                status = source_config["ingestion_status"]
                if status == "not_ready":
                    if source_config.get("indexes"):
                        raise BenchmarkV4BuildError("not_ready source must declare no source indexes")
                    continue
                if status != "ready":
                    raise BenchmarkV4BuildError(f"unsupported ingestion_status: {status!r}")
                ready_source_count += 1
                emitted = 0
                source_split_counts: Counter[str] = Counter()
                for observation in iter_source_observations(source_config, datasets_root):
                    source_record = _source_record_payload(observation)
                    if source_record["source_key"] != source_config["source_key"]:
                        raise BenchmarkV4BuildError("reader emitted a record for the wrong source")
                    identity = (
                        str(source_record["source_key"]),
                        str(source_record["source_sample_id"]),
                    )
                    if identity in seen_ids:
                        raise BenchmarkV4BuildError(f"duplicate source identity: {identity}")
                    seen_ids.add(identity)
                    if source_record["evaluation_eligibility"] == "strict":
                        raise BenchmarkV4BuildError(
                            "P1 v4 baseline has no verified strict cohort"
                        )
                    _materialize_observation(
                        observation,
                        source_record,
                        staging=staging,
                        writer=materialization_writer,
                        materialized=materialized,
                    )
                    source_writer.write(source_record)
                    normalization.update(observation, source_record)
                    statistics.update(observation, source_record)
                    source_split_counts[str(source_record["canonical_split"])] += 1
                    emitted += 1
                if emitted == 0:
                    raise BenchmarkV4BuildError(
                        f"ready source emitted no records: {source_config['source_key']}"
                    )
                source_key = str(source_config["source_key"])
                _assert_source_population(
                    source_config,
                    observed_pair_count=emitted,
                    observed_split_counts=source_split_counts,
                    observed_positive_count=statistics.positive_counts[source_key],
                    observed_no_target_count=statistics.no_target_counts[source_key],
                )

        if ready_source_count != 5:
            raise BenchmarkV4BuildError(
                f"P1 requires exactly five ready sources, found {ready_source_count}"
            )
        materialized_size_bytes = sum(
            int(row["size_bytes"]) for row in materialized.values()
        )
        if len(materialized) != expected_asset_count:
            raise BenchmarkV4BuildError(
                "materialized asset count differs from preflight: "
                f"expected={expected_asset_count}, observed={len(materialized)}"
            )
        if materialized_size_bytes != expected_asset_bytes:
            raise BenchmarkV4BuildError(
                "materialized byte count differs from preflight: "
                f"expected={expected_asset_bytes}, observed={materialized_size_bytes}"
            )

        source_index_sha256 = sha256_file(staging / _SOURCE_INDEX_PATH)
        normalization_payload = normalization.finalize(
            source_index_sha256=source_index_sha256
        )
        atomic_write_json(staging / _NORMALIZATION_PATH, normalization_payload)
        normalization_binding = sha256_file(staging / _NORMALIZATION_PATH)

        source_config_by_key = {
            str(source["source_key"]): source for source in source_configs
        }
        parent_count = 0
        with (
            _JsonlWriter(staging / _PARENT_INDEX_PATH) as parent_writer,
            _JsonlWriter(staging / _SPLIT_PATHS["train"]) as train_writer,
            _JsonlWriter(staging / _SPLIT_PATHS["val"]) as val_writer,
            _JsonlWriter(staging / _SPLIT_PATHS["test"]) as test_writer,
        ):
            split_writers = {
                "train": train_writer,
                "val": val_writer,
                "test": test_writer,
            }
            for source_record in _read_jsonl(staging / _SOURCE_INDEX_PATH):
                source_key = str(source_record["source_key"])
                parent = _parent_from_source(
                    source_record,
                    source_config_by_key[source_key],
                    normalization_binding=normalization_binding,
                )
                parent_count += 1
                parent_writer.write(parent)
                split = str(parent["canonical_split"])
                split_writers[split].write(
                    _split_projection(parent, all_line_number=parent_count)
                )

        statistics_payload = statistics.statistics_payload(
            normalization_binding_sha256=normalization_binding
        )
        if parent_count != statistics_payload["parent_count"]:
            raise BenchmarkV4BuildError("source and canonical parent counts diverged")
        if statistics_payload["strict_generalization_status"] != "unavailable":
            raise BenchmarkV4BuildError("strict cohort unexpectedly exists in P1 v4")

        risk_payload = statistics.risk_payload(source_configs)
        eligibility_payload = statistics.eligibility_payload()
        channel_catalog_payload = _channel_catalog(source_configs)
        atomic_write_json(staging / _CHANNEL_CATALOG_PATH, channel_catalog_payload)
        channel_catalog_binding = _artifact_binding(staging, _CHANNEL_CATALOG_PATH)
        source_registry_payload = _build_source_registry(
            source_configs=source_configs,
            config_binding=config_binding,
            config_payload_sha256=config_payload_sha256,
            source_contract_binding=source_contract_binding,
            source_inventory_binding=source_inventory_binding,
            schema_bindings=schema_bindings,
            channel_catalog_binding=channel_catalog_binding,
        )
        atomic_write_json(staging / _BUILD_CONFIG_PATH, config_payload)
        atomic_write_json(staging / _SOURCE_REGISTRY_PATH, source_registry_payload)
        atomic_write_json(staging / _STATISTICS_PATH, statistics_payload)
        atomic_write_json(staging / _DUPLICATE_RISK_PATH, risk_payload)
        atomic_write_json(staging / _ELIGIBILITY_PATH, eligibility_payload)

        bound_paths = (
            _BUILD_CONFIG_PATH,
            _SOURCE_REGISTRY_PATH,
            _CHANNEL_CATALOG_PATH,
            _MATERIALIZATION_INDEX_PATH,
            _NORMALIZATION_PATH,
            _SOURCE_INDEX_PATH,
            _PARENT_INDEX_PATH,
            *_SPLIT_PATHS.values(),
            _STATISTICS_PATH,
            _DUPLICATE_RISK_PATH,
            _ELIGIBILITY_PATH,
        )
        artifact_bindings = [
            _artifact_binding(staging, relative_path) for relative_path in bound_paths
        ]
        manifest = _manifest_payload(
            config_sha256=config_sha256,
            artifact_bindings=artifact_bindings,
            source_contract_sha256=str(source_contract_binding["sha256"]),
            source_inventory_sha256=str(source_inventory_binding["sha256"]),
            source_registry_sha256=sha256_file(staging / _SOURCE_REGISTRY_PATH),
            statistics=statistics_payload,
            normalization_binding=normalization_binding,
            channel_catalog_sha256=sha256_file(staging / _CHANNEL_CATALOG_PATH),
            materialization_index_sha256=sha256_file(
                staging / _MATERIALIZATION_INDEX_PATH
            ),
            materialized_asset_count=len(materialized),
            materialized_size_bytes=materialized_size_bytes,
        )
        atomic_write_json(staging / _MANIFEST_PATH, manifest)

    manifest_path = output_dir / _MANIFEST_PATH
    result = dict(manifest)
    result["manifest_sha256"] = sha256_file(manifest_path)
    result["benchmark_path"] = relative_path
    return result


__all__ = [
    "BENCHMARK_RELATIVE_PATH",
    "BUILDER_PROTOCOL",
    "NORMALIZATION_PROTOCOL",
    "SPLIT_PROJECTION_PROTOCOL",
    "BenchmarkV4BuildError",
    "build_benchmark_v4",
]
