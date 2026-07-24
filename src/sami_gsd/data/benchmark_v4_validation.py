"""P1 自包含 HDF5 Canonical Benchmark v4 的独立重放验证。

用途：不信任 builder summary，重新打开 source/copy、重算记录、通道目录与 lineage。
推荐入口：由 ``sami-gsd benchmark validate`` 调用 ``validate_benchmark_v4``。
输入：只读 benchmark、datasets、schemas；输出：repo ``outputs/`` 下全新原子 JSON 报告。
写行为：绝不修改 benchmark 或 datasets，拒绝覆盖验证报告。
阶段：P1；本模块不调用 builder，也不共享 builder 的统计实现。
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from sami_gsd.utilities.artifacts import (
    atomic_write_json,
    canonical_json_bytes,
    reject_non_finite,
    sha256_bytes,
    sha256_file,
)


VALIDATION_PROTOCOL = "sami_hdf5_materialized_copy_validator_v1"
BUILDER_PROTOCOL = "sami_hdf5_benchmark_v4_builder_v1"
MANIFEST_PROTOCOL = "sami_hdf5_materialized_copy_v1"
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


class BenchmarkV4ValidationError(ValueError):
    """validator 本身的调用边界无效，而非被验证 benchmark 的普通错误。"""


def _complete_counts(
    values: Mapping[str, int],
    keys: Sequence[str],
) -> dict[str, int]:
    """独立重放显式输出零人口，不继承 builder 的投影实现。"""

    return {key: int(values.get(key, 0)) for key in keys}


def _validate_source_population(
    source: Mapping[str, Any],
    replay: "_ReplayAccumulator",
    *,
    observed_split_counts: Mapping[str, int],
    errors: list[str],
) -> None:
    """独立比较现场人口与 source registry 绑定的 inventory 投影。"""

    source_key = str(source.get("source_key"))
    expected = {
        "pair_count": source.get("expected_pair_count"),
        "split_counts": source.get("expected_source_split_counts"),
        "positive_count": source.get("expected_positive_count"),
        "no_target_count": source.get("expected_no_target_count"),
    }
    observed = {
        "pair_count": replay.source_counts[source_key],
        "split_counts": dict(sorted(observed_split_counts.items())),
        "positive_count": replay.positive_counts[source_key],
        "no_target_count": replay.no_target_counts[source_key],
    }
    if observed != expected:
        errors.append(
            f"source_population_inventory_mismatch:{source_key}:"
            f"expected={expected}:observed={observed}"
        )


def _as_mapping(value: Any, *, context: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        payload = dict(value)
    elif callable(getattr(value, "to_mapping", None)):
        payload = value.to_mapping()
    elif callable(getattr(value, "model_dump", None)):
        payload = value.model_dump(mode="json")
    else:
        raise TypeError(f"{context} must expose a mapping projection")
    if not isinstance(payload, Mapping):
        raise TypeError(f"{context} projection must be a mapping")
    result = dict(payload)
    reject_non_finite(result)
    return result


def _portable_path(value: str, *, prefix: str | None = None) -> str:
    if not value or "\\" in value:
        raise BenchmarkV4ValidationError("logical path must be a non-empty POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in value.split("/")):
        raise BenchmarkV4ValidationError(f"non-portable logical path: {value!r}")
    if prefix is not None and not value.startswith(prefix):
        raise BenchmarkV4ValidationError(f"logical path must start with {prefix!r}: {value!r}")
    return value


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r}")


def _strict_json(path: Path) -> dict[str, Any]:
    payload = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=_reject_constant,
    )
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    reject_non_finite(payload)
    return payload


def _strict_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                payload = json.loads(line, parse_constant=_reject_constant)
            except (json.JSONDecodeError, ValueError) as error:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {error}") from error
            if not isinstance(payload, dict):
                raise ValueError(f"JSONL row must be an object at {path}:{line_number}")
            reject_non_finite(payload)
            yield payload


def _reject_machine_paths(value: Any, *, location: str = "$") -> None:
    """独立拒绝规范 artifact 中的本机绝对路径。"""

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
            raise ValueError(f"machine_path:{location}")
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


def _record_hash(payload: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json_bytes(_record_without_hash(payload)))


def _verify_record_hash(
    payload: Mapping[str, Any],
    *,
    context: str,
    errors: list[str],
) -> None:
    expected = payload.get("record_sha256")
    actual = _record_hash(payload)
    if expected != actual:
        errors.append(f"{context}:record_sha256_mismatch:{expected}:{actual}")


def _contract_from_mapping(class_name: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    from sami_gsd.contracts import benchmark_v4 as contracts

    if class_name == "NormalizationManifestV1":
        statistics = tuple(
            contracts.NormalizationChannelStatisticV1(**dict(row))
            for row in payload["statistics"]
        )
        instance = contracts.NormalizationManifestV1(
            **{
                **dict(payload),
                "statistics": statistics,
            }
        )
        return _as_mapping(instance, context=class_name)
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


def _observation_record(observation: Any) -> dict[str, Any]:
    projector = getattr(observation, "to_source_record_payload", None)
    if not callable(projector):
        raise TypeError("SourceObservation must expose to_source_record_payload()")
    payload = _contract_from_mapping(
        "Hdf5SourceRecordV1",
        _as_mapping(projector(), context="source record"),
    )
    _reject_machine_paths(payload)
    return payload


def _array(observation: Any, name: str) -> Any:
    if not hasattr(observation, name):
        raise TypeError(f"SourceObservation is missing {name!r}")
    return getattr(observation, name)


@dataclass
class _ReplayMoments:
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
            raise ValueError("normalization_non_finite_valid_pixel")
        batch_count = int(flat.size)
        batch_mean = float(flat.mean(dtype=np.float64))
        difference = flat - batch_mean
        batch_m2 = float(np.dot(difference, difference))
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
        self.m2 = self.m2 + batch_m2 + delta * delta * self.count * batch_count / total
        self.mean = self.mean + delta * batch_count / total
        self.count = total
        assert self.minimum is not None and self.maximum is not None
        self.minimum = min(self.minimum, batch_min)
        self.maximum = max(self.maximum, batch_max)

    def payload(
        self,
        source_key: str,
        channel_key: str,
        channel_index: int,
    ) -> dict[str, Any]:
        if self.count <= 0 or self.minimum is None or self.maximum is None:
            raise ValueError(f"normalization_empty:{source_key}:{channel_key}")
        variance = max(0.0, self.m2 / self.count)
        return {
            "source_key": source_key,
            "channel_key": channel_key,
            "channel_index": channel_index,
            "valid_pixel_count": self.count,
            "mean": self.mean,
            "std": math.sqrt(variance),
        }


class _ReplayAccumulator:
    """与 builder 分离的统计重放器；不读取任何 builder summary。"""

    def __init__(self) -> None:
        self.moments: dict[tuple[str, str], _ReplayMoments] = {}
        self.channel_indices: dict[tuple[str, str], int] = {}
        self.population = hashlib.sha256()
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

    def update(self, observation: Any, record: Mapping[str, Any]) -> None:
        import numpy as np

        source_key = str(record["source_key"])
        image = np.asarray(_array(observation, "image_values"))
        channel_valid = np.asarray(_array(observation, "channel_valid"), dtype=bool)
        pixel_valid = np.asarray(_array(observation, "pixel_valid"), dtype=bool)
        mask = np.asarray(_array(observation, "mask_values"))
        target_valid = np.asarray(_array(observation, "target_valid"), dtype=bool)
        if image.ndim != 3 or channel_valid.shape != (image.shape[0],):
            raise ValueError(f"input_validity_shape:{source_key}:{record['source_sample_id']}")
        if pixel_valid.shape != image.shape:
            raise ValueError(f"pixel_valid_shape:{source_key}:{record['source_sample_id']}")
        if mask.ndim != 2 or mask.shape != target_valid.shape:
            raise ValueError(f"target_validity_shape:{source_key}:{record['source_sample_id']}")
        if len(record["channels"]) != image.shape[0]:
            raise ValueError(f"channel_descriptor_count:{source_key}:{record['source_sample_id']}")

        valid_target = mask[target_valid]
        if valid_target.size == 0:
            raise ValueError(f"empty_target_valid:{source_key}:{record['source_sample_id']}")
        if not bool(np.isfinite(valid_target).all()) or not bool(np.isin(valid_target, (0, 1)).all()):
            raise ValueError(f"invalid_binary_target:{source_key}:{record['source_sample_id']}")
        positive_pixels = int(np.count_nonzero(valid_target))

        self.source_counts[source_key] += 1
        self.split_counts[str(record["canonical_split"])] += 1
        self.assurance_counts[str(record["split_assurance"])] += 1
        self.eligibility_counts[str(record["evaluation_eligibility"])] += 1
        self.valid_pixel_counts[source_key] += int(valid_target.size)
        self.positive_pixel_counts[source_key] += positive_pixels
        if positive_pixels:
            self.positive_counts[source_key] += 1
        else:
            self.no_target_counts[source_key] += 1
        group = record.get("group") or {}
        duplicate = record.get("duplicate_component") or {}
        self.group_completeness_counts[str(group.get("completeness", "unavailable"))] += 1
        self.duplicate_evidence_counts[
            str(duplicate.get("evidence_level", "unavailable"))
        ] += 1

        if record["canonical_split"] != "train":
            return
        population_row = {
            "source_key": source_key,
            "source_sample_id": record["source_sample_id"],
            "source_record_sha256": record["record_sha256"],
        }
        self.population.update(canonical_json_bytes(population_row))
        for index, descriptor in enumerate(record["channels"]):
            if int(descriptor["index"]) != index:
                raise ValueError(f"channel_index_order:{source_key}:{record['source_sample_id']}")
            if not bool(channel_valid[index]):
                continue
            key = (source_key, str(descriptor["channel_key"]))
            previous_index = self.channel_indices.setdefault(key, index)
            if previous_index != index:
                raise ValueError(
                    f"channel_index_drift:{source_key}:{descriptor['channel_key']}"
                )
            self.moments.setdefault(key, _ReplayMoments()).update(
                image[index][pixel_valid[index]]
            )

    def normalization(self, *, source_index_sha256: str) -> dict[str, Any]:
        rows = [
            moments.payload(
                source_key,
                channel_key,
                self.channel_indices[(source_key, channel_key)],
            )
            for (source_key, channel_key), moments in sorted(self.moments.items())
        ]
        payload = {
            "schema_version": "sami_normalization_manifest_v1",
            "protocol": NORMALIZATION_PROTOCOL,
            "population_split": "train",
            "source_index_sha256": source_index_sha256,
            "population_sha256": self.population.hexdigest(),
            "statistics": rows,
            "aggregate_sha256": sha256_bytes(canonical_json_bytes(rows)),
        }
        return _contract_from_mapping("NormalizationManifestV1", payload)

    def statistics(self, *, normalization_binding_sha256: str) -> dict[str, Any]:
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
        payload = {
            "schema_version": "sami_benchmark_statistics_v4",
            **core,
            "aggregate_sha256": sha256_bytes(canonical_json_bytes(core)),
        }
        return _contract_from_mapping("BenchmarkStatisticsV4", payload)

    def risk(self, sources: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return {
            "schema_version": "sami_benchmark_v4_duplicate_risk_report_v1",
            "protocol": "reported_incomplete_group_and_duplicate_evidence_v1",
            "group_completeness_counts": dict(sorted(self.group_completeness_counts.items())),
            "duplicate_evidence_counts": dict(sorted(self.duplicate_evidence_counts.items())),
            "training_blocked_by_incomplete_group_or_duplicate_evidence": False,
            "sources": [
                {
                    "source_key": source["source_key"],
                    "ingestion_status": source["ingestion_status"],
                    "risks": list(source.get("risks", ())),
                    "known_location_cross_split_conflict_count": source.get(
                        "known_location_cross_split_conflict_count"
                    ),
                }
                for source in sources
            ],
        }

    def eligibility(self) -> dict[str, Any]:
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


def _registered_rgb(
    source: Mapping[str, Any],
    *,
    normalization_binding: str,
) -> list[dict[str, Any]]:
    registered = source.get("registered_rgb")
    if registered is None:
        return []
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


def _expected_parent(
    record: Mapping[str, Any],
    source: Mapping[str, Any],
    *,
    normalization_binding: str,
) -> dict[str, Any]:
    payload = {
        "schema_version": "sami_canonical_parent_v4",
        "parent_id": _parent_id(str(record["source_key"]), str(record["source_sample_id"])),
        "source_key": record["source_key"],
        "source_sample_id": record["source_sample_id"],
        "canonical_split": record["canonical_split"],
        "split_assurance": record["split_assurance"],
        "evaluation_eligibility": record["evaluation_eligibility"],
        "channels": record["channels"],
        "image_ref": record["image"],
        "mask_ref": record["mask"],
        "validity_ref": record["validity"],
        "registered_views": _registered_rgb(
            source,
            normalization_binding=normalization_binding,
        ),
        "group": record["group"],
        "source_record_sha256": record["record_sha256"],
    }
    return _seal_parent(payload)


def _expected_projection(parent: Mapping[str, Any], line_number: int) -> dict[str, Any]:
    payload = {
        "schema_version": "sami_benchmark_v4_split_projection_v1",
        "protocol": SPLIT_PROJECTION_PROTOCOL,
        "parent_id": parent["parent_id"],
        "canonical_split": parent["canonical_split"],
        "all_line_number": line_number,
        "parent_record_sha256": parent["record_sha256"],
    }
    payload["record_sha256"] = sha256_bytes(canonical_json_bytes(payload))
    return payload


def _resolve_repo_binding(
    binding: Mapping[str, Any],
    *,
    repository_root: Path,
) -> Path:
    logical = _portable_path(str(binding["path"]))
    resolved = (repository_root / logical).resolve()
    try:
        resolved.relative_to(repository_root.resolve())
    except ValueError as error:
        raise BenchmarkV4ValidationError(f"binding escapes repository: {logical}") from error
    return resolved


def _validate_artifact_tree(
    benchmark_dir: Path,
    manifest: Mapping[str, Any],
    *,
    errors: list[str],
) -> None:
    bindings = manifest.get("artifact_bindings")
    if not isinstance(bindings, Mapping):
        errors.append("manifest_artifact_bindings_not_mapping")
        bindings = {}

    expected: dict[str, str] = {}
    for path_value, sha_value in bindings.items():
        try:
            logical = _portable_path(str(path_value))
        except Exception as error:
            errors.append(f"manifest_artifact_path:{error}")
            continue
        if logical in expected:
            errors.append(f"duplicate_artifact_binding:{logical}")
        if not isinstance(sha_value, str):
            errors.append(f"manifest_artifact_sha256_not_string:{logical}")
            continue
        expected[logical] = sha_value

    actual: set[str] = set()
    for path in benchmark_dir.rglob("*"):
        if path.is_symlink():
            errors.append(f"symlink_forbidden:{path.relative_to(benchmark_dir).as_posix()}")
            continue
            if path.is_file():
                logical = path.relative_to(benchmark_dir).as_posix()
                if not logical.startswith("assets/"):
                    actual.add(logical)
            if ".part-" in path.name or path.suffix == ".part":
                errors.append(f"partial_file_forbidden:{logical}")

    allowed = set(expected) | {_MANIFEST_PATH}
    for logical in sorted(actual - allowed):
        errors.append(f"unbound_artifact:{logical}")
    for logical in sorted(allowed - actual):
        errors.append(f"missing_artifact:{logical}")
    for logical, expected_sha in sorted(expected.items()):
        path = benchmark_dir / logical
        if not path.is_file():
            continue
        actual_sha = sha256_file(path)
        if expected_sha != actual_sha:
            errors.append(f"artifact_sha256_mismatch:{logical}")

    aggregate = sha256_bytes(
        canonical_json_bytes(dict(sorted(expected.items())))
    )
    if manifest.get("aggregate_sha256") != aggregate:
        errors.append("manifest_aggregate_sha256_mismatch")


def _resolve_benchmark_copy(
    benchmark_dir: Path,
    benchmark_logical_path: str,
) -> tuple[str, Path]:
    prefix = PurePosixPath("benchmark") / BENCHMARK_RELATIVE_PATH
    logical = PurePosixPath(_portable_path(benchmark_logical_path, prefix="benchmark/"))
    try:
        relative = logical.relative_to(prefix)
    except ValueError as error:
        raise BenchmarkV4ValidationError(
            f"benchmark copy escapes v4 Small: {benchmark_logical_path}"
        ) from error
    relative_text = _portable_path(relative.as_posix(), prefix="assets/")
    physical = benchmark_dir / relative_text
    resolved = physical.resolve()
    try:
        resolved.relative_to(benchmark_dir.resolve())
    except ValueError as error:
        raise BenchmarkV4ValidationError(
            f"benchmark copy escapes root: {benchmark_logical_path}"
        ) from error
    return relative_text, physical


def _validate_materialized_assets(
    benchmark_dir: Path,
    datasets_root: Path,
    records: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    *,
    errors: list[str],
) -> tuple[int, int]:
    """Replay the materialization ledger against source bytes and Benchmark copies."""

    from sami_gsd.data.hdf5_sources_v4 import resolve_dataset_logical_path

    ledger_path = benchmark_dir / _MATERIALIZATION_INDEX_PATH
    try:
        rows = list(_strict_jsonl(ledger_path))
    except Exception as error:
        errors.append(f"materialization_index_read:{error}")
        return 0, 0
    expected_keys = {
        "schema_version",
        "role",
        "source_key",
        "source_logical_path",
        "benchmark_logical_path",
        "sha256",
        "size_bytes",
    }
    ledger_by_path: dict[str, dict[str, Any]] = {}
    total_bytes = 0
    for line_number, row in enumerate(rows, start=1):
        if set(row) != expected_keys:
            errors.append(f"materialization_row_keys:{line_number}")
            continue
        if row.get("schema_version") != "sami_materialized_hdf5_asset_v1":
            errors.append(f"materialization_row_schema:{line_number}")
        if row.get("role") not in {"image", "mask"}:
            errors.append(f"materialization_row_role:{line_number}")
        try:
            relative, copy_path = _resolve_benchmark_copy(
                benchmark_dir,
                str(row["benchmark_logical_path"]),
            )
            source_path = resolve_dataset_logical_path(
                str(row["source_logical_path"]),
                datasets_root,
            )
        except Exception as error:
            errors.append(f"materialization_row_path:{line_number}:{error}")
            continue
        if relative in ledger_by_path:
            errors.append(f"materialization_duplicate_path:{relative}")
            continue
        ledger_by_path[relative] = row
        size = row.get("size_bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            errors.append(f"materialization_size_invalid:{relative}")
            continue
        total_bytes += size
        if not source_path.is_file():
            errors.append(f"materialization_source_missing:{row['source_logical_path']}")
            continue
        if not copy_path.is_file():
            errors.append(f"materialization_copy_missing:{relative}")
            continue
        if copy_path.is_symlink() or copy_path.stat().st_nlink != 1:
            errors.append(f"materialization_copy_must_be_independent_file:{relative}")
        if source_path.stat().st_size != size:
            errors.append(f"materialization_source_size_mismatch:{relative}")
        if copy_path.stat().st_size != size:
            errors.append(f"materialization_copy_size_mismatch:{relative}")
        source_sha = sha256_file(source_path)
        copy_sha = sha256_file(copy_path)
        if source_sha != row.get("sha256"):
            errors.append(f"materialization_source_sha256_mismatch:{relative}")
        if copy_sha != row.get("sha256"):
            errors.append(f"materialization_copy_sha256_mismatch:{relative}")

    record_rows: dict[str, dict[str, Any]] = {}
    for record in records:
        for role in ("image", "mask"):
            reference = record.get(role)
            if not isinstance(reference, Mapping):
                errors.append(
                    f"materialization_record_reference_missing:"
                    f"{record.get('source_key')}:{record.get('source_sample_id')}:{role}"
                )
                continue
            try:
                relative, _copy_path = _resolve_benchmark_copy(
                    benchmark_dir,
                    str(reference["benchmark_logical_path"]),
                )
            except Exception as error:
                errors.append(f"materialization_record_path:{error}")
                continue
            expected = {
                "schema_version": "sami_materialized_hdf5_asset_v1",
                "role": role,
                "source_key": record["source_key"],
                "source_logical_path": reference["source_logical_path"],
                "benchmark_logical_path": reference["benchmark_logical_path"],
                "sha256": reference["sha256"],
                "size_bytes": reference["size_bytes"],
            }
            previous = record_rows.setdefault(relative, expected)
            if previous != expected:
                errors.append(f"materialization_record_collision:{relative}")
    if record_rows != ledger_by_path:
        errors.append("materialization_ledger_record_projection_mismatch")

    actual_assets = {
        path.relative_to(benchmark_dir).as_posix()
        for path in (benchmark_dir / "assets").rglob("*")
        if path.is_file()
    }
    if actual_assets != set(ledger_by_path):
        errors.append("materialization_asset_tree_mismatch")
    if sha256_file(ledger_path) != manifest.get("materialization_index_sha256"):
        errors.append("manifest_materialization_index_sha256_mismatch")
    if len(ledger_by_path) != manifest.get("materialized_asset_count"):
        errors.append("manifest_materialized_asset_count_mismatch")
    if total_bytes != manifest.get("materialized_size_bytes"):
        errors.append("manifest_materialized_size_bytes_mismatch")
    return len(ledger_by_path), total_bytes


def _expected_channel_catalog(sources: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Independently rebuild the physical channel vocabulary."""

    grouped: dict[str, list[dict[str, Any]]] = {}
    invariants: dict[str, tuple[Any, ...]] = {}
    for source in sources:
        if source.get("ingestion_status") != "ready":
            continue
        for descriptor in source.get("channels", ()):
            key = str(descriptor["channel_key"])
            invariant = (
                descriptor["modality_family"],
                descriptor["wavelength_nm"],
                descriptor["wavelength_known"],
                descriptor["gsd_m"],
                descriptor["gsd_known"],
            )
            previous = invariants.setdefault(key, invariant)
            if previous != invariant:
                raise ValueError(f"channel_catalog_invariant_drift:{key}")
            grouped.setdefault(key, []).append(
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
    for token, key in enumerate(sorted(grouped)):
        invariant = invariants[key]
        entries.append(
            {
                "channel_token": token,
                "channel_key": key,
                "modality_family": invariant[0],
                "wavelength_nm": invariant[1],
                "wavelength_known": invariant[2],
                "gsd_m": invariant[3],
                "gsd_known": invariant[4],
                "source_bindings": sorted(
                    grouped[key],
                    key=lambda row: (str(row["source_key"]), int(row["source_index"])),
                ),
            }
        )
    return {
        "schema_version": "sami_channel_catalog_v1",
        "protocol": "explicit_channel_semantics_v1",
        "ordering_rule": "channel_token_is_lexicographic_channel_key_not_tensor_order",
        "entries": entries,
        "aggregate_sha256": sha256_bytes(canonical_json_bytes(entries)),
    }


def _validate_external_bindings(
    manifest: Mapping[str, Any],
    registry: Mapping[str, Any],
    *,
    repository_root: Path,
    errors: list[str],
) -> None:
    for role in ("source_contract", "source_inventory"):
        binding = registry.get(role)
        if not isinstance(binding, Mapping):
            errors.append(f"missing_{role}_binding")
            continue
        try:
            path = _resolve_repo_binding(binding, repository_root=repository_root)
        except Exception as error:
            errors.append(f"{role}_path:{error}")
            continue
        if not path.is_file():
            errors.append(f"{role}_missing:{binding.get('path')}")
        elif sha256_file(path) != binding.get("sha256"):
            errors.append(f"{role}_sha256_mismatch")
        manifest_digest = manifest.get(f"{role}_sha256")
        if manifest_digest != binding.get("sha256"):
            errors.append(f"manifest_{role}_sha256_mismatch")

    config_binding = registry.get("config")
    if not isinstance(config_binding, Mapping):
        errors.append("missing_config_binding")
    else:
        try:
            config_path = _resolve_repo_binding(
                config_binding,
                repository_root=repository_root,
            )
        except Exception as error:
            errors.append(f"config_path:{error}")
        else:
            if not config_path.is_file():
                errors.append(f"config_missing:{config_binding.get('path')}")
            elif sha256_file(config_path) != config_binding.get("sha256"):
                errors.append("config_sha256_mismatch")
        if manifest.get("config_sha256") != config_binding.get("sha256"):
            errors.append("manifest_config_sha256_mismatch")

    for binding in registry.get("schemas", ()):
        if not isinstance(binding, Mapping):
            errors.append("invalid_schema_binding")
            continue
        try:
            path = _resolve_repo_binding(binding, repository_root=repository_root)
        except Exception as error:
            errors.append(f"schema_path:{error}")
            continue
        if not path.is_file():
            errors.append(f"schema_missing:{binding.get('path')}")
        elif sha256_file(path) != binding.get("sha256"):
            errors.append(f"schema_sha256_mismatch:{binding.get('path')}")


def _report_payload(
    *,
    errors: Sequence[str],
    warnings: Sequence[str],
    manifest_sha256: str | None,
    artifact_count: int,
    source_record_count: int,
    parent_count: int,
    split_counts: Mapping[str, int],
    assurance_counts: Mapping[str, int],
    eligibility_counts: Mapping[str, int],
    normalization_binding: str | None,
    materialized_asset_count: int,
    materialized_size_bytes: int,
) -> dict[str, Any]:
    fallback_sha256 = "0" * 64
    payload = {
        "schema_version": "sami_benchmark_validation_report_v4",
        "protocol": VALIDATION_PROTOCOL,
        "errors": list(errors),
        "warnings": list(warnings),
        "benchmark_manifest_sha256": manifest_sha256 or fallback_sha256,
        "artifact_count": artifact_count,
        "materialized_asset_count": materialized_asset_count,
        "materialized_size_bytes": materialized_size_bytes,
        "source_record_count": source_record_count,
        "parent_count": parent_count,
        "split_counts": dict(sorted(split_counts.items())),
        "assurance_counts": dict(sorted(assurance_counts.items())),
        "eligibility_counts": dict(sorted(eligibility_counts.items())),
        "normalization_binding_sha256": normalization_binding or fallback_sha256,
    }
    payload["aggregate_sha256"] = sha256_bytes(canonical_json_bytes(payload))
    return _contract_from_mapping("BenchmarkValidationReportV4", payload)


def validate_benchmark_v4(
    benchmark_dir: Path,
    *,
    datasets_root: Path,
    schemas_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    """独立重放 v4 benchmark 并向 repo outputs 写一份新报告。

    该函数会重新打开全部 source HDF5 与 Benchmark copies，只允许由项目负责人运行。
    """

    from sami_gsd.data.hdf5_sources_v4 import iter_source_observations

    benchmark_dir = benchmark_dir.resolve()
    datasets_root = datasets_root.resolve()
    schemas_root = schemas_root.resolve()
    repository_root = schemas_root.parent.resolve()
    outputs_root = repository_root / "outputs"
    output_path = output_path.resolve()
    try:
        output_path.relative_to(outputs_root)
    except ValueError as error:
        raise BenchmarkV4ValidationError(
            f"validation report must be written below repository outputs/: {output_path}"
        ) from error
    try:
        output_path.relative_to(benchmark_dir)
    except ValueError:
        pass
    else:
        raise BenchmarkV4ValidationError("validation report must not be written inside benchmark")
    if not benchmark_dir.is_dir():
        raise FileNotFoundError(benchmark_dir)
    if not datasets_root.is_dir():
        raise FileNotFoundError(datasets_root)
    if not schemas_root.is_dir():
        raise FileNotFoundError(schemas_root)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite validation report: {output_path}")

    errors: list[str] = []
    warnings: list[str] = []
    manifest_sha256: str | None = None
    artifact_count = 0
    materialized_asset_count = 0
    materialized_size_bytes = 0
    source_record_count = 0
    parent_count = 0
    split_counts: Mapping[str, int] = {}
    assurance_counts: Mapping[str, int] = {}
    eligibility_counts: Mapping[str, int] = {}
    normalization_binding: str | None = None

    manifest_path = benchmark_dir / _MANIFEST_PATH
    if not manifest_path.is_file():
        errors.append("benchmark_manifest_missing")
        report = _report_payload(
            errors=errors,
            warnings=warnings,
            manifest_sha256=None,
            artifact_count=0,
            materialized_asset_count=0,
            materialized_size_bytes=0,
            source_record_count=0,
            parent_count=0,
            split_counts={},
            assurance_counts={},
            eligibility_counts={},
            normalization_binding=None,
        )
        atomic_write_json(output_path, report)
        return report

    try:
        manifest_raw = _strict_json(manifest_path)
    except Exception as error:
        errors.append(f"benchmark_manifest_json:{error}")
        manifest = {}
    else:
        try:
            manifest = _contract_from_mapping("BenchmarkManifestV4", manifest_raw)
        except Exception as error:
            errors.append(f"benchmark_manifest_contract:{error}")
            manifest = manifest_raw
    manifest_sha256 = sha256_file(manifest_path)
    artifact_bindings = manifest.get("artifact_bindings")
    artifact_count = len(artifact_bindings) if isinstance(artifact_bindings, Mapping) else 0
    if manifest.get("protocol") != MANIFEST_PROTOCOL:
        errors.append("benchmark_manifest_protocol_mismatch")
    if manifest.get("benchmark_relative_path") != BENCHMARK_RELATIVE_PATH:
        errors.append("benchmark_relative_path_mismatch")

    _validate_artifact_tree(benchmark_dir, manifest, errors=errors)

    try:
        registry = _strict_json(benchmark_dir / _SOURCE_REGISTRY_PATH)
        _verify_record_hash(registry, context="source_registry", errors=errors)
        if registry.get("protocol") != BUILDER_PROTOCOL:
            errors.append("source_registry_builder_protocol_mismatch")
        registry_sha256 = sha256_file(benchmark_dir / _SOURCE_REGISTRY_PATH)
        if manifest.get("source_registry_sha256") != registry_sha256:
            errors.append("manifest_source_registry_sha256_mismatch")
        _validate_external_bindings(
            manifest,
            registry,
            repository_root=repository_root,
            errors=errors,
        )
        build_config = _strict_json(benchmark_dir / _BUILD_CONFIG_PATH)
        build_config_payload_sha256 = sha256_bytes(canonical_json_bytes(build_config))
        if registry.get("config_payload_sha256") != build_config_payload_sha256:
            errors.append("source_registry_config_payload_sha256_mismatch")
        config_binding = registry.get("config")
        if isinstance(config_binding, Mapping):
            config_path = _resolve_repo_binding(
                config_binding,
                repository_root=repository_root,
            )
            from sami_gsd.contracts.benchmark_v4_config import (
                load_benchmark_v4_config,
            )

            live_config = load_benchmark_v4_config(
                config_path,
                repository_root=repository_root,
            ).model_dump(mode="json")
            if live_config != build_config:
                errors.append("live_config_semantic_replay_mismatch")
        else:
            errors.append("source_registry_config_binding_missing")
        sources_raw = registry["sources"]
        if not isinstance(sources_raw, list):
            raise ValueError("source registry sources must be a list")
        sources = tuple(dict(source) for source in sources_raw)
        config_sources = build_config.get("sources")
        if not isinstance(config_sources, list):
            errors.append("build_config_sources_not_list")
        elif config_sources != list(sources):
            errors.append("build_config_source_registry_mismatch")
    except Exception as error:
        errors.append(f"source_registry:{error}")
        registry = {}
        sources = ()

    try:
        indexed_channel_catalog = _strict_json(benchmark_dir / _CHANNEL_CATALOG_PATH)
        expected_channel_catalog = _expected_channel_catalog(sources)
        if indexed_channel_catalog != expected_channel_catalog:
            errors.append("channel_catalog_semantic_replay_mismatch")
        channel_catalog_sha256 = sha256_file(benchmark_dir / _CHANNEL_CATALOG_PATH)
        if manifest.get("channel_catalog_sha256") != channel_catalog_sha256:
            errors.append("manifest_channel_catalog_sha256_mismatch")
        registry_channel = registry.get("channel_catalog")
        if not isinstance(registry_channel, Mapping):
            errors.append("source_registry_channel_catalog_binding_missing")
        elif (
            registry_channel.get("path") != _CHANNEL_CATALOG_PATH
            or registry_channel.get("sha256") != channel_catalog_sha256
        ):
            errors.append("source_registry_channel_catalog_binding_mismatch")
    except Exception as error:
        errors.append(f"channel_catalog_replay:{error}")

    source_keys = [str(source.get("source_key")) for source in sources]
    if len(source_keys) != len(set(source_keys)):
        errors.append("source_registry_duplicate_source_key")
    if sum(source.get("ingestion_status") == "ready" for source in sources) != 5:
        errors.append("source_registry_ready_source_count_must_be_five")

    replay = _ReplayAccumulator()
    indexed_records: list[dict[str, Any]] = []
    try:
        indexed_iterator = iter(_strict_jsonl(benchmark_dir / _SOURCE_INDEX_PATH))
        for source in sources:
            status = source.get("ingestion_status")
            if status == "not_ready":
                if source.get("indexes"):
                    errors.append(f"not_ready_source_has_indexes:{source.get('source_key')}")
                continue
            if status != "ready":
                errors.append(f"invalid_ingestion_status:{source.get('source_key')}:{status}")
                continue
            emitted = 0
            source_split_counts: Counter[str] = Counter()
            for observation in iter_source_observations(source, datasets_root):
                try:
                    indexed = next(indexed_iterator)
                except StopIteration:
                    errors.append("source_index_shorter_than_live_replay")
                    break
                live = _observation_record(observation)
                if live.get("source_key") != source.get("source_key"):
                    errors.append(
                        f"live_source_key_mismatch:{source.get('source_key')}:{live.get('source_key')}"
                    )
                _verify_record_hash(indexed, context="source_record", errors=errors)
                try:
                    _reject_machine_paths(indexed)
                    indexed = _contract_from_mapping("Hdf5SourceRecordV1", indexed)
                except Exception as error:
                    errors.append(f"source_record_contract:{error}")
                if live != indexed:
                    errors.append(
                        f"source_record_replay_mismatch:{live.get('source_key')}:{live.get('source_sample_id')}"
                    )
                indexed_records.append(indexed)
                try:
                    replay.update(observation, live)
                except Exception as error:
                    errors.append(
                        f"source_statistics_replay:{live.get('source_key')}:{live.get('source_sample_id')}:{error}"
                    )
                source_split_counts[str(live["canonical_split"])] += 1
                emitted += 1
            if emitted == 0:
                errors.append(f"ready_source_emitted_no_rows:{source.get('source_key')}")
            _validate_source_population(
                source,
                replay,
                observed_split_counts=source_split_counts,
                errors=errors,
            )
        extra = next(indexed_iterator, None)
        if extra is not None:
            errors.append("source_index_longer_than_live_replay")
    except Exception as error:
        errors.append(f"source_replay_fatal:{error}")

    source_record_count = len(indexed_records)
    materialized_asset_count, materialized_size_bytes = _validate_materialized_assets(
        benchmark_dir,
        datasets_root,
        indexed_records,
        manifest,
        errors=errors,
    )
    normalization_path = benchmark_dir / _NORMALIZATION_PATH
    try:
        indexed_normalization = _contract_from_mapping(
            "NormalizationManifestV1",
            _strict_json(normalization_path),
        )
        live_normalization = replay.normalization(
            source_index_sha256=sha256_file(benchmark_dir / _SOURCE_INDEX_PATH),
        )
        if indexed_normalization != live_normalization:
            errors.append("normalization_semantic_replay_mismatch")
        normalization_binding = sha256_file(normalization_path)
        if manifest.get("normalization_binding_sha256") != normalization_binding:
            errors.append("manifest_normalization_binding_mismatch")
    except Exception as error:
        errors.append(f"normalization_replay:{error}")

    source_by_key = {str(source.get("source_key")): source for source in sources}
    indexed_parents: list[dict[str, Any]] = []
    try:
        parent_iterator = iter(_strict_jsonl(benchmark_dir / _PARENT_INDEX_PATH))
        for line_number, record in enumerate(indexed_records, start=1):
            try:
                parent = next(parent_iterator)
            except StopIteration:
                errors.append("canonical_parent_index_shorter_than_source_index")
                break
            _verify_record_hash(parent, context="canonical_parent", errors=errors)
            try:
                parent = _contract_from_mapping("CanonicalParentV4", parent)
                expected = _expected_parent(
                    record,
                    source_by_key[str(record["source_key"])],
                    normalization_binding=normalization_binding or ("0" * 64),
                )
                if parent != expected:
                    errors.append(
                        f"canonical_parent_replay_mismatch:{record['source_key']}:{record['source_sample_id']}"
                    )
            except Exception as error:
                errors.append(
                    f"canonical_parent_contract:{record.get('source_key')}:{record.get('source_sample_id')}:{error}"
                )
            indexed_parents.append(parent)
        if next(parent_iterator, None) is not None:
            errors.append("canonical_parent_index_longer_than_source_index")
    except Exception as error:
        errors.append(f"canonical_parent_replay_fatal:{error}")
    parent_count = len(indexed_parents)

    for split, relative_path in _SPLIT_PATHS.items():
        expected_rows = [
            _expected_projection(parent, line_number)
            for line_number, parent in enumerate(indexed_parents, start=1)
            if parent.get("canonical_split") == split
        ]
        try:
            actual_rows = list(_strict_jsonl(benchmark_dir / relative_path))
        except Exception as error:
            errors.append(f"split_projection_read:{split}:{error}")
            continue
        for row in actual_rows:
            _verify_record_hash(row, context=f"split_{split}", errors=errors)
        if actual_rows != expected_rows:
            errors.append(f"split_projection_replay_mismatch:{split}")

    try:
        live_statistics = replay.statistics(
            normalization_binding_sha256=normalization_binding or ("0" * 64)
        )
        indexed_statistics = _contract_from_mapping(
            "BenchmarkStatisticsV4",
            _strict_json(benchmark_dir / _STATISTICS_PATH),
        )
        if indexed_statistics != live_statistics:
            errors.append("statistics_semantic_replay_mismatch")
        split_counts = live_statistics["split_counts"]
        assurance_counts = live_statistics["assurance_counts"]
        eligibility_counts = live_statistics["eligibility_counts"]
        if eligibility_counts.get("strict", 0) != 0:
            errors.append("strict_population_must_be_zero")
        if live_statistics["strict_generalization_status"] != "unavailable":
            errors.append("strict_generalization_status_must_be_unavailable")
        if manifest.get("strict_generalization_status") != "unavailable":
            errors.append("manifest_strict_generalization_status_must_be_unavailable")
        if manifest.get("source_record_count") != source_record_count:
            errors.append("manifest_source_record_count_mismatch")
        if manifest.get("parent_count") != parent_count:
            errors.append("manifest_parent_count_mismatch")
        if manifest.get("split_counts") != split_counts:
            errors.append("manifest_split_counts_mismatch")
        if manifest.get("assurance_counts") != assurance_counts:
            errors.append("manifest_assurance_counts_mismatch")
        if manifest.get("eligibility_counts") != eligibility_counts:
            errors.append("manifest_eligibility_counts_mismatch")
    except Exception as error:
        errors.append(f"statistics_replay:{error}")

    try:
        indexed_risk = _strict_json(benchmark_dir / _DUPLICATE_RISK_PATH)
        if indexed_risk != replay.risk(sources):
            errors.append("duplicate_risk_semantic_replay_mismatch")
    except Exception as error:
        errors.append(f"duplicate_risk_replay:{error}")
    try:
        indexed_eligibility = _strict_json(benchmark_dir / _ELIGIBILITY_PATH)
        if indexed_eligibility != replay.eligibility():
            errors.append("evaluation_eligibility_semantic_replay_mismatch")
    except Exception as error:
        errors.append(f"evaluation_eligibility_replay:{error}")

    report = _report_payload(
        errors=errors,
        warnings=warnings,
        manifest_sha256=manifest_sha256,
        artifact_count=artifact_count,
        materialized_asset_count=materialized_asset_count,
        materialized_size_bytes=materialized_size_bytes,
        source_record_count=source_record_count,
        parent_count=parent_count,
        split_counts=split_counts,
        assurance_counts=assurance_counts,
        eligibility_counts=eligibility_counts,
        normalization_binding=normalization_binding,
    )
    atomic_write_json(output_path, report)
    return report


__all__ = [
    "VALIDATION_PROTOCOL",
    "BenchmarkV4ValidationError",
    "validate_benchmark_v4",
]
