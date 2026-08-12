"""Stage 4 train-only 扩展 Region Corpus 与 8,450 条组合索引。

该模块只消费 OA Benchmark 的人工 GT mask。500 条冻结 Corpus 作为只读成员引用，
新增 7,950 条资产独立发布；collection 只保存强身份索引，不复制或改写旧图像。
"""

from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

from oa_groundrag.artifacts.identity import (
    canonical_json,
    sha256_file,
    sha256_text,
    stable_hash,
)
from oa_groundrag.artifacts.io import first_symlink_component
from oa_groundrag.data.rs_general.io import (
    read_json,
    read_jsonl,
)
from oa_groundrag.artifacts.directory import AtomicArtifactDirectory
from oa_groundrag.vlm.config import _load_yaml

from ..contracts import EXPECTED_IDENTITY_FIELDS, fail
from ..region_contracts import (
    EVAL_MANIFEST_SCHEMA,
    MASK_RENDERER_VERSION,
    REGION_MANIFEST_SCHEMA,
    RGB_RENDERER_VERSION,
    RegionTargetStatus,
    _absolute,
    _benchmark,
    _config_path,
    _exact,
    _integer,
    _mapping,
    _ratio,
    _sha256,
    _string,
    parent_identity,
    size_bin,
)
from ..region import (
    RegionBenchmarkAccess,
    RegionBuildResult,
    annotation_guideline,
    annotation_queue_rows,
    ledger_rows,
    materialize_region_record,
    region_asset_identity,
)


EXTENSION_CONFIG_SCHEMA = "oa_groundrag.mask_grounded_region.extension_config.v2"
EXTENSION_MANIFEST_SCHEMA = "oa_groundrag.mask_grounded_region.extension_manifest.v2"
TRAIN_COLLECTION_SCHEMA = "oa_groundrag.mask_grounded_region.train_collection.v2"
TRAIN_COLLECTION_MEMBER_SCHEMA = "oa_groundrag.mask_grounded_region.train_collection_member.v2"
EXTENSION_SELECTION_ALGORITHM = "mask_grounded_region_extension_stratified_hash.v2"

EXTENSION_CORPUS_NAME = "mask_grounded_region_corpus_train_extension_v2_7950"
TRAIN_COLLECTION_NAME = "mask_grounded_region_train_collection_v2_8450"
SELECTION_SEED = 20260805
SOURCE_ORDER = (
    "gdcld",
    "lmhld",
    "landslidebench_agent",
    "landslide4sense",
    "multimodal_landslide",
)
BASE_COUNT = 500
EXTENSION_COUNT = 7_950
COLLECTION_COUNT = 8_450
BASE_PER_SOURCE = 100
EXTENSION_PER_SOURCE = 1_590
FINAL_PER_SOURCE = 1_690

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
BASE_REGION_ROOT = Path(os.path.abspath(
    REPOSITORY_ROOT.parent
    / "benchmark/oa_grounded_stage4_v1/region_corpus/mask_grounded_region_corpus_train_v1_500"
))
EVAL_DEV_ROOT = Path(os.path.abspath(
    REPOSITORY_ROOT.parent
    / "benchmark/oa_grounded_stage4_v1/eval_dev/oa_grounded_eval_dev_v1_100"
))
EXPANDED_STAGE4_ROOT = Path(os.path.abspath(
    REPOSITORY_ROOT.parent / "benchmark/oa_grounded_stage4_v2"
))
EXTENSION_ROOT = EXPANDED_STAGE4_ROOT / "region_corpus" / EXTENSION_CORPUS_NAME
COLLECTION_ROOT = EXPANDED_STAGE4_ROOT / "region_collection" / TRAIN_COLLECTION_NAME
EXTENSION_CONFIG_PATH = (
    REPOSITORY_ROOT
    / "configs/grounding/region_corpus_train_extension_v2_7950.yaml"
)


@dataclass(frozen=True)
class ArtifactBinding:
    root: Path
    schema_version: str
    manifest_sha256: str
    records_sha256: str
    ledger_sha256: str
    record_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "schema_version": self.schema_version,
            "manifest_sha256": self.manifest_sha256,
            "records_sha256": self.records_sha256,
            "ledger_sha256": self.ledger_sha256,
            "record_count": self.record_count,
        }


@dataclass(frozen=True)
class ExpandedRegionConfig:
    config_path: Path
    config_file_sha256: str
    semantic_sha256: str
    benchmark_root: Path
    benchmark_expected: Mapping[str, str]
    base_corpus: ArtifactBinding
    eval_dev: ArtifactBinding
    eval_baseline_parent_count: int
    extension_root: Path
    collection_root: Path
    source_order: tuple[str, ...]
    seed: int
    final_per_source: int
    base_per_source: int
    extension_per_source: int
    no_target_max_ratio: float
    candidate_pool_multiplier: int
    candidate_pool_min: int
    small_upper_exclusive: float
    medium_upper_exclusive: float
    context_margin_ratio: float
    overlay_alpha: float

    def semantic_dict(self) -> dict[str, Any]:
        return {
            "schema_version": EXTENSION_CONFIG_SCHEMA,
            "benchmark": {
                "root": str(self.benchmark_root),
                **dict(self.benchmark_expected),
            },
            "base_corpus": self.base_corpus.to_dict(),
            "eval_dev": {
                **self.eval_dev.to_dict(),
                "baseline_parent_count": self.eval_baseline_parent_count,
            },
            "outputs": {
                "extension_root": str(self.extension_root),
                "collection_root": str(self.collection_root),
            },
            "selection": {
                "split": "train",
                "seed": self.seed,
                "algorithm_version": EXTENSION_SELECTION_ALGORITHM,
                "source_order": list(self.source_order),
                "final_per_source": self.final_per_source,
                "base_per_source": self.base_per_source,
                "extension_per_source": self.extension_per_source,
                "no_target_max_ratio": self.no_target_max_ratio,
                "candidate_pool_multiplier": self.candidate_pool_multiplier,
                "candidate_pool_min": self.candidate_pool_min,
                "size_bins": {
                    "small_upper_exclusive": self.small_upper_exclusive,
                    "medium_upper_exclusive": self.medium_upper_exclusive,
                },
            },
            "rendering": {
                "context_margin_ratio": self.context_margin_ratio,
                "overlay_alpha": self.overlay_alpha,
                "rgb_renderer": RGB_RENDERER_VERSION,
                "mask_renderer": MASK_RENDERER_VERSION,
            },
        }


@dataclass(frozen=True)
class ExpandedRegionResult:
    extension_root: Path
    collection_root: Path
    extension_manifest_sha256: str
    collection_manifest_sha256: str
    extension_record_count: int
    collection_record_count: int


@dataclass(frozen=True)
class ExpandedCollectionEntry:
    index: Mapping[str, Any]
    record: Mapping[str, Any]
    queue: Mapping[str, Any]
    member_root: Path
    member_manifest_sha256: str


@dataclass(frozen=True)
class ExpandedCollectionContext:
    root: Path
    manifest: Mapping[str, Any]
    manifest_sha256: str
    entries: tuple[ExpandedCollectionEntry, ...]


def _artifact_binding(
    value: Any,
    *,
    base: Path,
    location: str,
    include_baseline_count: bool = False,
) -> tuple[ArtifactBinding, int | None]:
    row = _mapping(value, location)
    fields = (
        "root", "schema_version", "manifest_sha256", "records_sha256",
        "ledger_sha256", "record_count",
    )
    if include_baseline_count:
        fields = (*fields, "baseline_parent_count")
    _exact(row, fields, location)
    binding = ArtifactBinding(
        root=_absolute(base, row["root"], f"{location}.root"),
        schema_version=_string(row["schema_version"], f"{location}.schema_version"),
        manifest_sha256=_sha256(row["manifest_sha256"], f"{location}.manifest_sha256"),
        records_sha256=_sha256(row["records_sha256"], f"{location}.records_sha256"),
        ledger_sha256=_sha256(row["ledger_sha256"], f"{location}.ledger_sha256"),
        record_count=_integer(row["record_count"], f"{location}.record_count", 1),
    )
    baseline_count = (
        _integer(row["baseline_parent_count"], f"{location}.baseline_parent_count", 1)
        if include_baseline_count
        else None
    )
    return binding, baseline_count


def load_expanded_region_config(
    path: Path | str = EXTENSION_CONFIG_PATH,
) -> ExpandedRegionConfig:
    """严格装载固定 7,950+500 配置；test split 在配置层即被拒绝。"""

    config_path = _config_path(path)
    row = _load_yaml(config_path)
    _exact(
        row,
        ("schema_version", "benchmark", "base_corpus", "eval_dev", "outputs", "selection", "rendering"),
        "$",
    )
    if row["schema_version"] != EXTENSION_CONFIG_SCHEMA:
        fail("CONFIG_INVALID", f"仅支持 {EXTENSION_CONFIG_SCHEMA}")
    benchmark_root, benchmark_expected = _benchmark(
        row["benchmark"], base=config_path.parent, location="$.benchmark"
    )
    base_corpus, _ = _artifact_binding(
        row["base_corpus"], base=config_path.parent, location="$.base_corpus"
    )
    eval_dev, eval_parent_count = _artifact_binding(
        row["eval_dev"],
        base=config_path.parent,
        location="$.eval_dev",
        include_baseline_count=True,
    )
    outputs = _mapping(row["outputs"], "$.outputs")
    _exact(outputs, ("extension_root", "collection_root"), "$.outputs")
    selection = _mapping(row["selection"], "$.selection")
    _exact(
        selection,
        (
            "split", "seed", "algorithm_version", "source_order", "final_per_source",
            "base_per_source", "extension_per_source", "no_target_max_ratio",
            "candidate_pool_multiplier", "candidate_pool_min", "size_bins",
        ),
        "$.selection",
    )
    if selection["split"] != "train":
        fail("SPLIT_FORBIDDEN", "扩展 Corpus 只允许 train；val/test 被代码级拒绝")
    sources = selection["source_order"]
    if not isinstance(sources, list) or len(sources) != len(set(sources)):
        fail("CONFIG_INVALID", "source_order 必须为唯一列表")
    source_order = tuple(_string(item, "$.selection.source_order[]") for item in sources)
    size_bins = _mapping(selection["size_bins"], "$.selection.size_bins")
    _exact(
        size_bins,
        ("small_upper_exclusive", "medium_upper_exclusive"),
        "$.selection.size_bins",
    )
    rendering = _mapping(row["rendering"], "$.rendering")
    _exact(rendering, ("context_margin_ratio", "overlay_alpha"), "$.rendering")
    config = ExpandedRegionConfig(
        config_path=config_path,
        config_file_sha256=sha256_file(config_path),
        semantic_sha256="",
        benchmark_root=benchmark_root,
        benchmark_expected=benchmark_expected,
        base_corpus=base_corpus,
        eval_dev=eval_dev,
        eval_baseline_parent_count=int(eval_parent_count or 0),
        extension_root=_absolute(config_path.parent, outputs["extension_root"], "$.outputs.extension_root"),
        collection_root=_absolute(config_path.parent, outputs["collection_root"], "$.outputs.collection_root"),
        source_order=source_order,
        seed=_integer(selection["seed"], "$.selection.seed"),
        final_per_source=_integer(selection["final_per_source"], "$.selection.final_per_source", 1),
        base_per_source=_integer(selection["base_per_source"], "$.selection.base_per_source", 1),
        extension_per_source=_integer(selection["extension_per_source"], "$.selection.extension_per_source", 1),
        no_target_max_ratio=_ratio(selection["no_target_max_ratio"], "$.selection.no_target_max_ratio"),
        candidate_pool_multiplier=_integer(
            selection["candidate_pool_multiplier"], "$.selection.candidate_pool_multiplier", 1
        ),
        candidate_pool_min=_integer(selection["candidate_pool_min"], "$.selection.candidate_pool_min", 1),
        small_upper_exclusive=_ratio(
            size_bins["small_upper_exclusive"], "$.selection.size_bins.small_upper_exclusive"
        ),
        medium_upper_exclusive=_ratio(
            size_bins["medium_upper_exclusive"], "$.selection.size_bins.medium_upper_exclusive"
        ),
        context_margin_ratio=_ratio(rendering["context_margin_ratio"], "$.rendering.context_margin_ratio"),
        overlay_alpha=_ratio(rendering["overlay_alpha"], "$.rendering.overlay_alpha"),
    )
    fixed = (
        config.base_corpus.root == BASE_REGION_ROOT
        and config.base_corpus.schema_version == REGION_MANIFEST_SCHEMA
        and config.base_corpus.record_count == BASE_COUNT
        and config.eval_dev.root == EVAL_DEV_ROOT
        and config.eval_dev.schema_version == EVAL_MANIFEST_SCHEMA
        and config.eval_dev.record_count == 340
        and config.eval_baseline_parent_count == 100
        and config.extension_root == EXTENSION_ROOT
        and config.collection_root == COLLECTION_ROOT
        and config.source_order == SOURCE_ORDER
        and config.seed == SELECTION_SEED
        and config.final_per_source == FINAL_PER_SOURCE
        and config.base_per_source == BASE_PER_SOURCE
        and config.extension_per_source == EXTENSION_PER_SOURCE
        and config.final_per_source == config.base_per_source + config.extension_per_source
        and config.no_target_max_ratio == 0.20
        and config.candidate_pool_multiplier == 2
        and config.candidate_pool_min == 64
        and config.small_upper_exclusive == 0.01
        and config.medium_upper_exclusive == 0.10
        and config.context_margin_ratio == 0.15
        and config.overlay_alpha == 0.45
        and selection["algorithm_version"] == EXTENSION_SELECTION_ALGORITHM
    )
    if not fixed:
        fail("CONFIG_INVALID", "Stage 4 扩展 v2 的路径、数量、seed、分层与渲染合同已冻结")
    return replace(config, semantic_sha256=sha256_text(canonical_json(config.semantic_dict())))


def _record_parent(row: Mapping[str, Any]) -> str:
    return parent_identity(row)[0]


def _balanced_quotas(capacities: Mapping[str, int], total: int) -> dict[str, int]:
    names = ("small", "medium", "large")
    if total < 0 or sum(int(capacities.get(name, 0)) for name in names) < total:
        fail("SELECTION_QUOTA_UNMET", "target size-bin 容量不足")
    quotas = {name: 0 for name in names}
    remaining = total
    while remaining:
        progressed = False
        for name in names:
            if remaining and quotas[name] < int(capacities.get(name, 0)):
                quotas[name] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            fail("SELECTION_QUOTA_UNMET", "target size-bin 无法完成平衡分配")
    return quotas


def _parent_first_pool(
    rows: Sequence[Mapping[str, Any]],
    *,
    limit: int,
    base_parents: set[str],
    seed: int,
    key: str,
) -> list[Mapping[str, Any]]:
    ordered = sorted(rows, key=lambda row: stable_hash(seed, key, row["sample_id"]))
    representatives: list[Mapping[str, Any]] = []
    seen: set[str] = set(base_parents)
    remainder: list[Mapping[str, Any]] = []
    for row in ordered:
        parent = _record_parent(row)
        if parent not in seen:
            seen.add(parent)
            representatives.append(row)
        else:
            remainder.append(row)
    return (representatives + remainder)[:limit]


def _geometry_select(
    rows: Sequence[Mapping[str, Any]],
    *,
    quota: int,
    base_parents: set[str],
    seed: int,
    key: str,
    facts_provider: Callable[[Mapping[str, Any]], Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any]], dict[str, Mapping[str, Any]]]:
    candidates: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    for row in rows:
        facts = dict(facts_provider(row))
        component_count = facts.get("component_count", facts.get("fragment_count"))
        location = facts.get("location_3x3")
        if isinstance(component_count, bool) or not isinstance(component_count, int) or component_count < 1:
            fail("SELECTION_IDENTITY_MISMATCH", f"target component_count 非法：{row['sample_id']}")
        if not isinstance(location, str) or not location:
            fail("SELECTION_IDENTITY_MISMATCH", f"target location_3x3 非法：{row['sample_id']}")
        candidates.append((row, {"component_count": component_count, "location_3x3": location}))
    selected: list[Mapping[str, Any]] = []
    selected_facts: dict[str, Mapping[str, Any]] = {}
    selected_parents = set(base_parents)
    component_counts: Counter[str] = Counter()
    location_counts: Counter[str] = Counter()
    remaining = list(candidates)
    while len(selected) < quota:
        if not remaining:
            fail("SELECTION_QUOTA_UNMET", f"几何候选不足：{key}")

        def score(item: tuple[Mapping[str, Any], Mapping[str, Any]]) -> tuple[Any, ...]:
            row, facts = item
            parent = _record_parent(row)
            component = "single" if facts["component_count"] == 1 else "multiple"
            location = str(facts["location_3x3"])
            return (
                0 if parent not in selected_parents else 1,
                component_counts[component],
                location_counts[location],
                stable_hash(seed, key, row["sample_id"]),
            )

        best = min(range(len(remaining)), key=lambda index: score(remaining[index]))
        row, facts = remaining.pop(best)
        selected.append(row)
        selected_facts[str(row["sample_id"])] = facts
        selected_parents.add(_record_parent(row))
        component_counts["single" if facts["component_count"] == 1 else "multiple"] += 1
        location_counts[str(facts["location_3x3"])] += 1
    return selected, selected_facts


def select_extension_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    base_records: Sequence[Mapping[str, Any]],
    eval_parent_ids: set[str],
    source_order: Sequence[str],
    per_source_count: int,
    final_per_source_count: int,
    seed: int,
    no_target_max_ratio: float,
    candidate_pool_multiplier: int,
    candidate_pool_min: int,
    small_upper_exclusive: float,
    medium_upper_exclusive: float,
    facts_provider: Callable[[Mapping[str, Any]], Mapping[str, Any]],
) -> tuple[list[str], dict[str, Any]]:
    """确定性选择扩展 train rows，并在受容量约束时平衡 size/component/location。"""

    if any(row.get("split") != "train" for row in rows):
        fail("SPLIT_FORBIDDEN", "扩展选样输入只能包含 train；明确拒绝 val/test")
    sample_ids = [str(row.get("sample_id", "")) for row in rows]
    if any(not sample_id for sample_id in sample_ids) or len(sample_ids) != len(set(sample_ids)):
        fail("SELECTION_IDENTITY_MISMATCH", "train candidate sample_id 必须非空唯一")
    base_samples = {str(record["sample_id"]) for record in base_records}
    base_parents = {str(record["parent_id"]) for record in base_records}
    selected_ids: list[str] = []
    reports: dict[str, Any] = {}
    for source in source_order:
        source_rows = [
            row for row in rows
            if row.get("source") == source
            and str(row["sample_id"]) not in base_samples
            and _record_parent(row) not in eval_parent_ids
        ]
        if len(source_rows) < per_source_count:
            fail("SELECTION_QUOTA_UNMET", f"{source} 排除 base/eval parent 后不足 {per_source_count}")
        no_target = [row for row in source_rows if float(row.get("foreground_ratio", -1.0)) == 0.0]
        targets = [row for row in source_rows if float(row.get("foreground_ratio", -1.0)) > 0.0]
        if len(no_target) + len(targets) != len(source_rows):
            fail("SELECTION_IDENTITY_MISMATCH", f"{source} foreground_ratio 非法")
        base_no_target_count = sum(
            record.get("source") == source
            and record.get("target_status") == RegionTargetStatus.NO_TARGET.value
            for record in base_records
        )
        final_no_target_cap = int(final_per_source_count * no_target_max_ratio)
        extension_no_target_cap = max(0, final_no_target_cap - base_no_target_count)
        no_target_quota = min(len(no_target), extension_no_target_cap)
        target_total = per_source_count - no_target_quota
        bins: dict[str, list[Mapping[str, Any]]] = {name: [] for name in ("small", "medium", "large")}
        for row in targets:
            name = size_bin(
                float(row["foreground_ratio"]),
                small_upper=small_upper_exclusive,
                medium_upper=medium_upper_exclusive,
            )
            if name not in bins:
                fail("SELECTION_IDENTITY_MISMATCH", f"target size bin 非法：{row['sample_id']}")
            bins[name].append(row)
        quotas = _balanced_quotas({name: len(values) for name, values in bins.items()}, target_total)
        selected_source: list[Mapping[str, Any]] = []
        selected_geometry: dict[str, Mapping[str, Any]] = {}
        for name in ("small", "medium", "large"):
            quota = quotas[name]
            pool_limit = min(
                len(bins[name]),
                max(quota, candidate_pool_min, quota * candidate_pool_multiplier),
            )
            pool = _parent_first_pool(
                bins[name],
                limit=pool_limit,
                base_parents=base_parents,
                seed=seed,
                key=f"{source}:{name}:pool",
            )
            chosen, facts = _geometry_select(
                pool,
                quota=quota,
                base_parents=base_parents | {_record_parent(row) for row in selected_source},
                seed=seed,
                key=f"{source}:{name}:select",
                facts_provider=facts_provider,
            )
            selected_source.extend(chosen)
            selected_geometry.update(facts)
        no_target_pool = _parent_first_pool(
            no_target,
            limit=no_target_quota,
            base_parents=base_parents | {_record_parent(row) for row in selected_source},
            seed=seed,
            key=f"{source}:no_target",
        )
        selected_source.extend(no_target_pool)
        if len(selected_source) != per_source_count:
            fail("SELECTION_QUOTA_UNMET", f"{source} 实际选择数量错误")
        source_ids = [str(row["sample_id"]) for row in selected_source]
        if len(source_ids) != len(set(source_ids)):
            fail("SELECTION_IDENTITY_MISMATCH", f"{source} 选择结果重复")
        selected_ids.extend(source_ids)
        component_counts = Counter(
            "single" if facts["component_count"] == 1 else "multiple"
            for facts in selected_geometry.values()
        )
        location_counts = Counter(str(facts["location_3x3"]) for facts in selected_geometry.values())
        reports[source] = {
            "eligible_count": len(source_rows),
            "excluded_base_sample_count": sum(
                row.get("source") == source and str(row["sample_id"]) in base_samples for row in rows
            ),
            "excluded_eval_parent_count": sum(
                row.get("source") == source and _record_parent(row) in eval_parent_ids for row in rows
            ),
            "selected_count": len(selected_source),
            "target_count": target_total,
            "no_target_count": no_target_quota,
            "base_no_target_count": base_no_target_count,
            "final_no_target_count": base_no_target_count + no_target_quota,
            "final_no_target_cap": final_no_target_cap,
            "size_counts": quotas,
            "component_counts": dict(sorted(component_counts.items())),
            "location_3x3_counts": dict(sorted(location_counts.items())),
        }
    if len(selected_ids) != per_source_count * len(source_order) or len(selected_ids) != len(set(selected_ids)):
        fail("SELECTION_IDENTITY_MISMATCH", "扩展选择全局数量或唯一性错误")
    return selected_ids, reports


def _ordinary_root(root: Path, *, location: str) -> Path:
    result = Path(os.path.abspath(root))
    if first_symlink_component(result) is not None or not result.is_dir():
        fail("CORPUS_INVALID", f"{location} 必须是普通目录：{result}")
    return result


def _check_binding(binding: ArtifactBinding, report: Mapping[str, Any], *, location: str) -> None:
    expected = {
        "manifest_sha256": binding.manifest_sha256,
        "records_sha256": binding.records_sha256,
        "ledger_sha256": binding.ledger_sha256,
        "record_count": binding.record_count,
    }
    actual = {field: report.get(field) for field in expected}
    if actual != expected:
        fail("SELECTION_IDENTITY_MISMATCH", f"{location} 身份漂移", details={"expected": expected, "actual": actual})
    manifest = read_json(binding.root / "manifest.json")
    if not isinstance(manifest, dict) or manifest.get("schema_version") != binding.schema_version:
        fail("SCHEMA_MISMATCH", f"{location} schema_version 漂移")


def _validate_inputs(
    config: ExpandedRegionConfig,
    *,
    verify_source: bool,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], set[str]]:
    from ..region_validation import validate_eval_dev, validate_region_corpus

    base_report = validate_region_corpus(config.base_corpus.root, verify_source=verify_source)
    _check_binding(config.base_corpus, base_report, location="base_corpus")
    eval_report = validate_eval_dev(
        config.eval_dev.root,
        train_corpus_root=config.base_corpus.root,
        # train Corpus 只消费已发布 Eval-dev 的身份与 parent 清单；不得借验证排除项
        # 打开 OA Benchmark 的 val shard。Eval 的 source 验证由独立 Eval validator 完成。
        verify_source=False,
    )
    _check_binding(config.eval_dev, eval_report, location="eval_dev")
    eval_manifest = _mapping(read_json(config.eval_dev.root / "manifest.json"), "eval.manifest")
    eval_records = read_jsonl(config.eval_dev.root / "records.jsonl")
    baseline_ids = set(eval_manifest["selection"]["baseline_record_ids"])
    baselines = [record for record in eval_records if record.get("record_id") in baseline_ids]
    eval_parents = {str(record["parent_id"]) for record in baselines}
    if len(baselines) != config.eval_baseline_parent_count or len(eval_parents) != config.eval_baseline_parent_count:
        fail("PARENT_OVERLAP", "Eval baseline parent 身份或数量漂移")
    base_records = read_jsonl(config.base_corpus.root / "records.jsonl")
    return base_report, eval_report, base_records, eval_parents


def _artifact_contract(root: Path, schema_version: str, report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "root": str(root),
        "schema_version": schema_version,
        "manifest_sha256": report["manifest_sha256"],
        "records_sha256": report["records_sha256"],
        "ledger_sha256": report["ledger_sha256"],
        "record_count": report["record_count"],
    }


def build_region_extension(
    config_path: Path | str = EXTENSION_CONFIG_PATH,
) -> RegionBuildResult:
    """从 train GT 构建 7,950 条独立资产；从不读取 val/test shard。"""

    config = load_expanded_region_config(config_path)
    if config.extension_root.exists() or config.extension_root.is_symlink():
        fail("OUTPUT_EXISTS", f"扩展 Corpus 已存在，拒绝覆盖：{config.extension_root}")
    base_report, eval_report, base_records, eval_parents = _validate_inputs(config, verify_source=True)
    access = RegionBenchmarkAccess(
        config.benchmark_root,
        config.benchmark_expected,
        allowed_split="train",
        cache_size=8,
    )
    if access.source_order != config.source_order:
        fail("CONFIG_INVALID", "source_order 必须等于 OA manifest included_sources")

    def facts_provider(row: Mapping[str, Any]) -> Mapping[str, Any]:
        facts = access.load(str(row["sample_id"])).mask_facts
        if facts is None:
            fail("MASK_INVALID", f"target candidate mask 为空：{row['sample_id']}")
        return facts

    selected_ids, selection_report = select_extension_rows(
        access.rows,
        base_records=base_records,
        eval_parent_ids=eval_parents,
        source_order=config.source_order,
        per_source_count=config.extension_per_source,
        final_per_source_count=config.final_per_source,
        seed=config.seed,
        no_target_max_ratio=config.no_target_max_ratio,
        candidate_pool_multiplier=config.candidate_pool_multiplier,
        candidate_pool_min=config.candidate_pool_min,
        small_upper_exclusive=config.small_upper_exclusive,
        medium_upper_exclusive=config.medium_upper_exclusive,
        facts_provider=facts_provider,
    )
    with AtomicArtifactDirectory(config.extension_root) as writer:
        assert writer.staging is not None
        records = []
        for index, sample_id in enumerate(selected_ids, 1):
            records.append(materialize_region_record(
                record_key=f"extension_{index:06d}",
                loaded=access.load(sample_id),
                access=access,
                writer=writer,
                context_margin_ratio=config.context_margin_ratio,
                overlay_alpha=config.overlay_alpha,
            ))
        writer.write_jsonl("records.jsonl", records)
        queue = annotation_queue_rows(records, writer.staging)
        writer.write_jsonl("annotation_queue.jsonl", queue)
        writer.write_json("annotation_guideline.json", annotation_guideline())
        assets = [
            relative
            for record in records
            for relative in record["assets"].values()
            if isinstance(relative, str)
        ]
        ledger = ledger_rows(
            writer.staging,
            ("records.jsonl", "annotation_queue.jsonl", "annotation_guideline.json", *assets),
        )
        writer.write_jsonl("SHA256SUMS.jsonl", ledger)
        asset_set = set(assets)
        asset_bytes = sum(item["size_bytes"] for item in ledger if item["path"] in asset_set)
        base_contract = _artifact_contract(config.base_corpus.root, REGION_MANIFEST_SCHEMA, base_report)
        eval_contract = _artifact_contract(config.eval_dev.root, EVAL_MANIFEST_SCHEMA, eval_report)
        manifest = {
            "schema_version": EXTENSION_MANIFEST_SCHEMA,
            "corpus_name": EXTENSION_CORPUS_NAME,
            "corpus_id": sha256_text(canonical_json({
                "benchmark": config.benchmark_expected,
                "config": config.semantic_sha256,
                "base_manifest": base_report["manifest_sha256"],
                "eval_manifest": eval_report["manifest_sha256"],
                "sample_ids": selected_ids,
            })),
            "benchmark": {"root": str(config.benchmark_root), **dict(config.benchmark_expected)},
            "base_corpus": base_contract,
            "eval_dev": eval_contract,
            "eval_exclusion": {
                "baseline_parent_count": len(eval_parents),
                "ordered_parent_ids_sha256": sha256_text(canonical_json(sorted(eval_parents))),
            },
            "selection": {
                "split": "train",
                "algorithm_version": EXTENSION_SELECTION_ALGORITHM,
                "seed": config.seed,
                "config_file_sha256": config.config_file_sha256,
                "config_semantic_sha256": config.semantic_sha256,
                "source_order": list(config.source_order),
                "extension_per_source": config.extension_per_source,
                "no_target_max_ratio": config.no_target_max_ratio,
                "candidate_pool_multiplier": config.candidate_pool_multiplier,
                "candidate_pool_min": config.candidate_pool_min,
                "size_bins": {
                    "small_upper_exclusive": config.small_upper_exclusive,
                    "medium_upper_exclusive": config.medium_upper_exclusive,
                },
                "ordered_sample_ids": selected_ids,
                "ordered_sample_ids_sha256": sha256_text(canonical_json(selected_ids)),
                "sample_count": len(selected_ids),
                "source_reports": selection_report,
            },
            "rendering": {
                "context_margin_ratio": config.context_margin_ratio,
                "overlay_alpha": config.overlay_alpha,
                "rgb_renderer": RGB_RENDERER_VERSION,
                "mask_renderer": MASK_RENDERER_VERSION,
            },
            "records": {**next(item for item in ledger if item["path"] == "records.jsonl"), "count": len(records)},
            "annotation_queue": {
                **next(item for item in ledger if item["path"] == "annotation_queue.jsonl"),
                "count": len(queue),
            },
            "assets": {
                "count": len(asset_set),
                "total_bytes": asset_bytes,
                "role_counts": dict(sorted(Counter(path.split("/")[1] for path in asset_set).items())),
            },
            "ledger": {
                "path": "SHA256SUMS.jsonl",
                "entry_count": len(ledger),
                "size_bytes": writer.path("SHA256SUMS.jsonl").stat().st_size,
                "file_sha256": sha256_file(writer.path("SHA256SUMS.jsonl")),
                "root_sha256": sha256_text(canonical_json(ledger)),
            },
            "silver_generated": False,
            "expert_annotation_completed": False,
            "formal_acceptance": False,
        }
        writer.write_json("manifest.json", manifest)
        validate_region_extension(
            writer.staging,
            config_path=config.config_path,
            verify_source=True,
        )
        root = writer.publish()
    return RegionBuildResult(
        root=root,
        manifest_sha256=sha256_file(root / "manifest.json"),
        records_sha256=sha256_file(root / "records.jsonl"),
        ledger_sha256=sha256_file(root / "SHA256SUMS.jsonl"),
        record_count=len(records),
        asset_count=len(asset_set),
        asset_bytes=asset_bytes,
    )


def _manifest_artifact_binding(value: Any, *, location: str) -> ArtifactBinding:
    row = _mapping(value, location)
    _exact(
        row,
        ("root", "schema_version", "manifest_sha256", "records_sha256", "ledger_sha256", "record_count"),
        location,
    )
    root_text = _string(row["root"], f"{location}.root")
    if not Path(root_text).is_absolute():
        fail("PATH_ABSOLUTE", f"{location}.root 必须是绝对路径")
    return ArtifactBinding(
        root=Path(os.path.abspath(root_text)),
        schema_version=_string(row["schema_version"], f"{location}.schema_version"),
        manifest_sha256=_sha256(row["manifest_sha256"], f"{location}.manifest_sha256"),
        records_sha256=_sha256(row["records_sha256"], f"{location}.records_sha256"),
        ledger_sha256=_sha256(row["ledger_sha256"], f"{location}.ledger_sha256"),
        record_count=_integer(row["record_count"], f"{location}.record_count", 1),
    )


def validate_region_extension(
    root: Path | str,
    *,
    config_path: Path | str | None = None,
    verify_source: bool = True,
) -> dict[str, Any]:
    """重算扩展选择、GT 几何、资产和 base/Eval 排除身份。"""

    from ..region_validation import _validate_common, validate_eval_dev, validate_region_corpus

    resolved = _ordinary_root(Path(root), location="extension root")
    manifest, ledger, records, assets, access = _validate_common(
        resolved,
        expected_schema=EXTENSION_MANIFEST_SCHEMA,
        split="train",
        verify_source=verify_source,
    )
    _exact(
        manifest,
        (
            "schema_version", "corpus_name", "corpus_id", "benchmark", "base_corpus",
            "eval_dev", "eval_exclusion", "selection", "rendering", "records",
            "annotation_queue", "assets", "ledger", "silver_generated",
            "expert_annotation_completed", "formal_acceptance",
        ),
        "extension.manifest",
    )
    if (
        manifest["corpus_name"] != EXTENSION_CORPUS_NAME
        or manifest["silver_generated"] is not False
        or manifest["expert_annotation_completed"] is not False
        or manifest["formal_acceptance"] is not False
    ):
        fail("SCHEMA_MISMATCH", "扩展 Corpus 名称或冻结 false 状态非法")
    base_binding = _manifest_artifact_binding(manifest["base_corpus"], location="manifest.base_corpus")
    eval_binding = _manifest_artifact_binding(manifest["eval_dev"], location="manifest.eval_dev")
    if base_binding.schema_version != REGION_MANIFEST_SCHEMA or eval_binding.schema_version != EVAL_MANIFEST_SCHEMA:
        fail("SCHEMA_MISMATCH", "扩展 Corpus base/eval schema 非法")
    base_report = validate_region_corpus(base_binding.root, verify_source=verify_source)
    _check_binding(base_binding, base_report, location="manifest.base_corpus")
    eval_report = validate_eval_dev(
        eval_binding.root,
        train_corpus_root=base_binding.root,
        # 扩展 Corpus 的深验只重放 train GT；Eval-dev 在这里始终作为冻结排除合同读取。
        verify_source=False,
    )
    _check_binding(eval_binding, eval_report, location="manifest.eval_dev")
    selection = _mapping(manifest["selection"], "manifest.selection")
    _exact(
        selection,
        (
            "split", "algorithm_version", "seed", "config_file_sha256", "config_semantic_sha256",
            "source_order", "extension_per_source", "no_target_max_ratio",
            "candidate_pool_multiplier", "candidate_pool_min", "size_bins",
            "ordered_sample_ids", "ordered_sample_ids_sha256", "sample_count", "source_reports",
        ),
        "manifest.selection",
    )
    ids = selection["ordered_sample_ids"]
    if (
        selection["split"] != "train"
        or selection["algorithm_version"] != EXTENSION_SELECTION_ALGORITHM
        or selection["seed"] != SELECTION_SEED
        or selection["source_order"] != list(SOURCE_ORDER)
        or selection["extension_per_source"] != EXTENSION_PER_SOURCE
        or selection["sample_count"] != EXTENSION_COUNT
        or not isinstance(ids, list)
        or len(ids) != EXTENSION_COUNT
        or len(ids) != len(set(ids))
        or [record["sample_id"] for record in records] != ids
        or sha256_text(canonical_json(ids)) != selection["ordered_sample_ids_sha256"]
    ):
        fail("SELECTION_IDENTITY_MISMATCH", "扩展 Corpus ordered selection 身份非法")
    if Counter(record["source"] for record in records) != Counter({source: EXTENSION_PER_SOURCE for source in SOURCE_ORDER}):
        fail("SELECTION_QUOTA_UNMET", "扩展 Corpus 每来源必须精确 1,590 条")
    base_records = read_jsonl(base_binding.root / "records.jsonl")
    base_samples = {record["sample_id"] for record in base_records}
    eval_manifest = read_json(eval_binding.root / "manifest.json")
    eval_records = read_jsonl(eval_binding.root / "records.jsonl")
    baseline_ids = set(eval_manifest["selection"]["baseline_record_ids"])
    eval_parents = {record["parent_id"] for record in eval_records if record["record_id"] in baseline_ids}
    exclusion = _mapping(manifest["eval_exclusion"], "manifest.eval_exclusion")
    _exact(exclusion, ("baseline_parent_count", "ordered_parent_ids_sha256"), "manifest.eval_exclusion")
    if exclusion != {
        "baseline_parent_count": len(eval_parents),
        "ordered_parent_ids_sha256": sha256_text(canonical_json(sorted(eval_parents))),
    }:
        fail("SELECTION_IDENTITY_MISMATCH", "Eval parent exclusion 身份漂移")
    if base_samples & set(ids):
        fail("SELECTION_IDENTITY_MISMATCH", "extension 与 base sample 重复")
    if eval_parents & {record["parent_id"] for record in records}:
        fail("PARENT_OVERLAP", "extension 与 Eval baseline parent 相交")
    config = load_expanded_region_config(config_path or EXTENSION_CONFIG_PATH)
    # staging root 与最终 root 不同，因此只比较 manifest 中冻结的配置语义与成员身份。
    if (
        selection["config_file_sha256"] != config.config_file_sha256
        or selection["config_semantic_sha256"] != config.semantic_sha256
        or base_binding != config.base_corpus
        or eval_binding != config.eval_dev
        or selection["no_target_max_ratio"] != config.no_target_max_ratio
        or selection["candidate_pool_multiplier"] != config.candidate_pool_multiplier
        or selection["candidate_pool_min"] != config.candidate_pool_min
        or selection["size_bins"] != {
            "small_upper_exclusive": config.small_upper_exclusive,
            "medium_upper_exclusive": config.medium_upper_exclusive,
        }
    ):
        fail("CONFIG_IDENTITY_MISMATCH", "扩展 Corpus 与冻结配置身份不一致")
    reports = _mapping(selection["source_reports"], "manifest.selection.source_reports")
    _exact(reports, SOURCE_ORDER, "manifest.selection.source_reports")
    for source in SOURCE_ORDER:
        report = _mapping(reports[source], f"manifest.selection.source_reports.{source}")
        _exact(
            report,
            (
                "eligible_count", "excluded_base_sample_count", "excluded_eval_parent_count",
                "selected_count", "target_count", "no_target_count", "base_no_target_count",
                "final_no_target_count", "final_no_target_cap", "size_counts",
                "component_counts", "location_3x3_counts",
            ),
            f"manifest.selection.source_reports.{source}",
        )
        source_records = [record for record in records if record["source"] == source]
        targets = [
            record for record in source_records
            if record["target_status"] == RegionTargetStatus.TARGET_PRESENT.value
        ]
        no_targets = len(source_records) - len(targets)
        base_no_targets = sum(
            record["source"] == source
            and record["target_status"] == RegionTargetStatus.NO_TARGET.value
            for record in base_records
        )
        expected_selected = {
            "selected_count": len(source_records),
            "target_count": len(targets),
            "no_target_count": no_targets,
            "base_no_target_count": base_no_targets,
            "final_no_target_count": base_no_targets + no_targets,
            "final_no_target_cap": int(FINAL_PER_SOURCE * config.no_target_max_ratio),
            "size_counts": dict(sorted(Counter(
                size_bin(
                    float(record["program_facts"]["mask"]["area_ratio"]),
                    small_upper=config.small_upper_exclusive,
                    medium_upper=config.medium_upper_exclusive,
                )
                for record in targets
            ).items())),
            "component_counts": dict(sorted(Counter(
                "single" if record["program_facts"]["mask"]["fragment_count"] == 1 else "multiple"
                for record in targets
            ).items())),
            "location_3x3_counts": dict(sorted(Counter(
                str(record["program_facts"]["mask"]["location_3x3"])
                for record in targets
            ).items())),
        }
        if any(report[field] != value for field, value in expected_selected.items()):
            fail("SELECTION_IDENTITY_MISMATCH", f"{source} 选样分布与 records 不一致")
        if base_no_targets + no_targets > int(FINAL_PER_SOURCE * config.no_target_max_ratio):
            fail("SELECTION_QUOTA_UNMET", f"{source} 最终 no-target 超过 20% 上限")
    if verify_source:
        if access is None:
            fail("BENCHMARK_INVALID", "verify_source=true 时缺少 train access")

        def facts_provider(row: Mapping[str, Any]) -> Mapping[str, Any]:
            facts = access.load(str(row["sample_id"])).mask_facts
            if facts is None:
                fail("MASK_INVALID", f"target candidate mask 为空：{row['sample_id']}")
            return facts

        recomputed, reports = select_extension_rows(
            access.rows,
            base_records=base_records,
            eval_parent_ids=eval_parents,
            source_order=SOURCE_ORDER,
            per_source_count=EXTENSION_PER_SOURCE,
            final_per_source_count=FINAL_PER_SOURCE,
            seed=SELECTION_SEED,
            no_target_max_ratio=float(selection["no_target_max_ratio"]),
            candidate_pool_multiplier=int(selection["candidate_pool_multiplier"]),
            candidate_pool_min=int(selection["candidate_pool_min"]),
            small_upper_exclusive=float(selection["size_bins"]["small_upper_exclusive"]),
            medium_upper_exclusive=float(selection["size_bins"]["medium_upper_exclusive"]),
            facts_provider=facts_provider,
        )
        if recomputed != ids or canonical_json(reports) != canonical_json(selection["source_reports"]):
            fail("SELECTION_IDENTITY_MISMATCH", "扩展 Corpus 选样无法确定性重放")
    expected_corpus_id = sha256_text(canonical_json({
        "benchmark": {field: manifest["benchmark"][field] for field in EXPECTED_IDENTITY_FIELDS},
        "config": selection["config_semantic_sha256"],
        "base_manifest": base_report["manifest_sha256"],
        "eval_manifest": eval_report["manifest_sha256"],
        "sample_ids": ids,
    }))
    if manifest["corpus_id"] != expected_corpus_id:
        fail("SELECTION_IDENTITY_MISMATCH", "扩展 Corpus corpus_id 重算不一致")
    return {
        "valid": True,
        "root": str(resolved),
        "manifest_sha256": sha256_file(resolved / "manifest.json"),
        "records_sha256": sha256_file(resolved / "records.jsonl"),
        "ledger_sha256": sha256_file(resolved / "SHA256SUMS.jsonl"),
        "record_count": len(records),
        "asset_count": len(assets),
        "asset_bytes": sum(item["size_bytes"] for item in ledger if item["path"] in assets),
        "source_counts": dict(sorted(Counter(record["source"] for record in records).items())),
        "source_verified": verify_source,
    }


def _collection_member_rows(
    *,
    base_root: Path,
    extension_root: Path,
    source_order: Sequence[str],
) -> list[dict[str, Any]]:
    member_data = {
        "base": (base_root, read_jsonl(base_root / "records.jsonl"), read_jsonl(base_root / "annotation_queue.jsonl")),
        "extension": (
            extension_root,
            read_jsonl(extension_root / "records.jsonl"),
            read_jsonl(extension_root / "annotation_queue.jsonl"),
        ),
    }
    queue_by_member = {
        name: {row["record_id"]: row for row in queue}
        for name, (_, _, queue) in member_data.items()
    }
    rows: list[dict[str, Any]] = []
    for source in source_order:
        for member in ("base", "extension"):
            root, records, _ = member_data[member]
            for record in records:
                if record["source"] != source:
                    continue
                queue = queue_by_member[member].get(record["record_id"])
                if queue is None or queue.get("asset_identity_sha256") != region_asset_identity(root, record["assets"]):
                    fail("ANNOTATION_INVALID", f"{member}:{record['record_id']} queue/asset identity 不一致")
                rows.append({
                    "schema_version": TRAIN_COLLECTION_MEMBER_SCHEMA,
                    "ordinal": len(rows),
                    "member": member,
                    "record_id": record["record_id"],
                    "sample_id": record["sample_id"],
                    "parent_id": record["parent_id"],
                    "source": record["source"],
                    "target_status": record["target_status"],
                    "record_sha256": sha256_text(canonical_json(record)),
                    "asset_identity_sha256": queue["asset_identity_sha256"],
                })
    return rows


def build_train_collection(
    config_path: Path | str = EXTENSION_CONFIG_PATH,
) -> dict[str, Any]:
    """发布只读 base+extension 组合索引；不复制任一成员资产。"""

    config = load_expanded_region_config(config_path)
    if config.collection_root.exists() or config.collection_root.is_symlink():
        fail("OUTPUT_EXISTS", f"train collection 已存在，拒绝覆盖：{config.collection_root}")
    base_report, eval_report, base_records, eval_parents = _validate_inputs(config, verify_source=False)
    extension_report = validate_region_extension(
        config.extension_root,
        config_path=config.config_path,
        verify_source=False,
    )
    member_rows = _collection_member_rows(
        base_root=config.base_corpus.root,
        extension_root=config.extension_root,
        source_order=config.source_order,
    )
    if len(member_rows) != COLLECTION_COUNT:
        fail("SELECTION_QUOTA_UNMET", "collection 必须精确 8,450 条")
    with AtomicArtifactDirectory(config.collection_root) as writer:
        assert writer.staging is not None
        writer.write_jsonl("members.jsonl", member_rows)
        ledger = ledger_rows(writer.staging, ("members.jsonl",))
        writer.write_jsonl("SHA256SUMS.jsonl", ledger)
        member_contracts = {
            "base": _artifact_contract(config.base_corpus.root, REGION_MANIFEST_SCHEMA, base_report),
            "extension": _artifact_contract(config.extension_root, EXTENSION_MANIFEST_SCHEMA, extension_report),
        }
        manifest = {
            "schema_version": TRAIN_COLLECTION_SCHEMA,
            "collection_name": TRAIN_COLLECTION_NAME,
            "collection_id": sha256_text(canonical_json({
                "members": member_contracts,
                "eval_manifest": eval_report["manifest_sha256"],
                "ordered_record_ids": [row["record_id"] for row in member_rows],
            })),
            "split": "train",
            "intended_use": "model_assisted_training_supervision",
            "members": member_contracts,
            "eval_exclusion": {
                **_artifact_contract(config.eval_dev.root, EVAL_MANIFEST_SCHEMA, eval_report),
                "baseline_parent_count": len(eval_parents),
                "ordered_parent_ids_sha256": sha256_text(canonical_json(sorted(eval_parents))),
            },
            "selection": {
                "algorithm_version": EXTENSION_SELECTION_ALGORITHM,
                "seed": config.seed,
                "source_order": list(config.source_order),
                "base_per_source": config.base_per_source,
                "extension_per_source": config.extension_per_source,
                "final_per_source": config.final_per_source,
                "record_count": len(member_rows),
                "ordered_record_ids_sha256": sha256_text(canonical_json([row["record_id"] for row in member_rows])),
                "ordered_sample_ids_sha256": sha256_text(canonical_json([row["sample_id"] for row in member_rows])),
                "source_counts": dict(sorted(Counter(row["source"] for row in member_rows).items())),
                "member_counts": dict(sorted(Counter(row["member"] for row in member_rows).items())),
                "target_counts": dict(sorted(Counter(row["target_status"] for row in member_rows).items())),
            },
            "member_index": {**ledger[0], "count": len(member_rows)},
            "ledger": {
                "path": "SHA256SUMS.jsonl",
                "entry_count": len(ledger),
                "size_bytes": writer.path("SHA256SUMS.jsonl").stat().st_size,
                "file_sha256": sha256_file(writer.path("SHA256SUMS.jsonl")),
                "root_sha256": sha256_text(canonical_json(ledger)),
            },
            "reference_authority": "mixed_model_and_single_expert",
            "gold": False,
            "scientific_acceptance": False,
            "formal_acceptance": False,
        }
        writer.write_json("manifest.json", manifest)
        validate_expanded_region_collection(writer.staging, verify_members=True, verify_source=False)
        root = writer.publish()
    return {
        "valid": True,
        "root": str(root),
        "manifest_sha256": sha256_file(root / "manifest.json"),
        "ledger_sha256": sha256_file(root / "SHA256SUMS.jsonl"),
        "record_count": len(member_rows),
    }


def _load_member_payload(binding: ArtifactBinding) -> tuple[dict[str, Any], dict[str, Any]]:
    records = read_jsonl(binding.root / "records.jsonl")
    queue = read_jsonl(binding.root / "annotation_queue.jsonl")
    by_record = {str(row["record_id"]): row for row in records}
    by_queue = {str(row["record_id"]): row for row in queue}
    if len(by_record) != len(records) or len(by_queue) != len(queue) or set(by_record) != set(by_queue):
        fail("ANNOTATION_INVALID", f"member records/queue 身份不一致：{binding.root}")
    return by_record, by_queue


def _resolve_collection_entries(
    root: Path,
    manifest: Mapping[str, Any],
    member_rows: Sequence[Mapping[str, Any]],
) -> tuple[ExpandedCollectionEntry, ...]:
    del root
    member_contracts = _mapping(manifest["members"], "manifest.members")
    _exact(member_contracts, ("base", "extension"), "manifest.members")
    bindings = {
        name: _manifest_artifact_binding(value, location=f"manifest.members.{name}")
        for name, value in member_contracts.items()
    }
    payloads = {name: _load_member_payload(binding) for name, binding in bindings.items()}
    entries: list[ExpandedCollectionEntry] = []
    expected_fields = (
        "schema_version", "ordinal", "member", "record_id", "sample_id", "parent_id",
        "source", "target_status", "record_sha256", "asset_identity_sha256",
    )
    for index, value in enumerate(member_rows):
        row = _mapping(value, f"members[{index}]")
        _exact(row, expected_fields, f"members[{index}]")
        if row["schema_version"] != TRAIN_COLLECTION_MEMBER_SCHEMA or row["ordinal"] != index:
            fail("SCHEMA_MISMATCH", f"members[{index}] schema/ordinal 非法")
        member = row["member"]
        if member not in bindings:
            fail("INVALID_ENUM", f"members[{index}].member 非法")
        records, queues = payloads[str(member)]
        record = records.get(str(row["record_id"]))
        queue = queues.get(str(row["record_id"]))
        if record is None or queue is None:
            fail("SELECTION_IDENTITY_MISMATCH", f"members[{index}] 引用未知 record")
        expected = {
            "schema_version": TRAIN_COLLECTION_MEMBER_SCHEMA,
            "ordinal": index,
            "member": member,
            "record_id": record["record_id"],
            "sample_id": record["sample_id"],
            "parent_id": record["parent_id"],
            "source": record["source"],
            "target_status": record["target_status"],
            "record_sha256": sha256_text(canonical_json(record)),
            "asset_identity_sha256": queue["asset_identity_sha256"],
        }
        if canonical_json(row) != canonical_json(expected):
            fail("SELECTION_IDENTITY_MISMATCH", f"members[{index}] 与成员 record/queue 不一致")
        entries.append(ExpandedCollectionEntry(
            index=row,
            record=record,
            queue=queue,
            member_root=bindings[str(member)].root,
            member_manifest_sha256=bindings[str(member)].manifest_sha256,
        ))
    return tuple(entries)


def validate_expanded_region_collection(
    root: Path | str,
    *,
    verify_members: bool = True,
    verify_source: bool = False,
) -> dict[str, Any]:
    """验证组合索引及两个成员根；collection 自身不拥有或复制图像资产。"""

    from ..region_validation import _validate_files, validate_eval_dev, validate_region_corpus

    resolved = _ordinary_root(Path(root), location="collection root")
    manifest, ledger, ledger_files = _validate_files(resolved)
    _exact(
        manifest,
        (
            "schema_version", "collection_name", "collection_id", "split", "intended_use",
            "members", "eval_exclusion", "selection", "member_index", "ledger",
            "reference_authority", "gold", "scientific_acceptance", "formal_acceptance",
        ),
        "collection.manifest",
    )
    if (
        manifest["schema_version"] != TRAIN_COLLECTION_SCHEMA
        or manifest["collection_name"] != TRAIN_COLLECTION_NAME
        or manifest["split"] != "train"
        or manifest["intended_use"] != "model_assisted_training_supervision"
        or manifest["reference_authority"] != "mixed_model_and_single_expert"
        or manifest["gold"] is not False
        or manifest["scientific_acceptance"] is not False
        or manifest["formal_acceptance"] is not False
    ):
        fail("SCHEMA_MISMATCH", "train collection 固定 schema/split/接受状态非法")
    if ledger_files != {"members.jsonl"}:
        fail("LEDGER_FILESET_MISMATCH", "collection ledger 只能绑定 members.jsonl")
    member_rows = read_jsonl(resolved / "members.jsonl")
    members_item = next(item for item in ledger if item["path"] == "members.jsonl")
    if manifest["member_index"] != {**members_item, "count": len(member_rows)}:
        fail("LEDGER_MISMATCH", "collection member_index 身份不一致")
    bindings_row = _mapping(manifest["members"], "manifest.members")
    _exact(bindings_row, ("base", "extension"), "manifest.members")
    base_binding = _manifest_artifact_binding(bindings_row["base"], location="manifest.members.base")
    extension_binding = _manifest_artifact_binding(
        bindings_row["extension"], location="manifest.members.extension"
    )
    if base_binding.schema_version != REGION_MANIFEST_SCHEMA or extension_binding.schema_version != EXTENSION_MANIFEST_SCHEMA:
        fail("SCHEMA_MISMATCH", "collection member schema 非法")
    if base_binding.root != BASE_REGION_ROOT or extension_binding.root != EXTENSION_ROOT:
        fail("SELECTION_IDENTITY_MISMATCH", "collection member root 不是冻结 v1 base/v2 extension")
    if verify_members:
        base_report = validate_region_corpus(base_binding.root, verify_source=verify_source)
        extension_report = validate_region_extension(extension_binding.root, verify_source=verify_source)
        _check_binding(base_binding, base_report, location="collection.members.base")
        _check_binding(extension_binding, extension_report, location="collection.members.extension")
    entries = _resolve_collection_entries(resolved, manifest, member_rows)
    if len(entries) != COLLECTION_COUNT:
        fail("SELECTION_QUOTA_UNMET", "train collection 必须精确 8,450 条")
    ids = [entry.index["record_id"] for entry in entries]
    samples = [entry.index["sample_id"] for entry in entries]
    if len(ids) != len(set(ids)) or len(samples) != len(set(samples)):
        fail("SELECTION_IDENTITY_MISMATCH", "collection record/sample 必须全局唯一")
    selection = _mapping(manifest["selection"], "manifest.selection")
    _exact(
        selection,
        (
            "algorithm_version", "seed", "source_order", "base_per_source",
            "extension_per_source", "final_per_source", "record_count",
            "ordered_record_ids_sha256", "ordered_sample_ids_sha256", "source_counts",
            "member_counts", "target_counts",
        ),
        "manifest.selection",
    )
    expected_selection = {
        "algorithm_version": EXTENSION_SELECTION_ALGORITHM,
        "seed": SELECTION_SEED,
        "source_order": list(SOURCE_ORDER),
        "base_per_source": BASE_PER_SOURCE,
        "extension_per_source": EXTENSION_PER_SOURCE,
        "final_per_source": FINAL_PER_SOURCE,
        "record_count": COLLECTION_COUNT,
        "ordered_record_ids_sha256": sha256_text(canonical_json(ids)),
        "ordered_sample_ids_sha256": sha256_text(canonical_json(samples)),
        "source_counts": dict(sorted(Counter(entry.index["source"] for entry in entries).items())),
        "member_counts": dict(sorted(Counter(entry.index["member"] for entry in entries).items())),
        "target_counts": dict(sorted(Counter(entry.index["target_status"] for entry in entries).items())),
    }
    if canonical_json(selection) != canonical_json(expected_selection):
        fail("SELECTION_IDENTITY_MISMATCH", "collection selection 身份不一致")
    if expected_selection["source_counts"] != {source: FINAL_PER_SOURCE for source in SOURCE_ORDER}:
        fail("SELECTION_QUOTA_UNMET", "collection 每来源必须精确 1,690 条")
    if expected_selection["member_counts"] != {"base": BASE_COUNT, "extension": EXTENSION_COUNT}:
        fail("SELECTION_QUOTA_UNMET", "collection base/extension 数量非法")
    expected_member_sequence = [
        (source, member)
        for source in SOURCE_ORDER
        for member, count in (("base", BASE_PER_SOURCE), ("extension", EXTENSION_PER_SOURCE))
        for _ in range(count)
    ]
    if [(entry.index["source"], entry.index["member"]) for entry in entries] != expected_member_sequence:
        fail("SELECTION_IDENTITY_MISMATCH", "collection 必须按 source 内 base→extension 固定排序")
    eval_binding_row = _mapping(manifest["eval_exclusion"], "manifest.eval_exclusion")
    eval_binding = _manifest_artifact_binding(
        {key: value for key, value in eval_binding_row.items() if key not in {
            "baseline_parent_count", "ordered_parent_ids_sha256"
        }},
        location="manifest.eval_exclusion",
    )
    _exact(
        eval_binding_row,
        (
            "root", "schema_version", "manifest_sha256", "records_sha256", "ledger_sha256",
            "record_count", "baseline_parent_count", "ordered_parent_ids_sha256",
        ),
        "manifest.eval_exclusion",
    )
    if eval_binding.schema_version != EVAL_MANIFEST_SCHEMA:
        fail("SCHEMA_MISMATCH", "collection eval exclusion schema 非法")
    if eval_binding.root != EVAL_DEV_ROOT:
        fail("SELECTION_IDENTITY_MISMATCH", "collection Eval exclusion root 非冻结 Eval-dev")
    if verify_members:
        eval_report = validate_eval_dev(
            eval_binding.root,
            train_corpus_root=base_binding.root,
            # collection 是 train-only 索引，不因成员深验而打开 val shard。
            verify_source=False,
        )
        _check_binding(eval_binding, eval_report, location="collection.eval_exclusion")
    eval_manifest = read_json(eval_binding.root / "manifest.json")
    eval_records = read_jsonl(eval_binding.root / "records.jsonl")
    baseline_ids = set(eval_manifest["selection"]["baseline_record_ids"])
    eval_parents = {record["parent_id"] for record in eval_records if record["record_id"] in baseline_ids}
    if eval_binding_row["baseline_parent_count"] != len(eval_parents) or eval_binding_row[
        "ordered_parent_ids_sha256"
    ] != sha256_text(canonical_json(sorted(eval_parents))):
        fail("SELECTION_IDENTITY_MISMATCH", "collection Eval exclusion 身份不一致")
    if eval_parents & {entry.index["parent_id"] for entry in entries}:
        fail("PARENT_OVERLAP", "collection 与 Eval baseline parent 相交")
    expected_collection_id = sha256_text(canonical_json({
        "members": {"base": base_binding.to_dict(), "extension": extension_binding.to_dict()},
        "eval_manifest": eval_binding.manifest_sha256,
        "ordered_record_ids": ids,
    }))
    if manifest["collection_id"] != expected_collection_id:
        fail("SELECTION_IDENTITY_MISMATCH", "collection_id 重算不一致")
    return {
        "valid": True,
        "root": str(resolved),
        "manifest_sha256": sha256_file(resolved / "manifest.json"),
        "ledger_sha256": sha256_file(resolved / "SHA256SUMS.jsonl"),
        "record_count": len(entries),
        "source_counts": expected_selection["source_counts"],
        "member_counts": expected_selection["member_counts"],
        "formal_acceptance": False,
        "source_verified": verify_source,
    }


def load_expanded_collection_context(
    root: Path | str = COLLECTION_ROOT,
    *,
    verify_members: bool = True,
) -> ExpandedCollectionContext:
    """返回每条完整 Region record、queue 与其真实资产根，供逐条 VLM 消费。"""

    resolved = Path(os.path.abspath(Path(root)))
    validate_expanded_region_collection(
        resolved,
        verify_members=verify_members,
        verify_source=False,
    )
    manifest = _mapping(read_json(resolved / "manifest.json"), "collection.manifest")
    entries = _resolve_collection_entries(resolved, manifest, read_jsonl(resolved / "members.jsonl"))
    return ExpandedCollectionContext(
        root=resolved,
        manifest=manifest,
        manifest_sha256=sha256_file(resolved / "manifest.json"),
        entries=entries,
    )


def iter_expanded_collection_records(
    root: Path | str = COLLECTION_ROOT,
    *,
    verify_members: bool = True,
) -> Iterator[ExpandedCollectionEntry]:
    yield from load_expanded_collection_context(root, verify_members=verify_members).entries


def prepare_expanded_region_assets(
    config_path: Path | str = EXTENSION_CONFIG_PATH,
    *,
    verify_source: bool = True,
) -> ExpandedRegionResult:
    """幂等准备两个根：有效根复用，无效根拒绝覆盖，允许 collection 阶段断点续作。"""

    config = load_expanded_region_config(config_path)
    if (
        (config.collection_root.exists() or config.collection_root.is_symlink())
        and not (config.extension_root.exists() or config.extension_root.is_symlink())
    ):
        fail("SELECTION_IDENTITY_MISMATCH", "collection 已存在但其 extension 成员缺失，拒绝补造")
    if config.extension_root.exists() or config.extension_root.is_symlink():
        extension_report = validate_region_extension(
            config.extension_root,
            config_path=config.config_path,
            verify_source=verify_source,
        )
    else:
        built = build_region_extension(config.config_path)
        extension_report = {
            "manifest_sha256": built.manifest_sha256,
            "record_count": built.record_count,
        }
    if config.collection_root.exists() or config.collection_root.is_symlink():
        collection_report = validate_expanded_region_collection(
            config.collection_root,
            verify_members=True,
            verify_source=verify_source,
        )
    else:
        collection_report = build_train_collection(config.config_path)
    return ExpandedRegionResult(
        extension_root=config.extension_root,
        collection_root=config.collection_root,
        extension_manifest_sha256=str(extension_report["manifest_sha256"]),
        collection_manifest_sha256=str(collection_report["manifest_sha256"]),
        extension_record_count=int(extension_report["record_count"]),
        collection_record_count=int(collection_report["record_count"]),
    )
