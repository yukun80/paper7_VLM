"""Stage 4 v2 train-only Region Corpus 构建与共享 OA Benchmark 只读访问。"""

from __future__ import annotations

import math
import os
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from PIL import Image

from oa_groundrag.phase2.data import registry_from_benchmark
from oa_groundrag.phase3.common import (
    canonical_json,
    first_symlink_component,
    portable_relative_path,
    read_json,
    read_jsonl,
    safe_join,
    sha256_file,
    sha256_text,
)
from oa_groundrag.phase4.artifacts import AtomicArtifactDirectory
from oa_groundrag.phase4.contracts import MaskMode, RegionInventory, SelectionMode, SelectionRequest
from oa_groundrag.phase4.evidence import (
    context_crop_window,
    deterministic_mask_facts,
    render_binary_mask,
    render_context_crop,
    render_mask_overlay,
)
from oa_groundrag.phase4.regions import RegionSelector
from scripts.phase1_benchmark_build.benchmark_common import BenchmarkDataset

from .contracts import EXPECTED_IDENTITY_FIELDS, fail, no_target_mask_facts
from .pipeline import render_optical
from .region_contracts import (
    ANNOTATION_QUEUE_SCHEMA,
    COORDINATE_BASIS,
    FORMAL_ROLES,
    MASK_RENDERER_VERSION,
    REGION_MANIFEST_SCHEMA,
    REGION_RECORD_SCHEMA,
    REGION_SELECTION_ALGORITHM,
    RGB_RENDERER_VERSION,
    RegionCorpusConfig,
    RegionTargetStatus,
    RepresentationMode,
    annotation_state_template,
    empty_description_template,
    load_region_corpus_config,
    parent_identity,
    validate_region_record,
)


REGION_CORPUS_NAME = "mask_grounded_region_corpus_train_v1_500"


@dataclass(frozen=True)
class RegionLoadedSample:
    row: Mapping[str, Any]
    sample: Mapping[str, Any]
    mask: np.ndarray
    mask_facts: Mapping[str, Any] | None


@dataclass(frozen=True)
class RegionBuildResult:
    root: Path
    manifest_sha256: str
    records_sha256: str
    ledger_sha256: str
    record_count: int
    asset_count: int
    asset_bytes: int


def _ordinary_file(path: Path, *, code: str = "ASSET_INVALID", hardlink: bool = False) -> None:
    if not path.is_file() or path.is_symlink():
        fail(code, f"必须是普通文件：{path}")
    if hardlink and path.stat().st_nlink != 1:
        fail("ASSET_HARDLINK", f"不得使用 hardlink：{path}")


class RegionBenchmarkAccess:
    """验证 OA Benchmark 身份后，只打开调用方声明的 train 或 val shard。

    index 中的 test 元数据仅用于 split 泄漏检查；该类永远不会创建 test Dataset，
    也不会打开任何 test HDF5 payload。
    """

    def __init__(
        self,
        root: Path,
        expected: Mapping[str, str],
        *,
        allowed_split: str,
        cache_size: int = 8,
    ) -> None:
        if allowed_split not in {"train", "val"}:
            fail("SPLIT_FORBIDDEN", "RegionBenchmarkAccess 只允许 train 或 val，明确拒绝 test")
        if isinstance(cache_size, bool) or not isinstance(cache_size, int) or cache_size < 1:
            fail("CONFIG_INVALID", "RegionBenchmarkAccess cache_size 必须是正整数")
        self.root = Path(os.path.abspath(root))
        linked = first_symlink_component(self.root)
        if linked is not None or not self.root.is_dir():
            fail("BENCHMARK_INVALID", f"Benchmark root 非普通目录：{self.root}")
        self.allowed_split = allowed_split
        names = (
            "manifest.json",
            "index.jsonl",
            "build_config.json",
            "source_statistics.json",
            "SHA256SUMS.jsonl",
        )
        paths = {name: self.root / name for name in names}
        for path in paths.values():
            _ordinary_file(path, code="BENCHMARK_INVALID")
        ledger = read_jsonl(paths["SHA256SUMS.jsonl"])
        manifest = read_json(paths["manifest.json"])
        if not isinstance(manifest, dict):
            fail("BENCHMARK_INVALID", "OA manifest 必须是对象")
        actual = {
            "schema_version": str(manifest.get("schema_version", "")),
            "manifest_sha256": sha256_file(paths["manifest.json"]),
            "index_sha256": sha256_file(paths["index.jsonl"]),
            "payload_sha256": sha256_text(canonical_json(ledger)),
            "build_config_sha256": sha256_file(paths["build_config.json"]),
            "ledger_file_sha256": sha256_file(paths["SHA256SUMS.jsonl"]),
            "registry_sha256": sha256_text(canonical_json(registry_from_benchmark(self.root).to_dict())),
        }
        rows = read_jsonl(paths["index.jsonl"])
        split_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
        split_ids: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            split, sample_id = str(row.get("split", "")), str(row.get("sample_id", ""))
            if split not in {"train", "val", "test"} or not sample_id:
                fail("BENCHMARK_INVALID", "OA index split/sample_id 非法")
            if sample_id in split_ids[split]:
                fail("BENCHMARK_INVALID", f"重复 sample_id：{sample_id}")
            split_ids[split].add(sample_id)
            split_rows[split].append(row)
        if any(
            split_ids[left] & split_ids[right]
            for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
        ):
            fail("SPLIT_LEAKAGE", "OA index sample_id 跨 split 重复")
        actual["train_split_sha256"] = sha256_text(
            canonical_json([row["sample_id"] for row in split_rows["train"]])
        )
        for field in EXPECTED_IDENTITY_FIELDS:
            if str(expected[field]) != actual[field]:
                fail(
                    "BENCHMARK_IDENTITY_MISMATCH",
                    f"OA {field} 不匹配",
                    details={"expected": expected[field], "actual": actual[field]},
                )
        if manifest.get("files") != ledger or manifest.get("content_sha256") != actual["payload_sha256"]:
            fail("BENCHMARK_IDENTITY_MISMATCH", "OA manifest 与 ledger payload 不一致")
        self.rows = split_rows[allowed_split]
        self.rows_by_id = {str(row["sample_id"]): row for row in self.rows}
        self.split_ids = split_ids
        self.ledger = {str(item["path"]): item for item in ledger}
        self.dataset = BenchmarkDataset(
            self.root,
            split=allowed_split,
            auxiliary_policy="none",
            normalization="none",
        )
        self.dataset_indices = {
            str(row["sample_id"]): index for index, row in enumerate(self.dataset.rows)
        }
        if set(self.dataset_indices) != set(self.rows_by_id):
            fail("BENCHMARK_INVALID", f"BenchmarkDataset {allowed_split} membership 不一致")
        self._verified_shards: set[str] = set()
        # 大规模 Corpus 构建只需保留最近样本；无界缓存会让数千条 image/mask tensor 常驻内存。
        self._cache_size = cache_size
        self._cache: OrderedDict[str, RegionLoadedSample] = OrderedDict()
        self.manifest = manifest
        self.source_order = tuple(str(item) for item in manifest["included_sources"])
        self.expected = dict(expected)

    def reject_wrong_split(self, sample_id: str) -> None:
        if sample_id not in self.rows_by_id:
            for split, values in self.split_ids.items():
                if sample_id in values:
                    fail("SPLIT_FORBIDDEN", f"拒绝 {split} sample；当前只允许 {self.allowed_split}：{sample_id}")
            fail("SAMPLE_NOT_FOUND", f"sample 不存在：{sample_id}")

    def _verify_shard(self, relative: str) -> str:
        pure = portable_relative_path(relative, location="storage.shard")
        if len(pure.parts) < 4 or pure.parts[0] != "data" or pure.parts[2] != self.allowed_split:
            fail("SPLIT_FORBIDDEN", f"拒绝非 {self.allowed_split} shard：{relative}")
        item = self.ledger.get(relative)
        if item is None:
            fail("BENCHMARK_INVALID", f"shard 不在 OA ledger：{relative}")
        path = safe_join(self.root, relative, location="storage.shard")
        _ordinary_file(path, code="BENCHMARK_INVALID")
        if relative not in self._verified_shards:
            if path.stat().st_size != item.get("size_bytes") or sha256_file(path) != item.get("sha256"):
                fail("BENCHMARK_IDENTITY_MISMATCH", f"消费 shard 身份不符：{relative}")
            self._verified_shards.add(relative)
        return str(item["sha256"])

    def load(self, sample_id: str) -> RegionLoadedSample:
        self.reject_wrong_split(sample_id)
        if sample_id in self._cache:
            loaded = self._cache.pop(sample_id)
            self._cache[sample_id] = loaded
            return loaded
        row = self.rows_by_id[sample_id]
        self._verify_shard(str(row["storage"]["shard"]))
        sample = self.dataset[self.dataset_indices[sample_id]]
        if (
            sample["sample_id"] != sample_id
            or sample["metadata"]["split"] != self.allowed_split
            or sample["metadata"]["source"] != row["source"]
        ):
            fail("BENCHMARK_INVALID", f"Dataset sample identity 不一致：{sample_id}")
        raw = np.asarray(sample["mask"].cpu().numpy())
        if raw.shape[0] != 1 or not set(np.unique(raw).tolist()).issubset({0, 1}):
            fail("MASK_INVALID", f"GT mask 合同非法：{sample_id}")
        mask = raw[0].astype(bool)
        if not math.isclose(float(mask.mean()), float(row["foreground_ratio"]), rel_tol=0.0, abs_tol=1e-12):
            fail("BENCHMARK_INVALID", f"foreground_ratio 与 GT mask 不一致：{sample_id}")
        loaded = RegionLoadedSample(
            row=row,
            sample=sample,
            mask=mask,
            mask_facts=deterministic_mask_facts(mask) if bool(mask.any()) else None,
        )
        self._cache[sample_id] = loaded
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return loaded

    def shard_sha256(self, row: Mapping[str, Any]) -> str:
        return self._verify_shard(str(row["storage"]["shard"]))


def _save_png(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        fail("OUTPUT_EXISTS", f"资产已存在：{path}")
    image.save(path, format="PNG", optimize=False, compress_level=9)


def region_asset_identity(root: Path, assets: Mapping[str, Any]) -> str:
    """重算一条 Region record 的发布资产身份。

    单专家标注与训练导出必须重新绑定同一批 full/mask/crop，而不能只相信工作目录中
    保存的 record_id。这里保留 audit overlay 的身份，但它仍不会进入正式模型输入。
    """

    identities = {}
    for role, relative in sorted(assets.items()):
        if relative is None:
            identities[role] = None
            continue
        path = safe_join(root, str(relative), location=f"assets.{role}")
        _ordinary_file(path, hardlink=True)
        identities[role] = {"size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
    return sha256_text(canonical_json(identities))


def materialize_region_record(
    *,
    record_key: str,
    loaded: RegionLoadedSample,
    access: RegionBenchmarkAccess,
    writer: AtomicArtifactDirectory,
    effective_mask: np.ndarray | None = None,
    mask_source: str = "oa_auxseg_benchmark_gt",
    representation_mode: RepresentationMode | None = None,
    context_margin_ratio: float = 0.15,
    overlay_alpha: float = 0.45,
    include_audit_overlay: bool = True,
    shared_optical_path: str | None = None,
    crop_margin_ratio: float | None = None,
    counterfactual: Mapping[str, Any] | None = None,
    difficulty_proxy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """发布一条记录；程序事实来自 effective mask，绝不由 VLM 生成。"""

    row, sample = loaded.row, loaded.sample
    gt_mask = loaded.mask
    mask = gt_mask if effective_mask is None else np.asarray(effective_mask, dtype=bool)
    if mask.shape != gt_mask.shape:
        fail("MASK_INVALID", f"effective mask shape 不一致：{row['sample_id']}")
    target = bool(mask.any())
    if effective_mask is None:
        inventory = RegionInventory(
            str(row["sample_id"]), MaskMode.GT_MASK, gt_mask, (), str(row["record_sha256"])
        )
        selected = RegionSelector().select(inventory, SelectionRequest(mode=SelectionMode.GLOBAL))
        if target != (selected.mask is not None):
            fail("MASK_INVALID", f"RegionSelector target state 不一致：{row['sample_id']}")
    optical = render_optical(sample)
    stem = f"{record_key}_{sha256_text(str(row['sample_id']))[:16]}"
    optical_path = shared_optical_path or f"assets/optical_full/{stem}.png"
    if shared_optical_path is None:
        _save_png(optical, writer.path(optical_path))
    mask_path = f"assets/binary_mask/{stem}.png"
    _save_png(render_binary_mask(mask), writer.path(mask_path))
    target_status = RegionTargetStatus.TARGET_PRESENT if target else RegionTargetStatus.NO_TARGET
    if representation_mode is None:
        representation_mode = (
            RepresentationMode.FULL_PLUS_MASK_PLUS_CROP
            if target
            else RepresentationMode.FULL_PLUS_MASK
        )
    crop_path: str | None = None
    crop_window: list[int] | None = None
    if "context_crop" in FORMAL_ROLES[representation_mode]:
        if not target:
            fail("MASK_INVALID", "no-target 不能生成 context crop")
        margin = context_margin_ratio if crop_margin_ratio is None else crop_margin_ratio
        crop, window = render_context_crop(optical, mask, margin_ratio=margin)
        crop_path = f"assets/context_crop/{stem}.png"
        _save_png(crop, writer.path(crop_path))
        crop_window = list(window)
    overlay_path: str | None = None
    if target and include_audit_overlay:
        overlay_path = f"assets/audit_overlay/{stem}.png"
        _save_png(render_mask_overlay(optical, mask, alpha=overlay_alpha), writer.path(overlay_path))
    if target:
        mask_facts = {
            "status": "available",
            **deterministic_mask_facts(mask),
            "crop_window_xyxy_pixel_half_open": crop_window,
        }
    else:
        mask_facts = no_target_mask_facts()
    parent_id, parent_status = parent_identity(row)
    source_identity = {
        "benchmark_record_sha256": str(row["record_sha256"]),
        "source_record_sha256": str(row["source_record_sha256"]),
        "storage_shard": str(row["storage"]["shard"]),
        "storage_row": int(row["storage"]["row"]),
        "storage_shard_sha256": access.shard_sha256(row),
        "source_group_id": row.get("source_group_id"),
        "group_status": str(row.get("group_status", "unknown")),
    }
    assets = {
        "optical_full": optical_path,
        "binary_mask": mask_path,
        "context_crop": crop_path,
        "audit_overlay": overlay_path,
    }
    record = {
        "schema_version": REGION_RECORD_SCHEMA,
        "record_id": f"mgr_{sha256_text(canonical_json([record_key, row['sample_id'], mask_source]))[:24]}",
        "parent_id": parent_id,
        "parent_identity_status": parent_status.value,
        "sample_id": str(row["sample_id"]),
        "source": str(row["source"]),
        "split": access.allowed_split,
        "source_identity": source_identity,
        "mask_source": mask_source,
        "target_status": target_status.value,
        "representation_mode": representation_mode.value,
        "assets": assets,
        "formal_model_input_roles": list(FORMAL_ROLES[representation_mode]),
        "audit_only_roles": ["audit_overlay"] if overlay_path is not None else [],
        "program_facts": {
            "image": {
                "width_pixels": optical.width,
                "height_pixels": optical.height,
                "rgb_renderer": RGB_RENDERER_VERSION,
            },
            "mask": mask_facts,
            "coordinate_basis": COORDINATE_BASIS,
            "mask_renderer": MASK_RENDERER_VERSION,
            "difficulty_proxy": None if difficulty_proxy is None else dict(difficulty_proxy),
            "counterfactual": None if counterfactual is None else dict(counterfactual),
        },
        "description": empty_description_template(),
        "annotation": annotation_state_template(),
    }
    validate_region_record(record, expected_split=access.allowed_split)
    return record


def annotation_queue_rows(records: Iterable[Mapping[str, Any]], root: Path) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        rows.append({
            "schema_version": ANNOTATION_QUEUE_SCHEMA,
            "record_id": record["record_id"],
            "parent_id": record["parent_id"],
            "sample_id": record["sample_id"],
            "source": record["source"],
            "split": record["split"],
            "target_status": record["target_status"],
            "representation_mode": record["representation_mode"],
            "assets": dict(record["assets"]),
            "asset_identity_sha256": region_asset_identity(root, record["assets"]),
            "program_facts": dict(record["program_facts"]),
            "description_template": empty_description_template(),
            "annotation": annotation_state_template(),
        })
    return rows


def annotation_guideline() -> dict[str, Any]:
    return {
        "schema_version": "oa_groundrag.mask_grounded_region.annotation_guideline.v1",
        "language": "zh-CN",
        "scope": "仅描述人工 GT mask 指定区域、周围可见环境及二者视觉关系",
        "field_groups": [
            "target_appearance", "target_morphology", "surrounding_environment",
            "region_context_contrast", "possible_confusers", "evidence_sufficiency",
            "short_summary", "limitations",
        ],
        "forbidden_claims": [
            "发生时间", "确定触发原因", "精确位移或运动方向", "稳定性", "风险等级",
            "灾害规模等级", "实际威胁", "现场调查才能确认的地层岩性和内部结构",
        ],
        "rules": [
            "白色 binary mask 只是空间提示，不是真实地物颜色",
            "audit overlay 不得作为颜色、纹理或植被依据",
            "bbox、面积、质心、组件数和 crop window 是程序事实，不得改写",
            "证据不足时必须填写 limitations",
            "proxy 难度不是专家标签",
        ],
    }


def ledger_rows(root: Path, relatives: Iterable[str]) -> list[dict[str, Any]]:
    rows = []
    for relative in sorted(set(relatives)):
        path = safe_join(root, relative, location="ledger.path")
        _ordinary_file(path, hardlink=True)
        rows.append({"path": relative, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return rows


def frozen_stage4a_ids(config: RegionCorpusConfig) -> tuple[list[str], dict[str, Any]]:
    identity = config.stage4a_selection
    root = identity.root
    if first_symlink_component(root) is not None or not root.is_dir():
        fail("STAGE4A_IDENTITY_MISMATCH", f"Stage 4A root 非普通目录：{root}")
    paths = {
        "manifest": root / "manifest.json",
        "records": root / "records.jsonl",
        "ledger": root / "SHA256SUMS.jsonl",
    }
    for path in paths.values():
        _ordinary_file(path)
    actual = {
        "manifest_sha256": sha256_file(paths["manifest"]),
        "records_sha256": sha256_file(paths["records"]),
        "ledger_sha256": sha256_file(paths["ledger"]),
    }
    expected = {
        "manifest_sha256": identity.manifest_sha256,
        "records_sha256": identity.records_sha256,
        "ledger_sha256": identity.ledger_sha256,
    }
    if actual != expected:
        fail("STAGE4A_IDENTITY_MISMATCH", "Stage 4A manifest/records/ledger 身份漂移", details={
            "expected": expected, "actual": actual,
        })
    manifest = read_json(paths["manifest"])
    records = read_jsonl(paths["records"])
    ids = list(manifest["selection"]["ordered_sample_ids"])
    if (
        len(ids) != identity.sample_count
        or [record["sample_id"] for record in records] != ids
        or sha256_text(canonical_json(ids)) != identity.ordered_sample_ids_sha256
    ):
        fail("STAGE4A_IDENTITY_MISMATCH", "Stage 4A ordered sample IDs 不匹配")
    return ids, actual


def build_region_corpus(config_path: Path | str) -> RegionBuildResult:
    config = load_region_corpus_config(config_path)
    if config.output_root.exists() or config.output_root.is_symlink():
        fail("OUTPUT_EXISTS", f"Region Corpus output 已存在，拒绝覆盖：{config.output_root}")
    selected_ids, stage4a_actual = frozen_stage4a_ids(config)
    access = RegionBenchmarkAccess(config.benchmark_root, config.benchmark_expected, allowed_split="train")
    for sample_id in selected_ids:
        access.reject_wrong_split(sample_id)
    with AtomicArtifactDirectory(config.output_root) as writer:
        assert writer.staging is not None
        records = [
            materialize_region_record(
                record_key=f"train_{index + 1:06d}",
                loaded=access.load(sample_id),
                access=access,
                writer=writer,
                context_margin_ratio=config.context_margin_ratio,
                overlay_alpha=config.overlay_alpha,
            )
            for index, sample_id in enumerate(selected_ids)
        ]
        writer.write_jsonl("records.jsonl", records)
        queue = annotation_queue_rows(records, writer.staging)
        writer.write_jsonl("annotation_queue.jsonl", queue)
        writer.write_json("annotation_guideline.json", annotation_guideline())
        assets = [
            path
            for record in records
            for path in record["assets"].values()
            if isinstance(path, str)
        ]
        ledger = ledger_rows(
            writer.staging,
            ["records.jsonl", "annotation_queue.jsonl", "annotation_guideline.json", *assets],
        )
        writer.write_jsonl("SHA256SUMS.jsonl", ledger)
        source_counts = Counter(record["source"] for record in records)
        target_counts = Counter(record["target_status"] for record in records)
        component_counts = Counter(
            "no_target"
            if record["target_status"] == RegionTargetStatus.NO_TARGET.value
            else "single"
            if record["program_facts"]["mask"]["fragment_count"] == 1
            else "multiple"
            for record in records
        )
        location_counts = Counter(
            str(record["program_facts"]["mask"]["location_3x3"])
            for record in records
            if record["target_status"] == RegionTargetStatus.TARGET_PRESENT.value
        )
        asset_bytes = sum(item["size_bytes"] for item in ledger if item["path"].startswith("assets/"))
        manifest = {
            "schema_version": REGION_MANIFEST_SCHEMA,
            "corpus_name": REGION_CORPUS_NAME,
            "corpus_id": sha256_text(canonical_json({
                "benchmark": config.benchmark_expected,
                "config": config.semantic_sha256,
                "sample_ids": selected_ids,
            })),
            "benchmark": {"root": str(config.benchmark_root), **dict(config.benchmark_expected)},
            "selection": {
                "algorithm_version": REGION_SELECTION_ALGORITHM,
                "config_file_sha256": config.config_file_sha256,
                "config_semantic_sha256": config.semantic_sha256,
                "stage4a": {**config.stage4a_selection.to_dict(), **stage4a_actual},
                "ordered_sample_ids": selected_ids,
                "ordered_sample_ids_sha256": sha256_text(canonical_json(selected_ids)),
                "sample_count": len(selected_ids),
                "source_counts": dict(sorted(source_counts.items())),
                "target_counts": dict(sorted(target_counts.items())),
                "component_counts": dict(sorted(component_counts.items())),
                "location_3x3_counts": dict(sorted(location_counts.items())),
            },
            "rendering": {
                "context_margin_ratio": config.context_margin_ratio,
                "overlay_alpha": config.overlay_alpha,
                "rgb_renderer": RGB_RENDERER_VERSION,
                "mask_renderer": MASK_RENDERER_VERSION,
            },
            "records": {
                **next(item for item in ledger if item["path"] == "records.jsonl"),
                "count": len(records),
            },
            "annotation_queue": {
                **next(item for item in ledger if item["path"] == "annotation_queue.jsonl"),
                "count": len(queue),
            },
            "assets": {
                "count": len(set(assets)),
                "total_bytes": asset_bytes,
                "role_counts": dict(sorted(Counter(path.split("/")[1] for path in set(assets)).items())),
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
        from .region_validation import validate_region_corpus

        validate_region_corpus(writer.staging, verify_source=True)
        root = writer.publish()
    return RegionBuildResult(
        root=root,
        manifest_sha256=sha256_file(root / "manifest.json"),
        records_sha256=sha256_file(root / "records.jsonl"),
        ledger_sha256=sha256_file(root / "SHA256SUMS.jsonl"),
        record_count=len(records),
        asset_count=len(set(assets)),
        asset_bytes=asset_bytes,
    )
