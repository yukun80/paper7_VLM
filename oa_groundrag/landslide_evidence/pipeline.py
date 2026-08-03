"""Stage 4A train-only 选择、事实生成与原子发布。"""

from __future__ import annotations

import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image

from oa_groundrag.phase2.data import registry_from_benchmark
from oa_groundrag.phase2.model import optical_rgb_indices
from oa_groundrag.phase3.common import (
    canonical_json, first_symlink_component, portable_relative_path, read_json,
    read_jsonl, safe_join, sha256_file, sha256_text,
)
from oa_groundrag.phase4.artifacts import AtomicArtifactDirectory
from oa_groundrag.phase4.contracts import MaskMode, RegionInventory, SelectionMode, SelectionRequest
from oa_groundrag.phase4.evidence import (
    deterministic_auxiliary_facts, deterministic_mask_facts, render_context_crop,
    render_evidence_image, render_mask_overlay,
)
from oa_groundrag.phase4.regions import RegionSelector
from scripts.phase1_benchmark_build.benchmark_common import BenchmarkDataset

from .contracts import (
    FALSE_FLAGS, MANIFEST_SCHEMA, RECORD_SCHEMA, SUPPORTED_MODALITIES, ForegroundBin,
    PilotConfig, derive_claims, fail, load_config, no_target_mask_facts, unavailable_auxiliary_facts,
)


CORPUS_NAME = "landslide_evidence_corpus_v1_pilot_500"


@dataclass(frozen=True)
class LoadedSample:
    row: Mapping[str, Any]
    sample: Mapping[str, Any]
    mask: np.ndarray
    mask_facts: Mapping[str, Any] | None


@dataclass(frozen=True)
class BuildResult:
    root: Path
    manifest_sha256: str
    selected_ids_sha256: str
    record_count: int
    asset_count: int
    asset_bytes: int


def _regular(path: Path, *, code: str) -> None:
    if not path.is_file() or path.is_symlink():
        fail(code, f"必须是普通文件：{path}")


class OABenchmarkAccess:
    """冻结 OA metadata，并仅允许消费 train shard。"""

    def __init__(self, config: PilotConfig) -> None:
        self.config = config
        self.root = Path(os.path.abspath(config.benchmark_root))
        linked = first_symlink_component(self.root)
        if linked is not None or not self.root.is_dir():
            fail("BENCHMARK_INVALID", f"Benchmark root 非普通目录：{self.root}")
        names = ("manifest.json", "index.jsonl", "build_config.json", "source_statistics.json", "SHA256SUMS.jsonl")
        paths = {name: self.root / name for name in names}
        for path in paths.values():
            _regular(path, code="BENCHMARK_INVALID")
        expected = config.expected
        actual = {
            "manifest_sha256": sha256_file(paths["manifest.json"]),
            "index_sha256": sha256_file(paths["index.jsonl"]),
            "build_config_sha256": sha256_file(paths["build_config.json"]),
            "ledger_file_sha256": sha256_file(paths["SHA256SUMS.jsonl"]),
        }
        manifest = read_json(paths["manifest.json"])
        if not isinstance(manifest, dict):
            fail("BENCHMARK_INVALID", "OA manifest 必须是对象")
        ledger = read_jsonl(paths["SHA256SUMS.jsonl"])
        if manifest.get("schema_version") != expected["schema_version"]:
            fail("BENCHMARK_IDENTITY_MISMATCH", "OA manifest schema 不匹配")
        if manifest.get("files") != ledger:
            fail("BENCHMARK_IDENTITY_MISMATCH", "manifest.files 与 ledger 不一致")
        payload = sha256_text(canonical_json(ledger))
        actual["payload_sha256"] = payload
        if manifest.get("content_sha256") != payload:
            fail("BENCHMARK_IDENTITY_MISMATCH", "OA payload identity 不一致")
        if manifest.get("index_sha256") != actual["index_sha256"]:
            fail("BENCHMARK_IDENTITY_MISMATCH", "OA index identity 不一致")
        for key, value in actual.items():
            if expected[key] != value:
                fail("BENCHMARK_IDENTITY_MISMATCH", f"OA {key} 不匹配", details={
                    "expected": expected[key], "actual": value,
                })
        registry = registry_from_benchmark(self.root)
        registry_sha = sha256_text(canonical_json(registry.to_dict()))
        if registry_sha != expected["registry_sha256"]:
            fail("BENCHMARK_IDENTITY_MISMATCH", "OA registry identity 不一致")
        rows = read_jsonl(paths["index.jsonl"])
        self.split_ids: dict[str, set[str]] = defaultdict(set)
        train_rows: list[dict[str, Any]] = []
        for row in rows:
            split = str(row.get("split", ""))
            sample_id = str(row.get("sample_id", ""))
            if split not in {"train", "val", "test"} or not sample_id:
                fail("BENCHMARK_INVALID", "OA index split/sample_id 非法")
            if sample_id in self.split_ids[split]:
                fail("BENCHMARK_INVALID", f"重复 sample_id：{sample_id}")
            self.split_ids[split].add(sample_id)
            if split == "train":
                train_rows.append(row)
        split_pairs = (("train", "val"), ("train", "test"), ("val", "test"))
        if any(self.split_ids[left] & self.split_ids[right] for left, right in split_pairs):
            fail("SPLIT_LEAKAGE", "OA index sample_id 跨 split 重复")
        train_sha = sha256_text(canonical_json([row["sample_id"] for row in train_rows]))
        if train_sha != expected["train_split_sha256"]:
            fail("BENCHMARK_IDENTITY_MISMATCH", "ordered train split identity 不一致")
        self.train_split_sha256 = train_sha
        self.train_rows = train_rows
        self.rows_by_id = {str(row["sample_id"]): row for row in train_rows}
        if len(self.rows_by_id) != len(train_rows):
            fail("BENCHMARK_INVALID", "train sample_id 重复")
        self.ledger = {str(item["path"]): item for item in ledger}
        self.dataset = BenchmarkDataset(self.root, split="train", auxiliary_policy="all", normalization="none")
        self.dataset_indices = {str(row["sample_id"]): index for index, row in enumerate(self.dataset.rows)}
        if set(self.dataset_indices) != set(self.rows_by_id):
            fail("BENCHMARK_INVALID", "BenchmarkDataset train membership 不一致")
        self._verified_shards: set[str] = set()
        self._cache: dict[str, LoadedSample] = {}
        self.manifest = manifest
        self.registry_sha256 = registry_sha
        self.source_order = tuple(str(item) for item in manifest["included_sources"])

    def reject_non_train(self, sample_id: str) -> None:
        if sample_id in self.split_ids["val"] or sample_id in self.split_ids["test"]:
            fail("SPLIT_LEAKAGE", f"拒绝非 train sample：{sample_id}")
        if sample_id not in self.rows_by_id:
            fail("SAMPLE_NOT_FOUND", f"sample 不在 train：{sample_id}")

    def _verify_train_shard(self, relative: str) -> str:
        pure = portable_relative_path(relative, location="storage.shard")
        if len(pure.parts) < 4 or pure.parts[0] != "data" or pure.parts[2] != "train":
            fail("SPLIT_LEAKAGE", f"拒绝非 train shard：{relative}")
        item = self.ledger.get(relative)
        if item is None:
            fail("BENCHMARK_INVALID", f"shard 不在 OA ledger：{relative}")
        path = safe_join(self.root, relative, location="storage.shard")
        _regular(path, code="BENCHMARK_INVALID")
        if relative not in self._verified_shards:
            if path.stat().st_size != item.get("size_bytes") or sha256_file(path) != item.get("sha256"):
                fail("BENCHMARK_IDENTITY_MISMATCH", f"消费 shard 身份不符：{relative}")
            self._verified_shards.add(relative)
        return str(item["sha256"])

    def load(self, sample_id: str) -> LoadedSample:
        self.reject_non_train(sample_id)
        if sample_id in self._cache:
            return self._cache[sample_id]
        row = self.rows_by_id[sample_id]
        shard = str(row["storage"]["shard"])
        self._verify_train_shard(shard)
        sample = self.dataset[self.dataset_indices[sample_id]]
        if sample["sample_id"] != sample_id or sample["metadata"]["split"] != "train" or sample["metadata"]["source"] != row["source"]:
            fail("BENCHMARK_INVALID", f"Dataset sample identity 不一致：{sample_id}")
        mask = np.asarray(sample["mask"].cpu().numpy())
        if mask.shape[0] != 1 or not set(np.unique(mask).tolist()).issubset({0, 1}):
            fail("MASK_INVALID", f"GT mask 合同非法：{sample_id}")
        binary = mask[0].astype(bool)
        ratio = float(binary.mean())
        if not math.isclose(ratio, float(row["foreground_ratio"]), rel_tol=0.0, abs_tol=1e-12):
            fail("BENCHMARK_INVALID", f"foreground_ratio 与 GT mask 不一致：{sample_id}")
        facts = deterministic_mask_facts(binary) if bool(binary.any()) else None
        loaded = LoadedSample(row=row, sample=sample, mask=binary, mask_facts=facts)
        self._cache[sample_id] = loaded
        return loaded

    def shard_sha256(self, row: Mapping[str, Any]) -> str:
        return self._verify_train_shard(str(row["storage"]["shard"]))


def _bin_for(value: float, bins: Sequence[ForegroundBin]) -> ForegroundBin:
    for item in bins:
        if item.contains(value):
            return item
    fail("SELECTION_FAILED", f"target foreground ratio 未落入任何分层：{value}")
    raise AssertionError


def _rank(config: PilotConfig, sample_id: str) -> tuple[str, str]:
    return sha256_text(f"{config.seed}{config.algorithm_version}{sample_id}"), sample_id

def _bin_requests(total: int, bins: Sequence[ForegroundBin]) -> list[int]:
    base, remainder = divmod(total, len(bins))
    return [base + (1 if index < remainder else 0) for index in range(len(bins))]


def _component_balance(chosen: list[dict[str, Any]], candidates: Sequence[dict[str, Any]], access: OABenchmarkAccess) -> list[dict[str, Any]]:
    if len(chosen) < 2:
        return chosen
    counts = Counter(
        "single" if access.load(str(row["sample_id"])).mask_facts["fragment_count"] == 1 else "multiple"
        for row in chosen
    )
    if counts["single"] and counts["multiple"]:
        return chosen
    wanted = "multiple" if counts["single"] else "single"
    selected = {str(row["sample_id"]) for row in chosen}
    for row in candidates:
        if str(row["sample_id"]) in selected:
            continue
        facts = access.load(str(row["sample_id"])).mask_facts
        kind = "single" if facts["fragment_count"] == 1 else "multiple"
        if kind == wanted:
            return [*chosen[:-1], row]
    return chosen


def select_pilot(config: PilotConfig, access: OABenchmarkAccess) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """按 source/target/ratio 固定配额选择，容量不足仅在同 source/target 回填。"""

    if tuple(item.source for item in config.source_quotas) != access.source_order:
        fail("CONFIG_INVALID", "source quota 顺序必须等于 OA manifest source 顺序")
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in access.train_rows:
        source = str(row["source"])
        ratio = float(row["foreground_ratio"])
        key = (source, _bin_for(ratio, config.foreground_bins).name) if ratio > 0 else (source, "no_target")
        grouped[key].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: _rank(config, str(row["sample_id"])))
    selected: list[dict[str, Any]] = []
    source_reports: list[dict[str, Any]] = []
    for quota in config.source_quotas:
        requests = _bin_requests(quota.target, config.foreground_bins)
        source_selected: list[dict[str, Any]] = []
        bin_reports: list[dict[str, Any]] = []
        used: set[str] = set()
        for item, requested in zip(config.foreground_bins, requests, strict=True):
            candidates = grouped[(quota.source, item.name)]
            chosen = _component_balance(candidates[:requested], candidates, access)
            used.update(str(row["sample_id"]) for row in chosen)
            source_selected.extend(chosen)
            bin_reports.append({
                "name": item.name, "requested": requested, "achieved": len(chosen),
                "deficit": max(0, requested - len(chosen)),
            })
        target_deficit = quota.target - len(source_selected)
        if target_deficit > 0:
            leftovers = sorted((
                row for item in config.foreground_bins for row in grouped[(quota.source, item.name)]
                if str(row["sample_id"]) not in used
            ), key=lambda row: _rank(config, str(row["sample_id"])))
            source_selected.extend(leftovers[:target_deficit])
        no_candidates = grouped[(quota.source, "no_target")]
        no_selected = no_candidates[: quota.no_target]
        if len(source_selected) != quota.target or len(no_selected) != quota.no_target:
            fail("SELECTION_CAPACITY_INSUFFICIENT", f"{quota.source} 无法达到真实配额", details={
                "target_requested": quota.target, "target_achieved": len(source_selected),
                "no_target_requested": quota.no_target, "no_target_achieved": len(no_selected),
            })
        selected.extend(source_selected)
        selected.extend(no_selected)
        source_reports.append({
            "source": quota.source,
            "requested": {"total": quota.total, "target": quota.target, "no_target": quota.no_target},
            "achieved": {"total": len(source_selected) + len(no_selected), "target": len(source_selected), "no_target": len(no_selected)},
            "deficit": {"total": 0, "target": 0, "no_target": 0}, "foreground_bins": bin_reports,
        })
    if len(selected) != config.sample_count:
        fail("SELECTION_FAILED", "Pilot 总数未达到配置")
    if len({str(row["sample_id"]) for row in selected}) != len(selected):
        fail("SELECTION_FAILED", "Pilot sample 重复")
    for row in selected:
        loaded = access.load(str(row["sample_id"]))
        if bool(loaded.mask.any()) != (float(row["foreground_ratio"]) > 0):
            fail("BENCHMARK_INVALID", f"target state 与 GT mask 冲突：{row['sample_id']}")
    source_rank = {source: index for index, source in enumerate(access.source_order)}
    bin_rank = {item.name: index for index, item in enumerate(config.foreground_bins)}
    def final_rank(row: Mapping[str, Any]) -> tuple[Any, ...]:
        ratio = float(row["foreground_ratio"])
        return (
            source_rank[str(row["source"])],
            0 if ratio > 0 else 1,
            bin_rank[_bin_for(ratio, config.foreground_bins).name] if ratio > 0 else len(bin_rank),
            *_rank(config, str(row["sample_id"])),
        )
    selected.sort(key=final_rank)
    for report in source_reports:
        for bin_report in report["foreground_bins"]:
            rows = [row for row in selected if row["source"] == report["source"]
                    and float(row["foreground_ratio"]) > 0
                    and _bin_for(float(row["foreground_ratio"]), config.foreground_bins).name == bin_report["name"]]
            kinds = Counter(
                "single" if access.load(str(row["sample_id"])).mask_facts["fragment_count"] == 1 else "multiple"
                for row in rows
            )
            bin_report["achieved"] = len(rows)
            bin_report["deficit"] = max(0, bin_report["requested"] - len(rows))
            bin_report["components"] = {"single": kinds["single"], "multiple": kinds["multiple"],
                                        "both_covered": bool(kinds["single"] and kinds["multiple"])}
    signatures: Counter[str] = Counter()
    for row in selected:
        signature = canonical_json({
            "optical_channel_names": row["optical"]["channel_names"],
            "available_auxiliaries": sorted(row["auxiliaries"]),
        })
        signatures[signature] += 1
    return selected, {"sources": source_reports, "modality_signatures": [
        {"signature": signature, "count": count} for signature, count in sorted(signatures.items())
    ]}


def render_optical(sample: Mapping[str, Any]) -> Image.Image:
    values = np.asarray(sample["optical"].cpu().numpy(), dtype=np.float64)
    validity = np.asarray(sample["optical_pixel_valid"].cpu().numpy(), dtype=bool)
    channel_valid = np.asarray(sample["optical_channel_valid"].cpu().numpy(), dtype=bool)
    indices = optical_rgb_indices(sample["optical_channel_names"])
    rgb = np.zeros((*values.shape[-2:], 3), dtype=np.uint8)
    for output, index in enumerate(indices):
        valid = validity[index] & channel_valid[index] & np.isfinite(values[index])
        if not bool(valid.any()):
            fail("OPTICAL_UNAVAILABLE", "审计 RGB 通道没有有效像素")
        data = values[index][valid]
        low, high = (float(item) for item in np.percentile(data, (2, 98), method="linear"))
        if high <= low:
            rgb[..., output][valid] = 127
        else:
            rgb[..., output][valid] = np.clip((values[index][valid] - low) / (high - low) * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(rgb)


def _save_png(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        fail("OUTPUT_EXISTS", f"资产已存在：{path}")
    image.save(path, format="PNG", optimize=False, compress_level=9)


def _record_for(index: int, loaded: LoadedSample, config: PilotConfig, access: OABenchmarkAccess,
                writer: AtomicArtifactDirectory) -> dict[str, Any]:
    row, sample, mask = loaded.row, loaded.sample, loaded.mask
    sample_id = str(row["sample_id"])
    target = bool(mask.any())
    inventory = RegionInventory(sample_id, MaskMode.GT_MASK, mask, (), str(row["record_sha256"]))
    selected = RegionSelector().select(inventory, SelectionRequest(mode=SelectionMode.GLOBAL))
    if target != (selected.mask is not None):
        fail("MASK_INVALID", f"RegionSelector target state 不一致：{sample_id}")
    stem = f"{index + 1:06d}_{sha256_text(sample_id)[:16]}"
    optical_path = f"assets/optical/{stem}.png"
    mask_path = f"assets/mask/{stem}.png"
    optical = render_optical(sample)
    _save_png(optical, writer.path(optical_path))
    _save_png(Image.fromarray(mask.astype(np.uint8) * 255), writer.path(mask_path))
    overlay_path = crop_path = None
    mask_facts: dict[str, Any]
    if target:
        overlay_path = f"assets/overlay/{stem}.png"
        crop_path = f"assets/crop/{stem}.png"
        _save_png(render_mask_overlay(optical, mask, alpha=config.overlay_alpha), writer.path(overlay_path))
        crop, crop_window = render_context_crop(optical, mask, margin_ratio=config.context_margin_ratio)
        _save_png(crop, writer.path(crop_path))
        mask_facts = {"status": "available", **dict(loaded.mask_facts), "crop_window_xyxy_pixel_half_open": list(crop_window)}
    else:
        mask_facts = no_target_mask_facts()
    modality_facts: dict[str, dict[str, Any]] = {}
    auxiliary_assets: dict[str, str] = {}
    for modality in SUPPORTED_MODALITIES:
        if modality not in sample["auxiliaries"]:
            modality_facts[modality] = unavailable_auxiliary_facts()
            continue
        auxiliary = sample["auxiliaries"][modality]
        values = np.asarray(auxiliary["values"].cpu().numpy())
        valid = np.asarray(auxiliary["pixel_valid"].cpu().numpy())
        channels = np.asarray(auxiliary["channel_valid"].cpu().numpy())
        facts = deterministic_auxiliary_facts(
            values, valid, channels, channel_names=auxiliary["channel_names"],
            mask=mask if target else None, min_coverage=config.min_auxiliary_coverage,
        )
        modality_facts[modality] = {**facts, "unit": None, "sign_convention": None}
        if facts["status"] == "available":
            if values.shape[0] != 1:
                fail("AUXILIARY_INVALID", f"{modality}: Pilot 仅接受 registry 中的单通道视图")
            asset_path = f"assets/auxiliaries/{modality}/{stem}.png"
            effective = valid[0].astype(bool) & bool(channels[0]) & np.isfinite(values[0])
            _save_png(render_evidence_image(values[0], validity=effective, location=modality), writer.path(asset_path))
            auxiliary_assets[modality] = asset_path
    available = ["optical", *sorted(auxiliary_assets)]
    unavailable = sorted(set(SUPPORTED_MODALITIES) - set(auxiliary_assets))
    target_state = "target_present" if target else "no_target"
    allowed, forbidden, unavailable_claims = derive_claims(target_state=target_state, modality_facts=modality_facts)
    facts = {
        "image": {
            "width_pixels": optical.width, "height_pixels": optical.height,
            "coordinate_basis": "top_left_pixel_raster", "geographic_orientation": "unknown",
        },
        "mask": mask_facts, "modalities": modality_facts,
    }
    return {
        "schema_version": RECORD_SCHEMA, "record_id": f"le_{sha256_text(sample_id)[:24]}",
        "sample_id": sample_id, "source": str(row["source"]), "split": "train",
        "source_identity": {
            "benchmark_record_sha256": str(row["record_sha256"]), "source_record_sha256": str(row["source_record_sha256"]),
            "storage_shard": str(row["storage"]["shard"]), "storage_row": int(row["storage"]["row"]),
            "storage_shard_sha256": access.shard_sha256(row),
        },
        "mask_source": "oa_auxseg_benchmark_gt", "target_state": target_state,
        "declared_modality_signature": {
            "optical_channel_names": list(sample["optical_channel_names"]),
            "auxiliaries": {name: list(value["channel_names"]) for name, value in sorted(sample["auxiliaries"].items())},
        },
        "available_modalities": available, "unavailable_modalities": unavailable,
        "assets": {
            "optical": optical_path, "mask": mask_path, "overlay": overlay_path,
            "crop": crop_path, "auxiliaries": auxiliary_assets,
        },
        "program_facts": facts, "allowed_claims": allowed, "forbidden_claims": forbidden,
        "unavailable_claims": unavailable_claims, "silver_observation": None,
        "review_status": "not_reviewed", "case_kb_eligible": False, "adapter_training_eligible": False,
    }


def _ledger_rows(root: Path, relatives: Iterable[str]) -> list[dict[str, Any]]:
    rows = []
    for relative in sorted(set(relatives)):
        path = safe_join(root, relative, location="ledger.path")
        _regular(path, code="ARTIFACT_INVALID")
        rows.append({"path": relative, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return rows


def build_auto(config_path: Path | str) -> BuildResult:
    config = load_config(config_path)
    if config.output_root.exists() or config.output_root.is_symlink():
        fail("OUTPUT_EXISTS", f"Corpus output 已存在，拒绝覆盖：{config.output_root}")
    linked = first_symlink_component(config.output_root)
    if linked is not None:
        fail("OUTPUT_LINK", f"Corpus output 路径含 symlink：{linked}")
    access = OABenchmarkAccess(config)
    selected, stratification = select_pilot(config, access)
    selected_ids = [str(row["sample_id"]) for row in selected]
    selected_sha = sha256_text(canonical_json(selected_ids))
    with AtomicArtifactDirectory(config.output_root) as writer:
        assert writer.staging is not None
        records = [_record_for(index, access.load(sample_id), config, access, writer)
                   for index, sample_id in enumerate(selected_ids)]
        writer.write_jsonl("records.jsonl", records)
        asset_paths = [path for record in records for path in (
            record["assets"]["optical"], record["assets"]["mask"], record["assets"]["overlay"],
            record["assets"]["crop"], *record["assets"]["auxiliaries"].values(),
        ) if path is not None]
        ledger = _ledger_rows(writer.staging, ["records.jsonl", *asset_paths])
        writer.write_jsonl("SHA256SUMS.jsonl", ledger)
        role_counts = Counter("auxiliary" if path.startswith("assets/auxiliaries/") else path.split("/")[1]
                              for path in asset_paths)
        asset_bytes = sum(item["size_bytes"] for item in ledger if item["path"].startswith("assets/"))
        manifest = {
            "schema_version": MANIFEST_SCHEMA,
            "corpus_id": sha256_text(canonical_json({
                "benchmark": config.expected, "selection": config.semantic_sha256, "sample_ids": selected_ids,
            })),
            "corpus_name": CORPUS_NAME, "benchmark": {**dict(config.expected), "root": str(config.benchmark_root)},
            "train_split": {
                "name": "train", "source_record_count": len(access.train_rows),
                "ordered_sample_ids_sha256": access.train_split_sha256,
            },
            "selection": {
                "config_schema_version": config.semantic_dict()["schema_version"], "config_file_sha256": config.config_file_sha256,
                "config_semantic_sha256": config.semantic_sha256, "seed": config.seed,
                "algorithm_version": config.algorithm_version, "sample_count": len(selected_ids),
                "source_order": list(access.source_order),
                "foreground_bins": [item.__dict__ for item in config.foreground_bins],
                "stratification": stratification, "ordered_sample_ids": selected_ids,
                "ordered_sample_ids_sha256": selected_sha,
            },
            "rendering": {
                "context_margin_ratio": config.context_margin_ratio, "overlay_alpha": config.overlay_alpha,
                "min_auxiliary_coverage": config.min_auxiliary_coverage,
            },
            "records": next(item for item in ledger if item["path"] == "records.jsonl") | {"count": len(records)},
            "assets": {"count": len(asset_paths), "total_bytes": asset_bytes,
                       "role_counts": dict(sorted(role_counts.items()))},
            "ledger": {
                "path": "SHA256SUMS.jsonl", "entry_count": len(ledger),
                "size_bytes": writer.path("SHA256SUMS.jsonl").stat().st_size,
                "file_sha256": sha256_file(writer.path("SHA256SUMS.jsonl")),
                "root_sha256": sha256_text(canonical_json(ledger)),
            },
            **{flag: False for flag in FALSE_FLAGS},
        }
        writer.write_json("manifest.json", manifest)
        from .validation import validate_corpus

        validate_corpus(writer.staging, verify_source=True)
        root = writer.publish()
    return BuildResult(root, sha256_file(root / "manifest.json"), selected_sha, len(records), len(asset_paths), asset_bytes)
