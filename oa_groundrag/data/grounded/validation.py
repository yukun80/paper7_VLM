"""Landslide Evidence Corpus 的正式运行时验证器。"""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image

from oa_groundrag.artifacts.identity import (
    canonical_json,
    sha256_file,
    sha256_text,
)
from oa_groundrag.artifacts.io import first_symlink_component
from oa_groundrag.data.rs_general.io import (
    portable_relative_path,
    read_json,
    read_jsonl,
    safe_join,
)
from oa_groundrag.grounding.evidence import (
    context_crop_window, deterministic_auxiliary_facts, deterministic_mask_facts,
    render_context_crop, render_evidence_image, render_mask_overlay,
)

from .contracts import (
    CONFIG_SCHEMA, EXPECTED_IDENTITY_FIELDS, FALSE_FLAGS, MANIFEST_SCHEMA, SELECTION_ALGORITHM,
    SUPPORTED_MODALITIES, ForegroundBin, PilotConfig, SourceQuota, derive_claims, fail,
    no_target_mask_facts, unavailable_auxiliary_facts, validate_record,
)
from .pilot import (
    CORPUS_NAME, OABenchmarkAccess, render_optical, select_pilot,
)


MANIFEST_FIELDS = (
    "schema_version", "corpus_id", "corpus_name", "benchmark", "train_split", "selection",
    "rendering", "records", "assets", "ledger", *FALSE_FLAGS,
)


def _exact(row: Mapping[str, Any], fields: tuple[str, ...], *, location: str) -> None:
    if set(row) != set(fields):
        fail("SCHEMA_MISMATCH", f"{location} 字段不匹配", details={
            "missing": sorted(set(fields) - set(row)), "unknown": sorted(set(row) - set(fields)),
        })


def _mapping(value: Any, *, location: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        fail("TYPE_MISMATCH", f"{location} 必须是对象")
    return dict(value)


def _ordinary_file(path: Path, *, output: bool = True) -> None:
    if not path.is_file() or path.is_symlink():
        fail("ASSET_INVALID", f"缺失或链接文件：{path}")
    if output and path.stat().st_nlink != 1:
        fail("ASSET_HARDLINK", f"输出文件不得是 hardlink：{path}")


def _image(path: Path, *, mode: str) -> Image.Image:
    _ordinary_file(path)
    with Image.open(path) as handle:
        handle.load()
        if handle.format != "PNG":
            fail("ASSET_INVALID", f"资产必须是 PNG：{path}")
        return handle.convert(mode)


def _same_image(actual: Image.Image, expected: Image.Image, *, location: str) -> None:
    if actual.mode != expected.mode or actual.size != expected.size or not np.array_equal(np.asarray(actual), np.asarray(expected)):
        fail("ASSET_CONTENT_MISMATCH", f"资产像素与程序事实不一致：{location}")


def _assert_equal(actual: Any, expected: Any, *, location: str) -> None:
    if canonical_json(actual) != canonical_json(expected):
        fail("FACT_MISMATCH", f"{location} 与重算事实不一致")


def _source_access(manifest: Mapping[str, Any], root: Path) -> OABenchmarkAccess:
    benchmark = _mapping(manifest["benchmark"], location="manifest.benchmark")
    expected = dict(benchmark)
    benchmark_root = Path(str(expected.pop("root")))
    selection = _mapping(manifest["selection"], location="manifest.selection")
    bins = tuple(ForegroundBin(str(item["name"]), float(item["lower_exclusive"]),
                               float(item["upper_inclusive"])) for item in selection["foreground_bins"])
    source_reports = selection["stratification"]["sources"]
    quotas = tuple(SourceQuota(
        str(item["source"]), int(item["requested"]["total"]), int(item["requested"]["target"]),
        int(item["requested"]["no_target"]),
    ) for item in source_reports)
    config = PilotConfig(
        root / "manifest.json", str(selection["config_file_sha256"]), str(selection["config_semantic_sha256"]),
        benchmark_root, expected, root, int(selection["sample_count"]), int(selection["seed"]),
        str(selection["algorithm_version"]), quotas, bins,
        float(manifest["rendering"]["context_margin_ratio"]), float(manifest["rendering"]["overlay_alpha"]),
        float(manifest["rendering"]["min_auxiliary_coverage"]),
    )
    return OABenchmarkAccess(config)


def _validate_source_record(root: Path, record: Mapping[str, Any], access: OABenchmarkAccess,
                            optical: Image.Image, mask: np.ndarray, rendering: Mapping[str, Any]) -> None:
    loaded = access.load(str(record["sample_id"]))
    row, sample = loaded.row, loaded.sample
    if (record["source"] != row["source"] or record["mask_source"] != "oa_auxseg_benchmark_gt"
            or record["record_id"] != f"le_{sha256_text(str(record['sample_id']))[:24]}"):
        fail("RECORD_IDENTITY_MISMATCH", f"record/source/mask identity 非法：{record['sample_id']}")
    if not np.array_equal(mask, loaded.mask):
        fail("MASK_SOURCE_MISMATCH", f"导出 GT mask 与 OA train 不一致：{record['sample_id']}")
    _same_image(optical, render_optical(sample), location=f"{record['sample_id']}.optical")
    expected_identity = {
        "benchmark_record_sha256": str(row["record_sha256"]),
        "source_record_sha256": str(row["source_record_sha256"]),
        "storage_shard": str(row["storage"]["shard"]),
        "storage_row": int(row["storage"]["row"]),
        "storage_shard_sha256": access.shard_sha256(row),
    }
    _assert_equal(record["source_identity"], expected_identity, location="source_identity")
    signature = {"optical_channel_names": list(sample["optical_channel_names"]), "auxiliaries": {
        name: list(value["channel_names"]) for name, value in sorted(sample["auxiliaries"].items())
    }}
    _assert_equal(record["declared_modality_signature"], signature, location="modality_signature")
    target = bool(mask.any())
    expected_modalities: dict[str, Any] = {}
    expected_aux_assets: dict[str, str] = {}
    actual_aux_assets = _mapping(record["assets"]["auxiliaries"], location="assets.auxiliaries")
    for modality in SUPPORTED_MODALITIES:
        if modality not in sample["auxiliaries"]:
            expected_modalities[modality] = unavailable_auxiliary_facts()
            continue
        auxiliary = sample["auxiliaries"][modality]
        values = np.asarray(auxiliary["values"].cpu().numpy())
        validity = np.asarray(auxiliary["pixel_valid"].cpu().numpy())
        channels = np.asarray(auxiliary["channel_valid"].cpu().numpy())
        facts = deterministic_auxiliary_facts(
            values, validity, channels, channel_names=auxiliary["channel_names"],
            mask=mask if target else None, min_coverage=float(rendering["min_auxiliary_coverage"]),
        )
        expected_modalities[modality] = {**facts, "unit": None, "sign_convention": None}
        if facts["status"] == "available":
            path = actual_aux_assets.get(modality)
            if not isinstance(path, str):
                fail("ASSET_REFERENCE_MISMATCH", f"{modality} 有效但未引用资产")
            expected_aux_assets[modality] = path
            if values.shape[0] != 1:
                fail("AUXILIARY_INVALID", f"{modality}: 非单通道 Pilot 资产")
            effective = validity[0].astype(bool) & bool(channels[0]) & np.isfinite(values[0])
            expected_view = render_evidence_image(values[0], validity=effective, location=modality)
            _same_image(_image(safe_join(root, path, location="auxiliary.path"), mode="RGB"),
                        expected_view, location=f"{record['sample_id']}.{modality}")
    _assert_equal(record["program_facts"]["modalities"], expected_modalities, location="modality facts")
    if set(actual_aux_assets) != set(expected_aux_assets):
        fail("ASSET_REFERENCE_MISMATCH", "辅助资产集合与 validity 不一致")
    expected_available = ["optical", *sorted(expected_aux_assets)]
    expected_unavailable = sorted(set(SUPPORTED_MODALITIES) - set(expected_aux_assets))
    _assert_equal(record["available_modalities"], expected_available, location="available_modalities")
    _assert_equal(record["unavailable_modalities"], expected_unavailable, location="unavailable_modalities")


def _validate_record_assets(root: Path, record: Mapping[str, Any], rendering: Mapping[str, Any],
                            access: OABenchmarkAccess | None) -> set[str]:
    assets = _mapping(record["assets"], location="record.assets")
    _exact(assets, ("optical", "mask", "overlay", "crop", "auxiliaries"), location="record.assets")
    optical_path, mask_path = assets["optical"], assets["mask"]
    if not isinstance(optical_path, str) or not isinstance(mask_path, str):
        fail("ASSET_REFERENCE_MISMATCH", "optical/mask 路径必须存在")
    optical = _image(safe_join(root, optical_path, location="assets.optical"), mode="RGB")
    mask_image = _image(safe_join(root, mask_path, location="assets.mask"), mode="L")
    raw_mask = np.asarray(mask_image)
    if not set(np.unique(raw_mask).tolist()).issubset({0, 255}):
        fail("MASK_INVALID", "导出 mask 必须严格为 0/255")
    mask = raw_mask == 255
    if mask.shape != (optical.height, optical.width):
        fail("MASK_INVALID", "mask 与 optical shape 不一致")
    target = bool(mask.any())
    expected_state = "target_present" if target else "no_target"
    if record["target_state"] != expected_state:
        fail("MASK_INVALID", "target_state 与导出 mask 不一致")
    expected_image = {"width_pixels": optical.width, "height_pixels": optical.height,
                      "coordinate_basis": "top_left_pixel_raster", "geographic_orientation": "unknown"}
    _assert_equal(record["program_facts"]["image"], expected_image, location="image facts")
    referenced = {optical_path, mask_path}
    if target:
        if not isinstance(assets["overlay"], str) or not isinstance(assets["crop"], str):
            fail("ASSET_REFERENCE_MISMATCH", "target 必须引用 overlay/crop")
        expected_facts = {"status": "available", **deterministic_mask_facts(mask),
                          "crop_window_xyxy_pixel_half_open": list(context_crop_window(
                              mask, image_size=optical.size,
                              margin_ratio=float(rendering["context_margin_ratio"]))) }
        expected_overlay = render_mask_overlay(optical, mask, alpha=float(rendering["overlay_alpha"]))
        expected_crop, _ = render_context_crop(optical, mask, margin_ratio=float(rendering["context_margin_ratio"]))
        _same_image(_image(safe_join(root, assets["overlay"], location="assets.overlay"), mode="RGB"), expected_overlay, location="overlay")
        _same_image(_image(safe_join(root, assets["crop"], location="assets.crop"), mode="RGB"), expected_crop, location="crop")
        referenced.update((assets["overlay"], assets["crop"]))
    else:
        if assets["overlay"] is not None or assets["crop"] is not None:
            fail("ASSET_REFERENCE_MISMATCH", "no-target 的 overlay/crop 必须为 null")
        expected_facts = no_target_mask_facts()
    _assert_equal(record["program_facts"]["mask"], expected_facts, location="mask facts")
    auxiliary_assets = _mapping(assets["auxiliaries"], location="assets.auxiliaries")
    for modality, path in auxiliary_assets.items():
        if modality not in SUPPORTED_MODALITIES or not isinstance(path, str):
            fail("ASSET_REFERENCE_MISMATCH", "辅助资产 key/path 非法")
        referenced.add(path)
    modality_facts = _mapping(record["program_facts"]["modalities"], location="program_facts.modalities")
    if set(modality_facts) != set(SUPPORTED_MODALITIES):
        fail("SCHEMA_MISMATCH", "program_facts.modalities 集合非法")
    allowed, forbidden, unavailable = derive_claims(
        target_state=expected_state, modality_facts=modality_facts)
    _assert_equal(record["allowed_claims"], allowed, location="allowed_claims")
    _assert_equal(record["forbidden_claims"], forbidden, location="forbidden_claims")
    _assert_equal(record["unavailable_claims"], unavailable, location="unavailable_claims")
    if access is not None:
        _validate_source_record(root, record, access, optical, mask, rendering)
    return referenced


def validate_corpus(root: Path | str, *, verify_source: bool = True) -> dict[str, Any]:
    root = Path(os.path.abspath(Path(root)))
    if first_symlink_component(root) is not None or not root.is_dir():
        fail("CORPUS_INVALID", f"Corpus root 非普通目录：{root}")
    for child in root.rglob("*"):
        if child.is_symlink():
            fail("ASSET_LINK", f"Corpus 内不得包含 symlink：{child}")
    manifest_path = root / "manifest.json"
    ledger_path = root / "SHA256SUMS.jsonl"
    records_path = root / "records.jsonl"
    for path in (manifest_path, ledger_path, records_path): _ordinary_file(path)
    manifest = _mapping(read_json(manifest_path), location="manifest")
    _exact(manifest, MANIFEST_FIELDS, location="manifest")
    if manifest["schema_version"] != MANIFEST_SCHEMA:
        fail("SCHEMA_MISMATCH", "manifest schema_version 非法")
    if manifest["corpus_name"] != CORPUS_NAME:
        fail("SCHEMA_MISMATCH", "manifest corpus_name 非法")
    for flag in FALSE_FLAGS:
        if manifest[flag] is not False: fail("SCHEMA_MISMATCH", f"manifest.{flag} 必须为 false")
    ledger = read_jsonl(ledger_path)
    paths = [str(item.get("path", "")) for item in ledger]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        fail("LEDGER_INVALID", "ledger path 必须严格排序且唯一")
    ledger_files: set[str] = set()
    for index, item in enumerate(ledger):
        _exact(item, ("path", "size_bytes", "sha256"), location=f"ledger[{index}]")
        relative = str(item["path"])
        portable_relative_path(relative, location=f"ledger[{index}].path")
        path = safe_join(root, relative, location=f"ledger[{index}].path")
        _ordinary_file(path)
        if path.stat().st_size != item["size_bytes"] or sha256_file(path) != item["sha256"]:
            fail("LEDGER_MISMATCH", f"ledger 文件身份不符：{relative}")
        ledger_files.add(relative)
    actual_files = {path.relative_to(root).as_posix() for path in root.rglob("*")
                    if path.is_file() and not path.is_symlink()}
    if actual_files != ledger_files | {"manifest.json", "SHA256SUMS.jsonl"}:
        fail("LEDGER_FILESET_MISMATCH", "ledger 与实际文件集合不一致")
    ledger_contract = _mapping(manifest["ledger"], location="manifest.ledger")
    _exact(ledger_contract, ("path", "entry_count", "size_bytes", "file_sha256", "root_sha256"), location="manifest.ledger")
    expected_ledger = {
        "path": "SHA256SUMS.jsonl", "entry_count": len(ledger), "size_bytes": ledger_path.stat().st_size,
        "file_sha256": sha256_file(ledger_path), "root_sha256": sha256_text(canonical_json(ledger)),
    }
    _assert_equal(ledger_contract, expected_ledger, location="manifest.ledger")
    records = read_jsonl(records_path)
    records_contract = _mapping(manifest["records"], location="manifest.records")
    record_ledger = next((item for item in ledger if item["path"] == "records.jsonl"), None)
    if record_ledger is None:
        fail("LEDGER_INVALID", "ledger 缺少 records.jsonl")
    _assert_equal(records_contract, {**record_ledger, "count": len(records)}, location="manifest.records")
    benchmark = _mapping(manifest["benchmark"], location="manifest.benchmark")
    _exact(benchmark, ("root", *EXPECTED_IDENTITY_FIELDS), location="manifest.benchmark")
    selection = _mapping(manifest["selection"], location="manifest.selection")
    _exact(selection, (
        "config_schema_version", "config_file_sha256", "config_semantic_sha256", "seed", "algorithm_version",
        "sample_count", "source_order", "foreground_bins", "stratification", "ordered_sample_ids",
        "ordered_sample_ids_sha256",
    ), location="manifest.selection")
    if (
        selection["config_schema_version"] != CONFIG_SCHEMA
        or selection["algorithm_version"] != SELECTION_ALGORITHM
        or selection["sample_count"] != 500
    ):
        fail("SCHEMA_MISMATCH", "Pilot config/algorithm/sample_count 非法")
    expected_bins = [
        {"name": "very_small", "lower_exclusive": 0.0, "upper_inclusive": 0.01},
        {"name": "small", "lower_exclusive": 0.01, "upper_inclusive": 0.05},
        {"name": "medium", "lower_exclusive": 0.05, "upper_inclusive": 0.20},
        {"name": "large", "lower_exclusive": 0.20, "upper_inclusive": 1.0},
    ]
    fixed_quotas = {source: (100, 100 if source == "multimodal_landslide" else 75,
                             0 if source == "multimodal_landslide" else 25)
                    for source in ("gdcld", "lmhld", "landslidebench_agent", "landslide4sense", "multimodal_landslide")}
    stratification = _mapping(selection["stratification"], location="selection.stratification")
    _exact(stratification, ("sources", "modality_signatures"), location="selection.stratification")
    actual_quotas = {item["source"]: tuple(item["requested"][key] for key in ("total", "target", "no_target"))
                     for item in stratification["sources"]}
    if selection["seed"] != 20260803 or selection["foreground_bins"] != expected_bins or actual_quotas != fixed_quotas:
        fail("SELECTION_IDENTITY_MISMATCH", "Pilot seed/bin/source quota 不符合 v1 冻结合同")
    if not isinstance(selection["foreground_bins"], list):
        fail("SCHEMA_MISMATCH", "foreground_bins 必须是列表")
    for index, item in enumerate(selection["foreground_bins"]):
        _exact(_mapping(item, location=f"foreground_bins[{index}]"),
               ("name", "lower_exclusive", "upper_inclusive"), location=f"foreground_bins[{index}]")
    rendering = _mapping(manifest["rendering"], location="manifest.rendering")
    _exact(rendering, ("context_margin_ratio", "overlay_alpha", "min_auxiliary_coverage"), location="manifest.rendering")
    if rendering != {"context_margin_ratio": 0.15, "overlay_alpha": 0.45, "min_auxiliary_coverage": 0.8}:
        fail("SCHEMA_MISMATCH", "Pilot rendering 合同不符合 v1 冻结值")
    ordered_ids = selection.get("ordered_sample_ids")
    if not isinstance(ordered_ids, list) or not all(isinstance(item, str) and item for item in ordered_ids):
        fail("SCHEMA_MISMATCH", "selection.ordered_sample_ids 非法")
    record_ids = [str(record.get("sample_id", "")) for record in records]
    if record_ids != ordered_ids or len(record_ids) != len(set(record_ids)):
        fail("SELECTION_IDENTITY_MISMATCH", "records 与有序 sample IDs 不一致或重复")
    if len(records) != selection.get("sample_count"):
        fail("SELECTION_IDENTITY_MISMATCH", "selection sample_count 不一致")
    if selection.get("ordered_sample_ids_sha256") != sha256_text(canonical_json(ordered_ids)):
        fail("SELECTION_IDENTITY_MISMATCH", "ordered sample ID identity 不一致")
    access = _source_access(manifest, root) if verify_source else None
    if access is not None:
        expected_rows, expected_stratification = select_pilot(access.config, access)
        if [row["sample_id"] for row in expected_rows] != ordered_ids:
            fail("SELECTION_IDENTITY_MISMATCH", "有序 IDs 与确定性选择重算不一致")
        _assert_equal(selection["stratification"], expected_stratification, location="selection.stratification")
        if selection["source_order"] != list(access.source_order):
            fail("SELECTION_IDENTITY_MISMATCH", "source_order 与 OA manifest 不一致")
        expected_corpus_id = sha256_text(canonical_json({
            "benchmark": {key: benchmark[key] for key in EXPECTED_IDENTITY_FIELDS},
            "selection": selection["config_semantic_sha256"], "sample_ids": ordered_ids,
        }))
        if manifest["corpus_id"] != expected_corpus_id:
            fail("SELECTION_IDENTITY_MISMATCH", "corpus_id 不一致")
    referenced: set[str] = set()
    source_counts: Counter[str] = Counter()
    target_counts: Counter[str] = Counter()
    for index, value in enumerate(records):
        record = validate_record(value, location=f"records[{index}]")
        if access is not None:
            access.reject_non_train(record["sample_id"])
        referenced.update(_validate_record_assets(root, record, rendering, access))
        source_counts[record["source"]] += 1
        target_counts[record["target_state"]] += 1
    ledger_assets = {path for path in ledger_files if path.startswith("assets/")}
    if referenced != ledger_assets:
        fail("ASSET_REFERENCE_MISMATCH", "record 引用与 ledger 资产集合不一致")
    asset_contract = _mapping(manifest["assets"], location="manifest.assets")
    role_counts = Counter(
        "auxiliary" if path.startswith("assets/auxiliaries/") else path.split("/")[1]
        for path in ledger_assets
    )
    expected_assets = {
        "count": len(ledger_assets),
        "total_bytes": sum(int(item["size_bytes"]) for item in ledger if item["path"] in ledger_assets),
        "role_counts": dict(sorted(role_counts.items())),
    }
    _assert_equal(asset_contract, expected_assets, location="manifest.assets")
    if access is not None:
        train_contract = _mapping(manifest["train_split"], location="manifest.train_split")
        _assert_equal(train_contract, {
            "name": "train", "source_record_count": len(access.train_rows),
            "ordered_sample_ids_sha256": access.train_split_sha256,
        }, location="manifest.train_split")
    return {
        "valid": True, "root": str(root), "manifest_sha256": sha256_file(manifest_path),
        "record_count": len(records), "asset_count": len(ledger_assets), "asset_bytes": expected_assets["total_bytes"],
        "selected_ids_sha256": selection["ordered_sample_ids_sha256"],
        "source_counts": dict(sorted(source_counts.items())), "target_counts": dict(sorted(target_counts.items())),
        "source_verified": verify_source,
    }
