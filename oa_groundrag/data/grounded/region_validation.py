"""Stage 4 v2 Corpus 与 OA-GroundedEval-dev 的独立运行时验证器。"""

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
    context_crop_window,
    deterministic_mask_facts,
    deterministic_shift_mask,
    render_mask_overlay,
)

from .contracts import EXPECTED_IDENTITY_FIELDS, fail, no_target_mask_facts
from .pilot import render_optical
from .region_contracts import (
    ANNOTATION_QUEUE_SCHEMA,
    COUNTERFACTUAL_GROUP_SCHEMA,
    EVAL_MANIFEST_SCHEMA,
    FORMAL_ROLES,
    REGION_MANIFEST_SCHEMA,
    REGION_RECORD_SCHEMA,
    CounterfactualKind,
    RegionTargetStatus,
    RepresentationMode,
    parent_identity,
    validate_region_record,
)
from .region import (
    RegionBenchmarkAccess,
    frozen_stage4a_ids,
    region_asset_identity,
)


def _mapping(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        fail("TYPE_MISMATCH", f"{location} 必须是对象")
    return dict(value)


def _exact(value: Mapping[str, Any], fields: tuple[str, ...], location: str) -> None:
    if set(value) != set(fields):
        fail("SCHEMA_MISMATCH", f"{location} 字段不匹配", details={
            "missing": sorted(set(fields) - set(value)),
            "unknown": sorted(set(value) - set(fields)),
        })


def _ordinary(path: Path) -> None:
    if not path.is_file() or path.is_symlink():
        fail("ASSET_INVALID", f"缺失或链接文件：{path}")
    if path.stat().st_nlink != 1:
        fail("ASSET_HARDLINK", f"输出文件不得是 hardlink：{path}")


def _png(path: Path, *, mode: str) -> Image.Image:
    _ordinary(path)
    with Image.open(path) as handle:
        handle.load()
        if handle.format != "PNG" or handle.mode != mode:
            fail("ASSET_INVALID", f"PNG mode 必须是 {mode}：{path}")
        return handle.copy()


def _same_image(actual: Image.Image, expected: Image.Image, *, location: str) -> None:
    if actual.mode != expected.mode or actual.size != expected.size or not np.array_equal(
        np.asarray(actual), np.asarray(expected)
    ):
        fail("ASSET_CONTENT_MISMATCH", f"资产像素不一致：{location}")


def _same(actual: Any, expected: Any, *, location: str) -> None:
    if canonical_json(actual) != canonical_json(expected):
        fail("FACT_MISMATCH", f"{location} 与重算事实不一致")


def _validate_files(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]], set[str]]:
    root = Path(os.path.abspath(root))
    if first_symlink_component(root) is not None or not root.is_dir():
        fail("CORPUS_INVALID", f"资产根非普通目录：{root}")
    for child in root.rglob("*"):
        if child.is_symlink():
            fail("ASSET_LINK", f"资产根内不得包含 symlink：{child}")
    manifest_path, ledger_path = root / "manifest.json", root / "SHA256SUMS.jsonl"
    _ordinary(manifest_path)
    _ordinary(ledger_path)
    manifest = _mapping(read_json(manifest_path), "manifest")
    ledger = read_jsonl(ledger_path)
    paths = [str(item.get("path", "")) for item in ledger]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        fail("LEDGER_INVALID", "ledger path 必须严格排序且唯一")
    ledger_files: set[str] = set()
    for index, item in enumerate(ledger):
        _exact(item, ("path", "size_bytes", "sha256"), f"ledger[{index}]")
        relative = str(item["path"])
        portable_relative_path(relative, location=f"ledger[{index}].path")
        path = safe_join(root, relative, location=f"ledger[{index}].path")
        _ordinary(path)
        if path.stat().st_size != item["size_bytes"] or sha256_file(path) != item["sha256"]:
            fail("LEDGER_MISMATCH", f"ledger 文件身份不符：{relative}")
        ledger_files.add(relative)
    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    if actual_files != ledger_files | {"manifest.json", "SHA256SUMS.jsonl"}:
        fail("LEDGER_FILESET_MISMATCH", "ledger 与实际文件集合不一致")
    contract = _mapping(manifest["ledger"], "manifest.ledger")
    expected = {
        "path": "SHA256SUMS.jsonl",
        "entry_count": len(ledger),
        "size_bytes": ledger_path.stat().st_size,
        "file_sha256": sha256_file(ledger_path),
        "root_sha256": sha256_text(canonical_json(ledger)),
    }
    _same(contract, expected, location="manifest.ledger")
    return manifest, ledger, ledger_files


def validate_region_asset_files(root: Path | str) -> dict[str, Any]:
    """只验证发布根的 manifest/ledger/全文件身份，不访问 Benchmark shard。"""

    resolved = Path(os.path.abspath(Path(root)))
    manifest, ledger, ledger_files = _validate_files(resolved)
    if manifest.get("schema_version") not in {REGION_MANIFEST_SCHEMA, EVAL_MANIFEST_SCHEMA}:
        fail("SCHEMA_MISMATCH", "单专家工作流只接受 Stage 4 Region/Eval 发布根")
    return {
        "valid": True,
        "root": str(resolved),
        "manifest_sha256": sha256_file(resolved / "manifest.json"),
        "ledger_sha256": sha256_file(resolved / "SHA256SUMS.jsonl"),
        "ledger_entry_count": len(ledger),
        "bound_file_count": len(ledger_files),
    }


def _manifest_benchmark_identity(manifest: Mapping[str, Any]) -> tuple[Path, dict[str, str]]:
    benchmark = _mapping(manifest["benchmark"], "manifest.benchmark")
    _exact(benchmark, ("root", *EXPECTED_IDENTITY_FIELDS), "manifest.benchmark")
    root = Path(str(benchmark["root"]))
    expected = {field: str(benchmark[field]) for field in EXPECTED_IDENTITY_FIELDS}
    return root, expected


def _benchmark_access(manifest: Mapping[str, Any], *, split: str) -> RegionBenchmarkAccess:
    root, expected = _manifest_benchmark_identity(manifest)
    return RegionBenchmarkAccess(root, expected, allowed_split=split)


def _validate_queue(
    root: Path,
    records: list[dict[str, Any]],
    ledger: list[dict[str, Any]],
) -> None:
    queue_path = root / "annotation_queue.jsonl"
    guideline_path = root / "annotation_guideline.json"
    _ordinary(queue_path)
    _ordinary(guideline_path)
    queue = read_jsonl(queue_path)
    if len(queue) != len(records):
        fail("ANNOTATION_INVALID", "annotation queue 数量必须等于 records")
    by_record = {record["record_id"]: record for record in records}
    if len(by_record) != len(records):
        fail("SCHEMA_MISMATCH", "record_id 重复")
    for index, row in enumerate(queue):
        if row.get("schema_version") != ANNOTATION_QUEUE_SCHEMA:
            fail("SCHEMA_MISMATCH", f"annotation_queue[{index}] schema 非法")
        record = by_record.get(row.get("record_id"))
        if record is None or row.get("assets") != record["assets"] or row.get("program_facts") != record["program_facts"]:
            fail("ANNOTATION_INVALID", f"annotation_queue[{index}] 未绑定原 record")
        if row.get("asset_identity_sha256") != region_asset_identity(root, record["assets"]):
            fail("ANNOTATION_INVALID", f"annotation_queue[{index}] asset identity 不一致")
    for path in ("annotation_queue.jsonl", "annotation_guideline.json", "records.jsonl"):
        if not any(item["path"] == path for item in ledger):
            fail("LEDGER_INVALID", f"ledger 缺少 {path}")


def _expected_mask_and_crop(
    record: Mapping[str, Any],
    loaded: Any,
    access: RegionBenchmarkAccess,
    optical: Image.Image,
    rendering: Mapping[str, Any],
) -> tuple[np.ndarray, tuple[int, int, int, int] | None]:
    gt_mask = loaded.mask
    counterfactual = record["program_facts"].get("counterfactual")
    kind = None if counterfactual is None else counterfactual.get("kind")
    if kind == CounterfactualKind.EMPTY_MASK.value:
        return np.zeros_like(gt_mask), None
    if kind == CounterfactualKind.MASK_SHIFT.value:
        shifted, parameters = deterministic_shift_mask(gt_mask)
        expected = {"kind": kind, **parameters}
        _same(counterfactual, expected, location=f"{record['record_id']}.counterfactual")
        mask = shifted
    elif kind == CounterfactualKind.MASK_SWAP.value:
        donor_sample_id = counterfactual.get("donor_sample_id")
        if not isinstance(donor_sample_id, str):
            fail("COUNTERFACTUAL_INVALID", "mask_swap 缺少 donor_sample_id")
        donor = access.load(donor_sample_id)
        donor_parent, _ = parent_identity(donor.row)
        base_parent, _ = parent_identity(loaded.row)
        if (
            donor_parent != base_parent
            or donor.row.get("record_sha256") != counterfactual.get("donor_record_sha256")
            or donor.mask.shape != gt_mask.shape
            or np.array_equal(donor.mask, gt_mask)
        ):
            fail("COUNTERFACTUAL_INVALID", "mask_swap donor identity/shape/mask 非法")
        _same_image(render_optical(donor.sample), optical, location=f"{record['record_id']}.mask_swap.optical")
        mask = donor.mask
    else:
        mask = gt_mask
    if not bool(mask.any()) or "context_crop" not in record["formal_model_input_roles"]:
        return mask, None
    if kind == CounterfactualKind.CONTEXT_REMOVAL.value:
        ys, xs = np.nonzero(mask)
        return mask, (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    return mask, context_crop_window(
        mask,
        image_size=optical.size,
        margin_ratio=float(rendering["context_margin_ratio"]),
    )


def _validate_record_assets(
    root: Path,
    record: Mapping[str, Any],
    access: RegionBenchmarkAccess,
    *,
    rendering: Mapping[str, Any],
) -> set[str]:
    loaded = access.load(str(record["sample_id"]))
    row = loaded.row
    expected_parent, expected_parent_status = parent_identity(row)
    if record["parent_id"] != expected_parent or record["parent_identity_status"] != expected_parent_status.value:
        fail("RECORD_IDENTITY_MISMATCH", f"parent identity 不一致：{record['record_id']}")
    expected_source = {
        "benchmark_record_sha256": str(row["record_sha256"]),
        "source_record_sha256": str(row["source_record_sha256"]),
        "storage_shard": str(row["storage"]["shard"]),
        "storage_row": int(row["storage"]["row"]),
        "storage_shard_sha256": access.shard_sha256(row),
        "source_group_id": row.get("source_group_id"),
        "group_status": str(row.get("group_status", "unknown")),
    }
    _same(record["source_identity"], expected_source, location=f"{record['record_id']}.source_identity")
    assets = record["assets"]
    optical_path, mask_path = str(assets["optical_full"]), str(assets["binary_mask"])
    optical = _png(safe_join(root, optical_path, location="assets.optical_full"), mode="RGB")
    _same_image(optical, render_optical(loaded.sample), location=f"{record['record_id']}.optical_full")
    mask_image = _png(safe_join(root, mask_path, location="assets.binary_mask"), mode="L")
    raw = np.asarray(mask_image)
    if not set(np.unique(raw).tolist()).issubset({0, 255}):
        fail("MASK_INVALID", f"binary mask 非 0/255：{record['record_id']}")
    actual_mask = raw == 255
    expected_mask, expected_window = _expected_mask_and_crop(record, loaded, access, optical, rendering)
    if not np.array_equal(actual_mask, expected_mask):
        fail("MASK_SOURCE_MISMATCH", f"mask 与 GT/反事实重算不一致：{record['record_id']}")
    target = bool(actual_mask.any())
    expected_status = RegionTargetStatus.TARGET_PRESENT.value if target else RegionTargetStatus.NO_TARGET.value
    if record["target_status"] != expected_status:
        fail("MASK_INVALID", f"target_status 不一致：{record['record_id']}")
    if target:
        expected_facts = {
            "status": "available",
            **deterministic_mask_facts(actual_mask),
            "crop_window_xyxy_pixel_half_open": None if expected_window is None else list(expected_window),
        }
    else:
        expected_facts = no_target_mask_facts()
    _same(record["program_facts"]["mask"], expected_facts, location=f"{record['record_id']}.mask_facts")
    referenced = {optical_path, mask_path}
    crop_path = assets["context_crop"]
    if expected_window is None:
        if crop_path is not None:
            fail("ASSET_REFERENCE_MISMATCH", f"不应存在 context crop：{record['record_id']}")
    else:
        if not isinstance(crop_path, str):
            fail("ASSET_REFERENCE_MISMATCH", f"缺少 context crop：{record['record_id']}")
        _same_image(
            _png(safe_join(root, crop_path, location="assets.context_crop"), mode="RGB"),
            optical.crop(expected_window),
            location=f"{record['record_id']}.context_crop",
        )
        referenced.add(crop_path)
    overlay_path = assets["audit_overlay"]
    if overlay_path is not None:
        if not isinstance(overlay_path, str) or not target:
            fail("ASSET_REFERENCE_MISMATCH", f"audit overlay 引用非法：{record['record_id']}")
        _same_image(
            _png(safe_join(root, overlay_path, location="assets.audit_overlay"), mode="RGB"),
            render_mask_overlay(optical, actual_mask, alpha=float(rendering["overlay_alpha"])),
            location=f"{record['record_id']}.audit_overlay",
        )
        referenced.add(overlay_path)
    if "audit_overlay" in record["formal_model_input_roles"]:
        fail("ASSET_ROLE_LEAKAGE", f"overlay 进入正式输入：{record['record_id']}")
    return referenced


def _validate_common(
    root: Path,
    *,
    expected_schema: str,
    split: str,
    verify_source: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], set[str], RegionBenchmarkAccess | None]:
    manifest, ledger, ledger_files = _validate_files(root)
    if manifest.get("schema_version") != expected_schema:
        fail("SCHEMA_MISMATCH", "manifest schema_version 非法")
    for flag in ("expert_annotation_completed", "formal_acceptance"):
        if manifest.get(flag) is not False:
            fail("SCHEMA_MISMATCH", f"manifest.{flag} 必须为 false")
    records = read_jsonl(root / "records.jsonl")
    records_item = next((item for item in ledger if item["path"] == "records.jsonl"), None)
    queue_item = next((item for item in ledger if item["path"] == "annotation_queue.jsonl"), None)
    if records_item is None or queue_item is None:
        fail("LEDGER_INVALID", "ledger 缺少 records/annotation_queue")
    _same(manifest["records"], {**records_item, "count": len(records)}, location="manifest.records")
    queue_count = len(read_jsonl(root / "annotation_queue.jsonl"))
    _same(
        manifest["annotation_queue"],
        {**queue_item, "count": queue_count},
        location="manifest.annotation_queue",
    )
    access = _benchmark_access(manifest, split=split) if verify_source else None
    referenced: set[str] = set()
    for index, value in enumerate(records):
        record = validate_region_record(value, expected_split=split, location=f"records[{index}]")
        if access is not None:
            referenced.update(_validate_record_assets(root, record, access, rendering=manifest["rendering"]))
        else:
            referenced.update(path for path in record["assets"].values() if isinstance(path, str))
    ledger_assets = {path for path in ledger_files if path.startswith("assets/")}
    if referenced != ledger_assets:
        fail("ASSET_REFERENCE_MISMATCH", "records 引用与 ledger 资产集合不一致")
    _validate_queue(root, records, ledger)
    expected_assets = {
        "count": len(ledger_assets),
        "total_bytes": sum(item["size_bytes"] for item in ledger if item["path"] in ledger_assets),
        "role_counts": dict(sorted(Counter(path.split("/")[1] for path in ledger_assets).items())),
    }
    _same(manifest["assets"], expected_assets, location="manifest.assets")
    return manifest, ledger, records, ledger_assets, access


def validate_region_corpus(root: Path | str, *, verify_source: bool = True) -> dict[str, Any]:
    root = Path(os.path.abspath(Path(root)))
    manifest, ledger, records, assets, access = _validate_common(
        root,
        expected_schema=REGION_MANIFEST_SCHEMA,
        split="train",
        verify_source=verify_source,
    )
    if manifest.get("silver_generated") is not False or manifest.get("corpus_name") != "mask_grounded_region_corpus_train_v1_500":
        fail("SCHEMA_MISMATCH", "Region Corpus 固定状态/名称非法")
    selection = _mapping(manifest["selection"], "manifest.selection")
    ids = selection.get("ordered_sample_ids")
    if (
        not isinstance(ids, list)
        or len(ids) != 500
        or [record["sample_id"] for record in records] != ids
        or sha256_text(canonical_json(ids)) != selection.get("ordered_sample_ids_sha256")
    ):
        fail("SELECTION_IDENTITY_MISMATCH", "Region Corpus ordered IDs 不一致")
    _, benchmark_identity = _manifest_benchmark_identity(manifest)
    expected_corpus_id = sha256_text(canonical_json({
        "benchmark": benchmark_identity,
        "config": selection.get("config_semantic_sha256"),
        "sample_ids": ids,
    }))
    if manifest.get("corpus_id") != expected_corpus_id:
        fail("SELECTION_IDENTITY_MISMATCH", "Region Corpus corpus_id 重算不一致")
    if access is not None:
        from .region_contracts import FrozenStage4ASelection, RegionCorpusConfig

        stage4a = selection["stage4a"]
        frozen = FrozenStage4ASelection(
            Path(stage4a["root"]), stage4a["manifest_sha256"], stage4a["records_sha256"],
            stage4a["ledger_sha256"], stage4a["ordered_sample_ids_sha256"], int(stage4a["sample_count"]),
        )
        dummy = RegionCorpusConfig(
            root / "manifest.json", selection["config_file_sha256"], selection["config_semantic_sha256"],
            access.root, access.expected, frozen, root, float(manifest["rendering"]["context_margin_ratio"]),
            float(manifest["rendering"]["overlay_alpha"]),
        )
        frozen_ids, _ = frozen_stage4a_ids(dummy)
        if frozen_ids != ids:
            fail("STAGE4A_IDENTITY_MISMATCH", "Region Corpus 未精确复用 Stage 4A IDs")
    return {
        "valid": True,
        "root": str(root),
        "manifest_sha256": sha256_file(root / "manifest.json"),
        "records_sha256": sha256_file(root / "records.jsonl"),
        "ledger_sha256": sha256_file(root / "SHA256SUMS.jsonl"),
        "record_count": len(records),
        "asset_count": len(assets),
        "asset_bytes": sum(item["size_bytes"] for item in ledger if item["path"] in assets),
        "source_counts": dict(sorted(Counter(record["source"] for record in records).items())),
        "target_counts": dict(sorted(Counter(record["target_status"] for record in records).items())),
        "source_verified": verify_source,
    }


def validate_eval_dev(
    root: Path | str,
    *,
    train_corpus_root: Path | str,
    verify_source: bool = True,
) -> dict[str, Any]:
    root = Path(os.path.abspath(Path(root)))
    train_root = Path(os.path.abspath(Path(train_corpus_root)))
    train_report = validate_region_corpus(train_root, verify_source=verify_source)
    manifest, ledger, records, assets, _ = _validate_common(
        root,
        expected_schema=EVAL_MANIFEST_SCHEMA,
        split="val",
        verify_source=verify_source,
    )
    if manifest.get("development_only") is not True or manifest.get("sealed_test_accessed") is not False:
        fail("FORMAL_EVALUATION_FORBIDDEN", "Eval-dev 必须 development_only 且 sealed_test_accessed=false")
    selection = _mapping(manifest["selection"], "manifest.selection")
    baseline_ids = selection.get("baseline_record_ids")
    if not isinstance(baseline_ids, list) or len(baseline_ids) != 100:
        fail("SELECTION_IDENTITY_MISMATCH", "Eval-dev 必须有 100 个 baseline records")
    by_id = {record["record_id"]: record for record in records}
    baselines = [by_id.get(record_id) for record_id in baseline_ids]
    if any(record is None for record in baselines):
        fail("SELECTION_IDENTITY_MISMATCH", "baseline_record_ids 引用未知 record")
    baseline_records = [record for record in baselines if record is not None]
    if len({record["parent_id"] for record in baseline_records}) != 100:
        fail("PARENT_OVERLAP", "Eval baseline parent 必须唯一")
    train_records = read_jsonl(train_root / "records.jsonl")
    train_samples = {record["sample_id"] for record in train_records}
    train_parents = {record["parent_id"] for record in train_records}
    train_contract = _mapping(manifest.get("train_corpus"), "manifest.train_corpus")
    _exact(
        train_contract,
        ("root", "manifest_sha256", "records_sha256", "ledger_sha256", "parent_count"),
        "manifest.train_corpus",
    )
    expected_train_contract = {
        "root": str(train_root),
        "manifest_sha256": train_report["manifest_sha256"],
        "records_sha256": train_report["records_sha256"],
        "ledger_sha256": train_report["ledger_sha256"],
        "parent_count": len(train_parents),
    }
    _same(
        train_contract,
        expected_train_contract,
        location="manifest.train_corpus",
    )
    if train_samples & {record["sample_id"] for record in baseline_records}:
        fail("SPLIT_LEAKAGE", "train/val sample 相交")
    if train_parents & {record["parent_id"] for record in baseline_records}:
        fail("PARENT_OVERLAP", "train/val parent 相交")
    ordered_samples = [record["sample_id"] for record in baseline_records]
    ordered_parents = [record["parent_id"] for record in baseline_records]
    if (
        selection.get("sample_count") != len(baseline_records)
        or selection.get("ordered_sample_ids") != ordered_samples
        or selection.get("ordered_parent_ids") != ordered_parents
    ):
        fail("SELECTION_IDENTITY_MISMATCH", "Eval baseline sample/parent 顺序身份不一致")
    _, benchmark_identity = _manifest_benchmark_identity(manifest)
    expected_eval_id = sha256_text(canonical_json({
        "benchmark": benchmark_identity,
        "config": selection.get("config_semantic_sha256"),
        "baseline_samples": ordered_samples,
    }))
    if manifest.get("eval_id") != expected_eval_id:
        fail("SELECTION_IDENTITY_MISMATCH", "Eval eval_id 重算不一致")
    groups = read_jsonl(root / "counterfactual_groups.jsonl")
    target_baselines = [record for record in baseline_records if record["target_status"] == "target_present"]
    if len(groups) != len(target_baselines):
        fail("COUNTERFACTUAL_INVALID", "每个 target baseline 必须有一个反事实组")
    required = {
        CounterfactualKind.BASELINE.value,
        CounterfactualKind.EMPTY_MASK.value,
        CounterfactualKind.MASK_SHIFT.value,
        CounterfactualKind.CONTEXT_REMOVAL.value,
    }
    for index, group in enumerate(groups):
        if group.get("schema_version") != COUNTERFACTUAL_GROUP_SCHEMA:
            fail("SCHEMA_MISMATCH", f"counterfactual_groups[{index}] schema 非法")
        variants = group.get("variants")
        if not isinstance(variants, dict) or not required.issubset(variants):
            fail("COUNTERFACTUAL_INVALID", f"counterfactual_groups[{index}] 不完整")
        if any(record_id not in by_id for record_id in variants.values()):
            fail("COUNTERFACTUAL_INVALID", f"counterfactual_groups[{index}] 引用未知 record")
        if bool(group.get("mask_swap_eligible")) != (CounterfactualKind.MASK_SWAP.value in variants):
            fail("COUNTERFACTUAL_INVALID", f"counterfactual_groups[{index}] mask_swap eligibility 不一致")
    grouped_record_ids = set(baseline_ids)
    for group in groups:
        grouped_record_ids.update(group["variants"].values())
    if grouped_record_ids != set(by_id):
        fail("COUNTERFACTUAL_INVALID", "Eval records 必须且只能由 baseline/反事实组引用")
    if not any(item["path"] == "counterfactual_groups.jsonl" for item in ledger):
        fail("LEDGER_INVALID", "Eval ledger 缺少 counterfactual_groups.jsonl")
    return {
        "valid": True,
        "root": str(root),
        "manifest_sha256": sha256_file(root / "manifest.json"),
        "records_sha256": sha256_file(root / "records.jsonl"),
        "ledger_sha256": sha256_file(root / "SHA256SUMS.jsonl"),
        "record_count": len(records),
        "baseline_count": len(baseline_records),
        "counterfactual_group_count": len(groups),
        "asset_count": len(assets),
        "train_corpus_manifest_sha256": train_report["manifest_sha256"],
        "formal_acceptance": False,
        "sealed_test_accessed": False,
        "source_verified": verify_source,
    }
