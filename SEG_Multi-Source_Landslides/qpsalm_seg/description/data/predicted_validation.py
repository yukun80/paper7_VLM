#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deep replay, validation, and exact-fold merge for predicted regions."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

from qpsalm_seg.engine.checkpoint import CHECKPOINT_FORMAT
from qpsalm_seg.paths import (
    resolve_project_path,
    to_project_ref,
    validate_output_replacement_safety,
)

from .expert_contracts import require_frozen_expert_bridge, validate_expert_rows
from .oof import load_oof_manifest, oof_manifest_artifact_binding
from ..protocols.io import canonical_sha256, sha256_file, strict_json_loads
from .predicted_contracts import (
    FIXED_PREDICTION_ARTIFACT_PROTOCOL,
    OOF_MERGE_PROTOCOL,
    PREDICTED_REGION_FORMAT,
    atomic_write_prediction_json,
    atomic_write_prediction_jsonl,
    read_prediction_jsonl,
)
from .predicted_checkpoint import validate_oof_checkpoint


def revalidate_fixed_predicted_index(
    index_path: str | Path,
    *,
    split: str,
    expected_expert_gate_audit: dict[str, Any],
    require_complete: bool = True,
) -> dict[str, Any]:
    """Deeply replay a fixed val/test prediction publication before use."""
    if split not in {"val", "test"}:
        raise ValueError("fixed predicted artifact replay 只接受 val/test")
    path = resolve_project_path(index_path) or Path(index_path)
    report_path = path.parent / "report.json"
    if not path.is_file() or not report_path.is_file():
        raise FileNotFoundError("fixed predicted index/report 缺失")
    report = strict_json_loads(report_path.read_text(encoding="utf-8"))
    if (
        report.get("format") != PREDICTED_REGION_FORMAT
        or report.get("validation_protocol") != FIXED_PREDICTION_ARTIFACT_PROTOCOL
        or str(report.get("split")) != split
    ):
        raise ValueError("fixed predicted report protocol/split 非法")
    reported_index = resolve_project_path(str(report.get("index") or ""))
    if (
        reported_index is None
        or reported_index.resolve(strict=False) != path.resolve(strict=False)
        or str(report.get("index_sha256") or "") != sha256_file(path)
    ):
        raise ValueError("fixed predicted index path/hash 已漂移")
    source_path = resolve_project_path(str(report.get("source_bridge_index") or ""))
    if source_path is None or not source_path.is_file():
        raise FileNotFoundError("fixed predicted report 绑定的 expert source 缺失")
    expected_source = source_path.parent.parent / f"indexes/expert_{split}.jsonl"
    if (
        source_path.resolve(strict=False) != expected_source.resolve(strict=False)
        or str(report.get("source_bridge_index_sha256") or "") != sha256_file(source_path)
    ):
        raise ValueError("fixed predicted report 未绑定当前 expert split index")
    current_gate = require_frozen_expert_bridge(source_path.parent.parent)
    if (
        report.get("expert_gate_audit") != current_gate
        or current_gate != expected_expert_gate_audit
    ):
        raise ValueError("fixed predicted report 与当前 frozen Bridge gate 不一致")
    source_rows = read_prediction_jsonl(source_path)
    validate_expert_rows(source_rows, stage="bridge_expert", split=split)
    source_by_parent: dict[str, dict[str, Any]] = {}
    for source in source_rows:
        if str(source.get("split")) != split:
            raise ValueError("fixed predicted expert source 包含错误 split")
        if source.get("region_source") != "gt_global_mask":
            continue
        parent = str(source.get("parent_sample_id") or "")
        previous = source_by_parent.setdefault(parent, source)
        if not parent or previous != source:
            raise ValueError("fixed predicted expert source 的 global parent 缺失或重复")
    if not source_by_parent:
        raise ValueError("fixed predicted expert source 没有 gt_global_mask parent")

    checkpoint_path = resolve_project_path(str(report.get("checkpoint") or ""))
    if checkpoint_path is None or not checkpoint_path.is_file():
        raise FileNotFoundError("fixed predicted segmentation checkpoint 缺失")
    checkpoint_sha256 = sha256_file(checkpoint_path)
    if checkpoint_sha256 != str(report.get("checkpoint_sha256") or ""):
        raise ValueError("fixed predicted segmentation checkpoint hash 已漂移")
    checkpoint_payload = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    if (
        not isinstance(checkpoint_payload, dict)
        or checkpoint_payload.get("format") != CHECKPOINT_FORMAT
        or int(checkpoint_payload.get("step", -1)) <= 0
        or int(checkpoint_payload.get("step", -1))
        != int(report.get("checkpoint_step", -2))
    ):
        raise ValueError("fixed predicted segmentation checkpoint payload/step 非法")

    rows = read_prediction_jsonl(path)
    _revalidate_prediction_publication_directory(path, rows, split=split)
    validate_expert_rows(rows, stage="predicted_mask", split=split)
    by_parent: dict[str, dict[str, Any]] = {}
    mask_paths: set[str] = set()
    mask_inventory: list[dict[str, str]] = []
    mask_bytes = 0
    for row in rows:
        parent = str(row.get("parent_sample_id") or "")
        provenance = row.get("prediction_provenance")
        source = source_by_parent.get(parent)
        if (
            row.get("schema_version") != PREDICTED_REGION_FORMAT
            or str(row.get("split")) != split
            or row.get("region_source") != "predicted_proposal"
            or row.get("region_id") != "predicted_global"
            or not isinstance(provenance, dict)
            or source is None
        ):
            raise ValueError(f"fixed predicted row protocol/source 非法: {parent}")
        if (
            str(provenance.get("checkpoint") or "") != to_project_ref(checkpoint_path)
            or str(provenance.get("checkpoint_sha256") or "") != checkpoint_sha256
            or int(provenance.get("checkpoint_step", -1))
            != int(checkpoint_payload["step"])
            or str(provenance.get("split")) != split
            or provenance.get("out_of_fold_verified") is not True
            or provenance.get("checkpoint_fold") is not None
            or provenance.get("fold_audit") is not None
            or str(provenance.get("source_bridge_record_id") or "")
            != str(source.get("bridge_record_id") or source.get("sample_id") or "")
            or str(provenance.get("source_expert_record_sha256") or "")
            != canonical_sha256(source)
            or row.get("expert_target") != source.get("expert_target")
            or row.get("review") != source.get("review")
        ):
            raise ValueError(f"fixed predicted row provenance/source 已漂移: {parent}")
        if parent in by_parent:
            raise ValueError(f"fixed predicted parent 重复: {parent}")
        expected_mask_path = path.parent / "masks" / split / f"{parent}.npy"
        mask_path, size, mask_sha256 = _validate_prediction_mask(
            row,
            parent,
            expected_path=expected_mask_path,
        )
        if mask_path in mask_paths:
            raise ValueError(f"fixed predicted mask 被多个 parent 复用: {mask_path}")
        mask_paths.add(mask_path)
        mask_bytes += size
        mask_inventory.append({
            "parent_sample_id": parent,
            "path": str(row["region_mask"]["path"]),
            "sha256": mask_sha256,
        })
        by_parent[parent] = row
    eligible_parents = sorted(source_by_parent)
    try:
        requested_max_parents = int(report.get("requested_max_parents", -1))
    except (TypeError, ValueError) as error:
        raise ValueError("fixed predicted requested_max_parents 非法") from error
    if requested_max_parents < 0:
        raise ValueError("fixed predicted report 缺少 requested_max_parents")
    expected_parents = (
        eligible_parents[:requested_max_parents]
        if requested_max_parents > 0
        else eligible_parents
    )
    observed_parents = sorted(by_parent)
    if observed_parents != expected_parents:
        raise ValueError("fixed predicted index 未按确定性 prefix 覆盖 expert split parents")
    population_complete = expected_parents == eligible_parents
    if require_complete and not population_complete:
        raise ValueError("正式 fixed predicted index 必须完整覆盖 expert split global parents")
    expected_report = {
        "num_parents": len(observed_parents),
        "num_eligible_parents": len(eligible_parents),
        "population_complete": population_complete,
        "population_sha256": canonical_sha256(observed_parents),
        "mask_inventory_sha256": canonical_sha256(mask_inventory),
        "mask_bytes": mask_bytes,
    }
    if {name: report.get(name) for name in expected_report} != expected_report:
        raise ValueError("fixed predicted report population/mask statistics 未通过重放")
    return {
        "protocol": FIXED_PREDICTION_ARTIFACT_PROTOCOL,
        "report": to_project_ref(report_path),
        "report_sha256": sha256_file(report_path),
        "index": to_project_ref(path),
        "index_sha256": sha256_file(path),
        "split": split,
        "segmentation_checkpoint": to_project_ref(checkpoint_path),
        "segmentation_checkpoint_sha256": checkpoint_sha256,
        "segmentation_checkpoint_step": int(checkpoint_payload["step"]),
        "source_bridge_index": to_project_ref(source_path),
        "source_bridge_index_sha256": sha256_file(source_path),
        "num_parents": len(observed_parents),
        "population_complete": population_complete,
        "population_sha256": expected_report["population_sha256"],
        "mask_inventory_sha256": expected_report["mask_inventory_sha256"],
    }


def _validate_prediction_mask(
    row: dict[str, Any],
    parent: str,
    *,
    expected_path: Path | None = None,
) -> tuple[str, int, str]:
    mask = row.get("region_mask")
    if not isinstance(mask, dict) or not mask.get("path"):
        raise ValueError(f"OOF prediction 缺少 region_mask.path: {parent}")
    path = resolve_project_path(str(mask["path"])) or Path(str(mask["path"]))
    if (
        expected_path is not None
        and path.resolve(strict=False) != expected_path.resolve(strict=False)
    ):
        raise ValueError(
            f"fixed predicted mask path 不在确定性 publication 目录: {parent}"
        )
    if not path.is_file():
        raise FileNotFoundError(f"OOF prediction mask 不存在: {parent} path={path}")
    observed_hash = sha256_file(path)
    if observed_hash != str(mask.get("sha256") or ""):
        raise ValueError(f"OOF prediction mask hash 不一致: {parent}")
    values = np.load(path, allow_pickle=False)
    try:
        expected_shape = tuple(int(value) for value in mask.get("shape", []))
    except (TypeError, ValueError) as error:
        raise ValueError(f"OOF prediction mask shape metadata 非法: {parent}") from error
    if values.ndim != 2 or tuple(values.shape) != expected_shape:
        raise ValueError(
            f"OOF prediction mask shape 不一致: {parent} "
            f"observed={tuple(values.shape)} expected={expected_shape}"
        )
    if not np.isin(values, (0, 1)).all():
        raise ValueError(f"OOF prediction mask 不是 binary: {parent}")
    return (
        str(path.resolve(strict=False)),
        int(path.stat().st_size),
        observed_hash,
    )


def _revalidate_prediction_publication_directory(
    index_path: Path,
    rows: list[dict[str, Any]],
    *,
    split: str,
) -> None:
    """Require one deterministic mask file per indexed parent and no leftovers."""
    root = index_path.parent.resolve(strict=False)
    mask_root = (root / "masks" / split).resolve(strict=False)
    expected_files: set[str] = set()
    for row in rows:
        parent = str(row.get("parent_sample_id") or "")
        mask = row.get("region_mask")
        path = resolve_project_path(
            str(mask.get("path") or "") if isinstance(mask, dict) else ""
        )
        expected = (mask_root / f"{parent}.npy").resolve(strict=False)
        try:
            expected.relative_to(mask_root)
        except ValueError as exc:
            raise ValueError(
                f"predicted parent 逃逸 publication mask 目录: {parent!r}"
            ) from exc
        if not parent or path is None or path.resolve(strict=False) != expected:
            raise ValueError(
                f"predicted mask path 不在确定性 publication 目录: {parent!r}"
            )
        expected_files.add(str(expected))
    observed_files = {
        str(candidate.resolve(strict=False))
        for candidate in mask_root.rglob("*")
        if candidate.is_file()
    } if mask_root.is_dir() else set()
    if observed_files != expected_files:
        raise ValueError(
            "predicted mask 目录存在未绑定、临时或缺失文件"
        )
    part_files = sorted(
        str(candidate.relative_to(root))
        for candidate in root.rglob("*.part")
        if candidate.is_file()
    )
    if part_files:
        raise ValueError(
            f"predicted publication 残留 .part 文件: {part_files[:8]}"
        )


def _audit_oof_prediction_rows(
    rows: list[dict[str, Any]],
    *,
    manifest: dict[str, Any],
    fold_manifest: str | Path,
) -> dict[str, Any]:
    parent_to_fold = {
        str(parent): str(fold)
        for parent, fold in manifest["parent_to_fold"].items()
    }
    source_bridge = resolve_project_path(str(manifest["source_bridge_index"]))
    if source_bridge is None or not source_bridge.is_file():
        raise FileNotFoundError("OOF source expert_train 在 row replay 时缺失")
    source_by_parent: dict[str, dict[str, Any]] = {}
    for source in read_prediction_jsonl(source_bridge):
        if source.get("region_source") != "gt_global_mask":
            continue
        parent = str(source.get("parent_sample_id") or "")
        previous = source_by_parent.setdefault(parent, source)
        if previous != source:
            raise ValueError(f"OOF source expert_train 存在重复 gt_global_mask parent={parent}")

    merged: dict[str, dict[str, Any]] = {}
    fold_counts: Counter[str] = Counter()
    mask_paths: set[str] = set()
    mask_inventory: list[dict[str, Any]] = []
    mask_bytes = 0
    checkpoint_audits: dict[str, dict[str, Any]] = {}
    live_cache: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        parent = str(row.get("parent_sample_id") or "")
        provenance = row.get("prediction_provenance")
        if row.get("schema_version") != PREDICTED_REGION_FORMAT:
            raise ValueError(f"predicted region format 非法: {parent}")
        if (
            str(row.get("split")) != "train"
            or row.get("region_source") != "predicted_proposal"
            or row.get("region_id") != "predicted_global"
        ):
            raise ValueError(f"OOF merge 只接受 train predicted_global row: {parent}")
        if not parent or parent not in parent_to_fold:
            raise ValueError(f"prediction parent 不在 fold manifest: {parent}")
        if not isinstance(provenance, dict):
            raise ValueError(f"prediction 缺少 provenance: {parent}")
        fold = parent_to_fold[parent]
        if str(provenance.get("checkpoint_fold")) != fold:
            raise ValueError(f"prediction fold 归属错误: {parent}")
        if provenance.get("out_of_fold_verified") is not True:
            raise ValueError(f"prediction 未通过 OOF checkpoint 审计: {parent}")
        source = source_by_parent.get(parent)
        if source is None:
            raise ValueError(f"prediction parent 不在当前 expert_train gt_global_mask: {parent}")
        if (
            str(provenance.get("source_bridge_record_id") or "")
            != str(source.get("bridge_record_id") or source.get("sample_id") or "")
            or str(provenance.get("source_expert_record_sha256") or "")
            != canonical_sha256(source)
            or row.get("expert_target") != source.get("expert_target")
            or row.get("review") != source.get("review")
        ):
            raise ValueError(f"prediction expert source record 绑定不一致: {parent}")
        checkpoint_ref = str(provenance.get("checkpoint") or "")
        cache_key = (fold, checkpoint_ref)
        if cache_key not in live_cache:
            expected_holdout = (manifest["folds"][fold] or {}).get("holdout_index")
            live_cache[cache_key] = validate_oof_checkpoint(
                checkpoint_ref,
                manifest,
                fold,
                str(expected_holdout or ""),
                fold_manifest=fold_manifest,
            )
        live_audit = live_cache[cache_key]
        if provenance.get("fold_audit") != live_audit:
            raise ValueError(f"prediction fold audit 未通过当前 artifact 重放: {parent}")
        if (
            str(provenance.get("checkpoint_sha256") or "")
            != live_audit["checkpoint_sha256"]
            or int(provenance.get("checkpoint_step", -1))
            != int(live_audit["checkpoint_step"])
            or str(provenance.get("fold_manifest") or "")
            != str(live_audit["fold_manifest"])
            or str(provenance.get("fold_manifest_sha256") or "")
            != str(live_audit["fold_manifest_sha256"])
        ):
            raise ValueError(f"prediction checkpoint/manifest binding 不一致: {parent}")
        previous_audit = checkpoint_audits.setdefault(fold, live_audit)
        if previous_audit != live_audit:
            raise ValueError(f"同一 fold 使用了多个 segmentation checkpoint: fold={fold}")
        if parent in merged:
            raise ValueError(f"OOF prediction parent 重复: {parent}")
        mask_path, size, mask_sha256 = _validate_prediction_mask(row, parent)
        if mask_path in mask_paths:
            raise ValueError(f"多个 parent 复用了同一个 predicted mask: {mask_path}")
        mask_paths.add(mask_path)
        mask_bytes += size
        mask_inventory.append({
            "parent_sample_id": parent,
            "path": mask_path,
            "sha256": mask_sha256,
            "bytes": size,
        })
        fold_counts[fold] += 1
        merged[parent] = row
    validate_expert_rows(
        list(merged.values()), stage="predicted_mask", split="train"
    )
    missing = sorted(set(parent_to_fold) - set(merged))
    unexpected = sorted(set(merged) - set(parent_to_fold))
    if missing or unexpected:
        raise ValueError(
            "OOF prediction parent 覆盖不精确: "
            f"missing={len(missing)} unexpected={len(unexpected)} "
            f"examples={(missing + unexpected)[:8]}"
        )
    expected_fold_counts = Counter(parent_to_fold.values())
    if fold_counts != expected_fold_counts:
        raise ValueError(
            "OOF prediction fold 覆盖不一致: "
            f"observed={dict(fold_counts)} expected={dict(expected_fold_counts)}"
        )
    if set(checkpoint_audits) != set(manifest["folds"]):
        raise ValueError("OOF prediction 未绑定每个 fold 的唯一 checkpoint")
    ordered_rows = [merged[parent] for parent in sorted(merged)]
    return {
        "rows": ordered_rows,
        "num_parents": len(ordered_rows),
        "parents_per_fold": dict(sorted(fold_counts.items())),
        "num_unique_masks": len(mask_paths),
        "mask_bytes": mask_bytes,
        "mask_inventory_sha256": canonical_sha256(
            sorted(mask_inventory, key=lambda value: value["parent_sample_id"])
        ),
        "fold_checkpoint_audits": dict(sorted(checkpoint_audits.items())),
        "population_sha256": canonical_sha256(
            [str(row["parent_sample_id"]) for row in ordered_rows]
        ),
    }


def merge_oof_predictions(
    *,
    fold_manifest: str | Path,
    input_indexes: list[str | Path],
    output: str | Path,
) -> dict[str, Any]:
    """Merge OOF rows only after replaying sources, folds, checkpoints and masks."""
    if not input_indexes:
        raise ValueError("OOF merge 至少需要一个 fold prediction index")
    manifest = load_oof_manifest(fold_manifest)
    manifest_binding = oof_manifest_artifact_binding(fold_manifest)
    input_bindings: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    seen_inputs: set[Path] = set()
    for path_ref in input_indexes:
        path = resolve_project_path(path_ref) or Path(path_ref)
        if not path.is_file():
            raise FileNotFoundError(f"OOF prediction index 不存在: {path}")
        resolved = path.resolve(strict=False)
        if resolved in seen_inputs:
            raise ValueError(f"OOF merge 重复输入 index: {path}")
        seen_inputs.add(resolved)
        input_rows = read_prediction_jsonl(path)
        if not input_rows:
            raise ValueError(f"OOF prediction index 不能为空: {path}")
        _revalidate_prediction_publication_directory(
            path, input_rows, split="train"
        )
        rows.extend(input_rows)
        input_bindings.append({
            "path": to_project_ref(path),
            "sha256": sha256_file(path),
            "bytes": int(path.stat().st_size),
            "num_records": len(input_rows),
        })
    audit = _audit_oof_prediction_rows(
        rows, manifest=manifest, fold_manifest=fold_manifest
    )
    output_path = resolve_project_path(output) or Path(output)
    protected = {
        "fold-manifest": fold_manifest,
        **{
            f"input-{index}": path
            for index, path in enumerate(input_indexes)
        },
    }
    validate_output_replacement_safety(output_path, protected)
    validate_output_replacement_safety(
        output_path.with_suffix(".report.json"), protected
    )
    atomic_write_prediction_jsonl(output_path, audit["rows"])
    report = {
        "protocol": OOF_MERGE_PROTOCOL,
        "fold_manifest_artifact": manifest_binding,
        "expert_gate_audit": manifest["expert_gate_audit"],
        "num_folds": int(manifest["num_folds"]),
        "num_parents": audit["num_parents"],
        "parents_per_fold": audit["parents_per_fold"],
        "num_unique_masks": audit["num_unique_masks"],
        "mask_bytes": audit["mask_bytes"],
        "mask_inventory_sha256": audit["mask_inventory_sha256"],
        "population_sha256": audit["population_sha256"],
        "fold_checkpoint_audits": audit["fold_checkpoint_audits"],
        "inputs": input_bindings,
        "output": to_project_ref(output_path),
        "output_sha256": sha256_file(output_path),
        "output_bytes": int(output_path.stat().st_size),
    }
    atomic_write_prediction_json(output_path.with_suffix(".report.json"), report)
    return report


def revalidate_oof_merged_index(
    index_path: str | Path,
    *,
    expected_expert_gate_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Replay a published train OOF index before D4/M7 consumes it."""
    path = resolve_project_path(index_path) or Path(index_path)
    report_path = path.with_suffix(".report.json")
    if not path.is_file() or not report_path.is_file():
        raise FileNotFoundError("OOF merged index/report 缺失")
    report = strict_json_loads(report_path.read_text(encoding="utf-8"))
    if report.get("protocol") != OOF_MERGE_PROTOCOL:
        raise ValueError(
            f"OOF merge report protocol 非法: {report.get('protocol')!r}"
        )
    reported_output = resolve_project_path(str(report.get("output") or ""))
    if (
        reported_output is None
        or reported_output.resolve(strict=False) != path.resolve(strict=False)
        or str(report.get("output_sha256") or "") != sha256_file(path)
        or int(report.get("output_bytes", -1)) != int(path.stat().st_size)
    ):
        raise ValueError("OOF merged index path/hash/bytes 已漂移")
    manifest_binding = report.get("fold_manifest_artifact")
    if not isinstance(manifest_binding, dict):
        raise ValueError("OOF merge report 缺少 fold manifest artifact binding")
    manifest_path = resolve_project_path(str(manifest_binding.get("manifest") or ""))
    if manifest_path is None or not manifest_path.is_file():
        raise FileNotFoundError("OOF merge report 绑定的 fold manifest 缺失")
    current_manifest_binding = oof_manifest_artifact_binding(manifest_path)
    if manifest_binding != current_manifest_binding:
        raise ValueError("OOF fold manifest artifact 在 merge 后发生变化")
    manifest = load_oof_manifest(manifest_path)
    if (
        report.get("expert_gate_audit") != manifest["expert_gate_audit"]
        or (
            expected_expert_gate_audit is not None
            and report.get("expert_gate_audit") != expected_expert_gate_audit
        )
    ):
        raise ValueError("OOF merge report 与当前 frozen expert gate 不一致")
    inputs = report.get("inputs")
    if not isinstance(inputs, list) or not inputs:
        raise ValueError("OOF merge report 缺少输入 index bindings")
    for binding in inputs:
        if not isinstance(binding, dict):
            raise ValueError("OOF merge input binding 非法")
        input_path = resolve_project_path(str(binding.get("path") or ""))
        if (
            input_path is None
            or not input_path.is_file()
            or sha256_file(input_path) != str(binding.get("sha256") or "")
            or int(input_path.stat().st_size) != int(binding.get("bytes", -1))
            or len(read_prediction_jsonl(input_path)) != int(binding.get("num_records", -1))
        ):
            raise ValueError("OOF merge 的 fold prediction input 已漂移")
        _revalidate_prediction_publication_directory(
            input_path, read_prediction_jsonl(input_path), split="train"
        )
    audit = _audit_oof_prediction_rows(
        read_prediction_jsonl(path), manifest=manifest, fold_manifest=manifest_path
    )
    expected = {
        "num_folds": int(manifest["num_folds"]),
        "num_parents": audit["num_parents"],
        "parents_per_fold": audit["parents_per_fold"],
        "num_unique_masks": audit["num_unique_masks"],
        "mask_bytes": audit["mask_bytes"],
        "mask_inventory_sha256": audit["mask_inventory_sha256"],
        "population_sha256": audit["population_sha256"],
        "fold_checkpoint_audits": audit["fold_checkpoint_audits"],
    }
    observed = {name: report.get(name) for name in expected}
    if observed != expected:
        raise ValueError("OOF merge report 统计/checkpoint provenance 未通过当前重放")
    return {
        "protocol": OOF_MERGE_PROTOCOL,
        "report": to_project_ref(report_path),
        "report_sha256": sha256_file(report_path),
        "index": to_project_ref(path),
        "index_sha256": sha256_file(path),
        "fold_manifest_artifact": current_manifest_binding,
        "fold_checkpoint_audits": audit["fold_checkpoint_audits"],
        "num_parents": audit["num_parents"],
        "population_sha256": audit["population_sha256"],
        "mask_inventory_sha256": audit["mask_inventory_sha256"],
    }
