#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Description evaluation with GT, fixed prediction and end-to-end region protocols."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Iterable

import torch
from torch.utils.data import DataLoader
import numpy as np

from qpsalm_seg.data import MultiSourceLandslideDataset, qpsalm_collate
from qpsalm_seg.paths import resolve_project_path
from qpsalm_seg.visualize import restore_mask_to_original

from .common import description_amp_dtype, move_description_batch, write_json
from .backbone import transform_region_mask_to_cache
from .checkpoint import validate_description_stage_lineage
from .config import SegDescConfig
from .json_protocol import strict_json_loads
from .d_minus_one import revalidate_saved_d_minus_one_acceptance
from .counterfactuals import (
    COUNTERFACTUAL_MODES,
    counterfactual_backbone,
    counterfactual_region_masks,
    select_backbone_state,
)
from .cycle_localization import (
    CycleLocalizationProvider,
    cycle_region_iou,
    summarize_cycle_localization,
)
from .metrics import (
    DescriptionMetricAccumulator,
    caption_token_f1,
    bootstrap_mean_ci,
    finite_mean,
    structured_disagreement,
    unsupported_claim_counts,
)
from .data import REGION_INPUT_SOURCE_PROTOCOL, bridge_region_metadata
from .model import (
    SegmentationGroundedDescriptionModel,
    alignment_positive_mask,
    multi_positive_alignment_loss,
)
from .output_protocol import parse_description_output
from .target_audit import build_segmentation_instruction_source_binding
from .vision_cache import DescriptionVisionFeatureBank, description_cache_key


DESCRIPTION_EVALUATION_PROTOCOL = (
    "qpsalm_description_evaluation_v16_atomic_artifact_bound"
)
EVALUATION_PUBLICATION_PROTOCOL = (
    "qpsalm_description_evaluation_publication_v1_artifact_bound"
)
EVALUATION_MASK_ARTIFACT_PROTOCOL = (
    "qpsalm_description_evaluation_mask_artifact_v1_binary_npy"
)
EVALUATION_MASK_INVENTORY_PROTOCOL = (
    "qpsalm_description_evaluation_mask_inventory_v1_role_bound"
)
EVALUATION_CHECKPOINT_BINDING_PROTOCOL = (
    "qpsalm_description_evaluation_checkpoint_binding_v5_run_completion_bound"
)
SAME_IMAGE_RETRIEVAL_PROTOCOL = "qpsalm_same_image_region_retrieval_v2_parent_ranked"
END_TO_END_TARGET_PROTOCOL = "qpsalm_end_to_end_region_target_v3_source_bound"
COUNTERFACTUAL_INPUT_AUDIT_PROTOCOL = (
    "qpsalm_counterfactual_input_change_v1_state_fingerprinted"
)
EVALUATION_POPULATION_FIELDS = (
    "sample_id",
    "parent_sample_id",
    "task_family",
    "target_status",
    "source_dataset",
    "visual_image_path",
    "region_pair_id",
    "region_id",
    "region_source",
    "source_region_aliases",
    "region_mask_path",
    "split",
    "evaluation_mode",
    "instruction",
    "target_text",
    "reference_texts",
    "has_unavailable_modality",
    "end_to_end_segmentation_target",
    "region_input_mask_artifact",
    "region_input_source_binding",
)

_EVALUATION_MASK_ROLES = (
    "region_input",
    "cycle_prediction",
    "cycle_target",
    "cycle_source",
    "cycle_valid",
    "end_to_end_source",
)


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("utf-8"))
    digest.update(json.dumps(
        list(tensor.shape), separators=(",", ":"), allow_nan=False
    ).encode())
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _binary_mask_array(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    while array.ndim > 2 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2:
        raise ValueError(f"evaluation mask 必须是二维二值数组: shape={array.shape}")
    if not np.issubdtype(array.dtype, np.bool_):
        if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
            raise ValueError("evaluation mask 必须是有限数值或 bool")
        if not np.logical_or(array == 0, array == 1).all():
            raise ValueError("evaluation mask 含有非二值像素")
    return np.ascontiguousarray(array.astype(np.uint8, copy=False))


def _mask_content_sha256(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(b"uint8")
    digest.update(json.dumps(
        list(array.shape), separators=(",", ":"), allow_nan=False
    ).encode())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _mask_artifact_relative_path(role: str, sample_id: str) -> Path:
    if role not in _EVALUATION_MASK_ROLES:
        raise ValueError(f"未知 evaluation mask role: {role}")
    if not str(sample_id):
        raise ValueError("evaluation mask artifact 缺少 sample_id")
    key = hashlib.sha256(str(sample_id).encode("utf-8")).hexdigest()
    return Path("mask_artifacts") / role / f"{key}.npy"


def write_evaluation_mask_artifact(
    root: str | Path,
    *,
    role: str,
    sample_id: str,
    mask: torch.Tensor | np.ndarray,
) -> dict[str, Any]:
    """Atomically materialize the exact binary mask consumed by evaluation."""
    output = Path(root)
    relative = _mask_artifact_relative_path(role, sample_id)
    path = output / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    array = _binary_mask_array(mask)
    temporary = path.with_suffix(path.suffix + ".part")
    try:
        with temporary.open("wb") as handle:
            np.save(handle, array, allow_pickle=False)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "protocol": EVALUATION_MASK_ARTIFACT_PROTOCOL,
        "role": role,
        "sample_id": str(sample_id),
        "path": relative.as_posix(),
        "file_sha256": _file_sha256(path),
        "bytes": int(path.stat().st_size),
        "shape": list(array.shape),
        "dtype": "uint8",
        "positive_pixels": int(array.sum()),
        "content_sha256": _mask_content_sha256(array),
    }


def revalidate_evaluation_mask_artifact(
    root: str | Path,
    artifact: Any,
    *,
    expected_role: str,
    expected_sample_id: str,
) -> tuple[dict[str, Any], np.ndarray]:
    """Reopen one mask and recompute its complete semantic/file binding."""
    if not isinstance(artifact, dict):
        raise ValueError("evaluation mask artifact 缺失")
    output = Path(root).resolve(strict=False)
    relative = _mask_artifact_relative_path(expected_role, expected_sample_id)
    if (
        artifact.get("protocol") != EVALUATION_MASK_ARTIFACT_PROTOCOL
        or artifact.get("role") != expected_role
        or str(artifact.get("sample_id") or "") != str(expected_sample_id)
        or str(artifact.get("path") or "") != relative.as_posix()
    ):
        raise ValueError("evaluation mask artifact identity/path binding 非法")
    path = (output / relative).resolve(strict=False)
    try:
        path.relative_to(output)
    except ValueError as exc:
        raise ValueError("evaluation mask artifact 逃逸 evaluation 根目录") from exc
    if not path.is_file():
        raise FileNotFoundError(f"evaluation mask artifact 不存在: {path}")
    try:
        loaded = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"evaluation mask artifact 无法读取: {path}") from exc
    array = _binary_mask_array(loaded)
    rebuilt = {
        "protocol": EVALUATION_MASK_ARTIFACT_PROTOCOL,
        "role": expected_role,
        "sample_id": str(expected_sample_id),
        "path": relative.as_posix(),
        "file_sha256": _file_sha256(path),
        "bytes": int(path.stat().st_size),
        "shape": list(array.shape),
        "dtype": "uint8",
        "positive_pixels": int(array.sum()),
        "content_sha256": _mask_content_sha256(array),
    }
    if rebuilt != artifact:
        raise ValueError("evaluation mask artifact 文件或语义绑定已漂移")
    return rebuilt, array


def evaluation_mask_artifact_inventory(
    artifacts: Iterable[dict[str, Any]],
    *,
    materialized: bool = True,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {
        role: [] for role in _EVALUATION_MASK_ROLES
    }
    identities: set[tuple[str, str]] = set()
    for artifact in artifacts:
        role = str(artifact.get("role") or "")
        sample_id = str(artifact.get("sample_id") or "")
        identity = (role, sample_id)
        if role not in grouped or not sample_id or identity in identities:
            raise ValueError("evaluation mask artifact inventory identity 非法或重复")
        identities.add(identity)
        grouped[role].append(dict(artifact))
    roles = {}
    for role in _EVALUATION_MASK_ROLES:
        values = sorted(grouped[role], key=lambda value: str(value["sample_id"]))
        roles[role] = {
            "count": len(values),
            "bindings_sha256": _json_sha256(values),
        }
    return {
        "protocol": EVALUATION_MASK_INVENTORY_PROTOCOL,
        "materialized": bool(materialized),
        "num_artifacts": len(identities),
        "roles": roles,
    }


def _replay_region_input_source(
    root: Path,
    row: dict[str, Any],
    binding: Any,
    cache_bank: DescriptionVisionFeatureBank,
) -> tuple[np.ndarray, dict[str, Any] | None, np.ndarray]:
    if not isinstance(binding, dict) or binding.get("protocol") != REGION_INPUT_SOURCE_PROTOCOL:
        raise ValueError("formal region evaluation 缺少 source-to-cache projection binding")
    sample_id = str(row.get("sample_id") or "")
    parent = str(row.get("parent_sample_id") or "")
    if (
        str(binding.get("sample_id") or "") != sample_id
        or str(binding.get("parent_sample_id") or "") != parent
        or str(binding.get("region_id") or "") != str(row.get("region_id") or "")
        or str(binding.get("region_source") or "") != str(row.get("region_source") or "")
        or not isinstance(binding.get("render_transform"), dict)
    ):
        raise ValueError("region input source binding identity/cache 字段不一致")
    lookup_key = description_cache_key("multisource_parent", parent)
    try:
        cache_record = cache_bank.record("multisource_parent", parent)
    except (KeyError, OSError, RuntimeError, ValueError) as exc:
        raise ValueError(
            f"region input 无法从当前 M3 cache 重放 parent: {parent}"
        ) from exc
    views = cache_record.get("views")
    if not isinstance(views, list) or not views or not isinstance(views[0], dict):
        raise ValueError("region input M3 cache record 缺少 reference view")
    cache_transform = views[0].get("render_transform")
    if (
        str(cache_record.get("lookup_key") or "") != lookup_key
        or str(binding.get("cache_lookup_key") or "") != lookup_key
        or str(binding.get("cache_fingerprint") or "")
        != str(cache_record.get("cache_fingerprint") or "")
        or binding.get("render_transform") != cache_transform
    ):
        raise ValueError("region input source binding 与当前 M3 cache record 不一致")
    render_size = int(cache_bank.manifest.get("render_size") or 0)
    if render_size <= 0:
        raise ValueError("region input M3 cache render_size 非法")
    try:
        valid = torch.stack([
            torch.nn.functional.interpolate(
                view["valid_mask"].to(dtype=torch.float32)[None],
                size=(render_size, render_size),
                mode="nearest",
            )[0]
            for view in views
        ]).amax(0)
    except (KeyError, TypeError, RuntimeError, ValueError) as exc:
        raise ValueError("region input 无法从 M3 views 重放 union valid mask") from exc
    valid_array = _binary_mask_array(valid.bool())
    source = binding.get("source_mask")
    if not isinstance(source, dict):
        raise ValueError("region input source binding 缺少 source_mask")
    kind = str(source.get("kind") or "")
    source_artifact = None
    if kind == "binary_npy":
        if str(source.get("path") or "") != str(row.get("region_mask_path") or ""):
            raise ValueError("region input source path 与 evaluation row 不一致")
        path = resolve_project_path(str(source.get("path") or ""))
        if path is None or not path.is_file():
            raise FileNotFoundError("region input bound source mask 不存在")
        try:
            array = _binary_mask_array(np.load(path, allow_pickle=False))
        except (OSError, ValueError) as exc:
            raise ValueError("region input bound source mask 不是合法 binary NPY") from exc
        expected = {
            "kind": "binary_npy",
            "path": str(source["path"]),
            "file_sha256": _file_sha256(path),
            "bytes": int(path.stat().st_size),
            "shape": list(array.shape),
            "positive_pixels": int(array.sum()),
        }
        if source != expected:
            raise ValueError("region input bound source mask 文件/语义已漂移")
    elif kind == "null":
        try:
            shape = tuple(int(value) for value in source.get("shape", []))
        except (TypeError, ValueError) as exc:
            raise ValueError("null region input source shape 非法") from exc
        if (
            len(shape) != 2
            or min(shape) <= 0
            or source != {
                "kind": "null",
                "path": None,
                "file_sha256": None,
                "bytes": 0,
                "shape": list(shape),
                "positive_pixels": 0,
            }
            or row.get("region_mask_path") is not None
        ):
            raise ValueError("null region input source binding 非法")
        array = np.zeros(shape, dtype=np.uint8)
    elif kind == "evaluation_artifact":
        if str(row.get("evaluation_mode") or "") != "end_to_end":
            raise ValueError("evaluation_artifact source 只允许 end-to-end mode")
        source_artifact, array = revalidate_evaluation_mask_artifact(
            root,
            source.get("artifact"),
            expected_role="end_to_end_source",
            expected_sample_id=sample_id,
        )
        expected = {
            "kind": "evaluation_artifact",
            "artifact": source_artifact,
            "shape": list(array.shape),
            "positive_pixels": int(array.sum()),
        }
        if source != expected:
            raise ValueError("end-to-end source mask artifact binding 已漂移")
    else:
        raise ValueError(f"未知 region input source kind: {kind!r}")
    projected = transform_region_mask_to_cache(
        torch.from_numpy(array.astype(np.float32, copy=False))[None],
        dict(binding["render_transform"]),
    )
    return _binary_mask_array(projected), source_artifact, valid_array


def revalidate_evaluation_mask_artifacts(
    root: str | Path,
    generation_rows: Iterable[dict[str, Any]],
    report: dict[str, Any],
) -> dict[str, Any]:
    """Reopen every formal region/cycle mask and verify the report inventory."""
    output = Path(root).resolve(strict=False)
    cache_bank: DescriptionVisionFeatureBank | None = None
    if str(report.get("stage") or "") in {"bridge_expert", "predicted_mask"}:
        architecture = dict(
            (report.get("checkpoint_metadata") or {}).get(
                "description_architecture_spec"
            ) or {}
        )
        cache_binding = architecture.get("description_cache_artifact_binding")
        if not isinstance(cache_binding, dict):
            raise ValueError("formal region evaluation 缺少 M3 cache artifact binding")
        cache_ref = cache_binding.get("cache_dir")
        cache_root = resolve_project_path(str(cache_ref or ""))
        if cache_root is None:
            raise ValueError("formal region evaluation M3 cache_dir 非法")
        try:
            cache_bank = DescriptionVisionFeatureBank(cache_root, max_open_shards=1)
        except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
            raise ValueError("formal region evaluation 无法打开当前 M3 cache") from exc
        if cache_bank.artifact_binding() != cache_binding:
            raise ValueError("formal region evaluation M3 cache artifact 已漂移")
    artifacts: list[dict[str, Any]] = []
    region_arrays: dict[str, np.ndarray] = {}
    cache_valid_arrays: dict[str, np.ndarray] = {}
    for row in generation_rows:
        sample_id = str(row.get("sample_id") or "")
        binding, array = revalidate_evaluation_mask_artifact(
            output,
            row.get("region_input_mask_artifact"),
            expected_role="region_input",
            expected_sample_id=sample_id,
        )
        area = float(array.mean())
        if not np.isclose(
            float(row.get("region_area_fraction", -1.0)),
            area,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError("evaluation region_area_fraction 无法从实际 mask 重算")
        artifacts.append(binding)
        region_arrays[sample_id] = array
        if str(report.get("stage") or "") in {"bridge_expert", "predicted_mask"}:
            assert cache_bank is not None
            projected, source_artifact, cache_valid = _replay_region_input_source(
                output,
                row,
                row.get("region_input_source_binding"),
                cache_bank,
            )
            if not np.array_equal(projected, array):
                raise ValueError(
                    "evaluation region mask 与 bound source/cache transform 重放不一致"
                )
            if source_artifact is not None:
                artifacts.append(source_artifact)
            cache_valid_arrays[sample_id] = cache_valid

    if report.get("cycle_localization") is not None:
        cycle_path = output / "cycle_localization.jsonl"
        if not cycle_path.is_file():
            raise FileNotFoundError("evaluation cycle report 存在但 mask audit JSONL 缺失")
        for line in cycle_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = strict_json_loads(line)
            sample_id = str(row.get("sample_id") or "")
            loaded: dict[str, np.ndarray] = {}
            for role, field in (
                ("cycle_prediction", "prediction_mask_artifact"),
                ("cycle_target", "target_mask_artifact"),
                ("cycle_source", "source_mask_artifact"),
                ("cycle_valid", "valid_mask_artifact"),
            ):
                binding, mask_array = revalidate_evaluation_mask_artifact(
                    output,
                    row.get(field),
                    expected_role=role,
                    expected_sample_id=sample_id,
                )
                artifacts.append(binding)
                loaded[role] = mask_array
            cycle_audit = row.get("cycle_audit")
            if not isinstance(cycle_audit, dict) or not isinstance(
                cycle_audit.get("description_render_transform"), dict
            ):
                raise ValueError("cycle mask artifacts 缺少 description render transform")
            projected = _binary_mask_array(transform_region_mask_to_cache(
                torch.from_numpy(
                    loaded["cycle_source"].astype(np.float32, copy=False)
                )[None],
                dict(cycle_audit["description_render_transform"]),
            ))
            valid = loaded["cycle_valid"].astype(bool, copy=False)
            region = region_arrays.get(sample_id)
            cache_valid = cache_valid_arrays.get(sample_id)
            if (
                region is None
                or cache_valid is None
                or projected.shape != valid.shape
                or region.shape != valid.shape
                or not np.array_equal(
                    valid,
                    cache_valid.astype(bool, copy=False),
                )
                or not np.array_equal(
                    np.logical_and(projected.astype(bool, copy=False), valid),
                    loaded["cycle_prediction"].astype(bool, copy=False),
                )
                or not np.array_equal(
                    np.logical_and(region.astype(bool, copy=False), valid),
                    loaded["cycle_target"].astype(bool, copy=False),
                )
            ):
                raise ValueError(
                    "cycle effective masks 无法从 M3 valid/source projection/region 重放"
                )

    rebuilt = evaluation_mask_artifact_inventory(artifacts)
    if report.get("evaluation_mask_artifacts") != rebuilt:
        raise ValueError("evaluation mask artifact inventory 无法由逐条文件重算")
    expected_files = {
        str((output / str(value["path"])).resolve(strict=False))
        for value in artifacts
    }
    artifact_root = output / "mask_artifacts"
    observed_files = {
        str(path.resolve(strict=False))
        for path in artifact_root.rglob("*")
        if path.is_file()
    } if artifact_root.is_dir() else set()
    if observed_files != expected_files:
        raise ValueError(
            "evaluation mask artifact 目录存在未绑定、临时或缺失文件"
        )
    return rebuilt


def _backbone_state_sha256(state) -> str:
    """Fingerprint every task-neutral visual/dense input consumed by MGRR."""

    pyramids = []
    for sample in state.features.samples:
        sample_rows = []
        for pyramid in sample:
            instance_metadata = dict(pyramid.instance.metadata or {})
            content_hash = str(instance_metadata.get("content_hash") or "")
            sample_rows.append({
                "name": str(pyramid.instance.name),
                "family": str(pyramid.instance.family),
                "quality": float(pyramid.instance.quality),
                # Description Cache v1 already binds these source bytes. The
                # projected dense tensors are deterministic under the evaluated
                # checkpoint, so hashing multi-megabyte pyramids for every
                # generation would add I/O without stronger provenance.
                "content_hash": content_hash,
                "source_modalities": list(
                    instance_metadata.get("source_modalities") or []
                ),
                "render_transform": dict(
                    instance_metadata.get("render_transform") or {}
                ),
                "fallback_feature_sha256": (
                    None if content_hash else {
                        name: _tensor_sha256(getattr(pyramid, name))
                        for name in ("high", "detail", "mid", "low")
                    }
                ),
            })
        pyramids.append(sample_rows)
    visual = state.visual_evidence
    visual_payload = None
    if visual is not None:
        visual_payload = {
            "tokens": _tensor_sha256(visual.tokens),
            "token_mask": _tensor_sha256(visual.token_mask),
            "family_ids": _tensor_sha256(visual.family_ids),
            "token_counts": list(visual.token_counts),
            "view_segments": visual.view_segments,
            "cache_keys": list(visual.cache_keys),
            "cache_format": str(visual.cache_format),
        }
    return _json_sha256({
        "pyramids": pyramids,
        "valid_mask": _tensor_sha256(state.valid_mask),
        "active_subsets": [
            {
                "active_names": list(value.active_names),
                "dropped_names": list(value.dropped_names),
                "signature": str(value.signature),
                "is_full": bool(value.is_full),
            }
            for value in state.active_subsets
        ],
        "source_metadata": [
            {
                "component": row.get("component"),
                "parent_sample_id": row.get("parent_sample_id"),
                "cache_key": row.get("cache_key"),
                "counterfactual_modality_swap": row.get(
                    "counterfactual_modality_swap"
                ),
            }
            for row in state.metadata
        ],
        "reference_hw": list(state.reference_hw),
        "use_full_evidence": bool(state.use_full_evidence),
        "visual_evidence": visual_payload,
    })


def counterfactual_input_change_audit(
    *,
    mode: str,
    baseline_state,
    counterfactual_state,
    baseline_mask: torch.Tensor,
    counterfactual_mask: torch.Tensor,
) -> dict[str, Any]:
    baseline_mask_sha = _tensor_sha256(baseline_mask)
    changed_mask_sha = _tensor_sha256(counterfactual_mask)
    baseline_state_sha = _backbone_state_sha256(baseline_state)
    changed_state_sha = _backbone_state_sha256(counterfactual_state)
    dimensions = []
    if baseline_mask_sha != changed_mask_sha:
        dimensions.append("region_mask")
    if baseline_state_sha != changed_state_sha:
        dimensions.append("backbone_state")
    return {
        "protocol": COUNTERFACTUAL_INPUT_AUDIT_PROTOCOL,
        "mode": str(mode),
        "baseline_region_mask_sha256": baseline_mask_sha,
        "counterfactual_region_mask_sha256": changed_mask_sha,
        "baseline_backbone_state_sha256": baseline_state_sha,
        "counterfactual_backbone_state_sha256": changed_state_sha,
        "changed_dimensions": dimensions,
        "changed": bool(dimensions),
    }


def evaluation_population_sha256(rows: list[dict[str, Any]]) -> str:
    """Hash the exact generated sample/target/region population, not model text."""
    identities = [
        {key: row.get(key) for key in EVALUATION_POPULATION_FIELDS}
        for row in rows
    ]
    sample_ids = [str(value.get("sample_id") or "") for value in identities]
    if any(not value for value in sample_ids):
        raise ValueError("description evaluation population 存在空 sample_id")
    if len(sample_ids) != len(set(sample_ids)):
        duplicates = sorted(
            value for value, count in Counter(sample_ids).items() if count > 1
        )
        raise ValueError(f"description evaluation population 存在重复 sample_id: {duplicates[:8]}")
    return _json_sha256(sorted(identities, key=lambda value: str(value["sample_id"])))


def _publication_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"description evaluation publication 缺少 {label}: {path}")
    try:
        rows = [
            strict_json_loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"description evaluation publication 的 {label} 不是严格 JSONL"
        ) from exc
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(
            f"description evaluation publication 的 {label} 每行必须是 object"
        )
    return rows


def _publication_count(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"description evaluation publication {label} 必须是非负整数")
    return int(value)


def _publication_file_binding(
    root: Path,
    relative_path: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    path = (root / relative_path).resolve(strict=False)
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("description evaluation publication artifact 逃逸输出目录") from exc
    return {
        "path": relative_path,
        "sha256": _file_sha256(path),
        "bytes": int(path.stat().st_size),
        "records": len(rows),
    }


def build_evaluation_publication_audit(
    output_dir: str | Path,
    report: dict[str, Any],
) -> dict[str, Any]:
    """Reopen every standalone-evaluation artifact before publishing the report.

    The report hash deliberately excludes ``publication_audit`` so the audit can
    live inside the atomically written final report without a self-hash cycle.
    """
    root = (resolve_project_path(output_dir) or Path(output_dir)).resolve(
        strict=False
    )
    if report.get("protocol") != DESCRIPTION_EVALUATION_PROTOCOL:
        raise ValueError("description evaluation publication protocol 不兼容")
    payload = {
        key: value for key, value in report.items()
        if key != "publication_audit"
    }
    raw_path = root / "raw_generations.jsonl"
    counterfactual_path = root / "counterfactual_generations.jsonl"
    raw_rows = _publication_jsonl(raw_path, label="raw_generations")
    counterfactual_rows = _publication_jsonl(
        counterfactual_path, label="counterfactual_generations"
    )

    num_samples = _publication_count(report.get("num_samples"), label="num_samples")
    num_generated = _publication_count(
        report.get("num_generated"), label="num_generated"
    )
    coverage = report.get("generation_coverage")
    if not isinstance(coverage, dict):
        raise ValueError("description evaluation publication 缺少 generation_coverage")
    eligible = _publication_count(
        coverage.get("eligible_samples"), label="eligible_samples"
    )
    generated = _publication_count(
        coverage.get("generated_samples"), label="generated_samples"
    )
    sample_ids = [str(row.get("sample_id") or "") for row in raw_rows]
    if any(not value for value in sample_ids) or len(sample_ids) != len(set(sample_ids)):
        raise ValueError(
            "description evaluation publication raw sample_id 必须非空且唯一"
        )
    expected_fraction = len(raw_rows) / max(num_samples, 1)
    fraction = coverage.get("fraction")
    if (
        isinstance(fraction, bool)
        or not isinstance(fraction, (int, float))
        or not np.isfinite(float(fraction))
        or not np.isclose(
            float(fraction), expected_fraction, rtol=0.0, atol=1.0e-12
        )
        or num_samples != eligible
        or num_generated != len(raw_rows)
        or generated != len(raw_rows)
        or coverage.get("complete") is not (len(raw_rows) == num_samples)
        or coverage.get("population_sha256")
        != evaluation_population_sha256(raw_rows)
        or coverage.get("population_identity_fields")
        != list(EVALUATION_POPULATION_FIELDS)
    ):
        raise ValueError(
            "description evaluation publication generation population/count 绑定不一致"
        )

    raw_by_sample = {str(row["sample_id"]): row for row in raw_rows}
    counterfactual_keys: set[tuple[str, str]] = set()
    observed_counterfactual_counts: Counter[str] = Counter()
    for row in counterfactual_rows:
        sample_id = str(row.get("sample_id") or "")
        mode = str(row.get("mode") or "")
        key = (sample_id, mode)
        if (
            sample_id not in raw_by_sample
            or not mode
            or key in counterfactual_keys
            or (
                row.get("parent_sample_id") is not None
                and str(row.get("parent_sample_id"))
                != str(raw_by_sample[sample_id].get("parent_sample_id"))
            )
        ):
            raise ValueError(
                "description evaluation publication counterfactual identity 非法"
            )
        counterfactual_keys.add(key)
        observed_counterfactual_counts[mode] += 1
    sensitivity = report.get("counterfactual_sensitivity") or {}
    if not isinstance(sensitivity, dict):
        raise ValueError(
            "description evaluation publication counterfactual summary 必须是 object"
        )
    expected_counterfactual_counts: dict[str, int] = {}
    for mode, summary in sensitivity.items():
        if not isinstance(summary, dict):
            raise ValueError(
                "description evaluation publication counterfactual mode summary 非法"
            )
        expected_counterfactual_counts[str(mode)] = _publication_count(
            summary.get("n"), label=f"counterfactual_sensitivity.{mode}.n"
        )
    if (
        set(observed_counterfactual_counts) - set(expected_counterfactual_counts)
        or any(
            observed_counterfactual_counts.get(mode, 0) != count
            for mode, count in expected_counterfactual_counts.items()
        )
    ):
        raise ValueError(
            "description evaluation publication counterfactual 行数与报告不一致"
        )

    artifacts = {
        "raw_generations": _publication_file_binding(
            root, "raw_generations.jsonl", raw_rows
        ),
        "counterfactual_generations": _publication_file_binding(
            root, "counterfactual_generations.jsonl", counterfactual_rows
        ),
    }
    optional_specs = (
        (
            "end_to_end_target_audit",
            "end_to_end_target_audit.jsonl",
            report.get("end_to_end_coverage") is not None,
        ),
        (
            "cycle_localization",
            "cycle_localization.jsonl",
            report.get("cycle_localization") is not None,
        ),
    )
    for name, relative_path, required in optional_specs:
        path = root / relative_path
        if required:
            rows = _publication_jsonl(path, label=name)
            ids = [str(row.get("sample_id") or row.get("bridge_sample_id") or "") for row in rows]
            if any(not value for value in ids) or len(ids) != len(set(ids)):
                raise ValueError(
                    f"description evaluation publication {name} sample identity 非法"
                )
            if set(ids) - set(raw_by_sample):
                raise ValueError(
                    f"description evaluation publication {name} 超出 generation population"
                )
            if name == "end_to_end_target_audit" and len(rows) != len(raw_rows):
                raise ValueError(
                    "description evaluation publication end-to-end audit 未覆盖 generation"
                )
            if name == "cycle_localization" and len(rows) != _publication_count(
                (report.get("cycle_localization") or {}).get("evaluated_samples"),
                label="cycle_localization.evaluated_samples",
            ):
                raise ValueError(
                    "description evaluation publication cycle 行数与报告不一致"
                )
            artifacts[name] = _publication_file_binding(root, relative_path, rows)
        elif path.exists():
            raise ValueError(
                f"description evaluation publication 存在未声明的 {relative_path}"
            )

    checkpoint_ref = report.get("checkpoint")
    checkpoint = resolve_project_path(str(checkpoint_ref or ""))
    if checkpoint is None or not checkpoint.is_file():
        raise FileNotFoundError(
            "description evaluation publication checkpoint 不存在"
        )
    checkpoint_sha256 = str(report.get("checkpoint_sha256") or "")
    checkpoint_step = report.get("checkpoint_step")
    if (
        checkpoint_sha256 != _file_sha256(checkpoint)
        or isinstance(checkpoint_step, bool)
        or not isinstance(checkpoint_step, int)
        or checkpoint_step < 0
        or not isinstance(report.get("checkpoint_metadata"), dict)
        or not isinstance(report.get("checkpoint_binding"), dict)
        or (report.get("checkpoint_binding") or {}).get("protocol")
        != EVALUATION_CHECKPOINT_BINDING_PROTOCOL
    ):
        raise ValueError(
            "description evaluation publication checkpoint/metadata 绑定不一致"
        )
    if (root / "failure_report.json").exists():
        raise ValueError(
            "description evaluation publication 目录同时存在 failure_report"
        )
    temporary_files = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.suffix in {".tmp", ".part"}
    )
    if temporary_files:
        raise ValueError(
            "description evaluation publication 残留临时文件: "
            f"{temporary_files[:8]}"
        )
    return {
        "protocol": EVALUATION_PUBLICATION_PROTOCOL,
        "terminal_status": "published",
        "report_payload_sha256": _json_sha256(payload),
        "checkpoint": str(checkpoint_ref),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_step": int(checkpoint_step),
        "population_sha256": coverage["population_sha256"],
        "num_samples": num_samples,
        "num_generated": num_generated,
        "artifacts": artifacts,
    }


def revalidate_evaluation_publication(
    output_dir: str | Path,
    report: dict[str, Any],
) -> dict[str, Any]:
    """Rebuild and compare the terminal publication audit from disk."""
    observed = report.get("publication_audit")
    if not isinstance(observed, dict):
        raise ValueError(
            "formal description evaluation 缺少 terminal publication audit"
        )
    rebuilt = build_evaluation_publication_audit(output_dir, report)
    if observed != rebuilt:
        raise ValueError(
            "formal description evaluation publication artifact/report 已漂移"
        )
    return rebuilt


def validate_evaluation_checkpoint_binding(
    config: SegDescConfig,
    checkpoint_report: dict[str, Any],
    runtime_segmentation_migration: dict[str, Any],
    predicted_index_audit: dict[str, Any] | None,
    *,
    checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    """Bind evaluation data mode to the intended trained stage and segmenter."""
    metadata = dict(checkpoint_report.get("metadata") or {})
    checkpoint_stage = str(metadata.get("stage") or "")
    saved_config = dict(metadata.get("config") or {})
    saved_seed = saved_config.get("seed")
    if saved_seed is None or int(saved_seed) != int(config.seed):
        raise RuntimeError(
            "description evaluation seed 与 checkpoint 训练 seed 不一致: "
            f"evaluation={int(config.seed)} checkpoint={saved_seed!r}"
        )
    expected_checkpoint_stage = (
        "predicted_mask" if config.evaluation_mode == "end_to_end" else config.stage
    )
    if checkpoint_stage != expected_checkpoint_stage:
        raise RuntimeError(
            "description evaluation checkpoint stage 非法: "
            f"mode={config.evaluation_mode} data_stage={config.stage} "
            f"expected={expected_checkpoint_stage} observed={checkpoint_stage}"
        )
    expected_checkpoint_role = (
        "terminal_last"
        if checkpoint_stage in {"overfit", "bridge_auto"}
        else "validation_best"
    )
    observed_checkpoint_role = metadata.get("checkpoint_role")
    if observed_checkpoint_role != expected_checkpoint_role:
        raise RuntimeError(
            "description evaluation checkpoint role 非法: "
            f"stage={checkpoint_stage!r} expected={expected_checkpoint_role!r} "
            f"observed={observed_checkpoint_role!r}"
        )
    run_completion = None
    if checkpoint is not None:
        from .run_artifacts import (
            DESCRIPTION_TRAINING_COMPLETION_PROTOCOL,
            validate_checkpoint_run_completion,
        )
        try:
            run_completion = validate_checkpoint_run_completion(
                checkpoint,
                expected_completion_protocol=(
                    DESCRIPTION_TRAINING_COMPLETION_PROTOCOL
                ),
                expected_stage=checkpoint_stage,
                expected_role=expected_checkpoint_role,
            )
        except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
            raise RuntimeError(
                "description evaluation checkpoint 所属训练 run 未成功完成"
            ) from exc
    d_minus_one_acceptance = None
    stage_lineage = None
    if checkpoint_stage != "overfit":
        d_minus_one_acceptance = revalidate_saved_d_minus_one_acceptance(
            metadata.get("d_minus_one_acceptance"),
            expected_description_benchmark=config.description_benchmark,
        )
        if checkpoint_stage != "mmrs_caption":
            stage_lineage = validate_description_stage_lineage(
                metadata.get("stage_lineage"),
                expected_target_stage=checkpoint_stage,
            )
    saved_migration = dict(checkpoint_report.get("segmentation_migration") or {})
    saved_source_sha = str(saved_migration.get("source_sha256") or "")
    runtime_source_sha = str(runtime_segmentation_migration.get("source_sha256") or "")
    if not saved_source_sha or saved_source_sha != runtime_source_sha:
        raise RuntimeError(
            "description evaluation 当前 segmentation source 与 checkpoint lineage 不一致"
        )
    fixed_prediction_match: bool | None = None
    if config.evaluation_mode == "fixed_prediction":
        audit = dict(predicted_index_audit or {})
        predicted_source_sha = str(
            audit.get("segmentation_checkpoint_sha256") or ""
        )
        fixed_prediction_match = bool(
            predicted_source_sha and predicted_source_sha == saved_source_sha
        )
        if not fixed_prediction_match:
            raise RuntimeError(
                "fixed prediction masks 必须由 description checkpoint 绑定的同一 segmentation "
                "checkpoint 生成"
            )
    return {
        "protocol": EVALUATION_CHECKPOINT_BINDING_PROTOCOL,
        "evaluation_mode": config.evaluation_mode,
        "evaluation_data_stage": config.stage,
        "checkpoint_stage": checkpoint_stage,
        "checkpoint_role": observed_checkpoint_role,
        "expected_checkpoint_role": expected_checkpoint_role,
        "expected_checkpoint_stage": expected_checkpoint_stage,
        "run_completion": run_completion,
        "saved_segmentation_migration": saved_migration,
        "runtime_segmentation_migration": dict(runtime_segmentation_migration),
        "segmentation_source_sha256_match": True,
        "fixed_prediction_segmentation_source_match": fixed_prediction_match,
        "checkpoint_training_seed": int(saved_seed),
        "evaluation_seed": int(config.seed),
        "seed_match": True,
        "d_minus_one_acceptance": d_minus_one_acceptance,
        "stage_lineage": stage_lineage,
    }


def _counterfactual_parent_values(
    rows: list[dict[str, Any]], mode: str, field: str,
) -> list[float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if str(row.get("mode") or "") != mode:
            continue
        grouped[str(row["parent_sample_id"])].append(float(row[field]))
    return [sum(values) / len(values) for _parent, values in sorted(grouped.items())]


class EndToEndTargetResolver:
    """Map one Bridge region to the exact segmentation instruction that names it."""

    PROTOCOL = END_TO_END_TARGET_PROTOCOL
    GLOBAL_FAMILY_PRIORITY = {
        "global_landslide_segmentation": 0,
        "negative_aware_segmentation": 1,
        "multisource_evidence_segmentation": 2,
    }

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        ranked_global: dict[str, tuple[int, int]] = {}
        self.referring: dict[tuple[str, str], int] = {}
        for index, row in enumerate(rows):
            parent = str(row.get("parent_sample_id") or row.get("sample_id"))
            family = str(row.get("task_family") or "")
            if family in self.GLOBAL_FAMILY_PRIORITY:
                priority = self.GLOBAL_FAMILY_PRIORITY[family]
                if parent not in ranked_global or priority < ranked_global[parent][0]:
                    ranked_global[parent] = (priority, index)
            target_id = row.get("parent_referring_target_sample_id")
            if target_id:
                key = (parent, str(target_id))
                previous = self.referring.setdefault(key, index)
                if previous != index:
                    previous_row = rows[previous]
                    if str(previous_row.get("sample_id")) != str(row.get("sample_id")):
                        raise ValueError(f"重复 referring instruction identity: {key}")
        self.global_indices = {
            parent: index for parent, (_priority, index) in ranked_global.items()
        }

    @staticmethod
    def _empty_target(row: dict[str, Any]) -> bool:
        mask = row.get("mask") or {}
        if bool(mask.get("empty_mask")):
            return True
        positive = mask.get("positive_pixels")
        return positive is not None and int(positive) == 0

    @staticmethod
    def _aliases(metadata: dict[str, Any]) -> list[dict[str, Any]]:
        return sorted(
            (
                dict(value) for value in (metadata.get("source_region_aliases") or [])
                if isinstance(value, dict) and value.get("sample_id")
            ),
            key=lambda value: str(value["sample_id"]),
        )

    def _global(self, parent: str) -> tuple[int, str, str | None]:
        index = self.global_indices.get(parent)
        if index is None:
            raise KeyError(f"segmentation split 缺少 global instruction: parent={parent}")
        return index, "global_instruction", None

    def _referring(
        self,
        parent: str,
        aliases: list[dict[str, Any]],
        *,
        expected_family: str,
    ) -> tuple[int, str, str | None]:
        for alias in aliases:
            target_id = str(alias["sample_id"])
            index = self.referring.get((parent, target_id))
            if index is None:
                continue
            family = str(self.rows[index].get("task_family") or "")
            if family == expected_family:
                return index, "referring_alias", target_id
        raise KeyError(
            "segmentation split 缺少精确 referring instruction: "
            f"parent={parent} family={expected_family} "
            f"aliases={[value['sample_id'] for value in aliases[:8]]}"
        )

    def resolve(self, metadata: dict[str, Any]) -> dict[str, Any]:
        parent = str(metadata.get("parent_sample_id") or "")
        source = str(metadata.get("region_source") or "unknown")
        aliases = self._aliases(metadata)
        alias_id: str | None
        if source == "gt_global_mask":
            index, kind, alias_id = self._global(parent)
        elif source in {"gt_referring_mask", "pseudo_instance_component"}:
            if not aliases:
                raise KeyError(
                    f"{source} 没有可识别的 referring alias: "
                    f"parent={parent} region={metadata.get('region_id')}"
                )
            index, kind, alias_id = self._referring(
                parent, aliases, expected_family="referring_landslide_segmentation"
            )
        elif source == "no_target":
            if aliases:
                index, kind, alias_id = self._referring(
                    parent, aliases, expected_family="no_target_segmentation"
                )
            else:
                index, kind, alias_id = self._global(parent)
                if not self._empty_target(self.rows[index]):
                    raise KeyError(
                        "no_target region 既无 no-target alias，parent global target 也非空: "
                        f"parent={parent}"
                    )
                kind = "empty_global_instruction"
        else:
            raise KeyError(
                f"region_source={source!r} 没有端到端 segmentation target protocol"
            )
        row = self.rows[index]
        return {
            "protocol": self.PROTOCOL,
            "bridge_sample_id": str(metadata.get("sample_id") or ""),
            "dataset_index": int(index),
            "mapping_kind": kind,
            "alias_sample_id": alias_id,
            "segmentation_sample_id": str(row.get("sample_id")),
            "segmentation_task_family": str(row.get("task_family")),
            "parent_sample_id": parent,
            "bridge_region_id": str(metadata.get("region_id") or "unknown"),
            "bridge_region_source": source,
        }


class EndToEndMaskProvider:
    """Run frozen segmentation only for an exactly resolved Bridge target."""

    def __init__(
        self,
        model: SegmentationGroundedDescriptionModel,
        split: str,
        threshold: float,
    ) -> None:
        config = replace(
            model.segmentation.config,
            modality_dropout=0.0,
            train_hflip_prob=0.0,
            train_vflip_prob=0.0,
        )
        self.dataset = MultiSourceLandslideDataset(config, split)
        self.segmentation_source_binding = (
            build_segmentation_instruction_source_binding(
                config, split, self.dataset.rows
            )
        )
        self.resolver = EndToEndTargetResolver(self.dataset.rows)
        self.model = model
        self.threshold = float(threshold)
        self.cache: dict[str, tuple[torch.Tensor, dict[str, Any]]] = {}
        self.mapping_counts: Counter[str] = Counter()

    def require_targets(self, metadata_rows: Iterable[dict[str, Any]]) -> None:
        errors = []
        for metadata in metadata_rows:
            try:
                self.resolver.resolve(metadata)
            except KeyError as exc:
                errors.append(str(exc))
        if errors:
            raise KeyError(
                "end-to-end segmentation target 映射不完整: "
                f"count={len(errors)} examples={errors[:8]}"
            )

    @torch.no_grad()
    def predict(
        self,
        metadata: dict[str, Any],
        output_hw: tuple[int, int],
        *,
        return_source: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any]] | tuple[
        torch.Tensor, dict[str, Any], torch.Tensor
    ]:
        audit = self.resolver.resolve(metadata)
        cache_key = str(audit["segmentation_sample_id"])
        cached = self.cache.get(cache_key)
        if cached is None:
            item = self.dataset[int(audit["dataset_index"])]
            batch = qpsalm_collate([item])
            with self.model.controller.adapter_scope("default"):
                output = self.model.segmentation(batch)
            canvas = (
                torch.sigmoid(output.final_mask_logits[0, 0].float()).detach().cpu().numpy()
                >= self.threshold
            ).astype(np.uint8)
            segmentation_transform = dict(item["metadata"].get("resize_transform") or {})
            restored = restore_mask_to_original(canvas, segmentation_transform)
            if restored is None:
                raise ValueError(
                    "end-to-end mask 无法按 segmentation resize transform 恢复: "
                    f"sample={cache_key}"
                )
            original_mask = torch.from_numpy(restored.astype(np.float32))[None]
            cached = (original_mask, segmentation_transform)
            self.cache[cache_key] = cached
        original_mask, segmentation_transform = cached
        self.mapping_counts[str(audit["mapping_kind"])] += 1
        device = next(self.model.parameters()).device
        parent = str(audit["parent_sample_id"])
        cache_record = self.model.description_backbone.bank.record(
            "multisource_parent", parent
        )
        description_transform = dict(cache_record["views"][0]["render_transform"])
        description_mask = transform_region_mask_to_cache(
            original_mask, description_transform
        ).to(device=device)
        if tuple(description_mask.shape[-2:]) != tuple(output_hw):
            raise ValueError(
                "end-to-end description mask canvas 不一致: "
                f"parent={parent} observed={tuple(description_mask.shape[-2:])} "
                f"expected={tuple(output_hw)}"
            )
        audit = {
            **audit,
            "mask_threshold": self.threshold,
            "segmentation_resize_transform": segmentation_transform,
            "description_render_transform": description_transform,
            "original_mask_shape": list(original_mask.shape[-2:]),
        }
        if return_source:
            return description_mask.unsqueeze(0), audit, original_mask.clone()
        return description_mask.unsqueeze(0), audit

    def summary(self, dataset: Any) -> dict[str, Any]:
        return {
            "protocol": self.resolver.PROTOCOL,
            "source_bridge_rows": int(getattr(dataset, "end_to_end_source_count", len(dataset))),
            "eligible_bridge_rows_before_limit": int(
                getattr(dataset, "end_to_end_eligible_count", len(dataset))
            ),
            "evaluated_rows": len(dataset),
            "excluded_by_reason": dict(sorted(
                getattr(dataset, "end_to_end_exclusion_counts", {}).items()
            )),
            "mapping_counts": dict(sorted(self.mapping_counts.items())),
            "unique_segmentation_inferences": len(self.cache),
            "mask_threshold": self.threshold,
            "segmentation_source_binding": self.segmentation_source_binding,
        }


def _same_image_retrieval(
    region_embeddings: list[torch.Tensor],
    text_embeddings: list[torch.Tensor],
    parent_ids: list[str],
    phrase_labels: list[str] | None = None,
    sample_ids: list[str] | None = None,
) -> dict[str, Any]:
    if not region_embeddings:
        return {
            "protocol": SAME_IMAGE_RETRIEVAL_PROTOCOL,
            "population_identity_complete": sample_ids is not None,
            "population_sha256": _json_sha256([]) if sample_ids is not None else None,
            "num_queries": 0,
            "num_multi_candidate_queries": 0,
            "region_to_text_r1": None,
            "text_to_region_r1": None,
            "mean_r1": None,
            "region_to_text_r5": None,
            "text_to_region_r5": None,
            "mean_r5": None,
            "normalized_phrase_match": None,
            "modifier_accuracy": None,
            "region_to_text_ranking_margin": None,
            "text_to_region_ranking_margin": None,
            "mean_ranking_margin": None,
            "aggregation_unit": "parent",
            "per_parent": {},
            "per_parent_mean_r1": {},
        }
    region = torch.cat(region_embeddings).float()
    text = torch.cat(text_embeddings).float()
    if region.shape != text.shape or region.shape[0] != len(parent_ids):
        raise ValueError("DIOR retrieval embedding/metadata 数量不一致")
    labels = (
        [" ".join(str(value).casefold().split()) for value in phrase_labels]
        if phrase_labels is not None else [f"pair:{index}" for index in range(len(parent_ids))]
    )
    if len(labels) != len(parent_ids):
        raise ValueError("DIOR retrieval phrase label 数量不一致")
    identity_complete = sample_ids is not None
    resolved_sample_ids = list(sample_ids or [f"row:{index}" for index in range(len(parent_ids))])
    if len(resolved_sample_ids) != len(parent_ids):
        raise ValueError("DIOR retrieval sample identity 数量不一致")
    if identity_complete and len(set(resolved_sample_ids)) != len(resolved_sample_ids):
        raise ValueError("DIOR retrieval sample_id 必须唯一")
    population = sorted(
        [
            {
                "sample_id": str(sample),
                "parent_sample_id": str(parent),
                "normalized_phrase": str(label),
            }
            for sample, parent, label in zip(resolved_sample_ids, parent_ids, labels)
        ],
        key=lambda value: value["sample_id"],
    )
    def modifiers(value: str) -> tuple[str, ...]:
        tokens = [
            token for token in str(value).casefold().split()
            if token not in {"a", "an", "the", "of", "in", "on", "at"}
        ]
        return tuple(tokens[:-1]) if len(tokens) > 1 else ()

    eligible = 0
    ambiguous = 0
    parent_metrics: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for index, parent in enumerate(parent_ids):
        candidates = [value for value, current in enumerate(parent_ids) if current == parent]
        if len(candidates) < 2:
            continue
        candidate_tensor = torch.tensor(candidates, device=region.device)
        text_to_region_scores = text[index] @ region[candidate_tensor].T
        region_to_text_scores = region[index] @ text[candidate_tensor].T
        text_to_region_rank = [
            candidates[int(value)]
            for value in torch.argsort(text_to_region_scores, descending=True).tolist()
        ]
        region_to_text_rank = [
            candidates[int(value)]
            for value in torch.argsort(region_to_text_scores, descending=True).tolist()
        ]
        selected_region = text_to_region_rank[0]
        selected_text = region_to_text_rank[0]
        positives = {value for value in candidates if labels[value] == labels[index]}
        ambiguous += int(len(positives) > 1)
        negatives = [value for value in candidates if value not in positives]
        t2r_correct = float(selected_region in positives)
        r2t_correct = float(selected_text in positives)
        values = parent_metrics[str(parent)]
        values["text_to_region_r1"].append(t2r_correct)
        values["region_to_text_r1"].append(r2t_correct)
        values["text_to_region_r5"].append(
            float(bool(set(text_to_region_rank[:5]) & positives))
        )
        values["region_to_text_r5"].append(
            float(bool(set(region_to_text_rank[:5]) & positives))
        )
        values["normalized_phrase_match"].append(
            float(labels[selected_text] == labels[index])
        )
        values["modifier_accuracy"].append(
            float(modifiers(labels[selected_text]) == modifiers(labels[index]))
        )
        if negatives:
            negative_positions = [candidates.index(value) for value in negatives]
            positive_positions = [candidates.index(value) for value in positives]
            values["text_to_region_margin"].append(float(
                text_to_region_scores[positive_positions].max()
                - text_to_region_scores[negative_positions].max()
            ))
            values["region_to_text_margin"].append(float(
                region_to_text_scores[positive_positions].max()
                - region_to_text_scores[negative_positions].max()
            ))
        eligible += 1
    per_parent = {
        parent: {
            name: sum(values) / len(values)
            for name, values in sorted(metrics.items()) if values
        }
        for parent, metrics in sorted(parent_metrics.items())
    }

    def parent_macro(name: str) -> float | None:
        values = [metrics[name] for metrics in per_parent.values() if name in metrics]
        return sum(values) / len(values) if values else None

    r2t_r1 = parent_macro("region_to_text_r1")
    t2r_r1 = parent_macro("text_to_region_r1")
    return {
        "protocol": SAME_IMAGE_RETRIEVAL_PROTOCOL,
        "population_identity_complete": identity_complete,
        "population_sha256": _json_sha256(population),
        "num_queries": len(parent_ids),
        "num_multi_candidate_queries": eligible,
        "num_ambiguous_phrase_queries": ambiguous,
        "aggregation_unit": "parent",
        "region_to_text_r1": r2t_r1,
        "text_to_region_r1": t2r_r1,
        "mean_r1": (
            (r2t_r1 + t2r_r1) * 0.5
            if r2t_r1 is not None and t2r_r1 is not None else None
        ),
        "region_to_text_r5": parent_macro("region_to_text_r5"),
        "text_to_region_r5": parent_macro("text_to_region_r5"),
        "mean_r5": (
            (
                parent_macro("region_to_text_r5")
                + parent_macro("text_to_region_r5")
            ) * 0.5
            if parent_macro("region_to_text_r5") is not None
            and parent_macro("text_to_region_r5") is not None else None
        ),
        "normalized_phrase_match": parent_macro("normalized_phrase_match"),
        "modifier_accuracy": parent_macro("modifier_accuracy"),
        "region_to_text_ranking_margin": parent_macro("region_to_text_margin"),
        "text_to_region_ranking_margin": parent_macro("text_to_region_margin"),
        "mean_ranking_margin": (
            (
                parent_macro("region_to_text_margin")
                + parent_macro("text_to_region_margin")
            ) * 0.5
            if parent_macro("region_to_text_margin") is not None
            and parent_macro("text_to_region_margin") is not None else None
        ),
        "per_parent": per_parent,
        "per_parent_mean_r1": {
            parent: 0.5 * (
                metrics["region_to_text_r1"] + metrics["text_to_region_r1"]
            )
            for parent, metrics in per_parent.items()
        },
    }


def _counterfactual_modes(config: SegDescConfig) -> tuple[str, ...]:
    values = tuple(config.counterfactual_modes or COUNTERFACTUAL_MODES)
    invalid = sorted(set(values) - set(COUNTERFACTUAL_MODES))
    if invalid:
        raise ValueError(f"未知 counterfactual modes: {invalid}")
    return values


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = "".join(
        json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
        for row in rows
    )
    temporary = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    try:
        temporary.write_text(encoded, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


@torch.no_grad()
def evaluate_description(
    model: SegmentationGroundedDescriptionModel,
    loader: DataLoader,
    config: SegDescConfig,
    device: torch.device,
    *,
    split: str,
    output_dir: str | Path | None = None,
    max_generate_samples: int | None = None,
    run_counterfactuals: bool = True,
    publish_report: bool = True,
) -> dict[str, Any]:
    model.eval()
    resolved_output = (
        resolve_project_path(output_dir) or Path(output_dir)
        if output_dir is not None else None
    )
    if resolved_output is not None:
        resolved_output.mkdir(parents=True, exist_ok=True)
    amp_dtype = description_amp_dtype(config, device)
    autocast = device.type == "cuda" and config.amp_dtype != "fp32"
    metric = DescriptionMetricAccumulator()
    unavailable_metric = DescriptionMetricAccumulator()
    losses: list[float] = []
    generation_rows: list[dict[str, Any]] = []
    counterfactual_rows: list[dict[str, Any]] = []
    end_to_end_rows: list[dict[str, Any]] = []
    cycle_localization_rows: list[dict[str, Any]] = []
    mask_artifacts: list[dict[str, Any]] = []
    counterfactual_values: dict[str, list[float]] = {name: [] for name in _counterfactual_modes(config)}
    counterfactual_score_deltas: dict[str, list[float]] = {
        name: [] for name in _counterfactual_modes(config)
    }
    counterfactual_claim_deltas: dict[str, list[float]] = {
        name: [] for name in _counterfactual_modes(config)
    }
    region_embeddings: list[torch.Tensor] = []
    text_embeddings: list[torch.Tensor] = []
    retrieval_parents: list[str] = []
    retrieval_phrases: list[str] = []
    retrieval_sample_ids: list[str] = []
    generated = 0
    requested_generate = int(
        config.max_generate_samples if max_generate_samples is None else max_generate_samples
    )
    generate_limit = len(loader.dataset) if requested_generate <= 0 else min(
        requested_generate, len(loader.dataset)
    )
    counterfactual_counts = {name: 0 for name in _counterfactual_modes(config)}
    counterfactual_skipped_no_effect = {
        name: 0 for name in _counterfactual_modes(config)
    }
    counterfactual_skipped_unavailable = {
        name: 0 for name in _counterfactual_modes(config)
    }
    e2e = (
        EndToEndMaskProvider(model, split, config.segmentation_mask_threshold)
        if config.evaluation_mode == "end_to_end" else None
    )
    if e2e is not None:
        e2e.require_targets(
            bridge_region_metadata(row)
            for row in getattr(loader.dataset, "rows", [])
        )
    cycle = (
        CycleLocalizationProvider(
            model, split, config.segmentation_mask_threshold
        )
        if int(config.cycle_localization_samples) >= 0 else None
    )
    if cycle is not None:
        cycle.prepare(getattr(loader.dataset, "rows", []))
    cycle_target = (
        0 if cycle is None else (
            cycle.eligible_rows
            if int(config.cycle_localization_samples) == 0
            else min(
                int(config.cycle_localization_samples), cycle.eligible_rows
            )
        )
    )
    started = time.perf_counter()

    for batch_index, cpu_batch in enumerate(loader):
        batch = move_description_batch(cpu_batch, device)
        backbone = model.encode_description_requests(
            batch["requests"],
            include_spatial=config.stage not in {"mmrs_caption", "rsicap_caption"},
        )
        region_masks = batch["region_masks"]
        batch_e2e_audits: list[dict[str, Any] | None] = [None] * len(batch["metadata"])
        batch_e2e_source_masks: list[torch.Tensor | None] = [
            None
        ] * len(batch["metadata"])
        if e2e is not None:
            resolved = [
                e2e.predict(
                    row,
                    tuple(region_masks.shape[-2:]),
                    return_source=True,
                )
                for row in batch["metadata"]
            ]
            predicted = [value[0][0] for value in resolved]
            batch_e2e_audits = [value[1] for value in resolved]
            batch_e2e_source_masks = [value[2] for value in resolved]
            region_masks = torch.stack(predicted).to(device=device)

        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast):
            if config.stage == "dior_alignment":
                regions, texts = model.region_alignment_embeddings(
                    backbone, region_masks, batch["target_texts"]
                )
                temperature = model.alignment_temperature.float().clamp(0.01, 1.0)
                logits = regions @ texts.T / temperature
                positive_mask = alignment_positive_mask(
                    batch["target_texts"],
                    [str(row["parent_sample_id"]) for row in batch["metadata"]],
                    device=logits.device,
                )
                loss = multi_positive_alignment_loss(logits, positive_mask)
                region_embeddings.append(regions.detach().cpu())
                text_embeddings.append(texts.detach().cpu())
                retrieval_parents.extend(str(row["parent_sample_id"]) for row in batch["metadata"])
                retrieval_phrases.extend(str(value) for value in batch["target_texts"])
                retrieval_sample_ids.extend(str(row["sample_id"]) for row in batch["metadata"])
            else:
                output = model.describe_from_state(
                    backbone,
                    region_masks,
                    batch["instructions"],
                    target_texts=batch["target_texts"],
                    region_valid_mask=backbone.valid_mask,
                    protocol=config.region_protocol,
                    structured_output=batch["structured_outputs"],
                )
                if output.per_sample_loss is None:
                    raise RuntimeError("description validation 未产生 per-sample loss")
                loss = (output.per_sample_loss * batch["weights"]).sum() / batch["weights"].sum().clamp_min(1.0)
        losses.append(float(loss.detach().cpu()))

        if config.stage == "dior_alignment" or generate_limit <= generated:
            continue
        baseline_texts: list[str] = []
        batch_generation_count = min(region_masks.shape[0], generate_limit - generated)
        for sample_index in range(batch_generation_count):
            one_state = select_backbone_state(backbone, [sample_index])
            one_mask = region_masks[sample_index:sample_index + 1]
            sample_id = str(batch["metadata"][sample_index]["sample_id"])
            region_mask_artifact = (
                write_evaluation_mask_artifact(
                    resolved_output,
                    role="region_input",
                    sample_id=sample_id,
                    mask=one_mask,
                )
                if resolved_output is not None else None
            )
            if region_mask_artifact is not None:
                mask_artifacts.append(region_mask_artifact)
            region_source_binding = batch["metadata"][sample_index].get(
                "region_input_source_binding"
            )
            end_to_end_audit = batch_e2e_audits[sample_index]
            if end_to_end_audit is not None:
                source_artifact = (
                    write_evaluation_mask_artifact(
                        resolved_output,
                        role="end_to_end_source",
                        sample_id=sample_id,
                        mask=batch_e2e_source_masks[sample_index],
                    )
                    if resolved_output is not None else None
                )
                if source_artifact is not None:
                    mask_artifacts.append(source_artifact)
                original_binding = dict(region_source_binding or {})
                region_source_binding = {
                    **original_binding,
                    "source_mask": {
                        "kind": "evaluation_artifact",
                        "artifact": source_artifact,
                        "shape": list(source_artifact["shape"]),
                        "positive_pixels": int(source_artifact["positive_pixels"]),
                    } if source_artifact is not None else None,
                    "render_transform": dict(
                        end_to_end_audit["description_render_transform"]
                    ),
                }
                end_to_end_audit = {
                    **end_to_end_audit,
                    "region_input_mask_artifact": region_mask_artifact,
                    "region_input_source_binding": region_source_binding,
                }
                end_to_end_rows.append(end_to_end_audit)
            structured = bool(batch["structured_outputs"][sample_index])
            raw = model.generate_from_state(
                one_state,
                one_mask,
                batch["instructions"][sample_index],
                max_new_tokens=config.max_new_tokens,
                protocol=config.region_protocol,
                structured_output=structured,
            )
            baseline_texts.append(raw)
            details = metric.update(
                prediction=raw,
                target_text=batch["target_texts"][sample_index],
                references=batch["reference_texts"][sample_index],
                structured=structured,
                metadata=batch["metadata"][sample_index],
            )
            if bool(batch["metadata"][sample_index].get("has_unavailable_modality")):
                unavailable_metric.update(
                    prediction=raw,
                    target_text=batch["target_texts"][sample_index],
                    references=batch["reference_texts"][sample_index],
                    structured=structured,
                    metadata=batch["metadata"][sample_index],
                )
            generation_rows.append({
                **batch["metadata"][sample_index],
                "end_to_end_segmentation_target": end_to_end_audit,
                "region_input_mask_artifact": region_mask_artifact,
                "region_input_source_binding": region_source_binding,
                "split": split,
                "evaluation_mode": config.evaluation_mode,
                "instruction": batch["instructions"][sample_index],
                "target_text": batch["target_texts"][sample_index],
                "reference_texts": list(batch["reference_texts"][sample_index]),
                "raw_generation": raw,
                "raw_metrics": details,
                "region_area_fraction": float(one_mask.float().mean().cpu()),
            })
            if (
                cycle is not None
                and len(cycle_localization_rows) < cycle_target
                and cycle.eligible(str(batch["metadata"][sample_index]["sample_id"]))
            ):
                if not str(raw).strip():
                    cycle.runtime_skip_counts["empty_raw_generation"] += 1
                else:
                    cycle_mask, cycle_audit, cycle_source_mask = cycle.localize(
                        batch["metadata"][sample_index],
                        raw,
                        tuple(one_mask.shape[-2:]),
                        return_source=True,
                    )
                    cycle_metrics = cycle_region_iou(
                        cycle_mask,
                        one_mask[0],
                        one_state.valid_mask[0],
                    )
                    valid = one_state.valid_mask[0].detach().bool()
                    effective_prediction = cycle_mask.detach().bool() & valid
                    effective_target = one_mask[0].detach().bool() & valid
                    prediction_artifact = (
                        write_evaluation_mask_artifact(
                            resolved_output,
                            role="cycle_prediction",
                            sample_id=sample_id,
                            mask=effective_prediction,
                        )
                        if resolved_output is not None else None
                    )
                    target_artifact = (
                        write_evaluation_mask_artifact(
                            resolved_output,
                            role="cycle_target",
                            sample_id=sample_id,
                            mask=effective_target,
                        )
                        if resolved_output is not None else None
                    )
                    source_artifact = (
                        write_evaluation_mask_artifact(
                            resolved_output,
                            role="cycle_source",
                            sample_id=sample_id,
                            mask=cycle_source_mask,
                        )
                        if resolved_output is not None else None
                    )
                    valid_artifact = (
                        write_evaluation_mask_artifact(
                            resolved_output,
                            role="cycle_valid",
                            sample_id=sample_id,
                            mask=valid,
                        )
                        if resolved_output is not None else None
                    )
                    if all(value is not None for value in (
                        prediction_artifact,
                        target_artifact,
                        source_artifact,
                        valid_artifact,
                    )):
                        mask_artifacts.extend((
                            prediction_artifact,
                            target_artifact,
                            source_artifact,
                            valid_artifact,
                        ))
                    cycle_localization_rows.append({
                        **batch["metadata"][sample_index],
                        "split": split,
                        "evaluation_mode": config.evaluation_mode,
                        "region_protocol": config.region_protocol,
                        "cycle_audit": cycle_audit,
                        "prediction_mask_artifact": prediction_artifact,
                        "target_mask_artifact": target_artifact,
                        "source_mask_artifact": source_artifact,
                        "valid_mask_artifact": valid_artifact,
                        **cycle_metrics,
                    })
            generated += 1

        if not run_counterfactuals or all(
            value >= int(config.counterfactual_samples)
            for value in counterfactual_counts.values()
        ):
            continue
        for mode in _counterfactual_modes(config):
            if counterfactual_counts[mode] >= int(config.counterfactual_samples):
                continue
            counterfactual_inputs: list[dict[str, Any] | None] = [
                None for _ in batch["metadata"]
            ]
            counterfactual_unavailable = [False for _ in batch["metadata"]]
            try:
                if mode in {"region_swap", "cross_parent_region_swap"}:
                    cf_backbone = backbone
                    cf_masks = region_masks.clone()
                    resolver_name = (
                        "same_parent_region_swap"
                        if mode == "region_swap" else "cross_parent_region_swap"
                    )
                    resolver = getattr(loader.dataset, resolver_name, None)
                    if resolver is None:
                        continue
                    for sample_index, metadata in enumerate(batch["metadata"]):
                        resolved = resolver(
                            str(metadata["sample_id"]),
                            region_masks[sample_index],
                        )
                        if resolved is not None:
                            alternate, counterfactual_inputs[sample_index] = resolved
                            cf_masks[sample_index] = alternate.to(
                                device=region_masks.device,
                                dtype=region_masks.dtype,
                            )
                        else:
                            counterfactual_unavailable[sample_index] = True
                elif mode in {"full_mask", "zero_mask", "shuffled_mask"}:
                    cf_backbone = backbone
                    cf_masks = counterfactual_region_masks(region_masks, mode)
                elif mode == "modality_removal":
                    cf_backbone = counterfactual_backbone(backbone, mode)
                    cf_masks = region_masks
                else:
                    # Cross-parent donors are resolved per sample below. This
                    # remains valid when formal evaluation uses batch_size=1.
                    cf_backbone = backbone
                    cf_masks = region_masks
            except ValueError:
                continue
            for sample_index, baseline in enumerate(baseline_texts):
                if counterfactual_counts[mode] >= int(config.counterfactual_samples):
                    break
                one_state = None
                if mode == "cross_parent_modality_swap":
                    donor_resolver = getattr(
                        loader.dataset, "cross_parent_modality_swap_request", None
                    )
                    resolved_donor = (
                        donor_resolver(str(batch["metadata"][sample_index]["sample_id"]))
                        if donor_resolver is not None else None
                    )
                    if resolved_donor is None:
                        effective = False
                        counterfactual_unavailable[sample_index] = True
                    else:
                        donor_request, donor_audit = resolved_donor
                        pair_backbone = model.encode_description_requests([
                            batch["requests"][sample_index], donor_request,
                        ])
                        swapped_pair = counterfactual_backbone(
                            pair_backbone, "cross_parent_modality_swap"
                        )
                        one_state = select_backbone_state(swapped_pair, [0])
                        swap_audit = swapped_pair.metadata[0].get(
                            "counterfactual_modality_swap"
                        )
                        counterfactual_inputs[sample_index] = {
                            **donor_audit,
                            "applied_swap": swap_audit,
                        }
                        effective = swap_audit is not None
                elif mode in {
                    "full_mask", "zero_mask", "shuffled_mask", "region_swap",
                    "cross_parent_region_swap",
                }:
                    effective = not torch.equal(
                        cf_masks[sample_index], region_masks[sample_index]
                    )
                elif mode == "modality_removal":
                    effective = (
                        cf_backbone.active_subsets[sample_index].active_names
                        != backbone.active_subsets[sample_index].active_names
                    )
                else:
                    effective = not cf_backbone.active_subsets[
                        sample_index
                    ].signature.endswith(":none")
                if not effective:
                    if counterfactual_unavailable[sample_index]:
                        counterfactual_skipped_unavailable[mode] += 1
                    else:
                        counterfactual_skipped_no_effect[mode] += 1
                    continue
                if one_state is None:
                    one_state = select_backbone_state(cf_backbone, [sample_index])
                baseline_state = select_backbone_state(backbone, [sample_index])
                input_change_audit = counterfactual_input_change_audit(
                    mode=mode,
                    baseline_state=baseline_state,
                    counterfactual_state=one_state,
                    baseline_mask=region_masks[
                        sample_index:sample_index + 1
                    ],
                    counterfactual_mask=cf_masks[
                        sample_index:sample_index + 1
                    ],
                )
                if input_change_audit["changed"] is not True:
                    counterfactual_skipped_no_effect[mode] += 1
                    continue
                structured = bool(batch["structured_outputs"][sample_index])
                changed = model.generate_from_state(
                    one_state,
                    cf_masks[sample_index:sample_index + 1],
                    batch["instructions"][sample_index],
                    max_new_tokens=config.max_new_tokens,
                    protocol=config.region_protocol,
                    structured_output=structured,
                )
                if structured:
                    baseline_parsed = parse_description_output(baseline).parsed
                    changed_parsed = parse_description_output(changed).parsed
                    target_parsed = parse_description_output(
                        batch["target_texts"][sample_index]
                    ).parsed
                    sensitivity = structured_disagreement(
                        baseline_parsed,
                        changed_parsed,
                    )
                    baseline_score = 1.0 - structured_disagreement(
                        baseline_parsed, target_parsed
                    )
                    changed_score = 1.0 - structured_disagreement(
                        changed_parsed, target_parsed
                    )
                    baseline_claims = unsupported_claim_counts(
                        baseline_parsed, target_parsed
                    )[1]
                    changed_claims = unsupported_claim_counts(
                        changed_parsed, target_parsed
                    )[1]
                else:
                    sensitivity = 1.0 - caption_token_f1(changed, [baseline])
                    references = batch["reference_texts"][sample_index]
                    baseline_score = caption_token_f1(baseline, references)
                    changed_score = caption_token_f1(changed, references)
                    baseline_claims = changed_claims = 0
                score_delta = changed_score - baseline_score
                claim_delta = float(changed_claims - baseline_claims)
                counterfactual_values[mode].append(sensitivity)
                counterfactual_score_deltas[mode].append(score_delta)
                counterfactual_claim_deltas[mode].append(claim_delta)
                counterfactual_rows.append({
                    **batch["metadata"][sample_index],
                    "mode": mode,
                    "counterfactual_input": counterfactual_inputs[sample_index],
                    "input_change_audit": input_change_audit,
                    "baseline_generation": baseline,
                    "counterfactual_generation": changed,
                    "sensitivity": sensitivity,
                    "baseline_target_score": baseline_score,
                    "counterfactual_target_score": changed_score,
                    "target_score_delta": score_delta,
                    "factual_claim_count_delta": claim_delta,
                })
                counterfactual_counts[mode] += 1

    population_sha256 = evaluation_population_sha256(generation_rows)
    report = {
        "protocol": DESCRIPTION_EVALUATION_PROTOCOL,
        "stage": config.stage,
        "split": split,
        "evaluation_mode": config.evaluation_mode,
        "region_protocol": config.region_protocol,
        "expert_gate_audit": getattr(loader.dataset, "expert_gate_audit", None),
        "predicted_index_audit": getattr(loader.dataset, "predicted_index_audit", None),
        "source_filter_audit": getattr(loader.dataset, "source_filter_audit", None),
        "region_source_filter_audit": getattr(
            loader.dataset, "region_source_filter_audit", None
        ),
        "num_samples": len(loader.dataset),
        "num_generated": generated,
        "evaluation_limit_audit": {
            "protocol": "qpsalm_description_evaluation_limit_v1",
            "requested_max_samples": int(config.max_val_samples),
            "full_population_requested": int(config.max_val_samples) == 0,
            "dataset_rows_evaluated": len(loader.dataset),
        },
        "generation_coverage": {
            "requested": requested_generate,
            "eligible_samples": len(loader.dataset),
            "generated_samples": generated,
            "fraction": generated / max(len(loader.dataset), 1),
            "complete": generated == len(loader.dataset),
            "population_sha256": population_sha256,
            "population_identity_fields": list(EVALUATION_POPULATION_FIELDS),
        },
        "evaluation_mask_artifacts": evaluation_mask_artifact_inventory(
            mask_artifacts,
            materialized=resolved_output is not None,
        ),
        "statistics_protocol": {
            "aggregation_unit": "parent",
            "confidence": 0.95,
            "bootstrap_samples": 10000,
            "runtime_seed": int(config.seed),
            "formal_gate_recomputes_with_frozen_pilot_seed": True,
        },
        "mean_teacher_forced_loss": finite_mean(losses),
        "generation_metrics": metric.compute(),
        "unavailable_modality_generation_metrics": unavailable_metric.compute(),
        "primary_score_bootstrap_ci": bootstrap_mean_ci(
            [
                float((row.get("raw_metrics") or {}).get("raw_field_accuracy"))
                if (row.get("raw_metrics") or {}).get("raw_field_accuracy") is not None
                else float((row.get("raw_metrics") or {}).get("caption_token_f1", 0.0))
                for row in generation_rows
            ],
            seed=config.seed + 7919,
        ),
        "same_image_retrieval": _same_image_retrieval(
            region_embeddings,
            text_embeddings,
            retrieval_parents,
            retrieval_phrases,
            retrieval_sample_ids,
        ),
        "counterfactual_sensitivity": {
            name: {
                "requested": int(config.counterfactual_samples),
                "n": len(values),
                "num_effective_parents": len(_counterfactual_parent_values(
                    counterfactual_rows, name, "target_score_delta"
                )),
                "aggregation_unit": "parent",
                "coverage_complete": (
                    int(config.counterfactual_samples) > 0
                    and len(values) >= int(config.counterfactual_samples)
                ),
                "mean_disagreement": finite_mean(_counterfactual_parent_values(
                    counterfactual_rows, name, "sensitivity"
                )),
                "mean_target_score_delta": finite_mean(_counterfactual_parent_values(
                    counterfactual_rows, name, "target_score_delta"
                )),
                "paired_target_score_delta_ci": bootstrap_mean_ci(
                    _counterfactual_parent_values(
                        counterfactual_rows, name, "target_score_delta"
                    ),
                    seed=config.seed + 104729 * (1 + list(counterfactual_values).index(name)),
                    samples=10000,
                ),
                "mean_factual_claim_count_delta": finite_mean(_counterfactual_parent_values(
                    counterfactual_rows, name, "factual_claim_count_delta"
                )),
                "paired_factual_claim_count_delta_ci": bootstrap_mean_ci(
                    _counterfactual_parent_values(
                        counterfactual_rows, name, "factual_claim_count_delta"
                    ),
                    seed=config.seed + 130363 * (1 + list(counterfactual_values).index(name)),
                    samples=10000,
                ),
                "skipped_no_effect": counterfactual_skipped_no_effect[name],
                "skipped_unavailable": counterfactual_skipped_unavailable[name],
            }
            for name, values in counterfactual_values.items()
        },
        "end_to_end_coverage": e2e.summary(loader.dataset) if e2e is not None else None,
        "cycle_localization": (
            summarize_cycle_localization(
                cycle_localization_rows,
                cycle,
                requested=int(config.cycle_localization_samples),
                seed=int(config.seed) + 524287,
            )
            if cycle is not None else None
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }
    if resolved_output is not None:
        resolved = resolved_output
        _write_jsonl(resolved / "raw_generations.jsonl", generation_rows)
        _write_jsonl(resolved / "counterfactual_generations.jsonl", counterfactual_rows)
        if e2e is not None:
            _write_jsonl(resolved / "end_to_end_target_audit.jsonl", end_to_end_rows)
        if cycle is not None:
            _write_jsonl(
                resolved / "cycle_localization.jsonl", cycle_localization_rows
            )
        if publish_report:
            write_json(resolved / "eval_report.json", report)
    return report


def description_selection_score(report: dict[str, Any], stage: str, metric_name: str = "auto") -> float:
    if metric_name != "auto":
        path: Any = report
        for part in metric_name.split("."):
            path = path.get(part) if isinstance(path, dict) else None
        if path is None:
            raise KeyError(f"checkpoint metric 不存在: {metric_name}")
        return float(path)
    if stage == "dior_alignment":
        return float((report.get("same_image_retrieval") or {}).get("mean_r1") or 0.0)
    generation = report.get("generation_metrics") or {}
    if stage in {"bridge_auto", "bridge_expert", "predicted_mask", "overfit"}:
        return float(generation.get("structured_field_macro_f1") or 0.0)
    return float(generation.get("caption_token_f1") or 0.0)
