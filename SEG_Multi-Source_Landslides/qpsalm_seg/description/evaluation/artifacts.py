"""Binary mask artifacts and counterfactual input identity replay."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from qpsalm_seg.paths import resolve_project_path

from ..data.engineering_contracts import REGION_INPUT_SOURCE_PROTOCOL
from ..data.vision_cache import DescriptionVisionFeatureBank, description_cache_key
from ..protocols.region_geometry import (
    project_native_region_mask_to_cache,
)
from ..protocols.io import (
    canonical_sha256 as _json_sha256,
    sha256_file as _file_sha256,
    strict_json_loads,
    tensor_sha256 as _tensor_sha256,
)
from .contracts import (
    COUNTERFACTUAL_INPUT_AUDIT_PROTOCOL,
    EVALUATION_MASK_ARTIFACT_PROTOCOL,
    EVALUATION_MASK_INVENTORY_PROTOCOL,
)


_EVALUATION_MASK_ROLES = (
    "region_input",
    "cycle_prediction",
    "cycle_target",
    "cycle_source",
    "cycle_valid",
    "end_to_end_source",
)


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
    projected, source_mapping = project_native_region_mask_to_cache(
        torch.from_numpy(array.astype(np.float32, copy=False))[None],
        dict(binding["render_transform"]),
    )
    if binding.get("source_to_render_mapping") != source_mapping:
        raise ValueError("region input source-to-render mapping 不一致")
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
            projected_tensor, source_mapping = project_native_region_mask_to_cache(
                torch.from_numpy(
                    loaded["cycle_source"].astype(np.float32, copy=False)
                )[None],
                dict(cycle_audit["description_render_transform"]),
            )
            if cycle_audit.get("source_to_render_mapping") != source_mapping:
                raise ValueError("cycle source-to-render mapping 不一致")
            projected = _binary_mask_array(projected_tensor)
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
