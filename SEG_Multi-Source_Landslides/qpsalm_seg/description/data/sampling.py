"""Deterministic curriculum sampling and region-swap populations."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from ..protocols.io import strict_json_loads


def description_row_sample_id(row: dict[str, Any]) -> str:
    """Return the stable task identity used by evaluation metadata."""
    value = row.get("sample_id") or row.get("bridge_record_id")
    if not value:
        raise ValueError("description row 缺少 sample_id/bridge_record_id")
    return str(value)


def same_parent_region_swap_candidates(
    rows: list[dict[str, Any]],
    sample_id: str,
    *,
    catalog: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Return deterministic real-region candidates from the same parent.

    The alternate catalog may contain unreviewed Bridge region geometry, but
    its text target is never consumed.  Null/no-target rows are not regions and
    therefore cannot be used as a region-swap shortcut.
    """
    by_id = {description_row_sample_id(row): row for row in rows}
    current = by_id.get(str(sample_id))
    if current is None:
        return []
    parent = str(current.get("parent_sample_id") or "")
    current_region_id = str(current.get("region_id") or "")
    current_mask = (current.get("region_mask") or {}).get("path")
    candidates = []
    for row in catalog if catalog is not None else rows:
        candidate_id = description_row_sample_id(row)
        if candidate_id == str(sample_id):
            continue
        if str(row.get("parent_sample_id") or "") != parent:
            continue
        geometry = row.get("region_geometry") or {}
        has_box = geometry.get("type") == "box"
        has_mask = bool((row.get("region_mask") or {}).get("path"))
        if not (has_box or has_mask):
            continue
        if str(row.get("target_status") or "present") != "present":
            continue
        candidate_region_id = str(row.get("region_id") or "")
        candidate_mask = (row.get("region_mask") or {}).get("path")
        if (
            candidate_region_id
            and candidate_region_id == current_region_id
            and candidate_mask == current_mask
        ):
            continue
        candidates.append(row)
    return sorted(
        candidates,
        key=lambda row: (
            str(row.get("region_source") or "") == str(current.get("region_source") or ""),
            str(row.get("region_id") or ""),
            description_row_sample_id(row),
        ),
    )


def cross_parent_region_swap_candidates(
    rows: list[dict[str, Any]],
    sample_id: str,
    *,
    catalog: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Return deterministic real-region donors from different parents."""
    by_id = {description_row_sample_id(row): row for row in rows}
    current = by_id.get(str(sample_id))
    if current is None:
        return []
    parent = str(current.get("parent_sample_id") or "")
    candidates = []
    for row in catalog if catalog is not None else rows:
        candidate_id = description_row_sample_id(row)
        donor_parent = str(row.get("parent_sample_id") or "")
        if not donor_parent or donor_parent == parent:
            continue
        geometry = row.get("region_geometry") or {}
        has_box = geometry.get("type") == "box"
        has_mask = bool((row.get("region_mask") or {}).get("path"))
        if not (has_box or has_mask):
            continue
        if str(row.get("target_status") or "present") != "present":
            continue
        rank = hashlib.sha256(
            f"cross-parent-region:{sample_id}:{candidate_id}".encode("utf-8")
        ).hexdigest()
        candidates.append((rank, candidate_id, row))
    return [row for _rank, _candidate_id, row in sorted(candidates)]


def end_to_end_region_support(row: dict[str, Any]) -> tuple[bool, str]:
    """Return whether a Bridge row has an identifiable segmentation target.

    Global masks always map to the global segmentation instruction. Referring
    masks and pseudo components are valid only when inventory deduplication
    attached at least one referring-target alias. A pseudo component without
    such an alias has no language target for the segmentation model and must
    not silently fall back to whole-image segmentation.
    """
    source = str(row.get("region_source") or "unknown")
    if source == "gt_global_mask":
        return True, "global_instruction"
    aliases = [
        value for value in (row.get("source_region_aliases") or [])
        if isinstance(value, dict) and value.get("sample_id")
    ]
    if source == "gt_referring_mask":
        return (bool(aliases), "referring_alias" if aliases else "missing_referring_alias")
    if source == "pseudo_instance_component":
        return (
            bool(aliases),
            "component_with_referring_alias" if aliases else "component_without_language_target",
        )
    if source == "no_target":
        # Empty parents can map to an empty global instruction even when there
        # is no explicit no-target referring alias. The resolver verifies this.
        return True, "no_target_alias_or_empty_global"
    return False, f"unsupported_region_source:{source}"


def read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(strict_json_loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: 非法 JSONL") from exc
    return rows


def stable_weighted_index(
    seed: int,
    epoch: int,
    sample_id: str,
    weights: list[float],
) -> int:
    """Draw one deterministic epoch-specific answer from positive quality weights."""
    if not weights or any(not math.isfinite(value) or value < 0 for value in weights):
        raise ValueError("caption answer weights 必须是非空有限非负数列")
    total = sum(weights)
    if total <= 0:
        raise ValueError("caption answer weights 总和必须大于 0")
    digest = hashlib.sha256(f"{seed}:{epoch}:{sample_id}:weighted".encode()).digest()
    draw = int.from_bytes(digest[:8], "big") / float(2**64) * total
    cumulative = 0.0
    for index, weight in enumerate(weights):
        cumulative += weight
        if draw < cumulative:
            return index
    return len(weights) - 1


def caption_source_weights(
    rows: list[dict[str, Any]],
    *,
    stage: str,
    rsicap_mmrs_fraction: float,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Keep all parents while equalizing sources inside the D0/D1 protocol mix."""
    if stage not in {"mmrs_caption", "rsicap_caption"} or not rows:
        return {}, {"protocol": "not_applicable"}
    counts = Counter(str(row.get("source_dataset") or "unknown") for row in rows)
    source_group = {
        source: ("mmrs" if source.startswith("MMRS-") else "rsicap")
        for source in counts
    }
    sources_by_group: dict[str, list[str]] = {}
    for source, group in source_group.items():
        sources_by_group.setdefault(group, []).append(source)
    if stage == "mmrs_caption":
        group_mass = {"mmrs": 1.0}
    else:
        available_groups = set(sources_by_group)
        requested = {
            "rsicap": 1.0 - float(rsicap_mmrs_fraction),
            "mmrs": float(rsicap_mmrs_fraction),
        }
        normalizer = sum(requested[group] for group in available_groups)
        group_mass = {
            group: requested[group] / max(normalizer, 1.0e-12)
            for group in available_groups
        }
    raw_source_mass = {
        source: group_mass[group] / len(sources_by_group[group])
        for group, sources in sources_by_group.items()
        for source in sources
    }
    # Sum of per-row weights equals number of parents, so optimizer loss scale
    # remains comparable to stages that use unit weights.
    row_scale = float(len(rows))
    by_sample = {
        description_row_sample_id(row): (
            raw_source_mass[str(row.get("source_dataset") or "unknown")]
            / counts[str(row.get("source_dataset") or "unknown")]
            * row_scale
        )
        for row in rows
    }
    return by_sample, {
        "protocol": "qpsalm_caption_parent_epoch_source_weighting_v1",
        "stage": stage,
        "num_parents": len(rows),
        "source_counts": dict(sorted(counts.items())),
        "source_total_mass": {
            source: raw_source_mass[source]
            for source in sorted(raw_source_mass)
        },
        "group_total_mass": dict(sorted(group_mass.items())),
        "row_weight_mean": sum(by_sample.values()) / len(by_sample),
    }


def stable_subset(rows: list[dict[str, Any]], count: int, seed: int, namespace: str) -> list[dict[str, Any]]:
    if count >= len(rows):
        return list(rows)
    ranked = sorted(
        rows,
        key=lambda row: hashlib.sha256(
            f"{seed}:{namespace}:{row.get('sample_id') or row.get('bridge_record_id')}".encode()
        ).hexdigest(),
    )
    return ranked[:max(0, int(count))]


D_MINUS_ONE_CATEGORIES = ("global", "box", "mask", "null")


def select_d_minus_one_mixture(
    description_rows: list[dict[str, Any]],
    bridge_rows: list[dict[str, Any]],
    *,
    count: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build the protocol-required deterministic global/box/mask/null mix.

    Bridge bbox metadata describes a mask and is not a box-conditioned training
    example. D-1 therefore takes real full-image/box tasks from Description M1.1
    and mask/null tasks from Bridge rule candidates. The latter remain explicitly
    non-expert engineering supervision.
    """
    requested = int(count)
    if requested != 64:
        raise ValueError(
            "D-1 overfit 当前协议固定要求 64 条样本；"
            f"当前 requested={requested}"
        )
    pools: dict[str, list[dict[str, Any]]] = {
        "global": [
            row for row in description_rows
            if row.get("task_family") == "global_caption"
            and (row.get("region_geometry") or {}).get("type") == "full_image"
        ],
        "box": [
            row for row in description_rows
            if row.get("task_family") == "region_referring_expression"
            and (row.get("region_geometry") or {}).get("type") == "box"
        ],
        "mask": [
            row for row in bridge_rows
            if str(row.get("split") or "") == "train"
            and str(row.get("region_source") or "") != "no_target"
            and isinstance(row.get("region_mask"), dict)
            and bool((row.get("region_mask") or {}).get("path"))
        ],
        "null": [
            row for row in bridge_rows
            if str(row.get("split") or "") == "train"
            and (
                str(row.get("region_source") or "") == "no_target"
                or str(row.get("target_status") or "") == "absent"
            )
            and not row.get("region_mask")
        ],
    }
    missing = [name for name in D_MINUS_ONE_CATEGORIES if not pools[name]]
    if missing:
        raise RuntimeError(f"D-1 四路混合缺少真实类别: {missing}")

    base, remainder = divmod(requested, len(D_MINUS_ONE_CATEGORIES))
    quotas = {
        name: base + int(index < remainder)
        for index, name in enumerate(D_MINUS_ONE_CATEGORIES)
    }
    selected_by_category: dict[str, list[dict[str, Any]]] = {}
    for name in D_MINUS_ONE_CATEGORIES:
        if len(pools[name]) < quotas[name]:
            raise RuntimeError(
                f"D-1 category={name} 样本不足: "
                f"required={quotas[name]} available={len(pools[name])}"
            )
        selected = stable_subset(
            pools[name], quotas[name], seed, f"d_minus_one:{name}"
        )
        selected_by_category[name] = [
            {
                **row,
                "_d_minus_one_category": name,
                "_d_minus_one_item_kind": (
                    "description" if name in {"global", "box"} else "bridge"
                ),
                "_d_minus_one_target_authority": (
                    "description_benchmark_answer"
                    if name in {"global", "box"}
                    else "deterministic_rule_candidate_not_expert"
                ),
            }
            for row in selected
        ]

    # Round-robin order guarantees a bounded generation smoke observes all four
    # categories even when max_generate_samples is smaller than the population.
    mixed = [
        selected_by_category[name][offset]
        for offset in range(max(quotas.values()))
        for name in D_MINUS_ONE_CATEGORIES
        if offset < len(selected_by_category[name])
    ]

    native_sizes = set()
    for row in mixed:
        if row["_d_minus_one_item_kind"] == "description":
            visual = row.get("visual_ref") or {}
            size = (visual.get("height"), visual.get("width"))
        else:
            original = (row.get("visual_ref") or {}).get("original_size") or []
            size = tuple(original[:2]) if len(original) >= 2 else (None, None)
        if all(isinstance(value, int) and value > 0 for value in size):
            native_sizes.add(tuple(int(value) for value in size))
    return mixed, {
        "protocol": "qpsalm_d_minus_one_stratified_mixture_v1",
        "requested_samples": requested,
        "selected_samples": len(mixed),
        "sampling_seed": int(seed),
        "category_order": list(D_MINUS_ONE_CATEGORIES),
        "category_counts": {
            name: len(selected_by_category[name]) for name in D_MINUS_ONE_CATEGORIES
        },
        "category_available": {
            name: len(pools[name]) for name in D_MINUS_ONE_CATEGORIES
        },
        "native_source_sizes": [list(value) for value in sorted(native_sizes)],
        "num_native_source_sizes": len(native_sizes),
        "bridge_target_authority": "deterministic_rule_candidate_not_expert",
        "expert_truth_used": False,
        "category_region_token_policy": {
            "global": False,
            "box": True,
            "mask": True,
            "null": True,
        },
    }
