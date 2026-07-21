"""Leakage-safe parent grouping and deterministic split assignment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from sami_gsd.contracts.canonical import CanonicalParentV3
from sami_gsd.contracts.config import SplitSettings
from sami_gsd.utilities.artifacts import canonical_json_bytes, sha256_bytes


SPLIT_PROTOCOL_VERSION = "sami_parent_group_split_v2_union_constraints_coverage_rebalanced"
SplitName = Literal["train", "val", "test"]


class SplitAssignmentError(ValueError):
    """Raised when grouping constraints or forced split roles conflict."""


@dataclass(frozen=True)
class SplitAssignment:
    """Stable parent assignments and connected grouping evidence."""

    parent_to_split: dict[str, SplitName]
    components: tuple[dict[str, object], ...]
    coverage_parent_ids: tuple[str, ...]
    aggregate_sha256: str


class _UnionFind:
    """Deterministic disjoint set for parent-level constraints."""

    def __init__(self, values: tuple[str, ...]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        """Return one compressed root."""

        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            following = self.parent[value]
            self.parent[value] = root
            value = following
        return root

    def union(self, left: str, right: str) -> None:
        """Join roots with a lexical representative."""

        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            low, high = sorted((left_root, right_root))
            self.parent[high] = low


def _constraint_keys(parent: CanonicalParentV3, duplicate_cluster_id: str) -> tuple[str, ...]:
    """Return all group keys that must remain in one connected split."""

    source = parent.source
    prefix = source.dataset
    values = [
        f"{prefix}:source_group:{source.source_group_id}",
        f"duplicate:{duplicate_cluster_id}",
    ]
    for kind, value in (
        ("scene", source.scene_id),
        ("event", source.event_id),
        ("region", source.region_id),
    ):
        if value is not None:
            values.append(f"{prefix}:{kind}:{value}")
    return tuple(sorted(values))


def assign_parent_splits(
    parents: tuple[CanonicalParentV3, ...],
    *,
    duplicate_clusters: dict[str, str],
    settings: SplitSettings,
    seed: int,
    forced_splits: dict[str, SplitName] | None = None,
    coverage_parent_ids: tuple[str, ...] | None = None,
) -> SplitAssignment:
    """Union constraints, hash components, then deterministically guarantee coverage."""

    ordered = tuple(sorted(parents, key=lambda item: item.parent_id))
    parent_ids = tuple(parent.parent_id for parent in ordered)
    if not parent_ids or len(parent_ids) != len(set(parent_ids)):
        raise SplitAssignmentError("split assignment requires non-empty unique parent IDs")
    if set(duplicate_clusters) != set(parent_ids):
        raise SplitAssignmentError("duplicate cluster mapping must cover every parent exactly")
    forced = forced_splits or {}
    if not set(forced).issubset(parent_ids):
        raise SplitAssignmentError("forced split mapping contains an unknown parent")
    coverage = tuple(sorted(set(coverage_parent_ids or ())))
    if not set(coverage).issubset(parent_ids):
        raise SplitAssignmentError("coverage parent set contains an unknown parent")

    union = _UnionFind(parent_ids)
    first_by_key: dict[str, str] = {}
    keys_by_parent: dict[str, tuple[str, ...]] = {}
    for parent in ordered:
        keys = _constraint_keys(parent, duplicate_clusters[parent.parent_id])
        keys_by_parent[parent.parent_id] = keys
        for key in keys:
            if key in first_by_key:
                union.union(parent.parent_id, first_by_key[key])
            else:
                first_by_key[key] = parent.parent_id

    members_by_root: dict[str, list[str]] = {}
    for parent_id in parent_ids:
        members_by_root.setdefault(union.find(parent_id), []).append(parent_id)
    states: list[dict[str, object]] = []
    for members_list in sorted(members_by_root.values(), key=lambda values: tuple(sorted(values))):
        members = tuple(sorted(members_list))
        forced_values = {forced[parent_id] for parent_id in members if parent_id in forced}
        if len(forced_values) > 1:
            raise SplitAssignmentError(f"connected group has conflicting forced splits: {members}")
        if forced_values:
            split = next(iter(forced_values))
            assignment_reason = "forced_source_policy"
        else:
            fraction = int(
                sha256_bytes(canonical_json_bytes({"seed": seed, "members": members}))[:16],
                16,
            ) / float(16**16)
            if fraction < settings.train:
                split = "train"
            elif fraction < settings.train + settings.val:
                split = "val"
            else:
                split = "test"
            assignment_reason = "seeded_component_hash"
        states.append(
            {
                "component_id": f"group-{sha256_bytes(canonical_json_bytes(list(members)))[:20]}",
                "members": members,
                "constraint_keys": sorted({key for parent_id in members for key in keys_by_parent[parent_id]}),
                "split": split,
                "assignment_reason": assignment_reason,
                "forced": bool(forced_values),
                "coverage_count": sum(parent_id in coverage for parent_id in members),
            }
        )

    required_splits: tuple[SplitName, ...] = tuple(
        name
        for name, ratio in (("train", settings.train), ("val", settings.val), ("test", settings.test))
        if ratio > 0
    )
    coverage_states = [state for state in states if int(state["coverage_count"]) > 0]
    require_coverage = len(coverage) >= len(required_splits)
    if require_coverage and len(coverage_states) < len(required_splits):
        raise SplitAssignmentError("connected coverage groups cannot populate every requested split")
    target_ratios: dict[SplitName, float] = {
        "train": settings.train,
        "val": settings.val,
        "test": settings.test,
    }
    while require_coverage:
        covered = {state["split"] for state in coverage_states}
        missing = next((name for name in required_splits if name not in covered), None)
        if missing is None:
            break
        component_counts = {
            name: sum(state["split"] == name for state in coverage_states)
            for name in required_splits
        }
        parent_counts = {
            name: sum(
                int(state["coverage_count"])
                for state in coverage_states
                if state["split"] == name
            )
            for name in required_splits
        }
        candidates: list[tuple[float, str, dict[str, object]]] = []
        for state in coverage_states:
            donor = state["split"]
            if state["forced"] or donor == missing or component_counts[donor] <= 1:
                continue
            moved = int(state["coverage_count"])
            projected = dict(parent_counts)
            projected[donor] -= moved
            projected[missing] += moved
            total = float(sum(projected.values()))
            score = sum(
                ((projected[name] / total) - target_ratios[name]) ** 2
                for name in required_splits
            )
            tie = sha256_bytes(
                canonical_json_bytes(
                    {"seed": seed, "missing": missing, "members": state["members"]}
                )
            )
            candidates.append((score, tie, state))
        if not candidates:
            raise SplitAssignmentError("forced coverage groups cannot populate every requested split")
        selected = min(candidates, key=lambda item: (item[0], item[1]))[2]
        selected["split"] = missing
        selected["assignment_reason"] = "seeded_component_hash_coverage_rebalance"

    mapping: dict[str, SplitName] = {}
    components: list[dict[str, object]] = []
    for state in states:
        members = tuple(state["members"])
        split = state["split"]
        for parent_id in members:
            mapping[parent_id] = split
        components.append(
            {
                "component_id": state["component_id"],
                "parent_ids": list(members),
                "constraint_keys": state["constraint_keys"],
                "split": split,
                "assignment_reason": state["assignment_reason"],
            }
        )
    payload = {
        "protocol": SPLIT_PROTOCOL_VERSION,
        "seed": seed,
        "settings": settings.model_dump(mode="json"),
        "coverage_parent_ids": list(coverage),
        "components": components,
    }
    return SplitAssignment(
        parent_to_split=dict(sorted(mapping.items())),
        components=tuple(components),
        coverage_parent_ids=coverage,
        aggregate_sha256=sha256_bytes(canonical_json_bytes(payload)),
    )


def apply_parent_splits(
    parents: tuple[CanonicalParentV3, ...],
    assignment: SplitAssignment,
) -> tuple[CanonicalParentV3, ...]:
    """Return strict records with frozen splits; task expansion must follow this call."""

    if set(assignment.parent_to_split) != {parent.parent_id for parent in parents}:
        raise SplitAssignmentError("split assignment does not match the parent set")
    return tuple(
        CanonicalParentV3.model_validate(
            {
                **parent.model_dump(mode="json"),
                "split": assignment.parent_to_split[parent.parent_id],
            }
        )
        for parent in sorted(parents, key=lambda item: item.parent_id)
    )


__all__ = [
    "SPLIT_PROTOCOL_VERSION",
    "SplitAssignment",
    "SplitAssignmentError",
    "apply_parent_splits",
    "assign_parent_splits",
]
