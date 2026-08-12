"""外部三库的确定性 parent train/val 划分与内容重复防泄漏。"""

from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import Any, Iterable

from .assets import normalized_image_sha256
from .io import canonical_json, sha256_bytes, sha256_text, stable_hash
from .config import BuildConfig
from .contracts import (
    AdapterResult,
    EXTERNAL_SPLIT_SCHEMA_VERSION,
    LogicalRole,
    MediaType,
    PendingAsset,
    SourceExample,
    parent_id,
)
from .errors import AssetError, ReasonCode


Node = tuple[str, str]


class _UnionFind:
    def __init__(self, nodes: Iterable[Node]) -> None:
        self.parent = {node: node for node in nodes}

    def find(self, node: Node) -> Node:
        root = node
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[node] != node:
            next_node = self.parent[node]
            self.parent[node] = root
            node = next_node
        return root

    def union(self, left: Node, right: Node) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if left_root < right_root:
            self.parent[right_root] = left_root
        else:
            self.parent[left_root] = right_root


def _asset_cache_key(asset: PendingAsset) -> tuple[str, ...]:
    if asset.payload is not None:
        return ("payload", sha256_bytes(asset.payload))
    if asset.source_path is not None:
        return ("file", str(asset.source_path))
    assert asset.archive_path is not None and asset.archive_member is not None
    return ("archive", str(asset.archive_path), asset.archive_member)


def _nested_counts(
    examples: Iterable[SourceExample],
    *,
    outer: str,
) -> dict[str, dict[str, int]]:
    counts: Counter[tuple[str, str]] = Counter()
    for example in examples:
        outer_value = (
            example.source
            if outer == "source"
            else example.task_family.value
        )
        counts[(outer_value, example.logical_role.value)] += 1
    output: dict[str, dict[str, int]] = defaultdict(dict)
    for (outer_value, role), count in sorted(counts.items()):
        output[outer_value][role] = count
    return dict(output)


def assign_external_splits(
    results: list[AdapterResult],
    config: BuildConfig,
    *,
    check_content: bool,
) -> dict[str, Any]:
    """原地更新 external examples，返回可写入 audit/statistics 的确定性摘要。"""

    fingerprint_cache: dict[tuple[str, ...], str] = {}
    fingerprint_failures: dict[tuple[str, ...], AssetError] = {}
    fingerprint_errors: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    if check_content:
        unique_image_assets: dict[tuple[str, ...], PendingAsset] = {}
        for result in results:
            for example in result.examples:
                for asset in example.assets:
                    if asset.media_type is MediaType.IMAGE:
                        unique_image_assets.setdefault(
                            _asset_cache_key(asset),
                            asset,
                        )

        def fingerprint(
            item: tuple[tuple[str, ...], PendingAsset],
        ) -> tuple[tuple[str, ...], str | None, AssetError | None]:
            cache_key, asset = item
            try:
                return (
                    cache_key,
                    normalized_image_sha256(
                        asset,
                        policy=config.asset_policy,
                    ),
                    None,
                )
            except AssetError as error:
                return cache_key, None, error

        ordered_assets = sorted(unique_image_assets.items())
        if config.workers == 1 or len(ordered_assets) <= 1:
            fingerprint_rows = [fingerprint(item) for item in ordered_assets]
        else:
            with ThreadPoolExecutor(
                max_workers=min(config.workers, len(ordered_assets)),
                thread_name_prefix="rs-generaldesc-fingerprint",
            ) as executor:
                fingerprint_rows = list(executor.map(fingerprint, ordered_assets))
        for cache_key, digest, error in fingerprint_rows:
            if error is not None:
                fingerprint_failures[cache_key] = error
            else:
                assert digest is not None
                fingerprint_cache[cache_key] = digest

        for result in results:
            valid_examples: list[SourceExample] = []
            deep_rejected_examples = 0
            for example in result.examples:
                failures: list[tuple[PendingAsset, AssetError]] = []
                for asset in example.assets:
                    if asset.media_type is not MediaType.IMAGE:
                        continue
                    cache_key = _asset_cache_key(asset)
                    error = fingerprint_failures.get(cache_key)
                    if error is not None:
                        fingerprint_errors[example.source].add(cache_key)
                        failures.append((asset, error))
                if not failures:
                    valid_examples.append(example)
                    continue
                primary_error = failures[0][1]
                evidence = dict(primary_error.details)
                evidence["invalid_assets"] = [
                    {
                        "role": asset.role,
                        "source_ref": asset.source_ref,
                        "reason_code": error.code.value,
                    }
                    for asset, error in failures
                ]
                result.add_skip(
                    source_record_id=example.source_record_id,
                    source_split=example.source_split,
                    task_family=example.task_family.value,
                    reason_code=primary_error.code,
                    evidence=evidence,
                )
                deep_rejected_examples += 1
            result.examples = valid_examples
            result.audit["deep_rejected_examples"] = deep_rejected_examples
            result.audit["selected_examples"] = len(valid_examples)
            result.audit["task_counts"] = dict(
                sorted(
                    Counter(
                        example.task_family.value
                        for example in valid_examples
                    ).items()
                )
            )
            result.audit["parent_count"] = len(
                {example.parent_key for example in valid_examples}
            )
            result.audit["skip_count"] = len(result.skips)
            result.audit["skip_reason_counts"] = dict(
                sorted(
                    Counter(
                        skip.reason_code.value for skip in result.skips
                    ).items()
                )
            )

    external_examples = [
        example
        for result in results
        for example in result.examples
    ]
    nodes = sorted({(row.source, row.parent_key) for row in external_examples})
    if not nodes:
        return {
            "schema_version": EXTERNAL_SPLIT_SCHEMA_VERSION,
            "version": config.external_split.version,
            "seed": config.seed,
            "validation_percent": config.external_split.validation_percent,
            "content_dedup_checked": check_content,
            "assignment_sha256": sha256_text(canonical_json([])),
            "parent_role_counts": {},
            "record_role_counts": {},
            "source_role_counts": {},
            "task_role_counts": {},
            "content_component_count": 0,
            "duplicate_content_component_count": 0,
            "content_fingerprint_error_count": sum(
                len(values) for values in fingerprint_errors.values()
            ),
        }

    by_source: dict[str, list[Node]] = defaultdict(list)
    tasks_by_node: dict[Node, set[str]] = defaultdict(set)
    for example in external_examples:
        tasks_by_node[(example.source, example.parent_key)].add(
            example.task_family.value
        )
    for node in nodes:
        by_source[node[0]].append(node)
    role_by_node: dict[Node, LogicalRole] = {}
    for source, source_nodes in sorted(by_source.items()):
        ranked = sorted(
            source_nodes,
            key=lambda node: (
                stable_hash(
                    config.seed,
                    config.external_split.version,
                    source,
                    node[1],
                ),
                node[1],
            ),
        )
        if config.external_split.validation_percent == 0 or len(ranked) < 2:
            validation_count = 0
        else:
            validation_count = max(
                1,
                len(ranked)
                * config.external_split.validation_percent
                // 100,
            )
            validation_count = min(validation_count, len(ranked) - 1)
        validation_nodes: set[Node] = set()
        if validation_count:
            minimum_task_parents = (
                100 + config.external_split.validation_percent - 1
            ) // config.external_split.validation_percent
            task_parents: dict[str, set[Node]] = defaultdict(set)
            for node in ranked:
                for task in tasks_by_node[node]:
                    task_parents[task].add(node)
            uncovered = {
                task
                for task, task_nodes in task_parents.items()
                if len(task_nodes) >= minimum_task_parents
            }
            rank_by_node = {
                node: index for index, node in enumerate(ranked)
            }
            while uncovered and len(validation_nodes) < validation_count:
                candidate = min(
                    (
                        node
                        for node in ranked
                        if node not in validation_nodes
                        and tasks_by_node[node] & uncovered
                    ),
                    key=lambda node: (
                        -len(tasks_by_node[node] & uncovered),
                        rank_by_node[node],
                        node,
                    ),
                )
                validation_nodes.add(candidate)
                uncovered -= tasks_by_node[candidate]
            for node in ranked:
                if len(validation_nodes) >= validation_count:
                    break
                validation_nodes.add(node)
        for node in ranked:
            role_by_node[node] = (
                LogicalRole.EXTERNAL_VAL
                if node in validation_nodes
                else LogicalRole.EXTERNAL_TRAIN
            )

    union = _UnionFind(nodes)
    digest_nodes: dict[str, set[Node]] = defaultdict(set)
    if check_content:
        for example in external_examples:
            node = (example.source, example.parent_key)
            for asset in example.assets:
                if asset.media_type is not MediaType.IMAGE:
                    continue
                cache_key = _asset_cache_key(asset)
                digest = fingerprint_cache[cache_key]
                digest_nodes[digest].add(node)
        for digest in sorted(digest_nodes):
            members = sorted(digest_nodes[digest])
            for member in members[1:]:
                union.union(members[0], member)

    components: dict[Node, list[Node]] = defaultdict(list)
    for node in nodes:
        components[union.find(node)].append(node)
    component_by_node: dict[Node, str] = {}
    duplicate_components = 0
    for members in components.values():
        ordered = sorted(members)
        component_id = "component_" + sha256_text(
            canonical_json(
                [
                    parent_id(source, parent_key)
                    for source, parent_key in ordered
                ]
            )
        )
        for node in ordered:
            component_by_node[node] = component_id
        if len(ordered) > 1:
            duplicate_components += 1
        provisional_roles = {role_by_node[node] for node in ordered}
        if len(provisional_roles) <= 1:
            continue
        component_rank = int(
            stable_hash(
                config.seed,
                config.external_split.version,
                "content_component",
                component_id,
            ),
            16,
        )
        component_role = (
            LogicalRole.EXTERNAL_VAL
            if component_rank % 100
            < config.external_split.validation_percent
            else LogicalRole.EXTERNAL_TRAIN
        )
        for node in ordered:
            role_by_node[node] = component_role

    assignment_rows = [
        {
            "parent_id": parent_id(source, parent_key),
            "source": source,
            "logical_role": role_by_node[(source, parent_key)].value,
            "content_component_id": component_by_node[(source, parent_key)],
        }
        for source, parent_key in nodes
    ]
    assignment_sha256 = sha256_text(canonical_json(assignment_rows))

    for result in results:
        updated: list[SourceExample] = []
        for example in result.examples:
            node = (example.source, example.parent_key)
            role = role_by_node[node]
            split_provenance = {
                "type": "external_parent_split",
                "version": config.external_split.version,
                "seed": config.seed,
                "validation_percent": config.external_split.validation_percent,
                "logical_role": role.value,
                "content_component_id": component_by_node[node],
                "content_dedup_checked": check_content,
                "assignment_sha256": assignment_sha256,
            }
            updated.append(
                replace(
                    example,
                    logical_role=role,
                    provenance=(*example.provenance, split_provenance),
                )
            )
        result.examples = updated
        result.audit["external_split"] = {
            "version": config.external_split.version,
            "seed": config.seed,
            "validation_percent": config.external_split.validation_percent,
            "content_dedup_checked": check_content,
            "assignment_sha256": sha256_text(
                canonical_json(
                    [
                        row
                        for row in assignment_rows
                        if row["source"] == result.source
                    ]
                )
            ),
            "record_role_counts": dict(
                sorted(Counter(row.logical_role.value for row in updated).items())
            ),
            "parent_role_counts": dict(
                sorted(
                    Counter(
                        role_by_node[(result.source, parent_key)].value
                        for parent_key in {
                            row.parent_key for row in updated
                        }
                    ).items()
                )
            ),
            "task_role_counts": _nested_counts(updated, outer="task"),
            "content_fingerprint_error_count": len(
                fingerprint_errors[result.source]
            ),
        }

    all_updated = [
        example
        for result in results
        for example in result.examples
    ]
    return {
        "schema_version": EXTERNAL_SPLIT_SCHEMA_VERSION,
        "version": config.external_split.version,
        "seed": config.seed,
        "validation_percent": config.external_split.validation_percent,
        "content_dedup_checked": check_content,
        "assignment_sha256": assignment_sha256,
        "parent_role_counts": dict(
            sorted(
                Counter(role_by_node[node].value for node in nodes).items()
            )
        ),
        "record_role_counts": dict(
            sorted(Counter(row.logical_role.value for row in all_updated).items())
        ),
        "source_role_counts": _nested_counts(all_updated, outer="source"),
        "task_role_counts": _nested_counts(all_updated, outer="task"),
        "content_component_count": len(components),
        "duplicate_content_component_count": duplicate_components,
        "content_fingerprint_error_count": sum(
            len(values) for values in fingerprint_errors.values()
        ),
    }
