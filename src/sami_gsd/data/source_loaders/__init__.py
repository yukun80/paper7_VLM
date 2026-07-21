"""Resolved raw-source loaders allowed to feed Canonical materialization."""

from __future__ import annotations

from pathlib import Path

from sami_gsd.contracts.config import BenchmarkAuditConfig
from sami_gsd.data.materialize import SpatialParentInput
from sami_gsd.data.source_loaders.sen12 import load_sen12_parents


class SourceLoadingError(ValueError):
    """Raised when no configured source can safely feed a formal build."""


def load_resolved_spatial_parents(
    config: BenchmarkAuditConfig,
    *,
    datasets_root: Path,
) -> tuple[SpatialParentInput, ...]:
    """Load only licensed, task-approved sources with a unique greenfield policy."""

    eligible = [
        source
        for source in config.sources
        if source.enabled and source.license.allowed_for_training and "t1" in source.allowed_task_roles
    ]
    if not eligible:
        raise SourceLoadingError(
            "no training-eligible spatial source; approve exact source license evidence before a formal Small build"
        )
    parents: list[SpatialParentInput] = []
    for source in sorted(eligible, key=lambda item: item.source_key):
        if source.source_key == "sen12_landslides":
            parents.extend(
                load_sen12_parents(
                    source,
                    source_root=datasets_root / source.local_path,
                    limit=config.build.small_max_parents_per_source if config.mode == "small" else None,
                )
            )
        else:
            raise SourceLoadingError(
                f"training-approved source has no resolved canonical loader: {source.source_key}"
            )
    if not parents:
        raise SourceLoadingError("eligible source loaders emitted no annotated spatial parent")
    return tuple(sorted(parents, key=lambda item: item.parent_id))


__all__ = ["SourceLoadingError", "load_resolved_spatial_parents"]
