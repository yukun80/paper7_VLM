"""Replay D-1 artifact readiness from a serialized training config."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..protocols.config import serialized_segdesc_config_value
from ..data.artifact_readiness import (
    revalidate_saved_artifact_readiness_acceptance,
)


def replay_overfit_artifact_readiness(
    observations: dict[str, Any],
    composed_config: dict[str, Any] | None,
    readiness_path: Path | None,
) -> tuple[dict[str, Any] | None, str | None]:
    if composed_config is None or readiness_path is None:
        return None, "D-1 readiness config/report binding 缺失"
    try:
        acceptance = revalidate_saved_artifact_readiness_acceptance(
            observations.get("artifact_readiness_acceptance"),
            expected_description_benchmark=serialized_segdesc_config_value(
                composed_config, "description_benchmark"
            ),
            expected_bridge_benchmark=serialized_segdesc_config_value(
                composed_config, "bridge_benchmark"
            ),
            expected_unified_benchmark=serialized_segdesc_config_value(
                composed_config, "unified_benchmark"
            ),
            expected_description_cache=serialized_segdesc_config_value(
                composed_config, "description_vision_cache"
            ),
        )
        if acceptance.get("report") != str(
            readiness_path.resolve(strict=False)
        ):
            raise ValueError(
                "D-1 readiness acceptance 未绑定 source report"
            )
        return acceptance, None
    except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
        return None, f"{type(exc).__name__}: {exc}"
