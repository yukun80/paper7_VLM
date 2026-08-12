"""Grounded Corpus 构建与发布前验证编排。"""

from __future__ import annotations

from pathlib import Path

from .pilot import BuildResult, build_pilot_corpus
from .region import RegionBuildResult, build_region_assets
from .region_validation import validate_region_corpus
from .validation import validate_corpus


def build_auto(config_path: Path | str) -> BuildResult:
    """构建 pilot，并在原子发布前执行完整来源验证。"""

    return build_pilot_corpus(
        config_path,
        validator=lambda root: validate_corpus(root, verify_source=True),
    )


def build_region_corpus(config_path: Path | str) -> RegionBuildResult:
    """构建 Region Corpus，并在原子发布前执行完整来源验证。"""

    return build_region_assets(
        config_path,
        validator=lambda root: validate_region_corpus(root, verify_source=True),
    )
