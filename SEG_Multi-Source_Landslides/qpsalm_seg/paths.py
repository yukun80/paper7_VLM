#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""QPSALM 项目路径解析。

脚本作用：把索引中的 datasets/...、benchmark/... 逻辑引用映射到仓库同级
大数据目录，同时让 outputs/models_zoo/configs 等路径继续相对仓库根目录。
是否改写数据：不会。
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve_root_override(value: str | None, default: Path) -> Path:
    """解析环境变量覆盖；相对路径统一相对仓库根目录。"""
    if not value:
        return default.resolve(strict=False)
    path = Path(value).expanduser()
    if not path.is_absolute() and len(path.parts) == 1 and path.parts[0] in {"datasets", "benchmark"}:
        return _default_external_root(path.parts[0]).resolve(strict=False)
    resolved = path if path.is_absolute() else REPO_ROOT / path
    return resolved.resolve(strict=False)


def _default_external_root(name: str) -> Path:
    """优先同级存储目录，并在其不存在时兼容旧仓库内目录。"""
    sibling = REPO_ROOT.parent / name
    legacy = REPO_ROOT / name
    return sibling if sibling.exists() or not legacy.exists() else legacy


def _benchmark_override() -> str | None:
    value = os.environ.get("PAPER7_BENCHMARK_ROOT")
    if value:
        return value
    prefix = os.environ.get("BENCHMARK_PREFIX")
    if not prefix:
        return None
    prefix_path = Path(prefix)
    if not prefix_path.is_absolute() and prefix_path.parts and prefix_path.parts[0] == "benchmark":
        return str(_default_external_root("benchmark"))
    return str(prefix_path.parent)


DATASETS_ROOT = _resolve_root_override(
    os.environ.get("PAPER7_DATASETS_ROOT") or os.environ.get("DATASETS_ROOT"),
    _default_external_root("datasets"),
)
BENCHMARK_ROOT = _resolve_root_override(_benchmark_override(), _default_external_root("benchmark"))


def resolve_project_path(path_ref: str | Path | None) -> Path | None:
    """解析项目路径，保留绝对路径并识别 datasets/benchmark 逻辑前缀。"""
    if path_ref is None:
        return None
    base = str(path_ref).split("::", 1)[0]
    path = Path(base).expanduser()
    if path.is_absolute():
        return path.resolve(strict=False)
    if path.parts and path.parts[0] == "datasets":
        return DATASETS_ROOT.joinpath(*path.parts[1:]).resolve(strict=False)
    if path.parts and path.parts[0] == "benchmark":
        return BENCHMARK_ROOT.joinpath(*path.parts[1:]).resolve(strict=False)
    return (REPO_ROOT / path).resolve(strict=False)


def resolve_repo_path(path_ref: str | Path | None) -> Path | None:
    """兼容旧调用名；新代码应将其视为项目逻辑路径解析器。"""
    return resolve_project_path(path_ref)


def to_project_ref(path_ref: str | Path | None) -> str | None:
    """把物理路径转换为可移植项目引用。"""
    if path_ref is None:
        return None
    path = Path(path_ref)
    if not path.is_absolute():
        return path.as_posix()
    path = path.resolve(strict=False)
    for logical_root, physical_root in (("datasets", DATASETS_ROOT), ("benchmark", BENCHMARK_ROOT)):
        try:
            relative = path.relative_to(physical_root)
            return (Path(logical_root) / relative).as_posix()
        except ValueError:
            pass
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def validate_output_replacement_safety(
    output_ref: str | Path,
    protected_inputs: Mapping[str, str | Path | None],
) -> dict[str, str]:
    """Reject output/input overlap that could mutate replay-bound source evidence."""
    output = resolve_project_path(output_ref) or Path(output_ref)
    output = output.resolve(strict=False)
    resolved_inputs: dict[str, str] = {}
    for label, reference in protected_inputs.items():
        if reference is None or str(reference) == "":
            continue
        source = resolve_project_path(reference) or Path(reference)
        source = source.resolve(strict=False)
        resolved_inputs[str(label)] = str(source)
        source_inside_output = False
        output_inside_source = False
        try:
            source.relative_to(output)
            source_inside_output = True
        except ValueError:
            pass
        try:
            output.relative_to(source)
            output_inside_source = True
        except ValueError:
            pass
        if source_inside_output or output_inside_source:
            raise ValueError(
                f"{label} 与待替换 output 路径重叠，拒绝删除或覆盖输入: "
                f"input={source} output={output}"
            )
    return resolved_inputs
