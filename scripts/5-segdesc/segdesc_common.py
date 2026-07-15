#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Segmentation-description unified-index helpers; not a standalone entrypoint."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
BUILDER_VERSION = "qpsalm_segdesc_index_builder_v2_expert_gate_bound"
INDEX_SCHEMA = "qpsalm_segdesc_index_v1"
VALIDATION_PROTOCOL = "qpsalm_segdesc_index_validation_v2"
STATISTICS_PROTOCOL = "qpsalm_segdesc_index_statistics_v2"
BRIDGE_AWAITING_STATUS = "awaiting_expert_review"
BRIDGE_FROZEN_STATUS = "expert_pilot_frozen"
TASK_WEIGHTS = {
    "segmentation": 1.0,
    "global_caption": 1.0,
    "region_alignment": 1.0,
    "region_description_auto": 0.5,
    "region_description_expert": 1.0,
}
TASK_COMPONENTS = {
    "segmentation": "landslide_segmentation_v2",
    "global_caption": "description_v2",
    "region_alignment": "description_v2",
    "region_description_auto": "landslide_bridge_v1",
    "region_description_expert": "landslide_bridge_v1",
}
TASK_INDEX_NAMES = {
    "segmentation": {
        "instruction_train.jsonl", "instruction_val.jsonl", "instruction_test.jsonl",
    },
    "global_caption": {"train_eligible.jsonl", "dev.jsonl", "test.jsonl"},
    "region_alignment": {"train_eligible.jsonl", "dev.jsonl", "test.jsonl"},
    "region_description_auto": {"auto_train.jsonl"},
    "region_description_expert": {"expert_all.jsonl"},
}


def bridge_publication_policy(
    bridge_status: str,
    *,
    expert_index_present: bool,
    gate_present: bool,
) -> dict[str, bool]:
    """Resolve expert publication without inferring supervision from stale files."""
    if bridge_status == BRIDGE_FROZEN_STATUS:
        if not expert_index_present:
            raise ValueError("Bridge 已冻结但缺少 indexes/expert_all.jsonl")
        if not gate_present:
            raise ValueError("Bridge 已冻结但缺少 evaluation_gate_manifest.json")
        return {
            "expert_index_published": True,
            "bridge_gate_published": True,
            "stale_expert_index_ignored": False,
            "stale_bridge_gate_ignored": False,
        }
    if bridge_status == BRIDGE_AWAITING_STATUS:
        return {
            "expert_index_published": False,
            "bridge_gate_published": False,
            "stale_expert_index_ignored": bool(expert_index_present),
            "stale_bridge_gate_ignored": bool(gate_present),
        }
    raise ValueError(
        "统一索引只接受正式 Bridge prepare 或 frozen expert Pilot，"
        f"当前 status={bridge_status!r}"
    )


def benchmark_root() -> Path:
    configured = os.environ.get("PAPER7_BENCHMARK_ROOT") or os.environ.get("BENCHMARK_PREFIX")
    if configured:
        path = Path(configured).expanduser()
        return path if path.is_absolute() else (REPO_ROOT / path).resolve(strict=False)
    sibling = REPO_ROOT.parent / "benchmark"
    return sibling if sibling.exists() else REPO_ROOT / "benchmark"


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    parts = path.parts
    if parts and parts[0] == "benchmark":
        return benchmark_root().joinpath(*parts[1:])
    return REPO_ROOT / path


def project_ref(path: str | Path) -> str:
    resolved = Path(path).resolve(strict=False)
    root = benchmark_root().resolve(strict=False)
    try:
        return (Path("benchmark") / resolved.relative_to(root)).as_posix()
    except ValueError:
        try:
            return resolved.relative_to(REPO_ROOT.resolve(strict=False)).as_posix()
        except ValueError:
            return str(resolved)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: record must be an object")
            rows.append(value)
    return rows


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".part", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path: Path, payload: Any) -> None:
    _atomic_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    _atomic_text(path, "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_output(output_dir: Path, overwrite: bool, dry_run: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"输出目录非空；请显式使用 --overwrite: {output_dir}")
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
