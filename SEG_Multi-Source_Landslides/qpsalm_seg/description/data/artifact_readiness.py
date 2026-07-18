#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Live Description/Bridge/Unified/M3 readiness contracts.

用途：以只读方式重放首次 D-1 前的全部数据与 cache artifact，并消费已发布报告。
推荐调用：由 ``qpsalm-segdesc validate artifacts`` 发布，再由 D-1 trainer 重验。
输入：Description v4、Bridge v7、Unified v3、Description Cache M3 v3。
输出：确定性的 readiness report 或绑定当前 live artifact 的 acceptance。
写入行为：本模块只读；workflow 负责原子发布报告。
工作流阶段：M2/M3 到 D-1 的工程门禁。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from qpsalm_seg.paths import resolve_project_path

from .cache_migration import revalidate_published_cache_origin
from .engineering_contracts import (
    DESCRIPTION_BUILDER_VERSION,
    require_engineering_bridge,
    require_engineering_description,
)
from .expert_contracts import (
    BRIDGE_BUILDER_VERSION,
    require_frozen_expert_bridge,
)
from .unified_artifact import revalidate_unified_artifact
from .vision_cache import DescriptionVisionFeatureBank
from ..protocols.io import (
    nested_file_bindings_current,
    read_json,
    sha256_file,
)


ARTIFACT_READINESS_PROTOCOL = (
    "qpsalm_segdesc_artifact_readiness_v2_training_consumable"
)
ARTIFACT_READINESS_ACCEPTANCE_PROTOCOL = (
    "qpsalm_segdesc_artifact_readiness_acceptance_v1_live_replayed"
)


def _root(value: str | Path, *, label: str) -> Path:
    path = resolve_project_path(value) or Path(value)
    if not path.is_dir():
        raise FileNotFoundError(f"{label} 不存在: {path}")
    return path.resolve(strict=False)


def build_artifact_readiness_report(
    *,
    mode: str,
    description_benchmark: str | Path,
    bridge_benchmark: str | Path,
    unified_benchmark: str | Path,
    description_cache: str | Path,
) -> dict[str, Any]:
    """Compute the deterministic readiness report without writing files."""
    report: dict[str, Any] = {
        "protocol": ARTIFACT_READINESS_PROTOCOL,
        "mode": str(mode),
        "status": "engineering-invalid",
        "ready": False,
        "expert_truth_used": False,
        "inputs": {},
        "checks": {},
        "errors": [],
    }
    try:
        if mode not in {"small", "full"}:
            raise ValueError(f"mode 必须是 small/full: {mode!r}")
        description = _root(
            description_benchmark, label="Description benchmark"
        )
        bridge = _root(bridge_benchmark, label="Bridge benchmark")
        unified = _root(unified_benchmark, label="Unified benchmark")
        cache = _root(description_cache, label="Description M3 cache")
        report["inputs"] = {
            "description_benchmark": str(description),
            "bridge_benchmark": str(bridge),
            "unified_benchmark": str(unified),
            "description_cache": str(cache),
        }

        bank = DescriptionVisionFeatureBank(cache, max_open_shards=1)
        shard_replay = bank.verify_all_shards()
        report["cache"] = {
            "artifact_binding": bank.artifact_binding(),
            "shard_replay": shard_replay,
        }
        report["checks"]["m3_v3_all_shards_replayed"] = (
            shard_replay.get("all_verified") is True
        )
        cache_origin_audit = revalidate_published_cache_origin(cache, bank)
        report["cache"]["origin"] = cache_origin_audit
        cache_origin_checks = dict(cache_origin_audit.get("checks") or {})
        report["checks"]["m3_v3_origin_bound"] = bool(
            cache_origin_checks
            and all(cache_origin_checks.values())
        )

        description_audit = require_engineering_description(
            description, bank
        )
        bridge_audit = require_engineering_bridge(bridge, bank)
        description_validation = read_json(
            Path(description_audit["validation_report"]),
            label="Description validation",
        )
        bridge_validation = read_json(
            Path(bridge_audit["validation_report"]),
            label="Bridge validation",
        )
        report["description"] = description_audit
        report["bridge"] = bridge_audit
        report["checks"]["description_v4_live_bound"] = (
            description_audit.get("builder_version")
            == DESCRIPTION_BUILDER_VERSION
            and description_validation.get("mode") == mode
        )
        bridge_status = str(bridge_audit.get("status") or "")
        bridge_expert = dict(bridge_validation.get("expert") or {})
        frozen_expert_audit = (
            require_frozen_expert_bridge(bridge)
            if bridge_status == "expert_pilot_frozen"
            else None
        )
        report["frozen_expert_bridge"] = frozen_expert_audit
        report["checks"]["bridge_v7_live_bound"] = (
            bridge_audit.get("builder_version") == BRIDGE_BUILDER_VERSION
            and bridge_status
            in {"awaiting_expert_review", "expert_pilot_frozen"}
            and bridge_audit.get("expert_truth_used") is False
            and bridge_validation.get("mode") == mode
            and (
                (
                    bridge_status == "expert_pilot_frozen"
                    and frozen_expert_audit is not None
                )
                or (
                    bridge_status == "awaiting_expert_review"
                    and int(bridge_expert.get("expert_records", -1)) == 0
                    and bridge_expert.get("gate_frozen") is False
                )
            )
        )
        unified_audit = revalidate_unified_artifact(
            unified,
            mode=mode,
            bridge_status=bridge_status,
            description_root=description,
            bridge_root=bridge,
        )
        report["unified"] = unified_audit
        report["checks"]["unified_v3_live_bound"] = all(
            unified_audit["checks"].values()
        )
    except Exception as exc:
        report["errors"].append({
            "type": type(exc).__name__,
            "message": str(exc),
        })

    failed = [
        name for name, value in report["checks"].items()
        if value is not True
    ]
    if failed:
        report["errors"].append({
            "type": "ReadinessChecksFailed",
            "message": ", ".join(failed),
        })
    report["ready"] = bool(report["checks"]) and not report["errors"]
    report["status"] = (
        "engineering-valid" if report["ready"]
        else "engineering-invalid"
    )
    return report


def validate_artifact_readiness_report(
    report_reference: str | Path,
    *,
    expected_description_benchmark: str | Path,
    expected_bridge_benchmark: str | Path,
    expected_unified_benchmark: str | Path,
    expected_description_cache: str | Path,
) -> dict[str, Any]:
    """Recompute a published readiness report from its expected live roots."""
    path = resolve_project_path(report_reference) or Path(report_reference)
    if not path.is_file():
        raise FileNotFoundError(f"D-1 缺少 artifact readiness report: {path}")
    saved = read_json(path, label="SegDesc artifact readiness")
    if (
        saved.get("protocol") != ARTIFACT_READINESS_PROTOCOL
        or saved.get("status") != "engineering-valid"
        or saved.get("ready") is not True
        or saved.get("errors") != []
        or not saved.get("checks")
        or not all(value is True for value in saved["checks"].values())
    ):
        raise ValueError("D-1 artifact readiness report 未通过当前协议")
    current = build_artifact_readiness_report(
        mode=str(saved.get("mode") or ""),
        description_benchmark=expected_description_benchmark,
        bridge_benchmark=expected_bridge_benchmark,
        unified_benchmark=expected_unified_benchmark,
        description_cache=expected_description_cache,
    )
    if current != saved:
        raise ValueError(
            "D-1 artifact readiness report 与当前 Description/Bridge/Unified/M3 不一致"
        )
    return {
        "protocol": ARTIFACT_READINESS_ACCEPTANCE_PROTOCOL,
        "status": "engineering-valid",
        "report": str(path.resolve(strict=False)),
        "report_sha256": sha256_file(path),
        "mode": str(saved["mode"]),
        "inputs": dict(saved["inputs"]),
        "bridge_status": str(
            (saved.get("bridge") or {}).get("status") or ""
        ),
        "expert_truth_used": False,
        "errors": [],
    }


def revalidate_saved_artifact_readiness_acceptance(
    saved: Any,
    *,
    expected_description_benchmark: str | Path,
    expected_bridge_benchmark: str | Path,
    expected_unified_benchmark: str | Path,
    expected_description_cache: str | Path,
) -> dict[str, Any]:
    """Replay a launch-time acceptance without rescanning tensor shards.

    The overfit launch performs the expensive full report recomputation.  Later
    checkpoint/D0 gates bind that accepted report byte-for-byte and replay all
    nested file hashes; this avoids hashing the 23 GiB cache twice per gate.
    """
    if not isinstance(saved, dict):
        raise ValueError("D-1 checkpoint 缺少 artifact readiness acceptance")
    report_path = resolve_project_path(str(saved.get("report") or ""))
    if (
        saved.get("protocol") != ARTIFACT_READINESS_ACCEPTANCE_PROTOCOL
        or saved.get("status") != "engineering-valid"
        or saved.get("expert_truth_used") is not False
        or saved.get("errors") != []
        or report_path is None
        or not report_path.is_file()
        or sha256_file(report_path) != saved.get("report_sha256")
    ):
        raise ValueError("D-1 artifact readiness acceptance 已损坏或漂移")
    report = read_json(report_path, label="SegDesc artifact readiness")
    expected_inputs = {
        "description_benchmark": str(_root(
            expected_description_benchmark,
            label="Description benchmark",
        )),
        "bridge_benchmark": str(_root(
            expected_bridge_benchmark,
            label="Bridge benchmark",
        )),
        "unified_benchmark": str(_root(
            expected_unified_benchmark,
            label="Unified benchmark",
        )),
        "description_cache": str(_root(
            expected_description_cache,
            label="Description M3 cache",
        )),
    }
    if (
        report.get("protocol") != ARTIFACT_READINESS_PROTOCOL
        or report.get("status") != "engineering-valid"
        or report.get("ready") is not True
        or report.get("errors") != []
        or not report.get("checks")
        or not all(value is True for value in report["checks"].values())
        or report.get("inputs") != expected_inputs
        or saved.get("inputs") != expected_inputs
        or saved.get("mode") != report.get("mode")
        or saved.get("bridge_status")
        != str((report.get("bridge") or {}).get("status") or "")
        or not nested_file_bindings_current(report)
    ):
        raise ValueError(
            "D-1 artifact readiness acceptance 与当前绑定文件不一致"
        )
    return dict(saved)
