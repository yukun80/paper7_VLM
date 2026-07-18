#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""D0 preflight consumption and live launch revalidation.

用途：在正式 D0 写入前重放 ready preflight 的配置、D-1、benchmark 与 cache 绑定。
推荐调用：由 ``qpsalm-segdesc train`` 消费 ``--d0-preflight-report``。
输入：config v2、resolved config、preflight report 和 construction device。
输出：可原子写入正式 run 的 D0 preflight acceptance。
写入行为：本模块只读；发布由 training workflow 负责。
工作流阶段：D0 formal launch gate。
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
from typing import Any

from qpsalm_seg.paths import resolve_project_path

from ..protocols.config import (
    SEGDESC_CONFIG_PROTOCOL,
    SegDescConfig,
    require_serialized_segdesc_config,
)
from ..data.cache_migration import revalidate_published_cache_origin
from ..data.loaders import (
    build_description_dataset,
    build_description_loader,
    description_collator_audit,
)
from ..data.engineering_contracts import (
    require_engineering_bridge,
    require_engineering_description,
)
from ..data.vision_cache import DescriptionVisionFeatureBank
from ..evaluation.d_minus_one import validate_d_minus_one_gate
from ..protocols.io import canonical_sha256, read_json, sha256_file
from ..protocols.launch import build_d0_training_launch
from ..protocols.stages import DESCRIPTION_STREAM_SEED_OFFSETS
from ..protocols.versions import (
    D0_CONSTRUCTION_CONTRACT_PROTOCOL,
    D0_PREFLIGHT_ACCEPTANCE_PROTOCOL,
    D0_PREFLIGHT_PROTOCOL,
)
from ..training.engineering_gates import dataset_data_audit
from ..training.streams import description_stream_binding


def validate_d0_preflight_for_launch(
    config: SegDescConfig,
    *,
    config_reference: str | Path,
    report_reference: str | Path,
    device_name: str,
) -> dict[str, Any]:
    """Revalidate a ready preflight before the first formal D0 write."""
    report_path = resolve_project_path(report_reference) or Path(report_reference)
    report = read_json(report_path, label="D0 preflight report")
    checks = dict(report.get("checks") or {})
    launch = dict(report.get("formal_training_launch") or {})
    if (
        report.get("protocol") != D0_PREFLIGHT_PROTOCOL
        or report.get("config_protocol") != SEGDESC_CONFIG_PROTOCOL
        or report.get("stage") != "mmrs_caption"
        or report.get("status") != "engineering-valid"
        or report.get("ready") is not True
        or report.get("optimizer_steps") != 0
        or report.get("errors") != []
        or not checks
        or not all(value is True for value in checks.values())
        or launch.get("unique") is not True
    ):
        raise ValueError("D0 preflight report 未通过当前 construction-only 门禁")

    config_path = resolve_project_path(config_reference) or Path(config_reference)
    if not config_path.is_file():
        raise FileNotFoundError(
            f"D0 preflight 绑定的 resolved config 不存在: {config_path}"
        )
    if (
        launch.get("resolved_config") != str(config_path.resolve(strict=False))
        or launch.get("resolved_config_sha256") != sha256_file(config_path)
    ):
        raise ValueError("D0 preflight resolved config 路径或 SHA 已漂移")
    serialized = require_serialized_segdesc_config(
        read_json(config_path, label="D0 resolved config"),
        label="D0 resolved config",
    )
    if serialized != report.get("resolved_config"):
        raise ValueError("D0 preflight report 与 resolved config 内容不一致")
    if report.get("resolved_config_sha256") != canonical_sha256(serialized):
        raise ValueError("D0 preflight report 的 canonical config SHA 已漂移")
    expected_runtime = deepcopy(serialized)
    expected_runtime["training"]["output_dir"] = config.training.output_dir
    if config.to_dict() != expected_runtime:
        raise ValueError("D0 正式启动参数偏离 preflight resolved config")
    formal_output = str(
        (resolve_project_path(config.training.output_dir) or Path(
            config.training.output_dir
        )).resolve(strict=False)
    )
    if launch.get("output_dir") != formal_output:
        raise ValueError("D0 正式 output-dir 与 preflight 发布值不一致")
    if (
        report.get("construction_device") != str(device_name)
        or launch.get("device") != str(device_name)
    ):
        raise ValueError("D0 正式 device 与 preflight construction device 不一致")
    expected_launch = build_d0_training_launch(
        python_executable=sys.executable,
        config_path=config_path,
        config_sha256=sha256_file(config_path),
        seed=config.training.seed,
        device_name=device_name,
        d_minus_one_gate=str(config.training.d_minus_one_gate),
        output_dir=formal_output,
        preflight_report=report_path,
    )
    if launch != expected_launch:
        raise ValueError("D0 preflight 发布的唯一正式命令或 argv 已漂移")
    segmentation_config = (
        resolve_project_path(config.model.segmentation_config)
        or Path(config.model.segmentation_config)
    )
    if not segmentation_config.is_file():
        raise FileNotFoundError(
            f"D0 segmentation config 不存在: {segmentation_config}"
        )
    expected_sources = {
        "description_benchmark": str((
            resolve_project_path(config.data.description_benchmark)
            or Path(config.data.description_benchmark)
        ).resolve(strict=False)),
        "bridge_benchmark": str((
            resolve_project_path(config.data.bridge_benchmark)
            or Path(config.data.bridge_benchmark)
        ).resolve(strict=False)),
        "unified_benchmark": str((
            resolve_project_path(config.data.unified_benchmark)
            or Path(config.data.unified_benchmark)
        ).resolve(strict=False)),
        "segmentation_config": str(
            segmentation_config.resolve(strict=False)
        ),
        "segmentation_config_sha256": sha256_file(segmentation_config),
    }
    if report.get("source_bindings") != expected_sources:
        raise ValueError("D0 preflight source/config binding 已漂移")

    d_minus_one = validate_d_minus_one_gate(
        config.training.d_minus_one_gate or "",
        expected_description_benchmark=config.data.description_benchmark,
        expected_bridge_benchmark=config.data.bridge_benchmark,
        expected_unified_benchmark=config.data.unified_benchmark,
        expected_description_cache=config.model.description_vision_cache,
    )
    if d_minus_one != report.get("d_minus_one_acceptance"):
        raise ValueError("D0 preflight 的 D-1 acceptance 已漂移")
    bank = DescriptionVisionFeatureBank(config.model.description_vision_cache)
    cache_report = dict(report.get("description_cache") or {})
    cache_snapshot = bank.file_metadata_snapshot()
    cache_origin = revalidate_published_cache_origin(
        config.model.description_vision_cache, bank
    )
    cache_origin_checks = dict(cache_origin.get("checks") or {})
    if (
        bank.artifact_binding() != cache_report.get("artifact_binding")
        or cache_snapshot
        != dict(cache_report.get("shard_replay") or {}).get("metadata_snapshot")
        or cache_origin != cache_report.get("origin")
        or not cache_origin_checks
        or not all(cache_origin_checks.values())
    ):
        raise ValueError(
            "D0 preflight 后 Description/segmentation source cache artifact 已漂移"
        )
    description_root = (
        resolve_project_path(config.data.description_benchmark)
        or Path(config.data.description_benchmark)
    )
    bridge_root = (
        resolve_project_path(config.data.bridge_benchmark)
        or Path(config.data.bridge_benchmark)
    )
    description_audit = require_engineering_description(description_root, bank)
    bridge_audit = require_engineering_bridge(bridge_root, bank)
    live_bindings = {
        "description": description_audit,
        "bridge": bridge_audit,
        "expert_truth_used": False,
        "bridge_status": bridge_audit["status"],
    }
    if live_bindings != report.get("benchmark_bindings"):
        raise ValueError("D0 preflight 后 benchmark binding 已漂移")
    dataset = build_description_dataset(
        config, bank, split="train", training=True
    )
    dataset_audit = dataset_data_audit(dataset)
    loader = build_description_loader(
        dataset,
        config,
        training=True,
        sampler_seed=(
            int(config.training.seed)
            + DESCRIPTION_STREAM_SEED_OFFSETS["main"]
        ),
    )
    collator_audit = description_collator_audit(next(iter(loader)))
    stream_binding = description_stream_binding(
        "main",
        {"config": config, "dataset": dataset, "loader": loader},
        dataset_audit,
    )
    loader_report = dict(report.get("loader") or {})
    if (
        dataset_audit != report.get("dataset")
        or collator_audit != report.get("collator")
        or stream_binding != loader_report.get("stream_binding")
        or int(loader_report.get("batches", -1)) != len(loader)
        or int(loader_report.get("num_workers", -1))
        != int(config.data.num_workers)
        or loader_report.get("batch_sampler")
        != type(loader.batch_sampler).__name__
    ):
        raise ValueError(
            "D0 preflight 后 dataset/collator/formal loader binding 已漂移"
        )
    construction_contract = {
        "protocol": D0_CONSTRUCTION_CONTRACT_PROTOCOL,
        "segmentation_migration": dict(
            report.get("segmentation_migration") or {}
        ),
        "dataset": dataset_audit,
        "collator": collator_audit,
        "loader": loader_report,
        "trainable_parameters": dict(
            report.get("trainable_parameters") or {}
        ),
        "optimizer": dict(report.get("optimizer") or {}),
    }
    return {
        "protocol": D0_PREFLIGHT_ACCEPTANCE_PROTOCOL,
        "preflight_report": str(report_path.resolve(strict=False)),
        "preflight_report_sha256": sha256_file(report_path),
        "resolved_config": str(config_path.resolve(strict=False)),
        "resolved_config_sha256": sha256_file(config_path),
        "formal_output_dir": formal_output,
        "device": str(device_name),
        "d_minus_one_acceptance_protocol": d_minus_one.get("protocol"),
        "description_cache_metadata_snapshot": cache_snapshot,
        "description_cache_origin": cache_origin,
        "construction_contract": construction_contract,
        "status": "engineering-valid",
        "errors": [],
    }
