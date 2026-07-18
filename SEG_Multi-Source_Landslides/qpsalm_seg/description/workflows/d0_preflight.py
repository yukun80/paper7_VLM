#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""D0 construction-only preflight; never performs an optimizer step.

用途：在正式 D0 前重验 D-1、benchmark/cache、模型迁移、dataset/collator 与 optimizer。
推荐命令：``qpsalm-segdesc train ... --stage mmrs_caption --preflight-only``。
输入：config v2、D-1 gate、M3 v3 cache、Description v4、Bridge v7、segmentation 权重。
输出：原子写入 ``preflight_report.json``。
写入行为：只写训练 ``--output-dir``；不会执行 backward 或 optimizer.step。
工作流阶段：D0 launch gate。
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

from qpsalm_seg.paths import (
    resolve_project_path,
    validate_output_replacement_safety,
)

from ..data.loaders import (
    build_description_dataset,
    build_description_loader,
    description_collator_audit,
    description_device,
    set_description_seed,
)
from ..data.cache_migration import revalidate_published_cache_origin
from ..protocols.config import SEGDESC_CONFIG_PROTOCOL, SegDescConfig
from ..data.engineering_contracts import (
    require_engineering_bridge,
    require_engineering_description,
)
from ..data.vision_cache import DescriptionVisionFeatureBank
from ..protocols.io import (
    atomic_write_json,
    canonical_sha256,
    sha256_file,
)
from ..protocols.launch import build_d0_training_launch
from ..protocols.stages import DESCRIPTION_STREAM_SEED_OFFSETS, get_stage_spec
from ..protocols.versions import D0_PREFLIGHT_PROTOCOL
from ..training.runtime import (
    build_description_optimizer,
    build_segdesc_model,
    description_optimizer_audit,
    description_trainable_parameter_manifest,
)
from ..training.engineering_gates import dataset_data_audit
from ..training.streams import description_stream_binding
from ..evaluation.d_minus_one import validate_d_minus_one_gate


def run_d0_preflight(
    config: SegDescConfig,
    *,
    device_name: str,
    output_dir: str | Path,
    formal_output_dir: str | Path,
) -> dict[str, Any]:
    output = resolve_project_path(output_dir) or Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "preflight_report.json"
    report: dict[str, Any] = {
        "protocol": D0_PREFLIGHT_PROTOCOL,
        "config_protocol": config.protocol,
        "stage": config.training.stage,
        "ready": False,
        "status": "invalid",
        "optimizer_steps": 0,
        "construction_device": str(device_name),
        "resolved_config": config.to_dict(),
        "resolved_config_sha256": canonical_sha256(config.to_dict()),
        "checks": {},
        "errors": [],
    }
    try:
        if config.protocol != SEGDESC_CONFIG_PROTOCOL:
            raise ValueError("D0 preflight 只接受 qpsalm_segdesc_config_v2")
        spec = get_stage_spec(config.training.stage)
        if spec.name != "mmrs_caption" or spec.requires_d_minus_one_gate is not True:
            raise ValueError("D0 preflight 必须使用 stage=mmrs_caption")
        if not config.training.d_minus_one_gate:
            raise ValueError("D0 preflight 缺少 d_minus_one_gate")
        report["stage_spec"] = spec.to_dict()

        d_minus_one = validate_d_minus_one_gate(
            config.training.d_minus_one_gate,
            expected_description_benchmark=config.data.description_benchmark,
            expected_bridge_benchmark=config.data.bridge_benchmark,
            expected_unified_benchmark=config.data.unified_benchmark,
            expected_description_cache=(
                config.model.description_vision_cache
            ),
        )
        report["d_minus_one_acceptance"] = d_minus_one
        report["checks"]["d_minus_one"] = True

        bank = DescriptionVisionFeatureBank(config.model.description_vision_cache)
        cache_replay = bank.verify_all_shards()
        cache_origin = revalidate_published_cache_origin(
            config.model.description_vision_cache, bank
        )
        report["description_cache"] = {
            "artifact_binding": bank.artifact_binding(),
            "shard_replay": cache_replay,
            "origin": cache_origin,
        }
        cache_origin_checks = dict(cache_origin.get("checks") or {})
        report["checks"]["description_cache"] = bool(
            cache_replay.get("all_verified")
            and cache_origin_checks
            and all(cache_origin_checks.values())
        )
        if report["checks"]["description_cache"] is not True:
            failed = sorted(
                name
                for name, passed in cache_origin_checks.items()
                if passed is not True
            )
            raise RuntimeError(
                "D0 preflight Description cache provenance 未通过: "
                f"origin_checks={failed} "
                f"all_shards_replayed={cache_replay.get('all_verified')!r}"
            )

        description_root = (
            resolve_project_path(config.data.description_benchmark)
            or Path(config.data.description_benchmark)
        )
        bridge_root = (
            resolve_project_path(config.data.bridge_benchmark)
            or Path(config.data.bridge_benchmark)
        )
        description_audit = require_engineering_description(
            description_root, bank
        )
        bridge_audit = require_engineering_bridge(bridge_root, bank)
        report["benchmark_bindings"] = {
            "description": description_audit,
            "bridge": bridge_audit,
            "expert_truth_used": False,
            "bridge_status": bridge_audit["status"],
        }
        report["checks"]["benchmark_bindings"] = bool(
            bridge_audit["status"]
            in {"awaiting_expert_review", "expert_pilot_frozen"}
            and bridge_audit.get("expert_truth_used") is False
        )
        if report["checks"]["benchmark_bindings"] is not True:
            raise RuntimeError(
                "D0 preflight Description/Bridge engineering binding 未通过: "
                f"bridge_status={bridge_audit.get('status')!r} "
                f"expert_truth_used={bridge_audit.get('expert_truth_used')!r}"
            )

        set_description_seed(config.training.seed)
        device = description_device(device_name)
        model, segmentation_migration = build_segdesc_model(config, device)
        report["segmentation_migration"] = segmentation_migration
        report["checks"]["segmentation_migration"] = True

        dataset = build_description_dataset(
            config, bank, split="train", training=True
        )
        dataset_audit = dataset_data_audit(dataset)
        if int(dataset_audit.get("num_samples", 0)) <= 0:
            raise ValueError("D0 training dataset 为空")
        loader = build_description_loader(
            dataset,
            config,
            training=True,
            sampler_seed=(
                int(config.training.seed)
                + DESCRIPTION_STREAM_SEED_OFFSETS["main"]
            ),
        )
        batch = next(iter(loader))
        report["dataset"] = dataset_audit
        report["collator"] = description_collator_audit(batch)
        stream_binding = description_stream_binding(
            "main",
            {"config": config, "dataset": dataset, "loader": loader},
            dataset_audit,
        )
        report["loader"] = {
            "batches": len(loader),
            "num_workers": int(config.data.num_workers),
            "batch_sampler": type(loader.batch_sampler).__name__,
            "stream_binding": stream_binding,
        }
        report["checks"]["dataset_and_collator"] = True

        optimizer, scheduler = build_description_optimizer(model, config)
        parameter_manifest = description_trainable_parameter_manifest(
            model,
            list(optimizer.param_groups),
            stage=config.training.stage,
        )
        adapter_groups = [
            group for group in parameter_manifest["groups"]
            if group["name"] == "desc_adapter"
        ]
        if len(adapter_groups) != 1 or int(adapter_groups[0]["numel"]) <= 0:
            raise ValueError("D0 optimizer 未证明 desc_adapter 独占分组")
        if any(
            ".default." in name
            for group in parameter_manifest["groups"]
            for name in group["parameter_names"]
        ):
            raise ValueError("D0 trainable manifest 泄漏 segmentation default adapter")
        report["trainable_parameters"] = parameter_manifest
        report["optimizer"] = description_optimizer_audit(
            optimizer, scheduler
        )
        if optimizer.state:
            raise ValueError("D0 preflight optimizer 在 step 前不应有 state")
        report["checks"]["optimizer_and_adapter_isolation"] = True

        config_path = resolve_project_path(config.model.segmentation_config)
        if config_path is None or not config_path.is_file():
            raise FileNotFoundError(
                "D0 segmentation config 不存在: "
                f"{config.model.segmentation_config}"
            )
        report["source_bindings"] = {
            "description_benchmark": str(
                (resolve_project_path(config.data.description_benchmark) or Path(
                    config.data.description_benchmark
                )).resolve(strict=False)
            ),
            "bridge_benchmark": str(
                (resolve_project_path(config.data.bridge_benchmark) or Path(
                    config.data.bridge_benchmark
                )).resolve(strict=False)
            ),
            "unified_benchmark": str(
                (resolve_project_path(config.data.unified_benchmark) or Path(
                    config.data.unified_benchmark
                )).resolve(strict=False)
            ),
            "segmentation_config": str(config_path.resolve(strict=False)),
            "segmentation_config_sha256": sha256_file(config_path),
        }
        formal_output = (
            resolve_project_path(formal_output_dir) or Path(formal_output_dir)
        ).resolve(strict=False)
        preflight_output = output.resolve(strict=False)
        if (
            formal_output == preflight_output
            or formal_output.is_relative_to(preflight_output)
            or preflight_output.is_relative_to(formal_output)
        ):
            raise ValueError(
                "D0 正式 output-dir 必须与 preflight output-dir 完全分离"
            )
        if formal_output.exists() and (
            not formal_output.is_dir() or any(formal_output.iterdir())
        ):
            raise ValueError(
                "D0 正式 output-dir 已存在且非空；预检拒绝发布会覆盖既有 run 的命令"
            )
        validate_output_replacement_safety(formal_output, {
            "segmentation-config": config.model.segmentation_config,
            "segmentation-checkpoint": config.model.segmentation_checkpoint,
            "segmentation-vision-cache": config.model.segmentation_vision_cache,
            "description-vision-cache": config.model.description_vision_cache,
            "description-benchmark": config.data.description_benchmark,
            "bridge-benchmark": config.data.bridge_benchmark,
            "unified-benchmark": config.data.unified_benchmark,
            "d-minus-one-gate": config.training.d_minus_one_gate,
        })
        resolved_config_path = output / "d0_resolved_config.json"
        atomic_write_json(resolved_config_path, config.to_dict())
        resolved_config_sha256 = sha256_file(resolved_config_path)
        report["formal_training_launch"] = build_d0_training_launch(
            python_executable=sys.executable,
            config_path=resolved_config_path,
            config_sha256=resolved_config_sha256,
            seed=config.training.seed,
            device_name=device_name,
            d_minus_one_gate=str(config.training.d_minus_one_gate),
            output_dir=formal_output,
            preflight_report=report_path,
        )
        report["checks"]["formal_training_launch"] = True
        report["checks"]["no_optimizer_step"] = True
        report["ready"] = all(report["checks"].values())
        report["status"] = "engineering-valid" if report["ready"] else "invalid"
    except Exception as exc:
        report["errors"].append({
            "type": type(exc).__name__,
            "message": str(exc),
        })
        report["ready"] = False
        report["status"] = "invalid"
    atomic_write_json(report_path, report)
    return report
