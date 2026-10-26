"""Qwen3.5 full-logits 与监督位置 logits 的真实 64-LoRA 梯度等价门。"""

from __future__ import annotations

import gc
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import torch
from safetensors.torch import load_file as load_safetensors
from torch import Tensor
import torch.nn.functional as F

from oa_groundrag.artifacts.identity import sha256_file
from oa_groundrag.artifacts.io import atomic_write_json, first_symlink_component
from oa_groundrag.data.grounded.supervision.compact_training import (
    CompactTrainingMessageDataset,
)
from oa_groundrag.data.rs_general.dataset import RSGeneralDescDataset
from oa_groundrag.data.rs_general.io import read_json
from oa_groundrag.training.vlm.cuda_telemetry import allocator_environment_identity
from oa_groundrag.training.vlm.qwen35_supervised import (
    wrap_qwen35_for_supervised_position_training,
)
from oa_groundrag.training.vlm.trainer import set_global_seed
from oa_groundrag.vlm.backends import build_model_adapter, build_processor_adapter
from oa_groundrag.vlm.data import ExternalDescriptionDataset
from oa_groundrag.vlm.errors import ModelError, ReasonCode
from oa_groundrag.vlm.preflight import open_benchmark_access
from oa_groundrag.vlm.processing import DescriptionCollator

from .config import STAGE5_CONFIG_SCHEMA_V4, load_stage5_config, verify_warm_start_files
from .data import (
    REGION_TRAIN_ROLE,
    RegionSubsetDataset,
    Stage5MixedDataset,
    split_compact_by_parent,
)
from .resource_gate import _selected_batches
from .resource_profile import verify_stage5_resource_profile


LOSS_PARITY_SCHEMA = "rs_vlm.qwen3_5_supervised_position_parity.v1"
LOSS_ATOL = 1e-4
LOSS_RTOL = 1e-3
GRADIENT_RELATIVE_L2_MAX = 1e-3
GRADIENT_COSINE_MIN = 0.9999
EXPECTED_LORA_GRADIENTS = 64


def _move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device=device, non_blocking=False)
        if isinstance(value, Tensor)
        else value
        for key, value in batch.items()
    }


def _load_warm_state(model: Any, processor_identity: Mapping[str, Any], config: Any) -> None:
    verify_warm_start_files(config)
    manifest = read_json(config.warm_start.checkpoint_root / "manifest.json")
    if (
        manifest.get("cursor", {}).get("global_step") != 1000
        or manifest.get("model_identity") != model.identity.to_dict()
        or manifest.get("processor_identity") != dict(processor_identity)
        or manifest.get("trainable_parameter_names") != list(model.trainable_names)
    ):
        raise ModelError(
            ReasonCode.CHECKPOINT_INCOMPATIBLE,
            "Qwen3.5 loss parity warm-start identity 不兼容",
        )
    model.load_trainable_state_dict(
        load_safetensors(
            str(
                config.warm_start.checkpoint_root
                / "adapter"
                / "adapter_model.safetensors"
            ),
            device="cpu",
        )
    )


def _gradient_vector(model: Any) -> Tensor:
    parameters = dict(model.model.named_parameters())
    gradients = []
    for name in model.trainable_names:
        parameter = parameters.get(name)
        if parameter is None or parameter.grad is None:
            raise ModelError(
                ReasonCode.NONFINITE_NUMBER,
                f"Qwen3.5 parity 缺少 LoRA gradient：{name}",
            )
        gradient = parameter.grad.detach().float()
        if not bool(torch.isfinite(gradient).all()):
            raise ModelError(
                ReasonCode.NONFINITE_NUMBER,
                f"Qwen3.5 parity LoRA gradient 非有限：{name}",
            )
        gradients.append(gradient.cpu().reshape(-1))
    if len(gradients) != EXPECTED_LORA_GRADIENTS:
        raise ModelError(
            ReasonCode.MODEL_IDENTITY_MISMATCH,
            "Qwen3.5 parity 必须精确比较 64 个 LoRA gradients",
        )
    return torch.cat(gradients)


def _clear_gradients(model: Any) -> None:
    for parameter in model.model.parameters():
        if parameter.requires_grad:
            parameter.grad = None


def run_qwen35_loss_parity(
    config_path: Path | str,
) -> dict[str, Any]:
    """在一条冻结视觉最坏样本上串行比较 full/projected loss 与梯度。"""

    config = load_stage5_config(config_path)
    resource = config.resource_contract
    if (
        config.schema_version != STAGE5_CONFIG_SCHEMA_V4
        or resource is None
        or config.model.backend != "qwen3_5"
        or resource.allocator_profile != "native"
    ):
        raise ModelError(
            ReasonCode.TYPE_MISMATCH,
            "loss parity 只接受 native Qwen3.5 M7 v4 配置",
        )
    allocator_environment_identity(resource.allocator_profile)
    verify_stage5_resource_profile(config)
    output_root = resource.loss_parity_root
    linked = first_symlink_component(output_root)
    if linked is not None or output_root.exists() or output_root.is_symlink():
        raise ModelError(
            ReasonCode.OUTPUT_EXISTS,
            f"loss parity 要求全新普通根：{output_root}",
        )
    output_root.mkdir(parents=True)
    if not torch.cuda.is_available():
        raise ModelError(ReasonCode.CUDA_REQUIRED, "Qwen3.5 loss parity 要求 CUDA")
    device = torch.device("cuda:0")
    processor = build_processor_adapter(config.base)
    processor_identity = processor.identity()
    compact = CompactTrainingMessageDataset(config.data_contract.compact_training_root)
    split = split_compact_by_parent(compact, seed=config.run.seed)
    access = open_benchmark_access(config.base)
    try:
        with tempfile.TemporaryDirectory(prefix="m7_loss_parity_rs_general_") as temporary:
            canonical = RSGeneralDescDataset(
                config.data.benchmark_root,
                roles=("external_train",),
                task_families=config.data.task_families,
                load_assets=False,
                seed=config.run.seed,
                expected_manifest_sha256=config.data.expected_manifest_sha256,
                verifier=access.verifier,
            )
            replay = ExternalDescriptionDataset(
                canonical,
                derived_root=Path(temporary) / "replay",
                seed=config.run.seed,
                roles=("external_train",),
            )
            region = RegionSubsetDataset(
                compact,
                split.train_indices,
                logical_role=REGION_TRAIN_ROLE,
            )
            mixed = Stage5MixedDataset(region, replay)
            selected = _selected_batches(
                config=config,
                dataset=mixed,
                collator=DescriptionCollator(processor, training=True),
            )
            item, cpu_batch = min(
                selected,
                key=lambda value: int(value[1]["input_token_counts"][0]),
            )

        model = build_model_adapter(
            config.base,
            device=device,
            gradient_checkpointing=True,
        )
        _load_warm_state(model, processor_identity, config)
        batch = _move_batch(cpu_batch, device)
        model.train()
        torch.cuda.reset_peak_memory_stats(device)

        set_global_seed(config.run.seed)
        full_result = model.forward(batch)
        full_loss = full_result.loss
        full_loss.backward()
        torch.cuda.synchronize(device)
        full_peak = int(torch.cuda.max_memory_allocated(device))
        full_vector = _gradient_vector(model)
        full_loss_value = float(full_loss.detach().cpu())
        del full_loss, full_result
        _clear_gradients(model)
        gc.collect()
        torch.cuda.empty_cache()

        set_global_seed(config.run.seed)
        torch.cuda.reset_peak_memory_stats(device)
        projected_model = wrap_qwen35_for_supervised_position_training(model)
        projected_result = projected_model.forward(batch)
        projected_loss = projected_result.loss
        projected_loss.backward()
        torch.cuda.synchronize(device)
        projected_peak = int(torch.cuda.max_memory_allocated(device))
        projected_vector = _gradient_vector(projected_model)
        projected_loss_value = float(projected_loss.detach().cpu())

        relative_l2 = float(
            torch.linalg.vector_norm(projected_vector - full_vector)
            / torch.linalg.vector_norm(full_vector)
        )
        cosine = float(F.cosine_similarity(full_vector, projected_vector, dim=0))
        loss_abs_diff = abs(projected_loss_value - full_loss_value)
        loss_close = math.isclose(
            projected_loss_value,
            full_loss_value,
            abs_tol=LOSS_ATOL,
            rel_tol=LOSS_RTOL,
        )
        passed = (
            loss_close
            and relative_l2 <= GRADIENT_RELATIVE_L2_MAX
            and cosine >= GRADIENT_COSINE_MIN
        )
        report = {
            "schema_version": LOSS_PARITY_SCHEMA,
            "status": "passed" if passed else "failed",
            "config_semantic_sha256": config.semantic_sha256,
            "model_identity": model.identity.to_dict(),
            "processor_identity": processor_identity,
            "warm_start": {
                "checkpoint_manifest_sha256": (
                    config.warm_start.checkpoint_manifest_sha256
                ),
                "adapter_sha256": config.warm_start.adapter_sha256,
            },
            "sample": {
                "sequence_index": item["sequence_index"],
                "record_id": item["record_id"],
                "reasons": item["reasons"],
                "input_tokens": int(cpu_batch["input_token_counts"][0]),
                "supervised_tokens": int(cpu_batch["labels"][:, 1:].ne(-100).sum()),
                "image_count": int(cpu_batch["image_counts"][0]),
                "pixel_shape": list(cpu_batch["pixel_values"].shape),
                "image_grid_thw": cpu_batch["image_grid_thw"].tolist(),
            },
            "loss": {
                "full": full_loss_value,
                "projected": projected_loss_value,
                "absolute_difference": loss_abs_diff,
                "atol": LOSS_ATOL,
                "rtol": LOSS_RTOL,
                "passed": loss_close,
            },
            "gradients": {
                "tensor_count": len(model.trainable_names),
                "parameter_count": int(full_vector.numel()),
                "aggregate_relative_l2": relative_l2,
                "aggregate_cosine": cosine,
                "relative_l2_max": GRADIENT_RELATIVE_L2_MAX,
                "cosine_min": GRADIENT_COSINE_MIN,
            },
            "memory": {
                "full_peak_allocated_bytes": full_peak,
                "projected_peak_allocated_bytes": projected_peak,
                "saved_peak_allocated_bytes": full_peak - projected_peak,
            },
            "formal_acceptance": False,
            "scientific_acceptance": False,
        }
        atomic_write_json(output_root / "loss_parity_report.json", report)
        if not passed:
            raise ModelError(
                ReasonCode.NONFINITE_NUMBER,
                "Qwen3.5 监督位置 loss/LoRA gradient 等价门失败",
                details={"report": report},
            )
        del projected_loss, projected_result, projected_vector, full_vector, batch
        del projected_model, model
        gc.collect()
        torch.cuda.empty_cache()
        return {
            **report,
            "root": str(output_root),
            "report_sha256": sha256_file(
                output_root / "loss_parity_report.json"
            ),
        }
    except BaseException as error:
        failure_path = output_root / "loss_parity_failure.json"
        if not failure_path.exists() and not failure_path.is_symlink():
            atomic_write_json(
                failure_path,
                {
                    "schema_version": LOSS_PARITY_SCHEMA,
                    "status": "failed",
                    "exception_type": type(error).__name__,
                    "exception_message": str(error),
                    "config_semantic_sha256": config.semantic_sha256,
                },
            )
        raise


def verify_qwen35_loss_parity(config: Any) -> dict[str, Any]:
    resource = config.resource_contract
    if resource is None:
        raise ModelError(ReasonCode.TYPE_MISMATCH, "非 v4 配置没有 loss parity")
    root = resource.loss_parity_root
    linked = first_symlink_component(root)
    report_path = root / "loss_parity_report.json"
    if (
        linked is not None
        or not root.is_dir()
        or root.is_symlink()
        or not report_path.is_file()
        or report_path.is_symlink()
        or (root / "loss_parity_failure.json").exists()
    ):
        raise ModelError(
            ReasonCode.CHECKPOINT_CORRUPT,
            "Qwen3.5 loss parity 根/报告非法",
        )
    report = read_json(report_path)
    gradients = report.get("gradients", {})
    loss = report.get("loss", {})
    if (
        report.get("schema_version") != LOSS_PARITY_SCHEMA
        or report.get("status") != "passed"
        or report.get("model_identity", {}).get("backend") != "qwen3_5"
        or report.get("warm_start", {}).get("adapter_sha256")
        != config.warm_start.adapter_sha256
        or gradients.get("tensor_count") != EXPECTED_LORA_GRADIENTS
        or float(gradients.get("aggregate_relative_l2", math.inf))
        > GRADIENT_RELATIVE_L2_MAX
        or float(gradients.get("aggregate_cosine", -math.inf))
        < GRADIENT_COSINE_MIN
        or loss.get("passed") is not True
        or report.get("formal_acceptance") is not False
        or report.get("scientific_acceptance") is not False
    ):
        raise ModelError(
            ReasonCode.CHECKPOINT_CORRUPT,
            "Qwen3.5 loss parity 报告未通过严格验证",
        )
    return {
        "root": str(root),
        "report_sha256": sha256_file(report_path),
        "loss_absolute_difference": loss["absolute_difference"],
        "gradient_relative_l2": gradients["aggregate_relative_l2"],
        "gradient_cosine": gradients["aggregate_cosine"],
    }

