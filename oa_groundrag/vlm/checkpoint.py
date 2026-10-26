"""推理所需 checkpoint 与既有训练产物的只读身份验证。"""

from __future__ import annotations

import pickle
import random
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from safetensors import SafetensorError
from safetensors.torch import load_file as load_safetensors
from safetensors.torch import save_file as save_safetensors
from torch import Tensor

from oa_groundrag.artifacts.identity import (
    canonical_json,
    sha256_file,
    sha256_text,
)
from oa_groundrag.artifacts.io import (
    atomic_write_json,
    first_symlink_component,
)
from oa_groundrag.data.rs_general.io import read_json
from oa_groundrag.data.rs_general.errors import RSGeneralDescError

from oa_groundrag.grounding.contracts import (
    CHECKPOINT_SCHEMA_VERSION,
    SAMPLE_TRACE_SCHEMA_VERSION,
)
from .errors import CheckpointError, ReasonCode
from oa_groundrag.training.vlm.input_pipeline import (
    INPUT_PIPELINE_PREFETCH,
    INPUT_PIPELINE_SYNC,
)


_CHECKPOINT_MANIFEST_FIELDS = {
    "schema_version",
    "config_semantic_sha256",
    "config_snapshot_sha256",
    "benchmark_identity",
    "validation_selection_identity",
    "model_identity",
    "processor_identity",
    "training_layout",
    "trainable_parameter_names",
    "trainable_parameter_count",
    "cursor",
    "multireference_epoch",
    "files",
}
INFERENCE_CHECKPOINT_SCHEMA_VERSION = "rs_vlm.checkpoint.v2"
_INFERENCE_CHECKPOINT_MANIFEST_FIELDS = {
    "schema_version",
    "config_semantic_sha256",
    "benchmark_identity",
    "validation_selection_identity",
    "model_identity",
    "processor_identity",
    "training_layout",
    "trainable_parameter_names",
    "trainable_parameter_count",
    "cursor",
    "selection_metrics",
    "files",
}
_BASE_TRAINING_LAYOUT_FIELDS = {
    "physical_batch_size",
    "gradient_accumulation_steps",
    "effective_batch_size",
    "input_pipeline_backend",
    "num_workers",
    "prefetch_factor",
    "pin_memory",
    "sample_trace_schema_version",
}
_CUDA_CACHE_LAYOUT_FIELD = "cuda_cache_cleanup_interval_steps"
_CUDA_RESOURCE_LAYOUT_FIELD = "cuda_resource_identity"


def _validated_training_layout(
    value: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    fields = set(value) if isinstance(value, Mapping) else set()
    allowed_fields = {
        frozenset(_BASE_TRAINING_LAYOUT_FIELDS),
        frozenset((*_BASE_TRAINING_LAYOUT_FIELDS, _CUDA_CACHE_LAYOUT_FIELD)),
        frozenset(
            (
                *_BASE_TRAINING_LAYOUT_FIELDS,
                _CUDA_CACHE_LAYOUT_FIELD,
                _CUDA_RESOURCE_LAYOUT_FIELD,
            )
        ),
    }
    if not isinstance(value, Mapping) or frozenset(fields) not in allowed_fields:
        raise CheckpointError(
            ReasonCode.CHECKPOINT_INCOMPATIBLE,
            f"{label} training_layout 字段不匹配",
        )
    result = dict(value)
    for field in (
        "physical_batch_size",
        "gradient_accumulation_steps",
        "effective_batch_size",
    ):
        item = result[field]
        if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
            raise CheckpointError(
                ReasonCode.CHECKPOINT_INCOMPATIBLE,
                f"{label} training_layout.{field} 必须是正整数",
            )
    for field in ("num_workers", "prefetch_factor"):
        item = result[field]
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise CheckpointError(
                ReasonCode.CHECKPOINT_INCOMPATIBLE,
                f"{label} training_layout.{field} 必须是非负整数",
            )
    if not isinstance(result["pin_memory"], bool):
        raise CheckpointError(
            ReasonCode.CHECKPOINT_INCOMPATIBLE,
            f"{label} training_layout.pin_memory 必须是 bool",
        )
    if _CUDA_CACHE_LAYOUT_FIELD in result:
        interval = result[_CUDA_CACHE_LAYOUT_FIELD]
        if isinstance(interval, bool) or not isinstance(interval, int) or interval <= 0:
            raise CheckpointError(
                ReasonCode.CHECKPOINT_INCOMPATIBLE,
                f"{label} training_layout.{_CUDA_CACHE_LAYOUT_FIELD} 必须是正整数",
            )
    if _CUDA_RESOURCE_LAYOUT_FIELD in result and (
        not isinstance(result[_CUDA_RESOURCE_LAYOUT_FIELD], Mapping)
        or not result[_CUDA_RESOURCE_LAYOUT_FIELD]
    ):
        raise CheckpointError(
            ReasonCode.CHECKPOINT_INCOMPATIBLE,
            f"{label} training_layout.{_CUDA_RESOURCE_LAYOUT_FIELD} 必须是非空对象",
        )
    backend = result["input_pipeline_backend"]
    if (
        result["effective_batch_size"]
        != result["physical_batch_size"]
        * result["gradient_accumulation_steps"]
        or backend not in {INPUT_PIPELINE_SYNC, INPUT_PIPELINE_PREFETCH}
        or (
            backend == INPUT_PIPELINE_SYNC
            and (
                result["num_workers"] != 0
                or result["prefetch_factor"] != 0
                or result["pin_memory"]
            )
        )
        or (
            backend == INPUT_PIPELINE_PREFETCH
            and (
                result["num_workers"] <= 0
                or result["prefetch_factor"] <= 0
                or not result["pin_memory"]
            )
        )
        or result["sample_trace_schema_version"]
        != SAMPLE_TRACE_SCHEMA_VERSION
    ):
        raise CheckpointError(
            ReasonCode.CHECKPOINT_INCOMPATIBLE,
            f"{label} training_layout batch/input pipeline/trace 不兼容",
        )
    return result


def _read_checkpoint_json(path: Path) -> Any:
    try:
        return read_json(path)
    except RSGeneralDescError as error:
        raise CheckpointError(
            ReasonCode.CHECKPOINT_CORRUPT,
            f"checkpoint JSON 无法严格读取：{path.name}",
            details={"error": str(error)},
        ) from error


@dataclass(frozen=True)
class TrainingCursor:
    epoch: int
    sample_offset: int
    global_step: int
    micro_step: int

    def __post_init__(self) -> None:
        for name in ("epoch", "sample_offset", "global_step", "micro_step"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise CheckpointError(
                    ReasonCode.TYPE_MISMATCH,
                    f"TrainingCursor.{name} 必须是 >= 0 的整数",
                )

    def to_dict(self) -> dict[str, int]:
        return {
            "epoch": self.epoch,
            "sample_offset": self.sample_offset,
            "global_step": self.global_step,
            "micro_step": self.micro_step,
        }


@dataclass(frozen=True)
class CheckpointPayload:
    root: Path
    manifest: Mapping[str, Any]
    config_snapshot: Mapping[str, Any]
    trainable_state: Mapping[str, Tensor]
    optimizer_state: Mapping[str, Any]
    scheduler_state: Mapping[str, Any]
    rng_state: Mapping[str, Any]
    cursor: TrainingCursor
    sampler_state: Mapping[str, Any]


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        ),
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    expected = {"python", "numpy", "torch_cpu", "torch_cuda"}
    if set(state) != expected:
        raise CheckpointError(
            ReasonCode.CHECKPOINT_CORRUPT,
            "RNG state 字段不匹配",
        )
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    cuda_states = state["torch_cuda"]
    if cuda_states:
        if not torch.cuda.is_available():
            raise CheckpointError(
                ReasonCode.CHECKPOINT_INCOMPATIBLE,
                "checkpoint 含 CUDA RNG，但当前 CUDA 不可用",
            )
        torch.cuda.set_rng_state_all(cuda_states)


class CheckpointManager:
    def save(
        self,
        root: Path,
        *,
        trainable_state: Mapping[str, Tensor],
        cursor: TrainingCursor,
        config_semantic_sha256: str,
        benchmark_identity: Mapping[str, Any],
        validation_selection_identity: Mapping[str, Any],
        model_identity: Mapping[str, Any],
        processor_identity: Mapping[str, Any],
        training_layout: Mapping[str, Any],
        trainable_names: tuple[str, ...],
        selection_metrics: Mapping[str, Any] | None = None,
    ) -> Path:
        root = Path(root)
        normalized_training_layout = _validated_training_layout(
            training_layout,
            label="待保存 checkpoint",
        )
        linked = first_symlink_component(root)
        if linked is not None:
            raise CheckpointError(
                ReasonCode.OUTPUT_LINK,
                f"checkpoint root 含链接组件：{linked}",
            )
        if root.exists() or root.is_symlink():
            raise CheckpointError(
                ReasonCode.OUTPUT_EXISTS,
                f"checkpoint root 已存在：{root}",
            )
        if tuple(sorted(trainable_state)) != tuple(sorted(trainable_names)):
            raise CheckpointError(
                ReasonCode.CHECKPOINT_INCOMPATIBLE,
                "trainable state/name 清单不一致",
            )
        root.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(prefix=f".{root.name}.staging-", dir=root.parent)
        )
        try:
            adapter_dir = staging / "adapter"
            adapter_dir.mkdir()
            tensors = {
                name: value.detach().cpu().contiguous()
                for name, value in trainable_state.items()
            }
            if not tensors:
                raise CheckpointError(
                    ReasonCode.CHECKPOINT_INCOMPATIBLE,
                    "训练 checkpoint 不接受空 trainable state",
                )
            save_safetensors(
                tensors,
                str(adapter_dir / "adapter_model.safetensors"),
            )
            payload_files = sorted(
                path
                for path in staging.rglob("*")
                if path.is_file() and path.name != "manifest.json"
            )
            files = {
                path.relative_to(staging).as_posix(): {
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in payload_files
            }
            manifest = {
                "schema_version": INFERENCE_CHECKPOINT_SCHEMA_VERSION,
                "config_semantic_sha256": config_semantic_sha256,
                "benchmark_identity": dict(benchmark_identity),
                "validation_selection_identity": dict(
                    validation_selection_identity
                ),
                "model_identity": dict(model_identity),
                "processor_identity": dict(processor_identity),
                "training_layout": normalized_training_layout,
                "trainable_parameter_names": list(trainable_names),
                "trainable_parameter_count": int(
                    sum(value.numel() for value in tensors.values())
                ),
                "cursor": cursor.to_dict(),
                "selection_metrics": dict(selection_metrics or {}),
                "files": files,
            }
            atomic_write_json(staging / "manifest.json", manifest)
            if root.exists() or root.is_symlink():
                raise CheckpointError(
                    ReasonCode.OUTPUT_EXISTS,
                    f"发布前 checkpoint root 已存在：{root}",
                )
            staging.replace(root)
            return root
        except BaseException:
            if staging.exists():
                shutil.rmtree(staging)
            raise

    def load(
        self,
        root: Path,
        *,
        expected_config_semantic_sha256: str,
        expected_benchmark_identity: Mapping[str, Any],
        expected_validation_selection_identity: (
            Mapping[str, Any] | None
        ),
        expected_model_identity: Mapping[str, Any],
        expected_processor_identity: Mapping[str, Any],
        expected_training_layout: Mapping[str, Any],
        expected_trainable_names: tuple[str, ...],
    ) -> CheckpointPayload:
        root = Path(root)
        linked = first_symlink_component(root)
        if linked is not None:
            raise CheckpointError(
                ReasonCode.OUTPUT_LINK,
                f"checkpoint root 含链接组件：{linked}",
            )
        if root.is_symlink() or not root.is_dir():
            raise CheckpointError(
                ReasonCode.CHECKPOINT_CORRUPT,
                f"checkpoint root 不存在或是链接：{root}",
            )
        manifest = _read_checkpoint_json(self._file(root, "manifest.json"))
        if not isinstance(manifest, dict) or set(manifest) != (
            _CHECKPOINT_MANIFEST_FIELDS
        ):
            raise CheckpointError(
                ReasonCode.CHECKPOINT_CORRUPT,
                "checkpoint manifest 字段不匹配",
            )
        if manifest.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise CheckpointError(
                ReasonCode.CHECKPOINT_INCOMPATIBLE,
                "checkpoint schema 不兼容",
            )
        expected_layout = _validated_training_layout(
            expected_training_layout,
            label="当前运行",
        )
        saved_layout = _validated_training_layout(
            manifest.get("training_layout"),
            label="checkpoint",
        )
        comparisons = {
            "config_semantic_sha256": (
                expected_config_semantic_sha256,
                manifest.get("config_semantic_sha256"),
            ),
            "benchmark_identity": (
                dict(expected_benchmark_identity),
                manifest.get("benchmark_identity"),
            ),
            "model_identity": (
                dict(expected_model_identity),
                manifest.get("model_identity"),
            ),
            "processor_identity": (
                dict(expected_processor_identity),
                manifest.get("processor_identity"),
            ),
            "training_layout": (
                expected_layout,
                saved_layout,
            ),
            "trainable_parameter_names": (
                list(expected_trainable_names),
                manifest.get("trainable_parameter_names"),
            ),
        }
        validation_identity = manifest.get("validation_selection_identity")
        if not isinstance(validation_identity, dict) or not validation_identity:
            raise CheckpointError(
                ReasonCode.CHECKPOINT_CORRUPT,
                "checkpoint validation selection identity 非法",
            )
        if expected_validation_selection_identity is not None:
            comparisons["validation_selection_identity"] = (
                dict(expected_validation_selection_identity),
                validation_identity,
            )
        mismatches = {
            name: {"expected": expected, "actual": actual}
            for name, (expected, actual) in comparisons.items()
            if expected != actual
        }
        if mismatches:
            raise CheckpointError(
                ReasonCode.CHECKPOINT_INCOMPATIBLE,
                "checkpoint identity/config/trainable list 不兼容",
                details=mismatches,
            )
        files = manifest.get("files")
        if not isinstance(files, dict):
            raise CheckpointError(
                ReasonCode.CHECKPOINT_CORRUPT,
                "checkpoint manifest.files 非法",
            )
        linked_payload = next(
            (
                path
                for path in root.rglob("*")
                if path.is_symlink()
            ),
            None,
        )
        if linked_payload is not None:
            raise CheckpointError(
                ReasonCode.OUTPUT_LINK,
                f"checkpoint payload 含链接：{linked_payload}",
            )
        actual_files = {
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file() and path.name != "manifest.json"
        }
        if actual_files != set(files):
            raise CheckpointError(
                ReasonCode.CHECKPOINT_CORRUPT,
                "checkpoint payload 文件集与 manifest 不一致",
                details={
                    "missing": sorted(set(files) - actual_files),
                    "unexpected": sorted(actual_files - set(files)),
                },
            )
        for relative, contract in files.items():
            if (
                not isinstance(relative, str)
                or not isinstance(contract, dict)
                or set(contract) != {"size_bytes", "sha256"}
                or isinstance(contract.get("size_bytes"), bool)
                or not isinstance(contract.get("size_bytes"), int)
                or contract["size_bytes"] < 0
                or not isinstance(contract.get("sha256"), str)
                or len(contract["sha256"]) != 64
            ):
                raise CheckpointError(
                    ReasonCode.CHECKPOINT_CORRUPT,
                    "checkpoint file contract 非法",
                )
            path = self._file(root, relative)
            if (
                path.stat().st_size != contract["size_bytes"]
                or sha256_file(path) != contract["sha256"]
            ):
                raise CheckpointError(
                    ReasonCode.CHECKPOINT_CORRUPT,
                    f"checkpoint file hash/size 不一致：{relative}",
                )
        config_snapshot = _read_checkpoint_json(
            self._file(root, "config_snapshot.json")
        )
        if not isinstance(config_snapshot, dict):
            raise CheckpointError(
                ReasonCode.CHECKPOINT_CORRUPT,
                "config snapshot 必须是对象",
            )
        if sha256_text(canonical_json(config_snapshot)) != manifest.get(
            "config_snapshot_sha256"
        ):
            raise CheckpointError(
                ReasonCode.CHECKPOINT_CORRUPT,
                "config snapshot hash 不一致",
            )
        training_state = _read_checkpoint_json(
            self._file(root, "training_state.json")
        )
        if not isinstance(training_state, dict) or set(training_state) != {
            "cursor",
            "sampler_state",
            "multireference_epoch",
        }:
            raise CheckpointError(
                ReasonCode.CHECKPOINT_CORRUPT,
                "training_state 字段不匹配",
            )
        if not isinstance(training_state["sampler_state"], dict):
            raise CheckpointError(
                ReasonCode.CHECKPOINT_CORRUPT,
                "sampler_state 必须是对象",
            )
        multireference_epoch = training_state["multireference_epoch"]
        if (
            isinstance(multireference_epoch, bool)
            or not isinstance(multireference_epoch, int)
            or multireference_epoch < 0
        ):
            raise CheckpointError(
                ReasonCode.CHECKPOINT_CORRUPT,
                "multireference_epoch 必须是 >= 0 的整数",
            )
        cursor_row = training_state["cursor"]
        if not isinstance(cursor_row, dict) or set(cursor_row) != {
            "epoch",
            "sample_offset",
            "global_step",
            "micro_step",
        }:
            raise CheckpointError(
                ReasonCode.CHECKPOINT_CORRUPT,
                "cursor 字段不匹配",
            )
        cursor = TrainingCursor(**cursor_row)
        if cursor.to_dict() != manifest.get("cursor"):
            raise CheckpointError(
                ReasonCode.CHECKPOINT_CORRUPT,
                "manifest/training_state cursor 不一致",
            )
        if training_state["multireference_epoch"] != manifest.get(
            "multireference_epoch"
        ):
            raise CheckpointError(
                ReasonCode.CHECKPOINT_CORRUPT,
                "multireference epoch 不一致",
            )
        try:
            trainable = load_safetensors(
                str(self._file(root, "adapter/adapter_model.safetensors"))
            )
            optimizer = torch.load(
                self._file(root, "optimizer.pt"),
                map_location="cpu",
                weights_only=False,
            )
            scheduler = torch.load(
                self._file(root, "scheduler.pt"),
                map_location="cpu",
                weights_only=False,
            )
            rng = torch.load(
                self._file(root, "rng.pt"),
                map_location="cpu",
                weights_only=False,
            )
        except (
            EOFError,
            OSError,
            RuntimeError,
            SafetensorError,
            ValueError,
            pickle.UnpicklingError,
        ) as error:
            raise CheckpointError(
                ReasonCode.CHECKPOINT_CORRUPT,
                "checkpoint tensor/state 无法读取",
                details={"error": str(error)},
            ) from error
        if not all(
            isinstance(value, dict)
            for value in (optimizer, scheduler, rng)
        ):
            raise CheckpointError(
                ReasonCode.CHECKPOINT_CORRUPT,
                "optimizer/scheduler/RNG state 必须是对象",
            )
        if set(trainable) != set(expected_trainable_names):
            raise CheckpointError(
                ReasonCode.CHECKPOINT_INCOMPATIBLE,
                "adapter safetensors 参数名不兼容",
            )
        actual_parameter_count = sum(value.numel() for value in trainable.values())
        manifest_parameter_count = manifest["trainable_parameter_count"]
        if (
            isinstance(manifest_parameter_count, bool)
            or not isinstance(manifest_parameter_count, int)
            or manifest_parameter_count != actual_parameter_count
        ):
            raise CheckpointError(
                ReasonCode.CHECKPOINT_CORRUPT,
                "checkpoint trainable parameter count 不一致",
            )
        return CheckpointPayload(
            root=root,
            manifest=manifest,
            config_snapshot=config_snapshot,
            trainable_state=trainable,
            optimizer_state=optimizer,
            scheduler_state=scheduler,
            rng_state=rng,
            cursor=cursor,
            sampler_state=training_state["sampler_state"],
        )

    def inspect_trainable(
        self,
        root: Path,
        *,
        expected_config_semantic_sha256: str,
        expected_benchmark_identity: Mapping[str, Any],
        expected_validation_selection_identity: Mapping[str, Any] | None,
        expected_model_identity: Mapping[str, Any],
        expected_processor_identity: Mapping[str, Any],
        expected_training_layout: Mapping[str, Any],
        expected_trainable_names: tuple[str, ...],
    ) -> Mapping[str, Any]:
        """只读校验 inference 所需身份与 payload；不反序列化任何训练状态。"""

        root = Path(root)
        linked = first_symlink_component(root)
        if linked is not None or root.is_symlink() or not root.is_dir():
            raise CheckpointError(
                ReasonCode.OUTPUT_LINK,
                f"checkpoint root 不存在或含链接：{root}",
            )
        manifest = _read_checkpoint_json(self._file(root, "manifest.json"))
        schema_version = manifest.get("schema_version") if isinstance(manifest, dict) else None
        expected_fields = (
            _CHECKPOINT_MANIFEST_FIELDS
            if schema_version == CHECKPOINT_SCHEMA_VERSION
            else _INFERENCE_CHECKPOINT_MANIFEST_FIELDS
        )
        if not isinstance(manifest, dict) or set(manifest) != expected_fields:
            raise CheckpointError(
                ReasonCode.CHECKPOINT_CORRUPT,
                "checkpoint manifest 字段不匹配",
            )
        if schema_version not in {
            CHECKPOINT_SCHEMA_VERSION,
            INFERENCE_CHECKPOINT_SCHEMA_VERSION,
        }:
            raise CheckpointError(
                ReasonCode.CHECKPOINT_INCOMPATIBLE,
                "checkpoint schema 不兼容",
            )
        expected_layout = _validated_training_layout(
            expected_training_layout,
            label="当前运行",
        )
        saved_layout = _validated_training_layout(
            manifest.get("training_layout"),
            label="checkpoint",
        )
        validation_identity = manifest.get("validation_selection_identity")
        if not isinstance(validation_identity, dict) or not validation_identity:
            raise CheckpointError(
                ReasonCode.CHECKPOINT_CORRUPT,
                "checkpoint validation selection identity 非法",
            )
        comparisons = {
            "config_semantic_sha256": expected_config_semantic_sha256,
            "benchmark_identity": dict(expected_benchmark_identity),
            "model_identity": dict(expected_model_identity),
            "processor_identity": dict(expected_processor_identity),
            "training_layout": expected_layout,
            "trainable_parameter_names": list(expected_trainable_names),
        }
        if expected_validation_selection_identity is not None:
            comparisons["validation_selection_identity"] = dict(
                expected_validation_selection_identity
            )
        actual = {
            key: saved_layout if key == "training_layout" else manifest.get(key)
            for key in comparisons
        }
        mismatches = {
            key: {"expected": value, "actual": actual[key]}
            for key, value in comparisons.items()
            if value != actual[key]
        }
        if mismatches:
            raise CheckpointError(
                ReasonCode.CHECKPOINT_INCOMPATIBLE,
                "checkpoint inference identity 不兼容",
                details=mismatches,
            )
        files = manifest.get("files")
        if not isinstance(files, dict):
            raise CheckpointError(
                ReasonCode.CHECKPOINT_CORRUPT,
                "checkpoint manifest.files 非法",
            )
        actual_files = {
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file() and path.name != "manifest.json"
        }
        if any(path.is_symlink() for path in root.rglob("*")) or actual_files != set(files):
            raise CheckpointError(
                ReasonCode.CHECKPOINT_CORRUPT,
                "checkpoint payload 文件集或链接状态不合法",
            )
        for relative, contract in files.items():
            if (
                not isinstance(relative, str)
                or not isinstance(contract, dict)
                or set(contract) != {"size_bytes", "sha256"}
            ):
                raise CheckpointError(
                    ReasonCode.CHECKPOINT_CORRUPT,
                    "checkpoint file contract 非法",
                )
            path = self._file(root, relative)
            if (
                path.stat().st_size != contract.get("size_bytes")
                or sha256_file(path) != contract.get("sha256")
            ):
                raise CheckpointError(
                    ReasonCode.CHECKPOINT_CORRUPT,
                    f"checkpoint file hash/size 不一致：{relative}",
                )
        return manifest

    def load_trainable(
        self,
        root: Path,
        *,
        expected_config_semantic_sha256: str,
        expected_benchmark_identity: Mapping[str, Any],
        expected_validation_selection_identity: Mapping[str, Any] | None,
        expected_model_identity: Mapping[str, Any],
        expected_processor_identity: Mapping[str, Any],
        expected_training_layout: Mapping[str, Any],
        expected_trainable_names: tuple[str, ...],
    ) -> tuple[Mapping[str, Any], Mapping[str, Tensor]]:
        """校验后仅加载 LoRA tensor；不读取 optimizer、scheduler 或 RNG。"""

        keyword = {
            "expected_config_semantic_sha256": expected_config_semantic_sha256,
            "expected_benchmark_identity": expected_benchmark_identity,
            "expected_validation_selection_identity": expected_validation_selection_identity,
            "expected_model_identity": expected_model_identity,
            "expected_processor_identity": expected_processor_identity,
            "expected_training_layout": expected_training_layout,
            "expected_trainable_names": expected_trainable_names,
        }
        manifest = self.inspect_trainable(root, **keyword)
        try:
            trainable = load_safetensors(
                str(self._file(Path(root), "adapter/adapter_model.safetensors")),
                device="cpu",
            )
        except (OSError, RuntimeError, SafetensorError, ValueError) as error:
            raise CheckpointError(
                ReasonCode.CHECKPOINT_CORRUPT,
                "adapter safetensors 无法读取",
                details={"error": str(error)},
            ) from error
        if set(trainable) != set(expected_trainable_names):
            raise CheckpointError(
                ReasonCode.CHECKPOINT_INCOMPATIBLE,
                "adapter safetensors 参数名不兼容",
            )
        parameter_count = sum(value.numel() for value in trainable.values())
        if parameter_count != manifest.get("trainable_parameter_count"):
            raise CheckpointError(
                ReasonCode.CHECKPOINT_CORRUPT,
                "adapter 参数量与 manifest 不一致",
            )
        return manifest, trainable

    def load_trainable_for_inference(
        self,
        root: Path,
        *,
        expected_manifest_sha256: str,
        expected_adapter_sha256: str,
        expected_config_semantic_sha256: str,
        expected_model_identity: Mapping[str, Any],
        expected_processor_identity: Mapping[str, Any],
        expected_trainable_names: tuple[str, ...],
        expected_trainable_parameter_count: int,
    ) -> tuple[Mapping[str, Any], Mapping[str, Tensor]]:
        """按发布 manifest 身份只加载 LoRA，不重放训练数据合同。

        benchmark、monitor selection 与 training layout 已包含在精确 SHA 锚定的
        checkpoint manifest 中。推理重新核验当前模型、processor、LoRA 参数名、
        Adapter hash/size 和参数量；optimizer、scheduler、RNG、sampler、Benchmark、
        compact supervision 与评价资产均不会被打开。
        """

        root = Path(root)
        linked = first_symlink_component(root)
        if linked is not None or root.is_symlink() or not root.is_dir():
            raise CheckpointError(
                ReasonCode.OUTPUT_LINK,
                f"checkpoint root 不存在或含链接：{root}",
            )
        manifest_path = self._file(root, "manifest.json")
        if sha256_file(manifest_path) != expected_manifest_sha256:
            raise CheckpointError(
                ReasonCode.CHECKPOINT_INCOMPATIBLE,
                "checkpoint manifest SHA-256 与发布绑定不一致",
            )
        manifest = _read_checkpoint_json(manifest_path)
        schema_version = manifest.get("schema_version") if isinstance(manifest, dict) else None
        expected_fields = (
            _CHECKPOINT_MANIFEST_FIELDS
            if schema_version == CHECKPOINT_SCHEMA_VERSION
            else _INFERENCE_CHECKPOINT_MANIFEST_FIELDS
        )
        if not isinstance(manifest, dict) or set(manifest) != expected_fields:
            raise CheckpointError(
                ReasonCode.CHECKPOINT_CORRUPT,
                "checkpoint manifest 字段不匹配",
            )
        if schema_version not in {
            CHECKPOINT_SCHEMA_VERSION,
            INFERENCE_CHECKPOINT_SCHEMA_VERSION,
        }:
            raise CheckpointError(
                ReasonCode.CHECKPOINT_INCOMPATIBLE,
                "checkpoint schema 不兼容",
            )
        benchmark_identity = manifest.get("benchmark_identity")
        validation_identity = manifest.get("validation_selection_identity")
        training_layout = manifest.get("training_layout")
        if (
            not isinstance(benchmark_identity, dict)
            or not benchmark_identity
            or not isinstance(validation_identity, dict)
            or not validation_identity
            or not isinstance(training_layout, dict)
            or not training_layout
        ):
            raise CheckpointError(
                ReasonCode.CHECKPOINT_CORRUPT,
                "checkpoint 发布 manifest 缺少训练数据/selection/layout 身份",
            )
        saved_layout = _validated_training_layout(training_layout, label="checkpoint")
        if (
            isinstance(expected_trainable_parameter_count, bool)
            or not isinstance(expected_trainable_parameter_count, int)
            or expected_trainable_parameter_count <= 0
        ):
            raise CheckpointError(
                ReasonCode.CHECKPOINT_INCOMPATIBLE,
                "当前模型可训练参数量必须是正整数",
            )
        files = manifest.get("files")
        adapter_relative = "adapter/adapter_model.safetensors"
        legacy_training_files = {
            "optimizer.pt",
            "scheduler.pt",
            "rng.pt",
            "config_snapshot.json",
            "training_state.json",
        }
        expected_payload_files = (
            {adapter_relative, *legacy_training_files}
            if schema_version == CHECKPOINT_SCHEMA_VERSION
            else {adapter_relative}
        )
        if (
            not isinstance(files, dict)
            or set(files) != expected_payload_files
        ):
            raise CheckpointError(
                ReasonCode.CHECKPOINT_CORRUPT,
                "checkpoint manifest payload 清单与已发布合同不一致",
            )
        expected = {
            "config_semantic_sha256": expected_config_semantic_sha256,
            "model_identity": dict(expected_model_identity),
            "processor_identity": dict(expected_processor_identity),
            "training_layout": saved_layout,
            "trainable_parameter_names": list(expected_trainable_names),
        }
        mismatches = {
            key: {"expected": value, "actual": manifest.get(key)}
            for key, value in expected.items()
            if manifest.get(key) != value
        }
        if mismatches:
            raise CheckpointError(
                ReasonCode.CHECKPOINT_INCOMPATIBLE,
                "checkpoint inference identity 不兼容",
                details=mismatches,
            )
        adapter_contract = files.get(adapter_relative)
        if (
            not isinstance(adapter_contract, dict)
            or set(adapter_contract) != {"size_bytes", "sha256"}
            or adapter_contract.get("sha256") != expected_adapter_sha256
        ):
            raise CheckpointError(
                ReasonCode.CHECKPOINT_CORRUPT,
                "checkpoint manifest 的 Adapter 发布合同非法",
            )
        adapter_path = self._file(root, adapter_relative)
        if (
            adapter_path.stat().st_size != adapter_contract.get("size_bytes")
            or sha256_file(adapter_path) != expected_adapter_sha256
        ):
            raise CheckpointError(
                ReasonCode.CHECKPOINT_CORRUPT,
                "Adapter hash/size 与发布合同不一致",
            )
        try:
            trainable = load_safetensors(str(adapter_path), device="cpu")
        except (OSError, RuntimeError, SafetensorError, ValueError) as error:
            raise CheckpointError(
                ReasonCode.CHECKPOINT_CORRUPT,
                "adapter safetensors 无法读取",
                details={"error": str(error)},
            ) from error
        if set(trainable) != set(expected_trainable_names):
            raise CheckpointError(
                ReasonCode.CHECKPOINT_INCOMPATIBLE,
                "adapter safetensors 参数名不兼容",
            )
        parameter_count = sum(value.numel() for value in trainable.values())
        manifest_parameter_count = manifest.get("trainable_parameter_count")
        if (
            isinstance(manifest_parameter_count, bool)
            or not isinstance(manifest_parameter_count, int)
            or parameter_count != manifest_parameter_count
            or parameter_count != expected_trainable_parameter_count
        ):
            raise CheckpointError(
                ReasonCode.CHECKPOINT_INCOMPATIBLE,
                "Adapter、checkpoint manifest 与当前模型的可训练参数量不一致",
                details={
                    "adapter": parameter_count,
                    "manifest": manifest_parameter_count,
                    "current_model": expected_trainable_parameter_count,
                },
            )
        return manifest, trainable

    @staticmethod
    def _file(root: Path, relative: str) -> Path:
        path = root / relative
        linked = first_symlink_component(path)
        if linked is not None:
            raise CheckpointError(
                ReasonCode.OUTPUT_LINK,
                f"checkpoint 路径含链接组件：{linked}",
            )
        try:
            path.resolve().relative_to(root.resolve())
        except ValueError as error:
            raise CheckpointError(
                ReasonCode.PATH_ESCAPE,
                f"checkpoint 路径逃逸：{relative}",
            ) from error
        if not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1:
            raise CheckpointError(
                ReasonCode.OUTPUT_LINK,
                f"checkpoint 文件必须是普通单链接：{relative}",
            )
        return path
