"""原子 checkpoint、严格 identity 比对与可重复 resume。"""

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

from oa_groundrag.phase3.common import (
    atomic_write_json,
    canonical_json,
    first_symlink_component,
    read_json,
    sha256_file,
    sha256_text,
)
from oa_groundrag.phase3.errors import RSGeneralDescError

from .contracts import (
    CHECKPOINT_SCHEMA_VERSION,
    SAMPLE_TRACE_SCHEMA_VERSION,
)
from .errors import CheckpointError, ReasonCode
from .input_pipeline import (
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
_TRAINING_LAYOUT_FIELDS = {
    "physical_batch_size",
    "gradient_accumulation_steps",
    "effective_batch_size",
    "input_pipeline_backend",
    "num_workers",
    "prefetch_factor",
    "pin_memory",
    "sample_trace_schema_version",
}


def _validated_training_layout(
    value: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _TRAINING_LAYOUT_FIELDS:
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


def _save_torch(path: Path, value: Any) -> None:
    temporary = path.parent / f".{path.name}.tmp"
    if temporary.exists() or temporary.is_symlink():
        raise CheckpointError(
            ReasonCode.OUTPUT_EXISTS,
            f"临时 checkpoint 文件已存在：{temporary}",
        )
    torch.save(value, temporary)
    temporary.replace(path)


class CheckpointManager:
    def save(
        self,
        root: Path,
        *,
        trainable_state: Mapping[str, Tensor],
        optimizer_state: Mapping[str, Any],
        scheduler_state: Mapping[str, Any],
        cursor: TrainingCursor,
        sampler_state: Mapping[str, Any],
        multireference_epoch: int,
        config_snapshot: Mapping[str, Any],
        config_semantic_sha256: str,
        benchmark_identity: Mapping[str, Any],
        validation_selection_identity: Mapping[str, Any],
        model_identity: Mapping[str, Any],
        processor_identity: Mapping[str, Any],
        training_layout: Mapping[str, Any],
        trainable_names: tuple[str, ...],
        rng_state: Mapping[str, Any] | None = None,
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
        if (
            isinstance(multireference_epoch, bool)
            or not isinstance(multireference_epoch, int)
            or multireference_epoch < 0
        ):
            raise CheckpointError(
                ReasonCode.TYPE_MISMATCH,
                "multireference_epoch 必须是 >= 0 的整数",
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
            _save_torch(staging / "optimizer.pt", dict(optimizer_state))
            _save_torch(staging / "scheduler.pt", dict(scheduler_state))
            _save_torch(
                staging / "rng.pt",
                dict(rng_state or capture_rng_state()),
            )
            atomic_write_json(staging / "config_snapshot.json", config_snapshot)
            state_row = {
                "cursor": cursor.to_dict(),
                "sampler_state": dict(sampler_state),
                "multireference_epoch": multireference_epoch,
            }
            atomic_write_json(staging / "training_state.json", state_row)
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
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "config_semantic_sha256": config_semantic_sha256,
                "config_snapshot_sha256": sha256_text(
                    canonical_json(config_snapshot)
                ),
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
                "multireference_epoch": multireference_epoch,
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
