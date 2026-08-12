"""Stage 5 Mask-Grounded Adapter 的冻结配置合同。"""

from __future__ import annotations

from dataclasses import dataclass, replace
import os
from pathlib import Path
from typing import Any, Mapping

from oa_groundrag.artifacts.identity import (
    canonical_json,
    sha256_file,
    sha256_text,
)
from oa_groundrag.artifacts.io import first_symlink_component

from oa_groundrag.vlm.config import VLMConfig, _load_yaml, load_config
from oa_groundrag.grounding.contracts import MaskMode
from oa_groundrag.vlm.errors import ConfigError, ReasonCode


STAGE5_CONFIG_SCHEMA = "rs_vlm.mask_grounded_stage5_config.v1"


@dataclass(frozen=True)
class WarmStartContract:
    checkpoint_root: Path
    best_pointer_sha256: str
    checkpoint_step: int
    checkpoint_manifest_sha256: str
    adapter_sha256: str


@dataclass(frozen=True)
class Stage5DataContract:
    compact_training_root: Path
    eval_dev_root: Path
    split_seed: int
    train_ratio: float
    region_micro_ratio: float
    replay_micro_ratio: float
    replay_validation_parents: int


@dataclass(frozen=True)
class Stage5Config:
    """兼容现有 trainer 属性，同时把 Stage 5 身份加入 checkpoint SHA。"""

    schema_version: str
    base: VLMConfig
    data_contract: Stage5DataContract
    warm_start: WarmStartContract
    workflow_root: Path
    config_path: Path
    semantic_sha256: str

    @property
    def run(self):
        return self.base.run

    @property
    def data(self):
        return self.base.data

    @property
    def limits(self):
        return self.base.limits

    @property
    def model(self):
        return self.base.model

    @property
    def adaptation(self):
        return self.base.adaptation

    @property
    def training(self):
        return self.base.training

    @property
    def generation(self):
        return self.base.generation

    def semantic_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "trainer": self.base.semantic_dict(),
            "data_contract": {
                "compact_training_root": str(self.data_contract.compact_training_root),
                "eval_dev_root": str(self.data_contract.eval_dev_root),
                "split_seed": self.data_contract.split_seed,
                "train_ratio": self.data_contract.train_ratio,
                "region_micro_ratio": self.data_contract.region_micro_ratio,
                "replay_micro_ratio": self.data_contract.replay_micro_ratio,
                "replay_validation_parents": self.data_contract.replay_validation_parents,
            },
            "warm_start": {
                "checkpoint_root": str(self.warm_start.checkpoint_root),
                "best_pointer_sha256": self.warm_start.best_pointer_sha256,
                "checkpoint_step": self.warm_start.checkpoint_step,
                "checkpoint_manifest_sha256": self.warm_start.checkpoint_manifest_sha256,
                "adapter_sha256": self.warm_start.adapter_sha256,
            },
        }

    def snapshot_dict(self) -> dict[str, Any]:
        value = self.semantic_dict()
        value["workflow_root"] = str(self.workflow_root)
        value["trainer"]["run"] = self.base.snapshot_dict()["run"]
        value["semantic_sha256"] = self.semantic_sha256
        return value


def _mapping(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ConfigError(ReasonCode.TYPE_MISMATCH, f"{location} 必须是字符串键对象")
    return dict(value)


def _exact(value: Mapping[str, Any], fields: tuple[str, ...], location: str) -> None:
    if set(value) != set(fields):
        raise ConfigError(
            ReasonCode.UNKNOWN_FIELD,
            f"{location} 字段不匹配",
            details={
                "missing": sorted(set(fields) - set(value)),
                "unknown": sorted(set(value) - set(fields)),
            },
        )


def _path(base: Path, value: Any, location: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(ReasonCode.TYPE_MISMATCH, f"{location} 必须是路径字符串")
    path = Path(value)
    return Path(os.path.abspath(path if path.is_absolute() else base / path))


def _sha(value: Any, location: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ConfigError(ReasonCode.TYPE_MISMATCH, f"{location} 必须是小写 SHA-256")
    return value


def _integer(value: Any, location: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigError(ReasonCode.TYPE_MISMATCH, f"{location} 必须是 >= {minimum} 的整数")
    return value


def _ratio(value: Any, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(ReasonCode.TYPE_MISMATCH, f"{location} 必须是比例")
    result = float(value)
    if not 0.0 < result < 1.0:
        raise ConfigError(ReasonCode.TYPE_MISMATCH, f"{location} 必须位于 (0,1)")
    return result


def load_stage5_config(path: Path | str) -> Stage5Config:
    """读取配置并强制锁定负责人确认的训练、split、replay 与科学边界。"""

    config_path = Path(os.path.abspath(Path(path)))
    linked = first_symlink_component(config_path)
    if linked is not None or not config_path.is_file() or config_path.is_symlink():
        raise ConfigError(ReasonCode.OUTPUT_LINK, f"Stage 5 配置必须是普通文件：{config_path}")
    row = _load_yaml(config_path)
    _exact(
        row,
        (
            "schema_version", "base_config", "workflow_root", "data", "warm_start",
            "training", "generation",
        ),
        "$",
    )
    if row["schema_version"] != STAGE5_CONFIG_SCHEMA:
        raise ConfigError(ReasonCode.INVALID_ENUM, f"仅支持 {STAGE5_CONFIG_SCHEMA}")
    base_path = _path(config_path.parent, row["base_config"], "$.base_config")
    base = load_config(base_path)
    data = _mapping(row["data"], "$.data")
    _exact(
        data,
        (
            "compact_training_root", "eval_dev_root", "split_seed", "train_ratio",
            "region_micro_ratio", "replay_micro_ratio", "replay_validation_parents",
        ),
        "$.data",
    )
    data_contract = Stage5DataContract(
        compact_training_root=_path(config_path.parent, data["compact_training_root"], "$.data.compact_training_root"),
        eval_dev_root=_path(config_path.parent, data["eval_dev_root"], "$.data.eval_dev_root"),
        split_seed=_integer(data["split_seed"], "$.data.split_seed"),
        train_ratio=_ratio(data["train_ratio"], "$.data.train_ratio"),
        region_micro_ratio=_ratio(data["region_micro_ratio"], "$.data.region_micro_ratio"),
        replay_micro_ratio=_ratio(data["replay_micro_ratio"], "$.data.replay_micro_ratio"),
        replay_validation_parents=_integer(
            data["replay_validation_parents"],
            "$.data.replay_validation_parents",
            minimum=1,
        ),
    )
    if (
        data_contract.split_seed != 20260808
        or data_contract.train_ratio != 0.9
        or data_contract.region_micro_ratio != 0.9
        or data_contract.replay_micro_ratio != 0.1
        or data_contract.region_micro_ratio + data_contract.replay_micro_ratio != 1.0
        or data_contract.replay_validation_parents != 128
    ):
        raise ConfigError(ReasonCode.TYPE_MISMATCH, "Stage 5 split/replay 比例未按负责人决策冻结")
    warm = _mapping(row["warm_start"], "$.warm_start")
    _exact(
        warm,
        (
            "checkpoint_root", "best_pointer_sha256", "checkpoint_step",
            "checkpoint_manifest_sha256", "adapter_sha256",
        ),
        "$.warm_start",
    )
    warm_start = WarmStartContract(
        checkpoint_root=_path(config_path.parent, warm["checkpoint_root"], "$.warm_start.checkpoint_root"),
        best_pointer_sha256=_sha(warm["best_pointer_sha256"], "$.warm_start.best_pointer_sha256"),
        checkpoint_step=_integer(warm["checkpoint_step"], "$.warm_start.checkpoint_step", minimum=1),
        checkpoint_manifest_sha256=_sha(
            warm["checkpoint_manifest_sha256"],
            "$.warm_start.checkpoint_manifest_sha256",
        ),
        adapter_sha256=_sha(warm["adapter_sha256"], "$.warm_start.adapter_sha256"),
    )
    if warm_start.checkpoint_step != 1000:
        raise ConfigError(ReasonCode.CHECKPOINT_INCOMPATIBLE, "warm-start 必须是 RS-General step-1000")
    training = _mapping(row["training"], "$.training")
    _exact(
        training,
        (
            "learning_rate", "warmup_ratio", "max_steps", "epochs",
            "checkpoint_interval", "validation_interval", "max_input_tokens",
        ),
        "$.training",
    )
    required = {
        "learning_rate": 5e-5,
        "warmup_ratio": 0.05,
        "max_steps": 1000,
        "epochs": 2,
        "checkpoint_interval": 100,
        "validation_interval": 100,
        "max_input_tokens": 4096,
    }
    if training != required:
        raise ConfigError(ReasonCode.TYPE_MISMATCH, "Stage 5 training 超参数未按冻结值填写")
    generation = _mapping(row["generation"], "$.generation")
    _exact(generation, ("max_new_tokens",), "$.generation")
    if generation["max_new_tokens"] != 768:
        raise ConfigError(ReasonCode.TYPE_MISMATCH, "Stage 5 Region JSON generation 固定为 768 tokens")
    workflow_root = _path(config_path.parent, row["workflow_root"], "$.workflow_root")
    trainer_run = replace(
        base.run,
        name="mask_grounded_region_lora_qwen3vl_2b_rsinit_v1",
        seed=data_contract.split_seed,
        mask_mode=MaskMode.GT_MASK,
        output_root=workflow_root / "training",
        resume_checkpoint=None,
    )
    trainer_limits = replace(
        base.limits,
        max_input_tokens=4096,
        max_total_tokens=100_000_000,
        max_new_tokens=768,
    )
    trainer_training = replace(
        base.training,
        batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=5e-5,
        warmup_ratio=0.05,
        max_steps=1000,
        epochs=2,
        checkpoint_interval=100,
        validation_interval=100,
        validation_max_parents=1,  # split 完成后由 workflow 替换为真实 monitor parent 数。
    )
    trainer_generation = replace(base.generation, max_new_tokens=768)
    provisional_base = replace(
        base,
        run=trainer_run,
        limits=trainer_limits,
        training=trainer_training,
        generation=trainer_generation,
        semantic_sha256="stage5-provisional",
    )
    provisional = Stage5Config(
        schema_version=STAGE5_CONFIG_SCHEMA,
        base=provisional_base,
        data_contract=data_contract,
        warm_start=warm_start,
        workflow_root=workflow_root,
        config_path=config_path,
        semantic_sha256="",
    )
    semantic = sha256_text(canonical_json(provisional.semantic_dict()))
    return replace(provisional, semantic_sha256=semantic)


def with_monitor_parent_count(config: Stage5Config, count: int) -> Stage5Config:
    """把数据驱动的 monitor parent 数写入最终 checkpoint 配置身份。"""

    count = _integer(count, "monitor_parent_count", minimum=1)
    base = replace(
        config.base,
        training=replace(config.base.training, validation_max_parents=count),
    )
    provisional = replace(config, base=base, semantic_sha256="")
    return replace(
        provisional,
        semantic_sha256=sha256_text(canonical_json(provisional.semantic_dict())),
    )


def verify_warm_start_files(config: Stage5Config) -> dict[str, Any]:
    """只读校验旧 RS-General Adapter；绝不改写其 pointer/checkpoint。"""

    checkpoint = config.warm_start.checkpoint_root
    pointer = checkpoint.parents[1] / "best_checkpoint.json"
    manifest = checkpoint / "manifest.json"
    adapter = checkpoint / "adapter" / "adapter_model.safetensors"
    for path, expected, label in (
        (pointer, config.warm_start.best_pointer_sha256, "best pointer"),
        (manifest, config.warm_start.checkpoint_manifest_sha256, "checkpoint manifest"),
        (adapter, config.warm_start.adapter_sha256, "adapter"),
    ):
        linked = first_symlink_component(path)
        if linked is not None or not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1:
            raise ConfigError(ReasonCode.CHECKPOINT_CORRUPT, f"{label} 不是普通单链接文件：{path}")
        if sha256_file(path) != expected:
            raise ConfigError(ReasonCode.CHECKPOINT_CORRUPT, f"{label} SHA-256 漂移：{path}")
    return {
        "checkpoint_root": str(checkpoint),
        "checkpoint_step": config.warm_start.checkpoint_step,
        "best_pointer_sha256": config.warm_start.best_pointer_sha256,
        "checkpoint_manifest_sha256": config.warm_start.checkpoint_manifest_sha256,
        "adapter_sha256": config.warm_start.adapter_sha256,
    }
