"""Phase 2 OA-AuxSeg 的公共类型和严格配置合同。"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor


MODEL_SCHEMA_VERSION = "oa_auxseg_model_v6"
CHECKPOINT_SCHEMA_VERSION = "oa_auxseg_checkpoint_v6"
INFERENCE_SCHEMA_VERSION = "oa_auxseg_inference_v6"
CONFIG_SCHEMA_VERSION = "oa_auxseg_runtime_config_v5"
BENCHMARK_SCHEMA_VERSION = "oa_auxseg_hdf5_v1"
SUPPORTED_BACKBONE = "convnext_small"
PHASE2_INCLUDED_SOURCES = (
    "gdcld",
    "lmhld",
    "landslidebench_agent",
    "landslide4sense",
    "multimodal_landslide",
)
PHASE2_EXCLUDED_SOURCES = ("sen12landslides",)
SUPPORTED_AUXILIARY_ORDER = (
    "dem",
    "insar_velocity",
    "slope",
)
NULL_AUXILIARY = "__null__"
REGION_GEOMETRY_DIM = 8
VARIANTS = (
    "optical_only",
    "direct_concat",
    "mean_auxiliary_fusion",
    "cmnext_injection",
    "injection_quality",
    "proposed_dropout",
)


def ordered_auxiliary_names(names: Sequence[str]) -> tuple[str, ...]:
    normalized = {str(name) for name in names}
    unknown = normalized - set(SUPPORTED_AUXILIARY_ORDER)
    if unknown:
        raise ValueError(f"未注册辅助模态：{sorted(unknown)}")
    return tuple(
        name for name in SUPPORTED_AUXILIARY_ORDER if name in normalized
    )


def signature_key(names: Sequence[str]) -> str:
    payload = "\0".join(str(name) for name in names).encode("utf-8")
    return f"sig_{hashlib.sha256(payload).hexdigest()[:16]}"


def direct_signature_key(
    optical_names: Sequence[str],
    auxiliaries: Sequence[tuple[str, Sequence[str]]],
) -> str:
    parts = ["optical", *optical_names]
    for name, channel_names in auxiliaries:
        parts.extend(("aux", name, *channel_names))
    return signature_key(parts)


@dataclass(frozen=True)
class ModelRegistry:
    """由 Benchmark index 推导、随 checkpoint 固化的输入 registry。"""

    optical_signatures: tuple[tuple[str, ...], ...]
    auxiliary_channels: Mapping[str, tuple[str, ...]]
    available_auxiliaries: Mapping[tuple[str, ...], tuple[str, ...]]

    def __post_init__(self) -> None:
        if not self.optical_signatures:
            raise ValueError("registry 至少需要一个光学通道签名")
        unknown = set(self.auxiliary_channels) - set(SUPPORTED_AUXILIARY_ORDER)
        if unknown:
            raise ValueError(f"未注册辅助模态：{sorted(unknown)}")
        if len(set(self.optical_signatures)) != len(self.optical_signatures):
            raise ValueError("registry 光学通道签名不能重复")
        for name, channels in self.auxiliary_channels.items():
            if not channels:
                raise ValueError(f"{name}: 辅助模态通道签名不能为空")
            if len(set(channels)) != len(channels):
                raise ValueError(f"{name}: 辅助模态通道名不能重复")
        expected_order = self.auxiliary_order
        for signature in self.optical_signatures:
            if not signature:
                raise ValueError("光学通道签名不能为空")
            names = self.available_auxiliaries.get(signature, ())
            if set(names) - set(self.auxiliary_channels):
                raise ValueError(f"{signature}: availability 包含未知辅助模态")
            ordered = tuple(name for name in expected_order if name in set(names))
            if tuple(names) != ordered:
                raise ValueError(f"{signature}: availability 模态顺序不符合 registry")
        extra_signatures = (
            set(self.available_auxiliaries) - set(self.optical_signatures)
        )
        if extra_signatures:
            raise ValueError(
                f"availability 含未知光学通道签名：{sorted(extra_signatures)}"
            )

    @property
    def auxiliary_order(self) -> tuple[str, ...]:
        return ordered_auxiliary_names(tuple(self.auxiliary_channels))

    @property
    def modality_weight_order(self) -> tuple[str, ...]:
        return (*self.auxiliary_order, NULL_AUXILIARY)

    def region_feature_dim(self, region_projection_dim: int) -> int:
        return (
            region_projection_dim * 2
            + REGION_GEOMETRY_DIM
            + len(self.modality_weight_order)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "optical_signatures": [list(item) for item in self.optical_signatures],
            "auxiliary_order": list(self.auxiliary_order),
            "auxiliary_channels": {
                name: list(self.auxiliary_channels[name])
                for name in self.auxiliary_order
            },
            "available_auxiliaries": [
                {
                    "optical_channel_names": list(signature),
                    "modalities": list(
                        self.available_auxiliaries.get(signature, ())
                    ),
                }
                for signature in self.optical_signatures
            ],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ModelRegistry":
        expected_fields = {
            "optical_signatures",
            "auxiliary_order",
            "auxiliary_channels",
            "available_auxiliaries",
        }
        if set(value) != expected_fields:
            raise ValueError("registry 字段与当前 schema 不匹配")
        optical_signatures = tuple(
            tuple(str(name) for name in item)
            for item in value["optical_signatures"]
        )
        availability = {
            tuple(str(name) for name in row["optical_channel_names"]): tuple(
                str(name) for name in row["modalities"]
            )
            for row in value["available_auxiliaries"]
        }
        registry = cls(
            optical_signatures=optical_signatures,
            auxiliary_channels={
                str(name): tuple(str(channel) for channel in channels)
                for name, channels in value["auxiliary_channels"].items()
            },
            available_auxiliaries=availability,
        )
        recorded_order = tuple(str(name) for name in value["auxiliary_order"])
        if recorded_order != registry.auxiliary_order:
            raise ValueError("registry auxiliary_order 与当前受支持目录不一致")
        return registry


@dataclass(frozen=True)
class OAAuxSegConfig:
    variant: str
    decoder_dim: int = 512
    region_projection_dim: int = 128
    optical_stochastic_depth: float = 0.1
    auxiliary_drop_path: float = 0.1
    decoder_dropout: float = 0.1
    region_threshold: float = 0.5
    min_region_area: int = 16

    def __post_init__(self) -> None:
        if self.variant not in VARIANTS:
            raise ValueError(f"variant 必须是 {VARIANTS}，实际为 {self.variant!r}")
        if self.decoder_dim != 512:
            raise ValueError("Phase 2 v6 固定要求 decoder_dim=512")
        if self.region_projection_dim != 128:
            raise ValueError("Phase 2 v6 固定要求 region_projection_dim=128")
        for name, probability in (
            ("optical_stochastic_depth", self.optical_stochastic_depth),
            ("auxiliary_drop_path", self.auxiliary_drop_path),
            ("decoder_dropout", self.decoder_dropout),
        ):
            if not 0 <= probability < 1:
                raise ValueError(f"{name} 必须位于 [0,1)")
        if not 0 < self.region_threshold < 1:
            raise ValueError("region_threshold 必须位于 (0,1)")
        if self.min_region_area <= 0:
            raise ValueError("min_region_area 必须大于 0")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": MODEL_SCHEMA_VERSION,
            "variant": self.variant,
            "decoder_dim": self.decoder_dim,
            "region_projection_dim": self.region_projection_dim,
            "optical_stochastic_depth": self.optical_stochastic_depth,
            "auxiliary_drop_path": self.auxiliary_drop_path,
            "decoder_dropout": self.decoder_dropout,
            "region_threshold": self.region_threshold,
            "min_region_area": self.min_region_area,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "OAAuxSegConfig":
        if value.get("schema_version") != MODEL_SCHEMA_VERSION:
            raise ValueError(
                f"模型配置 schema 错误：{value.get('schema_version')!r}"
            )
        expected_fields = {
            "schema_version",
            "variant",
            "decoder_dim",
            "region_projection_dim",
            "optical_stochastic_depth",
            "auxiliary_drop_path",
            "decoder_dropout",
            "region_threshold",
            "min_region_area",
        }
        if set(value) != expected_fields:
            raise ValueError("模型配置字段与当前 schema 不匹配")
        return cls(
            variant=str(value["variant"]),
            decoder_dim=int(value["decoder_dim"]),
            region_projection_dim=int(value["region_projection_dim"]),
            optical_stochastic_depth=float(
                value["optical_stochastic_depth"]
            ),
            auxiliary_drop_path=float(value["auxiliary_drop_path"]),
            decoder_dropout=float(value["decoder_dropout"]),
            region_threshold=float(value["region_threshold"]),
            min_region_area=int(value["min_region_area"]),
        )


@dataclass
class PackedAuxiliary:
    sample_indices: Tensor
    values: Tensor
    pixel_valid: Tensor
    channel_valid: Tensor
    channel_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.sample_indices.ndim != 1:
            raise ValueError("sample_indices 必须是一维")
        if self.sample_indices.dtype != torch.int64:
            raise ValueError("sample_indices 必须是 int64")
        if self.sample_indices.unique().numel() != self.sample_indices.numel():
            raise ValueError("sample_indices 不能重复")
        if self.values.ndim != 4:
            raise ValueError("辅助模态 values 必须是 NCHW")
        if self.pixel_valid.shape != self.values.shape:
            raise ValueError("辅助模态 pixel_valid shape 必须等于 values")
        if self.channel_valid.shape != self.values.shape[:2]:
            raise ValueError("辅助模态 channel_valid 必须是 NC")
        if len(self.channel_names) != self.values.shape[1]:
            raise ValueError("辅助模态通道名数量不符")
        if self.sample_indices.shape[0] != self.values.shape[0]:
            raise ValueError("辅助模态 sample_indices 与 values 数量不符")

    def to(self, device: torch.device, *, non_blocking: bool = False) -> "PackedAuxiliary":
        return PackedAuxiliary(
            sample_indices=self.sample_indices.to(
                device=device, non_blocking=non_blocking
            ),
            values=self.values.to(device=device, non_blocking=non_blocking),
            pixel_valid=self.pixel_valid.to(
                device=device, non_blocking=non_blocking
            ),
            channel_valid=self.channel_valid.to(
                device=device, non_blocking=non_blocking
            ),
            channel_names=self.channel_names,
        )


@dataclass
class OAAuxSegBatch:
    optical: tuple[Tensor, ...]
    optical_pixel_valid: tuple[Tensor, ...]
    optical_channel_valid: tuple[Tensor, ...]
    optical_channel_names: tuple[tuple[str, ...], ...]
    auxiliaries: Mapping[str, PackedAuxiliary]

    def __post_init__(self) -> None:
        size = len(self.optical)
        if size == 0:
            raise ValueError("batch 不能为空")
        if not (
            len(self.optical_pixel_valid)
            == len(self.optical_channel_valid)
            == len(self.optical_channel_names)
            == size
        ):
            raise ValueError("光学 batch 字段数量不一致")
        spatial_size: tuple[int, int] | None = None
        for values, pixel_valid, channel_valid, names in zip(
            self.optical,
            self.optical_pixel_valid,
            self.optical_channel_valid,
            self.optical_channel_names,
            strict=True,
        ):
            if values.ndim != 3:
                raise ValueError("单样本 optical 必须是 CHW")
            if pixel_valid.shape != values.shape:
                raise ValueError("optical pixel_valid shape 不符")
            if channel_valid.shape != (values.shape[0],):
                raise ValueError("optical channel_valid shape 不符")
            if len(names) != values.shape[0]:
                raise ValueError("optical channel_names 数量不符")
            current_size = (int(values.shape[-2]), int(values.shape[-1]))
            if spatial_size is None:
                spatial_size = current_size
            elif current_size != spatial_size:
                raise ValueError("同一 batch 的空间尺寸必须一致")
        for name, packed in self.auxiliaries.items():
            if name not in SUPPORTED_AUXILIARY_ORDER:
                raise ValueError(f"未知辅助模态：{name}")
            if packed.sample_indices.numel() and (
                int(packed.sample_indices.min()) < 0
                or int(packed.sample_indices.max()) >= size
            ):
                raise ValueError(f"{name}: sample_indices 越界")
            if tuple(packed.values.shape[-2:]) != spatial_size:
                raise ValueError(f"{name}: 辅助模态空间尺寸与光学不一致")

    @property
    def batch_size(self) -> int:
        return len(self.optical)

    @property
    def spatial_size(self) -> tuple[int, int]:
        return int(self.optical[0].shape[-2]), int(self.optical[0].shape[-1])

    def to(self, device: torch.device, *, non_blocking: bool = False) -> "OAAuxSegBatch":
        return OAAuxSegBatch(
            optical=tuple(
                value.to(device=device, non_blocking=non_blocking)
                for value in self.optical
            ),
            optical_pixel_valid=tuple(
                value.to(device=device, non_blocking=non_blocking)
                for value in self.optical_pixel_valid
            ),
            optical_channel_valid=tuple(
                value.to(device=device, non_blocking=non_blocking)
                for value in self.optical_channel_valid
            ),
            optical_channel_names=self.optical_channel_names,
            auxiliaries={
                name: packed.to(device, non_blocking=non_blocking)
                for name, packed in self.auxiliaries.items()
            },
        )


@dataclass(frozen=True)
class CandidateRegion:
    region_id: int
    mask: Tensor
    bbox_xyxy: tuple[int, int, int, int]
    centroid_xy: tuple[float, float]
    area_pixels: int
    confidence: float


@dataclass
class OAAuxSegOutput:
    mask_logits: Tensor
    mask_probability: Tensor
    no_target_score: Tensor
    modality_names: tuple[str, ...]
    modality_weights: Tensor
    modality_weight_map_strides: tuple[int, ...]
    modality_weight_maps: tuple[Tensor, ...]
    candidate_regions: list[list[CandidateRegion]] | None = None
    region_features: list[Tensor] | None = None
    active_modalities: list[tuple[str, ...]] = field(default_factory=list)


@dataclass(frozen=True)
class RuntimeConfig:
    schema_version: str
    benchmark_root: str
    output_dir: str
    backbone: str
    backbone_weights: str
    variant: str
    seed: int = 20260724
    device: str = "cuda"
    batch_size: int = 8
    num_workers: int = 0
    normalization: str = "zscore"
    max_steps: int = 300
    eval_interval: int = 100
    checkpoint_interval: int = 100
    log_interval: int = 10
    backbone_lr: float = 3e-5
    new_lr: float = 3e-4
    weight_decay: float = 0.05
    warmup_ratio: float = 0.05
    min_lr_ratio: float = 0.05
    grad_clip: float = 1.0
    modality_dropout: float = 0.2
    train_sampler: str = "uniform"
    optical_stochastic_depth: float = 0.1
    auxiliary_drop_path: float = 0.1
    decoder_dropout: float = 0.1
    use_bf16: bool = True
    gradient_checkpointing: bool = True
    region_threshold: float = 0.5
    min_region_area: int = 16

    def __post_init__(self) -> None:
        if self.schema_version != CONFIG_SCHEMA_VERSION:
            raise ValueError(f"运行配置 schema 错误：{self.schema_version!r}")
        if self.variant not in VARIANTS:
            raise ValueError(f"非法 variant：{self.variant}")
        if self.backbone != SUPPORTED_BACKBONE:
            raise ValueError(
                f"Phase 2 v6 只支持 backbone={SUPPORTED_BACKBONE!r}"
            )
        if self.normalization not in {"none", "zscore"}:
            raise ValueError("normalization 必须是 none 或 zscore")
        if self.batch_size <= 0 or self.max_steps <= 0:
            raise ValueError("batch_size 和 max_steps 必须大于 0")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device 必须是 cpu 或 cuda")
        if self.num_workers < 0:
            raise ValueError("num_workers 不能小于 0")
        if not isinstance(self.gradient_checkpointing, bool):
            raise ValueError("gradient_checkpointing 必须是 bool")
        if min(
            self.eval_interval,
            self.checkpoint_interval,
            self.log_interval,
        ) <= 0:
            raise ValueError("日志、评价和 checkpoint 间隔必须大于 0")
        if min(self.backbone_lr, self.new_lr, self.grad_clip) <= 0:
            raise ValueError("学习率和 grad_clip 必须大于 0")
        if self.weight_decay < 0:
            raise ValueError("weight_decay 不能小于 0")
        if not 0 <= self.modality_dropout < 1:
            raise ValueError("modality_dropout 必须位于 [0,1)")
        if not 0 <= self.warmup_ratio < 1:
            raise ValueError("warmup_ratio 必须位于 [0,1)")
        if not 0 < self.min_lr_ratio <= 1:
            raise ValueError("min_lr_ratio 必须位于 (0,1]")
        if self.train_sampler not in {"uniform", "balanced_target_presence"}:
            raise ValueError(
                "train_sampler 必须是 uniform 或 balanced_target_presence"
            )
        if (
            self.train_sampler == "balanced_target_presence"
            and self.batch_size % 2 != 0
        ):
            raise ValueError(
                "balanced_target_presence 要求偶数 batch_size"
            )
        for name, probability in (
            ("optical_stochastic_depth", self.optical_stochastic_depth),
            ("auxiliary_drop_path", self.auxiliary_drop_path),
            ("decoder_dropout", self.decoder_dropout),
        ):
            if not 0 <= probability < 1:
                raise ValueError(f"{name} 必须位于 [0,1)")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RuntimeConfig":
        if value.get("schema_version") != CONFIG_SCHEMA_VERSION:
            raise ValueError(
                f"运行配置 schema 错误：{value.get('schema_version')!r}"
            )
        allowed = {field.name for field in cls.__dataclass_fields__.values()}
        unknown = set(value) - allowed
        missing = {
            "schema_version",
            "benchmark_root",
            "output_dir",
            "backbone",
            "backbone_weights",
            "variant",
        } - set(value)
        if unknown:
            raise ValueError(f"运行配置包含未知字段：{sorted(unknown)}")
        if missing:
            raise ValueError(f"运行配置缺少字段：{sorted(missing)}")
        return cls(**value)

    def with_overrides(self, **overrides: Any) -> "RuntimeConfig":
        values = {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }
        values.update({key: value for key, value in overrides.items() if value is not None})
        return RuntimeConfig(**values)

    def to_dict(self) -> dict[str, Any]:
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }

    def resolve_path(self, value: str, repo_root: Path) -> Path:
        path = Path(value)
        return path if path.is_absolute() else (repo_root / path).resolve()
