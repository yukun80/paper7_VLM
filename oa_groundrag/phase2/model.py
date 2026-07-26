"""Phase 2 v5 的 ConvNeXt-Small 与完整 DELIVER 式四阶段 OA-AuxSeg。"""

from __future__ import annotations

import hashlib
import itertools
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import torch
import torch.nn.functional as functional
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint as activation_checkpoint
from torchvision.models import convnext_small

from .contracts import (
    ModelRegistry,
    OAAuxSegBatch,
    OAAuxSegConfig,
    OAAuxSegOutput,
    PackedAuxiliary,
    direct_signature_key,
    signature_key,
)
from .fusion import (
    FFM_HEADS,
    FUSION_CONTRACT,
    MSPA_MLP_RATIOS,
    STAGE_CHANNELS,
    STAGE_DEPTHS,
    STAGE_STRIDES,
    AuxiliaryDownsample,
    ChannelLayerNorm2d,
    CMNeXtSpatialSelector,
    EqualSpatialSelector,
    MSPAStage,
    NullAwareSpatialSelector,
    ProgressiveFusion,
    SparseStageFeature,
    availability_weight_map,
    optical_only_weight_map,
    summarize_weight_maps,
)
from .regions import extract_regions_and_features
from .validity import (
    FeatureValidity,
    VALIDITY_CONTRACT,
    apply_support,
    resample_validity,
    validity_from_channels,
)


BACKBONE_NAME = "convnext_small"
BACKBONE_CONTRACT_NAME = "torchvision_convnext_small_shared_rgb_residual_stem"
BACKBONE_STAGE_CHANNELS = STAGE_CHANNELS
BACKBONE_STAGE_DEPTHS = STAGE_DEPTHS
EXPECTED_BACKBONE_SHA256 = (
    "0c510722adfd92966a2bd72b92f785ca05966bbac03cafe2f7a90b1f54bfab9a"
)
ARCHITECTURE_NAME = "oa_auxseg_deliver_full_v2"
OPTICAL_STEM_CONTRACT = {
    "type": "shared_official_rgb_plus_signature_extra_residual",
    "output_channels": 96,
    "rgb_branch": "shared_official_conv4_stride4",
    "extra_branch": "signature_specific_conv4_stride4_no_bias",
    "extra_branch_initialization": "zero",
    "normalization": "shared_official_layer_norm",
}
DECODER_CONTRACT = {
    "type": "segformer_style_four_scale",
    "decoder_dim": 512,
    "region_projection_dim": 128,
}
AUXILIARY_STEM_CONTRACT = {
    "type": "modality_specific_conv7_stride4_padding3",
    "output_channels": 96,
}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def optical_rgb_indices(signature: Sequence[str]) -> tuple[int, int, int]:
    """从已审计的 Benchmark 通道名返回 Red/Green/Blue 位置。"""

    names = tuple(str(name) for name in signature)
    explicit = {
        ("R", "G", "B"): (0, 1, 2),
        ("Red", "Green", "Blue"): (0, 1, 2),
        ("Blue", "Green", "Red", "NIR"): (2, 1, 0),
        (
            "s2_b02",
            "s2_b03",
            "s2_b04",
            "s2_b05",
            "s2_b06",
            "s2_b07",
            "s2_b08",
            "s2_b8a",
            "s2_b11",
            "s2_b12",
        ): (2, 1, 0),
        tuple(f"B{index:02d}" for index in range(1, 13)): (3, 2, 1),
    }
    if names not in explicit:
        raise ValueError(f"没有审计过的 RGB stem 映射：{names}")
    return explicit[names]


class SignatureStem(nn.Module):
    """共享官方 RGB stem，并为非 RGB 通道学习零初始化残差。"""

    def __init__(
        self,
        in_channels: int,
        *,
        rgb_indices: tuple[int, int, int],
        official_conv: nn.Conv2d,
    ) -> None:
        super().__init__()
        self.rgb_indices = rgb_indices
        rgb_set = set(rgb_indices)
        self.extra_indices = tuple(
            index for index in range(in_channels) if index not in rgb_set
        )
        self.extra_projection: nn.Conv2d | None
        if self.extra_indices:
            self.extra_projection = nn.Conv2d(
                len(self.extra_indices),
                official_conv.out_channels,
                kernel_size=official_conv.kernel_size,
                stride=official_conv.stride,
                padding=official_conv.padding,
                dilation=official_conv.dilation,
                groups=1,
                bias=False,
            )
            nn.init.zeros_(self.extra_projection.weight)
        else:
            self.extra_projection = None

    def forward(
        self,
        values: Tensor,
        pixel_valid: Tensor,
        channel_valid: Tensor,
        *,
        rgb_projection: nn.Conv2d,
        normalization: nn.Module,
        anchor_validity: FeatureValidity | None = None,
    ) -> tuple[Tensor, FeatureValidity, FeatureValidity]:
        source_validity = validity_from_channels(
            pixel_valid,
            channel_valid,
        )
        channelwise = pixel_valid.to(torch.bool) & channel_valid.to(
            torch.bool
        )[:, :, None, None]
        clean = torch.where(channelwise, values, torch.zeros_like(values))
        rgb = clean[:, self.rgb_indices]
        stem = rgb_projection(rgb)
        if self.extra_projection is not None:
            stem = stem + self.extra_projection(clean[:, self.extra_indices])
        stem = normalization(stem)
        source_anchor = (
            source_validity if anchor_validity is None else anchor_validity
        )
        target_validity = resample_validity(
            source_anchor,
            stem.shape[-2:],
        )
        return (
            apply_support(stem, target_validity),
            source_anchor,
            target_validity,
        )


class AuxiliaryStem(nn.Module):
    """模态独立的 raw→stride-4/96 adapter。"""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(
                in_channels,
                STAGE_CHANNELS[0],
                kernel_size=7,
                stride=4,
                padding=3,
            ),
            ChannelLayerNorm2d(STAGE_CHANNELS[0]),
            nn.GELU(),
        )

    def forward(
        self,
        values: Tensor,
        pixel_valid: Tensor,
        channel_valid: Tensor,
    ) -> tuple[Tensor, FeatureValidity]:
        source_validity = validity_from_channels(
            pixel_valid,
            channel_valid,
        )
        channelwise = pixel_valid.to(torch.bool) & channel_valid.to(
            torch.bool
        )[:, :, None, None]
        clean = torch.where(channelwise, values, torch.zeros_like(values))
        feature = self.projection(clean)
        validity = resample_validity(
            source_validity,
            feature.shape[-2:],
        )
        return apply_support(feature, validity), validity


class SegmentationDecoder(nn.Module):
    """512维 SegFormer-style 四尺度 decoder 与独立区域投影。"""

    def __init__(
        self,
        channels: Sequence[int],
        *,
        decoder_dim: int,
        region_projection_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.lateral = nn.ModuleList(
            nn.Conv2d(channel, decoder_dim, kernel_size=1, bias=False)
            for channel in channels
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(
                decoder_dim * len(channels),
                decoder_dim,
                kernel_size=1,
                bias=False,
            ),
            ChannelLayerNorm2d(decoder_dim),
            nn.GELU(),
            nn.Dropout2d(dropout),
        )
        self.classifier = nn.Conv2d(decoder_dim, 1, kernel_size=1)
        self.optical_region_projection = nn.Sequential(
            nn.Conv2d(
                channels[0],
                region_projection_dim,
                kernel_size=1,
                bias=False,
            ),
            ChannelLayerNorm2d(region_projection_dim),
            nn.GELU(),
        )
        self.fused_region_projection = nn.Sequential(
            nn.Conv2d(
                decoder_dim,
                region_projection_dim,
                kernel_size=1,
                bias=False,
            ),
            ChannelLayerNorm2d(region_projection_dim),
            nn.GELU(),
        )

    def forward(
        self,
        features: Sequence[Tensor],
        *,
        optical_feature4: Tensor,
        output_size: tuple[int, int],
        validity4: FeatureValidity,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if len(features) != len(self.lateral):
            raise ValueError("decoder feature 数量不符")
        target_size = features[0].shape[-2:]
        projected = []
        for layer, feature in zip(
            self.lateral,
            features,
            strict=True,
        ):
            value = layer(feature)
            if value.shape[-2:] != target_size:
                value = functional.interpolate(
                    value,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )
            projected.append(value)
        fused = self.fusion(torch.cat(projected, dim=1))
        fused = apply_support(fused, validity4)
        logits = functional.interpolate(
            self.classifier(fused),
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )
        optical_region = apply_support(
            self.optical_region_projection(optical_feature4),
            validity4,
        )
        fused_region = apply_support(
            self.fused_region_projection(fused),
            validity4,
        )
        return logits, optical_region, fused_region


class OAAuxSegModel(nn.Module):
    optical_channels = BACKBONE_STAGE_CHANNELS
    auxiliary_channels = STAGE_CHANNELS
    modality_weight_map_strides = STAGE_STRIDES

    def __init__(
        self,
        config: OAAuxSegConfig,
        registry: ModelRegistry,
        *,
        backbone_weights: Path | None,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.config = config
        self.variant = config.variant
        self.registry = registry
        self.modality_order = registry.auxiliary_order
        self.modality_weight_order = registry.modality_weight_order
        self.modality_index = {
            name: index for index, name in enumerate(self.modality_order)
        }
        self.region_feature_dim = registry.region_feature_dim(
            config.region_projection_dim
        )
        self.gradient_checkpointing = bool(gradient_checkpointing)

        base = convnext_small(
            weights=None,
            stochastic_depth_prob=config.optical_stochastic_depth,
        )
        self.backbone_sha256: str | None = None
        if backbone_weights is not None:
            path = Path(backbone_weights)
            if not path.is_file():
                raise FileNotFoundError(
                    f"ConvNeXt-Small 权重不存在：{path}"
                )
            digest = sha256_file(path)
            if digest != EXPECTED_BACKBONE_SHA256:
                raise ValueError(
                    "ConvNeXt-Small 权重 SHA-256 不符合固定合同："
                    f"{digest}"
                )
            state = torch.load(
                path,
                map_location="cpu",
                weights_only=True,
            )
            if not isinstance(state, dict) or not all(
                isinstance(key, str) and isinstance(value, Tensor)
                for key, value in state.items()
            ):
                raise ValueError(
                    "backbone 权重必须是 torchvision 官方 state_dict"
                )
            base.load_state_dict(state, strict=True)
            self.backbone_sha256 = digest

        official_stem = base.features[0]
        official_conv = official_stem[0]
        if not isinstance(official_conv, nn.Conv2d):
            raise RuntimeError("torchvision ConvNeXt stem 合同发生变化")
        self.optical_rgb_stem = official_conv
        self.optical_stem_norm = official_stem[1]
        self.backbone = nn.ModuleList(
            list(base.features.children())[1:]
        )

        self.optical_stems = nn.ModuleDict()
        self._optical_stem_keys: dict[tuple[str, ...], str] = {}
        self.optical_rgb_mapping: dict[
            tuple[str, ...], tuple[int, int, int]
        ] = {}
        for signature in registry.optical_signatures:
            rgb_indices = optical_rgb_indices(signature)
            key = signature_key(signature)
            self._optical_stem_keys[signature] = key
            self.optical_rgb_mapping[signature] = rgb_indices
            self.optical_stems[key] = SignatureStem(
                len(signature),
                rgb_indices=rgb_indices,
                official_conv=official_conv,
            )

        self.direct_stems = nn.ModuleDict()
        self._direct_stem_keys: dict[
            tuple[tuple[str, ...], tuple[str, ...]], str
        ] = {}
        for signature in registry.optical_signatures:
            available = registry.available_auxiliaries.get(signature, ())
            for count in range(1, len(available) + 1):
                for subset in itertools.combinations(available, count):
                    auxiliary_contract = tuple(
                        (name, registry.auxiliary_channels[name])
                        for name in subset
                    )
                    key = direct_signature_key(
                        signature,
                        auxiliary_contract,
                    )
                    self._direct_stem_keys[(signature, subset)] = key
                    channels = len(signature) + sum(
                        len(registry.auxiliary_channels[name])
                        for name in subset
                    )
                    self.direct_stems[key] = SignatureStem(
                        channels,
                        rgb_indices=optical_rgb_indices(signature),
                        official_conv=official_conv,
                    )

        self.auxiliary_adapters = nn.ModuleDict(
            {
                name: AuxiliaryStem(
                    len(registry.auxiliary_channels[name])
                )
                for name in self.modality_order
            }
        )
        rates = torch.linspace(
            0.0,
            config.auxiliary_drop_path,
            sum(STAGE_DEPTHS),
        ).tolist()
        self.auxiliary_stages = nn.ModuleList()
        offset = 0
        for channels, depth, ratio in zip(
            STAGE_CHANNELS,
            STAGE_DEPTHS,
            MSPA_MLP_RATIOS,
            strict=True,
        ):
            self.auxiliary_stages.append(
                MSPAStage(
                    channels,
                    depth=depth,
                    mlp_ratio=ratio,
                    drop_path_rates=rates[offset : offset + depth],
                )
            )
            offset += depth
        self.auxiliary_downsamples = nn.ModuleList(
            AuxiliaryDownsample(
                STAGE_CHANNELS[index],
                STAGE_CHANNELS[index + 1],
            )
            for index in range(3)
        )
        self.equal_selector = EqualSpatialSelector(
            self.modality_order
        )
        self.cmnext_selectors = nn.ModuleList(
            CMNeXtSpatialSelector(channels, self.modality_order)
            for channels in STAGE_CHANNELS
        )
        self.quality_selectors = nn.ModuleList(
            NullAwareSpatialSelector(channels, self.modality_order)
            for channels in STAGE_CHANNELS
        )
        self.mean_fusions = nn.ModuleList(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=1,
                bias=False,
            )
            for channels in STAGE_CHANNELS
        )
        self.progressive_fusions = nn.ModuleList(
            ProgressiveFusion(channels, heads)
            for channels, heads in zip(
                STAGE_CHANNELS,
                FFM_HEADS,
                strict=True,
            )
        )
        self.decoder = SegmentationDecoder(
            STAGE_CHANNELS,
            decoder_dim=config.decoder_dim,
            region_projection_dim=config.region_projection_dim,
            dropout=config.decoder_dropout,
        )

    def set_gradient_checkpointing(self, enabled: bool) -> None:
        self.gradient_checkpointing = bool(enabled)

    def model_contract(self) -> dict[str, object]:
        return {
            "config": self.config.to_dict(),
            "registry": self.registry.to_dict(),
            "architecture": ARCHITECTURE_NAME,
            "backbone": BACKBONE_CONTRACT_NAME,
            "backbone_stage_channels": list(BACKBONE_STAGE_CHANNELS),
            "backbone_stage_depths": list(BACKBONE_STAGE_DEPTHS),
            "backbone_sha256": self.backbone_sha256,
            "optical_stem_contract": dict(OPTICAL_STEM_CONTRACT),
            "auxiliary_stem_contract": dict(
                AUXILIARY_STEM_CONTRACT
            ),
            "optical_rgb_mapping": [
                {
                    "channel_names": list(signature),
                    "rgb_indices": list(
                        self.optical_rgb_mapping[signature]
                    ),
                    "extra_indices": list(
                        self.optical_stems[
                            self._optical_stem_keys[signature]
                        ].extra_indices
                    ),
                }
                for signature in self.registry.optical_signatures
            ],
            "fusion_contract": dict(FUSION_CONTRACT),
            "decoder_contract": dict(DECODER_CONTRACT),
            "validity_contract": dict(VALIDITY_CONTRACT),
            "modality_weight_order": list(
                self.modality_weight_order
            ),
            "modality_weight_map_strides": list(
                self.modality_weight_map_strides
            ),
            "region_feature_dim": self.region_feature_dim,
        }

    def _batch_modalities(
        self,
        batch: OAAuxSegBatch,
    ) -> tuple[str, ...]:
        unknown = set(batch.auxiliaries) - set(self.modality_order)
        if unknown:
            raise ValueError(
                f"batch 辅助模态未出现在模型 registry：{sorted(unknown)}"
            )
        ordered = tuple(
            name
            for name in self.modality_order
            if name in batch.auxiliaries
        )
        for name in ordered:
            actual = batch.auxiliaries[name].channel_names
            expected = self.registry.auxiliary_channels[name]
            if actual != expected:
                raise ValueError(
                    f"{name}: 通道合同 {actual}，预期 {expected}"
                )
        return ordered

    @staticmethod
    def _stack_validities(
        values: Sequence[FeatureValidity],
    ) -> FeatureValidity:
        return FeatureValidity(
            support=torch.cat(
                [item.support for item in values],
                dim=0,
            ),
            coverage=torch.cat(
                [item.coverage for item in values],
                dim=0,
            ),
        )

    def _encode_optical_stems(
        self,
        batch: OAAuxSegBatch,
    ) -> tuple[Tensor, FeatureValidity, FeatureValidity]:
        grouped: dict[tuple[str, ...], list[int]] = defaultdict(list)
        for index, names in enumerate(batch.optical_channel_names):
            grouped[names].append(index)
        outputs: list[Tensor | None] = [None] * batch.batch_size
        raw_validities: list[FeatureValidity | None] = (
            [None] * batch.batch_size
        )
        stage_validities: list[FeatureValidity | None] = (
            [None] * batch.batch_size
        )
        for signature in sorted(grouped):
            key = self._optical_stem_keys.get(signature)
            if key is None:
                raise ValueError(
                    f"未注册光学通道签名：{signature}"
                )
            indices = grouped[signature]
            feature, raw_validity, stage_validity = (
                self.optical_stems[key](
                    torch.stack(
                        [batch.optical[index] for index in indices]
                    ),
                    torch.stack(
                        [
                            batch.optical_pixel_valid[index]
                            for index in indices
                        ]
                    ),
                    torch.stack(
                        [
                            batch.optical_channel_valid[index]
                            for index in indices
                        ]
                    ),
                    normalization=self.optical_stem_norm,
                    rgb_projection=self.optical_rgb_stem,
                )
            )
            for local_index, sample_index in enumerate(indices):
                outputs[sample_index] = feature[local_index]
                raw_validities[sample_index] = FeatureValidity(
                    support=raw_validity.support[
                        local_index : local_index + 1
                    ],
                    coverage=raw_validity.coverage[
                        local_index : local_index + 1
                    ],
                )
                stage_validities[sample_index] = FeatureValidity(
                    support=stage_validity.support[
                        local_index : local_index + 1
                    ],
                    coverage=stage_validity.coverage[
                        local_index : local_index + 1
                    ],
                )
        if (
            any(value is None for value in outputs)
            or any(value is None for value in raw_validities)
            or any(value is None for value in stage_validities)
        ):
            raise RuntimeError("光学 stem 未覆盖全部样本")
        return (
            torch.stack(outputs),  # type: ignore[arg-type]
            self._stack_validities(  # type: ignore[arg-type]
                raw_validities
            ),
            self._stack_validities(  # type: ignore[arg-type]
                stage_validities
            ),
        )

    @staticmethod
    def _auxiliary_lookup(
        auxiliaries: Mapping[str, PackedAuxiliary],
    ) -> dict[str, dict[int, int]]:
        return {
            name: {
                int(sample_index): local_index
                for local_index, sample_index in enumerate(
                    packed.sample_indices.tolist()
                )
            }
            for name, packed in auxiliaries.items()
        }

    def _encode_direct_stems(
        self,
        batch: OAAuxSegBatch,
    ) -> tuple[Tensor, FeatureValidity, FeatureValidity]:
        lookup = self._auxiliary_lookup(batch.auxiliaries)
        batch_modalities = self._batch_modalities(batch)
        outputs: list[Tensor] = []
        raw_validities: list[FeatureValidity] = []
        stage_validities: list[FeatureValidity] = []
        for sample_index in range(batch.batch_size):
            signature = batch.optical_channel_names[sample_index]
            optical_values = batch.optical[
                sample_index
            ].unsqueeze(0)
            optical_pixel_valid = batch.optical_pixel_valid[
                sample_index
            ].unsqueeze(0)
            optical_channel_valid = batch.optical_channel_valid[
                sample_index
            ].unsqueeze(0)
            anchor = validity_from_channels(
                optical_pixel_valid,
                optical_channel_valid,
            )
            active_names: list[str] = []
            for name in batch_modalities:
                local_index = lookup[name].get(sample_index)
                if local_index is None:
                    continue
                packed = batch.auxiliaries[name]
                local_valid = (
                    packed.pixel_valid[
                        local_index : local_index + 1
                    ].to(torch.bool)
                    & packed.channel_valid[
                        local_index : local_index + 1
                    ].to(torch.bool)[:, :, None, None]
                )
                if bool(local_valid.any().item()):
                    active_names.append(name)
            names = tuple(active_names)
            if not names:
                key = self._optical_stem_keys[signature]
                feature, raw_validity, stage_validity = (
                    self.optical_stems[key](
                        optical_values,
                        optical_pixel_valid,
                        optical_channel_valid,
                        normalization=self.optical_stem_norm,
                        rgb_projection=self.optical_rgb_stem,
                    )
                )
            else:
                key = self._direct_stem_keys.get((signature, names))
                if key is None:
                    raise ValueError(
                        f"未注册 direct concat 签名：{signature} + {names}"
                    )
                values = [optical_values]
                pixel_valid = [optical_pixel_valid]
                channel_valid = [optical_channel_valid]
                for name in names:
                    packed = batch.auxiliaries[name]
                    local_index = lookup[name][sample_index]
                    values.append(
                        packed.values[
                            local_index : local_index + 1
                        ]
                    )
                    pixel_valid.append(
                        packed.pixel_valid[
                            local_index : local_index + 1
                        ]
                    )
                    channel_valid.append(
                        packed.channel_valid[
                            local_index : local_index + 1
                        ]
                    )
                feature, raw_validity, stage_validity = (
                    self.direct_stems[key](
                        torch.cat(values, dim=1),
                        torch.cat(pixel_valid, dim=1),
                        torch.cat(channel_valid, dim=1),
                        normalization=self.optical_stem_norm,
                        rgb_projection=self.optical_rgb_stem,
                        anchor_validity=anchor,
                    )
                )
            outputs.append(feature[0])
            raw_validities.append(raw_validity)
            stage_validities.append(stage_validity)
        return (
            torch.stack(outputs),
            self._stack_validities(raw_validities),
            self._stack_validities(stage_validities),
        )

    def _run_backbone_stage(
        self,
        stage: nn.Module,
        values: Tensor,
        validity: FeatureValidity,
        *,
        checkpoint_blocks: bool,
    ) -> Tensor:
        for block in stage.children():
            if (
                checkpoint_blocks
                and self.training
                and self.gradient_checkpointing
                and torch.is_grad_enabled()
            ):
                values = activation_checkpoint(
                    block,
                    values,
                    use_reentrant=False,
                )
            else:
                values = block(values)
            values = apply_support(values, validity)
        return values

    @staticmethod
    def _transition(
        layer: nn.Module,
        values: Tensor,
        validity: FeatureValidity,
    ) -> tuple[Tensor, FeatureValidity]:
        values = layer(apply_support(values, validity))
        target = resample_validity(
            validity,
            values.shape[-2:],
        )
        return apply_support(values, target), target

    def _encode_auxiliary_stems(
        self,
        batch: OAAuxSegBatch,
    ) -> dict[str, SparseStageFeature]:
        streams: dict[str, SparseStageFeature] = {}
        for modality in self._batch_modalities(batch):
            packed = batch.auxiliaries[modality]
            expected = self.registry.auxiliary_channels.get(modality)
            if expected != packed.channel_names:
                raise ValueError(
                    f"{modality}: 通道合同 {packed.channel_names}，"
                    f"预期 {expected}"
                )
            feature, validity = self.auxiliary_adapters[modality](
                packed.values,
                packed.pixel_valid,
                packed.channel_valid,
            )
            streams[modality] = SparseStageFeature(
                sample_indices=packed.sample_indices,
                feature=feature,
                validity=validity,
            )
        return streams

    def _downsample_streams(
        self,
        streams: Mapping[str, SparseStageFeature],
        *,
        stage_index: int,
    ) -> dict[str, SparseStageFeature]:
        layer = self.auxiliary_downsamples[stage_index]
        result: dict[str, SparseStageFeature] = {}
        for modality in self.modality_order:
            item = streams.get(modality)
            if item is None:
                continue
            feature, validity = layer(
                item.feature,
                item.validity,
            )
            result[modality] = SparseStageFeature(
                sample_indices=item.sample_indices,
                feature=feature,
                validity=validity,
            )
        return result

    def _propagate_auxiliary(
        self,
        streams: Mapping[str, SparseStageFeature],
        rectified_auxiliary: Tensor,
    ) -> dict[str, SparseStageFeature]:
        result: dict[str, SparseStageFeature] = {}
        for modality in self.modality_order:
            item = streams.get(modality)
            if item is None:
                continue
            residual = rectified_auxiliary.index_select(
                0,
                item.sample_indices,
            )
            feature = apply_support(
                item.feature + residual,
                item.validity,
            )
            result[modality] = SparseStageFeature(
                sample_indices=item.sample_indices,
                feature=feature,
                validity=item.validity,
            )
        return result

    def _select_auxiliary(
        self,
        *,
        stage_index: int,
        streams: Mapping[str, SparseStageFeature],
        optical: Tensor,
        optical_validity: FeatureValidity,
    ):
        if self.variant == "mean_auxiliary_fusion":
            return self.equal_selector(
                streams,
                optical,
                optical_validity,
            )
        if self.variant == "cmnext_injection":
            return self.cmnext_selectors[stage_index](
                streams,
                optical,
                optical_validity,
            )
        return self.quality_selectors[stage_index](
            streams,
            optical,
            optical_validity,
        )

    def _fuse_stage(
        self,
        *,
        stage_index: int,
        streams: Mapping[str, SparseStageFeature],
        optical: Tensor,
        optical_validity: FeatureValidity,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        selected = self._select_auxiliary(
            stage_index=stage_index,
            streams=streams,
            optical=optical,
            optical_validity=optical_validity,
        )
        if not bool(selected.validity.support.any().item()):
            return (
                optical,
                torch.zeros_like(optical),
                optical,
                selected.weight_map,
            )
        auxiliary = self.auxiliary_stages[stage_index](
            selected.feature,
            selected.validity,
            checkpoint_blocks=self.gradient_checkpointing,
        )
        if self.variant == "mean_auxiliary_fusion":
            delta = apply_support(
                self.mean_fusions[stage_index](auxiliary),
                selected.validity,
            )
            fused = apply_support(
                optical + delta,
                optical_validity,
            )
            return (
                fused,
                auxiliary,
                fused,
                selected.weight_map,
            )
        propagated, rectified_auxiliary, decoder = (
            self.progressive_fusions[stage_index](
                optical,
                auxiliary,
                optical_validity,
                selected.validity,
                checkpoint_fusion=self.gradient_checkpointing,
            )
        )
        return (
            propagated,
            rectified_auxiliary,
            decoder,
            selected.weight_map,
        )

    def _raw_auxiliary_coverages(
        self,
        batch: OAAuxSegBatch,
        *,
        target_size: tuple[int, int],
    ) -> dict[str, tuple[Tensor, FeatureValidity]]:
        result: dict[str, tuple[Tensor, FeatureValidity]] = {}
        for modality in self._batch_modalities(batch):
            packed = batch.auxiliaries[modality]
            validity = validity_from_channels(
                packed.pixel_valid,
                packed.channel_valid,
            )
            result[modality] = (
                packed.sample_indices,
                resample_validity(validity, target_size),
            )
        return result

    def _baseline_weight_map(
        self,
        batch: OAAuxSegBatch,
        optical_validity: FeatureValidity,
    ) -> Tensor:
        if self.variant == "optical_only":
            return optical_only_weight_map(
                optical_validity,
                weight_columns=len(self.modality_weight_order),
            )
        return availability_weight_map(
            self._raw_auxiliary_coverages(
                batch,
                target_size=optical_validity.spatial_size,
            ),
            modality_order=self.modality_order,
            optical_validity=optical_validity,
        )

    def _active_modalities(
        self,
        batch: OAAuxSegBatch,
    ) -> list[tuple[str, ...]]:
        values: list[list[str]] = [
            [] for _ in range(batch.batch_size)
        ]
        for modality in self._batch_modalities(batch):
            for sample_index in batch.auxiliaries[
                modality
            ].sample_indices.tolist():
                values[int(sample_index)].append(modality)
        return [tuple(items) for items in values]

    def forward(
        self,
        batch: OAAuxSegBatch,
        *,
        return_regions: bool = False,
    ) -> OAAuxSegOutput:
        if self.variant == "direct_concat":
            stem, raw_validity, validity4 = (
                self._encode_direct_stems(batch)
            )
        else:
            stem, raw_validity, validity4 = (
                self._encode_optical_stems(batch)
            )
        active_modalities = self._active_modalities(batch)
        use_auxiliary_path = self.variant not in {
            "optical_only",
            "direct_concat",
        }
        streams = (
            self._encode_auxiliary_stems(batch)
            if use_auxiliary_path
            else {}
        )

        raw4 = self._run_backbone_stage(
            self.backbone[0],
            stem,
            validity4,
            checkpoint_blocks=False,
        )
        optical_feature4 = raw4
        if use_auxiliary_path:
            propagated4, auxiliary4, decoder4, weight4 = (
                self._fuse_stage(
                    stage_index=0,
                    streams=streams,
                    optical=raw4,
                    optical_validity=validity4,
                )
            )
            streams = self._propagate_auxiliary(
                streams,
                auxiliary4,
            )
            streams = self._downsample_streams(
                streams,
                stage_index=0,
            )
        else:
            propagated4 = decoder4 = raw4
            weight4 = self._baseline_weight_map(
                batch,
                validity4,
            )

        value8, validity8 = self._transition(
            self.backbone[1],
            propagated4,
            validity4,
        )
        raw8 = self._run_backbone_stage(
            self.backbone[2],
            value8,
            validity8,
            checkpoint_blocks=False,
        )
        if use_auxiliary_path:
            propagated8, auxiliary8, decoder8, weight8 = (
                self._fuse_stage(
                    stage_index=1,
                    streams=streams,
                    optical=raw8,
                    optical_validity=validity8,
                )
            )
            streams = self._propagate_auxiliary(
                streams,
                auxiliary8,
            )
            streams = self._downsample_streams(
                streams,
                stage_index=1,
            )
        else:
            propagated8 = decoder8 = raw8
            weight8 = self._baseline_weight_map(
                batch,
                validity8,
            )

        value16, validity16 = self._transition(
            self.backbone[3],
            propagated8,
            validity8,
        )
        raw16 = self._run_backbone_stage(
            self.backbone[4],
            value16,
            validity16,
            checkpoint_blocks=True,
        )
        if use_auxiliary_path:
            propagated16, auxiliary16, decoder16, weight16 = (
                self._fuse_stage(
                    stage_index=2,
                    streams=streams,
                    optical=raw16,
                    optical_validity=validity16,
                )
            )
            streams = self._propagate_auxiliary(
                streams,
                auxiliary16,
            )
            streams = self._downsample_streams(
                streams,
                stage_index=2,
            )
        else:
            propagated16 = decoder16 = raw16
            weight16 = self._baseline_weight_map(
                batch,
                validity16,
            )

        value32, validity32 = self._transition(
            self.backbone[5],
            propagated16,
            validity16,
        )
        raw32 = self._run_backbone_stage(
            self.backbone[6],
            value32,
            validity32,
            checkpoint_blocks=False,
        )
        if use_auxiliary_path:
            _, _, decoder32, weight32 = self._fuse_stage(
                stage_index=3,
                streams=streams,
                optical=raw32,
                optical_validity=validity32,
            )
        else:
            decoder32 = raw32
            weight32 = self._baseline_weight_map(
                batch,
                validity32,
            )

        weight_maps = (weight4, weight8, weight16, weight32)
        optical_validities = (
            validity4,
            validity8,
            validity16,
            validity32,
        )
        weights = summarize_weight_maps(
            weight_maps,
            optical_validities,
        )
        logits, optical_map, fused_map = self.decoder(
            (decoder4, decoder8, decoder16, decoder32),
            optical_feature4=optical_feature4,
            output_size=batch.spatial_size,
            validity4=validity4,
        )
        logits = torch.where(
            raw_validity.support,
            logits,
            logits.new_full((), -20.0),
        )
        probability = torch.sigmoid(logits)
        probability = torch.where(
            raw_validity.support,
            probability,
            torch.zeros_like(probability),
        )
        max_probability = probability.flatten(1).amax(dim=1)
        any_valid = raw_validity.support.flatten(1).any(dim=1)
        no_target = torch.where(
            any_valid,
            1.0 - max_probability,
            torch.ones_like(max_probability),
        )
        candidate_regions = None
        region_features = None
        if return_regions:
            candidate_regions, region_features = (
                extract_regions_and_features(
                    probability=probability,
                    optical_feature=optical_map,
                    fused_feature=fused_map,
                    modality_weights=weights,
                    expected_feature_dim=self.region_feature_dim,
                    threshold=self.config.region_threshold,
                    min_area=self.config.min_region_area,
                )
            )
        return OAAuxSegOutput(
            mask_logits=logits,
            mask_probability=probability,
            no_target_score=no_target,
            modality_names=self.modality_weight_order,
            modality_weights=weights,
            modality_weight_map_strides=(
                self.modality_weight_map_strides
            ),
            modality_weight_maps=tuple(
                item.float() for item in weight_maps
            ),
            candidate_regions=candidate_regions,
            region_features=region_features,
            active_modalities=active_modalities,
        )
