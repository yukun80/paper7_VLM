"""Phase 2 v4 的完整 DELIVER 式 MSPA、空间选择、FRM 与 FFM。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
import torch.nn.functional as functional
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from .validity import (
    FeatureValidity,
    apply_support,
    masked_average_pool2d,
    masked_global_average,
    masked_global_max,
    resample_validity,
)


STAGE_STRIDES = (4, 8, 16, 32)
STAGE_CHANNELS = (96, 192, 384, 768)
STAGE_DEPTHS = (3, 3, 27, 3)
MSPA_MLP_RATIOS = (8, 8, 4, 4)
FFM_HEADS = (3, 6, 12, 24)
FUSION_CONTRACT = {
    "name": "deliver_full_four_stage_v1",
    "stages": list(STAGE_STRIDES),
    "channels": list(STAGE_CHANNELS),
    "mspa_depths": list(STAGE_DEPTHS),
    "mspa_mlp_ratios": list(MSPA_MLP_RATIOS),
    "mspa_pool_kernels": [3, 7, 11],
    "auxiliary_downsample": "channel_layer_norm_conv3_stride2_padding1",
    "frm_lambda_channel": 0.5,
    "frm_lambda_spatial": 0.5,
    "ffm_cross_attention": "full_channel_context",
    "ffm_reduction": 1,
    "ffm_heads": list(FFM_HEADS),
    "ffm_initialization": "deliver_linear_trunc_normal_conv_fanout_normal",
    "selectors": {
        "cmnext_injection": "deliver_sigmoid_masked_max",
        "injection_quality": "null_aware_spatial_softmax",
        "proposed_dropout": "null_aware_spatial_softmax",
    },
    "weight_map_summary": "coverage_pool_each_stage_then_equal_stage_mean",
}


class ChannelLayerNorm2d(nn.Module):
    """逐像素仅沿通道归一化。"""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.normalization = nn.LayerNorm(channels)

    def forward(self, values: Tensor) -> Tensor:
        return self.normalization(values.permute(0, 2, 3, 1)).permute(
            0, 3, 1, 2
        )


class DropPath(nn.Module):
    """不依赖 timm 的逐样本 stochastic depth。"""

    def __init__(self, probability: float) -> None:
        super().__init__()
        if not 0.0 <= probability < 1.0:
            raise ValueError("drop path probability 必须位于 [0,1)")
        self.probability = float(probability)

    def forward(self, values: Tensor) -> Tensor:
        if not self.training or self.probability == 0.0:
            return values
        keep = 1.0 - self.probability
        shape = (values.shape[0],) + (1,) * (values.ndim - 1)
        mask = values.new_empty(shape).bernoulli_(keep)
        return values * mask / keep


class ValidityAwareMSPABlock(nn.Module):
    """DELIVER MSPA 的 validity-aware 纯 PyTorch 实现。"""

    def __init__(
        self,
        channels: int,
        *,
        mlp_ratio: int,
        drop_path: float,
    ) -> None:
        super().__init__()
        hidden = channels * mlp_ratio
        self.norm1 = ChannelLayerNorm2d(channels)
        self.context = nn.Conv2d(
            channels,
            channels,
            kernel_size=7,
            padding=3,
            groups=channels,
        )
        self.attention_projection = nn.Conv2d(
            channels, channels, kernel_size=1
        )
        self.norm2 = ChannelLayerNorm2d(channels)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.Conv2d(
                hidden,
                hidden,
                kernel_size=3,
                padding=1,
                groups=hidden,
            ),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1),
        )
        self.channel_mixer = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid(),
        )
        self.drop_path = DropPath(drop_path)
        self.layer_scale_attention = nn.Parameter(
            torch.full((channels, 1, 1), 1e-2)
        )
        self.layer_scale_mlp = nn.Parameter(
            torch.full((channels, 1, 1), 1e-2)
        )

    def forward(
        self,
        values: Tensor,
        validity: FeatureValidity,
    ) -> Tensor:
        base = apply_support(values, validity)
        normalized = apply_support(self.norm1(base), validity)
        local = apply_support(self.context(normalized), validity)
        context = local
        for kernel_size in (3, 7, 11):
            context = context + masked_average_pool2d(
                local,
                validity.coverage,
                kernel_size=kernel_size,
            )
        attended = (
            torch.sigmoid(self.attention_projection(context)) * normalized
            + normalized
        )
        values = base + self.drop_path(
            self.layer_scale_attention * attended
        )
        values = apply_support(values, validity)
        channel = masked_global_average(
            values,
            validity.coverage,
        ).unsqueeze(1)
        channel = self.channel_mixer(channel).transpose(1, 2).unsqueeze(-1)
        channel_mixed = values * channel
        mlp = self.mlp(apply_support(self.norm2(values), validity))
        values = channel_mixed + self.drop_path(
            self.layer_scale_mlp * mlp
        )
        return apply_support(values, validity)


class MSPAStage(nn.Module):
    """一个完整辅助 stage，深度与 ConvNeXt-Small 对齐。"""

    def __init__(
        self,
        channels: int,
        *,
        depth: int,
        mlp_ratio: int,
        drop_path_rates: Sequence[float],
    ) -> None:
        super().__init__()
        if len(drop_path_rates) != depth:
            raise ValueError("MSPA drop path 数量与 depth 不一致")
        self.blocks = nn.ModuleList(
            ValidityAwareMSPABlock(
                channels,
                mlp_ratio=mlp_ratio,
                drop_path=drop_path_rates[index],
            )
            for index in range(depth)
        )
        self.output_norm = ChannelLayerNorm2d(channels)

    def forward(
        self,
        values: Tensor,
        validity: FeatureValidity,
        *,
        checkpoint_blocks: bool,
    ) -> Tensor:
        for block in self.blocks:
            if (
                checkpoint_blocks
                and self.training
                and torch.is_grad_enabled()
            ):
                values = activation_checkpoint(
                    lambda current, selected=block: selected(
                        current, validity
                    ),
                    values,
                    use_reentrant=False,
                )
            else:
                values = block(values, validity)
            values = apply_support(values, validity)
        return apply_support(self.output_norm(values), validity)


class AuxiliaryDownsample(nn.Module):
    """共享的辅助模态 stage 下采样。"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            ChannelLayerNorm2d(in_channels),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
        )

    def forward(
        self,
        values: Tensor,
        validity: FeatureValidity,
    ) -> tuple[Tensor, FeatureValidity]:
        values = self.projection(apply_support(values, validity))
        target = resample_validity(validity, values.shape[-2:])
        return apply_support(values, target), target


@dataclass
class SparseStageFeature:
    """按 sample_indices 稀疏保存的单模态 stage 特征。"""

    sample_indices: Tensor
    feature: Tensor
    validity: FeatureValidity


@dataclass
class SelectorResult:
    feature: Tensor
    validity: FeatureValidity
    weight_map: Tensor


def _scatter_feature(
    item: SparseStageFeature,
    *,
    batch_size: int,
) -> tuple[Tensor, Tensor]:
    feature = item.feature.new_zeros(
        batch_size,
        item.feature.shape[1],
        item.feature.shape[2],
        item.feature.shape[3],
    ).index_copy(0, item.sample_indices, item.feature)
    coverage = item.validity.coverage.new_zeros(
        batch_size,
        1,
        item.feature.shape[2],
        item.feature.shape[3],
    ).index_copy(0, item.sample_indices, item.validity.coverage)
    return feature, coverage


def optical_only_weight_map(
    optical_validity: FeatureValidity,
    *,
    weight_columns: int,
) -> Tensor:
    weights = optical_validity.coverage.new_zeros(
        optical_validity.coverage.shape[0],
        weight_columns,
        *optical_validity.spatial_size,
    )
    weights[:, -1:] = optical_validity.support.to(weights.dtype)
    return weights


def availability_weight_map(
    coverages: Mapping[str, tuple[Tensor, FeatureValidity]],
    *,
    modality_order: Sequence[str],
    optical_validity: FeatureValidity,
) -> Tensor:
    """按像素对实际有效模态等权，供 mean/direct 及诊断使用。"""

    batch_size = optical_validity.coverage.shape[0]
    height, width = optical_validity.spatial_size
    device = optical_validity.coverage.device
    columns = len(modality_order) + 1
    active = torch.zeros(
        batch_size,
        len(modality_order),
        height,
        width,
        device=device,
        dtype=torch.bool,
    )
    for modality_index, modality in enumerate(modality_order):
        item = coverages.get(modality)
        if item is None:
            continue
        indices, validity = item
        dense = validity.coverage.new_zeros(
            batch_size, 1, height, width
        ).index_copy(0, indices, validity.coverage)
        active[:, modality_index : modality_index + 1] = (
            dense > 0
        ) & optical_validity.support
    count = active.sum(dim=1, keepdim=True)
    weights = torch.zeros(
        batch_size,
        columns,
        height,
        width,
        device=device,
        dtype=torch.float32,
    )
    weights[:, :-1] = active.float() / count.clamp_min(1).float()
    weights[:, -1:] = (
        (count == 0) & optical_validity.support
    ).float()
    return weights


class EqualSpatialSelector(nn.Module):
    """逐像素等权聚合实际有效的稀疏辅助模态。"""

    def __init__(self, modality_order: Sequence[str]) -> None:
        super().__init__()
        self.modality_order = tuple(modality_order)

    def forward(
        self,
        streams: Mapping[str, SparseStageFeature],
        optical: Tensor,
        optical_validity: FeatureValidity,
    ) -> SelectorResult:
        coverages = {
            name: (item.sample_indices, item.validity)
            for name, item in streams.items()
        }
        weight_map = availability_weight_map(
            coverages,
            modality_order=self.modality_order,
            optical_validity=optical_validity,
        )
        output = optical.new_zeros(optical.shape)
        coverage = optical_validity.coverage.new_zeros(
            optical.shape[0], 1, *optical.shape[-2:]
        )
        for index, modality in enumerate(self.modality_order):
            item = streams.get(modality)
            if item is None:
                continue
            dense_feature, dense_coverage = _scatter_feature(
                item,
                batch_size=optical.shape[0],
            )
            local_weight = weight_map[:, index : index + 1].to(
                dense_feature.dtype
            )
            output = output + local_weight * dense_coverage.to(
                dense_feature.dtype
            ) * dense_feature
            coverage = coverage + weight_map[
                :, index : index + 1
            ] * dense_coverage
        validity = FeatureValidity(
            support=coverage > 0,
            coverage=coverage.clamp(0.0, 1.0),
        )
        return SelectorResult(
            feature=apply_support(output, validity),
            validity=validity,
            weight_map=weight_map,
        )


class SpatialScorePredictor(nn.Module):
    """DELIVER PredictorConv：depthwise 3×3 + pointwise。"""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
            ),
            nn.Conv2d(channels, 1, kernel_size=1),
        )

    def forward(self, values: Tensor) -> Tensor:
        return torch.sigmoid(self.network(values))


class CMNeXtSpatialSelector(nn.Module):
    """DELIVER 式 sigmoid 增强与 masked channel-wise max。"""

    def __init__(
        self,
        channels: int,
        modality_order: Sequence[str],
    ) -> None:
        super().__init__()
        self.modality_order = tuple(modality_order)
        self.predictors = nn.ModuleDict(
            {
                name: SpatialScorePredictor(channels)
                for name in self.modality_order
            }
        )

    def forward(
        self,
        streams: Mapping[str, SparseStageFeature],
        optical: Tensor,
        optical_validity: FeatureValidity,
    ) -> SelectorResult:
        candidates: list[Tensor] = []
        coverages: list[Tensor] = []
        diagnostics: list[Tensor] = []
        minimum = torch.finfo(optical.dtype).min
        for modality in self.modality_order:
            item = streams.get(modality)
            if item is None:
                candidates.append(torch.full_like(optical, minimum))
                coverages.append(
                    optical_validity.coverage.new_zeros(
                        optical.shape[0], 1, *optical.shape[-2:]
                    )
                )
                diagnostics.append(
                    optical_validity.coverage.new_zeros(
                        optical.shape[0], 1, *optical.shape[-2:]
                    )
                )
                continue
            score = self.predictors[modality](
                apply_support(item.feature, item.validity)
            )
            dense_feature, dense_coverage = _scatter_feature(
                item,
                batch_size=optical.shape[0],
            )
            dense_score = score.new_zeros(
                optical.shape[0], 1, *optical.shape[-2:]
            ).index_copy(0, item.sample_indices, score)
            effective_coverage = (
                dense_coverage * optical_validity.coverage
            )
            support = effective_coverage > 0
            enhanced = (1.0 + dense_score) * dense_feature
            candidates.append(
                enhanced.masked_fill(~support.expand_as(enhanced), minimum)
            )
            coverages.append(effective_coverage)
            diagnostics.append(dense_score * effective_coverage)
        stacked = torch.stack(candidates, dim=1)
        coverage_stack = torch.stack(coverages, dim=1)
        any_valid = coverage_stack.amax(dim=1) > 0
        output = stacked.amax(dim=1)
        output = torch.where(
            any_valid.expand_as(output),
            output,
            torch.zeros_like(output),
        )
        coverage = coverage_stack.amax(dim=1)
        raw = torch.cat(diagnostics, dim=1).float()
        denominator = raw.sum(dim=1, keepdim=True)
        weights = torch.zeros(
            optical.shape[0],
            len(self.modality_order) + 1,
            *optical.shape[-2:],
            device=optical.device,
            dtype=torch.float32,
        )
        weights[:, :-1] = raw / denominator.clamp_min(1e-6)
        weights[:, -1:] = (
            (denominator <= 0) & optical_validity.support
        ).float()
        weights = weights * optical_validity.support.float()
        validity = FeatureValidity(
            support=coverage > 0,
            coverage=coverage.clamp(0.0, 1.0),
        )
        return SelectorResult(
            feature=apply_support(output, validity),
            validity=validity,
            weight_map=weights,
        )


class JointSpatialScorer(nn.Module):
    """全通道局部光学兼容质量评分器。"""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.optical_norm = ChannelLayerNorm2d(channels)
        self.auxiliary_norm = ChannelLayerNorm2d(channels)
        self.network = nn.Sequential(
            nn.Conv2d(channels * 3 + 2, channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
            ),
            nn.GELU(),
            nn.Conv2d(channels, 1, kernel_size=1),
        )

    def forward(
        self,
        optical: Tensor,
        auxiliary: Tensor,
        auxiliary_coverage: Tensor,
        overlap: Tensor,
    ) -> Tensor:
        optical = self.optical_norm(optical)
        auxiliary = self.auxiliary_norm(auxiliary)
        return self.network(
            torch.cat(
                (
                    optical,
                    auxiliary,
                    optical * auxiliary,
                    auxiliary_coverage.to(optical.dtype),
                    overlap.to(optical.dtype),
                ),
                dim=1,
            )
        )


class NullSpatialScorer(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = ChannelLayerNorm2d(channels)
        self.network = nn.Sequential(
            nn.Conv2d(channels + 1, channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
            ),
            nn.GELU(),
            nn.Conv2d(channels, 1, kernel_size=1),
        )

    def forward(
        self,
        optical: Tensor,
        coverage: Tensor,
    ) -> Tensor:
        return self.network(
            torch.cat(
                (self.norm(optical), coverage.to(optical.dtype)),
                dim=1,
            )
        )


class NullAwareSpatialSelector(nn.Module):
    """逐 stage、逐像素的 sparse null-aware softmax。"""

    def __init__(
        self,
        channels: int,
        modality_order: Sequence[str],
    ) -> None:
        super().__init__()
        self.modality_order = tuple(modality_order)
        self.scorers = nn.ModuleDict(
            {
                name: JointSpatialScorer(channels)
                for name in self.modality_order
            }
        )
        self.null_scorer = NullSpatialScorer(channels)

    def forward(
        self,
        streams: Mapping[str, SparseStageFeature],
        optical: Tensor,
        optical_validity: FeatureValidity,
    ) -> SelectorResult:
        batch_size = optical.shape[0]
        height, width = optical.shape[-2:]
        logits: list[Tensor] = []
        dense_features: list[Tensor] = []
        dense_coverages: list[Tensor] = []
        for modality in self.modality_order:
            item = streams.get(modality)
            if item is None:
                logits.append(
                    optical.new_full(
                        (batch_size, 1, height, width),
                        float("-inf"),
                        dtype=torch.float32,
                    )
                )
                dense_features.append(torch.zeros_like(optical))
                dense_coverages.append(
                    optical_validity.coverage.new_zeros(
                        batch_size, 1, height, width
                    )
                )
                continue
            optical_local = optical.index_select(0, item.sample_indices)
            optical_validity_local = optical_validity.index_select(
                item.sample_indices
            )
            overlap = (
                item.validity.coverage
                * optical_validity_local.coverage
            )
            local_logits = self.scorers[modality](
                optical_local,
                item.feature,
                item.validity.coverage,
                overlap,
            ).float()
            local_logits = local_logits.masked_fill(
                overlap <= 0,
                float("-inf"),
            )
            dense_logit = optical.new_full(
                (batch_size, 1, height, width),
                float("-inf"),
                dtype=torch.float32,
            ).index_copy(0, item.sample_indices, local_logits)
            dense_feature, dense_coverage = _scatter_feature(
                item,
                batch_size=batch_size,
            )
            logits.append(dense_logit)
            dense_features.append(dense_feature)
            dense_coverages.append(
                dense_coverage * optical_validity.coverage
            )
        null_logits = self.null_scorer(
            optical,
            optical_validity.coverage,
        ).float()
        null_logits = torch.where(
            optical_validity.support,
            null_logits,
            torch.zeros_like(null_logits),
        )
        logits.append(null_logits)
        stacked_logits = torch.cat(logits, dim=1)
        weights = torch.softmax(stacked_logits, dim=1)
        weights = weights * optical_validity.support.float()
        output = torch.zeros_like(optical)
        coverage = optical_validity.coverage.new_zeros(
            batch_size, 1, height, width
        )
        for index, (feature, modality_coverage) in enumerate(
            zip(dense_features, dense_coverages, strict=True)
        ):
            local_weight = weights[:, index : index + 1]
            output = output + (
                local_weight.to(feature.dtype)
                * modality_coverage.to(feature.dtype)
                * feature
            )
            coverage = coverage + local_weight * modality_coverage
        validity = FeatureValidity(
            support=coverage > 0,
            coverage=coverage.clamp(0.0, 1.0),
        )
        return SelectorResult(
            feature=apply_support(output, validity),
            validity=validity,
            weight_map=weights,
        )


class FeatureRectification(nn.Module):
    """完整通道容量的 DELIVER FRM，加入 validity 硬边界。"""

    def __init__(
        self,
        channels: int,
        *,
        lambda_channel: float = 0.5,
        lambda_spatial: float = 0.5,
    ) -> None:
        super().__init__()
        self.lambda_channel = float(lambda_channel)
        self.lambda_spatial = float(lambda_spatial)
        self.channel_weights = nn.Sequential(
            nn.Linear(channels * 4, channels * 4),
            nn.ReLU(inplace=True),
            nn.Linear(channels * 4, channels * 2),
            nn.Sigmoid(),
        )
        self.spatial_weights = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 2, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        optical: Tensor,
        auxiliary: Tensor,
        optical_validity: FeatureValidity,
        auxiliary_validity: FeatureValidity,
    ) -> tuple[Tensor, Tensor]:
        batch, channels = optical.shape[:2]
        channel = self.channel_weights(
            torch.cat(
                (
                    masked_global_average(
                        optical, optical_validity.coverage
                    ),
                    masked_global_average(
                        auxiliary, auxiliary_validity.coverage
                    ),
                    masked_global_max(
                        optical, optical_validity.support
                    ),
                    masked_global_max(
                        auxiliary, auxiliary_validity.support
                    ),
                ),
                dim=1,
            )
        ).reshape(batch, 2, channels, 1, 1)
        spatial = self.spatial_weights(
            torch.cat((optical, auxiliary), dim=1)
        )
        optical_delta = (
            self.lambda_channel * channel[:, 1] * auxiliary
            + self.lambda_spatial * spatial[:, 1:2] * auxiliary
        )
        auxiliary_delta = (
            self.lambda_channel * channel[:, 0] * optical
            + self.lambda_spatial * spatial[:, 0:1] * optical
        )
        rectified_optical = optical + apply_support(
            optical_delta,
            auxiliary_validity,
        )
        rectified_auxiliary = auxiliary + apply_support(
            auxiliary_delta,
            auxiliary_validity,
        )
        return (
            apply_support(rectified_optical, optical_validity),
            apply_support(rectified_auxiliary, auxiliary_validity),
        )


class FullChannelCrossAttention(nn.Module):
    """DELIVER 的 full-width KᵀV channel-context cross attention。"""

    def __init__(self, channels: int, heads: int) -> None:
        super().__init__()
        if channels % heads != 0:
            raise ValueError("FFM channels 必须可被 heads 整除")
        self.channels = channels
        self.heads = heads
        self.head_dim = channels // heads
        self.scale = self.head_dim**-0.5
        self.optical_kv = nn.Linear(channels, channels * 2, bias=False)
        self.auxiliary_kv = nn.Linear(
            channels, channels * 2, bias=False
        )

    def _heads(self, values: Tensor) -> Tensor:
        return values.reshape(
            values.shape[0],
            values.shape[1],
            self.heads,
            self.head_dim,
        ).permute(0, 2, 1, 3)

    def _context(
        self,
        values: Tensor,
        coverage: Tensor,
        projection: nn.Linear,
    ) -> Tensor:
        key, value = projection(values).chunk(2, dim=-1)
        key = self._heads(key)
        value = self._heads(value)
        weights = coverage[:, None, :, None].to(value.dtype)
        denominator = weights.sum(dim=2, keepdim=True).clamp_min(1e-6)
        context = key.transpose(-2, -1) @ (value * weights)
        context = context / denominator
        return torch.softmax(context * self.scale, dim=-2)

    def forward(
        self,
        optical: Tensor,
        auxiliary: Tensor,
        optical_coverage: Tensor,
        auxiliary_coverage: Tensor,
    ) -> tuple[Tensor, Tensor]:
        optical_query = self._heads(optical)
        auxiliary_query = self._heads(auxiliary)
        optical_context = self._context(
            optical,
            optical_coverage,
            self.optical_kv,
        )
        auxiliary_context = self._context(
            auxiliary,
            auxiliary_coverage,
            self.auxiliary_kv,
        )
        optical_cross = (
            optical_query @ auxiliary_context
        ).permute(0, 2, 1, 3).reshape(
            optical.shape[0],
            optical.shape[1],
            self.channels,
        )
        auxiliary_cross = (
            auxiliary_query @ optical_context
        ).permute(0, 2, 1, 3).reshape(
            auxiliary.shape[0],
            auxiliary.shape[1],
            self.channels,
        )
        return optical_cross, auxiliary_cross


class FullChannelCrossPath(nn.Module):
    def __init__(self, channels: int, heads: int) -> None:
        super().__init__()
        self.optical_projection = nn.Linear(channels, channels * 2)
        self.auxiliary_projection = nn.Linear(channels, channels * 2)
        self.cross_attention = FullChannelCrossAttention(
            channels, heads
        )
        self.optical_output = nn.Linear(channels * 2, channels)
        self.auxiliary_output = nn.Linear(channels * 2, channels)
        self.optical_norm = nn.LayerNorm(channels)
        self.auxiliary_norm = nn.LayerNorm(channels)

    def forward(
        self,
        optical: Tensor,
        auxiliary: Tensor,
        optical_coverage: Tensor,
        auxiliary_coverage: Tensor,
    ) -> tuple[Tensor, Tensor]:
        optical_y, optical_query = functional.relu(
            self.optical_projection(optical)
        ).chunk(2, dim=-1)
        auxiliary_y, auxiliary_query = functional.relu(
            self.auxiliary_projection(auxiliary)
        ).chunk(2, dim=-1)
        optical_cross, auxiliary_cross = self.cross_attention(
            optical_query,
            auxiliary_query,
            optical_coverage,
            auxiliary_coverage,
        )
        optical = self.optical_norm(
            optical
            + self.optical_output(
                torch.cat((optical_y, optical_cross), dim=-1)
            )
        )
        auxiliary = self.auxiliary_norm(
            auxiliary
            + self.auxiliary_output(
                torch.cat((auxiliary_y, auxiliary_cross), dim=-1)
            )
        )
        return optical, auxiliary


class FullChannelEmbedding(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.residual = nn.Conv2d(
            channels * 2, channels, kernel_size=1, bias=False
        )
        self.embedding = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1),
            ChannelLayerNorm2d(channels),
        )
        self.output_norm = ChannelLayerNorm2d(channels)

    def forward(self, values: Tensor) -> Tensor:
        return self.output_norm(
            self.residual(values) + self.embedding(values)
        )


class FeatureFusion(nn.Module):
    def __init__(self, channels: int, heads: int) -> None:
        super().__init__()
        self.cross_path = FullChannelCrossPath(channels, heads)
        self.embedding = FullChannelEmbedding(channels)
        self.apply(self._initialize_deliver_weights)

    @staticmethod
    def _initialize_deliver_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            fan_out = (
                module.kernel_size[0]
                * module.kernel_size[1]
                * module.out_channels
                / module.groups
            )
            nn.init.normal_(module.weight, std=math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        optical: Tensor,
        auxiliary: Tensor,
        optical_validity: FeatureValidity,
        auxiliary_validity: FeatureValidity,
    ) -> Tensor:
        batch, channels, height, width = optical.shape
        optical_tokens = optical.flatten(2).transpose(1, 2)
        auxiliary_tokens = auxiliary.flatten(2).transpose(1, 2)
        optical_tokens, auxiliary_tokens = self.cross_path(
            optical_tokens,
            auxiliary_tokens,
            optical_validity.coverage.flatten(2).squeeze(1),
            auxiliary_validity.coverage.flatten(2).squeeze(1),
        )
        merged = torch.cat(
            (optical_tokens, auxiliary_tokens),
            dim=-1,
        ).transpose(1, 2).reshape(
            batch, channels * 2, height, width
        )
        return apply_support(
            self.embedding(merged),
            optical_validity,
        )


class ProgressiveFusion(nn.Module):
    """FRM 校正向深层传播，FFM 输出送 decoder。"""

    def __init__(self, channels: int, heads: int) -> None:
        super().__init__()
        self.rectification = FeatureRectification(channels)
        self.fusion = FeatureFusion(channels, heads)

    def _forward_active(
        self,
        optical: Tensor,
        auxiliary: Tensor,
        optical_support: Tensor,
        optical_coverage: Tensor,
        auxiliary_support: Tensor,
        auxiliary_coverage: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        optical_validity = FeatureValidity(
            support=optical_support,
            coverage=optical_coverage,
        )
        auxiliary_validity = FeatureValidity(
            support=auxiliary_support,
            coverage=auxiliary_coverage,
        )
        rectified_optical, rectified_auxiliary = self.rectification(
            optical,
            auxiliary,
            optical_validity,
            auxiliary_validity,
        )
        decoder = self.fusion(
            rectified_optical,
            rectified_auxiliary,
            optical_validity,
            auxiliary_validity,
        )
        return rectified_optical, rectified_auxiliary, decoder

    def forward(
        self,
        optical: Tensor,
        auxiliary: Tensor,
        optical_validity: FeatureValidity,
        auxiliary_validity: FeatureValidity,
        *,
        checkpoint_fusion: bool,
    ) -> tuple[Tensor, Tensor, Tensor]:
        active = (
            auxiliary_validity.coverage.flatten(1).sum(dim=1) > 0
        ) & optical_validity.support.flatten(1).any(dim=1)
        indices = active.nonzero(as_tuple=False).flatten()
        rectified_auxiliary = torch.zeros_like(auxiliary)
        if indices.numel() == 0:
            return optical, rectified_auxiliary, optical
        arguments = (
            optical.index_select(0, indices),
            auxiliary.index_select(0, indices),
            optical_validity.support.index_select(0, indices),
            optical_validity.coverage.index_select(0, indices),
            auxiliary_validity.support.index_select(0, indices),
            auxiliary_validity.coverage.index_select(0, indices),
        )
        if (
            checkpoint_fusion
            and self.training
            and torch.is_grad_enabled()
        ):
            rectified_optical_active, rectified_auxiliary_active, decoder_active = (
                activation_checkpoint(
                    self._forward_active,
                    *arguments,
                    use_reentrant=False,
                )
            )
        else:
            rectified_optical_active, rectified_auxiliary_active, decoder_active = (
                self._forward_active(*arguments)
            )
        propagated = optical.index_copy(
            0, indices, rectified_optical_active
        )
        rectified_auxiliary = rectified_auxiliary.index_copy(
            0, indices, rectified_auxiliary_active
        )
        decoder = optical.index_copy(0, indices, decoder_active)
        return propagated, rectified_auxiliary, decoder


def summarize_weight_maps(
    weight_maps: Sequence[Tensor],
    optical_validities: Sequence[FeatureValidity],
) -> Tensor:
    """将四尺度空间图转为稳定的 [B,M+1] 全局诊断摘要。"""

    if len(weight_maps) != len(optical_validities) or not weight_maps:
        raise ValueError("weight maps 与 optical validities 数量不一致")
    stage_summaries: list[Tensor] = []
    for weight_map, validity in zip(
        weight_maps,
        optical_validities,
        strict=True,
    ):
        coverage = validity.coverage.float()
        denominator = coverage.sum(dim=(2, 3)).clamp_min(1e-6)
        summary = (
            weight_map.float() * coverage
        ).sum(dim=(2, 3)) / denominator
        no_optical = ~validity.support.flatten(1).any(dim=1)
        if no_optical.any():
            summary = summary.clone()
            summary[no_optical] = 0
            summary[no_optical, -1] = 1
        stage_summaries.append(summary)
    output = torch.stack(stage_summaries, dim=0).mean(dim=0)
    denominator = output.sum(dim=1, keepdim=True)
    normalized = output / denominator.clamp_min(1e-6)
    empty = denominator.squeeze(1) <= 0
    if empty.any():
        normalized = normalized.clone()
        normalized[empty] = 0
        normalized[empty, -1] = 1
    return normalized
