"""Phase 2 v4 的 fractional validity 表达与 masked 统计。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import Tensor


VALIDITY_CONTRACT = {
    "name": "fractional_area_v1",
    "channel_reduction": "mean",
    "support_rule": "coverage_gt_zero",
    "downsample": "area",
    "upsample": "nearest",
}


@dataclass(frozen=True)
class FeatureValidity:
    """特征空间的硬 support 与软 coverage。"""

    support: Tensor
    coverage: Tensor

    def __post_init__(self) -> None:
        if self.support.ndim != 4 or self.support.shape[1] != 1:
            raise ValueError("validity support 必须是 [B,1,H,W]")
        if self.coverage.shape != self.support.shape:
            raise ValueError("validity coverage shape 必须等于 support")
        if self.support.dtype != torch.bool:
            raise ValueError("validity support 必须是 bool")
        if not self.coverage.is_floating_point():
            raise ValueError("validity coverage 必须是浮点 tensor")

    @property
    def spatial_size(self) -> tuple[int, int]:
        return int(self.support.shape[-2]), int(self.support.shape[-1])

    def index_select(self, indices: Tensor) -> "FeatureValidity":
        return FeatureValidity(
            support=self.support.index_select(0, indices),
            coverage=self.coverage.index_select(0, indices),
        )


def validity_from_channels(
    pixel_valid: Tensor,
    channel_valid: Tensor,
) -> FeatureValidity:
    """从逐通道 validity 构造 support 与有效通道比例。"""

    if pixel_valid.ndim != 4:
        raise ValueError("pixel_valid 必须是 [B,C,H,W]")
    if channel_valid.shape != pixel_valid.shape[:2]:
        raise ValueError("channel_valid 必须是 [B,C]")
    valid = pixel_valid.to(torch.bool) & channel_valid.to(torch.bool)[
        :, :, None, None
    ]
    support = valid.any(dim=1, keepdim=True)
    coverage = valid.float().mean(dim=1, keepdim=True)
    return FeatureValidity(support=support, coverage=coverage)


def resample_validity(
    validity: FeatureValidity,
    target_size: tuple[int, int],
) -> FeatureValidity:
    """下采样保留覆盖比例，上采样保持最近邻 support 语义。"""

    target_size = (int(target_size[0]), int(target_size[1]))
    if target_size == validity.spatial_size:
        return validity
    source_h, source_w = validity.spatial_size
    target_h, target_w = target_size
    if target_h <= source_h and target_w <= source_w:
        coverage = functional.interpolate(
            validity.coverage.float(),
            size=target_size,
            mode="area",
        )
    elif target_h >= source_h and target_w >= source_w:
        coverage = functional.interpolate(
            validity.coverage.float(),
            size=target_size,
            mode="nearest",
        )
    else:
        raise ValueError(
            "validity 不支持一个空间维上采样、另一维下采样"
        )
    coverage = coverage.clamp_(0.0, 1.0)
    return FeatureValidity(support=coverage > 0, coverage=coverage)


def validity_like(
    validity: FeatureValidity,
    feature: Tensor,
) -> FeatureValidity:
    return resample_validity(
        validity,
        (int(feature.shape[-2]), int(feature.shape[-1])),
    )


def apply_support(values: Tensor, validity: FeatureValidity) -> Tensor:
    if values.shape[0] != validity.support.shape[0] or values.shape[-2:] != (
        validity.support.shape[-2:]
    ):
        raise ValueError("feature 与 validity shape 不匹配")
    return values * validity.support.to(values.dtype)


def masked_global_average(values: Tensor, coverage: Tensor) -> Tensor:
    """coverage 加权的全局平均，零覆盖样本稳定返回零。"""

    weights = coverage.to(values.dtype)
    denominator = weights.sum(dim=(2, 3)).clamp_min(1e-6)
    return (values * weights).sum(dim=(2, 3)) / denominator


def masked_global_max(values: Tensor, support: Tensor) -> Tensor:
    """support 内全局最大，零覆盖样本稳定返回零。"""

    minimum = torch.finfo(values.dtype).min
    masked = values.masked_fill(~support, minimum)
    output = masked.amax(dim=(2, 3))
    any_valid = support.flatten(1).any(dim=1, keepdim=True)
    return torch.where(any_valid, output, torch.zeros_like(output))


def masked_statistics(
    values: Tensor,
    coverage: Tensor,
) -> tuple[Tensor, Tensor]:
    """coverage 加权的逐通道 mean/std。"""

    weights = coverage.to(values.dtype)
    denominator = weights.sum(dim=(2, 3)).clamp_min(1e-6)
    mean = (values * weights).sum(dim=(2, 3)) / denominator
    variance = (
        (values - mean[:, :, None, None]).square() * weights
    ).sum(dim=(2, 3)) / denominator
    return mean, variance.clamp_min(0).sqrt()


def masked_average_pool2d(
    values: Tensor,
    coverage: Tensor,
    *,
    kernel_size: int,
) -> Tensor:
    """忽略无效像素的 stride-1 局部平均。"""

    padding = kernel_size // 2
    weights = coverage.to(values.dtype)
    numerator = functional.avg_pool2d(
        values * weights,
        kernel_size=kernel_size,
        stride=1,
        padding=padding,
        count_include_pad=False,
    )
    denominator = functional.avg_pool2d(
        weights,
        kernel_size=kernel_size,
        stride=1,
        padding=padding,
        count_include_pad=False,
    )
    output = numerator / denominator.clamp_min(1e-6)
    return torch.where(denominator > 0, output, torch.zeros_like(output))
