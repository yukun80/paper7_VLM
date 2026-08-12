"""Phase 2 从语义 mask 确定性提取区域，并在稳定空间特征上 mask pooling。"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as functional
from scipy import ndimage
from torch import Tensor

from .contracts import CandidateRegion, REGION_GEOMETRY_DIM


def _pool_feature(feature: Tensor, mask: Tensor) -> Tensor:
    resized = functional.interpolate(
        mask[None, None].to(device=feature.device, dtype=torch.float32),
        size=feature.shape[-2:],
        mode="area",
    )[0, 0]
    denominator = resized.sum().clamp_min(1e-6)
    return (feature * resized.unsqueeze(0)).sum(dim=(1, 2)) / denominator


@torch.no_grad()
def extract_regions_and_features(
    *,
    probability: Tensor,
    optical_feature: Tensor,
    fused_feature: Tensor,
    modality_weights: Tensor,
    expected_feature_dim: int,
    threshold: float,
    min_area: int,
) -> tuple[list[list[CandidateRegion]], list[Tensor]]:
    if probability.ndim != 4 or probability.shape[1] != 1:
        raise ValueError("probability 必须是 B1HW")
    batch_size = probability.shape[0]
    if (
        optical_feature.ndim != 4
        or fused_feature.ndim != 4
        or modality_weights.ndim != 2
        or optical_feature.shape[0] != batch_size
        or fused_feature.shape[0] != batch_size
        or modality_weights.shape[0] != batch_size
    ):
        raise ValueError("区域特征输入的 batch 合同不一致")
    derived_feature_dim = (
        optical_feature.shape[1]
        + fused_feature.shape[1]
        + REGION_GEOMETRY_DIM
        + modality_weights.shape[1]
    )
    if derived_feature_dim != expected_feature_dim:
        raise ValueError(
            "region feature 合同不一致："
            f"derived={derived_feature_dim} expected={expected_feature_dim}"
        )
    batch_regions: list[list[CandidateRegion]] = []
    batch_features: list[Tensor] = []
    height, width = probability.shape[-2:]
    structure = np.ones((3, 3), dtype=np.uint8)
    for batch_index in range(probability.shape[0]):
        probability_cpu = (
            probability[batch_index, 0].detach().float().cpu().numpy()
        )
        labels, count = ndimage.label(
            probability_cpu >= threshold, structure=structure
        )
        components: list[dict[str, object]] = []
        for label_id in range(1, count + 1):
            mask_np = labels == label_id
            area = int(mask_np.sum())
            if area < min_area:
                continue
            ys, xs = np.nonzero(mask_np)
            x0, x1 = int(xs.min()), int(xs.max()) + 1
            y0, y1 = int(ys.min()), int(ys.max()) + 1
            confidence = float(probability_cpu[mask_np].mean())
            components.append(
                {
                    "mask": mask_np,
                    "area": area,
                    "bbox": (x0, y0, x1, y1),
                    "centroid": (float(xs.mean()), float(ys.mean())),
                    "confidence": confidence,
                }
            )
        components.sort(
            key=lambda item: (
                -int(item["area"]),
                int(item["bbox"][1]),
                int(item["bbox"][0]),
            )
        )
        regions: list[CandidateRegion] = []
        features: list[Tensor] = []
        for region_id, item in enumerate(components):
            mask = torch.from_numpy(
                np.asarray(item["mask"], dtype=np.bool_)
            )
            bbox = tuple(int(value) for value in item["bbox"])
            centroid = tuple(float(value) for value in item["centroid"])
            confidence = float(item["confidence"])
            optical_pooled = _pool_feature(
                optical_feature[batch_index], mask
            )
            fused_pooled = _pool_feature(fused_feature[batch_index], mask)
            geometry = torch.tensor(
                [
                    bbox[0] / width,
                    bbox[1] / height,
                    bbox[2] / width,
                    bbox[3] / height,
                    centroid[0] / max(width - 1, 1),
                    centroid[1] / max(height - 1, 1),
                    int(item["area"]) / (height * width),
                    confidence,
                ],
                device=fused_feature.device,
                dtype=fused_feature.dtype,
            )
            if geometry.shape != (REGION_GEOMETRY_DIM,):
                raise RuntimeError("region geometry 维度与公共合同不一致")
            features.append(
                torch.cat(
                    (
                        optical_pooled,
                        fused_pooled,
                        geometry,
                        modality_weights[batch_index].to(fused_feature.dtype),
                    )
                )
            )
            regions.append(
                CandidateRegion(
                    region_id=region_id,
                    mask=mask,
                    bbox_xyxy=bbox,
                    centroid_xy=centroid,
                    area_pixels=int(item["area"]),
                    confidence=confidence,
                )
            )
        if features:
            feature_tensor = torch.stack(features)
        else:
            feature_tensor = fused_feature.new_empty(
                (0, expected_feature_dim)
            )
        if feature_tensor.shape[1:] != (expected_feature_dim,):
            raise RuntimeError(
                "region feature 维度错误："
                f"预期 {expected_feature_dim}，实际 {tuple(feature_tensor.shape)}"
            )
        batch_regions.append(regions)
        batch_features.append(feature_tensor)
    return batch_regions, batch_features
