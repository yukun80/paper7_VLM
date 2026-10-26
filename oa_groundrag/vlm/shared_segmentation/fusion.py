"""Validity-gated residual injection into pre-merge Qwen visual features."""
from collections.abc import Mapping, Sequence
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from .contracts import AuxiliaryObservation, MODALITIES, validate_auxiliaries

class AuxiliaryFusion(nn.Module):
    def __init__(self, width: int, bottleneck: int = 32):
        super().__init__()
        self.adapters = nn.ModuleDict({name: nn.Sequential(nn.Linear(2, bottleneck), nn.GELU(), nn.Linear(bottleneck, width)) for name in MODALITIES})
        for layer in self.adapters.values():
            nn.init.zeros_(layer[-1].weight)
            nn.init.zeros_(layer[-1].bias)

    def forward(self, raw: Tensor, grid_thw: Tensor,
                observations: Sequence[Mapping[str, AuxiliaryObservation]],
                image_sizes: Sequence[tuple[int, int]], merge_size: int) -> Tensor:
        grids = grid_thw.tolist()
        if len(grids) != len(observations) or len(grids) != len(image_sizes):
            raise ValueError("Auxiliary/image count mismatch")
        chunks = raw.split([t*h*w for t,h,w in grids])
        fused = []
        for features, (t,h,w), aux, hw in zip(chunks, grids, observations, image_sizes):
            validate_auxiliaries(aux, hw)
            if t != 1 or h % merge_size or w % merge_size:
                raise ValueError("Only still images with divisible spatial grids are supported")
            delta = torch.zeros((h,w,raw.shape[-1]), device=raw.device, dtype=raw.dtype)
            for name, obs in aux.items():
                valid = obs.valid.to(raw.device)
                if not bool(valid.any()):
                    continue
                # Invalid values never participate in moments or interpolation.
                values = obs.values.to(device=raw.device, dtype=torch.float32)
                selected = values[valid]
                scale = selected.std(unbiased=False).clamp_min(1e-6)
                normalized = torch.where(valid, (values-selected.mean())/scale, 0).clamp(-8,8)
                coverage = F.interpolate(valid.float()[None], size=(h,w), mode="area")[0]
                mean = F.interpolate(normalized[None], size=(h,w), mode="area")[0] / coverage.clamp_min(1e-6)
                inputs = torch.stack((mean[0], coverage[0]), dim=-1).to(raw.dtype)
                delta = delta + self.adapters[name](inputs) * coverage.permute(1,2,0).to(raw.dtype)
            # Row-major spatial residual -> Qwen merge-block token order.
            m = merge_size
            packed = delta.reshape(h//m,m,w//m,m,-1).permute(0,2,1,3,4).reshape(h*w,-1)
            fused.append(features + packed)
        return torch.cat(fused)
