#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""视觉证据先验分支。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn
from .common import ConvBlock


class VisualEvidenceAdapter(nn.Module):
    """从多源 preview 图生成 proposal visual evidence embedding/feature。

    该分支不预测 bbox，也不消费 bbox prior。它对应当前路线中的弱空间
    证据：Qwen 或轻量 adapter 提供全局图文证据，而像素级边界仍交给
    PSALM-style mask tokens 与 decoder 学习。
    """

    def __init__(self, decoder_dim: int) -> None:
        super().__init__()
        d = int(decoder_dim)
        hidden = max(32, d // 2)
        self.stem = nn.Sequential(
            nn.Conv2d(3, hidden, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, hidden),
            nn.GELU(),
            ConvBlock(hidden, d),
        )
        self.context = nn.Sequential(
            nn.Conv2d(d, d, kernel_size=3, padding=2, dilation=2, groups=d, bias=False),
            nn.Conv2d(d, d, kernel_size=1, bias=False),
            nn.GroupNorm(1, d),
            nn.GELU(),
        )
        self.attention_head = nn.Conv2d(d, 1, kernel_size=1)
        self.embedding_proj = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, d),
        )
        self.feature_proj = ConvBlock(d, d)

    def forward(
        self,
        visual_preview: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        x = visual_preview.float().clamp(-1.0, 1.0)
        features = self.context(self.stem(x))
        attention_logits = self.attention_head(features)
        attention = torch.sigmoid(attention_logits)
        denom = attention.sum(dim=(2, 3)).clamp_min(1.0)
        pooled = (features * attention).sum(dim=(2, 3)) / denom
        embedding = self.embedding_proj(pooled)
        return {
            "visual_evidence_features": self.feature_proj(features),
            "visual_evidence_embedding": embedding,
            "visual_evidence_attention_logits": attention_logits,
            "visual_evidence_attention": attention,
            "visual_evidence_attention_mean": attention.mean(dim=(1, 2, 3)),
            "visual_evidence_attention_max": attention.flatten(1).max(dim=1).values,
        }


class CachedQwenVisualEvidenceBank(nn.Module):
    """读取离线 Qwen 图文 visual evidence embedding cache。

    cache embedding 是冻结 Qwen3-VL 对 ``visual_preview + proposal/condition
    prompt`` 的 pooled hidden state。训练时仅学习轻量 projection，把真实 VLM
    图文证据先验注入 PSALM mask-token proposal 分支。
    """

    def __init__(self, cache_path: str | Path, decoder_dim: int) -> None:
        super().__init__()
        path = Path(cache_path)
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[3] / path
        if not path.exists():
            raise FileNotFoundError(f"Qwen visual evidence cache 不存在: {path}")
        payload = torch.load(path, map_location="cpu")
        self._validate_cache(payload, path)
        keys = [str(item) for item in payload["keys"]]
        embeddings = payload["embeddings"].detach().float().contiguous()
        self.key_to_index = {key: idx for idx, key in enumerate(keys)}
        self.cache_path = str(path)
        self.hidden_size = int(embeddings.shape[1])
        self.backend = str(payload.get("backend") or "unknown")
        self.register_buffer("embeddings", embeddings, persistent=False)
        self.proj = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, decoder_dim),
            nn.GELU(),
            nn.Linear(decoder_dim, decoder_dim),
        )

    @staticmethod
    def _validate_cache(payload: Any, path: Path) -> None:
        if not isinstance(payload, dict):
            raise ValueError(f"Qwen visual cache 必须是 dict: {path}")
        fmt = payload.get("format")
        if fmt != "qpsalm_qwen_visual_evidence_cache_v1":
            raise ValueError(f"未知 Qwen visual cache format={fmt!r}: {path}")
        keys = payload.get("keys")
        embeddings = payload.get("embeddings")
        if not isinstance(keys, list) or not keys:
            raise ValueError(f"Qwen visual cache 缺少 keys 或 keys 为空: {path}")
        if not torch.is_tensor(embeddings) or embeddings.ndim != 2:
            raise ValueError(f"Qwen visual cache embeddings 必须是 [N,H] tensor: {path}")
        if len(keys) != int(embeddings.shape[0]):
            raise ValueError(
                f"Qwen visual cache keys/embeddings 数量不一致: keys={len(keys)} "
                f"embeddings={embeddings.shape[0]}"
            )

    def forward(self, keys: Sequence[str], device: torch.device | None = None) -> torch.Tensor:
        if device is None:
            device = self.embeddings.device
        missing = [key for key in keys if key not in self.key_to_index]
        if missing:
            raise KeyError(
                "Qwen visual evidence cache 缺少当前 batch 的 key。"
                f" 请用相同 train/val index 重新运行 qpsalm-cache-qwen-visual-evidence。missing={missing[0]!r}"
            )
        indices = torch.tensor([self.key_to_index[key] for key in keys], dtype=torch.long, device=self.embeddings.device)
        pooled = self.embeddings.index_select(0, indices).to(device)
        return self.proj(pooled.to(self.proj[1].weight.dtype))
