#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline Qwen multi-view token bank used by QMEF."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn

from qpsalm_seg.paths import resolve_repo_path


class CachedQwenVisualEvidenceBank(nn.Module):
    """Project cached per-view Qwen tokens; no local dense preview branch."""

    def __init__(self, cache_path: str | Path, decoder_dim: int) -> None:
        super().__init__()
        path = resolve_repo_path(cache_path) or Path(cache_path)
        if not path.exists():
            raise FileNotFoundError(f"Qwen multi-view cache 不存在: {path}")
        payload = torch.load(path, map_location="cpu")
        self._validate_cache(payload, path)
        keys = [str(item) for item in payload["lookup_keys"]]
        embeddings = payload["view_embeddings"].detach().float().contiguous()
        view_mask = payload["view_mask"].detach().bool().contiguous()
        self.key_to_index = {key: index for index, key in enumerate(keys)}
        self.cache_path = str(path)
        self.hidden_size = int(embeddings.shape[-1])
        self.backend = str(payload.get("backend") or "unknown")
        self.renderer_version = str(payload.get("renderer_version") or "unknown")
        self.pooling_method = str(payload.get("pooling_method") or "vision-token")
        self.register_buffer("embeddings", embeddings, persistent=False)
        self.register_buffer("view_mask", view_mask, persistent=False)
        self.proj = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, decoder_dim),
            nn.GELU(),
            nn.Linear(decoder_dim, decoder_dim),
        )

    @staticmethod
    def _validate_cache(payload: Any, path: Path) -> None:
        if not isinstance(payload, dict) or payload.get("format") != "qpsalm_qwen_multiview_cache_v2":
            raise ValueError(f"Qwen visual cache 必须是 qpsalm_qwen_multiview_cache_v2: {path}")
        keys = payload.get("lookup_keys")
        embeddings = payload.get("view_embeddings")
        view_mask = payload.get("view_mask")
        if not isinstance(keys, list) or not keys:
            raise ValueError(f"Qwen multi-view cache 缺少 lookup_keys: {path}")
        if not torch.is_tensor(embeddings) or embeddings.ndim != 3:
            raise ValueError(f"view_embeddings 必须是 [N,V,H]: {path}")
        if not torch.is_tensor(view_mask) or tuple(view_mask.shape) != tuple(embeddings.shape[:2]):
            raise ValueError(f"view_mask 必须与 [N,V] 对齐: {path}")
        if len(keys) != int(embeddings.shape[0]):
            raise ValueError(f"lookup_keys 与 embeddings 数量不一致: {path}")

    def forward(
        self,
        keys: Sequence[str],
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if device is None:
            device = self.embeddings.device
        missing = [key for key in keys if key not in self.key_to_index]
        if missing:
            raise KeyError(
                "Qwen multi-view cache 缺少当前 parent sample。"
                f" 请重新运行 qpsalm-cache-qwen-visual-evidence。missing={missing[0]!r}"
            )
        indices = torch.tensor([self.key_to_index[key] for key in keys], dtype=torch.long, device=self.embeddings.device)
        tokens = self.embeddings.index_select(0, indices).to(device)
        mask = self.view_mask.index_select(0, indices).to(device)
        return self.proj(tokens.to(self.proj[1].weight.dtype)), mask
