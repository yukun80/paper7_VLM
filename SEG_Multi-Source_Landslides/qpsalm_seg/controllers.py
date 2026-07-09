#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""语义控制器：Qwen3-VL 冻结路径与 tiny text-probe fallback。

脚本作用：把 instruction、condition prompt、模态/GSD token 编码成 decoder
可用的 condition embedding。
主要输入：batch["condition_text"]。
主要输出：[B, decoder_dim] condition embedding。
是否改写原始数据：不会。
典型用法：build_controller(config, device)。
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn

from .config import QPSalmConfig


REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_local_path(path_ref: str | Path) -> Path:
    """解析 repo 相对路径。"""
    path = Path(path_ref)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def validate_qwen_model_dir(model_path: str | Path) -> Path:
    """确认本地 Qwen3-VL 目录包含基本加载文件。"""
    path = resolve_local_path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Qwen3-VL 模型目录不存在: {path}. "
            "请把本地权重放到 models_zoo/Qwen3-VL-2B-Instruct 或用 --qwen-model-path 指定。"
        )
    if not path.is_dir():
        raise NotADirectoryError(f"Qwen3-VL 模型路径不是目录: {path}")
    required = ["config.json", "tokenizer_config.json"]
    missing = [name for name in required if not (path / name).exists()]
    has_weights = any(path.glob("*.safetensors")) or any(path.glob("pytorch_model*.bin")) or (path / "model.safetensors.index.json").exists()
    if not has_weights:
        missing.append("*.safetensors or pytorch_model*.bin")
    if missing:
        raise FileNotFoundError(
            f"Qwen3-VL 模型目录缺少必要文件: {path}; missing={missing}. "
            "请确认模型权重已完整下载。"
        )
    return path


def select_qwen_model_class() -> type[Any]:
    """选择当前 transformers 可用的 Qwen3-VL 文本/视觉语言模型类。"""
    model_import_errors: list[str] = []
    # 先用专用 Qwen3-VL 类；老/新 transformers 可退到 AutoModelForImageTextToText。
    for class_name in ["Qwen3VLForConditionalGeneration", "AutoModelForImageTextToText", "AutoModelForCausalLM"]:
        try:
            module = __import__("transformers", fromlist=[class_name])
            model_cls = getattr(module, class_name)
            return model_cls
        except Exception as exc:  # pragma: no cover - depends on optional env
            model_import_errors.append(f"{class_name}: {exc}")
    detail = "; ".join(model_import_errors)
    raise RuntimeError(
        "当前 transformers 缺少可加载 Qwen3-VL 的模型类。"
        f" 请升级 qwen3vl 环境中的 transformers，导入失败详情: {detail}"
    )


class TextProbeController(nn.Module):
    """轻量文本探针，用于 CPU smoke 和无 GPU 环境调试。"""

    def __init__(self, decoder_dim: int, vocab_size: int = 4096, max_tokens: int = 96) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.max_tokens = max_tokens
        self.embedding = nn.Embedding(vocab_size, decoder_dim)
        self.norm = nn.LayerNorm(decoder_dim)
        self.proj = nn.Sequential(
            nn.Linear(decoder_dim, decoder_dim),
            nn.GELU(),
            nn.Linear(decoder_dim, decoder_dim),
        )

    def _hash_token(self, token: str) -> int:
        digest = hashlib.sha1(token.encode("utf-8")).hexdigest()
        return int(digest[:8], 16) % self.vocab_size

    def _encode_text(self, text: str, device: torch.device) -> torch.Tensor:
        tokens = text.lower().replace(".", " ").replace(",", " ").split()
        if not tokens:
            tokens = ["landslide"]
        ids = [self._hash_token(token) for token in tokens[: self.max_tokens]]
        return torch.tensor(ids, dtype=torch.long, device=device)

    def forward(self, texts: Sequence[str], device: torch.device | None = None) -> torch.Tensor:
        if device is None:
            device = self.embedding.weight.device
        pooled = []
        for text in texts:
            ids = self._encode_text(text, device)
            pooled.append(self.embedding(ids).mean(dim=0))
        x = torch.stack(pooled, dim=0)
        return self.proj(self.norm(x))


class FrozenQwenController(nn.Module):
    """冻结 Qwen3-VL 语义控制器。

    第一版只用文本路径编码 instruction/condition/modality/scale token；多源像素
    特征由本项目的 modality adapters 处理。
    """

    def __init__(self, model_path: str | Path, decoder_dim: int, device: torch.device, allow_cpu: bool = False) -> None:
        super().__init__()
        if device.type != "cuda" and not allow_cpu:
            raise RuntimeError(
                "Qwen controller requires CUDA for this prototype. "
                "Current CUDA is unavailable; use controller=text_probe for smoke tests or enable allow_qwen_cpu."
            )
        try:
            from transformers import AutoTokenizer
        except Exception as exc:  # pragma: no cover - depends on optional env
            raise RuntimeError("transformers with Qwen3-VL tokenizer support is required for controller=qwen") from exc
        model_dir = validate_qwen_model_dir(model_path)
        model_cls = select_qwen_model_class()

        self.model_path = str(model_dir)
        self.model_class_name = getattr(model_cls, "__name__", str(model_cls))
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
        except Exception as exc:  # pragma: no cover - depends on optional env/model files
            raise RuntimeError(
                f"无法加载 Qwen3-VL tokenizer: model_path={model_dir}. "
                "请确认 tokenizer.json/tokenizer_config.json 完整，且 transformers 版本支持 Qwen3-VL。"
            ) from exc
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        try:
            self.model = model_cls.from_pretrained(
                str(model_dir),
                torch_dtype=dtype,
                trust_remote_code=True,
            )
        except Exception as exc:  # pragma: no cover - depends on optional env/model files
            raise RuntimeError(
                f"无法加载冻结 Qwen3-VL controller: model_path={model_dir}, "
                f"model_class={self.model_class_name}. "
                "请确认权重完整、transformers/qwen-vl 依赖匹配，并优先在 qwen3vl 环境中运行 "
                "qpsalm-cache-qwen-embeddings --backend qwen。"
            ) from exc
        self.model.eval()
        self.model.to(device)
        for param in self.model.parameters():
            param.requires_grad_(False)

        text_config = getattr(self.model.config, "text_config", None) or getattr(self.model.config, "llm_config", None)
        hidden = int(getattr(text_config, "hidden_size", getattr(self.model.config, "hidden_size", 2048)))
        self.proj = nn.Sequential(
            nn.Linear(hidden, decoder_dim),
            nn.GELU(),
            nn.Linear(decoder_dim, decoder_dim),
        )

    @torch.no_grad()
    def _pool_qwen(self, texts: Sequence[str], device: torch.device) -> torch.Tensor:
        encoded = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        outputs = self.model(
            **encoded,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        hidden = outputs.hidden_states[-1]
        mask = encoded["attention_mask"].to(hidden.dtype).unsqueeze(-1)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

    def forward(self, texts: Sequence[str], device: torch.device | None = None) -> torch.Tensor:
        if device is None:
            device = next(self.model.parameters()).device
        pooled = self._pool_qwen(texts, device)
        return self.proj(pooled.to(self.proj[0].weight.dtype))


class CachedQwenController(nn.Module):
    """缓存式冻结 Qwen controller。

    预计算阶段用本地 Qwen3-VL 把 condition text 编码为原始 hidden state；
    训练阶段只加载这些 hidden state，并训练一个轻量 projection 接入 mask decoder。
    """

    def __init__(self, cache_path: str | Path, decoder_dim: int) -> None:
        super().__init__()
        path = Path(cache_path)
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[2] / path
        if not path.exists():
            raise FileNotFoundError(f"Qwen condition embedding cache 不存在: {path}")
        payload = torch.load(path, map_location="cpu")
        self._validate_cache(payload, path)
        texts = [str(item) for item in payload["texts"]]
        embeddings = payload["embeddings"].detach().float().contiguous()
        self.text_to_index = {text: idx for idx, text in enumerate(texts)}
        self.cache_path = str(path)
        self.hidden_size = int(embeddings.shape[1])
        self.register_buffer("embeddings", embeddings, persistent=False)
        self.proj = nn.Sequential(
            nn.Linear(self.hidden_size, decoder_dim),
            nn.GELU(),
            nn.Linear(decoder_dim, decoder_dim),
        )

    @staticmethod
    def _validate_cache(payload: Any, path: Path) -> None:
        if not isinstance(payload, dict):
            raise ValueError(f"Qwen cache 必须是 dict: {path}")
        fmt = payload.get("format")
        if fmt != "qpsalm_qwen_condition_cache_v1":
            raise ValueError(f"未知 Qwen cache format={fmt!r}: {path}")
        texts = payload.get("texts")
        embeddings = payload.get("embeddings")
        if not isinstance(texts, list) or not texts:
            raise ValueError(f"Qwen cache 缺少 texts 或 texts 为空: {path}")
        if not torch.is_tensor(embeddings) or embeddings.ndim != 2:
            raise ValueError(f"Qwen cache embeddings 必须是 [N,H] tensor: {path}")
        if len(texts) != int(embeddings.shape[0]):
            raise ValueError(
                f"Qwen cache texts/embeddings 数量不一致: texts={len(texts)} embeddings={embeddings.shape[0]}"
            )

    def forward(self, texts: Sequence[str], device: torch.device | None = None) -> torch.Tensor:
        if device is None:
            device = self.embeddings.device
        missing = [text for text in texts if text not in self.text_to_index]
        if missing:
            preview = missing[0][:240]
            raise KeyError(
                "Qwen condition embedding cache 缺少当前 batch 需要编码的文本。"
                f" 请用相同 train/val index 重新运行 qpsalm-cache-qwen-embeddings。missing={preview!r}"
            )
        indices = torch.tensor([self.text_to_index[text] for text in texts], dtype=torch.long, device=self.embeddings.device)
        pooled = self.embeddings.index_select(0, indices).to(device)
        return self.proj(pooled.to(self.proj[0].weight.dtype))


def build_controller(config: QPSalmConfig, device: torch.device) -> nn.Module:
    """按配置创建语义控制器。"""
    if config.controller == "text_probe":
        return TextProbeController(config.decoder_dim)
    if config.controller == "qwen":
        return FrozenQwenController(
            model_path=config.qwen_model_path,
            decoder_dim=config.decoder_dim,
            device=device,
            allow_cpu=config.allow_qwen_cpu,
        )
    if config.controller in {"qwen_cache", "cached_qwen"}:
        if not config.condition_embedding_cache:
            raise ValueError("controller=qwen_cache 需要设置 condition_embedding_cache")
        return CachedQwenController(
            cache_path=config.condition_embedding_cache,
            decoder_dim=config.decoder_dim,
        )
    raise ValueError(f"未知 controller: {config.controller}")
