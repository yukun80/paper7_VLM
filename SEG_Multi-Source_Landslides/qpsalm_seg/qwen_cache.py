#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Qwen condition cache 覆盖检查工具。

脚本作用：根据当前 train/val/test index 重新派生 QPSALM 需要编码的
proposal/condition 文本，并检查 qwen_cache 是否完整覆盖。
主要输入：QPSalmConfig 与 condition_embedding_cache。
主要输出：coverage report；训练入口可在 forward 前提前失败。
是否改写原始数据：不会。
典型用法：assert_qwen_cache_coverage(config, splits=("train", "val"))。
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch

from .config import QPSalmConfig
from .data import (
    build_condition_prompt_text,
    build_condition_text,
    build_evidence_reasoning_text,
    build_proposal_context_text,
    iter_jsonl,
    resolve_repo_path,
    should_skip_row,
)


TEXT_BUILDERS = [
    ("condition_text", build_condition_text),
    ("proposal_context_text", build_proposal_context_text),
    ("condition_prompt_text", build_condition_prompt_text),
    ("evidence_reasoning_text", build_evidence_reasoning_text),
]


def _split_limit(config: QPSalmConfig, split: str) -> int | None:
    """按 Dataset 的 max_samples 语义推导当前 split 检查范围。"""
    value = config.max_train_samples if split == "train" else config.max_val_samples
    if value is None or value <= 0:
        return None
    return int(value)


def collect_required_qwen_texts(
    config: QPSalmConfig,
    splits: Iterable[str] = ("train", "val"),
) -> tuple[list[str], dict[str, Any]]:
    """收集当前配置会送入 controller 的唯一文本集合。"""
    texts: dict[str, None] = {}
    text_types: dict[str, set[str]] = defaultdict(set)
    type_sets: dict[str, set[str]] = defaultdict(set)
    split_reports: list[dict[str, Any]] = []
    for split in splits:
        index_path = resolve_repo_path(config.index_path(split))
        if index_path is None or not index_path.exists():
            raise FileNotFoundError(f"{split} index 不存在: {config.index_path(split)}")
        rows_seen = 0
        rows_kept = 0
        skipped: Counter[str] = Counter()
        limit = _split_limit(config, split)
        for row in iter_jsonl(index_path):
            rows_seen += 1
            reason = should_skip_row(row, config.core_templates)
            if reason is not None:
                skipped[reason] += 1
                continue
            for text_type, builder in TEXT_BUILDERS:
                text = builder(row)
                texts[text] = None
                text_types[text].add(text_type)
                type_sets[text_type].add(text)
            rows_kept += 1
            if limit is not None and rows_kept >= limit:
                break
        split_reports.append(
            {
                "split": split,
                "index": str(index_path),
                "rows_seen": rows_seen,
                "rows_kept": rows_kept,
                "max_samples": limit,
                "skipped": dict(skipped),
            }
        )
    report = {
        "splits": split_reports,
        "text_types": {name: len(values) for name, values in sorted(type_sets.items())},
        "text_type_map": {text: sorted(types) for text, types in text_types.items()},
    }
    return list(texts.keys()), report


def resolve_cache_path(cache_path: str | Path | None) -> Path:
    """解析并确认 cache 路径存在。"""
    if not cache_path:
        raise ValueError("controller=qwen_cache 需要 condition_embedding_cache")
    path = resolve_repo_path(cache_path)
    if path is None or not path.exists():
        raise FileNotFoundError(f"Qwen condition embedding cache 不存在: {cache_path}")
    return path


def load_qwen_cache_payload(cache_path: str | Path | None) -> tuple[Path, dict[str, Any]]:
    """读取并基础校验 qwen cache。"""
    path = resolve_cache_path(cache_path)
    payload = torch.load(path, map_location="cpu")
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
    return path, payload


def verify_qwen_cache_coverage(
    config: QPSalmConfig,
    splits: Iterable[str] = ("train", "val"),
    preview_limit: int = 8,
) -> dict[str, Any]:
    """生成 cache 覆盖报告，不直接抛错。"""
    required_texts, source_report = collect_required_qwen_texts(config, splits=splits)
    path, payload = load_qwen_cache_payload(config.condition_embedding_cache)
    cached_texts = {str(text) for text in payload["texts"]}
    missing = [text for text in required_texts if text not in cached_texts]
    text_type_map = source_report.get("text_type_map", {})
    preview = [
        {
            "text": text[:240],
            "types": text_type_map.get(text, []),
        }
        for text in missing[: max(1, int(preview_limit))]
    ]
    source_report_public = {
        key: value
        for key, value in source_report.items()
        if key != "text_type_map"
    }
    return {
        "ok": not missing,
        "cache": {
            "path": str(path),
            "backend": payload.get("backend"),
            "model_path": payload.get("model_path"),
            "device": payload.get("device"),
            "hidden_size": int(payload["embeddings"].shape[1]),
            "num_texts": len(payload["texts"]),
        },
        "required": {
            "num_texts": len(required_texts),
            "source": source_report_public,
        },
        "missing": {
            "num_texts": len(missing),
            "preview": preview,
        },
    }


def assert_qwen_cache_coverage(
    config: QPSalmConfig,
    splits: Iterable[str] = ("train", "val"),
) -> dict[str, Any]:
    """检查 cache 覆盖；缺文本时提前给出可操作错误。"""
    if config.controller not in {"qwen_cache", "cached_qwen"}:
        return {"skipped": True, "reason": f"controller={config.controller}"}
    report = verify_qwen_cache_coverage(config, splits=splits)
    if not report["ok"]:
        preview = report["missing"]["preview"]
        raise RuntimeError(
            "Qwen condition embedding cache 未覆盖当前配置需要的全部文本。"
            f" missing={report['missing']['num_texts']} cache={report['cache']['path']} "
            f"preview={preview}. 请用相同 train/val index 重新运行 qpsalm-cache-qwen-embeddings。"
        )
    return report
