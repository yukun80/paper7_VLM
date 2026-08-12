"""OA-AuxSeg 运行配置加载与路径解析。"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from .contracts import RuntimeConfig


def load_runtime_config(path: Path) -> RuntimeConfig:
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError("运行配置顶层必须是 JSON object")
    return RuntimeConfig.from_dict(value)


def resolve_runtime(
    config: RuntimeConfig,
    repo_root: Path,
) -> tuple[Path, Path, Path, torch.device]:
    benchmark_root = config.resolve_path(config.benchmark_root, repo_root)
    output_dir = config.resolve_path(config.output_dir, repo_root)
    backbone_weights = config.resolve_path(config.backbone_weights, repo_root)
    if not (benchmark_root / "index.jsonl").is_file():
        raise FileNotFoundError(
            f"Benchmark index 不存在：{benchmark_root / 'index.jsonl'}"
        )
    if config.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "配置要求 CUDA，但当前 torch.cuda.is_available() 为 False"
            )
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return benchmark_root, output_dir, backbone_weights, device
