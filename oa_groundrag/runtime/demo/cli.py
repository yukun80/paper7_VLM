#!/usr/bin/env python3
"""用途：启动 OA-GroundRAG 只读 Unified Demo Workbench。

命令：python scripts/infer/demo.py --config configs/runtime/demo_v1.yaml --port 7860。
输入：严格 Demo v1 配置、只读 Benchmark/Frozen Eval 与现有 Unified providers。
输出：本地 Gradio 浏览界面、独立 Demo Gallery、test receipt（若授权）和 Demo run。
写入：仅写配置绑定的独立 demo_root；不修改 Benchmark、模型、Bank 或正式 outputs。
所属能力：Instruction-Routed Unified Inference / Research Demo。
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

from .app import serve_demo
from .config import load_demo_config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="OA-GroundRAG Unified Demo Workbench（只读科研演示）",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Unified Demo v1 严格 YAML",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="本地回环端口，默认 7860",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = load_demo_config(args.config)
        serve_demo(config, port=args.port)
    except KeyboardInterrupt:
        return 130
    except Exception as error:
        print(
            f"Unified Demo 启动失败 [{type(error).__name__}]: {error}",
            file=sys.stderr,
            flush=True,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
