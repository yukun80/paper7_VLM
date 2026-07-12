#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""QPSALM 可选环境检查。

用途：检查 Python、benchmark 索引、Qwen 本地文件、torch/GPU 和动态库线索。
推荐运行命令：PYTHONPATH=SEG_Multi-Source_Landslides python -m
qpsalm_seg.cli.check_env --benchmark-dir benchmark/multisource_landslide_v2_small
主要输入：本地 repo 与 qwen3vl Python 环境。
主要输出：终端 JSON。
写入行为：只读检查，不修改任何数据或配置。
所属流程：可选开发诊断，不是 benchmark 构建或正式训练的前置门槛。
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

from qpsalm_seg.paths import resolve_project_path
from typing import Any


def run_probe(cmd: list[str], timeout: int) -> dict[str, Any]:
    start = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "seconds": round(time.time() - start, 2),
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip()[-2000:],
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "returncode": "timeout",
            "seconds": round(time.time() - start, 2),
            "stdout": (exc.stdout or "").strip() if isinstance(exc.stdout, str) else "",
            "stderr": (exc.stderr or "").strip() if isinstance(exc.stderr, str) else "",
        }


def run_torch_import_profile(python: str, timeout: int) -> dict[str, Any]:
    """运行 -X importtime，并提取最慢 import 项。"""
    cmd = [python, "-B", "-X", "importtime", "-c", "import torch"]
    start = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
        returncode: int | str = proc.returncode
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
    except subprocess.TimeoutExpired as exc:
        returncode = "timeout"
        stdout = exc.stdout if isinstance(exc.stdout, str) else (exc.stdout or b"").decode("utf-8", errors="replace")
        stderr = exc.stderr if isinstance(exc.stderr, str) else (exc.stderr or b"").decode("utf-8", errors="replace")

    pattern = re.compile(r"import time:\s+(\d+)\s+\|\s+(\d+)\s+\|\s+(.+)")
    rows: list[dict[str, Any]] = []
    for line in (stderr + "\n" + stdout).splitlines():
        match = pattern.match(line)
        if not match:
            continue
        self_us, cumulative_us, module = match.groups()
        rows.append(
            {
                "module": module.strip(),
                "self_seconds": round(int(self_us) / 1_000_000.0, 4),
                "cumulative_seconds": round(int(cumulative_us) / 1_000_000.0, 4),
            }
        )
    rows.sort(key=lambda item: item["self_seconds"], reverse=True)
    return {
        "ok": returncode == 0,
        "returncode": returncode,
        "seconds": round(time.time() - start, 2),
        "num_import_rows": len(rows),
        "slowest_self": rows[:20],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check QPSALM runtime environment.")
    parser.add_argument("--python", default=sys.executable, help="Python executable to probe.")
    parser.add_argument("--benchmark-dir", default="benchmark/multisource_landslide_v2_small")
    parser.add_argument("--qwen-dir", default="models_zoo/Qwen3-VL-2B-Instruct")
    parser.add_argument("--torch-timeout", type=int, default=120)
    parser.add_argument("--profile-torch-import", action="store_true")
    parser.add_argument("--skip-basic-torch-import", action="store_true")
    parser.add_argument("--skip-torch-ldd", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    benchmark = resolve_project_path(args.benchmark_dir)
    if benchmark is None:
        raise ValueError("benchmark_dir 不能为空")
    qwen = Path(args.qwen_dir)
    qwen_config: dict[str, Any] = {}
    if (qwen / "config.json").exists():
        try:
            qwen_config = json.loads((qwen / "config.json").read_text(encoding="utf-8"))
        except Exception as exc:
            qwen_config = {"config_read_error": str(exc)}
    report: dict[str, Any] = {
        "python": args.python,
        "benchmark": {
            "dir_exists": benchmark.exists(),
            "instruction_train": (benchmark / "indexes" / "instruction_train.jsonl").exists(),
            "instruction_val": (benchmark / "indexes" / "instruction_val.jsonl").exists(),
        },
        "qwen": {
            "dir_exists": qwen.exists(),
            "config_json": (qwen / "config.json").exists(),
            "tokenizer_json": (qwen / "tokenizer.json").exists(),
            "tokenizer_config_json": (qwen / "tokenizer_config.json").exists(),
            "preprocessor_config_json": (qwen / "preprocessor_config.json").exists(),
            "model_safetensors": (qwen / "model.safetensors").exists(),
            "model_type": qwen_config.get("model_type"),
            "architectures": qwen_config.get("architectures"),
        },
    }
    if not args.skip_basic_torch_import:
        report["torch_import"] = run_probe(
            [
                args.python,
                "-c",
                (
                    "import time; t=time.time(); import torch; "
                    "print({'seconds': round(time.time()-t, 2), "
                    "'version': torch.__version__, 'cuda': torch.cuda.is_available()})"
                ),
            ],
            timeout=args.torch_timeout,
        )
    else:
        report["torch_import"] = {"skipped": True}
    torch_c = (
        Path(args.python).parent.parent
        / "lib"
        / "python3.11"
        / "site-packages"
        / "torch"
        / "_C.cpython-311-x86_64-linux-gnu.so"
    )
    if args.skip_torch_ldd:
        report["torch_ldd"] = {"skipped": True}
    elif torch_c.exists():
        report["torch_ldd"] = run_probe(["ldd", str(torch_c)], timeout=20)
    else:
        report["torch_ldd"] = {"ok": False, "reason": f"not found: {torch_c}"}
    if args.profile_torch_import:
        report["torch_import_profile"] = run_torch_import_profile(args.python, timeout=args.torch_timeout)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
