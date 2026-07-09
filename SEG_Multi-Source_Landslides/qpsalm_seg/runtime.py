#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""运行时轻量预检，不导入 torch。"""

from __future__ import annotations

import subprocess
import sys


def torch_preflight(timeout: int = 120) -> tuple[bool, str]:
    """在子进程中检查 torch 是否能导入，避免主训练进程无提示卡住。"""
    cmd = [
        sys.executable,
        "-c",
        "import torch; print(torch.__version__, torch.cuda.is_available())",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        return False, f"torch import timed out after {timeout}s"
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip()
        return False, detail[-2000:]
    return True, proc.stdout.strip()
