"""Qwen3-VL processor 具体后端。"""

from __future__ import annotations

from ...processing import _LocalMultimodalProcessorAdapter


class Qwen3VLProcessorAdapter(_LocalMultimodalProcessorAdapter):
    """Qwen3-VL 的本地 AutoProcessor + qwen-vl-utils 适配器。"""

