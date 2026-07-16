#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""M3-M7 artifact 共用的严格标准 JSON 解码边界。"""

from __future__ import annotations

import json
from typing import Any


class NonFiniteJSONError(json.JSONDecodeError):
    """拒绝 Python 解码器额外接受的 NaN/Infinity token。"""

    def __init__(self, token: str) -> None:
        super().__init__(
            f"non-standard JSON numeric constant is forbidden: {token}",
            token,
            0,
        )


def _reject_nonfinite_json_constant(token: str) -> None:
    raise NonFiniteJSONError(token)


def strict_json_loads(payload: str | bytes | bytearray) -> Any:
    """解析 JSON，同时拒绝 Python 的非标准非有限数扩展。"""
    return json.loads(
        payload,
        parse_constant=_reject_nonfinite_json_constant,
    )
