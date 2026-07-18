#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""旧入口：转发到统一的 Description Vision Cache workflow。

用途：兼容现有 M3 构建命令；新命令使用 ``qpsalm-segdesc cache build``。
写入行为：由 workflow 严格限制在 ``--output-dir``，不修改 benchmark 或源 cache。
"""

from qpsalm_seg.description.workflows.cache_build import main


if __name__ == "__main__":
    main()
