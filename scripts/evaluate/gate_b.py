#!/usr/bin/env python3
"""用途：运行 Gate B prepare/generate/evaluate/verify/media 子命令。

命令：python scripts/evaluate/gate_b.py --help
输入：冻结协议、selection、Base/Adapter prediction 与 canonical Benchmark。
输出：新评价报告，或只读 verifier/media 查询结果。
写入：依子命令而定；verify/media 只读，正式身份不覆盖。
所属能力：RS-GeneralDesc scientific evaluation。
"""

from oa_groundrag.vlm.cli import entrypoint


if __name__ == "__main__":
    raise SystemExit(entrypoint())
