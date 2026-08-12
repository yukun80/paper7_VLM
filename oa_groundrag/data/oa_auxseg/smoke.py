#!/usr/bin/env python3
"""阶段 1B：验证可变光学通道与缺失辅助模态可以形成训练 batch。

命令：python 1_4_smoke_dataloader.py --benchmark-root .../small
输入：已通过 validator 的 Benchmark。
输出：stdout JSON smoke 结果；不修改 Benchmark。
写入：无。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from torch.utils.data import DataLoader, Subset

from benchmark_common import BenchmarkDataset, collate_benchmark_samples


def smoke(root: Path, *, normalization: str = "none") -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for policy in ("none", "single", "all"):
        dataset = BenchmarkDataset(
            root,
            auxiliary_policy=policy,
            normalization=normalization,
        )
        first_by_source: dict[str, int] = {}
        for index, row in enumerate(dataset.rows):
            first_by_source.setdefault(row["source"], index)
        indices = list(first_by_source.values())
        loader = DataLoader(
            Subset(dataset, indices),
            batch_size=len(indices),
            shuffle=False,
            num_workers=0,
            collate_fn=collate_benchmark_samples,
        )
        batch = next(iter(loader))
        if batch["mask"].ndim != 4 or batch["mask"].shape[1] != 1:
            raise ValueError(f"{policy}: mask batch shape={tuple(batch['mask'].shape)}")
        if len(batch["optical"]) != len(indices):
            raise ValueError(f"{policy}: optical 列表长度错误")
        optical_channels = [int(values.shape[0]) for values in batch["optical"]]
        results.append(
            {
                "policy": policy,
                "batch_size": len(indices),
                "mask_shape": list(batch["mask"].shape),
                "optical_channel_counts": optical_channels,
                "auxiliary_modalities": sorted(batch["auxiliaries"]),
                "source_order": [
                    metadata["source"] for metadata in batch["metadata"]
                ],
            }
        )
    return {"status": "pass", "normalization": normalization, "results": results}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument(
        "--normalization", choices=("none", "zscore"), default="none"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    print(
        json.dumps(
            smoke(args.benchmark_root, normalization=args.normalization),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"错误：{error}", file=sys.stderr)
        raise SystemExit(1)
