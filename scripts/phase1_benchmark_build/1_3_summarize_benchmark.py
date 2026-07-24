#!/usr/bin/env python3
"""阶段 1B：汇总 Benchmark 的数据源、split、通道、模态和标签统计。

命令：python 1_3_summarize_benchmark.py --benchmark-root .../small
输入：Benchmark manifest 与 index.jsonl。
输出：stdout JSON；不修改 Benchmark。
写入：无。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

from benchmark_common import read_json, read_jsonl


def summarize(root: Path) -> dict[str, Any]:
    root = root.resolve()
    manifest = read_json(root / "manifest.json")
    rows = read_jsonl(root / "index.jsonl")
    source_split: dict[str, Counter[str]] = defaultdict(Counter)
    source_labels: dict[str, Counter[str]] = defaultdict(Counter)
    optical_channels: dict[str, Counter[tuple[str, ...]]] = defaultdict(Counter)
    modality_combinations: dict[str, Counter[tuple[str, ...]]] = defaultdict(Counter)
    foreground_sum: Counter[str] = Counter()
    for row in rows:
        source = row["source"]
        source_split[source][row["split"]] += 1
        source_labels[source][
            "positive" if float(row["foreground_ratio"]) > 0 else "background"
        ] += 1
        optical_channels[source][tuple(row["optical"]["channel_names"])] += 1
        modality_combinations[source][tuple(sorted(row["auxiliaries"]))] += 1
        foreground_sum[source] += float(row["foreground_ratio"])
    return {
        "schema_version": manifest["schema_version"],
        "mode": manifest["mode"],
        "sample_count": len(rows),
        "index_sha256": manifest["index_sha256"],
        "sources": {
            source: {
                "sample_count": sum(source_split[source].values()),
                "split_counts": dict(source_split[source]),
                "label_counts": dict(source_labels[source]),
                "mean_foreground_ratio": (
                    foreground_sum[source] / sum(source_split[source].values())
                ),
                "optical_channel_signatures": [
                    {"channel_names": list(names), "count": count}
                    for names, count in sorted(optical_channels[source].items())
                ],
                "auxiliary_combinations": [
                    {"modalities": list(names), "count": count}
                    for names, count in sorted(modality_combinations[source].items())
                ],
            }
            for source in sorted(source_split)
        },
        "warnings": manifest.get("warnings", []),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    print(json.dumps(summarize(args.benchmark_root), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"错误：{error}", file=sys.stderr)
        raise SystemExit(1)
