#!/usr/bin/env python3
"""1-4 合并统一 metadata。

用途：
    合并 Sen12 和 GDCLD 已生成的样本清单，形成统一 benchmark metadata 和 split 文件。

输入：
    - `benchmark/<run>/intermediate/sen12_samples.jsonl`
    - `benchmark/<run>/intermediate/gdcld_samples.jsonl`

输出：
    - `benchmark/<run>/metadata.jsonl`
    - `benchmark/<run>/splits/train.jsonl`
    - `benchmark/<run>/splits/val.jsonl`
    - `benchmark/<run>/splits/test.jsonl`
    - `benchmark/<run>/splits/test_candidate.jsonl`

关键处理：
    - Sen12 已在 1-1 阶段按 `region_id + event_date` 做 deterministic split。
    - GDCLD 官方 test_data 保持为 `test`。
    - Future work 默认只作为 `test_candidate` 候选数据，不进入正式训练集。
    - 本步骤不改写图像和掩膜，只负责统一 annotation layer。

示例命令：
    python scripts/1-4_merge_annotations.py \
      --out-dir benchmark/geohazard_halluground_v0
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from geohazard_common import ensure_dir, read_jsonl, save_splits, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="合并 Sen12 与 GDCLD 样本 metadata，并生成 split 文件。")
    parser.add_argument("--out-dir", default="benchmark/geohazard_halluground_v0", help="流水线输出目录。")
    return parser.parse_args()


def load_optional_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        print(f"[提示] 未找到中间文件，跳过：{path}")
        return []
    return read_jsonl(path)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    intermediate = out_dir / "intermediate"
    ensure_dir(intermediate)

    rows: list[dict[str, Any]] = []
    rows.extend(load_optional_jsonl(intermediate / "sen12_samples.jsonl"))
    rows.extend(load_optional_jsonl(intermediate / "gdcld_samples.jsonl"))
    rows.sort(key=lambda row: (row.get("source_dataset", ""), row.get("split", ""), row.get("sample_id", "")))

    metadata_path = out_dir / "metadata.jsonl"
    write_jsonl(metadata_path, rows)
    save_splits(out_dir, rows)
    print(f"已写入统一 metadata：{metadata_path}，共 {len(rows)} 条。")
    print(f"已写入 split 文件：{out_dir / 'splits'}")


if __name__ == "__main__":
    main()
