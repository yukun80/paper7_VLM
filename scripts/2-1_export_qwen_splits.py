#!/usr/bin/env python3
"""Split Qwen-VL SFT JSONL by metadata split.

This prevents accidentally training on validation/test samples from the
combined `qwen_vl_sft.jsonl` exported by stage 1-5.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def sample_id_from_sft_id(row_id: str) -> tuple[str, str]:
    if "::" not in row_id:
        raise ValueError(f"SFT row id does not contain task suffix: {row_id}")
    sample_id, task = row_id.rsplit("::", 1)
    return sample_id, task


def parse_tasks(value: str) -> set[str]:
    tasks = {item.strip() for item in value.split(",") if item.strip()}
    allowed = {"classification", "grounding", "quality"}
    unknown = tasks - allowed
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown task(s): {', '.join(sorted(unknown))}")
    if not tasks:
        raise argparse.ArgumentTypeError("at least one task is required")
    return tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export train/val/test Qwen-VL SFT JSONL files from combined benchmark output.")
    parser.add_argument("--out-dir", default="benchmark/geohazard_halluground_v2_full", help="Benchmark run directory.")
    parser.add_argument("--metadata", default=None, help="Override metadata.jsonl path.")
    parser.add_argument("--sft", default=None, help="Override qwen_vl_sft.jsonl path.")
    parser.add_argument("--tasks", type=parse_tasks, default=parse_tasks("classification,grounding"), help="Comma-separated tasks to export.")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Optionally limit train rows for smoke training.")
    parser.add_argument("--seed", type=int, default=42, help="Seed used when limiting train rows.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    metadata_path = Path(args.metadata) if args.metadata else out_dir / "metadata.jsonl"
    sft_path = Path(args.sft) if args.sft else out_dir / "qwen_vl_sft.jsonl"

    metadata_rows = read_jsonl(metadata_path)
    split_by_sample = {row["sample_id"]: row["split"] for row in metadata_rows}
    if len(split_by_sample) != len(metadata_rows):
        raise SystemExit(f"{metadata_path}: duplicate sample_id detected")

    rows_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    task_counts: Counter[tuple[str, str]] = Counter()
    missing_metadata = 0
    skipped_task = 0

    for row in read_jsonl(sft_path):
        sample_id, task = sample_id_from_sft_id(row.get("id", ""))
        if task not in args.tasks:
            skipped_task += 1
            continue
        split = split_by_sample.get(sample_id)
        if split is None:
            missing_metadata += 1
            continue
        if split not in rows_by_split:
            continue
        rows_by_split[split].append(row)
        task_counts[(split, task)] += 1

    if missing_metadata:
        raise SystemExit(f"{missing_metadata} SFT rows did not match any metadata sample_id")

    if args.max_train_samples is not None:
        if args.max_train_samples <= 0:
            raise SystemExit("--max-train-samples must be positive")
        rng = random.Random(args.seed)
        train_rows = rows_by_split["train"]
        rng.shuffle(train_rows)
        rows_by_split["train"] = train_rows[: args.max_train_samples]

    for split, rows in rows_by_split.items():
        write_jsonl(out_dir / f"qwen_sft_{split}.jsonl", rows)

    print(f"metadata: {metadata_path}")
    print(f"sft: {sft_path}")
    print(f"tasks: {','.join(sorted(args.tasks))}")
    for split in ["train", "val", "test"]:
        print(f"{split}: {len(rows_by_split[split])} rows -> {out_dir / f'qwen_sft_{split}.jsonl'}")
    print("task counts:")
    for (split, task), count in sorted(task_counts.items()):
        print(f"  {split}/{task}: {count}")
    if skipped_task:
        print(f"skipped rows from disabled tasks: {skipped_task}")


if __name__ == "__main__":
    main()
