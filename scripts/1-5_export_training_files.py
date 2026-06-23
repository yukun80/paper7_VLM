#!/usr/bin/env python3
"""1-5 导出训练和评估文件。

用途：
    从统一 `metadata.jsonl` 导出 Qwen-VL SFT 样本和 COCO bbox 标注。

输入：
    - `benchmark/<run>/metadata.jsonl`

输出：
    - `benchmark/<run>/qwen_vl_sft.jsonl`
    - `benchmark/<run>/detection_coco.json`

关键处理：
    - SFT prompt 和 assistant 自然语言内容使用中文。
    - JSON 字段名、split 名称、`landslide/none` 等解析约定保持英文。
    - 初版 VLM 训练主要输出 bbox；segmentation mask 保留给 IoU/Dice 评估。
    - bbox 从 mask 派生，只能作为语义证据框，不能声称为实例级真值框。

示例命令：
python scripts/1-5_export_training_files.py \
    --out-dir benchmark/geohazard_halluground_v0 \
    --version v2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from geohazard_common import build_coco, ensure_dir, qwen_messages, read_jsonl, write_jsonl


def build_sft_rows(metadata_rows: list[dict[str, Any]], version: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample in metadata_rows:
        rows.append(qwen_messages(sample, "classification", version))
        rows.append(qwen_messages(sample, "grounding", version))
        if version == "v2":
            rows.append(qwen_messages(sample, "quality", version))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从 metadata.jsonl 导出 Qwen-VL SFT 和 COCO bbox 文件。")
    parser.add_argument("--out-dir", default="benchmark/geohazard_halluground_v0", help="流水线输出目录。")
    parser.add_argument("--version", choices=["v0", "v1", "v2"], default="v0", help="控制是否导出 V2 质量判断任务。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    metadata_path = out_dir / "metadata.jsonl"
    if not metadata_path.exists():
        raise SystemExit(f"缺少 metadata：{metadata_path}，请先运行 1-4_merge_annotations.py。")
    ensure_dir(out_dir)

    metadata_rows = read_jsonl(metadata_path)
    sft_rows = build_sft_rows(metadata_rows, args.version)
    sft_path = out_dir / "qwen_vl_sft.jsonl"
    write_jsonl(sft_path, sft_rows)

    # COCO 标准字段保持英文，类别名保持 landslide，方便后续检测评估脚本直接读取。
    coco_path = out_dir / "detection_coco.json"
    coco_path.write_text(json.dumps(build_coco(metadata_rows), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写入 Qwen-VL SFT 文件：{sft_path}，共 {len(sft_rows)} 条。")
    print(f"已写入 COCO bbox 文件：{coco_path}")


if __name__ == "__main__":
    main()
