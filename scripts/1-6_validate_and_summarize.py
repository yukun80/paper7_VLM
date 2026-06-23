#!/usr/bin/env python3
"""1-6 校验和统计汇总。

用途：
    校验 benchmark 输出完整性，并生成统计 JSON、统计图和随机抽查图。

输入：
    - `benchmark/<run>/metadata.jsonl`
    - `benchmark/<run>/vlm_views/`
    - `benchmark/<run>/segmentation_masks/`
    - `benchmark/<run>/segmentation_masks_redblack/`

输出：
    - `benchmark/<run>/validation_report.json`
    - `benchmark/<run>/summary.json`
    - `benchmark/<run>/figures/`
    - `benchmark/<run>/audit/`

关键处理：
    - 检查图像和 mask 是否存在、尺寸是否一致、mask 是否只有 0/1。
    - 检查红黑可视化标签是否存在、尺寸是否与图像一致。
    - 检查 bbox 是否越界、正样本 mask 面积是否大于 0。
    - 检查同一数据源/区域/事件/标注单元是否跨 split 泄漏。
    - audit 图叠加 mask 和 bbox，只用于人工抽查，不改变任何标注。
    - 全量数据较大时会显示进度条；可用快速参数缩小校验范围。

示例命令：
python scripts/1-6_validate_and_summarize.py \
    --out-dir benchmark/geohazard_halluground_v0 \
    --audit-samples 100
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from geohazard_common import ensure_dir, read_jsonl, save_audit, save_stats, validate_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="校验 benchmark 输出并生成统计图和抽查图。")
    parser.add_argument("--out-dir", default="benchmark/geohazard_halluground_v0", help="流水线输出目录。")
    parser.add_argument("--audit-samples", type=int, default=100, help="随机抽查可视化样本数量。")
    parser.add_argument("--seed", type=int, default=42, help="随机抽查的随机种子。")
    parser.add_argument("--max-validation-samples", type=int, default=None, help="仅校验前 N 个样本，用于快速诊断。默认校验全部样本。")
    parser.add_argument("--skip-pixel-check", action="store_true", help="跳过 mask 像元唯一值和正样本面积检查，只检查文件和尺寸。")
    parser.add_argument("--skip-stats", action="store_true", help="跳过 summary.json 和 figures 统计输出。")
    parser.add_argument("--skip-audit", action="store_true", help="跳过 audit 抽查图生成。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    metadata_path = out_dir / "metadata.jsonl"
    if not metadata_path.exists():
        raise SystemExit(f"缺少 metadata：{metadata_path}，请先运行 1-4_merge_annotations.py。")
    ensure_dir(out_dir)

    print(f"正在读取 metadata：{metadata_path}", flush=True)
    metadata_rows = read_jsonl(metadata_path)
    print(f"已读取 {len(metadata_rows)} 个样本，开始校验文件和标注一致性。", flush=True)
    checked_count = min(args.max_validation_samples, len(metadata_rows)) if args.max_validation_samples is not None else len(metadata_rows)
    errors, warnings_out = validate_outputs(
        metadata_rows,
        show_progress=True,
        max_samples=args.max_validation_samples,
        skip_pixel_check=args.skip_pixel_check,
    )
    report = {"errors": errors, "warnings": warnings_out, "sample_count": len(metadata_rows), "checked_sample_count": checked_count}
    report_path = out_dir / "validation_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写入校验报告：{report_path}", flush=True)

    # 统计和 audit 只读取已经生成的派生图像，不再接触原始大图，便于快速重复校验。
    if args.skip_stats:
        print("已跳过统计图和 summary.json 生成。", flush=True)
    else:
        print("开始生成 summary.json 和统计图。", flush=True)
        save_stats(out_dir, metadata_rows)
    if args.skip_audit:
        print("已跳过 audit 抽查图生成。", flush=True)
    else:
        print(f"开始生成 audit 抽查图，数量：{args.audit_samples}。", flush=True)
        save_audit(out_dir, metadata_rows, args.audit_samples, args.seed)
    if errors:
        print(f"[错误] 校验发现 {len(errors)} 个问题，详情见：{report_path}")
        raise SystemExit(1)
    print(f"校验通过：{report_path}，共检查 {checked_count} / {len(metadata_rows)} 个样本。")


if __name__ == "__main__":
    main()
