#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""步骤 2-2：应用任务指令模板，生成 instruction segmentation 索引。

脚本作用：读取 benchmark 的 final all.jsonl 与 referring_target_all.jsonl，根据
任务类型、空 mask、可用模态和 referring target 选择稳定短指令模板。
主要输入：indexes/all.jsonl、indexes/referring_target_all.jsonl、模板 YAML。
主要输出：indexes/instruction_*.jsonl、reports/instruction_statistics.json、
reports/instruction_build_report.json。
是否改写原始数据：不会改写 datasets/；不会覆盖 1-benchmark 的原始索引。
典型用法：
  python scripts/2-instruction/2-2_apply_instruction_templates.py \
    --benchmark-dir benchmark/multisource_landslide_v1_small
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from geohazard_instruction_common import (
    DEFAULT_BENCHMARK_ROOT,
    DEFAULT_TEMPLATE_CONFIG,
    choose_evidence_template,
    count_rows,
    instruction_index_paths,
    load_template_config,
    make_instruction_sample,
    make_referring_instruction_sample,
    template_map,
    to_repo_rel,
    validate_templates,
    write_instruction_split_indexes,
    write_json,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_SCRIPT_DIR = REPO_ROOT / "scripts" / "1-benchmark"
if str(BENCHMARK_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_SCRIPT_DIR))

from geohazard_benchmark_common import final_index_paths, read_jsonl, referring_target_index_paths  # noqa: E402


def is_supervised_mask_sample(sample: dict[str, Any]) -> bool:
    """只为带 mask 的监督样本生成 instruction 训练行。"""
    return sample.get("supervision", "mask") == "mask" and isinstance(sample.get("mask"), dict)


def build_parent_instruction_rows(
    samples: list[dict[str, Any]],
    templates: dict[str, dict[str, Any]],
    *,
    include_evidence_extra: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """从普通 final 样本生成 global/negative/evidence instruction 行。"""
    rows: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    for sample in samples:
        if not is_supervised_mask_sample(sample):
            decisions.append({"sample_id": sample.get("sample_id"), "action": "skipped", "reason": "not_supervised_mask"})
            continue
        mask = sample.get("mask") or {}
        if mask.get("empty_mask") is True:
            tids = ["negative_aware_landslide_v1"]
        else:
            tids = ["generic_landslide_v1"]
            if include_evidence_extra:
                evidence_tid = choose_evidence_template(sample)
                if evidence_tid and evidence_tid not in tids:
                    tids.append(evidence_tid)
        for tid in tids:
            template = templates[tid]
            rows.append(make_instruction_sample(sample, template))
        decisions.append({"sample_id": sample.get("sample_id"), "action": "generated", "template_ids": tids})
    return rows, decisions


def build_referring_instruction_rows(
    referring_targets: list[dict[str, Any]],
    templates: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """从 referring_target_all.jsonl 生成 instruction 行，文本由 YAML 模板渲染。"""
    template = templates["referring_rule_based_v1"]
    return [make_referring_instruction_sample(sample, template, config) for sample in referring_targets]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="把 benchmark final/referring 索引转换成 instruction 索引。")
    parser.add_argument("--benchmark-dir", type=Path, default=DEFAULT_BENCHMARK_ROOT, help="目标 benchmark 目录。")
    parser.add_argument("--template-config", type=Path, default=DEFAULT_TEMPLATE_CONFIG, help="任务指令模板 YAML 路径。")
    parser.add_argument("--no-referring", action="store_true", help="不纳入 referring_target_all.jsonl。")
    parser.add_argument("--no-evidence-extra", action="store_true", help="多模态样本不额外生成 evidence-aware 指令。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_template_config(args.template_config)
    errors, _warnings = validate_templates(config)
    if errors:
        raise SystemExit(f"模板配置存在错误，停止生成 instruction 索引: {errors[:5]}")
    templates = template_map(config)

    final_samples = read_jsonl(final_index_paths(args.benchmark_dir)["all"])
    if not final_samples:
        raise SystemExit(f"未找到 final 索引: {final_index_paths(args.benchmark_dir)['all']}")
    parent_rows, decisions = build_parent_instruction_rows(
        final_samples,
        templates,
        include_evidence_extra=not args.no_evidence_extra,
    )

    referring_targets = [] if args.no_referring else read_jsonl(referring_target_index_paths(args.benchmark_dir)["all"])
    referring_rows = build_referring_instruction_rows(referring_targets, templates, config) if referring_targets else []
    instruction_rows = parent_rows + referring_rows
    write_instruction_split_indexes(args.benchmark_dir, instruction_rows)

    stats = count_rows(instruction_rows)
    stats.update({
        "说明": "instruction 索引由 final all.jsonl 和 referring_target_all.jsonl 派生，训练读取路径仍指向 benchmark 内部。",
        "benchmark_dir": to_repo_rel(args.benchmark_dir),
        "template_config": to_repo_rel(args.template_config),
        "num_final_input_samples": len(final_samples),
        "num_parent_instruction_samples": len(parent_rows),
        "num_referring_target_input_samples": len(referring_targets),
        "num_referring_instruction_samples": len(referring_rows),
        "include_referring": not args.no_referring,
        "include_evidence_extra": not args.no_evidence_extra,
        "instruction_index": to_repo_rel(instruction_index_paths(args.benchmark_dir)["all"]),
    })
    build_report = {
        "说明": "任务指令模板应用报告。",
        "benchmark_dir": to_repo_rel(args.benchmark_dir),
        "template_config": to_repo_rel(args.template_config),
        "decisions_examples": decisions[:50],
        "statistics": stats,
    }
    write_json(args.benchmark_dir / "reports" / "instruction_statistics.json", stats)
    write_json(args.benchmark_dir / "reports" / "instruction_build_report.json", build_report)
    print(
        "instruction 索引生成完成: "
        f"parent={len(parent_rows)}, referring={len(referring_rows)}, total={len(instruction_rows)} -> "
        f"{to_repo_rel(instruction_index_paths(args.benchmark_dir)['all'])}"
    )


if __name__ == "__main__":
    main()
