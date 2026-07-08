#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""步骤 2-3：验证 instruction segmentation 索引。

脚本作用：检查 instruction_*.jsonl 中的模板字段、任务类别、mask 路径、
模态路径和 evidence-aware 指令与实际模态是否匹配。
主要输入：indexes/instruction_all.jsonl、模板 YAML、可选 referring_target_all.jsonl。
主要输出：reports/instruction_validation_report.json。
是否改写原始数据：不会改写 datasets/；只写 benchmark/ 下的验证报告。
典型用法：
  python scripts/2-instruction/2-3_validate_instruction_index.py \
    --benchmark-dir benchmark/multisource_landslide_v1_small
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from geohazard_instruction_common import (
    ACTIVE_OR_CHANGE_WORDS,
    DEFAULT_BENCHMARK_ROOT,
    DEFAULT_TEMPLATE_CONFIG,
    INSAR_MODALITIES,
    OPTICAL_MODALITIES,
    SAR_MODALITIES,
    TERRAIN_MODALITIES,
    count_rows,
    has_all,
    has_any,
    instruction_index_paths,
    load_template_config,
    modality_names,
    template_map,
    to_repo_rel,
    validate_templates,
    write_json,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_SCRIPT_DIR = REPO_ROOT / "scripts" / "1-benchmark"
if str(BENCHMARK_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_SCRIPT_DIR))

from geohazard_benchmark_common import (  # noqa: E402
    path_is_inside_benchmark,
    read_jsonl,
    referring_target_index_paths,
    resolve_repo_path,
)


def check_path(path_ref: str | None, benchmark_dir: Path) -> tuple[bool, str]:
    """检查路径存在且位于 benchmark 内部。"""
    if not path_ref:
        return False, "路径为空"
    path = resolve_repo_path(path_ref)
    if path is None:
        return False, "路径无法解析"
    if not path.exists():
        return False, f"路径不存在: {to_repo_rel(path)}"
    if not path_is_inside_benchmark(path_ref, benchmark_dir):
        return False, f"路径不在 benchmark 目录内: {path_ref}"
    return True, ""


def template_known(template_id: str, task_family: str, templates: dict[str, dict[str, Any]]) -> bool:
    """允许 referring_*_v1 匹配 referring_rule_based_v1 通配模板。"""
    if template_id in templates:
        return True
    return task_family == "referring_landslide_segmentation" and template_id.startswith("referring_") and template_id.endswith("_v1")


def validate_evidence_template(sample: dict[str, Any]) -> tuple[list[str], list[str]]:
    """检查 evidence-aware 模板是否要求了不存在的模态。"""
    errors: list[str] = []
    warnings: list[str] = []
    sid = sample.get("sample_id", "<missing_sample_id>")
    tid = str(sample.get("template_id") or "")
    names = modality_names(sample)
    if tid == "multisource_landslide_v1" and len(names) < 2:
        errors.append(f"{sid}: multisource_landslide_v1 至少需要 2 种可用模态，当前 {sorted(names)}")
    if tid == "terrain_evidence_landslide_v1":
        if not has_any(names, OPTICAL_MODALITIES):
            errors.append(f"{sid}: terrain_evidence_landslide_v1 缺少光学/多光谱模态")
        if not has_any(names, TERRAIN_MODALITIES):
            errors.append(f"{sid}: terrain_evidence_landslide_v1 缺少 DEM/slope 模态")
    if tid == "sar_terrain_landslide_v1":
        if "multispectral" not in names:
            errors.append(f"{sid}: sar_terrain_landslide_v1 缺少 multispectral 模态")
        if not has_any(names, SAR_MODALITIES):
            errors.append(f"{sid}: sar_terrain_landslide_v1 缺少 SAR 模态")
        if not has_any(names, TERRAIN_MODALITIES):
            errors.append(f"{sid}: sar_terrain_landslide_v1 缺少 DEM/slope 模态")
    if tid == "insar_evidence_landslide_v1":
        if not has_any(names, OPTICAL_MODALITIES):
            errors.append(f"{sid}: insar_evidence_landslide_v1 缺少光学/多光谱模态")
        if not has_any(names, SAR_MODALITIES):
            errors.append(f"{sid}: insar_evidence_landslide_v1 缺少 SAR 模态")
        if not has_any(names, TERRAIN_MODALITIES):
            errors.append(f"{sid}: insar_evidence_landslide_v1 缺少 DEM/slope 模态")
        if not has_all(names, INSAR_MODALITIES):
            errors.append(f"{sid}: insar_evidence_landslide_v1 缺少 insar_vel 模态")
    text = f"{(sample.get('instruction') or {}).get('text', '')} {(sample.get('instruction') or {}).get('text_zh', '')}".lower()
    if any(word in text for word in ACTIVE_OR_CHANGE_WORDS):
        warnings.append(f"{sid}: 第一版不应把活动/新增滑坡作为独立监督目标，请确认模板语义")
    return errors, warnings


def validate_instruction_sample(sample: dict[str, Any], benchmark_dir: Path, templates: dict[str, dict[str, Any]]) -> tuple[list[str], list[str]]:
    """验证单条 instruction 样本。"""
    errors: list[str] = []
    warnings: list[str] = []
    sid = sample.get("sample_id", "<missing_sample_id>")
    tid = str(sample.get("template_id") or "")
    family = str(sample.get("task_family") or "")
    instruction = sample.get("instruction") or {}

    if not tid:
        errors.append(f"{sid}: 缺少 template_id")
    if not family:
        errors.append(f"{sid}: 缺少 task_family")
    if tid and family and not template_known(tid, family, templates):
        errors.append(f"{sid}: template_id 不在模板配置中: {tid}")
    if instruction.get("template_id") != tid:
        errors.append(f"{sid}: instruction.template_id 与顶层 template_id 不一致")
    if instruction.get("task_family") != family:
        errors.append(f"{sid}: instruction.task_family 与顶层 task_family 不一致")
    if not instruction.get("text"):
        errors.append(f"{sid}: 缺少 instruction.text")
    if not instruction.get("text_zh"):
        warnings.append(f"{sid}: 缺少 instruction.text_zh")
    if sample.get("answer_format") != "binary_mask":
        errors.append(f"{sid}: answer_format 应为 binary_mask")

    modalities = sample.get("modalities") or {}
    if not isinstance(modalities, dict) or not modalities:
        errors.append(f"{sid}: 缺少 modalities")
    for name, modality in modalities.items():
        if not isinstance(modality, dict) or modality.get("available", True) is False:
            continue
        ok, message = check_path(modality.get("path"), benchmark_dir)
        if not ok:
            errors.append(f"{sid}: 模态 {name} {message}")

    mask = sample.get("mask") or {}
    if not isinstance(mask, dict):
        errors.append(f"{sid}: 缺少 mask")
    else:
        ok, message = check_path(mask.get("path"), benchmark_dir)
        if not ok:
            errors.append(f"{sid}: mask {message}")
        if mask.get("format") != "npy":
            errors.append(f"{sid}: mask 必须为 npy，当前 {mask.get('format')}")
        if mask.get("dtype") != "uint8":
            errors.append(f"{sid}: mask dtype 应为 uint8，当前 {mask.get('dtype')}")

    cur_errors, cur_warnings = validate_evidence_template(sample)
    errors.extend(cur_errors)
    warnings.extend(cur_warnings)
    if family == "referring_landslide_segmentation":
        target = sample.get("referring_target")
        if not isinstance(target, dict):
            errors.append(f"{sid}: referring instruction 缺少 referring_target")
        else:
            if target.get("category") not in {"position", "scale", "morphology", "count"}:
                errors.append(f"{sid}: referring target category 非法: {target.get('category')}")
            if not target.get("subtype"):
                errors.append(f"{sid}: referring target 缺少 subtype")
            if not isinstance(target.get("target_mask"), dict):
                errors.append(f"{sid}: referring target 缺少 target_mask")
    return errors, warnings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="验证 instruction segmentation 索引。")
    parser.add_argument("--benchmark-dir", type=Path, default=DEFAULT_BENCHMARK_ROOT, help="目标 benchmark 目录。")
    parser.add_argument("--template-config", type=Path, default=DEFAULT_TEMPLATE_CONFIG, help="任务指令模板 YAML 路径。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_template_config(args.template_config)
    template_errors, template_warnings = validate_templates(config)
    templates = template_map(config)

    rows = read_jsonl(instruction_index_paths(args.benchmark_dir)["all"])
    errors: list[str] = list(template_errors)
    warnings: list[str] = list(template_warnings)
    seen: dict[str, int] = {}
    for idx, row in enumerate(rows):
        sid = str(row.get("sample_id") or "")
        if not sid:
            errors.append(f"第 {idx} 行缺少 sample_id")
        elif sid in seen:
            errors.append(f"重复 sample_id: {sid}，首次行 {seen[sid]}，重复行 {idx}")
        else:
            seen[sid] = idx
        cur_errors, cur_warnings = validate_instruction_sample(row, args.benchmark_dir, templates)
        errors.extend(cur_errors)
        warnings.extend(cur_warnings)

    referring_source = read_jsonl(referring_target_index_paths(args.benchmark_dir)["all"])
    referring_rows = [row for row in rows if row.get("task_family") == "referring_landslide_segmentation"]
    if referring_source and len(referring_rows) < len(referring_source):
        errors.append(f"referring instruction 数量少于 referring_target_all.jsonl: {len(referring_rows)} < {len(referring_source)}")
    source_categories = Counter(str(row.get("category", "unknown")) for row in referring_source)
    output_categories = Counter(str((row.get("referring_target") or {}).get("category", "unknown")) for row in referring_rows)
    for category, count in source_categories.items():
        if output_categories.get(category, 0) < count:
            errors.append(f"referring 类别 {category} 数量减少: {output_categories.get(category, 0)} < {count}")

    stats = count_rows(rows)
    report = {
        "说明": "instruction 索引验证报告；errors 必须为空才建议用于 VLM-Seg 训练。",
        "benchmark_dir": to_repo_rel(args.benchmark_dir),
        "template_config": to_repo_rel(args.template_config),
        "num_samples": len(rows),
        "num_errors": len(errors),
        "num_warnings": len(warnings),
        "statistics": stats,
        "errors": errors[:200],
        "warnings": warnings[:200],
    }
    report_path = args.benchmark_dir / "reports" / "instruction_validation_report.json"
    write_json(report_path, report)
    print(f"instruction 索引验证完成: errors={len(errors)}, warnings={len(warnings)} -> {to_repo_rel(report_path)}")
    if errors:
        raise SystemExit("instruction 索引存在错误，请查看 instruction_validation_report.json")


if __name__ == "__main__":
    main()
