#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""多源滑坡任务指令模板公共工具库。

脚本作用：为 scripts/2-instruction/ 阶段脚本提供模板读取、模板校验、
instruction JSONL 路径、模态组合判断和训练样本构造函数。
主要输入：configs/instruction_templates/*.yaml 与 benchmark/indexes/*.jsonl。
主要输出：公共函数返回值；本文件不作为流程入口单独运行。
是否改写原始数据：不会改写 datasets/ 原始数据。
典型用法：由 2-1、2-2、2-3 脚本 import 后复用。
"""

from __future__ import annotations

import copy
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_SCRIPT_DIR = REPO_ROOT / "scripts" / "1-benchmark"
if str(BENCHMARK_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_SCRIPT_DIR))

from geohazard_benchmark_common import (  # noqa: E402
    DEFAULT_BENCHMARK_ROOT,
    ensure_dir,
    modality_combo,
    read_jsonl,
    to_repo_rel,
    write_json,
    write_jsonl,
)


DEFAULT_TEMPLATE_CONFIG = REPO_ROOT / "configs" / "instruction_templates" / "multisource_landslide_v1.yaml"
INSTRUCTION_SPLITS = ["train", "val", "test", "unlabeled", "extended_pool"]
OPTICAL_MODALITIES = {"optical_rgb", "optical_multiband", "multispectral"}
TERRAIN_MODALITIES = {"dem", "slope"}
SAR_MODALITIES = {"sar_asc", "sar_dsc"}
INSAR_MODALITIES = {"insar_vel"}
ACTIVE_OR_CHANGE_WORDS = {
    "active landslide",
    "newly appeared",
    "new landslide",
    "新增滑坡",
    "活动滑坡",
}
REFERRING_TEMPLATE_REQUIREMENTS = {
    "position": [
        "upper-left",
        "upper",
        "upper-right",
        "left",
        "center",
        "right",
        "lower-left",
        "lower",
        "lower-right",
    ],
    "scale": ["largest_landslide", "large_landslide", "small_landslide_patches"],
    "morphology": ["compact_landslide", "fragmented_landslides", "elongated_landslide"],
    "count": ["single_landslide", "multiple_landslides", "many_landslides"],
}


def instruction_index_paths(benchmark_dir: Path) -> dict[str, Path]:
    """返回第 2 阶段 instruction JSONL 输出路径。"""
    index_dir = benchmark_dir / "indexes"
    paths = {"all": index_dir / "instruction_all.jsonl"}
    for split in INSTRUCTION_SPLITS:
        paths[split] = index_dir / f"instruction_{split}.jsonl"
    return paths


def load_template_config(path: Path) -> dict[str, Any]:
    """读取 YAML 模板配置。"""
    if not path.exists():
        raise FileNotFoundError(f"模板配置不存在: {to_repo_rel(path)}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"模板配置不是 YAML dict: {to_repo_rel(path)}")
    templates = data.get("templates")
    if not isinstance(templates, list) or not templates:
        raise ValueError(f"模板配置缺少 templates 列表: {to_repo_rel(path)}")
    return data


def template_map(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """按 template_id 建立模板映射。"""
    out: dict[str, dict[str, Any]] = {}
    for template in config.get("templates") or []:
        tid = str(template.get("template_id") or "")
        if tid:
            out[tid] = template
    return out


def validate_templates(config: dict[str, Any]) -> tuple[list[str], list[str]]:
    """检查模板字段完整性；返回 errors/warnings。"""
    errors: list[str] = []
    warnings: list[str] = []
    seen: set[str] = set()
    required = {"template_id", "task_family", "text_en", "text_zh", "applicable_when", "target_mask_policy", "answer_format", "quality_flags"}
    for idx, template in enumerate(config.get("templates") or []):
        if not isinstance(template, dict):
            errors.append(f"第 {idx} 个模板不是 dict")
            continue
        tid = str(template.get("template_id") or "")
        if not tid:
            errors.append(f"第 {idx} 个模板缺少 template_id")
            continue
        if tid in seen:
            errors.append(f"重复 template_id: {tid}")
        seen.add(tid)
        missing = sorted(required - set(template))
        if missing:
            errors.append(f"{tid}: 缺少字段 {missing}")
        if template.get("task_family") == "referring_landslide_segmentation" and not template.get("template_id_pattern"):
            warnings.append(f"{tid}: referring 模板建议记录 template_id_pattern")
        if any(word in str(template.get("text_en", "")).lower() for word in ["newly appeared", "active landslide"]):
            warnings.append(f"{tid}: 第一版不建议把 active/newly appeared 写成独立监督目标")
    expected = {
        "generic_landslide_v1",
        "negative_aware_landslide_v1",
        "multisource_landslide_v1",
        "terrain_evidence_landslide_v1",
        "sar_terrain_landslide_v1",
        "insar_evidence_landslide_v1",
        "referring_rule_based_v1",
    }
    for tid in sorted(expected - seen):
        errors.append(f"缺少推荐初始模板: {tid}")

    referring_templates = config.get("referring_templates")
    if not isinstance(referring_templates, dict):
        errors.append("缺少 referring_templates 映射，无法从 referring_target 渲染文本")
    else:
        for category, subtypes in REFERRING_TEMPLATE_REQUIREMENTS.items():
            category_rules = referring_templates.get(category)
            if not isinstance(category_rules, dict):
                errors.append(f"referring_templates 缺少类别: {category}")
                continue
            for subtype in subtypes:
                rule = category_rules.get(subtype)
                if not isinstance(rule, dict):
                    errors.append(f"referring_templates.{category} 缺少子类: {subtype}")
                    continue
                for field in ["text_en", "text_zh"]:
                    if not rule.get(field):
                        errors.append(f"referring_templates.{category}.{subtype} 缺少 {field}")
    return errors, warnings


def modality_names(sample: dict[str, Any]) -> set[str]:
    """取样本中 available=True 的模态名。"""
    modalities = sample.get("modalities") or {}
    return {str(name) for name, info in modalities.items() if isinstance(info, dict) and info.get("available", True)}


def has_any(names: set[str], candidates: set[str]) -> bool:
    return bool(names & candidates)


def has_all(names: set[str], candidates: set[str]) -> bool:
    return candidates.issubset(names)


def choose_evidence_template(sample: dict[str, Any]) -> str | None:
    """为非空监督样本选择一个额外 evidence-aware 模板。"""
    names = modality_names(sample)
    if len(names) < 2:
        return None
    has_optical = has_any(names, OPTICAL_MODALITIES)
    has_terrain = has_any(names, TERRAIN_MODALITIES)
    has_sar = has_any(names, SAR_MODALITIES)
    has_insar = has_any(names, INSAR_MODALITIES)
    if has_optical and has_terrain and has_sar and has_insar:
        return "insar_evidence_landslide_v1"
    if "multispectral" in names and has_sar and has_terrain:
        return "sar_terrain_landslide_v1"
    if has_optical and has_terrain:
        return "terrain_evidence_landslide_v1"
    return "multisource_landslide_v1"


def make_instruction_object(template: dict[str, Any], *, text_en: str | None = None, text_zh: str | None = None, template_id: str | None = None) -> dict[str, Any]:
    """构造统一 instruction 字段。"""
    tid = template_id or str(template["template_id"])
    return {
        "language": "en",
        "template_id": tid,
        "task_family": template["task_family"],
        "text": text_en if text_en is not None else template["text_en"],
        "text_zh": text_zh if text_zh is not None else template["text_zh"],
        "answer_format": template["answer_format"],
    }


def make_instruction_sample(parent: dict[str, Any], template: dict[str, Any]) -> dict[str, Any]:
    """从普通 final 样本生成一条 instruction 训练样本。"""
    tid = str(template["template_id"])
    item = copy.deepcopy(parent)
    parent_id = str(parent.get("sample_id"))
    item["sample_id"] = f"{parent_id}__inst_{tid}"
    item["parent_sample_id"] = parent_id
    item["parent_task_type"] = parent.get("task_type")
    item["task_type"] = "instruction_landslide_segmentation"
    item["instruction"] = make_instruction_object(template)
    item["template_id"] = tid
    item["task_family"] = template["task_family"]
    item["target_mask_policy"] = template["target_mask_policy"]
    item["answer_format"] = template["answer_format"]
    item["instruction_source"] = "template_config"
    flags = set(item.get("quality_flags") or [])
    flags.update(template.get("quality_flags") or [])
    item["quality_flags"] = sorted(flags)
    return item


def safe_template_id_part(value: str) -> str:
    """把 referring category/subtype 转为稳定 template_id 片段。"""
    return re.sub(r"[^0-9A-Za-z]+", "_", value).strip("_").lower()


class SafeFormatDict(dict[str, Any]):
    """缺少占位符时保留原样，避免模板渲染因少量 grounding 字段缺失中断。"""

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def referring_render_context(ref_target: dict[str, Any]) -> dict[str, Any]:
    """从 referring target 的 grounding 中抽取安全模板占位符。"""
    grounding = ref_target.get("grounding") or {}
    labels = grounding.get("component_labels") or []
    context: dict[str, Any] = {
        "category": ref_target.get("category", "unknown"),
        "subtype": ref_target.get("subtype", "unknown"),
        "grid": grounding.get("grid") or ref_target.get("subtype", "unknown"),
        "component_count": grounding.get("component_count") or len(labels) or 0,
    }
    for key, value in grounding.items():
        if isinstance(value, (str, int, float, bool)):
            context[key] = value
    return context


def render_referring_text(ref_target: dict[str, Any], config: dict[str, Any]) -> tuple[str, str, str]:
    """根据 category/subtype 从 YAML referring_templates 渲染中英文指令。"""
    category = str(ref_target.get("category") or "unknown")
    subtype = str(ref_target.get("subtype") or "unknown")
    rule = ((config.get("referring_templates") or {}).get(category) or {}).get(subtype)
    if not isinstance(rule, dict):
        raise ValueError(f"缺少 referring target 模板: {category}/{subtype}")
    context = SafeFormatDict(referring_render_context(ref_target))
    text_en = str(rule["text_en"]).format_map(context)
    text_zh = str(rule["text_zh"]).format_map(context)
    template_id = str(rule.get("template_id") or f"referring_{safe_template_id_part(category)}_{safe_template_id_part(subtype)}_v1")
    return template_id, text_en, text_zh


def make_referring_instruction_sample(ref_target: dict[str, Any], template: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """从 referring_target_all.jsonl 行生成 instruction 训练样本，不重新生成 mask。"""
    item = copy.deepcopy(ref_target)
    target_sample_id = str(ref_target.get("sample_id"))
    tid, text_en, text_zh = render_referring_text(ref_target, config)
    target_mask = copy.deepcopy(item.pop("target_mask", {}))
    referring_target = {
        "target_sample_id": target_sample_id,
        "category": ref_target.get("category"),
        "subtype": ref_target.get("subtype"),
        "target_mask": target_mask,
        "grounding": copy.deepcopy(ref_target.get("grounding") or {}),
        "confidence": ref_target.get("confidence"),
    }
    item["sample_id"] = f"{target_sample_id}__inst_{tid}"
    item["parent_referring_target_sample_id"] = target_sample_id
    item["task_type"] = "referring_landslide_segmentation"
    item["mask"] = target_mask
    item["referring_target"] = referring_target
    item["instruction"] = make_instruction_object(template, text_en=text_en, text_zh=text_zh, template_id=tid)
    item["template_id"] = tid
    item["task_family"] = "referring_landslide_segmentation"
    item["target_mask_policy"] = "referring_target_mask"
    item["answer_format"] = "binary_mask"
    item["instruction_source"] = "referring_target_template"
    item["supervision"] = "mask"
    item.pop("category", None)
    item.pop("subtype", None)
    item.pop("grounding", None)
    item.pop("confidence", None)
    flags = set(item.get("quality_flags") or [])
    flags.update(template.get("quality_flags") or [])
    item["quality_flags"] = sorted(flags)
    return item


def write_instruction_split_indexes(benchmark_dir: Path, rows: list[dict[str, Any]]) -> None:
    """写出 instruction all/train/val/test JSONL。"""
    paths = instruction_index_paths(benchmark_dir)
    write_jsonl(paths["all"], sorted(rows, key=lambda row: str(row.get("sample_id", ""))))
    for split in INSTRUCTION_SPLITS:
        split_rows = [row for row in rows if row.get("split") == split]
        write_jsonl(paths[split], sorted(split_rows, key=lambda row: str(row.get("sample_id", ""))))


def count_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """生成 instruction 索引统计。"""
    by_split = Counter(str(row.get("split", "unknown")) for row in rows)
    by_template = Counter(str(row.get("template_id", "unknown")) for row in rows)
    by_family = Counter(str(row.get("task_family", "unknown")) for row in rows)
    by_combo = Counter(modality_combo(row) for row in rows)
    referring_categories = Counter(
        str((row.get("referring_target") or {}).get("category", "unknown"))
        for row in rows
        if row.get("task_family") == "referring_landslide_segmentation"
    )
    return {
        "num_samples": len(rows),
        "by_split": dict(sorted(by_split.items())),
        "by_template_id": dict(sorted(by_template.items())),
        "by_task_family": dict(sorted(by_family.items())),
        "by_modality_combo": dict(sorted(by_combo.items())),
        "referring_by_category": dict(sorted(referring_categories.items())),
    }


def write_template_report(path: Path, config: dict[str, Any], errors: list[str], warnings: list[str]) -> None:
    """写出模板校验报告。"""
    families = Counter(str(t.get("task_family", "unknown")) for t in config.get("templates") or [])
    report = {
        "说明": "任务指令模板配置校验报告。",
        "template_version": config.get("version"),
        "num_templates": len(config.get("templates") or []),
        "template_ids": [str(t.get("template_id")) for t in config.get("templates") or []],
        "num_errors": len(errors),
        "num_warnings": len(warnings),
        "task_family_counts": dict(sorted(families.items())),
        "errors": errors,
        "warnings": warnings,
    }
    write_json(path, report)
