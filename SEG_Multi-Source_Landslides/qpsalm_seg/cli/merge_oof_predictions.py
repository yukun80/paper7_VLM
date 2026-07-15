#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""合并并验证 D4 各 held-out fold 的 train predictions。

用途：将每个 OOF segmentation checkpoint 导出的 predicted_train_<fold>.jsonl 合并为
唯一的 predicted_train_oof.jsonl，并验证 parent 覆盖、fold 归属和审计状态。
推荐命令：PYTHONPATH=SEG_Multi-Source_Landslides python -m
qpsalm_seg.cli.merge_oof_predictions --fold-manifest
outputs/qpsalm_description/oof_folds_small/fold_manifest.json --input
outputs/qpsalm_description/predicted_fold0/predicted_train_0.jsonl --input
outputs/qpsalm_description/predicted_fold1/predicted_train_1.jsonl --input
outputs/qpsalm_description/predicted_fold2/predicted_train_2.jsonl --output
outputs/qpsalm_description/predicted_train_oof.jsonl
主要输出：一个可直接传给 train-description --predicted-index 的 JSONL。
写入行为：只写 --output 及同目录 report；不移动 mask，不修改各 fold 结果。
所属流程：M6 D4。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from qpsalm_seg.description.oof import load_oof_manifest
from qpsalm_seg.description.predicted_regions import PREDICTED_REGION_FORMAT
from qpsalm_seg.paths import resolve_project_path


def _read_jsonl(path_ref: str) -> list[dict]:
    path = resolve_project_path(path_ref) or Path(path_ref)
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge audited train OOF predictions")
    parser.add_argument("--fold-manifest", required=True)
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = load_oof_manifest(args.fold_manifest)
    parent_to_fold = {
        str(key): str(value) for key, value in manifest["parent_to_fold"].items()
    }
    merged: dict[str, dict] = {}
    input_hashes = {}
    for path_ref in args.input:
        path = resolve_project_path(path_ref) or Path(path_ref)
        input_hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        for row in _read_jsonl(path_ref):
            parent = str(row.get("parent_sample_id") or "")
            provenance = row.get("prediction_provenance") or {}
            if row.get("schema_version") != PREDICTED_REGION_FORMAT:
                raise ValueError(f"predicted region format 非法: {parent}")
            if str(row.get("split")) != "train":
                raise ValueError(f"OOF merge 只接受 train prediction: {parent}")
            if parent not in parent_to_fold:
                raise ValueError(f"prediction parent 不在 fold manifest: {parent}")
            if str(provenance.get("checkpoint_fold")) != parent_to_fold[parent]:
                raise ValueError(f"prediction fold 归属错误: {parent}")
            if provenance.get("out_of_fold_verified") is not True or not provenance.get("fold_audit"):
                raise ValueError(f"prediction 未通过 OOF checkpoint 审计: {parent}")
            if parent in merged:
                raise ValueError(f"OOF prediction parent 重复: {parent}")
            merged[parent] = row
    missing = sorted(set(parent_to_fold) - set(merged))
    if missing:
        raise ValueError(f"OOF prediction 未覆盖全部 fold parents: count={len(missing)} examples={missing[:8]}")
    rows = [merged[parent] for parent in sorted(merged)]
    output = resolve_project_path(args.output) or Path(args.output)
    _write_jsonl(output, rows)
    report = {
        "protocol": "qpsalm_predicted_region_oof_merge_v1",
        "fold_manifest": str(args.fold_manifest),
        "num_folds": manifest["num_folds"],
        "num_parents": len(rows),
        "inputs": input_hashes,
        "output": str(output),
    }
    report_path = output.with_suffix(".report.json")
    temporary = report_path.with_suffix(report_path.suffix + ".part")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(report_path)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
