#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在 val 或 test split 上评估 SANE/QMEF/PMRD checkpoint。

用途：加载新格式 checkpoint，计算分组指标并导出预测 mask 和多模态总览。
推荐运行命令：PYTHONPATH=SEG_Multi-Source_Landslides python -m
qpsalm_seg.cli.eval --config
SEG_Multi-Source_Landslides/configs/qpsalm_small_qwen_cached_core.yaml
--preset sane_qmef_pmrd --checkpoint outputs/RUN/checkpoint_best.pt --split val
--condition-embedding-cache CACHE.pt --output-dir outputs/RUN/eval --device cuda --skip-torch-preflight
主要输入：配置、preset、checkpoint、val/test 索引和对应 Qwen cache。
主要输出：eval_report.json、eval_manifest.json、mask exports、诊断表和可视化。
写入行为：写入 --output-dir；--overwrite-output 会清空该评估目录。
所属流程：独立验证/测试推理；完整多模态导出可加 --visualize-all 和 --export-multimodal-overview。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from qpsalm_seg.config import apply_config_overrides, load_config
from qpsalm_seg.presets import PRESET_CHOICES, apply_preset
from qpsalm_seg.runtime import torch_preflight


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Multi-Source Qwen-PSALM-Seg.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--preset", choices=PRESET_CHOICES, default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--benchmark-dir", default=None)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--target-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--val-index", default=None)
    parser.add_argument("--test-index", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--controller", choices=["qwen", "qwen_cache", "cached_qwen", "text_probe"], default=None)
    parser.add_argument("--qwen-model-path", default=None)
    parser.add_argument("--condition-embedding-cache", default=None)
    parser.add_argument("--visual-evidence-cache", default=None)
    parser.add_argument("--allow-qwen-cpu", action="store_true")
    parser.add_argument("--num-visualizations", type=int, default=None)
    parser.add_argument("--visualize-all", action="store_true")
    parser.add_argument("--export-multimodal-overview", action="store_true")
    parser.add_argument("--eval-threshold", type=float, default=None)
    parser.add_argument("--torch-timeout", type=int, default=120)
    parser.add_argument("--skip-torch-preflight", action="store_true")
    parser.add_argument("--print-full-report", action="store_true")
    parser.add_argument("--overwrite-output", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = apply_preset(load_config(args.config), args.preset)
    config = apply_config_overrides(
        config,
        {
            "benchmark_dir": args.benchmark_dir,
            "batch_size": args.batch_size,
            "target_size": args.target_size,
            "num_workers": args.num_workers,
            "max_val_samples": args.max_val_samples,
            "max_val_batches": args.max_val_batches,
            "val_index": args.val_index,
            "test_index": args.test_index,
            "output_dir": args.output_dir,
            "controller": args.controller,
            "qwen_model_path": args.qwen_model_path,
            "condition_embedding_cache": args.condition_embedding_cache,
            "visual_evidence_cache": args.visual_evidence_cache,
            "allow_qwen_cpu": True if args.allow_qwen_cpu else None,
            "num_visualizations": args.num_visualizations,
            "eval_threshold": args.eval_threshold,
        },
    )
    if not args.skip_torch_preflight:
        ok, message = torch_preflight(timeout=args.torch_timeout)
        if not ok:
            raise RuntimeError(f"PyTorch runtime is not ready: {message}")
        print(f"torch_preflight={message}")

    import torch
    from qpsalm_seg.data import MultiSourceLandslideDataset, SizeBucketBatchSampler, qpsalm_collate
    from qpsalm_seg.paths import resolve_repo_path
    from qpsalm_seg.qwen_cache import assert_qwen_cache_coverage
    from qpsalm_seg.train_eval import build_model, evaluate, load_checkpoint, resolve_device, utc_now, write_json

    device = resolve_device(args.device)
    cache_report = assert_qwen_cache_coverage(config, splits=(args.split,))
    if cache_report.get("ok"):
        print(
            "qwen_cache_coverage="
            f"required_texts={cache_report['required']['num_texts']} "
            f"cached_texts={cache_report['cache']['num_texts']} "
            f"backend={cache_report['cache'].get('backend')}"
        )
    dataset = MultiSourceLandslideDataset(config, split=args.split, max_samples=config.max_val_samples)
    loader_kwargs = {"num_workers": config.num_workers, "collate_fn": qpsalm_collate}
    if config.size_buckets:
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_sampler=SizeBucketBatchSampler(dataset, config.batch_size, shuffle=False, seed=config.seed),
            **loader_kwargs,
        )
    else:
        loader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=False, **loader_kwargs)
    model = build_model(config, device)
    step = load_checkpoint(args.checkpoint, model)
    out_dir = resolve_repo_path(config.output_dir) or Path(config.output_dir)
    if args.overwrite_output and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = evaluate(
        model,
        loader,
        device,
        max_batches=config.max_val_batches if config.max_val_batches and config.max_val_batches > 0 else None,
        visual_dir=out_dir / "eval_visualizations",
        num_visualizations=config.num_visualizations,
        visualize_all=bool(args.visualize_all),
        export_multimodal_overview=bool(args.export_multimodal_overview),
        threshold=float(config.eval_threshold),
        threshold_sweep=config.threshold_sweep,
    )
    report["checkpoint_step"] = step
    write_json(
        out_dir / "eval_manifest.json",
        {
            "created_at_utc": utc_now(),
            "created_by": "qpsalm-eval",
            "checkpoint": str(args.checkpoint),
            "checkpoint_step": step,
            "split": args.split,
            "preset": config.preset,
            "visualize_all": bool(args.visualize_all),
            "export_multimodal_overview": bool(args.export_multimodal_overview),
            "resolved_config": dict(config.__dict__),
        },
    )
    (out_dir / "eval_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.print_full_report:
        payload = report
    else:
        payload = {
            "eval_report": str(out_dir / "eval_report.json"),
            "checkpoint_step": step,
            "split": args.split,
            "overall": (report.get("metrics") or {}).get("overall"),
            "positive_only": (report.get("metrics") or {}).get("positive_only"),
            "negative_only": (report.get("metrics") or {}).get("negative_only"),
            "original_overall": (report.get("metrics_original_size") or {}).get("overall"),
            "canvas_vs_original_delta": report.get("canvas_vs_original_delta"),
            "coverage": report.get("coverage"),
        }
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
