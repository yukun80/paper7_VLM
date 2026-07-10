#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""一键运行 QPSALM Phase 1 训练闭环。

脚本作用：串联核心索引缓存、condition embedding 缓存、训练、eval reload 和
run summary，减少手动拼接多条命令造成的实验不可复现。
主要输入：benchmark/multisource_landslide_v1_small instruction 索引与 YAML 配置。
主要输出：index cache、condition cache、checkpoint、validation/eval JSON、PNG 可视化和 run_summary.json。
是否改写原始数据：不会，只写 outputs 下的实验产物。
典型用法：python -m qpsalm_seg.cli.run_phase1 --device cuda --embedding-backend qwen。
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
from datetime import datetime, timezone
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from qpsalm_seg.cli.cache_index import cache_split
from qpsalm_seg.cli.cache_qwen_embeddings import collect_condition_texts, hash_smoke_embeddings, qwen_embeddings
from qpsalm_seg.cli.cache_qwen_visual_evidence import (
    collect_visual_evidence_samples,
    hash_smoke_embeddings as hash_smoke_visual_embeddings,
    qwen_visual_embeddings,
)
from qpsalm_seg.cli.compare_runs import compare_run_summaries, read_run_summary
from qpsalm_seg.cli.diagnose_run import default_diagnose_args, diagnose
from qpsalm_seg.cli.summarize_run import summarize_run
from qpsalm_seg.analysis_tables import export_analysis_tables
from qpsalm_seg.config import LOSS_STAGE_CHOICES, QPSalmConfig, apply_loss_stage, load_config, parse_combo_loss_weights
from qpsalm_seg.data import MultiSourceLandslideDataset, qpsalm_collate, resolve_repo_path
from qpsalm_seg.qwen_cache import assert_qwen_cache_coverage
from qpsalm_seg.thresholding import recommend_thresholds
from qpsalm_seg.train_eval import atomic_torch_save, build_model, evaluate, load_checkpoint, resolve_device, train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an end-to-end Multi-Source Qwen-PSALM-Seg Phase 1 experiment.")
    parser.add_argument("--config", default="SEG_Multi-Source_Landslides/configs/qpsalm_small_qwen_cached_core.yaml")
    parser.add_argument("--output-root", default="outputs/qpsalm_phase1")
    parser.add_argument("--run-name", default="core")
    parser.add_argument(
        "--mode",
        choices=["baseline", "box-prior", "both"],
        default="baseline",
        help="baseline is the main evidence route; box-prior/both are legacy ablations.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-device", default=None)

    parser.add_argument("--index-cache-dir", default=None)
    parser.add_argument("--index-strategy", choices=["first", "balanced-canonical"], default="balanced-canonical")
    parser.add_argument("--samples-per-combo", type=int, default=4)
    parser.add_argument("--max-index-rows", type=int, default=None)
    parser.add_argument("--max-index-samples", type=int, default=None)
    parser.add_argument("--reuse-index-cache", action="store_true")

    parser.add_argument("--controller", choices=["qwen_cache", "text_probe"], default="qwen_cache")
    parser.add_argument("--embedding-backend", choices=["qwen", "hash-smoke"], default="qwen")
    parser.add_argument("--embedding-cache", default=None)
    parser.add_argument("--embedding-device", default=None)
    parser.add_argument("--qwen-model-path", default=None)
    parser.add_argument("--embedding-batch-size", type=int, default=1)
    parser.add_argument("--hash-hidden-size", type=int, default=1024)
    parser.add_argument("--reuse-embedding-cache", action="store_true")
    parser.add_argument("--allow-qwen-cpu", action="store_true")
    parser.add_argument(
        "--visual-evidence-backend",
        choices=["off", "qwen", "hash-smoke"],
        default="off",
        help="Optional Qwen image-text visual evidence cache backend.",
    )
    parser.add_argument("--visual-evidence-cache", default=None)
    parser.add_argument("--visual-evidence-batch-size", type=int, default=1)
    parser.add_argument("--hash-visual-hidden-size", type=int, default=1024)
    parser.add_argument("--reuse-visual-evidence-cache", action="store_true")

    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--target-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--val-interval", type=int, default=None)
    parser.add_argument("--save-interval", type=int, default=None)
    parser.add_argument("--keep-recent-checkpoints", type=int, default=None)
    parser.add_argument("--visualize-interval", type=int, default=None)
    parser.add_argument("--log-interval", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--num-visualizations", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument("--grad-accum-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--modality-dropout", type=float, default=None)
    parser.add_argument("--use-gsd-film", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--use-spatial-modality-gate", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--use-query-modality-attention", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--query-modality-feature-weight", type=float, default=None)
    parser.add_argument("--use-evidence-reasoning", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--evidence-reasoning-weight", type=float, default=None)
    parser.add_argument("--selection-evidence-weight", type=float, default=None)
    parser.add_argument("--use-visual-evidence", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--visual-evidence-weight", type=float, default=None)
    parser.add_argument("--visual-evidence-feature-weight", type=float, default=None)
    parser.add_argument("--loss-stage", choices=LOSS_STAGE_CHOICES, default=None)
    parser.add_argument("--train-hflip-prob", type=float, default=None)
    parser.add_argument("--train-vflip-prob", type=float, default=None)
    parser.add_argument("--use-focal-loss", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--boundary-loss-weight", type=float, default=None)
    parser.add_argument("--box-boundary-loss-weight", type=float, default=0.05)
    parser.add_argument("--condition-ranking-loss-weight", type=float, default=None)
    parser.add_argument("--selection-ranking-loss-weight", type=float, default=None)
    parser.add_argument("--foreground-bce-pos-weight", type=float, default=None)
    parser.add_argument("--mask-bce-weight", type=float, default=None)
    parser.add_argument("--mask-dice-weight", type=float, default=None)
    parser.add_argument("--mask-tversky-weight", type=float, default=None)
    parser.add_argument("--tversky-alpha", type=float, default=None)
    parser.add_argument("--tversky-beta", type=float, default=None)
    parser.add_argument("--proposal-cls-weight", type=float, default=None)
    parser.add_argument("--condition-cls-weight", type=float, default=None)
    parser.add_argument("--proposal-mask-weight", type=float, default=None)
    parser.add_argument("--empty-mask-suppression-weight", type=float, default=None)
    parser.add_argument("--empty-proposal-suppression-weight", type=float, default=None)
    parser.add_argument("--proposal-positive-weight", type=float, default=None)
    parser.add_argument("--condition-positive-weight", type=float, default=None)
    parser.add_argument("--evidence-positive-weight", type=float, default=None)
    parser.add_argument("--query-diversity-loss-weight", type=float, default=None)
    parser.add_argument("--proposal-mask-diversity-loss-weight", type=float, default=None)
    parser.add_argument("--gate-entropy-loss-weight", type=float, default=None)
    parser.add_argument("--proposal-soft-target-topk", type=int, default=None)
    parser.add_argument("--proposal-soft-target-temperature", type=float, default=None)
    parser.add_argument("--query-usage-balance-loss-weight", type=float, default=None)
    parser.add_argument("--evidence-cls-weight", type=float, default=None)
    parser.add_argument("--evidence-ranking-loss-weight", type=float, default=None)
    parser.add_argument("--selection-proposal-weight", type=float, default=None)
    parser.add_argument("--selection-condition-weight", type=float, default=None)
    parser.add_argument("--selection-temperature", type=float, default=None)
    parser.add_argument("--final-foreground-gate-weight", type=float, default=None)
    parser.add_argument(
        "--final-mask-fusion",
        choices=["weighted_average", "topk_weighted_average", "topk_noisy_or"],
        default=None,
    )
    parser.add_argument("--final-topk", type=int, default=None)
    parser.add_argument("--final-noisy-or-epsilon", type=float, default=None)
    parser.add_argument(
        "--canonical-combo-loss-weights",
        default=None,
        help="Canonical combo loss weights, e.g. 'dem+s2=2.5,dem+s1+s2=1.5'.",
    )
    parser.add_argument("--eval-threshold", type=float, default=None)
    parser.add_argument("--eval-best-threshold", action="store_true")
    parser.add_argument("--min-visualizations", type=int, default=4)
    parser.add_argument("--resume-existing", action="store_true")
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json_file(path: Path) -> dict[str, Any] | None:
    """读取可选 JSON object；不存在或不是 object 时返回 None。"""
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def json_ready(value: Any) -> Any:
    """把 Namespace/Path 等对象转为 JSON 可写结构。"""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, argparse.Namespace):
        return {key: json_ready(item) for key, item in vars(value).items()}
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean_dir_for_overwrite(path: Path, enabled: bool) -> None:
    """显式 --overwrite 时清理旧 run/eval 目录，避免遗留 PNG/JSON 混入新结果。"""
    if enabled and path.exists():
        shutil.rmtree(path)


def release_cuda_memory() -> None:
    """释放阶段间 GPU cache，避免 Qwen cache 阶段挤占后续训练显存。"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def write_phase1_manifest(run_root: Path, args: argparse.Namespace, base_config: QPSalmConfig, modes: list[str]) -> dict[str, Any]:
    """写全局 phase1 manifest。"""
    manifest = {
        "created_at_utc": utc_now(),
        "argv": list(sys.argv),
        "args": json_ready(args),
        "base_config": json_ready(base_config.__dict__),
        "modes": modes,
        "run_root": str(run_root),
    }
    write_json(run_root / "phase1_manifest.json", manifest)
    return manifest


def write_run_manifest(
    run_dir: Path,
    eval_dir: Path,
    mode: str,
    config: QPSalmConfig,
    train_index: Path,
    val_index: Path,
    embedding_cache: Path | None,
    visual_evidence_cache: Path | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """写单个分支 manifest。"""
    manifest = {
        "created_at_utc": utc_now(),
        "mode": mode,
        "run_dir": str(run_dir),
        "eval_dir": str(eval_dir),
        "train_index": str(train_index),
        "val_index": str(val_index),
        "condition_embedding_cache": str(embedding_cache) if embedding_cache is not None else None,
        "visual_evidence_cache": str(visual_evidence_cache) if visual_evidence_cache is not None else None,
        "checkpoint_last": str(run_dir / "checkpoint_last.pt"),
        "checkpoint_best": str(run_dir / "checkpoint_best.pt"),
        "eval_report": str(eval_dir / "eval_report.json"),
        "args": json_ready(args),
        "resolved_config": json_ready(config.__dict__),
    }
    write_json(run_dir / "run_manifest.json", manifest)
    return manifest


def cache_core_indexes(config: QPSalmConfig, out_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    """生成 Phase 1 核心模板 train/val 缓存索引。"""
    train_path = out_dir / "qpsalm_core_train.jsonl"
    val_path = out_dir / "qpsalm_core_val.jsonl"
    summary_path = out_dir / "summary.json"
    if args.reuse_index_cache and train_path.exists() and val_path.exists() and summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    out_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for split in ["train", "val"]:
        index_rel = config.train_index if split == "train" else config.val_index
        reports.append(
            cache_split(
                split=split,
                benchmark_dir=config.benchmark_path(),
                index_rel=str(index_rel),
                core_templates=list(config.core_templates),
                output_dir=out_dir,
                max_rows=args.max_index_rows,
                max_samples=args.max_index_samples,
                strategy=args.index_strategy,
                samples_per_combo=int(args.samples_per_combo),
            )
        )
    summary = {
        "output_dir": str(out_dir),
        "reports": reports,
        "train_index_override": str(train_path),
        "val_index_override": str(val_path),
    }
    write_json(summary_path, summary)
    return summary


def cache_condition_embeddings(
    config_path: str,
    config: QPSalmConfig,
    train_index: Path,
    val_index: Path,
    output_path: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """生成或复用 condition embedding cache。"""
    coverage_config = replace(
        config,
        train_index=str(train_index),
        val_index=str(val_index),
        controller="qwen_cache",
        condition_embedding_cache=str(output_path),
    )
    if args.controller == "text_probe":
        return {"skipped": True, "reason": "controller=text_probe"}
    if output_path.exists() and args.reuse_embedding_cache:
        coverage = assert_qwen_cache_coverage(coverage_config, splits=("train", "val"))
        return {"output": str(output_path), "reused": True, "coverage": coverage}
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"condition cache 已存在，使用 --reuse-embedding-cache 或 --overwrite: {output_path}")
    texts, source_report = collect_condition_texts(config_path, str(train_index), str(val_index))
    if not texts:
        raise RuntimeError("没有收集到 condition_text，无法生成 embedding cache。")
    if args.embedding_backend == "hash-smoke":
        embeddings = hash_smoke_embeddings(texts, hidden_size=int(args.hash_hidden_size))
        device_name = "cpu"
    else:
        embedding_device = args.embedding_device or args.device
        device = resolve_device(embedding_device)
        embeddings = qwen_embeddings(
            texts=texts,
            model_path=config.qwen_model_path,
            decoder_dim=config.decoder_dim,
            device=device,
            batch_size=max(1, int(args.embedding_batch_size)),
            allow_cpu=bool(args.allow_qwen_cpu or config.allow_qwen_cpu),
        )
        device_name = str(device)
    payload = {
        "format": "qpsalm_qwen_condition_cache_v1",
        "backend": args.embedding_backend,
        "model_path": config.qwen_model_path,
        "device": device_name,
        "hidden_size": int(embeddings.shape[1]),
        "texts": texts,
        "embeddings": embeddings.contiguous(),
        "source": source_report,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(payload, output_path)
    coverage = assert_qwen_cache_coverage(coverage_config, splits=("train", "val"))
    return {
        "output": str(output_path),
        "backend": args.embedding_backend,
        "num_texts": len(texts),
        "hidden_size": int(embeddings.shape[1]),
        "model_path": config.qwen_model_path,
        "text_types": source_report.get("text_types"),
        "source": source_report["splits"],
        "coverage": coverage,
    }


def cache_visual_evidence_embeddings(
    config: QPSalmConfig,
    train_index: Path,
    val_index: Path,
    output_path: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """生成或复用 Qwen visual evidence embedding cache。"""
    if args.visual_evidence_backend == "off" or not bool(config.use_visual_evidence):
        return {"skipped": True, "reason": f"backend={args.visual_evidence_backend}, use_visual_evidence={config.use_visual_evidence}"}
    if output_path.exists() and args.reuse_visual_evidence_cache:
        payload = torch.load(output_path, map_location="cpu")
        return {
            "output": str(output_path),
            "reused": True,
            "backend": payload.get("backend") if isinstance(payload, dict) else None,
            "num_samples": len(payload.get("keys", [])) if isinstance(payload, dict) else None,
            "hidden_size": int(payload["embeddings"].shape[1]) if isinstance(payload, dict) and torch.is_tensor(payload.get("embeddings")) else None,
        }
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"visual evidence cache 已存在，使用 --reuse-visual-evidence-cache 或 --overwrite: {output_path}"
        )
    cache_config = replace(
        config,
        train_index=str(train_index),
        val_index=str(val_index),
        visual_evidence_cache=str(output_path),
    )
    samples, source_report = collect_visual_evidence_samples(cache_config)
    if not samples:
        raise RuntimeError("没有收集到 visual evidence 样本，无法生成 visual cache。")
    if args.visual_evidence_backend == "hash-smoke":
        embeddings = hash_smoke_visual_embeddings(samples, hidden_size=int(args.hash_visual_hidden_size))
        device_name = "cpu"
    else:
        embedding_device = args.embedding_device or args.device
        device = resolve_device(embedding_device)
        embeddings = qwen_visual_embeddings(
            samples=samples,
            model_path=config.qwen_model_path,
            device=device,
            batch_size=max(1, int(args.visual_evidence_batch_size)),
            allow_cpu=bool(args.allow_qwen_cpu or config.allow_qwen_cpu),
        )
        device_name = str(device)
    payload = {
        "format": "qpsalm_qwen_visual_evidence_cache_v1",
        "backend": args.visual_evidence_backend,
        "model_path": config.qwen_model_path,
        "device": device_name,
        "hidden_size": int(embeddings.shape[1]),
        "keys": [str(item["key"]) for item in samples],
        "texts": [str(item["text"]) for item in samples],
        "metadata": [item["metadata"] for item in samples],
        "embeddings": embeddings.contiguous(),
        "source": source_report,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(payload, output_path)
    return {
        "output": str(output_path),
        "backend": args.visual_evidence_backend,
        "num_samples": len(samples),
        "hidden_size": int(embeddings.shape[1]),
        "model_path": config.qwen_model_path,
        "source": source_report["splits"],
    }


def apply_common_overrides(config: QPSalmConfig, args: argparse.Namespace) -> QPSalmConfig:
    """应用 runner CLI 中的通用训练覆盖项。"""
    updates: dict[str, Any] = {}
    for key in [
        "batch_size",
        "target_size",
        "num_workers",
        "max_steps",
        "max_train_samples",
        "max_val_samples",
        "max_val_batches",
        "num_visualizations",
        "val_interval",
        "save_interval",
        "keep_recent_checkpoints",
        "visualize_interval",
        "log_interval",
        "lr",
        "weight_decay",
        "warmup_steps",
        "grad_clip",
        "grad_accum_steps",
        "seed",
        "qwen_model_path",
        "modality_dropout",
        "use_gsd_film",
        "use_spatial_modality_gate",
        "use_query_modality_attention",
        "query_modality_feature_weight",
        "use_evidence_reasoning",
        "evidence_reasoning_weight",
        "selection_evidence_weight",
        "use_visual_evidence",
        "visual_evidence_weight",
        "visual_evidence_feature_weight",
        "train_hflip_prob",
        "train_vflip_prob",
        "use_focal_loss",
        "condition_ranking_loss_weight",
        "selection_ranking_loss_weight",
        "foreground_bce_pos_weight",
        "mask_bce_weight",
        "mask_dice_weight",
        "mask_tversky_weight",
        "tversky_alpha",
        "tversky_beta",
        "proposal_cls_weight",
        "condition_cls_weight",
        "proposal_mask_weight",
        "empty_mask_suppression_weight",
        "empty_proposal_suppression_weight",
        "proposal_positive_weight",
        "condition_positive_weight",
        "evidence_positive_weight",
        "query_diversity_loss_weight",
        "proposal_mask_diversity_loss_weight",
        "gate_entropy_loss_weight",
        "proposal_soft_target_topk",
        "proposal_soft_target_temperature",
        "query_usage_balance_loss_weight",
        "evidence_cls_weight",
        "evidence_ranking_loss_weight",
        "selection_proposal_weight",
        "selection_condition_weight",
        "selection_temperature",
        "final_foreground_gate_weight",
        "final_mask_fusion",
        "final_topk",
        "final_noisy_or_epsilon",
        "eval_threshold",
    ]:
        value = getattr(args, key)
        if value is not None:
            updates[key] = value
    combo_weights = parse_combo_loss_weights(args.canonical_combo_loss_weights)
    if combo_weights is not None:
        updates["canonical_combo_loss_weights"] = combo_weights
    return replace(config, **updates)


def build_run_config(
    base: QPSalmConfig,
    args: argparse.Namespace,
    train_index: Path,
    val_index: Path,
    embedding_cache: Path | None,
    visual_evidence_cache: Path | None,
    output_dir: Path,
    use_box_prior: bool,
) -> QPSalmConfig:
    """构建单分支配置；box prior 仅保留为显式 legacy ablation。"""
    config = apply_loss_stage(base, args.loss_stage or base.loss_stage)
    config = apply_common_overrides(config, args)
    if use_box_prior:
        boundary = args.boundary_loss_weight if args.boundary_loss_weight is not None else args.box_boundary_loss_weight
    else:
        boundary = args.boundary_loss_weight if args.boundary_loss_weight is not None else 0.0
    return replace(
        config,
        output_dir=str(output_dir),
        train_index=str(train_index),
        val_index=str(val_index),
        controller=args.controller,
        condition_embedding_cache=str(embedding_cache) if embedding_cache is not None else None,
        visual_evidence_cache=str(visual_evidence_cache) if visual_evidence_cache is not None else None,
        allow_qwen_cpu=bool(args.allow_qwen_cpu or config.allow_qwen_cpu),
        use_box_prior=use_box_prior,
        boundary_loss_weight=float(boundary),
    )


def completed_summary_matches(summary: dict[str, Any], config: QPSalmConfig) -> bool:
    """判断已有 run_summary 是否能代表当前目标配置。"""
    if not (summary.get("acceptance") or {}).get("phase1_smoke_ready"):
        return False
    if not (summary.get("checkpoint_best") or {}).get("exists"):
        return False
    if not (summary.get("artifacts") or {}).get("validation_best", {}).get("exists"):
        return False
    saved_config = summary.get("config") or {}
    return (
        saved_config.get("max_steps") == config.max_steps
        and saved_config.get("target_size") == config.target_size
        and saved_config.get("num_mask_tokens") == config.num_mask_tokens
        and saved_config.get("max_val_batches") == config.max_val_batches
        and saved_config.get("use_box_prior") == config.use_box_prior
        and saved_config.get("loss_stage", "full") == config.loss_stage
        and bool(saved_config.get("use_gsd_film", True)) == bool(config.use_gsd_film)
        and bool(saved_config.get("use_spatial_modality_gate", True)) == bool(config.use_spatial_modality_gate)
        and bool(saved_config.get("use_query_modality_attention", True)) == bool(config.use_query_modality_attention)
        and float(saved_config.get("query_modality_feature_weight", 0.35))
        == float(config.query_modality_feature_weight)
        and bool(saved_config.get("use_evidence_reasoning", True)) == bool(config.use_evidence_reasoning)
        and float(saved_config.get("evidence_reasoning_weight", 0.35)) == float(config.evidence_reasoning_weight)
        and float(saved_config.get("selection_evidence_weight", 0.25)) == float(config.selection_evidence_weight)
        and bool(saved_config.get("use_visual_evidence", True)) == bool(config.use_visual_evidence)
        and (saved_config.get("visual_evidence_cache") or None) == (config.visual_evidence_cache or None)
        and float(saved_config.get("visual_evidence_weight", 0.25)) == float(config.visual_evidence_weight)
        and float(saved_config.get("visual_evidence_feature_weight", 0.15))
        == float(config.visual_evidence_feature_weight)
        and int(saved_config.get("keep_recent_checkpoints", 2)) == int(config.keep_recent_checkpoints)
        and float(saved_config.get("boundary_loss_weight", 0.0)) == float(config.boundary_loss_weight)
        and float(saved_config.get("condition_ranking_loss_weight", 0.0))
        == float(config.condition_ranking_loss_weight)
        and float(saved_config.get("selection_ranking_loss_weight", 0.0))
        == float(config.selection_ranking_loss_weight)
        and float(saved_config.get("foreground_bce_pos_weight", 1.0))
        == float(config.foreground_bce_pos_weight)
        and float(saved_config.get("mask_tversky_weight", 0.0)) == float(config.mask_tversky_weight)
        and float(saved_config.get("tversky_alpha", 0.3)) == float(config.tversky_alpha)
        and float(saved_config.get("tversky_beta", 0.7)) == float(config.tversky_beta)
        and float(saved_config.get("eval_threshold", 0.5)) == float(config.eval_threshold)
        and float(saved_config.get("empty_mask_suppression_weight", 0.0))
        == float(config.empty_mask_suppression_weight)
        and float(saved_config.get("empty_proposal_suppression_weight", 0.0))
        == float(config.empty_proposal_suppression_weight)
        and float(saved_config.get("proposal_positive_weight", 1.0)) == float(config.proposal_positive_weight)
        and float(saved_config.get("condition_positive_weight", 1.0)) == float(config.condition_positive_weight)
        and float(saved_config.get("evidence_positive_weight", 1.0)) == float(config.evidence_positive_weight)
        and float(saved_config.get("query_diversity_loss_weight", 0.0))
        == float(config.query_diversity_loss_weight)
        and float(saved_config.get("proposal_mask_diversity_loss_weight", 0.0))
        == float(config.proposal_mask_diversity_loss_weight)
        and float(saved_config.get("gate_entropy_loss_weight", 0.0)) == float(config.gate_entropy_loss_weight)
        and int(saved_config.get("proposal_soft_target_topk", 1)) == int(config.proposal_soft_target_topk)
        and float(saved_config.get("proposal_soft_target_temperature", 0.10))
        == float(config.proposal_soft_target_temperature)
        and float(saved_config.get("query_usage_balance_loss_weight", 0.0))
        == float(config.query_usage_balance_loss_weight)
        and float(saved_config.get("evidence_cls_weight", 0.0)) == float(config.evidence_cls_weight)
        and float(saved_config.get("evidence_ranking_loss_weight", 0.0))
        == float(config.evidence_ranking_loss_weight)
        and float(saved_config.get("train_hflip_prob", 0.0)) == float(config.train_hflip_prob)
        and float(saved_config.get("train_vflip_prob", 0.0)) == float(config.train_vflip_prob)
        and float(saved_config.get("selection_proposal_weight", 1.0)) == float(config.selection_proposal_weight)
        and float(saved_config.get("selection_condition_weight", 1.0)) == float(config.selection_condition_weight)
        and float(saved_config.get("selection_temperature", 1.0)) == float(config.selection_temperature)
        and float(saved_config.get("final_foreground_gate_weight", 0.0))
        == float(config.final_foreground_gate_weight)
        and str(saved_config.get("final_mask_fusion", "weighted_average")) == str(config.final_mask_fusion)
        and int(saved_config.get("final_topk", 3)) == int(config.final_topk)
        and float(saved_config.get("final_noisy_or_epsilon", 1.0e-5))
        == float(config.final_noisy_or_epsilon)
        and (saved_config.get("canonical_combo_loss_weights") or {})
        == (config.canonical_combo_loss_weights or {})
        and int(saved_config.get("grad_accum_steps", 1)) == int(config.grad_accum_steps)
    )


def run_eval(config: QPSalmConfig, checkpoint: Path, device_name: str, eval_dir: Path) -> dict[str, Any]:
    """从 checkpoint reload 并运行 eval。"""
    device = resolve_device(device_name)
    ds = MultiSourceLandslideDataset(config, split="val", max_samples=config.max_val_samples)
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=qpsalm_collate,
    )
    model = build_model(config, device)
    step = load_checkpoint(checkpoint, model)
    report = evaluate(
        model,
        loader,
        device,
        max_batches=config.max_val_batches if config.max_val_batches and config.max_val_batches > 0 else None,
        visual_dir=eval_dir / "eval_visualizations",
        num_visualizations=config.num_visualizations,
        threshold=float(config.eval_threshold),
        threshold_sweep=config.threshold_sweep,
    )
    report["checkpoint_step"] = step
    eval_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        eval_dir / "eval_manifest.json",
        {
            "created_at_utc": utc_now(),
            "created_by": "qpsalm-run-phase1",
            "eval_dir": str(eval_dir),
            "checkpoint": str(checkpoint),
            "checkpoint_step": step,
            "device": device_name,
            "eval_report": str(eval_dir / "eval_report.json"),
            "resolved_config": json_ready(config.__dict__),
        },
    )
    write_json(eval_dir / "eval_report.json", report)
    return report


def select_eval_checkpoint(run_dir: Path) -> Path:
    """优先使用验证 Dice 最佳 checkpoint，缺失时退回 last。"""
    best = run_dir / "checkpoint_best.pt"
    if best.exists():
        return best
    return run_dir / "checkpoint_last.pt"


def write_diagnosis(run_dir: Path) -> dict[str, Any]:
    """基于 run_summary 生成低精度/覆盖率诊断报告。"""
    summary_path = run_dir / "run_summary.json"
    summary = read_run_summary(summary_path)
    summary["_summary_path"] = str(summary_path)
    report = diagnose(summary, default_diagnose_args())
    output_path = run_dir / "diagnose_report.json"
    write_json(output_path, report)
    return {
        "path": str(output_path),
        "metric_block": report.get("metric_block"),
        "overall": report.get("overall"),
        "issues": report.get("issues"),
        "recommendations": report.get("recommendations"),
    }


def existing_calibrated_eval(eval_dir: Path) -> dict[str, Any] | None:
    """读取已有 calibrated eval 摘要，供 --skip-completed 路径复用。"""
    report_path = eval_dir / "eval_report.json"
    report = read_json_file(report_path)
    if report is None:
        return None
    return {
        "eval_dir": str(eval_dir),
        "eval_report": str(report_path),
        "threshold": report.get("threshold"),
        "overall": (report.get("metrics") or {}).get("overall") if isinstance(report.get("metrics"), dict) else None,
    }


def ensure_threshold_postprocess(
    mode: str,
    run_dir: Path,
    run_root: Path,
    config: QPSalmConfig,
    eval_checkpoint: Path,
    eval_device: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """补齐阈值推荐和可选 best-threshold eval，兼容新 run 与 skipped run。"""
    threshold_recommendations = recommend_thresholds(
        run_dir / "run_summary.json",
        block_name="eval",
        group_prefixes=("canonical_combo=",),
        eval_device=eval_device,
    )
    threshold_recommendations_path = run_dir / "threshold_recommendations.json"
    write_json(threshold_recommendations_path, threshold_recommendations)

    calibrated_eval: dict[str, Any] | None = None
    best_by_dice = threshold_recommendations.get("best_by_dice")
    best_threshold = best_by_dice.get("threshold") if isinstance(best_by_dice, dict) else None
    if args.eval_best_threshold and isinstance(best_threshold, (int, float)):
        calibrated_eval_dir = run_root / f"{mode}_eval_best_dice_threshold"
        existing = None if args.overwrite else existing_calibrated_eval(calibrated_eval_dir)
        if existing is not None:
            calibrated_eval = existing
        else:
            clean_dir_for_overwrite(calibrated_eval_dir, enabled=bool(args.overwrite))
            calibrated_config = replace(config, eval_threshold=float(best_threshold))
            calibrated_report = run_eval(calibrated_config, eval_checkpoint, eval_device, calibrated_eval_dir)
            calibrated_eval = {
                "eval_dir": str(calibrated_eval_dir),
                "eval_report": str(calibrated_eval_dir / "eval_report.json"),
                "threshold": float(best_threshold),
                "overall": (calibrated_report.get("metrics") or {}).get("overall"),
            }
    return {
        "threshold_recommendations": {
            "path": str(threshold_recommendations_path),
            "best_by_dice": threshold_recommendations.get("best_by_dice"),
            "eval_command_best_dice": threshold_recommendations.get("eval_command_best_dice"),
        },
        "calibrated_eval": calibrated_eval,
    }


def run_one_mode(
    mode: str,
    base_config: QPSalmConfig,
    args: argparse.Namespace,
    train_index: Path,
    val_index: Path,
    embedding_cache: Path | None,
    visual_evidence_cache: Path | None,
    run_root: Path,
) -> dict[str, Any]:
    """运行一个分支；baseline 是主路线，box-prior 只用于遗留消融。"""
    use_box_prior = mode == "box-prior"
    run_dir = run_root / mode
    eval_dir = run_root / f"{mode}_eval"
    clean_dir_for_overwrite(run_dir, enabled=bool(args.overwrite))
    clean_dir_for_overwrite(eval_dir, enabled=bool(args.overwrite))
    config = build_run_config(
        base=base_config,
        args=args,
        train_index=train_index,
        val_index=val_index,
        embedding_cache=embedding_cache,
        visual_evidence_cache=visual_evidence_cache,
        output_dir=run_dir,
        use_box_prior=use_box_prior,
    )
    summary_path = run_dir / "run_summary.json"
    checkpoint_path = run_dir / "checkpoint_last.pt"
    if args.skip_completed and summary_path.exists():
        existing_summary = read_run_summary(summary_path)
        if completed_summary_matches(existing_summary, config):
            eval_device = args.eval_device or args.device
            postprocess = ensure_threshold_postprocess(
                mode=mode,
                run_dir=run_dir,
                run_root=run_root,
                config=config,
                eval_checkpoint=select_eval_checkpoint(run_dir),
                eval_device=eval_device,
                args=args,
            )
            diagnosis = write_diagnosis(run_dir)
            return {
                "mode": mode,
                "run_dir": str(run_dir),
                "eval_dir": str(eval_dir),
                "skipped": True,
                "reason": "completed_run_summary_matches_config",
                "eval_overall": (existing_summary.get("eval") or {}).get("overall"),
                "summary": {
                    "path": str(summary_path),
                    "phase1_smoke_ready": (existing_summary.get("acceptance") or {}).get("phase1_smoke_ready"),
                },
                "threshold_recommendations": postprocess["threshold_recommendations"],
                "calibrated_eval": postprocess["calibrated_eval"],
                "diagnosis": diagnosis,
            }
    if checkpoint_path.exists() and not args.overwrite and not args.resume_existing:
        raise FileExistsError(
            f"run_dir 已有 checkpoint。使用 --resume-existing 续训，--skip-completed 跳过，"
            f"或 --overwrite 重跑: {checkpoint_path}"
        )
    run_manifest = write_run_manifest(
        run_dir=run_dir,
        eval_dir=eval_dir,
        mode=mode,
        config=config,
        train_index=train_index,
        val_index=val_index,
        embedding_cache=embedding_cache,
        visual_evidence_cache=visual_evidence_cache,
        args=args,
    )
    resume_path = str(checkpoint_path) if checkpoint_path.exists() and args.resume_existing else None
    train_result = train(config, device_name=args.device, resume=resume_path)
    eval_device = args.eval_device or args.device
    eval_checkpoint = select_eval_checkpoint(run_dir)
    eval_report = run_eval(config, eval_checkpoint, eval_device, eval_dir)
    summary = summarize_run(
        run_dir_ref=run_dir,
        eval_dir_ref=eval_dir,
        min_visualizations=int(args.min_visualizations),
    )
    postprocess = ensure_threshold_postprocess(
        mode=mode,
        run_dir=run_dir,
        run_root=run_root,
        config=config,
        eval_checkpoint=eval_checkpoint,
        eval_device=eval_device,
        args=args,
    )
    diagnosis = write_diagnosis(run_dir)
    return {
        "mode": mode,
        "run_dir": str(run_dir),
        "eval_dir": str(eval_dir),
        "train_result": train_result,
        "eval_overall": (eval_report.get("metrics") or {}).get("overall"),
        "manifest": {
            "path": str(run_dir / "run_manifest.json"),
            "created_at_utc": run_manifest["created_at_utc"],
        },
        "summary": {
            "path": str(run_dir / "run_summary.json"),
            "phase1_smoke_ready": summary.get("acceptance", {}).get("phase1_smoke_ready"),
        },
        "threshold_recommendations": postprocess["threshold_recommendations"],
        "calibrated_eval": postprocess["calibrated_eval"],
        "diagnosis": diagnosis,
    }


def main() -> None:
    args = parse_args()
    if args.overwrite and (args.resume_existing or args.skip_completed):
        raise ValueError("--overwrite 不能与 --resume-existing 或 --skip-completed 同时使用")
    if args.resume_existing or args.skip_completed:
        args.reuse_index_cache = True
        args.reuse_embedding_cache = True
        args.reuse_visual_evidence_cache = True
    base_config = load_config(args.config)
    base_config = apply_common_overrides(base_config, args)
    output_root = resolve_repo_path(args.output_root)
    if output_root is None:
        raise FileNotFoundError(args.output_root)
    run_root = output_root / args.run_name
    index_dir = resolve_repo_path(args.index_cache_dir) if args.index_cache_dir else run_root / "index_cache"
    if index_dir is None:
        raise FileNotFoundError(args.index_cache_dir)
    modes = ["baseline", "box-prior"] if args.mode == "both" else [args.mode]
    phase_manifest = write_phase1_manifest(run_root, args, base_config, modes)
    index_summary = cache_core_indexes(base_config, index_dir, args)
    train_index = Path(index_summary["train_index_override"])
    val_index = Path(index_summary["val_index_override"])

    embedding_cache: Path | None = None
    embedding_summary: dict[str, Any]
    if args.controller == "qwen_cache":
        embedding_cache = resolve_repo_path(args.embedding_cache) if args.embedding_cache else run_root / "condition_cache.pt"
        if embedding_cache is None:
            raise FileNotFoundError(args.embedding_cache)
        embedding_summary = cache_condition_embeddings(
            config_path=args.config,
            config=base_config,
            train_index=train_index,
            val_index=val_index,
            output_path=embedding_cache,
            args=args,
        )
        release_cuda_memory()
    else:
        embedding_summary = {"skipped": True, "reason": "controller=text_probe"}

    visual_evidence_cache: Path | None = None
    visual_evidence_summary: dict[str, Any]
    if args.visual_evidence_backend != "off" and bool(base_config.use_visual_evidence):
        visual_evidence_cache = (
            resolve_repo_path(args.visual_evidence_cache)
            if args.visual_evidence_cache
            else run_root / "visual_evidence_cache.pt"
        )
        if visual_evidence_cache is None:
            raise FileNotFoundError(args.visual_evidence_cache)
        visual_evidence_summary = cache_visual_evidence_embeddings(
            config=base_config,
            train_index=train_index,
            val_index=val_index,
            output_path=visual_evidence_cache,
            args=args,
        )
        release_cuda_memory()
    else:
        visual_evidence_summary = {
            "skipped": True,
            "reason": f"visual_evidence_backend={args.visual_evidence_backend}, use_visual_evidence={base_config.use_visual_evidence}",
        }

    run_reports = [
        run_one_mode(
            mode=mode,
            base_config=base_config,
            args=args,
            train_index=train_index,
            val_index=val_index,
            embedding_cache=embedding_cache,
            visual_evidence_cache=visual_evidence_cache,
            run_root=run_root,
        )
        for mode in modes
    ]
    comparison: dict[str, Any] | None = None
    if {"baseline", "box-prior"}.issubset(set(modes)):
        summary_paths = {item["mode"]: Path(item["summary"]["path"]) for item in run_reports}
        comparison = compare_run_summaries(
            read_run_summary(summary_paths["baseline"]),
            read_run_summary(summary_paths["box-prior"]),
            baseline_name="baseline",
            candidate_name="box-prior",
        )
        comparison_path = run_root / "comparison_baseline_vs_box-prior.json"
        write_json(comparison_path, comparison)
        comparison["path"] = str(comparison_path)
    summary_inputs = [Path(item["summary"]["path"]) for item in run_reports if item.get("summary", {}).get("path")]
    for item in run_reports:
        calibrated = item.get("calibrated_eval") if isinstance(item, dict) else None
        report_path = calibrated.get("eval_report") if isinstance(calibrated, dict) else None
        if report_path:
            summary_inputs.append(Path(report_path))
    analysis_tables = export_analysis_tables(summary_inputs, run_root / "analysis_tables") if summary_inputs else None
    final_summary = {
        "run_root": str(run_root),
        "config": args.config,
        "manifest": {
            "path": str(run_root / "phase1_manifest.json"),
            "created_at_utc": phase_manifest["created_at_utc"],
        },
        "index_cache": index_summary,
        "condition_cache": embedding_summary,
        "visual_evidence_cache": visual_evidence_summary,
        "runs": run_reports,
        "comparison": comparison,
        "analysis_tables": analysis_tables,
    }
    write_json(run_root / "phase1_summary.json", final_summary)
    print(json.dumps(final_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
