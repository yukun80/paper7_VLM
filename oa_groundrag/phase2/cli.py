"""OA-AuxSeg Phase 2 单一命令行入口。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .engine import (
    load_runtime_config,
    run_evaluation,
    run_inference,
    run_smoke,
    run_training,
)
from .progress import format_compact_training_report


def _path_from_repo(value: str, repo_root: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="OA-GroundRAG Phase 2 OA-AuxSeg 统一入口"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="训练任一消融或 proposed 模型")
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--resume", type=str)
    train.add_argument(
        "--full-report-json",
        action="store_true",
        help="结束时将完整训练报告 JSON 输出到 stdout",
    )

    overfit = subparsers.add_parser(
        "overfit", help="全部 train、全可用辅助模态的容量验收"
    )
    overfit.add_argument("--config", type=Path, required=True)
    overfit.add_argument("--resume", type=str)
    overfit.add_argument(
        "--full-report-json",
        action="store_true",
        help="结束时将完整训练报告 JSON 输出到 stdout",
    )

    evaluate = subparsers.add_parser("evaluate", help="严格重载 checkpoint 并评价")
    evaluate.add_argument("--config", type=Path, required=True)
    evaluate.add_argument("--checkpoint", type=str, required=True)
    evaluate.add_argument("--split", choices=("train", "val", "test"), default="val")
    evaluate.add_argument("--output", type=str, required=True)

    infer = subparsers.add_parser("infer", help="原子导出 JSONL 与 NPZ")
    infer.add_argument("--config", type=Path, required=True)
    infer.add_argument("--checkpoint", type=str, required=True)
    infer.add_argument("--split", choices=("train", "val", "test"), default="val")
    infer.add_argument("--source", type=str)
    infer.add_argument("--limit", type=int)
    infer.add_argument("--output-dir", type=str, required=True)

    smoke = subparsers.add_parser(
        "smoke", help="六个 variant 的真实异构 batch forward/backward/step"
    )
    smoke.add_argument("--config", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[2]
    config = load_runtime_config(arguments.config)
    if arguments.command == "train":
        report = run_training(
            config,
            repo_root=repo_root,
            capacity_overfit=False,
            resume_checkpoint=(
                _path_from_repo(arguments.resume, repo_root)
                if arguments.resume
                else None
            ),
        )
    elif arguments.command == "overfit":
        if config.variant != "proposed_dropout":
            raise ValueError("容量过拟合验收必须使用 proposed_dropout 结构")
        if config.max_steps > 1000:
            raise ValueError("容量过拟合验收最多允许 1000 steps")
        report = run_training(
            config,
            repo_root=repo_root,
            capacity_overfit=True,
            resume_checkpoint=(
                _path_from_repo(arguments.resume, repo_root)
                if arguments.resume
                else None
            ),
        )
    elif arguments.command == "evaluate":
        report = run_evaluation(
            config,
            repo_root=repo_root,
            checkpoint_path=_path_from_repo(arguments.checkpoint, repo_root),
            split=arguments.split,
            output_path=_path_from_repo(arguments.output, repo_root),
        )
    elif arguments.command == "infer":
        if arguments.limit is not None and arguments.limit <= 0:
            raise ValueError("--limit 必须大于 0")
        report = run_inference(
            config,
            repo_root=repo_root,
            checkpoint_path=_path_from_repo(arguments.checkpoint, repo_root),
            split=arguments.split,
            source=arguments.source,
            limit=arguments.limit,
            output_dir=_path_from_repo(arguments.output_dir, repo_root),
        )
    elif arguments.command == "smoke":
        report = run_smoke(config, repo_root=repo_root)
    else:
        raise AssertionError(arguments.command)
    if (
        arguments.command in {"train", "overfit"}
        and not arguments.full_report_json
    ):
        print(format_compact_training_report(report))
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
