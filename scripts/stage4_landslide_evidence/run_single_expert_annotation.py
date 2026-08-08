#!/usr/bin/env python3
"""用途：运行 Stage 4 单专家模型草稿、人工核验和 train-only 训练消息导出。

命令：python scripts/stage4_landslide_evidence/run_single_expert_annotation.py --help
输入：../benchmark 中已冻结的 train Region Corpus 或 val Eval-dev、仓库 prompt、本地 Qwen 配置。
输出：../benchmark/oa_grounded_stage4_v1 下的工作根、单专家 package 和训练 messages。
写入：只写全新工作/发布根；工作快照原子替换，正式 package/messages 拒绝覆盖。
阶段：Stage 4 单专家 annotation 与后续可训练数据准备；不是 Gold 或正式科学评价。
GPU：仅 generate-annotation-drafts 使用本地 GPU；其余命令 CPU-only。
split：训练只允许 500 条 train；开发参考只允许 100 条 val baseline；test 永久拒绝。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from oa_groundrag.landslide_evidence.contracts import LandslideEvidenceError
from oa_groundrag.landslide_evidence.single_expert import (
    AnnotationIntendedUse,
    create_annotation_project,
)
from oa_groundrag.landslide_evidence.single_expert_package import (
    export_verified_annotations,
    validate_verified_annotation_package,
)
from oa_groundrag.landslide_evidence.single_expert_training import (
    export_training_messages,
)
from oa_groundrag.landslide_evidence.single_expert_drafting import (
    generate_annotation_drafts,
)
from oa_groundrag.landslide_evidence.single_expert_workbench import (
    serve_annotation_workbench,
)
from oa_groundrag.landslide_evidence.single_expert_workflow import (
    TrainWorkflowPaths,
    run_train_annotation_workflow,
)
from oa_groundrag.phase3.errors import RSGeneralDescError
from oa_groundrag.phase4.errors import Phase4Error


BENCHMARK_STAGE4_ROOT = REPO_ROOT.parent / "benchmark" / "oa_grounded_stage4_v1"
TRAIN_WORKFLOW_PATHS = TrainWorkflowPaths(
    corpus_root=(
        BENCHMARK_STAGE4_ROOT
        / "region_corpus"
        / "mask_grounded_region_corpus_train_v1_500"
    ),
    project_root=BENCHMARK_STAGE4_ROOT / "work" / "stage4_train_expert_v1",
    annotation_package_root=(
        BENCHMARK_STAGE4_ROOT
        / "annotations"
        / "expert_verified_train_supervision_v1_500"
    ),
    training_messages_root=(
        BENCHMARK_STAGE4_ROOT
        / "training_messages"
        / "mask_grounded_region_training_messages_train_v1_500"
    ),
    prompt_path=(
        REPO_ROOT
        / "configs"
        / "stage4_landslide_evidence"
        / "single_expert_prompt_v1.txt"
    ),
    draft_config_path=(
        REPO_ROOT
        / "configs"
        / "stage4_landslide_evidence"
        / "single_expert_qwen3vl_8b_v1.yaml"
    ),
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 4 单专家最小可训练标注入口")
    commands = parser.add_subparsers(dest="command", required=True)

    workflow = commands.add_parser(
        "run-train-workflow",
        help="创建/恢复 train 草稿和 UI，并在 500/500 后自动发布训练资产",
    )
    workflow.add_argument("--port", type=int, default=7860)

    create = commands.add_parser("create-annotation-project", help="创建 train 或 val-baseline 工作根")
    create.add_argument("--asset-root", type=Path, required=True)
    create.add_argument("--output-root", type=Path, required=True)
    create.add_argument(
        "--intended-use",
        choices=tuple(value.value for value in AnnotationIntendedUse),
        required=True,
    )
    create.add_argument("--prompt", type=Path)
    create.add_argument("--train-project-root", type=Path)

    generate = commands.add_parser("generate-annotation-drafts", help="本地 Qwen3-VL-8B 一次生成草稿")
    generate.add_argument("--project-root", type=Path, required=True)
    generate.add_argument("--config", type=Path, required=True)
    generate.add_argument("--partition", choices=("calibration", "remaining", "all"), required=True)
    generate.add_argument("--limit", type=int)

    serve = commands.add_parser("serve-annotation", help="启动仅监听 127.0.0.1 的 Gradio 工作台")
    serve.add_argument("--project-root", type=Path, required=True)
    serve.add_argument(
        "--partition",
        choices=("calibration", "remaining", "all"),
        required=True,
        help="train 使用 calibration/remaining；val baseline 使用 all",
    )
    serve.add_argument(
        "--view",
        choices=("pending", "all"),
        default="pending",
        help="pending 仅显示已有草稿且尚未核验的记录；all 可重新修改已核验记录",
    )
    serve.add_argument("--port", type=int, default=7860)

    export = commands.add_parser("export-verified-annotations", help="原子发布完整单专家 package")
    export.add_argument("--project-root", type=Path, required=True)
    export.add_argument("--output-root", type=Path, required=True)

    validate = commands.add_parser("validate-annotations", help="独立验证单专家 package")
    validate.add_argument("--asset-root", type=Path, required=True)
    validate.add_argument("--package-root", type=Path, required=True)

    training = commands.add_parser("export-training-messages", help="导出 500 条 train assistant-only messages")
    training.add_argument("--asset-root", type=Path, required=True)
    training.add_argument("--annotations-root", type=Path, required=True)
    training.add_argument("--output-root", type=Path, required=True)
    return parser


def _print(value: Any, *, stream: Any = sys.stdout) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True), file=stream)


def entrypoint(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "run-train-workflow":
            result = run_train_annotation_workflow(
                paths=TRAIN_WORKFLOW_PATHS,
                port=args.port,
                progress_callback=lambda value: _print(value, stream=sys.stderr),
            )
        elif args.command == "create-annotation-project":
            result = create_annotation_project(
                asset_root=args.asset_root,
                output_root=args.output_root,
                intended_use=args.intended_use,
                prompt_path=args.prompt,
                train_project_root=args.train_project_root,
            )
        elif args.command == "generate-annotation-drafts":
            result = generate_annotation_drafts(
                project_root=args.project_root,
                config_path=args.config,
                partition=args.partition,
                limit=args.limit,
                progress_callback=lambda value: _print(value, stream=sys.stderr),
            )
        elif args.command == "serve-annotation":
            serve_annotation_workbench(
                project_root=args.project_root,
                partition=args.partition,
                view=args.view,
                port=args.port,
            )
            result = {"ok": True, "stopped": True, "formal_acceptance": False}
        elif args.command == "export-verified-annotations":
            result = export_verified_annotations(
                project_root=args.project_root,
                output_root=args.output_root,
            )
        elif args.command == "validate-annotations":
            package = validate_verified_annotation_package(
                asset_root=args.asset_root,
                package_root=args.package_root,
            )
            result = {
                "ok": True,
                "root": str(package.root),
                "annotation_count": len(package.annotations),
                "training_eligible": package.manifest["training_eligible"],
                "reference_authority": "single_expert",
                "formal_acceptance": False,
            }
        elif args.command == "export-training-messages":
            result = export_training_messages(
                asset_root=args.asset_root,
                annotations_root=args.annotations_root,
                output_root=args.output_root,
            )
        else:
            raise AssertionError("unreachable")
        _print(result)
        return 0
    except (LandslideEvidenceError, Phase4Error, RSGeneralDescError) as error:
        _print(
            {
                "ok": False,
                "code": error.code,
                "message": str(error),
                "details": error.details,
            },
            stream=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(entrypoint())
