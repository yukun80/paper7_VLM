#!/usr/bin/env python3
"""用途：管理保留的 Grounded train supervision 与 compact 训练资产。

命令：既有工作流命令保留；扩展重建必须显式提供已审核配置，仓库不再内置退役配方。
输入：train-only Region collection、监督资产、prompt、本地 VLM 配置及可选重建配置。
输出：模型辅助监督或 6,974 条 compact identity/record 统计。
写入：仅在显式调用生产命令时写新根；验证命令只读，不访问 test。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from oa_groundrag.data.grounded.contracts import LandslideEvidenceError
from oa_groundrag.data.grounded.supervision.compact_training import (
    load_compact_training_messages,
    publish_compact_training_messages,
)
from oa_groundrag.data.grounded.supervision.workflow import (
    ModelAssistedWorkflowPaths,
    prepare_expanded_corpus,
    run_model_assisted_train_workflow,
)
from oa_groundrag.artifacts.identity import sha256_file
from oa_groundrag.data.rs_general.errors import RSGeneralDescError
from oa_groundrag.vlm.errors import VLMError


BENCHMARK_ROOT = REPO_ROOT.parent / "benchmark"
STAGE4_V1_ROOT = BENCHMARK_ROOT / "oa_grounded_stage4_v1"
STAGE4_V2_ROOT = BENCHMARK_ROOT / "oa_grounded_stage4_v2"
MODEL_ASSISTED_WORKFLOW_PATHS = ModelAssistedWorkflowPaths(
    # 退役配方不再随仓库发布。生产命令必须通过 --extension-config 显式替换。
    extension_config_path=Path("/RETIRED_CONFIG_MUST_BE_EXPLICIT"),
    extension_root=(
        STAGE4_V2_ROOT
        / "region_corpus"
        / "mask_grounded_region_corpus_train_extension_v2_7950"
    ),
    collection_root=(
        STAGE4_V2_ROOT
        / "region_collection"
        / "mask_grounded_region_train_collection_v2_8450"
    ),
    legacy_project_root=STAGE4_V1_ROOT / "work" / "stage4_train_expert_v1",
    project_root=(
        STAGE4_V2_ROOT
        / "work"
        / "mask_grounded_region_model_assisted_train_v2_8450"
    ),
    annotation_package_root=(
        STAGE4_V2_ROOT
        / "annotations"
        / "mask_grounded_region_model_assisted_supervision_train_v2_source8450"
    ),
    training_messages_root=(
        STAGE4_V2_ROOT
        / "training_messages"
        / "mask_grounded_region_model_assisted_training_messages_train_v2_source8450"
    ),
    prompt_path=(
        REPO_ROOT
        / "configs"
        / "grounding"
        / "prompts"
        / "single_expert_prompt_v1.txt"
    ),
    draft_config_path=(
        REPO_ROOT
        / "configs"
        / "grounding"
        / "single_expert_qwen3vl_8b_v1.yaml"
    ),
)
COMPACT_TRAINING_ROOT = (
    STAGE4_V2_ROOT
    / "training_messages"
    / "mask_grounded_region_compact_training_messages_train_v3_6974"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Grounded train-only supervision 与 compact 入口",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("prepare-expanded-corpus", "用显式配置准备或验证扩展 Region collection"),
        ("run-train-workflow", "用显式配置运行模型辅助 train-only workflow"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument(
            "--extension-config",
            type=Path,
            default=None,
            help="负责人审核的显式 extension 配置；仓库不提供退役 Eval-dev 配方",
        )
    commands.add_parser(
        "publish-compact-training",
        help="从现有 train-only messages 发布 compact 资产",
    )
    commands.add_parser(
        "validate-compact-training",
        help="严格验证 compact 消息、collection 资产和 assistant 合同",
    )
    return parser


def _print(value: Any, *, stream: Any = sys.stdout) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True), file=stream)


def _explicit_workflow_paths(value: Path | None) -> ModelAssistedWorkflowPaths:
    if value is None:
        raise LandslideEvidenceError(
            "RETIRED_UPSTREAM_UNAVAILABLE",
            "扩展重建配置已退役；必须显式提供 --extension-config",
        )
    path = Path(value)
    if not path.is_file() or path.is_symlink():
        raise LandslideEvidenceError(
            "CONFIG_INVALID",
            f"--extension-config 必须是普通文件：{path}",
        )
    return ModelAssistedWorkflowPaths(
        path,
        MODEL_ASSISTED_WORKFLOW_PATHS.extension_root,
        MODEL_ASSISTED_WORKFLOW_PATHS.collection_root,
        MODEL_ASSISTED_WORKFLOW_PATHS.legacy_project_root,
        MODEL_ASSISTED_WORKFLOW_PATHS.project_root,
        MODEL_ASSISTED_WORKFLOW_PATHS.annotation_package_root,
        MODEL_ASSISTED_WORKFLOW_PATHS.training_messages_root,
        MODEL_ASSISTED_WORKFLOW_PATHS.prompt_path,
        MODEL_ASSISTED_WORKFLOW_PATHS.draft_config_path,
    )


def entrypoint(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "prepare-expanded-corpus":
            result = prepare_expanded_corpus(
                paths=_explicit_workflow_paths(args.extension_config),
                progress_callback=lambda value: _print(value, stream=sys.stderr),
            )
        elif args.command == "run-train-workflow":
            result = run_model_assisted_train_workflow(
                paths=_explicit_workflow_paths(args.extension_config),
                progress_callback=lambda value: _print(value, stream=sys.stderr),
            )
        elif args.command == "publish-compact-training":
            result = publish_compact_training_messages(
                source_training_root=MODEL_ASSISTED_WORKFLOW_PATHS.training_messages_root,
                output_root=COMPACT_TRAINING_ROOT,
            )
        elif args.command == "validate-compact-training":
            artifact = load_compact_training_messages(COMPACT_TRAINING_ROOT)
            result = {
                "ok": True,
                "root": str(artifact.root),
                "compact_id": artifact.manifest["compact_id"],
                "manifest_sha256": sha256_file(artifact.root / "manifest.json"),
                "record_count": len(artifact.rows),
                "authority_counts": artifact.manifest["authority_counts"],
                "formal_acceptance": False,
            }
        else:
            raise AssertionError("unreachable")
        _print(result)
        return 0
    except (LandslideEvidenceError, VLMError, RSGeneralDescError) as error:
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
