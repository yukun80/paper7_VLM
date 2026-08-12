#!/usr/bin/env python3
"""用途：准备 Stage 4 v2 扩展 Region、模型辅助监督及 Stage 5 compact 训练资产。

命令：提供扩展准备、生成工作流、compact 发布与 compact 严格验证四个固定子命令。
输入：冻结 v1 train Region、既有 500 条单专家工作根、仓库 prompt 与本地 Qwen 配置。
输出：../benchmark/oa_grounded_stage4_v2 下的 7,950 扩展、8,450 集合、工作与发布根。
写入：只创建固定新根；合法既有根复用，非法既有根失败，正式 package/messages 不覆盖。
阶段：Stage 4 监督准备与 Stage 5 compact 发布；不是 Gold、人工共识或正式科学评价。
运行：草稿只由本地模型一次 runtime 补齐；不查询 GPU，不调用外部 API，不启动 UI。
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
from oa_groundrag.landslide_evidence.compact_training import (
    load_compact_training_messages,
    publish_compact_training_messages,
)
from oa_groundrag.landslide_evidence.model_assisted_workflow import (
    ModelAssistedWorkflowPaths,
    prepare_expanded_corpus,
    run_model_assisted_train_workflow,
)
from oa_groundrag.phase3.common import sha256_file
from oa_groundrag.phase3.errors import RSGeneralDescError
from oa_groundrag.phase4.errors import Phase4Error


BENCHMARK_ROOT = REPO_ROOT.parent / "benchmark"
STAGE4_V1_ROOT = BENCHMARK_ROOT / "oa_grounded_stage4_v1"
STAGE4_V2_ROOT = BENCHMARK_ROOT / "oa_grounded_stage4_v2"
MODEL_ASSISTED_WORKFLOW_PATHS = ModelAssistedWorkflowPaths(
    extension_config_path=(
        REPO_ROOT
        / "configs"
        / "stage4_landslide_evidence"
        / "region_corpus_train_extension_v2_7950.yaml"
    ),
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
COMPACT_TRAINING_ROOT = (
    STAGE4_V2_ROOT
    / "training_messages"
    / "mask_grounded_region_compact_training_messages_train_v3_6974"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stage 4 v2 模型辅助 train supervision 固定入口",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "prepare-expanded-corpus",
        help="准备或严格复用 7,950 扩展与 8,450 Region 集合",
    )
    commands.add_parser(
        "run-train-workflow",
        help="一次 runtime 补齐缺失草稿，并在 8,450/8,450 后发布训练资产",
    )
    commands.add_parser(
        "publish-compact-training",
        help="从已验证 v2 messages 原子发布独立 6,974 条 compact 训练资产",
    )
    commands.add_parser(
        "validate-compact-training",
        help="严格验证 compact 消息、collection 资产和 assistant 合同",
    )
    return parser


def _print(value: Any, *, stream: Any = sys.stdout) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True), file=stream)


def entrypoint(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "prepare-expanded-corpus":
            result = prepare_expanded_corpus(
                paths=MODEL_ASSISTED_WORKFLOW_PATHS,
                progress_callback=lambda value: _print(value, stream=sys.stderr),
            )
        elif args.command == "run-train-workflow":
            result = run_model_assisted_train_workflow(
                paths=MODEL_ASSISTED_WORKFLOW_PATHS,
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
