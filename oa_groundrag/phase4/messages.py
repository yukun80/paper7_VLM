"""OA evidence 消息构造；External 消息继续由 Phase 2 renderer 提供。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from oa_groundrag.phase3.common import canonical_json

from .contracts import (
    MODEL_OUTPUT_SCHEMA_VERSION,
    EvidenceBundle,
    TargetStatus,
)
from .errors import ContractError, ReasonCode


def _asset_path(root: Path, relative: str) -> str:
    path = root / relative
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise ContractError(
            ReasonCode.PATH_ESCAPE,
            f"evidence asset 路径逃逸：{relative}",
        ) from error
    if not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1:
        raise ContractError(
            ReasonCode.OUTPUT_LINK,
            f"evidence asset 必须是普通单链接文件：{relative}",
        )
    return str(path.resolve())


def build_mask_grounded_messages(
    bundle: EvidenceBundle,
    *,
    evidence_root: Path,
    instruction: str,
    assistant_target: str | None = None,
) -> list[dict[str, Any]]:
    if not isinstance(instruction, str) or not instruction.strip():
        raise ContractError(
            ReasonCode.TYPE_MISMATCH,
            "instruction 必须是非空字符串",
        )
    content: list[dict[str, Any]] = []
    role_descriptions = {
        "optical_full": "全幅光学影像",
        "mask_overlay": "已有目标 mask 的光学叠加图",
        "context_crop": "已有目标 mask 的上下文裁剪",
    }
    for asset in bundle.assets:
        role = role_descriptions.get(asset.role, asset.role)
        content.append(
            {
                "type": "text",
                "text": f"Evidence {asset.evidence_id}: {role}.",
            }
        )
        content.append(
            {
                "type": "image",
                "image": _asset_path(evidence_root, asset.relative_path),
            }
        )
    contract = {
        "required_schema_version": MODEL_OUTPUT_SCHEMA_VERSION,
        "allowed_top_level_fields": [
            "schema_version",
            "target_status",
            "description",
            "answer",
            "evidence_sufficiency",
            "claims",
            "limitations",
        ],
        "claim_contract": {
            "fields": ["text", "evidence_ids"],
            "valid_evidence_ids": sorted(bundle.evidence_ids),
        },
        "program_computed_facts_are_read_only": dict(
            bundle.deterministic_facts
        ),
        "known_limitations": list(bundle.limitations),
        "target_status": bundle.target_status.value,
    }
    if bundle.target_status is TargetStatus.NO_TARGET:
        target_rule = (
            "The program reports no target. Do not imply target presence and do "
            "not invent a box, crop, location, morphology, or region geometry."
        )
    else:
        target_rule = (
            "Describe only the selected existing mask region. Do not restate or "
            "modify program-computed geometry as a model-generated fact."
        )
    content.append(
        {
            "type": "text",
            "text": (
                f"{instruction.strip()}\n"
                f"{target_rule}\n"
                "Every claim must cite one or more valid evidence_ids. "
                "Respect alignment, coverage, unit, and sign limitations. "
                "Return one strict JSON object and no surrounding prose.\n"
                f"Contract: {canonical_json(contract)}"
            ),
        }
    )
    messages: list[dict[str, Any]] = [{"role": "user", "content": content}]
    if assistant_target is not None:
        if not isinstance(assistant_target, str) or not assistant_target.strip():
            raise ContractError(
                ReasonCode.TYPE_MISMATCH,
                "assistant_target 必须是非空字符串或 null",
            )
        messages.append(
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": assistant_target.strip()}
                ],
            }
        )
    return messages


def strip_assistant_message(
    messages: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if not messages or messages[0].get("role") != "user":
        raise ContractError(
            ReasonCode.TYPE_MISMATCH,
            "messages 必须以 user 开始",
        )
    result = [dict(message) for message in messages]
    if result[-1].get("role") == "assistant":
        result.pop()
    if len(result) != 1:
        raise ContractError(
            ReasonCode.TYPE_MISMATCH,
            "phase4 单轮消息必须只有一个 user 和可选 assistant",
        )
    return result
