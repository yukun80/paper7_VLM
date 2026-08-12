"""Mask-grounded evidence 消息构造；External 消息由 phase3 renderer 提供。"""

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


def build_mask_grounded_region_messages(
    record: Mapping[str, Any],
    *,
    asset_root: Path,
    representation_mode: str | None = None,
    assistant_target: str | None = None,
    allow_audit_only: bool = False,
    instruction: str | None = None,
) -> list[dict[str, Any]]:
    """构建 Stage 4 v2 有序多图消息，不改变旧 generic/Gate B renderer。

    二值 mask 作为独立图像发送；audit overlay 只有调用方显式启用消融时才可进入消息。
    """

    from oa_groundrag.landslide_evidence.region_contracts import (
        FORMAL_ROLES,
        RepresentationMode,
        validate_region_record,
    )
    from oa_groundrag.phase4.outputs import region_output_contract

    row = validate_region_record(record)
    try:
        mode = RepresentationMode(representation_mode or row["representation_mode"])
    except (TypeError, ValueError) as error:
        raise ContractError(ReasonCode.INVALID_ENUM, "representation_mode 非法") from error
    if mode is RepresentationMode.OVERLAY_AUDIT_BASELINE:
        if not allow_audit_only:
            raise ContractError(
                ReasonCode.ASSET_ROLE_LEAKAGE,
                "overlay_audit_baseline 必须显式 allow_audit_only=True",
            )
        roles = ("audit_overlay",)
    else:
        roles = FORMAL_ROLES[mode]
    assets = row["assets"]
    for role in roles:
        if not isinstance(assets.get(role), str):
            raise ContractError(ReasonCode.ASSET_MISSING, f"representation 缺少资产：{role}")
        if role == "binary_mask":
            from PIL import Image
            from oa_groundrag.phase4.evidence import binary_mask_array

            with Image.open(_asset_path(asset_root, str(assets[role]))) as mask_image:
                # 先严格转为 bool，确认 0/255 资产不会被误当灰度置信度；正式 Qwen wrapper
                # 仍消费同一无损 PNG 路径，不另行实现 tokenizer 或视觉预处理。
                binary_mask_array(mask_image)
    role_text = {
        "optical_full": (
            "第1幅图是完整、未加标记的原始光学影像。它用于观察整个遥感场景和目标周围环境。"
        ),
        "binary_mask": (
            "下一幅图是与原图严格对齐的单通道二值关注区域 mask；白色仅表示关注位置，"
            "不是地物真实颜色。"
        ),
        "context_crop": (
            "下一幅图是直接从原始 RGB 裁剪的无标记区域图。crop 边缘不是目标边界。"
        ),
        "audit_overlay": (
            "这是一幅仅用于审计消融的彩色 overlay；其颜色不得作为地物证据。"
        ),
    }
    content: list[dict[str, Any]] = []
    for role in roles:
        content.append({"type": "text", "text": role_text[role]})
        content.append({
            "type": "image",
            "image": _asset_path(asset_root, str(assets[role])),
        })
    contract = {
        "strict_output_contract": region_output_contract(row["target_status"]),
        "program_facts_are_read_only": row["program_facts"],
        "formal_model_input_roles": list(roles) if mode is not RepresentationMode.OVERLAY_AUDIT_BASELINE else [],
        "audit_only": mode is RepresentationMode.OVERLAY_AUDIT_BASELINE,
    }
    if instruction is not None and (
        not isinstance(instruction, str) or not instruction.strip()
    ):
        raise ContractError(
            ReasonCode.TYPE_MISMATCH,
            "instruction 必须是非空字符串或 null",
        )
    instruction_prefix = (
        "" if instruction is None else f"用户任务：{instruction.strip()}\n"
    )
    content.append({
        "type": "text",
        "text": (
            instruction_prefix
            + "先观察完整影像中的总体遥感场景，再定位 mask 指定区域；只描述当前影像直接支持"
            "的视觉事实，并严格区分目标内部、周围环境、以及区域与环境之间的视觉差异。"
            "不要把白色 mask 当作真实"
            "颜色，不要把 crop 边缘当作目标边界，不要重写 bbox、面积、质心、组件数或 crop window。"
            "可以指出视觉混淆对象，但禁止断言发生时间、触发原因、精确运动、稳定性、风险、"
            "灾害规模、实际威胁或现场地质结构。证据不足时必须写 limitations。"
            "target_status 和 evidence_sufficiency 只能使用合同列出的英文 ASCII 枚举，禁止翻译。"
            "必须保持所有嵌套对象、数组及字段类型，short_summary 必须为非空字符串。"
            "不要复制统一的保守答案；每个无法判断都必须有与当前影像相关的具体可见性原因。"
            "只返回一个严格 JSON 对象，不允许额外 prose。\nContract: "
            + canonical_json(contract)
        ),
    })
    messages: list[dict[str, Any]] = [{"role": "user", "content": content}]
    if assistant_target is not None:
        if not isinstance(assistant_target, str) or not assistant_target.strip():
            raise ContractError(ReasonCode.TYPE_MISMATCH, "assistant_target 必须是非空字符串或 null")
        messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": assistant_target.strip()}],
        })
    return messages


def render_mask_grounded_region_messages(
    *,
    asset_root: Path,
    output_root: Path,
    representation_mode: str | None = None,
    allow_audit_only: bool = False,
) -> dict[str, Any]:
    """将已验证 records 渲染为新的 message artifact；不调用 processor 或模型。"""

    import os

    from oa_groundrag.landslide_evidence.contracts import fail
    from oa_groundrag.landslide_evidence.region_pipeline import ledger_rows
    from oa_groundrag.phase3.common import read_jsonl, sha256_file, sha256_text
    from oa_groundrag.phase4.artifacts import AtomicArtifactDirectory

    asset_root = Path(os.path.abspath(asset_root))
    output_root = Path(os.path.abspath(output_root))
    if output_root.exists() or output_root.is_symlink():
        fail("OUTPUT_EXISTS", f"message output 已存在：{output_root}")
    records = read_jsonl(asset_root / "records.jsonl")
    queue = {row["record_id"]: row for row in read_jsonl(asset_root / "annotation_queue.jsonl")}
    source_manifest_sha = sha256_file(asset_root / "manifest.json")
    rows = []
    for record in records:
        messages = build_mask_grounded_region_messages(
            record,
            asset_root=asset_root,
            representation_mode=representation_mode,
            allow_audit_only=allow_audit_only,
        )
        mode = representation_mode or record["representation_mode"]
        rows.append({
            "schema_version": "rs_vlm.mask_grounded_region_messages.v2",
            "record_id": record["record_id"],
            "parent_id": record["parent_id"],
            "representation_mode": mode,
            "asset_manifest_sha256": source_manifest_sha,
            "asset_identity_sha256": queue[record["record_id"]]["asset_identity_sha256"],
            "messages_sha256": sha256_text(canonical_json(messages)),
            "messages": messages,
        })
    with AtomicArtifactDirectory(output_root) as writer:
        assert writer.staging is not None
        writer.write_jsonl("messages.jsonl", rows)
        ledger = ledger_rows(writer.staging, ["messages.jsonl"])
        writer.write_jsonl("SHA256SUMS.jsonl", ledger)
        writer.write_json("manifest.json", {
            "schema_version": "rs_vlm.mask_grounded_region_message_artifact.v1",
            "source_root": str(asset_root),
            "source_manifest_sha256": source_manifest_sha,
            "record_count": len(rows),
            "representation_mode_override": representation_mode,
            "audit_only": allow_audit_only,
            "messages_sha256": sha256_file(writer.path("messages.jsonl")),
            "ledger_sha256": sha256_file(writer.path("SHA256SUMS.jsonl")),
            "model_invoked": False,
        })
        root = writer.publish()
    return {
        "ok": True,
        "root": str(root),
        "manifest_sha256": sha256_file(root / "manifest.json"),
        "record_count": len(rows),
        "model_invoked": False,
    }
