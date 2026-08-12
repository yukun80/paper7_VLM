"""Stage 4 单专家本地 Gradio 核验工作台；不调用外部 API。"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import time
from typing import Any, Mapping, MutableMapping, Sequence

from oa_groundrag.grounding.contracts import TargetStatus
from oa_groundrag.vlm.errors import ContractError
from oa_groundrag.grounding.outputs import (
    REGION_OUTPUT_SCHEMA_VERSION,
    RegionDraftQualityStatus,
    RegionEvidenceSufficiency,
    assess_region_draft_quality,
    parse_region_model_output,
)

from ..contracts import LandslideEvidenceError, fail
from .project import (
    annotation_work_item,
    default_region_description,
    load_annotation_project,
    load_model_drafts,
    load_verified_work,
    save_annotation_work,
    verify_annotation_work,
)


ANNOTATION_PARTITIONS = ("calibration", "remaining", "all")
ANNOTATION_VIEWS = ("pending", "all")
LOOPBACK_PROXY_BYPASS = ("127.0.0.1", "localhost")

# 表单顺序是 UI 回调和永久测试共同依赖的稳定合同。数组字段在界面中采用“一行一项”，
# 空行会被忽略；重复项必须交给严格 parser 拒绝，不能在界面层静默去重。
ANNOTATION_FORM_FIELDS = (
    "target_status",
    "target_appearance.tone",
    "target_appearance.texture",
    "target_appearance.vegetation_or_exposure",
    "target_appearance.homogeneity",
    "target_appearance.boundary_visibility",
    "target_morphology.shape",
    "target_morphology.fragmentation",
    "target_morphology.qualitative_orientation",
    "surrounding_environment.land_cover",
    "surrounding_environment.nearby_objects",
    "surrounding_environment.visible_terrain_context",
    "surrounding_environment.human_disturbance",
    "region_context_contrast.tone_contrast",
    "region_context_contrast.texture_contrast",
    "region_context_contrast.vegetation_contrast",
    "region_context_contrast.boundary_transition",
    "region_context_contrast.adjacency",
    "possible_confusers",
    "evidence_sufficiency",
    "short_summary",
    "limitations",
)
_FORM_ARRAY_POSITIONS = frozenset({9, 10, 11, 12, 17, 18, 21})
_NO_TARGET_LOCKED_POSITIONS = frozenset(range(1, 19))
_FORM_START = 10


def _canonical_editor_text(value: Mapping[str, Any]) -> str:
    """生成供专家审阅的稳定缩进 JSON；正式身份仍由 canonical serializer 计算。"""

    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def _lines_to_items(value: Any, *, location: str) -> list[str]:
    """把多行输入转换为数组；只忽略空行，不修复重复项。"""

    if not isinstance(value, str):
        fail("ANNOTATION_INVALID", f"{location} 必须是多行文本")
    return [line.strip() for line in value.splitlines() if line.strip()]


def description_to_form_values(
    value: str | Mapping[str, Any],
) -> tuple[str, ...]:
    """严格解析描述并映射到混合核验表单，避免表单与 parser 合同漂移。"""

    parsed = parse_region_model_output(value).to_dict()
    appearance = parsed["target_appearance"]
    morphology = parsed["target_morphology"]
    surrounding = parsed["surrounding_environment"]
    contrast = parsed["region_context_contrast"]
    raw_values: tuple[Any, ...] = (
        parsed["target_status"],
        appearance["tone"],
        appearance["texture"],
        appearance["vegetation_or_exposure"],
        appearance["homogeneity"],
        appearance["boundary_visibility"],
        morphology["shape"],
        morphology["fragmentation"],
        morphology["qualitative_orientation"],
        surrounding["land_cover"],
        surrounding["nearby_objects"],
        surrounding["visible_terrain_context"],
        surrounding["human_disturbance"],
        contrast["tone_contrast"],
        contrast["texture_contrast"],
        contrast["vegetation_contrast"],
        contrast["boundary_transition"],
        contrast["adjacency"],
        parsed["possible_confusers"],
        parsed["evidence_sufficiency"],
        parsed["short_summary"],
        parsed["limitations"],
    )
    return tuple(
        "\n".join(value) if index in _FORM_ARRAY_POSITIONS else str(value)
        for index, value in enumerate(raw_values)
    )


def form_values_to_description(values: Sequence[Any]) -> dict[str, Any]:
    """由当前表单重建并严格验证专家最终描述。"""

    if len(values) != len(ANNOTATION_FORM_FIELDS):
        fail(
            "ANNOTATION_INVALID",
            "专家表单字段数量不匹配",
            details={
                "expected": len(ANNOTATION_FORM_FIELDS),
                "actual": len(values),
            },
        )
    normalized: list[Any] = []
    for index, value in enumerate(values):
        if index in _FORM_ARRAY_POSITIONS:
            normalized.append(
                _lines_to_items(value, location=ANNOTATION_FORM_FIELDS[index])
            )
        elif not isinstance(value, str):
            fail(
                "ANNOTATION_INVALID",
                f"{ANNOTATION_FORM_FIELDS[index]} 必须是字符串",
            )
        else:
            normalized.append(value.strip())
    row = {
        "schema_version": REGION_OUTPUT_SCHEMA_VERSION,
        "target_status": normalized[0],
        "target_appearance": {
            "tone": normalized[1],
            "texture": normalized[2],
            "vegetation_or_exposure": normalized[3],
            "homogeneity": normalized[4],
            "boundary_visibility": normalized[5],
        },
        "target_morphology": {
            "shape": normalized[6],
            "fragmentation": normalized[7],
            "qualitative_orientation": normalized[8],
        },
        "surrounding_environment": {
            "land_cover": normalized[9],
            "nearby_objects": normalized[10],
            "visible_terrain_context": normalized[11],
            "human_disturbance": normalized[12],
        },
        "region_context_contrast": {
            "tone_contrast": normalized[13],
            "texture_contrast": normalized[14],
            "vegetation_contrast": normalized[15],
            "boundary_transition": normalized[16],
            "adjacency": normalized[17],
        },
        "possible_confusers": normalized[18],
        "evidence_sufficiency": normalized[19],
        "short_summary": normalized[20],
        "limitations": normalized[21],
    }
    return parse_region_model_output(row).to_dict()


def apply_advanced_json(
    editor_text: str,
    *,
    expected_target_status: str | None = None,
) -> tuple[tuple[str, ...], dict[str, Any], str]:
    """严格把高级 JSON 同步回表单；禁止借此改变当前 record 的 target status。"""

    parsed = parse_region_model_output(editor_text).to_dict()
    if (
        expected_target_status is not None
        and parsed["target_status"] != expected_target_status
    ):
        fail(
            "ANNOTATION_INVALID",
            "高级 JSON 的 target_status 与当前 record 不一致",
        )
    return (
        description_to_form_values(parsed),
        parsed,
        _canonical_editor_text(parsed),
    )


def form_interactive_flags(
    target_status: str,
    *,
    has_record: bool = True,
) -> tuple[bool, ...]:
    """返回稳定控件锁定合同；target_status 永远只读。"""

    if target_status not in {
        TargetStatus.TARGET_PRESENT.value,
        TargetStatus.NO_TARGET.value,
        "",
    }:
        fail("ANNOTATION_INVALID", f"未知 target_status：{target_status}")
    return tuple(
        bool(
            has_record
            and position != 0
            and not (
                target_status == TargetStatus.NO_TARGET.value
                and position in _NO_TARGET_LOCKED_POSITIONS
            )
        )
        for position in range(len(ANNOTATION_FORM_FIELDS))
    )


def ensure_loopback_proxy_bypass(
    environ: MutableMapping[str, str] | None = None,
) -> dict[str, str]:
    """为 Gradio 回环自检补齐精确代理绕过项，不改写进程外的代理配置。"""

    target = os.environ if environ is None else environ
    updated: dict[str, str] = {}
    for variable in ("NO_PROXY", "no_proxy"):
        values: list[str] = []
        seen: set[str] = set()
        for raw_value in target.get(variable, "").split(","):
            value = raw_value.strip()
            if value and value not in seen:
                values.append(value)
                seen.add(value)
        # `127.*` 等宽泛写法对部分 httpx 版本无效，因此必须保留并追加精确地址。
        for value in LOOPBACK_PROXY_BYPASS:
            if value not in seen:
                values.append(value)
                seen.add(value)
        target[variable] = ",".join(values)
        updated[variable] = target[variable]
    return updated


def _close_after_launch_failure(app: Any, error: Exception) -> None:
    """关闭可能已部分启动的 Gradio 服务，并以统一 reason code 报告启动失败。"""

    details = {
        "error_type": type(error).__name__,
        "error_message": str(error),
    }
    try:
        app.close()
    except Exception as close_error:  # pragma: no cover - 只记录第三方清理异常
        details.update({
            "close_error_type": type(close_error).__name__,
            "close_error_message": str(close_error),
        })
    fail(
        "ANNOTATION_UI_START_FAILED",
        "Gradio 本地标注服务启动失败",
        details=details,
    )


def _view_options(partition: str, view: str) -> tuple[str, str]:
    if partition not in ANNOTATION_PARTITIONS:
        fail("ANNOTATION_INVALID", f"未知 annotation partition：{partition}")
    if view not in ANNOTATION_VIEWS:
        fail("ANNOTATION_INVALID", f"未知 annotation view：{view}")
    return partition, view


def _visible_assignments(
    project_root: Path,
    *,
    partition: str,
    view: str,
) -> list[dict[str, Any]]:
    """只把已有草稿送入专家队列，避免把未生成的 500 条 assignment 混入核验。"""

    partition, view = _view_options(partition, view)
    _, assignments = load_annotation_project(project_root)
    available = {row["partition"] for row in assignments}
    if partition not in available:
        fail(
            "SPLIT_FORBIDDEN",
            f"当前项目不包含 partition={partition}；可用值为 {sorted(available)}",
        )
    drafts = load_model_drafts(project_root, assignments=assignments)
    verified = load_verified_work(
        project_root,
        assignments=assignments,
        drafts=drafts,
    )
    return [
        row
        for row in assignments
        if row["partition"] == partition
        and row["record_id"] in drafts
        and (view == "all" or row["record_id"] not in verified)
    ]


def annotation_partition_status(
    project_root: Path | str,
    *,
    partition: str,
) -> dict[str, Any]:
    """重算一个工作分区的草稿和核验进度，供 UI 自动关闭与状态机复用。"""

    root = Path(project_root).resolve()
    partition, _ = _view_options(partition, "pending")
    _, assignments = load_annotation_project(root)
    selected = [row for row in assignments if row["partition"] == partition]
    if not selected:
        fail("SPLIT_FORBIDDEN", f"当前项目不包含 partition={partition}")
    drafts = load_model_drafts(root, assignments=assignments)
    verified = load_verified_work(
        root,
        assignments=assignments,
        drafts=drafts,
    )
    record_ids = {str(row["record_id"]) for row in selected}
    drafted = sum(record_id in drafts for record_id in record_ids)
    completed = sum(record_id in verified for record_id in record_ids)
    return {
        "partition": partition,
        "total": len(record_ids),
        "drafted": drafted,
        "verified": completed,
        "complete": completed == len(record_ids),
        "formal_acceptance": False,
    }


def annotation_partition_draft_quality(
    project_root: Path | str,
    *,
    partition: str,
) -> dict[str, Any]:
    """重算一个分区的草稿语义质量；该诊断不写入最终 annotation。"""

    root = Path(project_root).resolve()
    partition, _ = _view_options(partition, "pending")
    _, assignments = load_annotation_project(root)
    selected = [row for row in assignments if row["partition"] == partition]
    if not selected:
        fail("SPLIT_FORBIDDEN", f"当前项目不包含 partition={partition}")
    drafts = load_model_drafts(root, assignments=assignments)
    quality_counts = {status.value: 0 for status in RegionDraftQualityStatus}
    parse_invalid = 0
    missing = 0
    target_template_copies = 0
    for assignment in selected:
        draft = drafts.get(str(assignment["record_id"]))
        if draft is None:
            missing += 1
            continue
        if draft.get("parse_status") != "valid" or not isinstance(
            draft.get("description"),
            Mapping,
        ):
            parse_invalid += 1
            continue
        assessment = assess_region_draft_quality(draft["description"])
        quality_counts[assessment.status.value] += 1
        if (
            assignment["target_status"] == TargetStatus.TARGET_PRESENT.value
            and bool(assessment.metrics["template_match"])
        ):
            target_template_copies += 1
    return {
        "partition": partition,
        "total": len(selected),
        "drafted": len(selected) - missing,
        "missing": missing,
        "parse_invalid": parse_invalid,
        "quality_counts": quality_counts,
        "target_template_copies": target_template_copies,
        "formal_acceptance": False,
    }


def annotation_view_item(
    project_root: Path | str,
    ordinal: int,
    *,
    partition: str,
    view: str,
) -> dict[str, Any]:
    """返回当前 partition/view 中的一条；ordinal 是视图内序号而非全局序号。"""

    root = Path(project_root).resolve()
    if isinstance(ordinal, bool) or not isinstance(ordinal, int):
        fail("ANNOTATION_INVALID", "ordinal 非法")
    visible = _visible_assignments(root, partition=partition, view=view)
    if not visible:
        # 项目至少包含 100/500 条 assignment。复用核心读取可继续验证 status 和资产身份，
        # 但空视图不会把这条隐藏 assignment 暴露给人工界面。
        item = annotation_work_item(root, 0)
        item.update(
            {
                "empty": True,
                "ordinal": 0,
                "total": 0,
                "global_ordinal": None,
                "record_id": "",
                "source": "",
                "split": "",
                "target_status": "",
                "partition": partition,
                "view": view,
                "optical_full": None,
                "binary_mask": None,
                "context_crop": None,
                "audit_overlay": None,
                "program_facts": {},
                "model_draft": None,
                "editor_text": "",
                "verified": False,
            }
        )
        return item
    index = ordinal % len(visible)
    assignment = visible[index]
    item = annotation_work_item(root, int(assignment["ordinal"]))
    item.update(
        {
            "empty": False,
            "ordinal": index,
            "total": len(visible),
            "global_ordinal": int(assignment["ordinal"]),
            "partition": partition,
            "view": view,
        }
    )
    return item


def _status_text(item: dict[str, Any], message: str = "") -> str:
    status = item["status"]
    prefix = f"{message}\n\n" if message else ""
    progress = (
        prefix
        + f"进度：{status['verified']}/{status['total']} 已核验；"
        + f"草稿 {status['drafted']}/{status['total']}；"
        + f"解析失败 {status['failed_drafts']}。"
    )
    if item.get("empty"):
        return (
            progress
            + f"\n\n当前 partition=`{item['partition']}`、view=`{item['view']}` "
            + "没有可显示记录。"
        )
    return progress + f"\n\n当前视图共有 {item['total']} 条。"


def _item_values(item: dict[str, Any], message: str = "") -> tuple[Any, ...]:
    """把工作项展开为稳定的 Gradio 输出序列。"""

    if item.get("empty"):
        meta = (
            "### 当前队列已完成或尚无草稿\n"
            f"partition=`{item['partition']}` · view=`{item['view']}`"
        )
        return (
            0,
            meta,
            "",
            None,
            None,
            None,
            {},
            None,
            {},
            "",
            *("" for _ in ANNOTATION_FORM_FIELDS),
            {},
            "",
            None,
            "",
            _status_text(item, message),
        )
    meta = (
        f"### {item['ordinal'] + 1}/{item['total']} · `{item['record_id']}`\n"
        f"source=`{item['source']}` · split=`{item['split']}` · "
        f"target=`{item['target_status']}` · partition=`{item['partition']}` · "
        f"global_ordinal=`{item['global_ordinal']}` · "
        f"verified=`{str(item['verified']).lower()}`"
    )
    editor_warning = ""
    try:
        parsed_editor = parse_region_model_output(item["editor_text"]).to_dict()
    except ContractError:
        # 历史工作快照允许暂存不完整 JSON；混合表单必须从严格模板恢复，但高级区仍
        # 保留原始文本，便于专家找回未完成编辑。
        parsed_editor = default_region_description(item["target_status"])
        editor_warning = (
            "⚠️ 已保存的高级 JSON 尚未通过严格解析；表单暂以空模板恢复，"
            "原始文本保留在高级编辑区。"
        )
    form_values = description_to_form_values(parsed_editor)
    canonical_preview = parse_region_model_output(parsed_editor).to_dict()
    advanced_text = (
        item["editor_text"]
        if editor_warning
        else _canonical_editor_text(canonical_preview)
    )
    crop_warning = item.get("crop_warning")
    crop_warning_text = (
        f"⚠️ {crop_warning['message']}\n\n"
        f"crop={crop_warning['width_pixels']}×{crop_warning['height_pixels']} px；"
        f"reasons={crop_warning['reasons']}"
        if isinstance(crop_warning, Mapping)
        else "局部裁剪尺寸未触发窄幅或极端长宽比提示。"
    )
    draft_quality = item.get("draft_quality") or {
        "status": "not_evaluated",
        "issues": ["model_draft_invalid_or_missing"],
        "metrics": {},
    }
    return (
        item["ordinal"],
        meta,
        item["record_id"],
        item["optical_full"],
        item["binary_mask"],
        item["context_crop"],
        item["program_facts"],
        item["model_draft"],
        draft_quality,
        crop_warning_text,
        *form_values,
        canonical_preview,
        advanced_text,
        item["audit_overlay"],
        editor_warning or "表单与 canonical JSON 已同步。",
        _status_text(item, message),
    )


def apply_annotation_action(
    *,
    project_root: Path | str,
    action: str,
    ordinal: int,
    record_id: str,
    editor_text: str,
    partition: str,
    view: str,
) -> tuple[Any, ...]:
    """执行 UI 的保存或最终核验动作，便于在不启动浏览器时做永久回归测试。"""

    root = Path(project_root).resolve()
    current = annotation_view_item(
        root,
        int(ordinal),
        partition=partition,
        view=view,
    )
    if current.get("empty") or current["record_id"] != record_id:
        fail("ANNOTATION_INVALID", "工作台 record_id 与当前视图不一致，请刷新后重试")
    if action == "save":
        save_annotation_work(
            project_root=root,
            record_id=record_id,
            editor_text=editor_text,
        )
        message = "已原子保存未核验编辑。"
    elif action == "verify":
        verify_annotation_work(
            project_root=root,
            record_id=record_id,
            editor_text=editor_text,
        )
        message = "严格验证通过，已标记 expert_verified。"
    else:
        fail("ANNOTATION_INVALID", f"未知 annotation action：{action}")
    next_ordinal = int(ordinal)
    if action == "verify" and view == "all":
        next_ordinal += 1
    # pending 中当前记录核验后会自动消失，同一视图序号自然指向下一条。
    return _item_values(
        annotation_view_item(
            root,
            next_ordinal,
            partition=partition,
            view=view,
        ),
        message,
    )


def apply_form_annotation_action(
    *,
    project_root: Path | str,
    action: str,
    ordinal: int,
    record_id: str,
    form_values: Sequence[Any],
    partition: str,
    view: str,
) -> tuple[Any, ...]:
    """从当前分组表单生成 canonical JSON 后执行保存或核验。"""

    description = form_values_to_description(form_values)
    current = annotation_view_item(
        project_root,
        int(ordinal),
        partition=partition,
        view=view,
    )
    if (
        current.get("empty")
        or current["record_id"] != record_id
        or description["target_status"] != current["target_status"]
    ):
        fail(
            "ANNOTATION_INVALID",
            "专家表单 target_status/record_id 与当前工作项不一致",
        )
    return apply_annotation_action(
        project_root=project_root,
        action=action,
        ordinal=ordinal,
        record_id=record_id,
        editor_text=_canonical_editor_text(description),
        partition=partition,
        view=view,
    )


def preview_form_values(
    values: Sequence[Any],
) -> tuple[dict[str, Any], str, str]:
    """实时生成只读 canonical 预览；错误只显示诊断，不保存或宽松修复。"""

    try:
        description = form_values_to_description(values)
    except (ContractError, LandslideEvidenceError) as error:
        return (
            {
                "schema_valid": False,
                "error_type": type(error).__name__,
                "error": str(error),
            },
            "",
            "⚠️ 当前表单尚未通过严格合同；保存和核验会被拒绝。",
        )
    return (
        description,
        _canonical_editor_text(description),
        "✅ 当前表单已通过严格 JSON、enum、重复项和禁区结论检查。",
    )


def create_annotation_app(
    *,
    project_root: Path | str,
    partition: str,
    view: str,
) -> Any:
    """构建字段表单、canonical 预览和高级 JSON 三层混合核验界面。"""

    import gradio as gr

    root = Path(project_root).resolve()
    partition, view = _view_options(partition, view)
    load_annotation_project(root)
    initial = annotation_view_item(root, 0, partition=partition, view=view)

    def component_updates(values: tuple[Any, ...]) -> tuple[Any, ...]:
        """按当前 target status 锁定 no-target 的区域描述控件。"""

        result = list(values)
        has_record = bool(result[2])
        target_status = str(result[_FORM_START])
        flags = form_interactive_flags(target_status, has_record=has_record)
        for position, interactive in enumerate(flags):
            result[_FORM_START + position] = gr.update(
                value=result[_FORM_START + position],
                interactive=interactive,
            )
        return tuple(result)

    def render(values: tuple[Any, ...]) -> tuple[Any, ...]:
        return component_updates(values)

    def navigate(index: int, delta: int) -> tuple[Any, ...]:
        return render(_item_values(
            annotation_view_item(
                root,
                int(index) + delta,
                partition=partition,
                view=view,
            )
        ))

    def save(index: int, record_id: str, *values: Any) -> tuple[Any, ...]:
        return render(apply_form_annotation_action(
            project_root=root,
            action="save",
            ordinal=int(index),
            record_id=record_id,
            form_values=values,
            partition=partition,
            view=view,
        ))

    def verify(index: int, record_id: str, *values: Any) -> tuple[Any, ...]:
        return render(apply_form_annotation_action(
            project_root=root,
            action="verify",
            ordinal=int(index),
            record_id=record_id,
            form_values=values,
            partition=partition,
            view=view,
        ))

    def synchronize_form(*values_and_editor: Any) -> tuple[Any, Any, str]:
        values = values_and_editor[:-1]
        current_editor = values_and_editor[-1]
        preview, editor_text, message = preview_form_values(values)
        # 专家输入一个尚未完整的字段时，高级区保留上一份合法 JSON，避免覆盖可恢复内容。
        return preview, editor_text or current_editor, message

    def synchronize_advanced(
        editor_text: str,
        target_status: str,
    ) -> tuple[Any, ...]:
        values, preview, normalized = apply_advanced_json(
            editor_text,
            expected_target_status=target_status,
        )
        updates: list[Any] = []
        flags = form_interactive_flags(target_status)
        for value, interactive in zip(values, flags, strict=True):
            updates.append(gr.update(value=value, interactive=interactive))
        return (
            *updates,
            preview,
            normalized,
            "✅ 高级 JSON 已严格解析并同步到分组表单。",
        )

    with gr.Blocks(title="Stage 4 单专家 Mask-Grounded Annotation") as app:
        gr.Markdown(
            "# Stage 4 单专家核验工作台\n"
            "模型输出只是草稿。请依据未修改 RGB、独立二值 mask 和 clean crop 修正；"
            "禁止写入发生时间、触发原因、稳定性、风险、精确运动或现场地质断言。"
        )
        index_state = gr.State(initial["ordinal"])
        meta = gr.Markdown()
        record_id = gr.Textbox(label="record_id", interactive=False)
        with gr.Row():
            optical = gr.Image(
                label="Original Full RGB",
                type="filepath",
                image_mode="RGB",
                interactive=False,
            )
            mask = gr.Image(
                label="Binary Mask (white=region)",
                type="filepath",
                image_mode="L",
                interactive=False,
            )
            crop = gr.Image(
                label="Clean Context Crop",
                type="filepath",
                image_mode="RGB",
                interactive=False,
            )
        crop_warning = gr.Markdown()
        with gr.Row():
            facts = gr.JSON(label="程序事实（只读）", open=False)
            model_draft = gr.JSON(label="模型草稿与解析状态（只读）", open=False)
            draft_quality = gr.JSON(label="草稿信息量诊断（实时重算，只读）", open=False)

        gr.Markdown(
            "## 专家最终结构化描述\n"
            "先核对总体场景，再核对目标外观/形态、周围环境和区域—环境关系。"
            "数组字段每行填写一项；空行忽略，重复项会被严格拒绝。"
        )
        target_status = gr.Textbox(label="target_status（程序事实，只读）", interactive=False)
        with gr.Accordion("1. 目标外观 target_appearance", open=True):
            with gr.Row():
                tone = gr.Textbox(label="tone")
                texture = gr.Textbox(label="texture")
                vegetation = gr.Textbox(label="vegetation_or_exposure")
            with gr.Row():
                homogeneity = gr.Textbox(label="homogeneity")
                boundary_visibility = gr.Textbox(label="boundary_visibility")
        with gr.Accordion("2. 目标形态 target_morphology", open=True):
            with gr.Row():
                shape = gr.Textbox(label="shape")
                fragmentation = gr.Textbox(label="fragmentation")
                orientation = gr.Textbox(label="qualitative_orientation")
        with gr.Accordion("3. 周围环境 surrounding_environment", open=True):
            with gr.Row():
                land_cover = gr.Textbox(label="land_cover（每行一项）", lines=4)
                nearby_objects = gr.Textbox(label="nearby_objects（每行一项）", lines=4)
            with gr.Row():
                terrain_context = gr.Textbox(
                    label="visible_terrain_context（每行一项）",
                    lines=4,
                )
                human_disturbance = gr.Textbox(
                    label="human_disturbance（每行一项）",
                    lines=4,
                )
        with gr.Accordion("4. 区域—环境关系 region_context_contrast", open=True):
            with gr.Row():
                tone_contrast = gr.Textbox(label="tone_contrast")
                texture_contrast = gr.Textbox(label="texture_contrast")
            with gr.Row():
                vegetation_contrast = gr.Textbox(label="vegetation_contrast")
                boundary_transition = gr.Textbox(label="boundary_transition")
            adjacency = gr.Textbox(label="adjacency（每行一项）", lines=4)
        with gr.Accordion("5. 混淆项、充分性、摘要与限制", open=True):
            confusers = gr.Textbox(label="possible_confusers（每行一项）", lines=4)
            sufficiency = gr.Dropdown(
                choices=[item.value for item in RegionEvidenceSufficiency],
                label="evidence_sufficiency",
                allow_custom_value=False,
            )
            summary = gr.Textbox(label="short_summary", lines=3)
            limitations = gr.Textbox(label="limitations（每行一项）", lines=5)

        form_components = [
            target_status,
            tone,
            texture,
            vegetation,
            homogeneity,
            boundary_visibility,
            shape,
            fragmentation,
            orientation,
            land_cover,
            nearby_objects,
            terrain_context,
            human_disturbance,
            tone_contrast,
            texture_contrast,
            vegetation_contrast,
            boundary_transition,
            adjacency,
            confusers,
            sufficiency,
            summary,
            limitations,
        ]
        canonical_preview = gr.JSON(
            label="专家最终 canonical JSON（只读，始终可见）",
            open=True,
        )
        form_status = gr.Markdown()
        with gr.Accordion("高级 JSON 编辑（可选）", open=False):
            advanced_editor = gr.Textbox(
                label="高级 JSON；修改后点击“应用 JSON 到表单”",
                lines=28,
                max_lines=64,
                interactive=True,
            )
            apply_json_button = gr.Button("应用 JSON 到表单")
        with gr.Accordion("Audit-only overlay（不得据此判断颜色/纹理）", open=False):
            overlay = gr.Image(label="Audit Overlay", type="filepath", image_mode="RGB", interactive=False)
        status = gr.Markdown()
        with gr.Row():
            previous = gr.Button("上一条")
            save_button = gr.Button("保存未核验编辑")
            verify_button = gr.Button("核验完成", variant="primary")
            next_button = gr.Button("下一条")

        outputs = [
            index_state, meta, record_id, optical, mask, crop, facts,
            model_draft, draft_quality, crop_warning,
            *form_components,
            canonical_preview, advanced_editor, overlay, form_status, status,
        ]
        app.load(fn=lambda: render(_item_values(initial)), outputs=outputs)
        previous.click(fn=lambda index: navigate(index, -1), inputs=index_state, outputs=outputs)
        next_button.click(fn=lambda index: navigate(index, 1), inputs=index_state, outputs=outputs)
        for component in form_components[1:]:
            component.input(
                fn=synchronize_form,
                inputs=[*form_components, advanced_editor],
                outputs=[canonical_preview, advanced_editor, form_status],
                show_progress="hidden",
            )
        apply_json_button.click(
            fn=synchronize_advanced,
            inputs=[advanced_editor, target_status],
            outputs=[
                *form_components,
                canonical_preview,
                advanced_editor,
                form_status,
            ],
        )
        save_button.click(
            fn=save,
            inputs=[index_state, record_id, *form_components],
            outputs=outputs,
        )
        verify_button.click(
            fn=verify,
            inputs=[index_state, record_id, *form_components],
            outputs=outputs,
        )
    return app


def serve_annotation_workbench(
    *,
    project_root: Path | str,
    partition: str,
    view: str = "pending",
    port: int = 7860,
    auto_close_partition: bool = False,
    poll_interval_seconds: float = 1.0,
) -> None:
    """只绑定回环地址；一键模式在当前分区全部核验后自动关闭服务。"""

    if isinstance(port, bool) or not isinstance(port, int) or not 1024 <= port <= 65535:
        fail("ANNOTATION_INVALID", "port 必须位于 [1024,65535]")
    if not isinstance(auto_close_partition, bool):
        fail("ANNOTATION_INVALID", "auto_close_partition 必须是布尔值")
    if (
        isinstance(poll_interval_seconds, bool)
        or not isinstance(poll_interval_seconds, (int, float))
        or not math.isfinite(float(poll_interval_seconds))
        or not 0.1 <= float(poll_interval_seconds) <= 10.0
    ):
        fail("ANNOTATION_INVALID", "poll_interval_seconds 必须位于 [0.1,10.0]")
    root = Path(project_root).resolve()
    project, _ = load_annotation_project(root)
    asset_root = Path(project["asset_root"]).resolve()
    app = create_annotation_app(
        project_root=root,
        partition=partition,
        view=view,
    )
    launch_options = {
        "server_name": "127.0.0.1",
        "server_port": port,
        "share": False,
        "inbrowser": False,
        "show_error": True,
        "allowed_paths": [str(asset_root)],
        "blocked_paths": [str(Path.cwd() / "models_zoo")],
    }
    # Gradio 会通过 httpx 回访 startup-events；精确 bypass 可避免 localhost 被代理为 503。
    ensure_loopback_proxy_bypass()
    if not auto_close_partition:
        try:
            app.launch(**launch_options)
        except Exception as error:
            _close_after_launch_failure(app, error)
        return
    try:
        app.launch(prevent_thread_lock=True, **launch_options)
    except Exception as error:
        _close_after_launch_failure(app, error)
    try:
        while not annotation_partition_status(root, partition=partition)["complete"]:
            time.sleep(float(poll_interval_seconds))
    finally:
        # Gradio 的非阻塞 launch 必须显式 close，避免 calibration 完成后残留端口。
        app.close()
