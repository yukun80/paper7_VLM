"""Stage 4 train-only 单专家标注的一键、可恢复状态机。"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Callable, Mapping

from oa_groundrag.phase3.common import (
    atomic_write_json,
    atomic_write_text,
    read_json,
    sha256_file,
)

from .contracts import fail
from .single_expert import (
    AnnotationIntendedUse,
    CALIBRATION_COUNT,
    TRAIN_ANNOTATION_COUNT,
    _exact,
    _mapping,
    _ordinary_file,
    create_annotation_project,
    load_annotation_project,
)
from .single_expert_drafting import generate_annotation_drafts
from .single_expert_package import (
    export_verified_annotations,
    validate_verified_annotation_package,
)
from .single_expert_training import (
    export_training_messages,
    load_training_message_artifact,
)
from .single_expert_workbench import (
    annotation_partition_draft_quality,
    annotation_partition_status,
    serve_annotation_workbench,
)


TRAIN_WORKFLOW_SCHEMA = "oa_groundrag.mask_grounded_region.train_workflow.v1"
TRAIN_WORKFLOW_STATE_SCHEMA = (
    "oa_groundrag.mask_grounded_region.train_workflow_state.v1"
)
TRAIN_WORKFLOW_PHASES = (
    "calibration",
    "awaiting_prompt_confirmation",
    "remaining",
    "complete",
)


@dataclass(frozen=True)
class TrainWorkflowPaths:
    """一键流程的冻结输入与输出位置；测试可注入隔离的临时根。"""

    corpus_root: Path
    project_root: Path
    annotation_package_root: Path
    training_messages_root: Path
    prompt_path: Path
    draft_config_path: Path

    def absolute(self) -> "TrainWorkflowPaths":
        """统一为词法绝对路径，不通过 resolve 接受 symlink 别名。"""

        return TrainWorkflowPaths(*(
            Path(os.path.abspath(path))
            for path in (
                self.corpus_root,
                self.project_root,
                self.annotation_package_root,
                self.training_messages_root,
                self.prompt_path,
                self.draft_config_path,
            )
        ))


def _result(stage: str, paths: TrainWorkflowPaths, **details: Any) -> dict[str, Any]:
    return {
        "schema_version": TRAIN_WORKFLOW_SCHEMA,
        "ok": True,
        "stage": stage,
        "project_root": str(paths.project_root),
        "annotation_package_root": str(paths.annotation_package_root),
        "training_messages_root": str(paths.training_messages_root),
        "annotator": "expert",
        "formal_acceptance": False,
        **details,
    }


def _workflow_state_path(project_root: Path) -> Path:
    return project_root / "workflow_state.json"


def _write_workflow_phase(project_root: Path, phase: str) -> dict[str, Any]:
    if phase not in TRAIN_WORKFLOW_PHASES:
        fail("ANNOTATION_INVALID", f"未知 train workflow phase：{phase}")
    value = {
        "schema_version": TRAIN_WORKFLOW_STATE_SCHEMA,
        "phase": phase,
        "formal_acceptance": False,
    }
    atomic_write_json(_workflow_state_path(project_root), value)
    return value


def _load_or_initialize_workflow_state(project_root: Path) -> dict[str, Any]:
    """持久化 calibration 人工边界，避免异常退出后一次重跑意外越过该边界。"""

    path = _workflow_state_path(project_root)
    if not path.exists() and not path.is_symlink():
        project, _ = load_annotation_project(project_root)
        phase = (
            "remaining"
            if project["frozen_prompt_sha256"] is not None
            or project["frozen_draft_config_sha256"] is not None
            else "calibration"
        )
        return _write_workflow_phase(project_root, phase)
    _ordinary_file(path, location="workflow_state")
    value = _mapping(read_json(path), "workflow_state")
    _exact(
        value,
        ("schema_version", "phase", "formal_acceptance"),
        "workflow_state",
    )
    if (
        value["schema_version"] != TRAIN_WORKFLOW_STATE_SCHEMA
        or value["phase"] not in TRAIN_WORKFLOW_PHASES
        or value["formal_acceptance"] is not False
    ):
        fail("ANNOTATION_INVALID", "train workflow state 非法")
    return value


def _validate_project_binding(paths: TrainWorkflowPaths) -> None:
    project, assignments = load_annotation_project(paths.project_root)
    prompt_frozen = project["frozen_prompt_sha256"] is not None
    config_frozen = project["frozen_draft_config_sha256"] is not None
    if (
        project["asset_root"] != str(paths.corpus_root)
        or project["split"] != "train"
        or project["intended_use"]
        != AnnotationIntendedUse.TRAIN_SUPERVISION.value
        or len(assignments) != TRAIN_ANNOTATION_COUNT
        or prompt_frozen != config_frozen
    ):
        fail("ANNOTATION_INVALID", "一键工作根未绑定冻结的 500 条 train Corpus")


def _reuse_or_publish_training_assets(
    paths: TrainWorkflowPaths,
    *,
    progress: Callable[[str], None],
) -> dict[str, Any]:
    """发布中断可从合法 package 继续；任何非法既有根都直接失败。"""

    package_exists = paths.annotation_package_root.exists() or paths.annotation_package_root.is_symlink()
    messages_exists = paths.training_messages_root.exists() or paths.training_messages_root.is_symlink()
    if messages_exists and not package_exists:
        fail("OUTPUT_EXISTS", "training messages 已存在但 annotation package 缺失，拒绝覆盖")
    if package_exists:
        progress("annotation_package_validating")
        package = validate_verified_annotation_package(
            asset_root=paths.corpus_root,
            package_root=paths.annotation_package_root,
        )
        if package.manifest["training_eligible"] is not True:
            fail("SPLIT_FORBIDDEN", "既有 annotation package 不可用于 train supervision")
        progress("annotation_package_reused")
    else:
        progress("annotation_package_publishing")
        export_verified_annotations(
            project_root=paths.project_root,
            output_root=paths.annotation_package_root,
        )
        validate_verified_annotation_package(
            asset_root=paths.corpus_root,
            package_root=paths.annotation_package_root,
        )
        progress("annotation_package_published")
    if messages_exists:
        progress("training_messages_validating")
        artifact = load_training_message_artifact(paths.training_messages_root)
        progress("training_messages_reused")
    else:
        progress("training_messages_publishing")
        export_training_messages(
            asset_root=paths.corpus_root,
            annotations_root=paths.annotation_package_root,
            output_root=paths.training_messages_root,
        )
        artifact = load_training_message_artifact(paths.training_messages_root)
        progress("training_messages_published")
    return _result(
        "complete",
        paths,
        annotation_count=TRAIN_ANNOTATION_COUNT,
        training_message_count=len(artifact.rows),
        training_eligible=True,
    )


def run_train_annotation_workflow(
    *,
    paths: TrainWorkflowPaths,
    port: int = 7860,
    runtime: Any | None = None,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
    serve_callback: Callable[..., None] = serve_annotation_workbench,
) -> dict[str, Any]:
    """推进到下一个人工边界；模型单次加载后仍严格逐记录生成并原子保存。"""

    frozen = paths.absolute()

    def emit(event: str, **details: Any) -> None:
        if progress_callback is not None:
            progress_callback({
                "schema_version": TRAIN_WORKFLOW_SCHEMA,
                "event": event,
                **details,
            })

    def draft_progress(value: Mapping[str, Any]) -> None:
        payload = dict(value)
        draft_event = str(payload.pop("event", "progress"))
        emit(f"draft_{draft_event}", **payload)

    package_exists = (
        frozen.annotation_package_root.exists()
        or frozen.annotation_package_root.is_symlink()
    )
    messages_exists = (
        frozen.training_messages_root.exists()
        or frozen.training_messages_root.is_symlink()
    )
    project_exists = frozen.project_root.exists() or frozen.project_root.is_symlink()
    if package_exists or messages_exists:
        emit("published_assets_detected")
        if project_exists:
            _validate_project_binding(frozen)
            _load_or_initialize_workflow_state(frozen.project_root)
        result = _reuse_or_publish_training_assets(
            frozen,
            progress=lambda event: emit(event),
        )
        if project_exists:
            _write_workflow_phase(frozen.project_root, "complete")
        return result

    if not project_exists:
        emit("project_creating")
        create_annotation_project(
            asset_root=frozen.corpus_root,
            output_root=frozen.project_root,
            intended_use=AnnotationIntendedUse.TRAIN_SUPERVISION.value,
            prompt_path=frozen.prompt_path,
        )
        emit("project_created")
    _validate_project_binding(frozen)
    workflow_state = _load_or_initialize_workflow_state(frozen.project_root)

    calibration = annotation_partition_status(
        frozen.project_root,
        partition="calibration",
    )
    if workflow_state["phase"] == "calibration":
        if not calibration["complete"] and calibration["drafted"] < CALIBRATION_COUNT:
            emit(
                "calibration_generation_starting",
                drafted=calibration["drafted"],
                total=CALIBRATION_COUNT,
            )
            generate_annotation_drafts(
                project_root=frozen.project_root,
                config_path=frozen.draft_config_path,
                partition="calibration",
                runtime=runtime,
                progress_callback=draft_progress,
            )
        if not calibration["complete"]:
            calibration = annotation_partition_status(
                frozen.project_root,
                partition="calibration",
            )
            if calibration["drafted"] != CALIBRATION_COUNT:
                fail("ANNOTATION_INVALID", "calibration 草稿未完整落盘，不得启动核验 UI")
            quality = annotation_partition_draft_quality(
                frozen.project_root,
                partition="calibration",
            )
            emit("calibration_quality_assessed", **quality)
            if quality["target_template_copies"]:
                fail(
                    "DRAFT_QUALITY_FAILED",
                    "calibration target 草稿复制空模板，拒绝启动核验 UI",
                    details=quality,
                )
            emit("calibration_ui_starting", port=port)
            serve_callback(
                project_root=frozen.project_root,
                partition="calibration",
                view="pending",
                port=port,
                auto_close_partition=True,
            )
        calibration = annotation_partition_status(
            frozen.project_root,
            partition="calibration",
        )
        if not calibration["complete"]:
            fail("ANNOTATION_INVALID", "calibration UI 已结束但 20 条尚未全部核验")
        _write_workflow_phase(
            frozen.project_root,
            "awaiting_prompt_confirmation",
        )
        emit("calibration_complete")
        # 此处是有意的人工作业边界：负责人可调整 prompt/config，下一次同命令才冻结。
        return _result(
            "calibration_complete_prompt_adjustment_required",
            frozen,
            calibration_verified=CALIBRATION_COUNT,
            next_action="rerun_same_command_after_prompt_review",
        )
    if workflow_state["phase"] == "awaiting_prompt_confirmation":
        if not calibration["complete"]:
            fail("ANNOTATION_INVALID", "calibration 状态回退，拒绝确认 remaining")
        project, _ = load_annotation_project(frozen.project_root)
        if (
            project["frozen_prompt_sha256"] is not None
            or project["frozen_draft_config_sha256"] is not None
        ):
            fail(
                "ANNOTATION_INVALID",
                "awaiting_prompt_confirmation 阶段不应已冻结 prompt/config",
            )
        # calibration 始终使用建项时的工作副本；只有负责人在
        # 20 条核验后重新执行一键命令，才将仓库 prompt 同步为
        # remaining 的候选冻结版，避免 calibration 中途漂移。
        _ordinary_file(frozen.prompt_path, location="workflow.prompt_path")
        prompt_text = frozen.prompt_path.read_text(encoding="utf-8")
        if not prompt_text.strip():
            fail("ANNOTATION_INVALID", "remaining 确认后的 repo prompt 不能为空")
        atomic_write_text(frozen.project_root / "prompt.txt", prompt_text)
        prompt_sha256 = sha256_file(frozen.project_root / "prompt.txt")
        emit(
            "prompt_synchronized",
            prompt_sha256=prompt_sha256,
        )
        _write_workflow_phase(frozen.project_root, "remaining")
        emit(
            "prompt_confirmation_recorded",
            prompt_sha256=prompt_sha256,
        )
    elif workflow_state["phase"] != "remaining":
        fail(
            "ANNOTATION_INVALID",
            "workflow phase=complete 但正式发布根缺失，拒绝隐式重建",
        )
    if not calibration["complete"]:
        fail("ANNOTATION_INVALID", "20 条 calibration 未完成，不得进入 remaining")

    remaining = annotation_partition_status(
        frozen.project_root,
        partition="remaining",
    )
    if remaining["drafted"] < remaining["total"]:
        emit(
            "remaining_generation_starting",
            drafted=remaining["drafted"],
            total=remaining["total"],
        )
        generate_annotation_drafts(
            project_root=frozen.project_root,
            config_path=frozen.draft_config_path,
            partition="remaining",
            runtime=runtime,
            progress_callback=draft_progress,
        )
    remaining = annotation_partition_status(
        frozen.project_root,
        partition="remaining",
    )
    if remaining["drafted"] != remaining["total"]:
        fail("ANNOTATION_INVALID", "remaining 草稿未完整落盘，不得启动核验 UI")
    if not remaining["complete"]:
        emit("remaining_ui_starting", port=port)
        serve_callback(
            project_root=frozen.project_root,
            partition="remaining",
            view="pending",
            port=port,
            auto_close_partition=True,
        )
        remaining = annotation_partition_status(
            frozen.project_root,
            partition="remaining",
        )
        if not remaining["complete"]:
            fail("ANNOTATION_INVALID", "remaining UI 已结束但 480 条尚未全部核验")
        emit("remaining_complete")
    result = _reuse_or_publish_training_assets(
        frozen,
        progress=lambda event: emit(event),
    )
    _write_workflow_phase(frozen.project_root, "complete")
    return result
