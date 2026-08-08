"""Stage 4 v2 模型辅助 train supervision 的固定路径、可恢复工作流。"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Callable, Mapping

from .contracts import fail


MODEL_ASSISTED_WORKFLOW_SCHEMA = (
    "oa_groundrag.mask_grounded_region.model_assisted_train_workflow.v2"
)
MODEL_ASSISTED_RECORD_COUNT = 8_450
EXTENSION_RECORD_COUNT = 7_950


@dataclass(frozen=True)
class ModelAssistedWorkflowPaths:
    """模型辅助流程的全部冻结输入与输出位置。"""

    extension_config_path: Path
    extension_root: Path
    collection_root: Path
    legacy_project_root: Path
    project_root: Path
    annotation_package_root: Path
    training_messages_root: Path
    prompt_path: Path
    draft_config_path: Path

    def absolute(self) -> "ModelAssistedWorkflowPaths":
        """转换为词法绝对路径，不通过 ``resolve`` 接受 symlink 别名。"""

        return ModelAssistedWorkflowPaths(*(
            Path(os.path.abspath(path))
            for path in (
                self.extension_config_path,
                self.extension_root,
                self.collection_root,
                self.legacy_project_root,
                self.project_root,
                self.annotation_package_root,
                self.training_messages_root,
                self.prompt_path,
                self.draft_config_path,
            )
        ))


def _exists(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _field(value: Any, name: str, *, location: str) -> Any:
    if isinstance(value, Mapping):
        if name in value:
            return value[name]
    elif hasattr(value, name):
        return getattr(value, name)
    fail("ANNOTATION_INVALID", f"{location} 缺少 {name}")


def _integer_field(value: Any, name: str, *, location: str) -> int:
    child = _field(value, name, location=location)
    if isinstance(child, bool) or not isinstance(child, int) or child < 0:
        fail("ANNOTATION_INVALID", f"{location}.{name} 必须是非负整数")
    return child


def _result(
    stage: str,
    paths: ModelAssistedWorkflowPaths,
    **details: Any,
) -> dict[str, Any]:
    return {
        "schema_version": MODEL_ASSISTED_WORKFLOW_SCHEMA,
        "ok": True,
        "stage": stage,
        "extension_root": str(paths.extension_root),
        "collection_root": str(paths.collection_root),
        "project_root": str(paths.project_root),
        "annotation_package_root": str(paths.annotation_package_root),
        "training_messages_root": str(paths.training_messages_root),
        "source_record_count": MODEL_ASSISTED_RECORD_COUNT,
        "reference_authority": "mixed_model_and_single_expert",
        "formal_acceptance": False,
        **details,
    }


def _emit(
    callback: Callable[[Mapping[str, Any]], None] | None,
    event: str,
    **details: Any,
) -> None:
    if callback is not None:
        callback({
            "schema_version": MODEL_ASSISTED_WORKFLOW_SCHEMA,
            "event": event,
            **details,
        })


def prepare_expanded_corpus(
    *,
    paths: ModelAssistedWorkflowPaths,
    verify_source: bool = True,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """准备或复用固定 v2 扩展 Corpus，并严格重验 8,450 条集合。"""

    from .expanded_region import (
        prepare_expanded_region_assets,
        validate_expanded_region_collection,
        validate_region_extension,
    )

    frozen = paths.absolute()
    _emit(progress_callback, "expanded_assets_preparing")
    prepared = prepare_expanded_region_assets(
        frozen.extension_config_path,
        verify_source=verify_source,
    )
    prepared_extension = Path(str(_field(
        prepared,
        "extension_root",
        location="prepare_expanded_region_assets result",
    )))
    prepared_collection = Path(str(_field(
        prepared,
        "collection_root",
        location="prepare_expanded_region_assets result",
    )))
    if (
        Path(os.path.abspath(prepared_extension)) != frozen.extension_root
        or Path(os.path.abspath(prepared_collection)) != frozen.collection_root
    ):
        fail("FACT_MISMATCH", "expanded Region 产物未写入冻结 v2 根")
    if (
        _integer_field(
            prepared,
            "extension_record_count",
            location="prepare_expanded_region_assets result",
        )
        != EXTENSION_RECORD_COUNT
        or _integer_field(
            prepared,
            "collection_record_count",
            location="prepare_expanded_region_assets result",
        )
        != MODEL_ASSISTED_RECORD_COUNT
    ):
        fail("ANNOTATION_INVALID", "expanded Region 数量不是 7,950 / 8,450")
    _emit(progress_callback, "extension_validating")
    extension_report = validate_region_extension(
        frozen.extension_root,
        config_path=frozen.extension_config_path,
        verify_source=verify_source,
    )
    _emit(progress_callback, "collection_validating")
    collection_report = validate_expanded_region_collection(
        frozen.collection_root,
        verify_members=True,
        verify_source=False,
    )
    _emit(progress_callback, "expanded_assets_ready")
    return _result(
        "expanded_corpus_ready",
        frozen,
        extension_record_count=EXTENSION_RECORD_COUNT,
        collection_record_count=MODEL_ASSISTED_RECORD_COUNT,
        extension_manifest_sha256=_field(
            prepared,
            "extension_manifest_sha256",
            location="prepare_expanded_region_assets result",
        ),
        collection_manifest_sha256=_field(
            prepared,
            "collection_manifest_sha256",
            location="prepare_expanded_region_assets result",
        ),
        extension_validation=dict(extension_report),
        collection_validation=dict(collection_report),
    )


def _project_status(status: Any) -> tuple[int, int]:
    total = _integer_field(status, "total", location="model-assisted project.status")
    drafted = _integer_field(
        status,
        "drafted",
        location="model-assisted project.status",
    )
    valid_drafts = _integer_field(
        status,
        "valid_drafts",
        location="model-assisted project.status",
    )
    invalid_drafts = _integer_field(
        status,
        "invalid_drafts",
        location="model-assisted project.status",
    )
    pending = _integer_field(
        status,
        "pending",
        location="model-assisted project.status",
    )
    complete = _field(
        status,
        "complete",
        location="model-assisted project.status",
    )
    formal_acceptance = _field(
        status,
        "formal_acceptance",
        location="model-assisted project.status",
    )
    if (
        total != MODEL_ASSISTED_RECORD_COUNT
        or drafted > total
        or drafted != valid_drafts + invalid_drafts
        or pending != total - drafted
        or not isinstance(complete, bool)
        or complete is not (drafted == total)
        or formal_acceptance is not False
    ):
        fail("ANNOTATION_INVALID", "model-assisted project 数量或状态非法")
    return total, drafted


def _package_counts(package: Any) -> dict[str, Any]:
    manifest = _field(
        package,
        "manifest",
        location="model-assisted supervision package",
    )
    record_count = _integer_field(
        manifest,
        "record_count",
        location="supervision package.manifest",
    )
    eligible_count = _integer_field(
        manifest,
        "eligible_count",
        location="supervision package.manifest",
    )
    excluded_count = _integer_field(
        manifest,
        "excluded_count",
        location="supervision package.manifest",
    )
    authority_counts = _field(
        manifest,
        "authority_counts",
        location="supervision package.manifest",
    )
    exclusion_counts = _field(
        manifest,
        "exclusion_counts",
        location="supervision package.manifest",
    )
    if not isinstance(authority_counts, Mapping) or not isinstance(
        exclusion_counts,
        Mapping,
    ):
        fail("ANNOTATION_INVALID", "supervision package 动态计数必须是对象")
    expert_count = _integer_field(
        authority_counts,
        "expert_verified",
        location="supervision package.manifest.authority_counts",
    )
    model_count = _integer_field(
        authority_counts,
        "model_generated_unreviewed",
        location="supervision package.manifest.authority_counts",
    )
    reason_counts: dict[str, int] = {}
    for reason, value in exclusion_counts.items():
        if (
            not isinstance(reason, str)
            or not reason
            or isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
        ):
            fail("ANNOTATION_INVALID", "supervision package exclusion_counts 非法")
        reason_counts[reason] = value
    if (
        record_count != MODEL_ASSISTED_RECORD_COUNT
        or eligible_count + excluded_count != record_count
        or expert_count + model_count != eligible_count
        or sum(reason_counts.values()) != excluded_count
        or _field(
            manifest,
            "reference_authority",
            location="supervision package.manifest",
        )
        != "mixed_model_and_single_expert"
    ):
        fail("ANNOTATION_INVALID", "supervision package 动态计数或 authority 非法")
    return {
        "eligible_count": eligible_count,
        "expert_count": expert_count,
        "model_count": model_count,
        "excluded_count": excluded_count,
        "reason_counts": dict(sorted(reason_counts.items())),
    }


def _message_count(artifact: Any, *, expected: int) -> int:
    rows = _field(artifact, "rows", location="model-assisted training messages")
    try:
        count = len(rows)
    except TypeError:
        fail("ANNOTATION_INVALID", "model-assisted training messages rows 不可计数")
        raise AssertionError("unreachable")
    if count != expected:
        fail(
            "ANNOTATION_INVALID",
            "动态 training messages 数量必须等于 supervision eligible_count",
            details={"messages": count, "eligible_count": expected},
        )
    return count


def _reuse_or_publish_training_assets(
    paths: ModelAssistedWorkflowPaths,
    *,
    project_drafted: int | None,
    progress_callback: Callable[[Mapping[str, Any]], None] | None,
) -> dict[str, Any]:
    from .model_assisted import (
        export_model_assisted_supervision,
        export_model_assisted_training_messages,
        load_model_assisted_training_messages,
        validate_model_assisted_supervision,
    )

    package_exists = _exists(paths.annotation_package_root)
    messages_exists = _exists(paths.training_messages_root)
    if messages_exists and not package_exists:
        fail(
            "OUTPUT_EXISTS",
            "动态 training messages 已存在但 supervision package 缺失，拒绝覆盖",
        )
    if package_exists:
        _emit(progress_callback, "supervision_package_validating")
        package = validate_model_assisted_supervision(
            collection_root=paths.collection_root,
            package_root=paths.annotation_package_root,
        )
        _emit(progress_callback, "supervision_package_reused")
    else:
        if project_drafted != MODEL_ASSISTED_RECORD_COUNT:
            fail(
                "ANNOTATION_INVALID",
                "只有 8,450 条草稿全部严格落盘后才能发布 supervision package",
                details={"drafted": project_drafted},
            )
        _emit(progress_callback, "supervision_package_publishing")
        export_model_assisted_supervision(
            project_root=paths.project_root,
            output_root=paths.annotation_package_root,
        )
        package = validate_model_assisted_supervision(
            collection_root=paths.collection_root,
            package_root=paths.annotation_package_root,
        )
        _emit(progress_callback, "supervision_package_published")
    counts = _package_counts(package)
    if messages_exists:
        _emit(progress_callback, "training_messages_validating")
        artifact = load_model_assisted_training_messages(paths.training_messages_root)
        _emit(progress_callback, "training_messages_reused")
    else:
        _emit(progress_callback, "training_messages_publishing")
        export_model_assisted_training_messages(
            collection_root=paths.collection_root,
            package_root=paths.annotation_package_root,
            output_root=paths.training_messages_root,
        )
        artifact = load_model_assisted_training_messages(paths.training_messages_root)
        _emit(progress_callback, "training_messages_published")
    message_count = _message_count(artifact, expected=counts["eligible_count"])
    return _result(
        "complete",
        paths,
        draft_count=MODEL_ASSISTED_RECORD_COUNT,
        training_message_count=message_count,
        training_eligible=True,
        expert_count=counts["expert_count"],
        model_count=counts["model_count"],
        excluded_count=counts["excluded_count"],
        reason_counts=counts["reason_counts"],
    )


def run_model_assisted_train_workflow(
    *,
    paths: ModelAssistedWorkflowPaths,
    runtime: Any | None = None,
    verify_source: bool = True,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """准备、恢复并发布模型辅助监督；每次仅一次 runtime 生成所有缺失项。"""

    from .model_assisted import (
        create_model_assisted_project,
        generate_model_assisted_drafts,
        model_assisted_project_status,
    )

    frozen = paths.absolute()
    prepare_expanded_corpus(
        paths=frozen,
        verify_source=verify_source,
        progress_callback=progress_callback,
    )

    package_exists = _exists(frozen.annotation_package_root)
    messages_exists = _exists(frozen.training_messages_root)
    project_exists = _exists(frozen.project_root)
    if messages_exists and not package_exists:
        fail(
            "OUTPUT_EXISTS",
            "动态 training messages 已存在但 supervision package 缺失，拒绝覆盖",
        )

    project_drafted: int | None = None
    if project_exists:
        _emit(progress_callback, "project_validating")
        status = model_assisted_project_status(frozen.project_root)
        _, project_drafted = _project_status(status)
        _emit(
            progress_callback,
            "project_reused",
            drafted=project_drafted,
            total=MODEL_ASSISTED_RECORD_COUNT,
        )

    if package_exists:
        return _reuse_or_publish_training_assets(
            frozen,
            project_drafted=project_drafted,
            progress_callback=progress_callback,
        )

    if not project_exists:
        _emit(progress_callback, "project_creating")
        create_model_assisted_project(
            collection_root=frozen.collection_root,
            legacy_project_root=frozen.legacy_project_root,
            config_path=frozen.draft_config_path,
            output_root=frozen.project_root,
        )
        _emit(progress_callback, "project_created")
        status = model_assisted_project_status(frozen.project_root)
        _, project_drafted = _project_status(status)

    assert project_drafted is not None
    if project_drafted < MODEL_ASSISTED_RECORD_COUNT:
        _emit(
            progress_callback,
            "missing_drafts_generation_starting",
            drafted=project_drafted,
            missing=MODEL_ASSISTED_RECORD_COUNT - project_drafted,
        )

        def draft_progress(value: Mapping[str, Any]) -> None:
            payload = dict(value)
            event = str(payload.pop("event", "progress"))
            _emit(progress_callback, f"draft_{event}", **payload)

        generate_model_assisted_drafts(
            project_root=frozen.project_root,
            config_path=frozen.draft_config_path,
            runtime=runtime,
            progress_callback=draft_progress,
        )
        status = model_assisted_project_status(frozen.project_root)
        _, project_drafted = _project_status(status)
        _emit(
            progress_callback,
            "missing_drafts_generation_finished",
            drafted=project_drafted,
            missing=MODEL_ASSISTED_RECORD_COUNT - project_drafted,
        )
    if project_drafted != MODEL_ASSISTED_RECORD_COUNT:
        fail(
            "ANNOTATION_INVALID",
            "模型生成结束后仍不足 8,450 条严格草稿，拒绝发布",
            details={"drafted": project_drafted},
        )
    return _reuse_or_publish_training_assets(
        frozen,
        project_drafted=project_drafted,
        progress_callback=progress_callback,
    )
