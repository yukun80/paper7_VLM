"""只读 Unified Demo Workbench 的 Gradio Blocks 与本地服务入口。"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

from oa_groundrag.runtime.contracts import RegionSource, UnifiedTask

from .access import DemoTestAccessController
from .catalog import (
    BenchmarkCatalog,
    BenchmarkFilter,
    BenchmarkRecord,
)
from .config import DemoConfig, load_demo_config
from .gallery import DemoGalleryStore
from .i18n import (
    DEFAULT_LOCALE,
    EVIDENCE_HEADERS,
    GALLERY_HEADERS,
    RUN_MODE_SINGLE,
    RUN_MODE_SUITE,
    SOURCE_FILTER_ALL,
    MessageSpec,
    candidate_choices,
    candidate_gallery as localized_candidate_gallery,
    modality_choices,
    preview_gallery,
    region_source_choices,
    render_messages,
    run_mode_choices,
    size_choices,
    source_choices,
    split_choices,
    target_choices,
    task_choices,
    tr,
)
from .runner import (
    WAITING_FOR_CANDIDATE,
    DemoCandidateSelection,
    DemoRunSummary,
    UnifiedDemoRunner,
)


DEMO_MODE = "Demo / Qualitative Exploration"
_DEMO_CSS = """
.scientific-boundary { border-left: 5px solid #b45309; padding-left: 12px; }
.readonly-boundary { border-left: 5px solid #1d4ed8; padding-left: 12px; }
"""


@dataclass(frozen=True)
class DemoWorkbenchServices:
    config: DemoConfig
    catalog: BenchmarkCatalog
    gallery: DemoGalleryStore
    runner: UnifiedDemoRunner
    test_access: DemoTestAccessController


def build_demo_services(
    config: DemoConfig | Path | str,
    *,
    runtime: Any | None = None,
) -> DemoWorkbenchServices:
    resolved = config if isinstance(config, DemoConfig) else load_demo_config(config)
    access = DemoTestAccessController(
        demo_root=resolved.demo_root,
        allow_test_demo=resolved.allow_test_demo,
        benchmark_identity=resolved.benchmark.identity,
        config_sha256=resolved.config_sha256,
    )
    catalog = BenchmarkCatalog(
        resolved.benchmark,
        access_controller=access,
    )
    gallery = DemoGalleryStore(resolved.demo_root)
    runner = UnifiedDemoRunner(resolved, catalog, runtime=runtime)
    return DemoWorkbenchServices(
        config=resolved,
        catalog=catalog,
        gallery=gallery,
        runner=runner,
        test_access=access,
    )


def _gallery_rows(store: DemoGalleryStore) -> list[list[Any]]:
    return [[
        entry.sample_id,
        entry.split,
        entry.source,
        ", ".join(entry.demo_tags),
        entry.note,
        ", ".join(task.value for task in entry.selected_tasks),
        entry.updated_at,
    ] for entry in store.list_current()]


def _split_tags(value: str) -> tuple[str, ...]:
    return tuple(
        token.strip()
        for line in (value or "").splitlines()
        for token in line.split(",")
        if token.strip()
    )


def _selection_for_record(record: BenchmarkRecord) -> dict[str, Any]:
    return {
        "mode": DEMO_MODE,
        "sample_id": record.sample_id,
        "split": record.split,
        "source": record.source,
    }


def _resolve_viewer_asset(run_root: Path, value: Any) -> str | None:
    if value is None:
        return None
    path = Path(str(value))
    if not path.is_absolute():
        path = run_root / path
    path = Path(os.path.abspath(path))
    try:
        path.relative_to(run_root)
    except ValueError:
        return None
    return str(path) if path.is_file() and not path.is_symlink() else None


def ensure_demo_loopback_proxy_bypass(
    environ: MutableMapping[str, str] | None = None,
) -> dict[str, str]:
    """Gradio startup self-check 只绕过精确本机回环地址。"""

    target = os.environ if environ is None else environ
    updated: dict[str, str] = {}
    for name in ("NO_PROXY", "no_proxy"):
        values = [value.strip() for value in target.get(name, "").split(",") if value.strip()]
        for loopback in ("127.0.0.1", "localhost", "::1"):
            if loopback not in values:
                values.append(loopback)
        target[name] = ",".join(dict.fromkeys(values))
        updated[name] = target[name]
    return updated


def create_demo_app(
    config: DemoConfig | Path | str,
    *,
    services: DemoWorkbenchServices | None = None,
) -> Any:
    """构建 Workbench；Gradio 延迟导入，普通 runtime/import 不受可选依赖影响。"""

    service = services or build_demo_services(config)
    matplotlib_cache = service.config.demo_root / ".cache" / "matplotlib"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
    try:
        import gradio as gr
    except ImportError as error:  # pragma: no cover - 由 CLI 环境诊断覆盖
        raise RuntimeError(
            "Unified Demo 需要可选依赖：pip install -e '.[demo]'"
        ) from error

    cfg = service.config
    catalog_sources = tuple(service.catalog.sources)
    catalog_modalities = tuple(service.catalog.modalities)
    default_suite = [task.value for task in cfg.defaults.suite_tasks]
    default_record = service.catalog.filtered(
        BenchmarkFilter(split=cfg.defaults.split)
    )[0]
    initial_selection = _selection_for_record(default_record)
    private_event = {"api_visibility": "private"}

    def selected_tasks(
        task_mode: str,
        single_task: str,
        suite_tasks: Sequence[str],
    ) -> tuple[UnifiedTask, ...]:
        tasks = [single_task] if task_mode == RUN_MODE_SINGLE else list(suite_tasks)
        return service.runner.canonical_tasks(tasks)

    def record_values(
        locale: str,
        record: BenchmarkRecord,
        *,
        message: MessageSpec | None = None,
    ) -> tuple[Any, ...]:
        service.runner.clear_spatial_snapshot()
        selection = _selection_for_record(record)
        lock = record.split == "test" and not cfg.allow_test_demo
        if lock:
            browser_messages = (MessageSpec.create("status.browser.test_locked"),)
            optical = reference = None
            preview_state: dict[str, Any] = {
                "full_optical": None,
                "optical_channels": (),
                "auxiliary_channels": (),
            }
            channel_metadata: dict[str, Any] = {"optical": [], "auxiliary": []}
            receipt = None
        else:
            loaded = service.catalog.load(record, action="BROWSE")
            optical = loaded.optical_image
            reference = loaded.reference_mask
            preview_state = {
                "full_optical": loaded.optical_image,
                "optical_channels": loaded.optical_channel_previews,
                "auxiliary_channels": loaded.auxiliary_channel_previews,
            }
            channel_metadata = {
                "optical": [
                    value.to_dict() for value in loaded.optical_channel_previews
                ],
                "auxiliary": [
                    value.to_dict() for value in loaded.auxiliary_channel_previews
                ],
            }
            receipt = None if loaded.test_receipt is None else loaded.test_receipt.receipt_id
            browser_messages = (
                message or MessageSpec.create("status.browser.loaded"),
            )
            if record.split == "test":
                browser_messages = (
                    MessageSpec.create("status.browser.test_access"),
                )
        optical_gallery, auxiliary_gallery = preview_gallery(
            locale,
            preview_state["full_optical"],
            preview_state["optical_channels"],
            preview_state["auxiliary_channels"],
        )
        metadata = {
            **record.to_dict(),
            "test_access_receipt_id": receipt,
            "benchmark_modified": False,
            "qualitative_demo_only": True,
        }
        spatial_inputs = {
            "optical": {
                "channel_names": list(record.optical_channel_names),
                "role": "spatial boundary anchor",
            },
            "auxiliary_modalities": {
                name: {
                    "channel_names": list(channels),
                    "role": "OA-AuxSeg optional evidence",
                    "preview_label": "Spatial Expert Input Preview",
                    "p0_mllm_boundary": "Not formal MLLM grounded input in current P0",
                }
                for name, channels in record.auxiliary_channel_names.items()
            },
        }
        formal_inputs = {
            "current_preview": [] if lock else ["Full Optical"],
            "grounded_runtime_when_applicable": ["Full Optical", "Binary Mask", "Context Crop"],
            "auxiliary_modalities_formally_consumed_by_mllm": False,
            "reference_or_gt_mask_role": "Reference / Audit Only",
        }
        candidate_messages = (MessageSpec.create("status.candidate.cleared"),)
        candidate_payload = {
            "snapshot": None,
            "options": [],
            "gallery_tokens": [],
        }
        return (
            selection,
            render_messages(locale, browser_messages),
            browser_messages,
            metadata,
            optical,
            reference,
            spatial_inputs,
            formal_inputs,
            record.sample_id,
            gr.update(choices=[], value=None),
            optical_gallery,
            auxiliary_gallery,
            preview_state,
            channel_metadata,
            [],
            [],
            candidate_payload,
            None,
            None,
            None,
            render_messages(locale, candidate_messages),
            candidate_messages,
        )

    def current_filter(
        split: str,
        source: str,
        query: str,
        target: str,
        size: str,
        modalities: Sequence[str] | None,
    ) -> BenchmarkFilter:
        return BenchmarkFilter(
            split=split,
            source=None if source == SOURCE_FILTER_ALL else source,
            sample_id_query=query.strip() or None,
            target_status=target,
            size=size,
            modalities=tuple(modalities or ()),
        )

    def load_default(locale: str) -> tuple[Any, ...]:
        try:
            return record_values(locale, default_record)
        except Exception as error:
            raise gr.Error(tr(locale, "error.browser_operation")) from error

    def apply_filter(locale: str, *values: Any) -> tuple[Any, ...]:
        try:
            filters = current_filter(*values)
            rows = service.catalog.filtered(filters)
            if not rows:
                raise gr.Error(tr(locale, "error.no_samples"))
            return record_values(
                locale,
                rows[0],
                message=MessageSpec.create(
                    "status.browser.filtered",
                    total=len(rows),
                    position=1,
                ),
            )
        except gr.Error:
            raise
        except Exception as error:
            raise gr.Error(tr(locale, "error.browser_operation")) from error

    def navigate(
        locale: str,
        selection: Mapping[str, Any],
        delta: int,
        *values: Any,
    ) -> tuple[Any, ...]:
        try:
            filters = current_filter(*values)
            record, position, total = service.catalog.navigate(
                filters,
                current_sample_id=selection.get("sample_id"),
                delta=delta,
            )
            return record_values(
                locale,
                record,
                message=MessageSpec.create(
                    "status.browser.filtered",
                    total=total,
                    position=position + 1,
                ),
            )
        except Exception as error:
            raise gr.Error(tr(locale, "error.browser_operation")) from error

    def random_record(locale: str, *values: Any) -> tuple[Any, ...]:
        try:
            filters = current_filter(*values)
            record = service.catalog.random_record(filters)
            return record_values(
                locale,
                record,
                message=MessageSpec.create("status.browser.random"),
            )
        except Exception as error:
            raise gr.Error(tr(locale, "error.browser_operation")) from error

    def locate_exact(locale: str, sample_id: str) -> tuple[Any, ...]:
        try:
            record = service.catalog.locate(sample_id.strip())
            base = record_values(
                locale,
                record,
                message=MessageSpec.create("status.browser.located"),
            )
            return (
                *base,
                record.split,
                record.source,
                record.target_status,
                record.size,
                list(record.available_modalities),
            )
        except Exception as error:
            raise gr.Error(tr(locale, "error.browser_operation")) from error

    def mutate_gallery(
        action: str,
        locale: str,
        selection: Mapping[str, Any],
        tags: str,
        note: str,
        tasks: Sequence[str],
    ) -> tuple[Any, str, tuple[MessageSpec, ...]]:
        if selection.get("mode") != DEMO_MODE:
            raise gr.Error(tr(locale, "error.gallery_mode"))
        try:
            record = service.catalog.locate(
                str(selection.get("sample_id", "")),
                split=str(selection.get("split", "")),
            )
            if action == "remove":
                revision = service.gallery.remove(
                    benchmark_identity=service.catalog.identity,
                    split=record.split,
                    sample_id=record.sample_id,
                )
            else:
                receipt = None
                if record.split == "test":
                    receipt = service.test_access.issue(
                        sample_id=record.sample_id,
                        action="GALLERY",
                    )
                revision = service.gallery.upsert(
                    benchmark_identity=service.catalog.identity,
                    sample_id=record.sample_id,
                    split=record.split,
                    source=record.source,
                    demo_tags=_split_tags(tags),
                    note=note,
                    selected_tasks=tasks,
                    test_receipt=receipt,
                )
            messages = (
                MessageSpec.create("status.gallery.published", revision=revision),
            )
            return (
                _gallery_rows(service.gallery),
                render_messages(locale, messages),
                messages,
            )
        except Exception as error:
            raise gr.Error(tr(locale, "error.gallery_operation")) from error

    def execute(
        locale: str,
        selection: Mapping[str, Any],
        task_mode: str,
        single_task: str,
        suite_tasks: Sequence[str],
        instruction: str,
        knowledge_question: str,
        user_mask: str | None,
        candidate_token: Any,
        interpretation_source: str,
    ) -> tuple[Any, ...]:
        try:
            selected = selected_tasks(task_mode, single_task, suite_tasks)
            pure_knowledge = selected == (UnifiedTask.KNOWLEDGE_QA,)
            sample_id = str(selection.get("sample_id", ""))
            split = str(selection.get("split", ""))
            record = None if pure_knowledge else service.catalog.locate(
                sample_id,
                split=split,
            )
            overrides: dict[UnifiedTask, str] = {}
            if instruction.strip():
                overrides.update({
                    task: instruction.strip()
                    for task in selected
                    if task is not UnifiedTask.KNOWLEDGE_QA
                })
            if knowledge_question.strip() and UnifiedTask.KNOWLEDGE_QA in selected:
                overrides[UnifiedTask.KNOWLEDGE_QA] = knowledge_question.strip()
            source = RegionSource(interpretation_source)
            candidate = (
                None
                if candidate_token in {None, ""}
                or source is RegionSource.USER_MASK
                or UnifiedTask.REGION_INTERPRETATION not in selected
                else DemoCandidateSelection.from_token(str(candidate_token))
            )
            summary = service.runner.run(
                record=record,
                tasks=selected,
                instructions=overrides,
                user_mask=user_mask,
                candidate_selection=candidate,
                region_interpretation_source=source,
                data_mode=str(selection.get("mode")),
            )
            return run_output_values(locale, summary, selection)
        except Exception as error:
            raise gr.Error(tr(locale, "error.runner_operation")) from error

    def run_selected_candidate(
        locale: str,
        selection: Mapping[str, Any],
        instruction: str,
        candidate_token: Any,
    ) -> tuple[Any, ...]:
        if candidate_token in {None, ""}:
            raise gr.Error(tr(locale, "error.candidate_required"))
        try:
            sample_id = str(selection.get("sample_id", ""))
            split = str(selection.get("split", ""))
            record = service.catalog.locate(sample_id, split=split)
            overrides = (
                {}
                if not instruction.strip()
                else {UnifiedTask.REGION_INTERPRETATION: instruction.strip()}
            )
            summary = service.runner.run(
                record=record,
                tasks=(UnifiedTask.REGION_INTERPRETATION,),
                instructions=overrides,
                candidate_selection=DemoCandidateSelection.from_token(
                    str(candidate_token)
                ),
                region_interpretation_source=RegionSource.OA_AUXSEG_CANDIDATE,
                data_mode=str(selection.get("mode")),
            )
            return run_output_values(locale, summary, selection)
        except Exception as error:
            raise gr.Error(tr(locale, "error.runner_operation")) from error

    def current_input_banner(
        locale: str,
        selection: Mapping[str, Any],
        task_mode: str,
        single_task: str,
        suite_tasks: Sequence[str],
    ) -> str:
        try:
            selected = selected_tasks(task_mode, single_task, suite_tasks)
        except Exception:
            return tr(locale, "status.banner.invalid_tasks")
        if selected == (UnifiedTask.KNOWLEDGE_QA,):
            return tr(locale, "status.banner.knowledge")
        return tr(
            locale,
            "status.banner.visual",
            mode=(
                tr(locale, "choice.mode.demo")
                if selection.get("mode") == DEMO_MODE
                else str(selection.get("mode"))
            ),
            split=str(selection.get("split")),
            sample_id=str(selection.get("sample_id")),
            source=str(selection.get("source")),
        )

    def effective_prompt_preview(
        locale: str,
        task_mode: str,
        single_task: str,
        suite_tasks: Sequence[str],
        instruction: str,
        knowledge_question: str,
    ) -> Mapping[str, Any]:
        try:
            selected = selected_tasks(task_mode, single_task, suite_tasks)
        except Exception as error:
            raise gr.Error(tr(locale, "error.runner_operation")) from error
        values: list[dict[str, str]] = []
        for task in selected:
            if task is UnifiedTask.KNOWLEDGE_QA and knowledge_question.strip():
                prompt = knowledge_question.strip()
                source = "USER_OVERRIDE"
            elif task is not UnifiedTask.KNOWLEDGE_QA and instruction.strip():
                prompt = instruction.strip()
                source = "USER_OVERRIDE"
            else:
                prompt = cfg.defaults.prompts[task]
                source = "CONFIG_DEFAULT"
            values.append({
                "task": task.value,
                "effective_prompt": prompt,
                "prompt_source": source,
            })
        return {
            "task_order": [task.value for task in selected],
            "prompts": values,
            "read_only_preview": True,
        }

    def candidate_component_values(
        locale: str,
        selection: Mapping[str, Any],
    ) -> tuple[Any, ...]:
        sample_id = str(selection.get("sample_id", ""))
        split = str(selection.get("split", ""))
        payload = service.runner.candidate_ui_payload(
            sample_id=sample_id,
            split=split,
        )
        if payload["snapshot"] is None:
            messages = (MessageSpec.create("status.candidate.none"),)
            return (
                gr.update(choices=[], value=None),
                [],
                [],
                payload,
                None,
                None,
                None,
                render_messages(locale, messages),
                messages,
            )
        snapshot = payload["snapshot"]
        messages = (MessageSpec.create(
            "status.candidate.snapshot",
            snapshot_id=snapshot["snapshot_id"],
            source_task=snapshot["source_task"],
        ),)
        return (
            gr.update(choices=candidate_choices(locale, payload), value=None),
            localized_candidate_gallery(locale, payload),
            payload["gallery_tokens"],
            payload,
            None,
            None,
            None,
            render_messages(locale, messages),
            messages,
        )

    def candidate_preview_values(
        locale: str,
        selection: Mapping[str, Any],
        token: Any,
    ) -> tuple[Any, ...]:
        if token in {None, ""}:
            messages = (MessageSpec.create("status.candidate.unselected"),)
            return None, None, None, render_messages(locale, messages), messages
        try:
            sample_id = str(selection.get("sample_id", ""))
            split = str(selection.get("split", ""))
            selected = service.runner.resolve_candidate_selection(
                str(token),
                sample_id=sample_id,
                split=split,
            )
            mask, overlay, metadata = service.runner.candidate_preview(
                selected,
                sample_id=sample_id,
                split=split,
            )
            messages = (
                MessageSpec.create("status.candidate.global")
                if metadata.get("explicit_global_confirmed")
                else MessageSpec.create(
                    "status.candidate.selected",
                    candidate_id=metadata.get("candidate_id"),
                ),
            )
            return (
                mask,
                overlay,
                metadata,
                render_messages(locale, messages),
                messages,
            )
        except Exception as error:
            raise gr.Error(tr(locale, "error.candidate_operation")) from error

    def candidate_gallery_selected(
        locale: str,
        selection: Mapping[str, Any],
        tokens: Sequence[str],
        event: gr.SelectData,
    ) -> tuple[Any, ...]:
        index = event.index
        if isinstance(index, (tuple, list)):
            index = index[0]
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(tokens):
            raise gr.Error(tr(locale, "error.candidate_index"))
        token = str(tokens[index])
        mask, overlay, metadata, status, messages = candidate_preview_values(
            locale,
            selection,
            token,
        )
        return gr.update(value=token), mask, overlay, metadata, status, messages

    # ``from __future__ import annotations`` 会把局部延迟导入的 Gradio 类型保存为字符串；
    # 恢复真实事件类型，确保 Gallery.select 将 SelectData 作为 event arg 注入。
    candidate_gallery_selected.__annotations__["event"] = gr.SelectData

    def run_output_values(
        locale: str,
        summary: DemoRunSummary,
        selection: Mapping[str, Any],
    ) -> tuple[Any, ...]:
        task_values = [item.task.value for item in summary.tasks]
        successful = [item.task for item in summary.tasks if item.status == "SUCCESS"]
        if (
            summary.active_snapshot_id is not None
            and service.runner.active_snapshot is not None
            and service.runner.active_snapshot.source_task.value in task_values
        ):
            result_value = service.runner.active_snapshot.source_task.value
        elif successful:
            result_value = successful[-1].value
        else:
            result_value = task_values[0]
        success_count = sum(item.status == "SUCCESS" for item in summary.tasks)
        failure_count = sum(item.status == "FAILED" for item in summary.tasks)
        pending_count = sum(
            item.status == WAITING_FOR_CANDIDATE for item in summary.tasks
        )
        messages = [MessageSpec.create(
            "status.run.published",
            run_id=summary.run_id,
            success=success_count,
            failed=failure_count,
            waiting=pending_count,
        )]
        if summary.input_scope == "KNOWLEDGE_ONLY":
            messages.append(MessageSpec.create("status.run.knowledge"))
        if pending_count:
            messages.append(MessageSpec.create("status.run.pending"))
        if summary.sealed_test_accessed:
            messages.append(MessageSpec.create("status.run.test"))
        run_messages = tuple(messages)
        candidate_values = candidate_component_values(locale, selection)
        return (
            summary.to_dict(),
            summary.to_dict(),
            gr.update(choices=task_choices(locale, task_values), value=result_value),
            render_messages(locale, run_messages),
            run_messages,
            *candidate_values,
        )

    def viewer_values(
        locale: str,
        run_state: Mapping[str, Any],
        task: str,
    ) -> tuple[Any, ...]:
        if not run_state or not task:
            return (None,) * 16
        try:
            run_root = Path(str(run_state["run_root"]))
            viewer = service.runner.load_viewer(run_root, task)
            preview = viewer["input_preview"]
            spatial = viewer["spatial_result"]
            grounded = viewer["grounded_understanding"]
            knowledge = viewer["knowledge_augmentation"]
            formal = preview["mllm_formal_grounded_inputs"]
            packet = knowledge.get("evidence_packet") or {}
            evidence_rows = [[
                item.get("knowledge_type"),
                item.get("source_title"),
                item.get("pdf_page"),
                item.get("section"),
                item.get("evidence_id"),
                item.get("text", item.get("content", "")),
            ] for item in packet.get("items", [])]
            response = viewer.get("response") or {}
            original = preview["spatial_expert_inputs"].get("full_optical")
            if original is None:
                original = preview.get("direct_mllm_visual_inputs", {}).get("full_optical")
            return (
                _resolve_viewer_asset(run_root, original),
                _resolve_viewer_asset(run_root, spatial.get("predicted_mask")),
                _resolve_viewer_asset(run_root, spatial.get("mask_probability")),
                _resolve_viewer_asset(run_root, spatial.get("overlay")),
                spatial,
                _resolve_viewer_asset(run_root, formal.get("full_optical")),
                _resolve_viewer_asset(run_root, formal.get("binary_mask")),
                _resolve_viewer_asset(run_root, formal.get("context_crop")),
                grounded.get("programmatic_facts"),
                grounded.get("pass1_structured_observation"),
                grounded.get("limitations"),
                evidence_rows,
                knowledge.get("pass2_interpretation"),
                knowledge.get("citations"),
                response.get("text"),
                {
                    "final_text": response.get("text"),
                    "request_preview": viewer.get("request_preview"),
                    **viewer["execution_trace"],
                    "failure": viewer.get("failure"),
                    "pending": viewer.get("pending"),
                },
            )
        except Exception as error:
            raise gr.Error(tr(locale, "error.runner_operation")) from error

    with gr.Blocks(title=tr(DEFAULT_LOCALE, "app.document_title")) as app:
        app_header = gr.Markdown(
            tr(DEFAULT_LOCALE, "app.header"),
            elem_classes="scientific-boundary",
        )
        language_selector = gr.Radio(
            [("中文", "zh"), ("English", "en")],
            value=DEFAULT_LOCALE,
            label=tr(DEFAULT_LOCALE, "app.language.label"),
        )
        locale_state = gr.State(DEFAULT_LOCALE)
        document_title_state = gr.Textbox(
            value=tr(DEFAULT_LOCALE, "app.document_title"),
            visible=False,
        )
        selection_state = gr.State(initial_selection)
        run_state = gr.State({})
        browser_message_state = gr.State(())
        gallery_message_state = gr.State(())
        run_message_state = gr.State(())
        candidate_message_state = gr.State((MessageSpec.create("status.candidate.none"),))
        preview_state = gr.State({
            "full_optical": None,
            "optical_channels": (),
            "auxiliary_channels": (),
        })
        candidate_payload_state = gr.State({
            "snapshot": None,
            "options": [],
            "gallery_tokens": [],
        })

        with gr.Tab(tr(DEFAULT_LOCALE, "tab.browser")) as browser_tab:
            browser_intro = gr.Markdown(tr(DEFAULT_LOCALE, "browser.intro"))
            with gr.Row():
                split = gr.Dropdown(
                    split_choices(DEFAULT_LOCALE),
                    value=cfg.defaults.split,
                    label=tr(DEFAULT_LOCALE, "browser.split.label"),
                )
                source_filter = gr.Dropdown(
                    source_choices(DEFAULT_LOCALE, catalog_sources),
                    value=SOURCE_FILTER_ALL,
                    label=tr(DEFAULT_LOCALE, "browser.source.label"),
                )
                target_filter = gr.Dropdown(
                    target_choices(DEFAULT_LOCALE),
                    value="all",
                    label=tr(DEFAULT_LOCALE, "browser.target.label"),
                )
                size_filter = gr.Dropdown(
                    size_choices(DEFAULT_LOCALE),
                    value="all",
                    label=tr(DEFAULT_LOCALE, "browser.size.label"),
                )
            with gr.Row():
                modality_filter = gr.Dropdown(
                    modality_choices(DEFAULT_LOCALE, catalog_modalities),
                    value=[],
                    multiselect=True,
                    label=tr(DEFAULT_LOCALE, "browser.modalities.label"),
                )
                sample_query = gr.Textbox(
                    label=tr(DEFAULT_LOCALE, "browser.sample_query.label")
                )
                current_sample = gr.Textbox(
                    label=tr(DEFAULT_LOCALE, "browser.current_sample.label"),
                    interactive=False,
                )
            with gr.Row():
                apply_button = gr.Button(tr(DEFAULT_LOCALE, "browser.apply"))
                previous_button = gr.Button(tr(DEFAULT_LOCALE, "browser.previous"))
                next_button = gr.Button(tr(DEFAULT_LOCALE, "browser.next"))
                random_button = gr.Button(tr(DEFAULT_LOCALE, "browser.random"))
                locate_button = gr.Button(
                    tr(DEFAULT_LOCALE, "browser.locate"),
                    variant="primary",
                )
            browser_status = gr.Markdown()
            browser_meta = gr.JSON(
                label=tr(DEFAULT_LOCALE, "browser.metadata.label"),
                open=False,
            )
            with gr.Row():
                browser_optical = gr.Image(
                    label=tr(DEFAULT_LOCALE, "browser.optical.label"),
                    image_mode="RGB",
                    interactive=False,
                )
                with gr.Accordion(
                    tr(DEFAULT_LOCALE, "browser.reference.accordion"),
                    open=False,
                ) as reference_accordion:
                    browser_reference = gr.Image(
                        label=tr(DEFAULT_LOCALE, "browser.reference.label"),
                        image_mode="L",
                        interactive=False,
                    )
            optical_channel_gallery = gr.Gallery(
                label=tr(DEFAULT_LOCALE, "browser.optical_channels.label"),
                type="pil",
                columns=4,
                rows=2,
                height="auto",
                object_fit="contain",
                interactive=False,
            )
            auxiliary_channel_gallery = gr.Gallery(
                label=tr(DEFAULT_LOCALE, "browser.auxiliary.label"),
                type="pil",
                columns=4,
                rows=2,
                height="auto",
                object_fit="contain",
                interactive=False,
            )
            channel_preview_metadata = gr.JSON(
                label=tr(DEFAULT_LOCALE, "browser.channel_metadata.label"),
                open=False,
            )
            with gr.Row():
                spatial_inputs = gr.JSON(
                    label=tr(DEFAULT_LOCALE, "browser.spatial_inputs.label"),
                    open=True,
                )
                formal_inputs = gr.JSON(
                    label=tr(DEFAULT_LOCALE, "browser.formal_inputs.label"),
                    open=True,
                )

        with gr.Tab(tr(DEFAULT_LOCALE, "tab.gallery")) as gallery_tab:
            gallery_intro = gr.Markdown(tr(DEFAULT_LOCALE, "gallery.intro"))
            with gr.Row():
                gallery_tags = gr.Textbox(
                    label=tr(DEFAULT_LOCALE, "gallery.tags.label")
                )
                gallery_tasks = gr.Dropdown(
                    task_choices(DEFAULT_LOCALE),
                    value=default_suite,
                    multiselect=True,
                    label=tr(DEFAULT_LOCALE, "gallery.tasks.label"),
                )
            gallery_note = gr.Textbox(
                label=tr(DEFAULT_LOCALE, "gallery.note.label"),
                lines=3,
            )
            with gr.Row():
                gallery_add = gr.Button(
                    tr(DEFAULT_LOCALE, "gallery.add"),
                    variant="primary",
                )
                gallery_remove = gr.Button(tr(DEFAULT_LOCALE, "gallery.remove"))
            gallery_status = gr.Markdown()
            gallery_table = gr.Dataframe(
                headers=GALLERY_HEADERS[DEFAULT_LOCALE],
                value=_gallery_rows(service.gallery),
                interactive=False,
                type="array",
                label=tr(DEFAULT_LOCALE, "gallery.table.label"),
            )

        with gr.Tab(tr(DEFAULT_LOCALE, "tab.runner")) as runner_tab:
            runner_intro = gr.Markdown(tr(DEFAULT_LOCALE, "runner.intro"))
            current_execution_input = gr.Markdown(
                current_input_banner(
                    DEFAULT_LOCALE,
                    initial_selection,
                    RUN_MODE_SUITE,
                    UnifiedTask.VLM_ONLY.value,
                    default_suite,
                ),
                elem_classes="readonly-boundary",
            )
            with gr.Row():
                task_mode = gr.Radio(
                    run_mode_choices(DEFAULT_LOCALE),
                    value=RUN_MODE_SUITE,
                    label=tr(DEFAULT_LOCALE, "runner.mode.label"),
                )
                single_task = gr.Dropdown(
                    task_choices(DEFAULT_LOCALE),
                    value=UnifiedTask.VLM_ONLY.value,
                    label=tr(DEFAULT_LOCALE, "runner.single_task.label"),
                )
                suite_tasks = gr.Dropdown(
                    task_choices(DEFAULT_LOCALE),
                    value=default_suite,
                    multiselect=True,
                    label=tr(DEFAULT_LOCALE, "runner.suite_tasks.label"),
                )
            instruction = gr.Textbox(
                label=tr(DEFAULT_LOCALE, "runner.instruction.label"),
                lines=2,
            )
            knowledge_question = gr.Textbox(
                label=tr(DEFAULT_LOCALE, "runner.question.label"),
                lines=2,
            )
            prompt_preview = gr.JSON(
                value=effective_prompt_preview(
                    DEFAULT_LOCALE,
                    RUN_MODE_SUITE,
                    UnifiedTask.VLM_ONLY.value,
                    default_suite,
                    "",
                    "",
                ),
                label=tr(DEFAULT_LOCALE, "runner.prompts.label"),
                open=False,
            )
            with gr.Row():
                user_mask = gr.File(
                    label=tr(DEFAULT_LOCALE, "runner.user_mask.label"),
                    file_types=[".png"],
                    type="filepath",
                )
                interpretation_source = gr.Dropdown(
                    region_source_choices(DEFAULT_LOCALE),
                    value=RegionSource.OA_AUXSEG_CANDIDATE.value,
                    label=tr(DEFAULT_LOCALE, "runner.region_source.label"),
                )
                candidate_selector = gr.Dropdown(
                    choices=[],
                    value=None,
                    allow_custom_value=False,
                    label=tr(DEFAULT_LOCALE, "runner.candidate_selector.label"),
                )
            run_button = gr.Button(
                tr(DEFAULT_LOCALE, "runner.run"),
                variant="primary",
            )
            run_status = gr.Markdown()
            run_summary = gr.JSON(
                label=tr(DEFAULT_LOCALE, "runner.summary.label"),
                open=True,
            )
            candidate_gallery_tokens = gr.State([])
            candidate_intro = gr.Markdown(tr(DEFAULT_LOCALE, "candidate.intro"))
            candidate_gallery = gr.Gallery(
                label=tr(DEFAULT_LOCALE, "candidate.gallery.label"),
                type="filepath",
                columns=4,
                rows=2,
                height="auto",
                object_fit="contain",
                interactive=False,
            )
            with gr.Row():
                candidate_mask = gr.Image(
                    label=tr(DEFAULT_LOCALE, "candidate.mask.label"),
                    type="filepath",
                    image_mode="L",
                    interactive=False,
                )
                candidate_overlay = gr.Image(
                    label=tr(DEFAULT_LOCALE, "candidate.overlay.label"),
                    type="filepath",
                    interactive=False,
                )
                candidate_metadata = gr.JSON(
                    label=tr(DEFAULT_LOCALE, "candidate.metadata.label"),
                    open=True,
                )
            candidate_status = gr.Markdown(
                render_messages(
                    DEFAULT_LOCALE,
                    (MessageSpec.create("status.candidate.none"),),
                )
            )
            run_selected_candidate_button = gr.Button(
                tr(DEFAULT_LOCALE, "candidate.run"),
                variant="primary",
            )
            result_task = gr.Dropdown(
                choices=[],
                label=tr(DEFAULT_LOCALE, "result.task.label"),
            )
            with gr.Row():
                result_original = gr.Image(
                    label=tr(DEFAULT_LOCALE, "result.original.label"),
                    type="filepath",
                    interactive=False,
                )
                result_mask = gr.Image(
                    label=tr(DEFAULT_LOCALE, "result.mask.label"),
                    type="filepath",
                    image_mode="L",
                    interactive=False,
                )
                result_probability = gr.Image(
                    label=tr(DEFAULT_LOCALE, "result.probability.label"),
                    type="filepath",
                    image_mode="L",
                    interactive=False,
                )
                result_overlay = gr.Image(
                    label=tr(DEFAULT_LOCALE, "result.overlay.label"),
                    type="filepath",
                    interactive=False,
                )
            spatial_result = gr.JSON(
                label=tr(DEFAULT_LOCALE, "result.spatial.label"),
                open=True,
            )
            raw_model_output = gr.Textbox(
                label=tr(DEFAULT_LOCALE, "result.raw_output.label"),
                lines=4,
                interactive=False,
            )
            grounded_heading = gr.Markdown(tr(DEFAULT_LOCALE, "grounded.heading"))
            with gr.Row():
                grounded_full = gr.Image(
                    label=tr(DEFAULT_LOCALE, "grounded.full.label"),
                    type="filepath",
                    interactive=False,
                )
                grounded_mask = gr.Image(
                    label=tr(DEFAULT_LOCALE, "grounded.mask.label"),
                    type="filepath",
                    image_mode="L",
                    interactive=False,
                )
                grounded_crop = gr.Image(
                    label=tr(DEFAULT_LOCALE, "grounded.crop.label"),
                    type="filepath",
                    interactive=False,
                )
            with gr.Row():
                program_facts = gr.JSON(
                    label=tr(DEFAULT_LOCALE, "grounded.facts.label"),
                    open=True,
                )
                pass1 = gr.JSON(
                    label=tr(DEFAULT_LOCALE, "grounded.pass1.label"),
                    open=True,
                )
                grounded_limitations = gr.JSON(
                    label=tr(DEFAULT_LOCALE, "grounded.limitations.label"),
                    open=False,
                )
            knowledge_heading = gr.Markdown(tr(DEFAULT_LOCALE, "knowledge.heading"))
            evidence_table = gr.Dataframe(
                headers=EVIDENCE_HEADERS[DEFAULT_LOCALE],
                interactive=False,
                type="array",
                label=tr(DEFAULT_LOCALE, "knowledge.evidence.label"),
            )
            with gr.Row():
                pass2 = gr.JSON(
                    label=tr(DEFAULT_LOCALE, "knowledge.pass2.label"),
                    open=True,
                )
                citations = gr.JSON(
                    label=tr(DEFAULT_LOCALE, "knowledge.citations.label"),
                    open=True,
                )
            with gr.Accordion(
                tr(DEFAULT_LOCALE, "trace.accordion"),
                open=False,
            ) as trace_accordion:
                execution_trace = gr.JSON(
                    label=tr(DEFAULT_LOCALE, "trace.label"),
                    open=True,
                )

        static_i18n_bindings = [
            (app_header, "value", "app.header"),
            (language_selector, "label", "app.language.label"),
            (browser_tab, "label", "tab.browser"),
            (browser_intro, "value", "browser.intro"),
            (sample_query, "label", "browser.sample_query.label"),
            (current_sample, "label", "browser.current_sample.label"),
            (apply_button, "value", "browser.apply"),
            (previous_button, "value", "browser.previous"),
            (next_button, "value", "browser.next"),
            (random_button, "value", "browser.random"),
            (locate_button, "value", "browser.locate"),
            (browser_meta, "label", "browser.metadata.label"),
            (browser_optical, "label", "browser.optical.label"),
            (reference_accordion, "label", "browser.reference.accordion"),
            (browser_reference, "label", "browser.reference.label"),
            (channel_preview_metadata, "label", "browser.channel_metadata.label"),
            (spatial_inputs, "label", "browser.spatial_inputs.label"),
            (formal_inputs, "label", "browser.formal_inputs.label"),
            (gallery_tab, "label", "tab.gallery"),
            (gallery_intro, "value", "gallery.intro"),
            (gallery_tags, "label", "gallery.tags.label"),
            (gallery_note, "label", "gallery.note.label"),
            (gallery_add, "value", "gallery.add"),
            (gallery_remove, "value", "gallery.remove"),
            (runner_tab, "label", "tab.runner"),
            (runner_intro, "value", "runner.intro"),
            (instruction, "label", "runner.instruction.label"),
            (knowledge_question, "label", "runner.question.label"),
            (prompt_preview, "label", "runner.prompts.label"),
            (user_mask, "label", "runner.user_mask.label"),
            (run_button, "value", "runner.run"),
            (run_summary, "label", "runner.summary.label"),
            (candidate_intro, "value", "candidate.intro"),
            (candidate_mask, "label", "candidate.mask.label"),
            (candidate_overlay, "label", "candidate.overlay.label"),
            (candidate_metadata, "label", "candidate.metadata.label"),
            (run_selected_candidate_button, "value", "candidate.run"),
            (result_original, "label", "result.original.label"),
            (result_mask, "label", "result.mask.label"),
            (result_probability, "label", "result.probability.label"),
            (result_overlay, "label", "result.overlay.label"),
            (spatial_result, "label", "result.spatial.label"),
            (raw_model_output, "label", "result.raw_output.label"),
            (grounded_heading, "value", "grounded.heading"),
            (grounded_full, "label", "grounded.full.label"),
            (grounded_mask, "label", "grounded.mask.label"),
            (grounded_crop, "label", "grounded.crop.label"),
            (program_facts, "label", "grounded.facts.label"),
            (pass1, "label", "grounded.pass1.label"),
            (grounded_limitations, "label", "grounded.limitations.label"),
            (knowledge_heading, "value", "knowledge.heading"),
            (pass2, "label", "knowledge.pass2.label"),
            (citations, "label", "knowledge.citations.label"),
            (trace_accordion, "label", "trace.accordion"),
            (execution_trace, "label", "trace.label"),
        ]

        def switch_language(
            locale: str,
            current_preview_state: Mapping[str, Any],
            current_candidate_payload: Mapping[str, Any],
            current_candidate_token: Any,
            current_run_state: Mapping[str, Any],
            current_result_task: Any,
            current_selection: Mapping[str, Any],
            current_task_mode: str,
            current_single_task: str,
            current_suite_tasks: Sequence[str],
            current_split: str,
            current_source: str,
            current_target: str,
            current_size: str,
            current_modalities: Sequence[str],
            current_gallery_tasks: Sequence[str],
            current_region_source: str,
            browser_messages: Sequence[MessageSpec],
            gallery_messages: Sequence[MessageSpec],
            run_messages: Sequence[MessageSpec],
            candidate_messages: Sequence[MessageSpec],
        ) -> tuple[Any, ...]:
            static_updates = [
                gr.update(**{field: tr(locale, key)})
                for _, field, key in static_i18n_bindings
            ]
            optical_gallery, auxiliary_gallery = preview_gallery(
                locale,
                current_preview_state.get("full_optical"),
                current_preview_state.get("optical_channels", ()),
                current_preview_state.get("auxiliary_channels", ()),
            )
            options = current_candidate_payload.get("options", ())
            valid_tokens = {str(option["token"]) for option in options}
            if (
                current_candidate_token not in {None, ""}
                and str(current_candidate_token) not in valid_tokens
            ):
                raise gr.Error(tr(locale, "error.i18n_state"))
            candidate_value = (
                None
                if current_candidate_token in {None, ""}
                else str(current_candidate_token)
            )
            run_task_values = [
                str(item["task"])
                for item in current_run_state.get("tasks", ())
                if isinstance(item, Mapping) and item.get("task")
            ]
            if (
                current_result_task not in {None, ""}
                and str(current_result_task) not in run_task_values
            ):
                raise gr.Error(tr(locale, "error.i18n_state"))
            result_value = (
                None
                if current_result_task in {None, ""}
                else str(current_result_task)
            )
            special_updates = [
                gr.update(
                    choices=split_choices(locale),
                    value=current_split,
                    label=tr(locale, "browser.split.label"),
                ),
                gr.update(
                    choices=source_choices(locale, catalog_sources),
                    value=current_source,
                    label=tr(locale, "browser.source.label"),
                ),
                gr.update(
                    choices=target_choices(locale),
                    value=current_target,
                    label=tr(locale, "browser.target.label"),
                ),
                gr.update(
                    choices=size_choices(locale),
                    value=current_size,
                    label=tr(locale, "browser.size.label"),
                ),
                gr.update(
                    choices=modality_choices(locale, catalog_modalities),
                    value=list(current_modalities or ()),
                    label=tr(locale, "browser.modalities.label"),
                ),
                gr.update(
                    choices=task_choices(locale),
                    value=list(current_gallery_tasks or ()),
                    label=tr(locale, "gallery.tasks.label"),
                ),
                gr.update(
                    choices=run_mode_choices(locale),
                    value=current_task_mode,
                    label=tr(locale, "runner.mode.label"),
                ),
                gr.update(
                    choices=task_choices(locale),
                    value=current_single_task,
                    label=tr(locale, "runner.single_task.label"),
                ),
                gr.update(
                    choices=task_choices(locale),
                    value=list(current_suite_tasks or ()),
                    label=tr(locale, "runner.suite_tasks.label"),
                ),
                gr.update(
                    choices=region_source_choices(locale),
                    value=current_region_source,
                    label=tr(locale, "runner.region_source.label"),
                ),
                gr.update(
                    choices=candidate_choices(locale, current_candidate_payload),
                    value=candidate_value,
                    label=tr(locale, "runner.candidate_selector.label"),
                ),
                gr.update(
                    value=localized_candidate_gallery(locale, current_candidate_payload),
                    label=tr(locale, "candidate.gallery.label"),
                ),
                gr.update(
                    choices=task_choices(locale, run_task_values),
                    value=result_value,
                    label=tr(locale, "result.task.label"),
                ),
                gr.update(
                    headers=GALLERY_HEADERS[locale],
                    label=tr(locale, "gallery.table.label"),
                ),
                gr.update(
                    headers=EVIDENCE_HEADERS[locale],
                    label=tr(locale, "knowledge.evidence.label"),
                ),
                gr.update(
                    value=optical_gallery,
                    label=tr(locale, "browser.optical_channels.label"),
                ),
                gr.update(
                    value=auxiliary_gallery,
                    label=tr(locale, "browser.auxiliary.label"),
                ),
                render_messages(locale, browser_messages),
                render_messages(locale, gallery_messages),
                render_messages(locale, run_messages),
                render_messages(locale, candidate_messages),
                current_input_banner(
                    locale,
                    current_selection,
                    current_task_mode,
                    current_single_task,
                    current_suite_tasks,
                ),
            ]
            return (
                locale,
                tr(locale, "app.document_title"),
                *static_updates,
                *special_updates,
            )

        browser_outputs = [
            selection_state,
            browser_status,
            browser_message_state,
            browser_meta,
            browser_optical,
            browser_reference,
            spatial_inputs,
            formal_inputs,
            current_sample,
            candidate_selector,
            optical_channel_gallery,
            auxiliary_channel_gallery,
            preview_state,
            channel_preview_metadata,
            candidate_gallery,
            candidate_gallery_tokens,
            candidate_payload_state,
            candidate_mask,
            candidate_overlay,
            candidate_metadata,
            candidate_status,
            candidate_message_state,
        ]
        filter_inputs = [split, source_filter, sample_query, target_filter, size_filter, modality_filter]
        app.load(
            fn=load_default,
            inputs=locale_state,
            outputs=browser_outputs,
            api_visibility="private",
        )
        apply_button.click(
            fn=apply_filter,
            inputs=[locale_state, *filter_inputs],
            outputs=browser_outputs,
            **private_event,
        )
        previous_button.click(
            fn=lambda locale, state, *values: navigate(locale, state, -1, *values),
            inputs=[locale_state, selection_state, *filter_inputs],
            outputs=browser_outputs,
            **private_event,
        )
        next_button.click(
            fn=lambda locale, state, *values: navigate(locale, state, 1, *values),
            inputs=[locale_state, selection_state, *filter_inputs],
            outputs=browser_outputs,
            **private_event,
        )
        random_button.click(
            fn=random_record,
            inputs=[locale_state, *filter_inputs],
            outputs=browser_outputs,
            **private_event,
        )
        locate_button.click(
            fn=locate_exact,
            inputs=[locale_state, sample_query],
            outputs=[*browser_outputs, split, source_filter, target_filter, size_filter, modality_filter],
            **private_event,
        )

        gallery_add.click(
            fn=lambda locale, state, tags, note, tasks: mutate_gallery(
                "upsert", locale, state, tags, note, tasks
            ),
            inputs=[locale_state, selection_state, gallery_tags, gallery_note, gallery_tasks],
            outputs=[gallery_table, gallery_status, gallery_message_state],
            **private_event,
        )
        gallery_remove.click(
            fn=lambda locale, state, tags, note, tasks: mutate_gallery(
                "remove", locale, state, tags, note, tasks
            ),
            inputs=[locale_state, selection_state, gallery_tags, gallery_note, gallery_tasks],
            outputs=[gallery_table, gallery_status, gallery_message_state],
            **private_event,
        )
        banner_inputs = [locale_state, selection_state, task_mode, single_task, suite_tasks]
        selection_state.change(
            fn=current_input_banner,
            inputs=banner_inputs,
            outputs=current_execution_input,
            show_progress="hidden",
            **private_event,
        )
        for component in (task_mode, single_task, suite_tasks):
            component.input(
                fn=current_input_banner,
                inputs=banner_inputs,
                outputs=current_execution_input,
                show_progress="hidden",
                **private_event,
            )
        prompt_inputs = [
            locale_state,
            task_mode,
            single_task,
            suite_tasks,
            instruction,
            knowledge_question,
        ]
        for component in (
            task_mode,
            single_task,
            suite_tasks,
            instruction,
            knowledge_question,
        ):
            component.input(
                fn=effective_prompt_preview,
                inputs=prompt_inputs,
                outputs=prompt_preview,
                show_progress="hidden",
                **private_event,
            )
        candidate_selector.input(
            fn=candidate_preview_values,
            inputs=[locale_state, selection_state, candidate_selector],
            outputs=[
                candidate_mask,
                candidate_overlay,
                candidate_metadata,
                candidate_status,
                candidate_message_state,
            ],
            show_progress="hidden",
            **private_event,
        )
        candidate_gallery.select(
            fn=candidate_gallery_selected,
            inputs=[locale_state, selection_state, candidate_gallery_tokens],
            outputs=[
                candidate_selector,
                candidate_mask,
                candidate_overlay,
                candidate_metadata,
                candidate_status,
                candidate_message_state,
            ],
            show_progress="hidden",
            **private_event,
        )
        run_outputs = [
            run_state,
            run_summary,
            result_task,
            run_status,
            run_message_state,
            candidate_selector,
            candidate_gallery,
            candidate_gallery_tokens,
            candidate_payload_state,
            candidate_mask,
            candidate_overlay,
            candidate_metadata,
            candidate_status,
            candidate_message_state,
        ]
        run_event = run_button.click(
            fn=execute,
            inputs=[
                locale_state,
                selection_state,
                task_mode,
                single_task,
                suite_tasks,
                instruction,
                knowledge_question,
                user_mask,
                candidate_selector,
                interpretation_source,
            ],
            outputs=run_outputs,
            concurrency_limit=1,
            concurrency_id="oa_groundrag_demo_gpu",
            **private_event,
        )
        candidate_run_event = run_selected_candidate_button.click(
            fn=run_selected_candidate,
            inputs=[locale_state, selection_state, instruction, candidate_selector],
            outputs=run_outputs,
            concurrency_limit=1,
            concurrency_id="oa_groundrag_demo_gpu",
            **private_event,
        )
        viewer_outputs = [
            result_original,
            result_mask,
            result_probability,
            result_overlay,
            spatial_result,
            grounded_full,
            grounded_mask,
            grounded_crop,
            program_facts,
            pass1,
            grounded_limitations,
            evidence_table,
            pass2,
            citations,
            raw_model_output,
            execution_trace,
        ]
        run_event.then(
            fn=viewer_values,
            inputs=[locale_state, run_state, result_task],
            outputs=viewer_outputs,
            **private_event,
        )
        candidate_run_event.then(
            fn=viewer_values,
            inputs=[locale_state, run_state, result_task],
            outputs=viewer_outputs,
            **private_event,
        )
        result_task.input(
            fn=viewer_values,
            inputs=[locale_state, run_state, result_task],
            outputs=viewer_outputs,
            show_progress="hidden",
            **private_event,
        )

        switch_inputs = [
            language_selector,
            preview_state,
            candidate_payload_state,
            candidate_selector,
            run_state,
            result_task,
            selection_state,
            task_mode,
            single_task,
            suite_tasks,
            split,
            source_filter,
            target_filter,
            size_filter,
            modality_filter,
            gallery_tasks,
            interpretation_source,
            browser_message_state,
            gallery_message_state,
            run_message_state,
            candidate_message_state,
        ]
        switch_outputs = [
            locale_state,
            document_title_state,
            *(component for component, _, _ in static_i18n_bindings),
            split,
            source_filter,
            target_filter,
            size_filter,
            modality_filter,
            gallery_tasks,
            task_mode,
            single_task,
            suite_tasks,
            interpretation_source,
            candidate_selector,
            candidate_gallery,
            result_task,
            gallery_table,
            evidence_table,
            optical_channel_gallery,
            auxiliary_channel_gallery,
            browser_status,
            gallery_status,
            run_status,
            candidate_status,
            current_execution_input,
        ]
        language_event = language_selector.input(
            fn=switch_language,
            inputs=switch_inputs,
            outputs=switch_outputs,
            queue=False,
            show_progress="hidden",
            concurrency_limit=None,
            **private_event,
        )
        language_event.then(
            fn=None,
            inputs=document_title_state,
            outputs=None,
            js="(title) => { document.title = title; }",
            queue=False,
            show_progress="hidden",
            **private_event,
        )
    return app.queue(api_open=False, default_concurrency_limit=1)


def _blocked_paths(config: DemoConfig) -> list[str]:
    blocked = [
        config.unified.repository_root / "models_zoo",
        config.benchmark.root,
    ]
    outputs = config.unified.repository_root / "outputs"
    if outputs.is_dir():
        for child in outputs.iterdir():
            try:
                config.demo_root.relative_to(child)
            except ValueError:
                blocked.append(child)
    return [str(path) for path in blocked if path.exists()]


def serve_demo(
    config: DemoConfig | Path | str,
    *,
    port: int = 7860,
    prevent_thread_lock: bool = False,
) -> Any:
    """仅绑定 127.0.0.1；禁用 share、公开 API、MCP 与 monitoring。"""

    if isinstance(port, bool) or not isinstance(port, int) or not 1024 <= port <= 65535:
        raise ValueError("port 必须位于 [1024,65535]")
    resolved = config if isinstance(config, DemoConfig) else load_demo_config(config)
    services = build_demo_services(resolved)
    app = create_demo_app(resolved, services=services)
    ensure_demo_loopback_proxy_bypass()
    allowed = [str(resolved.demo_root)]
    try:
        app.launch(
            server_name="127.0.0.1",
            server_port=port,
            share=False,
            inbrowser=False,
            show_error=True,
            prevent_thread_lock=prevent_thread_lock,
            footer_links=[],
            allowed_paths=allowed,
            blocked_paths=_blocked_paths(resolved),
            enable_monitoring=False,
            strict_cors=True,
            mcp_server=False,
            max_file_size="64mb",
            css=_DEMO_CSS,
        )
    except Exception:
        app.close()
        raise
    return app
