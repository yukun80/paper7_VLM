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
    source_choices = ["All", *service.catalog.sources]
    modality_choices = list(service.catalog.modalities)
    default_suite = [task.value for task in cfg.defaults.suite_tasks]
    default_record = service.catalog.filtered(
        BenchmarkFilter(split=cfg.defaults.split)
    )[0]
    initial_selection = _selection_for_record(default_record)
    private_event = {"api_visibility": "private"}

    def record_values(record: BenchmarkRecord, *, message: str = "") -> tuple[Any, ...]:
        service.runner.clear_spatial_snapshot()
        selection = _selection_for_record(record)
        lock = record.split == "test" and not cfg.allow_test_demo
        if lock:
            status = (
                "🔒 **test locked** — metadata-only entry. No HDF5 payload, inference, "
                "or Gallery access occurred. Set `allow_test_demo: true` only with owner authorization."
            )
            optical = reference = None
            optical_gallery: list[Any] = []
            auxiliary_gallery: list[Any] = []
            channel_metadata: dict[str, Any] = {"optical": [], "auxiliary": []}
            receipt = None
        else:
            loaded = service.catalog.load(record, action="BROWSE")
            optical = loaded.optical_image
            reference = loaded.reference_mask
            optical_gallery = [
                (loaded.optical_image, "Full Optical / deterministic RGB preview"),
                *(value.gallery_value() for value in loaded.optical_channel_previews),
            ]
            auxiliary_gallery = [
                value.gallery_value() for value in loaded.auxiliary_channel_previews
            ]
            channel_metadata = {
                "optical": [
                    value.to_dict() for value in loaded.optical_channel_previews
                ],
                "auxiliary": [
                    value.to_dict() for value in loaded.auxiliary_channel_previews
                ],
            }
            receipt = None if loaded.test_receipt is None else loaded.test_receipt.receipt_id
            status = message or "✅ Benchmark payload loaded read-only."
            if record.split == "test":
                status = (
                    "⚠️ **test Demo access recorded** — blind/sealed evaluation property is permanently false; "
                    "this is not formal test evaluation."
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
        return (
            selection,
            status,
            metadata,
            optical,
            reference,
            spatial_inputs,
            formal_inputs,
            record.sample_id,
            gr.update(choices=[], value=None),
            optical_gallery,
            auxiliary_gallery,
            channel_metadata,
            [],
            [],
            None,
            None,
            None,
            "Candidate selection cleared: sample selection changed.",
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
            source=None if source == "All" else source,
            sample_id_query=query.strip() or None,
            target_status=target,
            size=size,
            modalities=tuple(modalities or ()),
        )

    def apply_filter(*values: Any) -> tuple[Any, ...]:
        filters = current_filter(*values)
        rows = service.catalog.filtered(filters)
        if not rows:
            raise gr.Error("当前人工筛选条件没有样本")
        return record_values(rows[0], message=f"✅ 筛选结果 {len(rows)} 条；显示第 1 条。")

    def navigate(selection: Mapping[str, Any], delta: int, *values: Any) -> tuple[Any, ...]:
        filters = current_filter(*values)
        record, position, total = service.catalog.navigate(
            filters,
            current_sample_id=selection.get("sample_id"),
            delta=delta,
        )
        return record_values(record, message=f"✅ 筛选结果 {total} 条；当前第 {position + 1} 条。")

    def random_record(*values: Any) -> tuple[Any, ...]:
        filters = current_filter(*values)
        record = service.catalog.random_record(filters)
        return record_values(record, message="✅ 按当前人工筛选条件随机选择；未读取模型分数。")

    def locate_exact(sample_id: str) -> tuple[Any, ...]:
        record = service.catalog.locate(sample_id.strip())
        base = record_values(record, message="✅ 已按 exact sample_id 定位并同步筛选状态。")
        return (
            *base,
            record.split,
            record.source,
            record.target_status,
            record.size,
            list(record.available_modalities),
        )

    def mutate_gallery(
        action: str,
        selection: Mapping[str, Any],
        tags: str,
        note: str,
        tasks: Sequence[str],
    ) -> tuple[Any, str]:
        if selection.get("mode") != DEMO_MODE:
            raise gr.Error("Demo Gallery 只接受当前 Benchmark Browser selection")
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
        return _gallery_rows(service.gallery), (
            f"✅ Gallery revision `{revision}` published. Qualitative demo only; "
            "formal/scientific acceptance remain false."
        )

    def execute(
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
        tasks = [single_task] if task_mode == "Single Task" else list(suite_tasks)
        selected = service.runner.canonical_tasks(tasks)
        pure_knowledge = selected == (UnifiedTask.KNOWLEDGE_QA,)
        sample_id = str(selection.get("sample_id", ""))
        split = str(selection.get("split", ""))
        record = None if pure_knowledge else service.catalog.locate(
            sample_id,
            split=split,
        )
        overrides: dict[UnifiedTask, str] = {}
        if instruction.strip():
            overrides.update({task: instruction.strip() for task in selected if task is not UnifiedTask.KNOWLEDGE_QA})
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
        return run_output_values(summary, selection)

    def run_selected_candidate(
        selection: Mapping[str, Any],
        instruction: str,
        candidate_token: Any,
    ) -> tuple[Any, ...]:
        if candidate_token in {None, ""}:
            raise gr.Error(
                "请先从 Candidate Preview 选择具体 candidate，或显式选择 global mask"
            )
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
        return run_output_values(summary, selection)

    def current_input_banner(
        selection: Mapping[str, Any],
        task_mode: str,
        single_task: str,
        suite_tasks: Sequence[str],
    ) -> str:
        tasks = [single_task] if task_mode == "Single Task" else list(suite_tasks)
        try:
            selected = service.runner.canonical_tasks(tasks)
        except Exception:
            return "⚠️ 请选择至少一个有效 UnifiedTask。"
        if selected == (UnifiedTask.KNOWLEDGE_QA,):
            return (
                "### Current execution input\n"
                "**KNOWLEDGE_ONLY — No Benchmark payload consumed.** 当前 Browser "
                "selection 只是 UI metadata，不会打开 HDF5、构造 spatial input 或创建 test receipt。"
            )
        return (
            "### Current execution input\n"
            f"mode=`{selection.get('mode')}` · split=`{selection.get('split')}` · "
            f"sample=`{selection.get('sample_id')}` · source=`{selection.get('source')}`"
        )

    def effective_prompt_preview(
        task_mode: str,
        single_task: str,
        suite_tasks: Sequence[str],
        instruction: str,
        knowledge_question: str,
    ) -> Mapping[str, Any]:
        tasks = [single_task] if task_mode == "Single Task" else list(suite_tasks)
        selected = service.runner.canonical_tasks(tasks)
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
        selection: Mapping[str, Any],
    ) -> tuple[Any, ...]:
        sample_id = str(selection.get("sample_id", ""))
        split = str(selection.get("split", ""))
        payload = service.runner.candidate_ui_payload(
            sample_id=sample_id,
            split=split,
        )
        if payload["snapshot"] is None:
            return (
                gr.update(choices=[], value=None),
                [],
                [],
                None,
                None,
                None,
                "No valid spatial result for the current sample.",
            )
        snapshot = payload["snapshot"]
        return (
            gr.update(choices=payload["choices"], value=None),
            payload["gallery"],
            payload["gallery_tokens"],
            None,
            None,
            None,
            (
                f"Spatial snapshot `{snapshot['snapshot_id']}` from "
                f"`{snapshot['source_task']}`; select a candidate or explicitly confirm global."
            ),
        )

    def candidate_preview_values(
        selection: Mapping[str, Any],
        token: Any,
    ) -> tuple[Any, ...]:
        if token in {None, ""}:
            return None, None, None, "No candidate selected."
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
        return mask, overlay, metadata, (
            "Explicit global fallback confirmed."
            if metadata.get("explicit_global_confirmed")
            else f"Candidate `{metadata.get('candidate_id')}` selected for interpretation."
        )

    def candidate_gallery_selected(
        selection: Mapping[str, Any],
        tokens: Sequence[str],
        event: gr.SelectData,
    ) -> tuple[Any, ...]:
        index = event.index
        if isinstance(index, (tuple, list)):
            index = index[0]
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(tokens):
            raise gr.Error("Candidate Gallery selection index 非法")
        token = str(tokens[index])
        mask, overlay, metadata, status = candidate_preview_values(selection, token)
        return gr.update(value=token), mask, overlay, metadata, status

    # ``from __future__ import annotations`` 会把局部延迟导入的 Gradio 类型保存为字符串；
    # 恢复真实事件类型，确保 Gallery.select 将 SelectData 作为 event arg 注入。
    candidate_gallery_selected.__annotations__["event"] = gr.SelectData

    def run_output_values(
        summary: DemoRunSummary,
        selection: Mapping[str, Any],
    ) -> tuple[Any, ...]:
        task_choices = [item.task.value for item in summary.tasks]
        successful = [item.task for item in summary.tasks if item.status == "SUCCESS"]
        if (
            summary.active_snapshot_id is not None
            and service.runner.active_snapshot is not None
            and service.runner.active_snapshot.source_task.value in task_choices
        ):
            result_value = service.runner.active_snapshot.source_task.value
        elif successful:
            result_value = successful[-1].value
        else:
            result_value = task_choices[0]
        success_count = sum(item.status == "SUCCESS" for item in summary.tasks)
        failure_count = sum(item.status == "FAILED" for item in summary.tasks)
        pending_count = sum(
            item.status == WAITING_FOR_CANDIDATE for item in summary.tasks
        )
        status = (
            f"✅ Demo run `{summary.run_id}` published: "
            f"success={success_count}, failed={failure_count}, waiting={pending_count}. "
            "Engineering evidence only; not scientific acceptance."
        )
        if summary.input_scope == "KNOWLEDGE_ONLY":
            status += " KNOWLEDGE_ONLY: No Benchmark payload consumed."
        if pending_count:
            status += " REGION_INTERPRETATION is waiting for explicit candidate/global selection."
        if summary.sealed_test_accessed:
            status += " Test receipt recorded; blind/sealed property is false."
        candidate_values = candidate_component_values(selection)
        return (
            summary.to_dict(),
            summary.to_dict(),
            gr.update(choices=task_choices, value=result_value),
            status,
            *candidate_values,
        )

    def viewer_values(run_state: Mapping[str, Any], task: str) -> tuple[Any, ...]:
        if not run_state or not task:
            return (None,) * 15
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
            {
                "final_text": response.get("text"),
                "request_preview": viewer.get("request_preview"),
                **viewer["execution_trace"],
                "failure": viewer.get("failure"),
                "pending": viewer.get("pending"),
            },
        )

    with gr.Blocks(title="OA-GroundRAG Unified Demo Workbench") as app:
        gr.Markdown(
            "# OA-GroundRAG Unified Demo Workbench\n"
            "**Read-only inference workbench.** Benchmark、checkpoint、Adapter、Text Bank "
            "和正式 outputs 均不修改；唯一持久写入为独立 Demo Gallery、test receipt 与 Demo run。",
            elem_classes="scientific-boundary",
        )
        selection_state = gr.State(initial_selection)
        run_state = gr.State({})

        with gr.Tab("A. Benchmark Browser"):
            gr.Markdown(
                "### Demo / Qualitative Exploration\n"
                "筛选与随机选择只读取 canonical index 和人工条件，不读取模型得分、不做 Top-K。"
            )
            with gr.Row():
                split = gr.Dropdown(["train", "val", "test"], value=cfg.defaults.split, label="split")
                source_filter = gr.Dropdown(source_choices, value="All", label="source")
                target_filter = gr.Dropdown(
                    ["all", "target", "no_target"], value="all", label="target / no-target"
                )
                size_filter = gr.Dropdown(
                    ["all", "empty", "small", "medium", "large"], value="all", label="size"
                )
            with gr.Row():
                modality_filter = gr.Dropdown(
                    modality_choices,
                    value=[],
                    multiselect=True,
                    label="available modalities (all selected must exist)",
                )
                sample_query = gr.Textbox(label="sample_id filter / exact lookup")
                current_sample = gr.Textbox(label="current sample_id", interactive=False)
            with gr.Row():
                apply_button = gr.Button("应用筛选")
                previous_button = gr.Button("上一条")
                next_button = gr.Button("下一条")
                random_button = gr.Button("随机样本")
                locate_button = gr.Button("指定 sample_id", variant="primary")
            browser_status = gr.Markdown()
            browser_meta = gr.JSON(label="Sample metadata", open=False)
            with gr.Row():
                browser_optical = gr.Image(
                    label="Full Optical / RGB preview", image_mode="RGB", interactive=False
                )
                with gr.Accordion("Reference / Audit Only (never USER_MASK)", open=False):
                    browser_reference = gr.Image(
                        label="Reference / Audit Only", image_mode="L", interactive=False
                    )
            optical_channel_gallery = gr.Gallery(
                label="Optical / Multispectral channel previews — deterministic per-channel rendering",
                type="pil",
                columns=4,
                rows=2,
                height="auto",
                object_fit="contain",
                interactive=False,
            )
            auxiliary_channel_gallery = gr.Gallery(
                label=(
                    "Spatial Expert Input Preview — auxiliary modalities | "
                    "Not formal MLLM grounded input in current P0"
                ),
                type="pil",
                columns=4,
                rows=2,
                height="auto",
                object_fit="contain",
                interactive=False,
            )
            channel_preview_metadata = gr.JSON(
                label="Channel values / validity / display transform",
                open=False,
            )
            with gr.Row():
                spatial_inputs = gr.JSON(label="Spatial Expert Inputs", open=True)
                formal_inputs = gr.JSON(label="MLLM Formal Grounded Inputs", open=True)

        with gr.Tab("B. Demo Gallery"):
            gr.Markdown(
                "### Qualitative Demo Gallery\n"
                "人工选择、独立 revision；不是 Gold、evaluation selection、benchmark score 或 scientific acceptance。"
            )
            with gr.Row():
                gallery_tags = gr.Textbox(label="demo_tags (comma/newline separated)")
                gallery_tasks = gr.Dropdown(
                    [task.value for task in UnifiedTask],
                    value=default_suite,
                    multiselect=True,
                    label="selected tasks",
                )
            gallery_note = gr.Textbox(label="note", lines=3)
            with gr.Row():
                gallery_add = gr.Button("加入 / 更新 Gallery", variant="primary")
                gallery_remove = gr.Button("从当前视图移除（保留 tombstone）")
            gallery_status = gr.Markdown()
            gallery_table = gr.Dataframe(
                headers=["sample_id", "split", "source", "demo_tags", "note", "selected_tasks", "updated_at"],
                value=_gallery_rows(service.gallery),
                interactive=False,
                type="array",
                label="Current qualitative selection",
            )

        with gr.Tab("C. Task Runner / Result Viewer / Trace"):
            gr.Markdown(
                "### Task Routing\n"
                "Six explicit UnifiedTask only; Suite is preflighted in full and executed in a fixed capability order."
            )
            current_execution_input = gr.Markdown(
                current_input_banner(
                    initial_selection,
                    "Task Suite",
                    UnifiedTask.VLM_ONLY.value,
                    default_suite,
                ),
                elem_classes="readonly-boundary",
            )
            with gr.Row():
                task_mode = gr.Radio(["Single Task", "Task Suite"], value="Task Suite", label="run mode")
                single_task = gr.Dropdown(
                    [task.value for task in UnifiedTask], value=UnifiedTask.VLM_ONLY.value, label="single task"
                )
                suite_tasks = gr.Dropdown(
                    [task.value for task in UnifiedTask],
                    value=default_suite,
                    multiselect=True,
                    label="suite tasks (canonical order enforced)",
                )
            instruction = gr.Textbox(
                label="visual / region instruction (blank uses task-specific config prompt)", lines=2
            )
            knowledge_question = gr.Textbox(
                label="KNOWLEDGE_QA question (blank uses config prompt)", lines=2
            )
            prompt_preview = gr.JSON(
                value=effective_prompt_preview(
                    "Task Suite",
                    UnifiedTask.VLM_ONLY.value,
                    default_suite,
                    "",
                    "",
                ),
                label="Effective prompts sent by Demo orchestration (read only)",
                open=False,
            )
            with gr.Row():
                user_mask = gr.File(
                    label="Independent user/demo mask — strict PNG-L 0/255",
                    file_types=[".png"],
                    type="filepath",
                )
                interpretation_source = gr.Dropdown(
                    [RegionSource.OA_AUXSEG_CANDIDATE.value, RegionSource.USER_MASK.value],
                    value=RegionSource.OA_AUXSEG_CANDIDATE.value,
                    label="REGION_INTERPRETATION region source",
                )
                candidate_selector = gr.Dropdown(
                    choices=[],
                    value=None,
                    allow_custom_value=False,
                    label="Candidate selection (same sample + same spatial snapshot only)",
                )
            run_button = gr.Button("运行只读 Demo inference", variant="primary")
            run_status = gr.Markdown()
            run_summary = gr.JSON(label="Suite summary", open=True)
            candidate_gallery_tokens = gr.State([])
            gr.Markdown(
                "### Candidate Preview\n"
                "Only real OA-AuxSeg candidates are shown. No GT/reference mask and no automatic Top-1 selection."
            )
            candidate_gallery = gr.Gallery(
                label="OA-AuxSeg candidate overlays",
                type="filepath",
                columns=4,
                rows=2,
                height="auto",
                object_fit="contain",
                interactive=False,
            )
            with gr.Row():
                candidate_mask = gr.Image(
                    label="Selected candidate / explicit global mask",
                    type="filepath",
                    image_mode="L",
                    interactive=False,
                )
                candidate_overlay = gr.Image(
                    label="Selected candidate overlay",
                    type="filepath",
                    interactive=False,
                )
                candidate_metadata = gr.JSON(
                    label="Candidate ID / bbox / area / confidence / binding",
                    open=True,
                )
            candidate_status = gr.Markdown("No valid spatial result for the current sample.")
            run_selected_candidate_button = gr.Button(
                "运行所选 REGION_INTERPRETATION",
                variant="primary",
            )
            result_task = gr.Dropdown(choices=[], label="Result task")
            with gr.Row():
                result_original = gr.Image(label="Original image", type="filepath", interactive=False)
                result_mask = gr.Image(label="Predicted / User Mask", type="filepath", image_mode="L", interactive=False)
                result_probability = gr.Image(label="Mask probability", type="filepath", image_mode="L", interactive=False)
                result_overlay = gr.Image(label="Predicted overlay", type="filepath", interactive=False)
            spatial_result = gr.JSON(label="Spatial Result / candidates", open=True)
            gr.Markdown("### Grounded Understanding")
            with gr.Row():
                grounded_full = gr.Image(label="Full RGB", type="filepath", interactive=False)
                grounded_mask = gr.Image(label="Binary Mask", type="filepath", image_mode="L", interactive=False)
                grounded_crop = gr.Image(label="Context Crop", type="filepath", interactive=False)
            with gr.Row():
                program_facts = gr.JSON(label="Programmatic Facts", open=True)
                pass1 = gr.JSON(label="Pass-1 structured observation", open=True)
                grounded_limitations = gr.JSON(label="limitations", open=False)
            gr.Markdown("### Knowledge Augmentation")
            evidence_table = gr.Dataframe(
                headers=["knowledge_type", "source_title", "page", "section", "Evidence ID", "content"],
                interactive=False,
                type="array",
                label="Retrieved evidence",
            )
            with gr.Row():
                pass2 = gr.JSON(label="Pass-2 interpretation", open=True)
                citations = gr.JSON(label="citations", open=True)
            with gr.Accordion("Execution Trace", open=False):
                execution_trace = gr.JSON(label="ExecutionPlan / provider order / fallback / CUDA peak", open=True)

        browser_outputs = [
            selection_state,
            browser_status,
            browser_meta,
            browser_optical,
            browser_reference,
            spatial_inputs,
            formal_inputs,
            current_sample,
            candidate_selector,
            optical_channel_gallery,
            auxiliary_channel_gallery,
            channel_preview_metadata,
            candidate_gallery,
            candidate_gallery_tokens,
            candidate_mask,
            candidate_overlay,
            candidate_metadata,
            candidate_status,
        ]
        filter_inputs = [split, source_filter, sample_query, target_filter, size_filter, modality_filter]
        app.load(
            fn=lambda: record_values(default_record),
            outputs=browser_outputs,
            api_visibility="private",
        )
        apply_button.click(fn=apply_filter, inputs=filter_inputs, outputs=browser_outputs, **private_event)
        previous_button.click(
            fn=lambda state, *values: navigate(state, -1, *values),
            inputs=[selection_state, *filter_inputs],
            outputs=browser_outputs,
            **private_event,
        )
        next_button.click(
            fn=lambda state, *values: navigate(state, 1, *values),
            inputs=[selection_state, *filter_inputs],
            outputs=browser_outputs,
            **private_event,
        )
        random_button.click(fn=random_record, inputs=filter_inputs, outputs=browser_outputs, **private_event)
        locate_button.click(
            fn=locate_exact,
            inputs=sample_query,
            outputs=[*browser_outputs, split, source_filter, target_filter, size_filter, modality_filter],
            **private_event,
        )

        gallery_add.click(
            fn=lambda state, tags, note, tasks: mutate_gallery("upsert", state, tags, note, tasks),
            inputs=[selection_state, gallery_tags, gallery_note, gallery_tasks],
            outputs=[gallery_table, gallery_status],
            **private_event,
        )
        gallery_remove.click(
            fn=lambda state, tags, note, tasks: mutate_gallery("remove", state, tags, note, tasks),
            inputs=[selection_state, gallery_tags, gallery_note, gallery_tasks],
            outputs=[gallery_table, gallery_status],
            **private_event,
        )
        banner_inputs = [selection_state, task_mode, single_task, suite_tasks]
        for component in (selection_state, task_mode, single_task, suite_tasks):
            component.change(
                fn=current_input_banner,
                inputs=banner_inputs,
                outputs=current_execution_input,
                show_progress="hidden",
                **private_event,
            )
        prompt_inputs = [
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
            component.change(
                fn=effective_prompt_preview,
                inputs=prompt_inputs,
                outputs=prompt_preview,
                show_progress="hidden",
                **private_event,
            )
        candidate_selector.change(
            fn=candidate_preview_values,
            inputs=[selection_state, candidate_selector],
            outputs=[
                candidate_mask,
                candidate_overlay,
                candidate_metadata,
                candidate_status,
            ],
            show_progress="hidden",
            **private_event,
        )
        candidate_gallery.select(
            fn=candidate_gallery_selected,
            inputs=[selection_state, candidate_gallery_tokens],
            outputs=[
                candidate_selector,
                candidate_mask,
                candidate_overlay,
                candidate_metadata,
                candidate_status,
            ],
            show_progress="hidden",
            **private_event,
        )
        run_outputs = [
            run_state,
            run_summary,
            result_task,
            run_status,
            candidate_selector,
            candidate_gallery,
            candidate_gallery_tokens,
            candidate_mask,
            candidate_overlay,
            candidate_metadata,
            candidate_status,
        ]
        run_event = run_button.click(
            fn=execute,
            inputs=[
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
            inputs=[selection_state, instruction, candidate_selector],
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
            execution_trace,
        ]
        run_event.then(
            fn=viewer_values,
            inputs=[run_state, result_task],
            outputs=viewer_outputs,
            **private_event,
        )
        candidate_run_event.then(
            fn=viewer_values,
            inputs=[run_state, result_task],
            outputs=viewer_outputs,
            **private_event,
        )
        result_task.change(
            fn=viewer_values,
            inputs=[run_state, result_task],
            outputs=viewer_outputs,
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
