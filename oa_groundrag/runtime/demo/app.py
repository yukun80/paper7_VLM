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
    FrozenEvaluationCatalog,
    FrozenEvaluationItem,
)
from .config import DemoConfig, load_demo_config
from .gallery import DemoGalleryStore
from .runner import DemoRunSummary, UnifiedDemoRunner


DEMO_MODE = "Demo / Qualitative Exploration"
FROZEN_MODE = "Frozen Evaluation / Read Only"
_DEMO_CSS = """
.scientific-boundary { border-left: 5px solid #b45309; padding-left: 12px; }
.readonly-boundary { border-left: 5px solid #1d4ed8; padding-left: 12px; }
"""


@dataclass(frozen=True)
class DemoWorkbenchServices:
    config: DemoConfig
    catalog: BenchmarkCatalog
    frozen: Mapping[str, FrozenEvaluationCatalog]
    gallery: DemoGalleryStore
    runner: UnifiedDemoRunner
    test_access: DemoTestAccessController


def build_demo_services(
    config: DemoConfig | Path | str,
    *,
    runtime: Any | None = None,
    verify_frozen_payloads: bool = True,
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
    frozen = {
        binding.name: FrozenEvaluationCatalog(
            binding,
            verify_payloads=verify_frozen_payloads,
        )
        for binding in resolved.frozen_evaluations
    }
    gallery = DemoGalleryStore(resolved.demo_root)
    runner = UnifiedDemoRunner(resolved, catalog, runtime=runtime)
    return DemoWorkbenchServices(
        config=resolved,
        catalog=catalog,
        frozen=frozen,
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
        "frozen_name": None,
        "frozen_ordinal": None,
    }


def _selection_for_frozen(item: FrozenEvaluationItem) -> dict[str, Any]:
    return {
        "mode": FROZEN_MODE,
        "sample_id": item.sample_id,
        "split": "val",
        "source": item.baseline_record["source"],
        "frozen_name": item.evaluation_name,
        "frozen_ordinal": item.ordinal,
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
    frozen_names = list(service.frozen)
    default_suite = [task.value for task in cfg.defaults.suite_tasks]
    default_record = service.catalog.filtered(
        BenchmarkFilter(split=cfg.defaults.split)
    )[0]
    initial_selection = _selection_for_record(default_record)
    private_event = {"api_visibility": "private"}

    def record_values(record: BenchmarkRecord, *, message: str = "") -> tuple[Any, ...]:
        selection = _selection_for_record(record)
        lock = record.split == "test" and not cfg.allow_test_demo
        if lock:
            status = (
                "🔒 **test locked** — metadata-only entry. No HDF5 payload, inference, "
                "or Gallery access occurred. Set `allow_test_demo: true` only with owner authorization."
            )
            optical = reference = None
            receipt = None
        else:
            loaded = service.catalog.load(record, action="BROWSE")
            optical = loaded.optical_image
            reference = loaded.reference_mask
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
                name: {"channel_names": list(channels), "role": "OA-AuxSeg optional evidence"}
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

    def frozen_values(name: str, ordinal: int) -> tuple[Any, ...]:
        item = service.frozen[name].item(int(ordinal))
        variants = {
            kind: {
                "record_id": record.get("record_id"),
                "representation_mode": record.get("representation_mode"),
                "formal_model_input_roles": record.get("formal_model_input_roles"),
                "target_status": record.get("target_status"),
            }
            for kind, record in item.counterfactual_records.items()
        }
        return (
            _selection_for_frozen(item),
            item.ordinal,
            (
                "🔒 **Frozen Evaluation / Read Only** — 100 baseline identity and ledger verified; "
                "selection cannot be added, removed, replaced, or sent to Gallery."
            ),
            item.to_dict(),
            None if item.asset_path("optical_full") is None else str(item.asset_path("optical_full")),
            None if item.asset_path("binary_mask") is None else str(item.asset_path("binary_mask")),
            None if item.asset_path("context_crop") is None else str(item.asset_path("context_crop")),
            variants,
            gr.update(
                choices=list(item.counterfactual_records),
                value="baseline_correct_mask",
            ),
            gr.update(choices=[], value=None),
        )

    def frozen_variant_values(name: str, ordinal: int, variant: str) -> tuple[Any, ...]:
        item = service.frozen[name].item(int(ordinal))
        if variant not in item.counterfactual_records:
            raise gr.Error(f"当前 Frozen item 不含变体：{variant}")
        paths = [item.asset_path(role, variant=variant) for role in (
            "optical_full", "binary_mask", "context_crop"
        )]
        return tuple(None if path is None else str(path) for path in paths)

    def mutate_gallery(
        action: str,
        selection: Mapping[str, Any],
        tags: str,
        note: str,
        tasks: Sequence[str],
    ) -> tuple[Any, str]:
        if selection.get("mode") != DEMO_MODE:
            raise gr.Error("Frozen Evaluation selection 不能进入 Demo Gallery")
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
        candidate_id: Any,
        interpretation_source: str,
    ) -> tuple[Any, ...]:
        sample_id = str(selection.get("sample_id", ""))
        split = str(selection.get("split", ""))
        record = service.catalog.locate(sample_id, split=split)
        tasks = [single_task] if task_mode == "Single Task" else list(suite_tasks)
        selected = service.runner.canonical_tasks(tasks)
        overrides: dict[UnifiedTask, str] = {}
        if instruction.strip():
            overrides.update({task: instruction.strip() for task in selected if task is not UnifiedTask.KNOWLEDGE_QA})
        if knowledge_question.strip() and UnifiedTask.KNOWLEDGE_QA in selected:
            overrides[UnifiedTask.KNOWLEDGE_QA] = knowledge_question.strip()
        source = RegionSource(interpretation_source)
        candidate = None if candidate_id in {None, ""} else int(candidate_id)
        frozen_item = None
        if selection.get("mode") == FROZEN_MODE:
            frozen_name = str(selection["frozen_name"])
            frozen_item = service.frozen[frozen_name].item(int(selection["frozen_ordinal"]))
        summary = service.runner.run(
            record=record,
            tasks=selected,
            instructions=overrides,
            user_mask=user_mask,
            candidate_region_id=candidate,
            region_interpretation_source=source,
            data_mode=str(selection.get("mode")),
            frozen_item=frozen_item,
        )
        task_choices = [item.task.value for item in summary.tasks]
        candidate_choices = list(
            service.runner.candidate_choices(sample_id, split=split)
        )
        status = (
            f"✅ Demo run `{summary.run_id}` published: "
            f"{sum(item.status == 'SUCCESS' for item in summary.tasks)}/{len(summary.tasks)} tasks succeeded. "
            "Engineering evidence only; not scientific acceptance."
        )
        if summary.sealed_test_accessed:
            status += " Test receipt recorded; blind/sealed property is false."
        return (
            summary.to_dict(),
            summary.to_dict(),
            gr.update(choices=task_choices, value=task_choices[0]),
            status,
            gr.update(choices=candidate_choices, value=None),
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
        return (
            _resolve_viewer_asset(run_root, preview["spatial_expert_inputs"]["full_optical"]),
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
                **viewer["execution_trace"],
                "failure": viewer.get("failure"),
            },
        )

    with gr.Blocks(title="OA-GroundRAG Unified Demo Workbench") as app:
        gr.Markdown(
            "# OA-GroundRAG Unified Demo Workbench\n"
            "**Read-only inference workbench.** Benchmark/Frozen assets、checkpoint、Adapter、Text Bank "
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
            with gr.Row():
                spatial_inputs = gr.JSON(label="Spatial Expert Inputs", open=True)
                formal_inputs = gr.JSON(label="MLLM Formal Grounded Inputs", open=True)

        with gr.Tab("B. Frozen Evaluation / Read Only"):
            gr.Markdown(
                "### Frozen Evaluation / Read Only\n"
                "Selection identity is immutable. Poor images or model failures are not replaceable. "
                "Any new inference is a Demo run (`selection_unchanged=true`, `formal_evaluation=false`).",
                elem_classes="readonly-boundary",
            )
            with gr.Row():
                frozen_name = gr.Dropdown(frozen_names, value=frozen_names[0], label="Frozen selection")
                frozen_ordinal = gr.Number(value=0, precision=0, minimum=0, maximum=99, label="ordinal (0-99)")
                frozen_variant = gr.Dropdown(
                    ["baseline_correct_mask"],
                    value="baseline_correct_mask",
                    label="counterfactual variant (read only)",
                )
                frozen_previous = gr.Button("上一条")
                frozen_next = gr.Button("下一条")
                frozen_load = gr.Button("载入只读样本", variant="primary")
            frozen_status = gr.Markdown()
            frozen_meta = gr.JSON(label="Frozen identity / status", open=True)
            with gr.Row():
                frozen_optical = gr.Image(label="Frozen Full RGB", type="filepath", interactive=False)
                frozen_reference = gr.Image(
                    label="Reference / Counterfactual / Audit Only (never runtime input)",
                    type="filepath",
                    image_mode="L",
                    interactive=False,
                )
                frozen_crop = gr.Image(
                    label="Frozen Context Crop / Audit Preview",
                    type="filepath",
                    interactive=False,
                )
            frozen_variants = gr.JSON(label="Counterfactual variants (read only)", open=False)

        with gr.Tab("C. Demo Gallery"):
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

        with gr.Tab("D–F. Task Runner / Result Viewer / Trace"):
            gr.Markdown(
                "### Task Routing\n"
                "Six explicit UnifiedTask only; Suite is preflighted in full and executed in a fixed capability order."
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
                    label="candidate ID (latest spatial result for same sample only)",
                )
            run_button = gr.Button("运行只读 Demo inference", variant="primary")
            run_status = gr.Markdown()
            run_summary = gr.JSON(label="Suite summary", open=True)
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

        frozen_outputs = [
            selection_state,
            frozen_ordinal,
            frozen_status,
            frozen_meta,
            frozen_optical,
            frozen_reference,
            frozen_crop,
            frozen_variants,
            frozen_variant,
            candidate_selector,
        ]
        frozen_load.click(fn=frozen_values, inputs=[frozen_name, frozen_ordinal], outputs=frozen_outputs, **private_event)
        frozen_previous.click(
            fn=lambda name, ordinal: frozen_values(name, int(ordinal) - 1),
            inputs=[frozen_name, frozen_ordinal],
            outputs=frozen_outputs,
            **private_event,
        )
        frozen_next.click(
            fn=lambda name, ordinal: frozen_values(name, int(ordinal) + 1),
            inputs=[frozen_name, frozen_ordinal],
            outputs=frozen_outputs,
            **private_event,
        )
        frozen_variant.change(
            fn=frozen_variant_values,
            inputs=[frozen_name, frozen_ordinal, frozen_variant],
            outputs=[frozen_optical, frozen_reference, frozen_crop],
            show_progress="hidden",
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
            outputs=[run_state, run_summary, result_task, run_status, candidate_selector],
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
    allowed = [str(resolved.demo_root), *(str(item.root) for item in resolved.frozen_evaluations)]
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
