#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gradio 6.20 benchmark 对话式分割页面。

用途：浏览 val/test 样本、选择活动模态并使用原始或自定义指令重复推理。
运行方式：不作为独立入口；由 ``qpsalm-demo`` 调用。
"""

from __future__ import annotations

from typing import Any

import gradio as gr

from qpsalm_seg.inference import InferenceSession
from qpsalm_seg.presentation import presentation_arrays


def _choices(values) -> list[str]:
    return ["all", *sorted({str(value) for value in values})]


def build_demo(session: InferenceSession) -> gr.Blocks:
    catalog = list(session.catalog)
    if not catalog:
        raise RuntimeError(f"split={session.split} 没有可用样本")
    datasets = _choices(entry.dataset_name for entry in catalog)
    families = _choices(entry.family_combo for entry in catalog)
    tasks = _choices(entry.task_family for entry in catalog)
    initial = catalog[0]

    def filter_samples(dataset_name, family_name, task_name, query):
        entries = session.filter_catalog(
            dataset_name=dataset_name,
            family_combo_name=family_name,
            task_family=task_name,
            query=query,
        )
        values = [entry.sample_id for entry in entries]
        return gr.Dropdown(choices=values, value=values[0] if values else None)

    def load_sample(sample_id):
        if not sample_id:
            return "", "", gr.CheckboxGroup(choices=[], value=[]), {}
        defaults = session.sample_defaults(sample_id)
        modalities = list(defaults["modality_names"])
        metadata = {
            key: defaults[key] for key in (
                "sample_id", "parent_sample_id", "dataset_name", "task_family",
                "family_combo", "raw_combo", "instruction", "condition",
            )
        }
        return (
            defaults["instruction"],
            defaults["condition"],
            gr.CheckboxGroup(choices=modalities, value=modalities),
            metadata,
        )

    def run_prediction(sample_id, instruction, condition, modalities, threshold, history):
        if not sample_id:
            raise gr.Error("请先选择样本")
        defaults = session.sample_defaults(sample_id)
        task_override = str(instruction or "").strip()
        condition_override = str(condition or "").strip()
        result = session.predict(
            sample_id,
            instruction=(task_override if task_override != defaults["instruction"] else None),
            condition=(condition_override if condition_override != defaults["condition"] else None),
            active_modalities=modalities,
            threshold=float(threshold),
        )
        modality_gallery, prediction_gallery = presentation_arrays(result)
        note = "参考指标" if result.metrics_are_reference_only else "Benchmark GT 指标"
        assistant = (
            f"sample `{sample_id}`  |  Q{result.selected_query}  |  "
            f"Dice `{result.metrics.get('dice', 0):.4f}`  |  "
            f"IoU `{result.metrics.get('iou', 0):.4f}`  |  "
            f"mask `{result.diagnostics['mask_area']}` px  |  "
            f"{note}  |  `{result.latency_seconds:.3f}s`"
        )
        messages = list(history or [])
        messages.extend([
            {"role": "user", "content": task_override or defaults["instruction"]},
            {"role": "assistant", "content": assistant},
        ])
        diagnostics = {
            "checkpoint_step": result.checkpoint_step,
            "metrics": result.metrics,
            "metrics_are_reference_only": result.metrics_are_reference_only,
            "latency_seconds": result.latency_seconds,
            **result.diagnostics,
        }
        return modality_gallery, prediction_gallery, result.final_mask * 255, diagnostics, messages, messages

    with gr.Blocks(title="Multi-Source Qwen-PSALM-Seg") as demo:
        history_state = gr.State([])
        with gr.Row():
            with gr.Column(scale=1, min_width=300):
                dataset_filter = gr.Dropdown(datasets, value="all", label="Dataset")
                family_filter = gr.Dropdown(families, value="all", label="Modality family combo")
                task_filter = gr.Dropdown(tasks, value="all", label="Task family")
                search = gr.Textbox(label="Sample ID / instruction filter")
                sample = gr.Dropdown(
                    choices=[entry.sample_id for entry in catalog],
                    value=initial.sample_id,
                    label="Sample",
                    filterable=True,
                )
                instruction = gr.Textbox(value=initial.instruction, label="Instruction", lines=3)
                initial_defaults = session.sample_defaults(initial.sample_id)
                condition = gr.Textbox(value=initial_defaults["condition"], label="Condition", lines=2)
                modalities = gr.CheckboxGroup(
                    choices=list(initial.modality_names),
                    value=list(initial.modality_names),
                    label="Active modalities",
                )
                threshold = gr.Slider(0.05, 0.95, value=float(session.config.eval_threshold), step=0.05, label="Mask threshold")
                run = gr.Button("运行分割", variant="primary")
                metadata = gr.JSON(initial_defaults, label="Sample metadata")
            with gr.Column(scale=2):
                modality_gallery = gr.Gallery(label="Active modality views", columns=3, object_fit="contain")
                prediction_gallery = gr.Gallery(label="Segmentation result", columns=3, object_fit="contain")
                final_mask = gr.Image(label="Final binary mask", type="numpy", image_mode="L")
                diagnostics = gr.JSON(label="Diagnostics")
                chatbot = gr.Chatbot(label="Instruction history", layout="panel", height=300)

        filter_inputs = [dataset_filter, family_filter, task_filter, search]
        for component in filter_inputs:
            component.change(filter_samples, filter_inputs, sample)
        sample.change(load_sample, sample, [instruction, condition, modalities, metadata])
        run.click(
            run_prediction,
            [sample, instruction, condition, modalities, threshold, history_state],
            [modality_gallery, prediction_gallery, final_mask, diagnostics, chatbot, history_state],
            concurrency_limit=1,
        )
    return demo
