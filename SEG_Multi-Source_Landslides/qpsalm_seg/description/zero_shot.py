#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Native Qwen3-VL single-image zero-shot caption baseline for D-1."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image
import torch
from tqdm import tqdm

from qpsalm_seg.controllers import select_qwen_model_class, validate_qwen_model_dir
from qpsalm_seg.paths import resolve_project_path

from .metrics import bootstrap_mean_ci, caption_token_f1


def _rows(benchmark: str | Path, split: str, max_samples: int) -> list[dict[str, Any]]:
    root = resolve_project_path(benchmark) or Path(benchmark)
    values = [
        json.loads(line)
        for line in (root / f"indexes/{split}.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    values = [row for row in values if row.get("task_family") == "global_caption"]
    return values[:max_samples] if max_samples > 0 else values


@torch.no_grad()
def evaluate_zero_shot_global_caption(
    *,
    model_path: str | Path,
    benchmark: str | Path,
    split: str,
    output_dir: str | Path,
    device: torch.device,
    max_samples: int,
    max_new_tokens: int,
    seed: int,
    load_4bit: bool,
) -> dict[str, Any]:
    from transformers import AutoProcessor, BitsAndBytesConfig

    model_dir = validate_qwen_model_dir(model_path)
    processor = AutoProcessor.from_pretrained(str(model_dir), local_files_only=True)
    load_args: dict[str, Any] = {
        "local_files_only": True,
        "torch_dtype": torch.bfloat16 if device.type == "cuda" else torch.float32,
        "attn_implementation": "sdpa",
    }
    if load_4bit:
        load_args.update({
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            ),
            "device_map": {"": device.index or 0},
        })
    model = select_qwen_model_class().from_pretrained(str(model_dir), **load_args)
    if not load_4bit:
        model.to(device)
    model.eval()
    rows = _rows(benchmark, split, max_samples)
    outputs = []
    scores = []
    for row in tqdm(rows, desc="qwen-zero-shot-caption", unit="sample"):
        image_path = resolve_project_path(row["visual_ref"]["path"])
        if image_path is None:
            raise ValueError(f"无法解析图片: {row['visual_ref']['path']}")
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": str(row["instruction"])},
            ],
        }]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[prompt], images=[image], return_tensors="pt", padding=True)
        inputs = {name: value.to(device) if torch.is_tensor(value) else value for name, value in inputs.items()}
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        input_length = int(inputs["input_ids"].shape[1])
        prediction = processor.batch_decode(
            generated[:, input_length:], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
        references = [
            str(value["text"])
            for value in row.get("answers", [])
            if float(value.get("caption_quality_weight", 1.0)) > 0
        ]
        score = caption_token_f1(prediction, references)
        scores.append(score)
        outputs.append({
            "sample_id": row["sample_id"],
            "parent_sample_id": row["parent_sample_id"],
            "source_dataset": row["source_dataset"],
            "instruction": row["instruction"],
            "prediction": prediction,
            "references": references,
            "caption_token_f1": score,
        })
    target = resolve_project_path(output_dir) or Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    (target / "raw_generations.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in outputs), encoding="utf-8"
    )
    report = {
        "protocol": "qpsalm_qwen_zero_shot_global_caption_v1",
        "split": split,
        "num_samples": len(outputs),
        "caption_token_f1": sum(scores) / max(len(scores), 1),
        "bootstrap_ci": bootstrap_mean_ci(scores, seed=seed),
        "load_4bit": load_4bit,
        "region_capability_claimed": False,
    }
    (target / "eval_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report
