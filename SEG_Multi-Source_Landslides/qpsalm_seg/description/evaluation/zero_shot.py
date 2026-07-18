#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Native Qwen3-VL single-image zero-shot caption baseline for D-1."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image
import torch
from tqdm import tqdm

from qpsalm_seg.controllers import select_qwen_model_class, validate_qwen_model_dir
from qpsalm_seg.paths import resolve_project_path

from ..protocols.io import (
    atomic_write_json,
    atomic_write_jsonl,
    canonical_sha256,
    sha256_file as _sha256_file,
    strict_json_loads,
)
from .metrics import bootstrap_mean_ci, caption_token_f1


ZERO_SHOT_PROTOCOL = (
    "qpsalm_qwen_zero_shot_global_caption_v4_model_bytes_bound"
)
ZERO_SHOT_INPUT_PROTOCOL = (
    "qpsalm_d_minus_one_zero_shot_input_binding_v2_materialized_image"
)
ZERO_SHOT_MODEL_IDENTITY_PROTOCOL = (
    "qpsalm_zero_shot_model_identity_v1_weights_tokenizer_bound"
)
DESCRIPTION_BUILDER_VERSION = "description_benchmark_m1_v4_answer_trace"


def build_zero_shot_input_audit(
    benchmark: str | Path, split: str, max_samples: int, seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = resolve_project_path(benchmark) or Path(benchmark)
    index_path = root / f"indexes/{split}.jsonl"
    report_path = root / "reports/validation_report.json"
    if not index_path.is_file() or not report_path.is_file():
        raise FileNotFoundError(
            f"zero-shot 缺少 Description index/report: {index_path}, {report_path}"
        )
    validation = strict_json_loads(report_path.read_text(encoding="utf-8"))
    if (
        validation.get("builder_version") != DESCRIPTION_BUILDER_VERSION
        or validation.get("errors")
        or int(
            validation.get(
                "verified_perceptual_duplicate_cross_split_groups", -1
            )
        ) != 0
    ):
        raise RuntimeError(
            "zero-shot baseline 要求 engineering-valid Description M1.1 v4，"
            "且 verified cross-split cluster 必须为零"
        )
    values = [
        strict_json_loads(line)
        for line in index_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    values = [row for row in values if row.get("task_family") == "global_caption"]
    if max_samples > 0:
        values = sorted(
            values,
            key=lambda row: hashlib.sha256(
                f"{seed}:d-minus-one-zero-shot:{row['sample_id']}".encode()
            ).hexdigest(),
        )[:max_samples]
    if not values:
        raise RuntimeError("zero-shot global-caption population 为空")
    sample_ids = [str(row.get("sample_id") or "") for row in values]
    if (
        any(not sample_id for sample_id in sample_ids)
        or len(sample_ids) != len(set(sample_ids))
        or any(str(row.get("split") or "") != str(split) for row in values)
    ):
        raise RuntimeError("zero-shot population sample_id/split 非法")
    materialized_root = (root / "data").resolve(strict=False)
    identities = []
    image_identities = []
    for row in values:
        visual = dict(row.get("visual_ref") or {})
        visual_ref = str(visual.get("path") or "")
        expected_sha256 = str(visual.get("sha256") or "")
        image_path = resolve_project_path(visual_ref)
        if image_path is None or not image_path.is_file():
            raise FileNotFoundError(
                f"zero-shot materialized image 不存在: {visual_ref}"
            )
        try:
            image_path.resolve(strict=False).relative_to(materialized_root)
        except ValueError as exc:
            raise RuntimeError(
                f"zero-shot image 不属于当前 M1.1 data/: {visual_ref}"
            ) from exc
        observed_sha256 = _sha256_file(image_path)
        if (
            visual.get("storage_mode") != "materialized_copy"
            or len(expected_sha256) != 64
            or observed_sha256 != expected_sha256
        ):
            raise RuntimeError(
                f"zero-shot materialized image SHA/storage 非法: {visual_ref}"
            )
        image_identity = {
            "visual_ref": visual_ref,
            "visual_sha256": observed_sha256,
        }
        image_identities.append(image_identity)
        identities.append({
            "sample_id": str(row["sample_id"]),
            "parent_sample_id": str(row["parent_sample_id"]),
            "source_dataset": str(row["source_dataset"]),
            "instruction_sha256": hashlib.sha256(
                str(row["instruction"]).encode("utf-8")
            ).hexdigest(),
            "answer_sha256": sorted(
                hashlib.sha256(
                    str(answer.get("text") or "").encode("utf-8")
                ).hexdigest()
                for answer in row.get("answers", [])
                if float(answer.get("caption_quality_weight", 1.0)) > 0
            ),
            **image_identity,
        })
    encoded = json.dumps(
        identities,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return values, {
        "protocol": ZERO_SHOT_INPUT_PROTOCOL,
        "benchmark_root": str(root.resolve(strict=False)),
        "builder_version": validation.get("builder_version"),
        "index": str(index_path.resolve(strict=False)),
        "index_sha256": _sha256_file(index_path),
        "validation_report": str(report_path.resolve(strict=False)),
        "validation_report_sha256": _sha256_file(report_path),
        "split": str(split),
        "requested_max_samples": int(max_samples),
        "selected_samples": len(values),
        "population_sha256": hashlib.sha256(encoded).hexdigest(),
        "materialized_images": len(image_identities),
        "materialized_image_population_sha256": hashlib.sha256(json.dumps(
            image_identities,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")).hexdigest(),
        "sampling_seed": int(seed),
        "sampling_policy": "sha256_ranked_global_caption_v1",
    }


def build_zero_shot_model_identity(model_dir: Path) -> dict[str, Any]:
    """Bind every local file that can change Qwen loading or generation."""
    accepted_suffixes = {
        ".bin", ".jinja", ".json", ".model", ".safetensors",
        ".tiktoken", ".txt",
    }
    paths = sorted(
        path for path in model_dir.iterdir()
        if path.is_file() and path.suffix.casefold() in accepted_suffixes
    )
    files = {
        path.name: {
            "bytes": int(path.stat().st_size),
            "sha256": _sha256_file(path),
        }
        for path in paths
    }
    weight_files = sorted(
        name for name in files
        if Path(name).suffix.casefold() in {".bin", ".safetensors"}
    )
    if "config.json" not in files:
        raise FileNotFoundError(f"Qwen model 缺少 config.json: {model_dir}")
    if not weight_files:
        raise FileNotFoundError(f"Qwen model 缺少本地 weight 文件: {model_dir}")
    return {
        "protocol": ZERO_SHOT_MODEL_IDENTITY_PROTOCOL,
        "model_dir": str(model_dir.resolve(strict=False)),
        "files": files,
        "weight_files": weight_files,
        "snapshot_sha256": canonical_sha256(files),
    }


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
    model_dir = validate_qwen_model_dir(model_path)
    rows, input_audit = build_zero_shot_input_audit(
        benchmark, split, max_samples, seed
    )
    model_audit = build_zero_shot_model_identity(model_dir)
    from transformers import AutoProcessor, BitsAndBytesConfig

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
    atomic_write_jsonl(target / "raw_generations.jsonl", outputs)
    generation_path = target / "raw_generations.jsonl"
    nonempty = sum(bool(str(row["prediction"]).strip()) for row in outputs)
    checks = {
        "description_input_valid": input_audit["builder_version"]
        == DESCRIPTION_BUILDER_VERSION,
        "materialized_images_bound": input_audit["materialized_images"]
        == len(rows),
        "all_selected_samples_generated": len(outputs)
        == int(input_audit["selected_samples"]),
        "all_predictions_nonempty": nonempty == len(outputs),
        "no_region_capability_claim": True,
        "model_bytes_bound": bool(model_audit["snapshot_sha256"]),
    }
    errors = [name for name, passed in checks.items() if not passed]
    report = {
        "protocol": ZERO_SHOT_PROTOCOL,
        "status": "engineering-valid" if not errors else "engineering-invalid",
        "split": split,
        "num_samples": len(outputs),
        "num_nonempty_predictions": nonempty,
        "caption_token_f1": sum(scores) / max(len(scores), 1),
        "bootstrap_ci": bootstrap_mean_ci(scores, seed=seed),
        "statistics_seed": int(seed),
        "load_4bit": load_4bit,
        "region_capability_claimed": False,
        "input_audit": input_audit,
        "model_audit": model_audit,
        "raw_generations": str(generation_path.resolve(strict=False)),
        "raw_generations_sha256": _sha256_file(generation_path),
        "checks": checks,
        "errors": errors,
    }
    atomic_write_json(target / "eval_report.json", report)
    return report
