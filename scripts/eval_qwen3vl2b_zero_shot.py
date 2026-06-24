#!/usr/bin/env python3
"""Run zero-shot Qwen3-VL-2B inference and save raw responses.

This script intentionally stores raw model outputs only. JSON parsing and
metric computation should be implemented as a separate evaluation step.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image


def load_llava(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def prompt_from_qwen_row(row: dict[str, Any]) -> tuple[list[str], str, str]:
    user = row["messages"][0]
    assistant = row["messages"][1]
    content = user["content"]
    images = [item["image"] for item in content if item.get("type") == "image"]
    texts = [item["text"] for item in content if item.get("type") == "text"]
    return images, "\n".join(texts), assistant.get("content", "")


def prompt_from_llava_row(row: dict[str, Any]) -> tuple[list[str], str, str]:
    image_value = row["image"]
    images = image_value if isinstance(image_value, list) else [image_value]
    human = row["conversations"][0]["value"]
    prompt = human.replace("<image>", "").strip()
    target = row["conversations"][1].get("value", "")
    return images, prompt, target


def build_messages(images: list[str], prompt: str) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "image", "image": image} for image in images]
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def load_model(model_id: str):
    try:
        import torch
        from qwen_vl_utils import process_vision_info
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    except ImportError as exc:
        raise SystemExit(
            "Missing Qwen3-VL inference dependencies. Install torch, transformers from source, qwen-vl-utils, and accelerate."
        ) from exc

    model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, dtype="auto", device_map="auto")
    processor = AutoProcessor.from_pretrained(model_id)
    return torch, process_vision_info, model, processor


def generate_response(torch_mod, process_vision_info, model, processor, messages: list[dict[str, Any]], max_new_tokens: int) -> str:
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to(model.device)
    with torch_mod.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    return processor.batch_decode(generated_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Qwen3-VL-2B zero-shot inference on GeoHazard test data.")
    parser.add_argument("--input", default="benchmark/geohazard_halluground_v2_full/llava_test.json", help="llava_test.json or qwen_sft_test.jsonl.")
    parser.add_argument("--output", default="outputs/qwen3vl2b_zeroshot/raw_predictions.jsonl", help="Output JSONL path.")
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-2B-Instruct", help="Hugging Face model id or local path.")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit samples for a smoke run.")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Generation token budget.")
    parser.add_argument("--dry-run", action="store_true", help="Validate input/images and write prompts without loading the model.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if input_path.suffix == ".jsonl":
        raw_rows = read_jsonl(input_path)
        row_parser = prompt_from_qwen_row
    else:
        raw_rows = load_llava(input_path)
        row_parser = prompt_from_llava_row
    if args.max_samples is not None:
        raw_rows = raw_rows[: args.max_samples]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    runtime = None
    if not args.dry_run:
        runtime = load_model(args.model_id)

    with output_path.open("w", encoding="utf-8") as f:
        for row in raw_rows:
            images, prompt, target = row_parser(row)
            missing = [path for path in images if not Path(path).exists()]
            if missing:
                raise SystemExit(f"{row.get('id')}: missing image(s): {missing[:3]}")
            for path in images:
                with Image.open(path) as im:
                    im.verify()
            messages = build_messages(images, prompt)
            response = "" if args.dry_run else generate_response(*runtime, messages=messages, max_new_tokens=args.max_new_tokens)
            f.write(
                json.dumps(
                    {
                        "id": row.get("id"),
                        "images": images,
                        "prompt": prompt,
                        "target": target,
                        "response": response,
                        "dry_run": args.dry_run,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"wrote {len(raw_rows)} rows -> {output_path}")


if __name__ == "__main__":
    main()
