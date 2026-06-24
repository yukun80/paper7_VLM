#!/usr/bin/env python3
"""Convert GeoHazard Qwen messages JSONL to LLaVA-style JSON.

The target format is compatible with Qwen-VL-Series-Finetune SFT scripts:
`image` is a path or a list of paths, and `conversations` contains human/gpt
turns with `<image>` tokens in the human prompt.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def content_items(message: dict[str, Any]) -> list[dict[str, Any]]:
    content = message.get("content")
    if not isinstance(content, list):
        raise ValueError("user message content must be a list")
    return content


def assistant_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(item.get("text", "")) for item in content if item.get("type") == "text").strip()
    return str(content)


def convert_row(row: dict[str, Any]) -> dict[str, Any]:
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError(f"{row.get('id')}: expected at least user and assistant messages")
    user = messages[0]
    assistant = messages[1]
    items = content_items(user)

    images = [str(item["image"]) for item in items if item.get("type") == "image" and item.get("image")]
    texts = [str(item.get("text", "")).strip() for item in items if item.get("type") == "text" and str(item.get("text", "")).strip()]
    if not images:
        raise ValueError(f"{row.get('id')}: no image content found")

    image_tokens = "\n".join("<image>" for _ in images)
    prompt_text = "\n".join(texts)
    human_value = f"{image_tokens}\n{prompt_text}" if prompt_text else image_tokens

    converted: dict[str, Any] = {
        "id": row["id"],
        "image": images[0] if len(images) == 1 else images,
        "conversations": [
            {"from": "human", "value": human_value},
            {"from": "gpt", "value": assistant_text(assistant)},
        ],
    }
    if row.get("task"):
        converted["task"] = row["task"]
    return converted


def convert_file(input_path: Path, output_path: Path, check_images: bool) -> None:
    rows = read_jsonl(input_path)
    converted: list[dict[str, Any]] = []
    missing_images: list[str] = []
    for row in rows:
        item = convert_row(row)
        image_value = item["image"]
        paths = image_value if isinstance(image_value, list) else [image_value]
        if check_images:
            for path in paths:
                if not Path(path).exists():
                    missing_images.append(path)
                    if len(missing_images) >= 20:
                        break
        converted.append(item)
        if len(missing_images) >= 20:
            break
    if missing_images:
        raise SystemExit("missing image paths:\n" + "\n".join(missing_images))
    write_json(output_path, converted)
    print(f"{input_path} -> {output_path} ({len(converted)} rows)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert qwen_sft_{split}.jsonl files to LLaVA JSON.")
    parser.add_argument("--out-dir", default="benchmark/geohazard_halluground_v2_full", help="Benchmark run directory.")
    parser.add_argument("--splits", default="train,val,test", help="Comma-separated split names to convert.")
    parser.add_argument("--check-images", action="store_true", help="Fail if referenced image paths do not exist.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    splits = [item.strip() for item in args.splits.split(",") if item.strip()]
    if not splits:
        raise SystemExit("--splits must contain at least one split")
    for split in splits:
        input_path = out_dir / f"qwen_sft_{split}.jsonl"
        output_path = out_dir / f"llava_{split}.json"
        if not input_path.exists():
            raise SystemExit(f"missing input file: {input_path}")
        convert_file(input_path, output_path, args.check_images)


if __name__ == "__main__":
    main()
