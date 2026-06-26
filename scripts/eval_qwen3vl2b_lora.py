#!/usr/bin/env python3
"""加载 Qwen3-VL-2B 基座模型和 LoRA adapter，对测试集做推理。

本脚本只保存 raw predictions JSONL，不在这里计算指标。后续分类、
grounding 的 JSON 解析和指标统计应放在单独脚本中完成。

Dry-run：只检查输入和图像路径，不加载模型。

```bash
python scripts/eval_qwen3vl2b_lora.py \
  --input benchmark/geohazard_halluground_v2_full/llava_test.json \
  --output outputs/qwen3vl2b_stage1_lora_eval/dry_run.jsonl \
  --base-model /home/yukun/codes/paper7_VLM/models/Qwen3-VL-2B-Instruct \
  --adapter-path outputs/qwen3vl2b_stage1_lora \
  --max-samples 20 \
  --dry-run
```

20 条 LoRA 推理 smoke：

```bash
python scripts/eval_qwen3vl2b_lora.py \
  --input benchmark/geohazard_halluground_v2_full/llava_test.json \
  --output outputs/qwen3vl2b_stage1_lora_eval/raw_predictions_smoke.jsonl \
  --base-model /home/yukun/codes/paper7_VLM/models/Qwen3-VL-2B-Instruct \
  --adapter-path outputs/qwen3vl2b_stage1_lora \
  --max-samples 20 \
  --max-new-tokens 256
```

如果只想在终端直接看模型回答，额外加入：

```bash
  --print-responses
```

如果希望同步保存可视化结果，额外加入：

```bash
  --visualize --vis-dir inputs/test1/visualizations
```
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


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


def prompt_from_qwen_row(row: dict[str, Any]) -> tuple[list[str], str, str, str | None]:
    # 处理 qwen_sft_*.jsonl：图像和文本 prompt 都在 messages[0]["content"] 中。
    user = row["messages"][0]
    assistant = row["messages"][1]
    content = user["content"]
    images = [item["image"] for item in content if item.get("type") == "image"]
    texts = [item["text"] for item in content if item.get("type") == "text"]
    return images, "\n".join(texts), assistant.get("content", ""), row.get("task")


def prompt_from_llava_row(row: dict[str, Any]) -> tuple[list[str], str, str, str | None]:
    # 处理 LLaVA JSON list：image 字段可为单图或多图，human 文本中包含 <image> 占位符。
    image_value = row["image"]
    images = image_value if isinstance(image_value, list) else [image_value]
    human = row["conversations"][0]["value"]
    prompt = human.replace("<image>", "").strip()
    target = row["conversations"][1].get("value", "")
    return images, prompt, target, row.get("task")


def task_from_id(row_id: str | None) -> str | None:
    if not row_id or "::" not in row_id:
        return None
    return row_id.rsplit("::", 1)[1]


def safe_filename(value: str | None) -> str:
    name = value or "sample"
    return re.sub(r"[^0-9A-Za-z._-]+", "_", name).strip("_") or "sample"


def parse_response_json(response: str) -> tuple[dict[str, Any] | None, str | None]:
    text = response.strip()
    if not text:
        return None, "empty_response"
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None, None if isinstance(parsed, dict) else "json_not_object"
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                parsed = json.loads(text[start : end + 1])
                return parsed if isinstance(parsed, dict) else None, None if isinstance(parsed, dict) else "json_not_object"
            except json.JSONDecodeError:
                pass
    return None, "parse_failed"


def bbox_1000_to_pixels(bbox: Any, width: int, height: int) -> tuple[int, int, int, int] | None:
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(item) for item in bbox]
    except (TypeError, ValueError):
        return None
    x1 = max(0, min(width, round(x1 / 1000.0 * width)))
    y1 = max(0, min(height, round(y1 / 1000.0 * height)))
    x2 = max(0, min(width, round(x2 / 1000.0 * width)))
    y2 = max(0, min(height, round(y2 / 1000.0 * height)))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def visible_bbox_for_drawing(
    pixel_bbox: tuple[int, int, int, int],
    width: int,
    height: int,
    line_width: int,
) -> tuple[int, int, int, int] | None:
    # 仅做最小可见性约束：贴到图像边界时，将红线向图内移动少量像素。
    max_x = max(0, width - 1)
    max_y = max(0, height - 1)
    x1, y1, x2, y2 = pixel_bbox
    cx1 = max(0, min(max_x, x1))
    cy1 = max(0, min(max_y, y1))
    cx2 = max(0, min(max_x, x2))
    cy2 = max(0, min(max_y, y2))
    if cx2 <= cx1 or cy2 <= cy1:
        return None

    inset = max(1, line_width // 2)
    adjusted = (
        inset if x1 <= 0 else cx1,
        inset if y1 <= 0 else cy1,
        max_x - inset if x2 >= width else cx2,
        max_y - inset if y2 >= height else cy2,
    )
    ax1, ay1, ax2, ay2 = adjusted

    # 极窄框内收后可能失效，此时退回到 clamp 后的坐标。
    if ax2 <= ax1 or ay2 <= ay1:
        return cx1, cy1, cx2, cy2
    return adjusted


def draw_text_box(draw: ImageDraw.ImageDraw, lines: list[str]) -> None:
    text = "\n".join(line for line in lines if line)
    if not text:
        return
    bbox = draw.multiline_textbbox((8, 8), text)
    pad = 5
    bg = (255, 255, 255)
    fg = (0, 0, 0)
    draw.rectangle((bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad), fill=bg, outline=(220, 0, 0))
    draw.multiline_text((8, 8), text, fill=fg, spacing=3)


def visualize_prediction(row_id: str | None, task: str | None, images: list[str], response: str, vis_dir: Path) -> Path | None:
    # 第一版只处理单图样本；多图 pre/post 可视化后续再扩展。
    if len(images) != 1:
        print(f"skip visualization for multi-image sample: {row_id}")
        return None

    vis_dir.mkdir(parents=True, exist_ok=True)
    image_path = Path(images[0])
    parsed, parse_error = parse_response_json(response)

    with Image.open(image_path) as im:
        canvas = im.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    width, height = canvas.size

    status = parse_error or "ok"
    hazard_present = parsed.get("hazard_present") if parsed else None
    hazard_type = parsed.get("hazard_type") if parsed else None
    bbox = parsed.get("bbox_0_1000") if parsed else None
    pixel_bbox = bbox_1000_to_pixels(bbox, width, height)

    if pixel_bbox:
        visible_bbox = visible_bbox_for_drawing(pixel_bbox, width, height, line_width=3)
        if visible_bbox:
            draw.rectangle(visible_bbox, outline=(255, 0, 0), width=3)
            status = "bbox_drawn"
        else:
            status = "invalid bbox"
    elif bbox not in (None, []):
        status = "invalid bbox"
    elif not parse_error:
        status = "no bbox"

    draw_text_box(
        draw,
        [
            f"id: {row_id}",
            f"task: {task}",
            f"hazard_present: {hazard_present}",
            f"hazard_type: {hazard_type}",
            f"status: {status}",
        ],
    )

    out_path = vis_dir / f"{safe_filename(row_id)}.png"
    canvas.save(out_path)
    return out_path


def build_messages(images: list[str], prompt: str) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "image", "image": image} for image in images]
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def validate_adapter(adapter_path: Path) -> None:
    # LoRA adapter 至少需要这两个文件；缺任意一个都不能正确加载微调权重。
    missing = [
        str(path)
        for path in [adapter_path / "adapter_config.json", adapter_path / "adapter_model.safetensors"]
        if not path.exists()
    ]
    if missing:
        raise SystemExit("missing LoRA adapter file(s):\n" + "\n".join(missing))


def load_lora_model(base_model: str, adapter_path: str, merge_lora: bool):
    try:
        import torch
        from peft import PeftModel
        from qwen_vl_utils import process_vision_info
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    except ImportError as exc:
        raise SystemExit(
            "Missing LoRA inference dependencies. Install torch, transformers, peft, qwen-vl-utils, and accelerate."
        ) from exc

    # 先加载 Qwen3-VL 基座模型，再挂载训练得到的 LoRA adapter。
    model = Qwen3VLForConditionalGeneration.from_pretrained(base_model, dtype="auto", device_map="auto")
    model = PeftModel.from_pretrained(model, adapter_path)
    if merge_lora:
        # 仅用于推理时临时合并；不会改写磁盘上的 adapter 文件。
        model = model.merge_and_unload()
    model.eval()
    processor = AutoProcessor.from_pretrained(base_model)
    return torch, process_vision_info, model, processor


def generate_response(torch_mod, process_vision_info, model, processor, messages: list[dict[str, Any]], max_new_tokens: int) -> str:
    # Qwen processor 会把图像路径和文本 prompt 统一转换成模型可接收的张量输入。
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to(model.device)
    with torch_mod.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    return processor.batch_decode(generated_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Qwen3-VL-2B LoRA inference on GeoHazard test data.")
    parser.add_argument("--input", default="benchmark/geohazard_halluground_v2_full/llava_test.json", help="llava_test.json or qwen_sft_test.jsonl.")
    parser.add_argument("--output", default="outputs/qwen3vl2b_stage1_lora_eval/raw_predictions.jsonl", help="Output JSONL path.")
    parser.add_argument("--base-model", default="/home/yukun/codes/paper7_VLM/models/Qwen3-VL-2B-Instruct", help="Base Qwen3-VL model path or Hugging Face id.")
    parser.add_argument("--adapter-path", default="outputs/qwen3vl2b_stage1_lora", help="LoRA adapter directory.")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit samples for a smoke run.")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Generation token budget.")
    parser.add_argument("--merge-lora", action="store_true", help="Merge LoRA into the base model after loading.")
    parser.add_argument("--print-responses", action="store_true", help="Print each model response to stdout while saving JSONL.")
    parser.add_argument("--visualize", action="store_true", help="Save visualization PNGs for single-image predictions.")
    parser.add_argument("--vis-dir", default=None, help="Visualization output directory. Defaults to OUTPUT parent / visualizations.")
    parser.add_argument("--dry-run", action="store_true", help="Validate input/images and write prompts without loading the model.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    adapter_path = Path(args.adapter_path)
    validate_adapter(adapter_path)

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
    vis_dir = Path(args.vis_dir) if args.vis_dir else output_path.parent / "visualizations"

    runtime = None
    if not args.dry_run:
        runtime = load_lora_model(args.base_model, args.adapter_path, args.merge_lora)

    with output_path.open("w", encoding="utf-8") as f:
        for row in raw_rows:
            images, prompt, target, task = row_parser(row)
            # dry-run 同样会检查图像文件是否存在且可打开，但不会加载模型。
            missing = [path for path in images if not Path(path).exists()]
            if missing:
                raise SystemExit(f"{row.get('id')}: missing image(s): {missing[:3]}")
            for path in images:
                with Image.open(path) as im:
                    im.verify()
            messages = build_messages(images, prompt)
            response = "" if args.dry_run else generate_response(*runtime, messages=messages, max_new_tokens=args.max_new_tokens)
            row_task = task or task_from_id(row.get("id"))
            if args.print_responses:
                print(f"\n== {row.get('id')} ==")
                print(response if response else "[empty response]")
            if args.visualize:
                vis_path = visualize_prediction(row.get("id"), row_task, images, response, vis_dir)
                if vis_path:
                    print(f"visualized -> {vis_path}")
            f.write(
                json.dumps(
                    {
                        "id": row.get("id"),
                        "task": row_task,
                        "images": images,
                        "prompt": prompt,
                        "target": target,
                        "response": response,
                        "base_model": args.base_model,
                        "adapter_path": args.adapter_path,
                        "dry_run": args.dry_run,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"wrote {len(raw_rows)} rows -> {output_path}")


if __name__ == "__main__":
    main()
