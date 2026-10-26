"""RS-GeneralDesc 单元/合成测试的小型真实形状 fixture。"""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image, ImageDraw

from oa_groundrag.data.rs_general.io import atomic_write_json


def make_image(
    path: Path,
    *,
    size: tuple[int, int] = (8, 6),
    color: tuple[int, int, int] = (60, 120, 180),
    orientation: int | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", size, color)
    if orientation is None:
        image.save(path)
    else:
        exif = Image.Exif()
        exif[274] = orientation
        image.save(path, exif=exif)


def mask_bytes(
    *,
    size: tuple[int, int] = (8, 6),
    positive: bool = True,
) -> bytes:
    from io import BytesIO

    array = np.zeros((size[1], size[0]), dtype=np.uint8)
    if positive:
        array[1:3, 2:5] = 255
    output = BytesIO()
    Image.fromarray(array).save(output, format="PNG")
    return output.getvalue()


def make_rsgpt(root: Path) -> None:
    make_image(root / "RSICap/images/cap.png", color=(60, 120, 180))
    make_image(root / "RSICap/images/unsafe.png", color=(61, 121, 181))
    make_image(root / "RSICap/images/unused.png", color=(62, 122, 182))
    make_image(root / "RSIEval/images/eval.png", color=(63, 123, 183))
    atomic_write_json(
        root / "RSICap/captions.json",
        {
            "annotations": [
                {
                    "image_id": "1",
                    "filename": "cap.png",
                    "text_input": "Describe the image.",
                    "text_output": "Several buildings are visible.",
                },
                {
                    "image_id": "2",
                    "filename": "unsafe.png",
                    "text_input": "Describe the image.",
                    "text_output": "This probably shows a dangerous risk.",
                },
            ]
        },
    )
    atomic_write_json(
        root / "RSIEval/annotations.json",
        {
            "annotations": [
                {
                    "filename": "eval.png",
                    "caption": "A road and trees are visible.",
                    "qa_pairs": [
                        {
                            "question": "Are trees visible?",
                            "answer": "Yes.",
                            "type": "presence",
                        },
                        {
                            "question": "Are trees visible?",
                            "answer": "Yes.",
                            "type": "presence",
                        },
                        {
                            "question": "What season is shown?",
                            "answer": "Summer.",
                            "type": "reasoning",
                        },
                        {
                            "question": "What color is the roof?",
                            "answer": "Blue.",
                            "type": "color",
                        },
                        {
                            "question": "How many buildings are visible?",
                            "answer": "Three.",
                            "type": "quantity",
                        },
                        {
                            "question": "What is the overall image attribute?",
                            "answer": "The image is clear.",
                            "type": "image",
                        },
                        {
                            "question": "Which area is larger?",
                            "answer": "The forest area is larger.",
                            "type": "area_comp",
                        },
                        {
                            "question": "What scene is visible?",
                            "answer": "An urban scene.",
                            "type": "scene",
                        },
                        {
                            "question": "What is the road orientation?",
                            "answer": "East-west.",
                            "type": "road_ori",
                        },
                    ],
                }
            ]
        },
    )


def make_mmrs(root: Path) -> None:
    image_ref = "caption/demo/images/a.png"
    image_ref_b = "caption/demo/images/b.png"
    vqa_ref = "VQA/demo/images/v.png"
    rsvg_ref = "RSVG/demo/images/r.png"
    make_image(root / image_ref, color=(70, 130, 190))
    make_image(root / image_ref_b, color=(73, 133, 193))
    make_image(root / vqa_ref, color=(71, 131, 191))
    make_image(root / rsvg_ref, color=(72, 132, 192))
    atomic_write_json(
        root / "json/caption/demo.json",
        [
            {
                "image": f"data/{image_ref}",
                "conversations": [
                    {"from": "human", "value": "Caption this image."},
                    {"from": "gpt", "value": "A road beside buildings."},
                    {"from": "human", "value": "Caption this image."},
                    {"from": "gpt", "value": "A road beside buildings."},
                    {"from": "human", "value": "Caption this image."},
                    {"from": "gpt", "value": "Buildings line a road."},
                ],
            },
            {
                "image": f"data/{image_ref_b}",
                "conversations": [
                    {"from": "human", "value": "Caption this image."},
                    {"from": "gpt", "value": "Fields surround a settlement."},
                    {"from": "human", "value": "Caption this image."},
                    {"from": "gpt", "value": "A settlement lies among fields."},
                ],
            },
        ],
    )
    atomic_write_json(
        root / "json/VQA/demo.json",
        [
            {
                "image": f"data/{vqa_ref}",
                "conversations": [
                    {"from": "human", "value": "Is water visible?"},
                    {"from": "gpt", "value": "No."},
                ],
            },
            {
                "image": f"data/{vqa_ref}",
                "conversations": [
                    {"from": "human", "value": "Is water visible?"},
                    {"from": "gpt", "value": "No."},
                ],
            },
        ],
    )
    atomic_write_json(
        root / "json/RSVG/rsvg_trainval.json",
        [
            {
                "image": f"data/{rsvg_ref}",
                "conversations": [
                    {
                        "from": "human",
                        "value": "Describe this region :[0.1,0.2,0.5,0.8]",
                    },
                    {"from": "gpt", "value": "A building."},
                    {"from": "human", "value": "Where is the building?"},
                    {"from": "gpt", "value": "[0.1,0.2,0.5,0.8]"},
                ],
            },
            {
                "image": f"data/{rsvg_ref}",
                "conversations": [
                    {
                        "from": "human",
                        "value": "Describe this region :[0.2,0.2,0.2,0.5]",
                    },
                    {"from": "gpt", "value": "A zero box."},
                ],
            },
        ],
    )
    (root / "json/total.json").write_text("[]\n", encoding="utf-8")


def make_disaster(root: Path) -> None:
    instruct = root / "DisasterM3_Instruct"
    bench = root / "DisasterM3_Bench"
    make_image(
        instruct / "train_images/pre.png",
        color=(80, 140, 200),
    )
    make_image(
        instruct / "train_images/post.png",
        color=(81, 141, 201),
    )
    make_image(
        instruct / "box_train_images/rel.png",
        color=(82, 142, 202),
    )
    relation_path = instruct / "box_train_images/rel.png"
    with Image.open(relation_path) as relation_handle:
        relation_image = relation_handle.convert("RGB")
    relation_draw = ImageDraw.Draw(relation_image)
    relation_draw.rectangle((0, 0, 2, 2), outline=(255, 0, 0), width=1)
    relation_draw.rectangle((4, 2, 6, 4), outline=(0, 0, 255), width=1)
    relation_image.save(relation_path)
    rows: list[dict[str, Any]] = [
        {
            "pre_image_path": "train_images\\pre.png",
            "post_image_path": "train_images\\post.png",
            "post_image_type": "Optical",
            "ground_truth": "Buildings and roads",
            "prompts": "Identify visible land cover.",
            "training_answer": "Buildings and roads are visible.",
            "task": "Disaster Bearing Bodies Recognition",
        },
        {
            "pre_image_path": "train_images\\pre.png",
            "post_image_path": "train_images\\post.png",
            "post_image_type": "Optical",
            "ground_truth": 3,
            "prompts": "How many intact buildings are visible?",
            "task": "Building Damage Counting",
        },
        {
            "pre_image_path": "train_images\\pre.png",
            "post_image_path": "train_images\\post.png",
            "post_image_type": "Optical",
            "ground_truth": "12%",
            "prompts": "What road area is intact?",
            "task": "Road Damage Counting",
        },
        {
            "pre_image_path": "train_images\\pre.png",
            "post_image_path": "train_images\\post.png",
            "post_image_type": "Optical",
            "ground_truth": (
                "DISASTER: Flood\n"
                "BUILDING: Roof changes are visible.\n"
                "ROAD: Roads remain visible.\n"
                "CONCLUSION: The flood caused risk."
            ),
            "prompts": ["Describe visible changes."],
            "task": "disaster caption",
        },
        {
            "pre_image_path": "train_images\\pre.png",
            "post_image_path": "train_images\\post.png",
            "image_type": "Optical",
            "ground_truth": "masks\\flooding\\valid.png",
            "prompts": "Outline flooded regions.",
            "cls_description": "Flooded regions.",
            "task": "Referring Expression Segmentation",
        },
        {
            "pre_image_path": "train_images\\pre.png",
            "post_image_path": "train_images\\post.png",
            "image_type": "Optical",
            "ground_truth": "masks\\flooding\\empty.png",
            "prompts": "Outline flooded regions.",
            "cls_description": "Flooded regions.",
            "task": "Referring Expression Segmentation",
        },
        {
            "pre_image_path": "train_images\\pre.png",
            "post_image_path": "train_images\\post.png",
            "post_image_type": "Optical",
            "ground_truth": "Residential and forest",
            "prompts": "Identify land use before the event.",
            "task": "Disaster Scene Recognition",
        },
        {
            "image_path": "train_images\\box_train_images\\rel.png",
            "objects": {
                "1": [1, 3, 2, 2],
                "2": [0, 0, 3, 3],
                "3": [2, 1, 2, 2],
                "4": [4, 2, 3, 3],
            },
            "ground_truth": "Object 3 is left of object 4.",
            "prompts": "Describe the spatial relation.",
            "image_type": "Optical",
            "task": "relational reasoning",
        },
        {
            "pre_image_path": "train_images\\pre.png",
            "post_image_path": "train_images\\post.png",
            "post_image_type": "SAR",
            "ground_truth": "SAR row",
            "prompts": "Describe.",
            "task": "Disaster Scene Recognition",
        },
        {
            "pre_image_path": "train_images\\pre.png",
            "post_image_path": "train_images\\post.png",
            "post_image_type": "Optical",
            "ground_truth": "flood",
            "prompts": "What disaster?",
            "task": "Disaster Type Recognition",
        },
    ]
    atomic_write_json(instruct / "train_release.json", rows)
    atomic_write_json(bench / "benchmark_release.json", [])
    (bench / "test_images").mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(instruct / "masks.zip", "w") as archive:
        archive.writestr("masks/flooding/valid.png", mask_bytes(positive=True))
        archive.writestr("masks/flooding/empty.png", mask_bytes(positive=False))


def make_all_sources(base: Path) -> dict[str, Path]:
    roots = {
        "rsgpt": base / "rsgpt",
        "mmrs1m": base / "mmrs",
        "disasterm3": base / "disaster",
    }
    make_rsgpt(roots["rsgpt"])
    make_mmrs(roots["mmrs1m"])
    make_disaster(roots["disasterm3"])
    return roots


def build_config_dict(
    roots: dict[str, Path],
    output_root: Path,
    *,
    workers: int = 1,
    profile: str = "smoke",
    max_assets: int | None = 64,
    seed: int = 7,
    validation_percent: int = 5,
) -> dict[str, Any]:
    return {
        "schema_version": "rs_generaldesc.build_config.v1",
        "profile": profile,
        "seed": seed,
        "output_root": str(output_root),
        "sources": {
            "rsgpt": {"enabled": True, "root": str(roots["rsgpt"])},
            "mmrs1m": {"enabled": True, "root": str(roots["mmrs1m"])},
            "disasterm3": {
                "enabled": True,
                "root": str(roots["disasterm3"]),
            },
        },
        "external_split": {
            "version": "external_parent_split.v1",
            "validation_percent": validation_percent,
        },
        "text_policy": {
            "version": "fixture.v1",
            "forbidden_patterns": [
                r"\b(?:risk|danger|probably|likely)\b",
            ],
            "reasoning_forbidden_patterns": [
                r"\b(?:season|summer|winter|used for)\b",
            ],
        },
        "asset_policy": {
            "image_format": "png",
            "image_quality": 95,
            "jpeg_subsampling": 0,
            "target_size": None,
        },
        "sharding": {"records_per_shard": 8},
        "workers": workers,
        "limits": {
            "max_parents": 32,
            "max_records": 40,
            "max_assets": max_assets,
            "max_copied_bytes": 50_000_000,
            "max_validation_candidates_per_task": 10,
            "per_task": {},
        },
        "deletion_allowlist": False,
    }


def write_build_config(
    path: Path,
    roots: dict[str, Path],
    output_root: Path,
    **kwargs: Any,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            build_config_dict(roots, output_root, **kwargs),
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    return path


def write_export_config(
    path: Path,
    benchmark_root: Path,
    output_root: Path,
    *,
    purpose: str = "training",
    roles: list[str] | None = None,
    task_families: list[str] | None = None,
) -> Path:
    default_tasks = [
        "global_caption",
        "visual_qa",
        "bbox_region_caption",
        "scene_understanding",
        "object_count",
        "spatial_relation",
        "visible_change_report",
    ]
    value = {
        "schema_version": "rs_generaldesc.qwen_export.v1",
        "profile": "description_multitask.v1",
        "benchmark_root": str(benchmark_root),
        "output_root": str(output_root),
        "seed": 7,
        "purpose": purpose,
        "roles": roles if roles is not None else ["external_train"],
        "task_families": task_families or default_tasks,
        "template_version": "qwen3vl_messages.v2",
    }
    path.write_text(
        yaml.safe_dump(value, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return path
