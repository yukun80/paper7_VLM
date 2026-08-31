"""Stage 4 单专家工作流的本地 Qwen3-VL-8B 一次性草稿生成。"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from oa_groundrag.artifacts.identity import (
    canonical_json,
    sha256_file,
    sha256_text,
)
from oa_groundrag.artifacts.io import (
    atomic_write_json,
    first_symlink_component,
)
from oa_groundrag.data.rs_general.io import read_json
from oa_groundrag.vlm.config import (
    AdaptationSection,
    ModelSection,
    _load_yaml,
)
from oa_groundrag.grounding.contracts import MaskMode
from oa_groundrag.vlm.data import DescriptionSample
from oa_groundrag.vlm.errors import VLMError
from oa_groundrag.vlm.backends.qwen3_vl.model import Qwen3VLModelAdapter
from oa_groundrag.grounding.outputs import (
    assess_region_draft_quality,
    parse_region_model_output,
)
from oa_groundrag.vlm.processing import DescriptionCollator
from oa_groundrag.vlm.backends.qwen3_vl.processing import (
    Qwen3VLProcessorAdapter,
)

from ..contracts import fail
from .project import (
    DRAFT_MODEL_REPOSITORY,
    DRAFT_MODEL_REVISION,
    DRAFT_CONFIG_SCHEMA,
    MODEL_DRAFT_FAILURE_SCHEMA,
    MODEL_DRAFT_RUN_SCHEMA,
    MODEL_DRAFT_SCHEMA,
    AnnotationIntendedUse,
    append_model_draft,
    build_annotation_draft_messages,
    load_annotation_asset,
    load_annotation_project,
    load_model_drafts,
    register_draft_run,
)


QWEN_REPOSITORY = DRAFT_MODEL_REPOSITORY
QWEN_REVISION = DRAFT_MODEL_REVISION


@dataclass(frozen=True)
class LocalDraftConfig:
    config_path: Path
    config_file_sha256: str
    semantic_sha256: str
    model_path: Path
    processor_path: Path
    repository: str
    revision: str
    local_files_only: bool
    trust_remote_code: bool
    dtype: str
    attn_implementation: str
    min_pixels: int
    max_pixels: int
    max_images: int
    max_input_tokens: int
    max_new_tokens: int
    do_sample: bool
    temperature: float
    top_p: float
    seed: int

    def semantic_dict(self) -> dict[str, Any]:
        return {
            "schema_version": DRAFT_CONFIG_SCHEMA,
            "model": {
                "path": str(self.model_path),
                "processor_path": str(self.processor_path),
                "repository": self.repository,
                "revision": self.revision,
                "local_files_only": self.local_files_only,
                "trust_remote_code": self.trust_remote_code,
                "dtype": self.dtype,
                "attn_implementation": self.attn_implementation,
            },
            "processor": {
                "min_pixels": self.min_pixels,
                "max_pixels": self.max_pixels,
                "max_images": self.max_images,
                "max_input_tokens": self.max_input_tokens,
            },
            "generation": {
                "max_new_tokens": self.max_new_tokens,
                "do_sample": self.do_sample,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "seed": self.seed,
            },
        }


def _mapping(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        fail("CONFIG_INVALID", f"{location} 必须是字符串键对象")
    return dict(value)


def _exact(value: Mapping[str, Any], fields: Sequence[str], location: str) -> None:
    if set(value) != set(fields):
        fail(
            "CONFIG_INVALID",
            f"{location} 字段不匹配",
            details={
                "missing": sorted(set(fields) - set(value)),
                "unknown": sorted(set(value) - set(fields)),
            },
        )


def _positive_int(value: Any, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        fail("CONFIG_INVALID", f"{location} 必须是正整数")
    return value


def _boolean(value: Any, location: str) -> bool:
    if not isinstance(value, bool):
        fail("CONFIG_INVALID", f"{location} 必须是布尔值")
    return value


def _number(value: Any, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        fail("CONFIG_INVALID", f"{location} 必须是有限数值")
    result = float(value)
    if not math.isfinite(result):
        fail("CONFIG_INVALID", f"{location} 必须是有限数值")
    return result


def _text(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        fail("CONFIG_INVALID", f"{location} 必须是非空字符串")
    return value.strip()


def _absolute(base: Path, value: Any, location: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        fail("CONFIG_INVALID", f"{location} 必须是路径字符串")
    path = Path(value)
    return Path(os.path.abspath(path if path.is_absolute() else base / path))


def load_local_draft_config(path: Path | str) -> LocalDraftConfig:
    """严格加载已预注册的本地 8B 草稿配置；不触发模型下载。"""

    config_path = Path(os.path.abspath(Path(path)))
    if (
        first_symlink_component(config_path) is not None
        or not config_path.is_file()
        or config_path.is_symlink()
        or config_path.stat().st_nlink != 1
    ):
        fail("CONFIG_INVALID", f"draft config 必须是无 symlink 的普通文件：{config_path}")
    raw = _mapping(_load_yaml(config_path), "$ draft config")
    _exact(raw, ("schema_version", "model", "processor", "generation"), "$")
    if raw["schema_version"] != DRAFT_CONFIG_SCHEMA:
        fail("CONFIG_INVALID", f"仅支持 {DRAFT_CONFIG_SCHEMA}")
    model = _mapping(raw["model"], "$.model")
    _exact(
        model,
        (
            "path", "processor_path", "repository", "revision", "local_files_only",
            "trust_remote_code", "dtype", "attn_implementation",
        ),
        "$.model",
    )
    processor = _mapping(raw["processor"], "$.processor")
    _exact(processor, ("min_pixels", "max_pixels", "max_images", "max_input_tokens"), "$.processor")
    generation = _mapping(raw["generation"], "$.generation")
    _exact(generation, ("max_new_tokens", "do_sample", "temperature", "top_p", "seed"), "$.generation")
    values = LocalDraftConfig(
        config_path=config_path,
        config_file_sha256=sha256_file(config_path),
        semantic_sha256="",
        model_path=_absolute(config_path.parent, model["path"], "$.model.path"),
        processor_path=_absolute(config_path.parent, model["processor_path"], "$.model.processor_path"),
        repository=_text(model["repository"], "$.model.repository"),
        revision=_text(model["revision"], "$.model.revision"),
        local_files_only=_boolean(model["local_files_only"], "$.model.local_files_only"),
        trust_remote_code=_boolean(model["trust_remote_code"], "$.model.trust_remote_code"),
        dtype=_text(model["dtype"], "$.model.dtype"),
        attn_implementation=_text(model["attn_implementation"], "$.model.attn_implementation"),
        min_pixels=_positive_int(processor["min_pixels"], "$.processor.min_pixels"),
        max_pixels=_positive_int(processor["max_pixels"], "$.processor.max_pixels"),
        max_images=_positive_int(processor["max_images"], "$.processor.max_images"),
        max_input_tokens=_positive_int(processor["max_input_tokens"], "$.processor.max_input_tokens"),
        max_new_tokens=_positive_int(generation["max_new_tokens"], "$.generation.max_new_tokens"),
        do_sample=_boolean(generation["do_sample"], "$.generation.do_sample"),
        temperature=_number(generation["temperature"], "$.generation.temperature"),
        top_p=_number(generation["top_p"], "$.generation.top_p"),
        seed=_positive_int(generation["seed"], "$.generation.seed"),
    )
    if (
        values.repository != QWEN_REPOSITORY
        or values.revision != QWEN_REVISION
        or not values.local_files_only
        or values.trust_remote_code
        or values.dtype != "bfloat16"
        or values.attn_implementation != "sdpa"
        or values.min_pixels < 3136
        or values.max_pixels < values.min_pixels
        or values.max_pixels > 401408
        or values.max_images != 3
        or values.max_input_tokens != 4096
        or not 256 <= values.max_new_tokens <= 1024
        or values.do_sample
        or values.temperature != 0.0
        or values.top_p != 1.0
        or values.seed != 20260804
    ):
        fail("CONFIG_INVALID", "single-expert draft v1 要求冻结 8B 身份、安全视觉上限和 greedy generation")
    return replace(
        values,
        semantic_sha256=sha256_text(canonical_json(values.semantic_dict())),
    )


def _move_batch(batch: Mapping[str, Any], device: Any) -> dict[str, Any]:
    import torch

    return {
        key: value.to(device=device, non_blocking=False) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


class LocalQwenDraftRuntime:
    """只复用仓库 Qwen wrapper；不自行实现 tokenizer/chat template。"""

    def __init__(self, config: LocalDraftConfig) -> None:
        import torch

        if not torch.cuda.is_available():
            fail("CUDA_REQUIRED", "Qwen3-VL-8B 草稿生成要求显式 CUDA；CPU fallback 被拒绝")
        if first_symlink_component(config.model_path) is not None or not config.model_path.is_dir():
            fail("ASSET_MISSING", f"本地 Qwen3-VL-8B 尚不存在：{config.model_path}")
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        self.config = config
        self.device = torch.device("cuda")
        self.processor = Qwen3VLProcessorAdapter(
            processor_path=config.processor_path,
            local_files_only=True,
            trust_remote_code=False,
            min_pixels=config.min_pixels,
            max_pixels=config.max_pixels,
            max_images=config.max_images,
            max_input_tokens=config.max_input_tokens,
        )
        model_config = ModelSection(
            path=config.model_path,
            processor_path=config.processor_path,
            local_files_only=True,
            dtype=config.dtype,
            attn_implementation=config.attn_implementation,
            trust_remote_code=False,
        )
        adaptation = AdaptationSection(
            strategy="prompt_only",
            target_modules=(),
            rank=0,
            alpha=0,
            dropout=0.0,
            freeze_vision=True,
            freeze_merger=True,
        )
        self.model = Qwen3VLModelAdapter.load(
            model_config,
            adaptation,
            device=self.device,
            gradient_checkpointing=False,
        )
        self.collator = DescriptionCollator(self.processor, training=False)
        self.model_identity = self.model.identity.to_dict()
        self.processor_identity = self.processor.identity()

    def generate(self, messages: Sequence[Mapping[str, Any]]) -> str:
        sample = DescriptionSample(
            record_id="annotation_draft",
            parent_id="annotation_draft",
            logical_role="train",
            task_family="mask_grounded_region_description",
            messages=tuple(messages),
            reference_responses=(),
            mask_mode=MaskMode.GT_MASK,
            evidence_ids=(),
            provenance={},
        )
        batch = _move_batch(self.collator([sample]), self.device)
        values = self.model.generate_text(
            batch,
            processor=self.processor.processor,
            max_new_tokens=self.config.max_new_tokens,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
        )
        if len(values) != 1:
            fail("PREDICTION_INVALID", "单条 annotation draft 必须返回一个文本结果")
        # 空文本也是一次已完成生成，由上层保存为 versioned failure，避免重复调用模型。
        return values[0]


def generate_annotation_drafts(
    *,
    project_root: Path | str,
    config_path: Path | str,
    partition: str,
    limit: int | None = None,
    runtime: Any | None = None,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """按 calibration/remaining/all 顺序一次生成；已有 record 永不重复生成。"""

    def progress(event: str, **details: Any) -> None:
        if progress_callback is not None:
            progress_callback({
                "event": event,
                "partition": partition,
                **details,
            })

    config = load_local_draft_config(config_path)
    project, assignments = load_annotation_project(project_root)
    if partition not in {"calibration", "remaining", "all"}:
        fail("ANNOTATION_INVALID", f"partition 非法：{partition}")
    use = AnnotationIntendedUse(project["intended_use"])
    if use is AnnotationIntendedUse.TRAIN_SUPERVISION and partition not in {"calibration", "remaining"}:
        fail("SPLIT_FORBIDDEN", "train project 只允许 calibration/remaining")
    if use is AnnotationIntendedUse.DEV_REFERENCE and partition != "all":
        fail("SPLIT_FORBIDDEN", "dev reference 只允许 all baseline partition")
    frozen_config_sha = project["frozen_draft_config_sha256"]
    if frozen_config_sha is not None and config.semantic_sha256 != frozen_config_sha:
        fail("ANNOTATION_INVALID", "当前 draft config 与 calibration 后冻结身份不一致")
    if partition == "remaining":
        status = read_json(Path(project_root) / "status.json")
        if status.get("calibration_verified") != status.get("calibration_total") or status.get("calibration_total") != 20:
            fail("ANNOTATION_INVALID", "20 条 calibration 未全部专家核验，不得生成 remaining")
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0):
        fail("ANNOTATION_INVALID", "limit 必须是正整数或 null")
    drafts = load_model_drafts(project_root, assignments=assignments)
    candidates = [
        row for row in assignments
        if row["partition"] == partition and row["record_id"] not in drafts
    ]
    if limit is not None:
        candidates = candidates[:limit]
    if not candidates:
        progress("partition_already_drafted", generated=0)
        return {
            "ok": True,
            "generated": 0,
            "partition": partition,
            "message": "partition 已无待生成记录",
            "formal_acceptance": False,
        }
    root = Path(os.path.abspath(Path(project_root)))
    prompt_path = root / "prompt.txt"
    prompt = prompt_path.read_text(encoding="utf-8")
    prompt_sha = sha256_file(prompt_path)
    if project["frozen_prompt_sha256"] is not None and project["frozen_prompt_sha256"] != prompt_sha:
        fail("ANNOTATION_INVALID", "冻结 prompt 已漂移")
    if partition in {"remaining", "all"}:
        frozen_prompt = project["frozen_prompt_sha256"]
        frozen_config = project["frozen_draft_config_sha256"]
        if (frozen_prompt is None) != (frozen_config is None):
            fail("ANNOTATION_INVALID", "project prompt/config 冻结状态不完整")
        if frozen_prompt is None:
            # 第二次 train 命令在加载 8B 前即冻结输入；即使随后 CUDA/OOM，重跑也不能
            # 悄然更换 prompt 或 generation config。
            project["frozen_prompt_sha256"] = prompt_sha
            project["frozen_draft_config_sha256"] = config.semantic_sha256
            atomic_write_json(root / "project.json", project)
        elif (
            frozen_prompt != prompt_sha
            or frozen_config != config.semantic_sha256
        ):
            fail("ANNOTATION_INVALID", "当前 prompt/config 与项目冻结身份不一致")
        progress(
            "inputs_frozen",
            prompt_sha256=prompt_sha,
            config_semantic_sha256=config.semantic_sha256,
        )
    context = load_annotation_asset(project["asset_root"])
    records = {row["record_id"]: row for row in context.records}
    progress("model_loading", pending=len(candidates))
    active_runtime = runtime or LocalQwenDraftRuntime(config)
    if (
        not isinstance(getattr(active_runtime, "model_identity", None), Mapping)
        or not isinstance(getattr(active_runtime, "processor_identity", None), Mapping)
        or not callable(getattr(active_runtime, "generate", None))
    ):
        fail("ANNOTATION_INVALID", "local draft runtime identity/生成接口非法")
    progress("model_ready", pending=len(candidates))
    record_ids = [row["record_id"] for row in candidates]
    run_basis = {
        "config_semantic_sha256": config.semantic_sha256,
        "asset_manifest_sha256": project["asset_manifest_sha256"],
        "prompt_sha256": prompt_sha,
        "partition": partition,
        "record_ids": record_ids,
    }
    run_id = "draft_run_" + sha256_text(canonical_json(run_basis))[:24]
    generation = {
        "max_new_tokens": config.max_new_tokens,
        "do_sample": False,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": config.seed,
        "single_attempt": True,
    }
    run = {
        "schema_version": MODEL_DRAFT_RUN_SCHEMA,
        "draft_run_id": run_id,
        "config_sha256": config.config_file_sha256,
        "config_semantic_sha256": config.semantic_sha256,
        "config": config.semantic_dict(),
        "model_repository": config.repository,
        "model_revision": config.revision,
        "model_identity": dict(active_runtime.model_identity),
        "processor_identity": dict(active_runtime.processor_identity),
        "prompt_text": prompt,
        "prompt_sha256": prompt_sha,
        "generation": generation,
        "partition": partition,
        "record_ids": record_ids,
        "record_ids_sha256": sha256_text(canonical_json(record_ids)),
        "formal_acceptance": False,
    }
    register_draft_run(
        root,
        draft_run=run,
        freeze_prompt=partition in {"remaining", "all"},
    )
    generated = 0
    invalid = 0
    quality_counts: dict[str, int] = {}
    target_template_copies = 0
    for candidate_index, assignment in enumerate(candidates, start=1):
        record = records[assignment["record_id"]]
        messages = build_annotation_draft_messages(
            record,
            asset_root=context.root,
            prompt_text=prompt,
        )
        messages_sha = sha256_text(canonical_json(messages))
        raw = active_runtime.generate(messages)
        failure: dict[str, Any] | None = None
        description: dict[str, Any] | None = None
        quality_status: str | None = None
        parse_status = "valid"
        try:
            parsed = parse_region_model_output(raw)
            if parsed.target_status.value != assignment["target_status"]:
                fail("ANNOTATION_INVALID", "模型 draft target_status 与程序事实不一致")
            description = parsed.to_dict()
            quality = assess_region_draft_quality(parsed)
            quality_counts[quality.status.value] = (
                quality_counts.get(quality.status.value, 0) + 1
            )
            quality_status = quality.status.value
            if (
                assignment["target_status"] == "target_present"
                and quality.metrics["template_match"] is True
            ):
                target_template_copies += 1
        except VLMError as error:
            parse_status = "invalid"
            description = None
            failure = {
                "schema_version": MODEL_DRAFT_FAILURE_SCHEMA,
                "code": error.code.value,
                "message": str(error),
                "details": dict(error.details),
            }
            invalid += 1
        except Exception as error:
            # 仅解析/合同错误形成 failure；CUDA/OOM 等发生在 generate()，会直接中止且可续跑。
            parse_status = "invalid"
            description = None
            failure = {
                "schema_version": MODEL_DRAFT_FAILURE_SCHEMA,
                "code": "ANNOTATION_INVALID",
                "message": str(error),
                "details": {},
            }
            invalid += 1
        draft_id = "draft_" + sha256_text(canonical_json({
            "draft_run_id": run_id,
            "record_id": assignment["record_id"],
            "asset_identity_sha256": assignment["asset_identity_sha256"],
        }))[:24]
        append_model_draft(root, draft={
            "schema_version": MODEL_DRAFT_SCHEMA,
            "draft_id": draft_id,
            "draft_run_id": run_id,
            "record_id": assignment["record_id"],
            "asset_identity_sha256": assignment["asset_identity_sha256"],
            "messages_sha256": messages_sha,
            "raw_output": raw,
            "parse_status": parse_status,
            "description": description,
            "failure": failure,
        })
        generated += 1
        progress(
            "draft_persisted",
            record_id=assignment["record_id"],
            current=candidate_index,
            total=len(candidates),
            parse_status=parse_status,
            quality_status=quality_status,
        )
    completed_ids = set(load_model_drafts(root, assignments=assignments))
    result = {
        "ok": True,
        "draft_run_id": run_id,
        "partition": partition,
        "generated": generated,
        "invalid": invalid,
        "quality_counts": dict(sorted(quality_counts.items())),
        "target_template_copies": target_template_copies,
        "remaining": sum(
            row["partition"] == partition
            and row["record_id"] not in completed_ids
            for row in assignments
        ),
        "formal_acceptance": False,
    }
    progress("generation_complete", **result)
    return result
