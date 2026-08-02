"""调用 phase4 库实现的薄 CLI。"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Sequence

import torch

from oa_groundrag.phase3.dataset import RSGeneralDescDataset
from oa_groundrag.phase3.errors import RSGeneralDescError

from .checkpoint import CheckpointManager
from .config import apply_runtime_overrides, load_config
from .data import ExternalDescriptionDataset
from .evaluation import evaluate_predictions
from .errors import (
    CheckpointError,
    ModelError,
    Phase4Error,
    PredictionError,
    ReasonCode,
)
from .inference import run_inference
from .model import Qwen3VLModelAdapter
from .preflight import BenchmarkAccess, open_benchmark_access, run_preflight
from .processing import DescriptionCollator, Qwen3VLProcessorAdapter
from .progress import compact_training_result
from .smoke import run_bounded_external_smoke
from .trainer import DescriptionTrainer, training_layout_identity
from .validation import select_bounded_external_validation


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rs-vlm",
        description=(
            "算法 Phase 3 / 仓库 phase4：RS-VLM train/infer/evaluate"
        ),
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "train", "infer", "smoke"):
        child = subcommands.add_parser(name)
        child.add_argument("--config", type=Path, required=True)
        if name == "preflight":
            child.add_argument("--output-root", type=Path)
        elif name == "smoke":
            child.add_argument("--output-root", type=Path)
        elif name == "train":
            child.add_argument("--resume-checkpoint", type=Path)
            child.add_argument("--output-root", type=Path)
            child.add_argument("--stop-after-steps", type=int)
            child.add_argument("--log-interval", type=int)
        elif name == "infer":
            child.add_argument("--checkpoint", type=Path)
            child.add_argument("--output-root", type=Path, required=True)
            child.add_argument("--limit", type=int, default=1)
    evaluate = subcommands.add_parser("evaluate")
    evaluate.add_argument("--config", type=Path, required=True)
    evaluate.add_argument("--predictions", type=Path, required=True)
    evaluate.add_argument("--output-root", type=Path, required=True)
    return parser


def _processor(config):
    return Qwen3VLProcessorAdapter(
        processor_path=config.model.processor_path,
        local_files_only=config.model.local_files_only,
        trust_remote_code=config.model.trust_remote_code,
        min_pixels=config.limits.min_pixels,
        max_pixels=config.limits.max_pixels,
        max_images=config.limits.max_images,
        max_input_tokens=config.limits.max_input_tokens,
    )


def _external_dataset(
    config,
    derived_root: Path,
    access: BenchmarkAccess,
):
    canonical = RSGeneralDescDataset(
        config.data.benchmark_root,
        roles=config.data.roles,
        task_families=config.data.task_families,
        load_assets=False,
        seed=config.run.seed,
        expected_manifest_sha256=config.data.expected_manifest_sha256,
        verifier=access.verifier,
    )
    return ExternalDescriptionDataset(
        canonical,
        derived_root=derived_root,
        seed=config.run.seed,
    )


def _absolute_cli_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    return Path(os.path.abspath(path))


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    config = load_config(arguments.config)
    if arguments.command in {"preflight", "train"}:
        config = apply_runtime_overrides(
            config,
            output_root=_absolute_cli_path(arguments.output_root),
            resume_checkpoint=(
                _absolute_cli_path(arguments.resume_checkpoint)
                if arguments.command == "train"
                else None
            ),
            log_interval=(
                arguments.log_interval
                if arguments.command == "train"
                else None
            ),
        )
    if arguments.command == "preflight":
        result = run_preflight(config)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if arguments.command == "smoke":
        target = run_bounded_external_smoke(
            config,
            output_root=arguments.output_root,
        )
        print(target)
        return 0
    if arguments.command == "evaluate":
        target = evaluate_predictions(
            arguments.predictions,
            output_root=arguments.output_root,
            expected_mask_mode=config.run.mask_mode,
            formal=False,
        )
        print(target)
        return 0
    access = open_benchmark_access(config)
    if arguments.command == "train":
        if config.adaptation.strategy == "prompt_only":
            raise ModelError(
                ReasonCode.MODEL_IDENTITY_MISMATCH,
                "prompt-only baseline 有 0 个训练参数；请使用 infer/evaluate",
            )
        run_preflight(
            config,
            require_new_output=config.run.resume_checkpoint is None,
            access=access,
        )
        if not torch.cuda.is_available():
            raise ModelError(
                ReasonCode.CUDA_REQUIRED,
                "真实 Qwen3-VL LoRA 训练要求可用 CUDA；CPU 仅用于 tiny fixture",
            )
    elif arguments.command == "infer":
        if arguments.limit <= 0:
            raise PredictionError(
                ReasonCode.TYPE_MISMATCH,
                "--limit 必须 > 0",
            )
        if (
            config.adaptation.strategy == "lora"
            and arguments.checkpoint is None
        ):
            raise CheckpointError(
                ReasonCode.CHECKPOINT_INCOMPATIBLE,
                "LoRA infer 必须显式给 --checkpoint",
            )
        run_preflight(config, require_new_output=False, access=access)
    identity = access.identity
    device = torch.device(
        "cuda"
        if arguments.command == "train"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if arguments.command == "train":
        print(
            "[preflight] "
            f"benchmark={identity.build_id} "
            f"config_sha256={config.semantic_sha256} "
            f"output={config.run.output_root}",
            file=sys.stderr,
            flush=True,
        )
        print(
            f"[load] model={config.model.path} device={device}",
            file=sys.stderr,
            flush=True,
        )
    processor = _processor(config)
    model = Qwen3VLModelAdapter.load(
        config.model,
        config.adaptation,
        device=device,
        gradient_checkpointing=(
            arguments.command == "train"
            and config.training.gradient_checkpointing
        ),
    )
    if arguments.command == "train":
        with tempfile.TemporaryDirectory(
            prefix="rs_generaldesc_external_derived_"
        ) as temporary:
            canonical = RSGeneralDescDataset(
                config.data.benchmark_root,
                roles=("external_train", "external_val"),
                task_families=config.data.task_families,
                load_assets=False,
                seed=config.run.seed,
                expected_manifest_sha256=config.data.expected_manifest_sha256,
                verifier=access.verifier,
            )
            dataset = ExternalDescriptionDataset(
                canonical,
                derived_root=Path(temporary) / "train",
                seed=config.run.seed,
                roles=("external_train",),
            )
            validation_dataset = ExternalDescriptionDataset(
                canonical,
                derived_root=Path(temporary) / "validation",
                seed=config.run.seed,
                roles=("external_val",),
            )
            validation_selection = select_bounded_external_validation(
                validation_dataset,
                benchmark_build_id=identity.build_id,
                benchmark_payload_sha256=identity.payload_sha256,
                seed=config.run.seed,
                max_parents=config.training.validation_max_parents,
            )
            collator = DescriptionCollator(processor, training=True)
            trainer = DescriptionTrainer(
                config=config,
                model=model,
                collator=collator,
                validation_dataset=validation_dataset,
                validation_collator=DescriptionCollator(
                    processor,
                    training=True,
                ),
                validation_selection=validation_selection,
                benchmark_identity=identity,
                processor_identity=processor.identity(),
                device=device,
            )
            result = trainer.fit(
                dataset,
                resume_checkpoint=config.run.resume_checkpoint,
                stop_after_steps=arguments.stop_after_steps,
            )
        print(compact_training_result(result.to_dict()))
        return 0
    if arguments.command == "infer":
        with tempfile.TemporaryDirectory(
            prefix="rs_generaldesc_external_infer_derived_"
        ) as temporary:
            dataset = _external_dataset(
                config,
                Path(temporary) / "derived",
                access,
            )
            if config.adaptation.strategy == "lora":
                assert arguments.checkpoint is not None
                payload = CheckpointManager().load(
                    arguments.checkpoint,
                    expected_config_semantic_sha256=config.semantic_sha256,
                    expected_benchmark_identity=(
                        identity.training_identity_dict()
                    ),
                    expected_validation_selection_identity=None,
                    expected_model_identity=model.identity.to_dict(),
                    expected_processor_identity=processor.identity(),
                    expected_training_layout=training_layout_identity(config),
                    expected_trainable_names=model.trainable_names,
                )
                model.load_trainable_state_dict(payload.trainable_state)
            samples = [
                dataset[index]
                for index in range(min(arguments.limit, len(dataset)))
            ]
            target = run_inference(
                config=config,
                samples=samples,
                collator=DescriptionCollator(processor, training=False),
                model=model,
                processor=processor.processor,
                output_root=arguments.output_root,
            )
        print(target)
        return 0
    raise AssertionError("unreachable")


def entrypoint(argv: Sequence[str] | None = None) -> int:
    try:
        return main(argv)
    except (Phase4Error, RSGeneralDescError) as error:
        print(
            json.dumps(
                {
                    "status": "error",
                    "reason_code": error.code.value,
                    "message": str(error),
                    "details": dict(error.details),
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(entrypoint())
