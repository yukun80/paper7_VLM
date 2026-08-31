"""调用 Shared VLM 库实现的薄 CLI。"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Sequence

import torch

from oa_groundrag.data.rs_general.dataset import RSGeneralDescDataset
from oa_groundrag.data.rs_general.errors import RSGeneralDescError

from .checkpoint import CheckpointManager
from .config import apply_runtime_overrides, load_config
from .data import ExternalDescriptionDataset
from oa_groundrag.evaluation.vlm import evaluate_predictions
from .errors import (
    CheckpointError,
    ModelError,
    VLMError,
    PredictionError,
    ReasonCode,
)
from .inference import run_inference
from oa_groundrag.evaluation.rs_general.acceptance import verify_gate_b_acceptance
from oa_groundrag.evaluation.rs_general.comparison import compare_gate_b_families
from oa_groundrag.evaluation.rs_general.metrics import evaluate_gate_b
from oa_groundrag.evaluation.rs_general.generation import generate_gate_b
from oa_groundrag.evaluation.rs_general.media import (
    DEFAULT_GATE_B_BENCHMARK_ROOT,
    locate_gate_b_media,
)
from oa_groundrag.evaluation.rs_general.selection import prepare_gate_b
from .backends import build_model_adapter, build_processor_adapter
from .preflight import BenchmarkAccess, open_benchmark_access, run_preflight
from .processing import DescriptionCollator
from oa_groundrag.training.vlm.progress import compact_training_result
from oa_groundrag.evaluation.vlm_smoke import run_bounded_external_smoke
from oa_groundrag.training.vlm.trainer import DescriptionTrainer, training_layout_identity
from oa_groundrag.training.vlm.validation import select_bounded_external_validation


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rs-vlm",
        description=(
            "Shared RS-Geohazard MLLM train/infer/evaluate"
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
    gate_prepare = subcommands.add_parser("gate-b-prepare")
    gate_prepare.add_argument("--protocol", type=Path, required=True)
    gate_prepare.add_argument("--training-root", type=Path, required=True)
    gate_prepare.add_argument("--output-root", type=Path, required=True)
    gate_generate = subcommands.add_parser("gate-b-generate")
    gate_generate.add_argument("--protocol", type=Path, required=True)
    gate_generate.add_argument("--selection", type=Path, required=True)
    gate_generate.add_argument(
        "--model-role",
        choices=("base", "adapter"),
        required=True,
    )
    gate_generate.add_argument("--training-root", type=Path)
    gate_generate.add_argument("--output-root", type=Path, required=True)
    gate_evaluate = subcommands.add_parser("gate-b-evaluate")
    gate_evaluate.add_argument("--protocol", type=Path, required=True)
    gate_evaluate.add_argument("--selection", type=Path, required=True)
    gate_evaluate.add_argument("--base-run", type=Path, required=True)
    gate_evaluate.add_argument("--adapter-run", type=Path, required=True)
    gate_evaluate.add_argument("--output-root", type=Path, required=True)
    gate_verify = subcommands.add_parser("gate-b-verify")
    gate_verify.add_argument("--protocol", type=Path, required=True)
    gate_verify.add_argument("--selection", type=Path, required=True)
    gate_verify.add_argument("--training-root", type=Path, required=True)
    gate_verify.add_argument("--base-run", type=Path, required=True)
    gate_verify.add_argument("--adapter-run", type=Path, required=True)
    gate_verify.add_argument("--evaluation-root", type=Path, required=True)
    gate_verify.add_argument(
        "--expected-protocol-file-sha256",
        required=True,
    )
    gate_verify.add_argument("--expected-report-sha256", required=True)
    family_compare = subcommands.add_parser("gate-b-family-compare")
    family_compare.add_argument("--reference-protocol", type=Path, required=True)
    family_compare.add_argument("--reference-selection", type=Path, required=True)
    family_compare.add_argument("--reference-adapter-run", type=Path, required=True)
    family_compare.add_argument(
        "--reference-evaluation-root",
        type=Path,
        required=True,
    )
    family_compare.add_argument(
        "--expected-reference-report-sha256",
        required=True,
    )
    family_compare.add_argument(
        "--expected-reference-predictions-sha256",
        required=True,
    )
    family_compare.add_argument("--candidate-protocol", type=Path, required=True)
    family_compare.add_argument("--candidate-selection", type=Path, required=True)
    family_compare.add_argument("--candidate-adapter-run", type=Path, required=True)
    family_compare.add_argument("--output-root", type=Path, required=True)
    gate_locate = subcommands.add_parser("gate-b-locate-media")
    gate_locate.add_argument("--predictions", type=Path, required=True)
    gate_locate.add_argument("--line-number", type=int, required=True)
    gate_locate.add_argument(
        "--benchmark-root",
        type=Path,
        default=DEFAULT_GATE_B_BENCHMARK_ROOT,
    )
    return parser


def _processor(config):
    return build_processor_adapter(config)


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
    if arguments.command == "gate-b-locate-media":
        media_paths = locate_gate_b_media(
            arguments.predictions,
            line_number=arguments.line_number,
            benchmark_root=arguments.benchmark_root,
        )
        for media in media_paths:
            print(f"{media.role}\t{media.path}")
        return 0
    if arguments.command == "gate-b-prepare":
        target = prepare_gate_b(
            arguments.protocol,
            training_root=arguments.training_root,
            output_root=arguments.output_root,
        )
        print(target)
        return 0
    if arguments.command == "gate-b-generate":
        outcome = generate_gate_b(
            arguments.protocol,
            arguments.selection,
            model_role=arguments.model_role,
            training_root=arguments.training_root,
            output_root=arguments.output_root,
        )
        print(outcome.root)
        return 0 if outcome.valid_for_evaluation else 2
    if arguments.command == "gate-b-evaluate":
        outcome = evaluate_gate_b(
            arguments.protocol,
            arguments.selection,
            base_run=arguments.base_run,
            adapter_run=arguments.adapter_run,
            output_root=arguments.output_root,
        )
        print(outcome.root)
        if outcome.status == "invalid":
            return 2
        return 0 if outcome.gate_b_passed else 1
    if arguments.command == "gate-b-verify":
        verification = verify_gate_b_acceptance(
            arguments.protocol,
            arguments.selection,
            training_root=arguments.training_root,
            base_run=arguments.base_run,
            adapter_run=arguments.adapter_run,
            evaluation_root=arguments.evaluation_root,
            expected_protocol_file_sha256=(
                arguments.expected_protocol_file_sha256
            ),
            expected_report_sha256=arguments.expected_report_sha256,
        )
        print(
            json.dumps(
                verification.to_dict(),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if arguments.command == "gate-b-family-compare":
        outcome = compare_gate_b_families(
            reference_protocol=arguments.reference_protocol,
            reference_selection=arguments.reference_selection,
            reference_adapter_run=arguments.reference_adapter_run,
            reference_evaluation_root=arguments.reference_evaluation_root,
            expected_reference_report_sha256=(
                arguments.expected_reference_report_sha256
            ),
            expected_reference_predictions_sha256=(
                arguments.expected_reference_predictions_sha256
            ),
            candidate_protocol=arguments.candidate_protocol,
            candidate_selection=arguments.candidate_selection,
            candidate_adapter_run=arguments.candidate_adapter_run,
            output_root=arguments.output_root,
        )
        print(outcome.root)
        return 0
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
    model = build_model_adapter(
        config,
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


_GATE_B_INFRASTRUCTURE_MARKERS = (
    "cuda out of memory",
    "cuda error",
    "cuda driver",
    "cublas",
    "cudnn",
    "nccl",
    "no kernel image is available",
)


def _gate_b_infrastructure_runtime_error(
    argv: Sequence[str] | None,
    error: RuntimeError,
) -> bool:
    values = list(sys.argv[1:] if argv is None else argv)
    if not values or values[0] != "gate-b-generate":
        return False
    if isinstance(error, torch.cuda.OutOfMemoryError):
        return True
    message = str(error).lower()
    return any(marker in message for marker in _GATE_B_INFRASTRUCTURE_MARKERS)


def entrypoint(argv: Sequence[str] | None = None) -> int:
    try:
        return main(argv)
    except (VLMError, RSGeneralDescError) as error:
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
    except RuntimeError as error:
        if not _gate_b_infrastructure_runtime_error(argv, error):
            raise
        print(
            json.dumps(
                {
                    "status": "invalid",
                    "reason_code": ReasonCode.GATE_B_RUN_INVALID.value,
                    "message": str(error),
                    "details": {"exception_type": type(error).__name__},
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(entrypoint())
