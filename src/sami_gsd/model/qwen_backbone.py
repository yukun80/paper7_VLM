"""Official online Qwen3-VL native multi-image wrapper for P2."""

from __future__ import annotations

import importlib.metadata
import inspect
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional

from sami_gsd.contracts.model import SamiModelConfig
from sami_gsd.model.cache import QwenBackboneCache, build_cache_key, compare_backbone_states
from sami_gsd.model.rendering import RENDERER_REVISION
from sami_gsd.model.sensor_adapter import ADAPTER_REVISION
from sami_gsd.model.states import (
    MultiImageBatch,
    QwenBackboneState,
    SpatialFeatureLevel,
    ViewBackboneState,
    ViewTransform,
)
from sami_gsd.utilities.artifacts import canonical_json_bytes, sha256_bytes, sha256_file


WRAPPER_REVISION = "sami_qwen3_vl_official_native_multi_image_wrapper_v1"


def _aggregate_files(root: Path, relative_paths: tuple[str, ...]) -> str:
    """Hash a stable list of exact local model/processor files."""

    records: list[dict[str, object]] = []
    for relative_path in relative_paths:
        path = root / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"required local Qwen file is missing: {path}")
        records.append(
            {
                "path": relative_path,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return sha256_bytes(canonical_json_bytes(records))


def fingerprint_local_qwen(model_root: Path) -> tuple[str, str]:
    """Return exact model-weight/config and official-processor fingerprints."""

    root = model_root.resolve()
    weight_files = tuple(sorted(path.name for path in root.glob("*.safetensors")))
    if not weight_files:
        raise FileNotFoundError(f"no local safetensors weights found below {root}")
    optional_index = tuple(sorted(path.name for path in root.glob("*.safetensors.index.json")))
    model_paths = ("config.json",) + optional_index + weight_files
    processor_candidates = (
        "chat_template.json",
        "merges.txt",
        "preprocessor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "video_preprocessor_config.json",
        "vocab.json",
    )
    processor_paths = tuple(path for path in processor_candidates if (root / path).is_file())
    required_processor = {"chat_template.json", "preprocessor_config.json", "tokenizer_config.json"}
    if not required_processor.issubset(processor_paths):
        raise FileNotFoundError("local Qwen directory lacks official processor/chat-template files")
    return _aggregate_files(root, model_paths), _aggregate_files(root, processor_paths)


def _code_revision(model: Any) -> str:
    """Bind the installed Transformers version and exact Qwen model source bytes."""

    source = inspect.getsourcefile(model.__class__)
    if source is None:
        raise RuntimeError("cannot resolve installed Qwen3-VL model source")
    transformers_version = importlib.metadata.version("transformers")
    return sha256_bytes(
        canonical_json_bytes(
            {
                "transformers_version": transformers_version,
                "model_source_sha256": sha256_file(Path(source)),
            }
        )
    )


class QwenBackboneWrapper:
    """Expose task-neutral state from exactly one official Qwen3-VL forward."""

    def __init__(
        self,
        config: SamiModelConfig,
        *,
        model: Any,
        processor: Any,
        device: torch.device,
        model_fingerprint: str,
        processor_fingerprint: str,
        qwen_code_revision: str,
    ) -> None:
        self.config = config
        self.model = model
        self.processor = processor
        self.device = device
        self.model_fingerprint = model_fingerprint
        self.processor_fingerprint = processor_fingerprint
        self.qwen_code_revision = qwen_code_revision
        self._validate_official_components()

    @classmethod
    def from_pretrained(
        cls,
        config: SamiModelConfig,
        *,
        repository_root: Path,
        device: str,
    ) -> "QwenBackboneWrapper":
        """Load the local official processor/model without network fallback."""

        try:
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        except ImportError as error:
            raise RuntimeError("install the root model extra before running P2: pip install -e '.[model]'") from error

        model_root = (repository_root.resolve() / config.model.model_path).resolve()
        if not model_root.is_relative_to(repository_root.resolve()):
            raise ValueError("model_path must resolve below the repository root")
        model_fingerprint, processor_fingerprint = fingerprint_local_qwen(model_root)
        dtype = torch.bfloat16 if config.model.dtype == "bfloat16" else torch.float32
        processor = AutoProcessor.from_pretrained(
            model_root,
            local_files_only=config.model.local_files_only,
            trust_remote_code=config.model.trust_remote_code,
        )
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_root,
            local_files_only=config.model.local_files_only,
            trust_remote_code=config.model.trust_remote_code,
            dtype=dtype,
            attn_implementation=config.model.attention_implementation,
        )
        resolved_device = torch.device(device)
        if resolved_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("requested CUDA for P2 smoke but CUDA is not available")
        model.to(resolved_device)
        model.eval()
        for parameter in model.model.visual.parameters():
            parameter.requires_grad_(False)
        return cls(
            config,
            model=model,
            processor=processor,
            device=resolved_device,
            model_fingerprint=model_fingerprint,
            processor_fingerprint=processor_fingerprint,
            qwen_code_revision=_code_revision(model),
        )

    def _validate_official_components(self) -> None:
        """Reject a generic/legacy controller masquerading as the official backend."""

        if self.model.__class__.__name__ != "Qwen3VLForConditionalGeneration":
            raise TypeError("P2 requires official Qwen3VLForConditionalGeneration")
        if self.processor.__class__.__name__ != "Qwen3VLProcessor":
            raise TypeError("P2 requires official Qwen3VLProcessor")
        image_processor = self.processor.image_processor
        if int(image_processor.patch_size) != self.config.processor.patch_size:
            raise ValueError("official processor patch size differs from P2 config")
        if int(image_processor.merge_size) != self.config.processor.spatial_merge_size:
            raise ValueError("official processor spatial merge size differs from P2 config")
        visual = self.model.model.visual
        if int(visual.patch_size) != self.config.processor.patch_size:
            raise ValueError("official model visual patch size differs from processor contract")
        if int(visual.spatial_merge_size) != self.config.processor.spatial_merge_size:
            raise ValueError("official model merge size differs from processor contract")

    def _prompts(self, batch: MultiImageBatch) -> tuple[str, ...]:
        """Build official-chat-template prompts with interleaved cards/images."""

        prompts: list[str] = []
        for parent in batch.parents:
            content: list[dict[str, str]] = []
            for view in parent.views:
                card = canonical_json_bytes(view.sensor_card.payload()).decode("utf-8").strip()
                content.append({"type": "text", "text": f"Sensor card: {card}"})
                content.append({"type": "image"})
            content.append({"type": "text", "text": self.config.processor.user_instruction})
            messages = [
                {"role": "system", "content": self.config.processor.system_prompt},
                {"role": "user", "content": content},
            ]
            prompt = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            if not isinstance(prompt, str) or not prompt:
                raise ValueError("official chat template returned an empty/non-string prompt")
            prompts.append(prompt)
        return tuple(prompts)

    def _key_payload(
        self,
        batch: MultiImageBatch,
        prompts: tuple[str, ...],
        *,
        return_spatial_features: bool,
    ) -> dict[str, Any]:
        """Bind every model/processor/input field that can affect state bytes."""

        return {
            "wrapper_revision": WRAPPER_REVISION,
            "adapter_revision": ADAPTER_REVISION,
            "renderer_revision": RENDERER_REVISION,
            "batch": batch.identity_payload(),
            "prompt_version": self.config.processor.prompt_version,
            "prompt_sha256": [sha256_bytes(prompt.encode("utf-8")) for prompt in prompts],
            "model_weight_hash": self.model_fingerprint,
            "processor_hash": self.processor_fingerprint,
            "qwen_code_revision": self.qwen_code_revision,
            "pixel_budget": batch.profile.model_dump(mode="json"),
            "view_order": [list(parent.view_ids) for parent in batch.parents],
            "dtype": self.config.model.dtype,
            "return_spatial_features": return_spatial_features,
        }

    def encode(
        self,
        batch: MultiImageBatch,
        *,
        return_spatial_features: bool,
        cache_store: object | None = None,
    ) -> QwenBackboneState:
        """Run one official native multi-image forward or return a strict cache hit."""

        prompts = self._prompts(batch)
        key_payload = self._key_payload(
            batch,
            prompts,
            return_spatial_features=return_spatial_features,
        )
        cache_key, key_payload_sha256 = build_cache_key(key_payload)
        cache: QwenBackboneCache | None
        if cache_store is None:
            cache = None
        elif isinstance(cache_store, QwenBackboneCache):
            cache = cache_store
        else:
            raise TypeError("cache_store must be QwenBackboneCache or None")
        if cache is not None:
            cached = cache.get(cache_key, key_payload_sha256=key_payload_sha256)
            if cached is not None:
                return cached

        flat_images = [view.image for view in batch.flat_views]
        processor_outputs = self.processor(
            text=list(prompts),
            images=flat_images,
            padding=True,
            return_tensors="pt",
            min_pixels=self.config.processor.processor_min_pixels,
            max_pixels=batch.profile.reference_max_pixels,
        )
        if "pixel_values" not in processor_outputs or "image_grid_thw" not in processor_outputs:
            raise ValueError("official processor did not emit pixel_values and image_grid_thw")
        image_grid_thw_cpu = processor_outputs["image_grid_thw"].detach().cpu()
        if image_grid_thw_cpu.shape != (len(flat_images), 3):
            raise ValueError("official processor image_grid_thw does not match effective view count")
        model_inputs = {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in processor_outputs.items()
        }

        captured: dict[str, Any] = {}

        def capture_visual(_module: Any, _inputs: Any, output: Any) -> None:
            captured["visual"] = output

        def capture_language(_module: Any, _inputs: Any, output: Any) -> None:
            captured["language_hidden"] = output[0]

        visual_handle = self.model.model.visual.register_forward_hook(capture_visual)
        language_handle = self.model.model.language_model.register_forward_hook(capture_language)
        try:
            with torch.inference_mode():
                self.model(
                    **model_inputs,
                    use_cache=False,
                    output_hidden_states=False,
                    return_dict=True,
                    logits_to_keep=1,
                )
        finally:
            visual_handle.remove()
            language_handle.remove()
        if set(captured) != {"visual", "language_hidden"}:
            raise RuntimeError("official Qwen forward hooks did not capture visual and language states")
        state = self._assemble_state(
            batch,
            prompts,
            processor_outputs=processor_outputs,
            image_grid_thw=image_grid_thw_cpu,
            visual_output=captured["visual"],
            language_hidden=captured["language_hidden"],
            cache_key=cache_key,
            return_spatial_features=return_spatial_features,
        )
        if cache is not None:
            cache.write(state, key_payload_sha256=key_payload_sha256)
            cached = cache.get(cache_key, key_payload_sha256=key_payload_sha256)
            if cached is None:
                raise RuntimeError("newly published cache entry cannot be reopened")
            compare_backbone_states(state, cached, self.config.cache.equivalence)
        return state

    def _assemble_state(
        self,
        batch: MultiImageBatch,
        prompts: tuple[str, ...],
        *,
        processor_outputs: Any,
        image_grid_thw: torch.Tensor,
        visual_output: Any,
        language_hidden: torch.Tensor,
        cache_key: str,
        return_spatial_features: bool,
    ) -> QwenBackboneState:
        """Split flattened official outputs back into exact parent/view grids."""

        merge_size = self.config.processor.spatial_merge_size
        split_sizes = [int(row.prod().item()) // (merge_size * merge_size) for row in image_grid_thw]
        final_visual = visual_output.pooler_output
        if isinstance(final_visual, torch.Tensor):
            final_per_view = tuple(torch.split(final_visual, split_sizes))
        else:
            final_per_view = tuple(final_visual)
        if len(final_per_view) != len(split_sizes):
            raise ValueError("official final visual feature count differs from view count")
        deepstack = tuple(getattr(visual_output, "deepstack_features", ()) or ())
        deepstack_per_level = tuple(tuple(torch.split(level, split_sizes)) for level in deepstack)
        layer_indexes = tuple(int(index) for index in self.model.config.vision_config.deepstack_visual_indexes)
        if len(layer_indexes) != len(deepstack_per_level):
            raise ValueError("official DeepStack indexes/features differ")

        input_ids = processor_outputs["input_ids"].to(language_hidden.device)
        image_token_id = int(self.model.config.image_token_id)
        language_tokens: list[torch.Tensor] = []
        view_offset = 0
        for parent_index, parent in enumerate(batch.parents):
            positions = torch.nonzero(input_ids[parent_index] == image_token_id, as_tuple=False).flatten()
            parent_sizes = split_sizes[view_offset : view_offset + len(parent.views)]
            if int(positions.numel()) != sum(parent_sizes):
                raise ValueError(f"image placeholder tokens do not match view grids for {parent.parent_id}")
            token_offset = 0
            for token_count in parent_sizes:
                selected_positions = positions[token_offset : token_offset + token_count]
                language_tokens.append(language_hidden[parent_index, selected_positions])
                token_offset += token_count
            view_offset += len(parent.views)

        view_states: list[ViewBackboneState] = []
        for index, (prepared_view, grid_row, token_count) in enumerate(
            zip(batch.flat_views, image_grid_thw, split_sizes, strict=True)
        ):
            temporal, grid_h, grid_w = (int(item) for item in grid_row.tolist())
            if temporal != 1:
                raise ValueError("P2 image view unexpectedly produced a temporal grid")
            if grid_h % merge_size or grid_w % merge_size:
                raise ValueError("official processor grid is not divisible by spatial merge size")
            merged_hw = (grid_h // merge_size, grid_w // merge_size)
            if merged_hw[0] * merged_hw[1] != token_count:
                raise ValueError("merged Qwen grid does not reconstruct token count")
            valid_mask = functional.interpolate(
                prepared_view.valid_mask[None, None].to(dtype=torch.float32, device=self.device),
                size=merged_hw,
                mode="nearest",
            )[0, 0].to(dtype=torch.bool)
            levels: list[SpatialFeatureLevel] = []
            if return_spatial_features:
                for layer_index, per_view in zip(layer_indexes, deepstack_per_level, strict=True):
                    features = per_view[index]
                    if features.shape[0] != token_count:
                        raise ValueError("DeepStack token count differs from merged grid")
                    levels.append(
                        SpatialFeatureLevel(
                            level=f"deepstack_layer_{layer_index}",
                            features=features.reshape(*merged_hw, features.shape[-1]).permute(2, 0, 1),
                        )
                    )
                final_features = final_per_view[index]
                if final_features.shape[0] != token_count:
                    raise ValueError("final visual token count differs from merged grid")
                levels.append(
                    SpatialFeatureLevel(
                        level="vision_final",
                        features=final_features.reshape(*merged_hw, final_features.shape[-1]).permute(2, 0, 1),
                    )
                )
            modality = prepared_view.modality
            view_states.append(
                ViewBackboneState(
                    parent_id=prepared_view.parent_id,
                    view_id=modality.modality_id,
                    role=prepared_view.role,
                    sensor_card=prepared_view.sensor_card,
                    language_aligned_visual_tokens=language_tokens[index],
                    spatial_features=tuple(levels),
                    valid_mask=valid_mask,
                    transform=ViewTransform(
                        source_hw=prepared_view.source_hw,
                        rendered_hw=prepared_view.rendered_hw,
                        processor_grid_thw=(temporal, grid_h, grid_w),
                        merged_grid_hw=merged_hw,
                        alignment_status=modality.alignment_status,
                        source_to_reference_transform=(
                            None
                            if modality.source_to_reference_transform is None
                            else tuple(
                                step.model_dump(mode="json")
                                for step in modality.source_to_reference_transform
                            )
                        ),
                        reference_to_source_transform=(
                            None
                            if modality.reference_to_source_transform is None
                            else tuple(
                                step.model_dump(mode="json")
                                for step in modality.reference_to_source_transform
                            )
                        ),
                    ),
                    image_sha256=prepared_view.image_sha256,
                    valid_sha256=prepared_view.valid_sha256,
                )
            )
        state = QwenBackboneState(
            schema_version="sami_qwen_backbone_state_v1",
            parent_ids=tuple(parent.parent_id for parent in batch.parents),
            view_order=tuple(parent.view_ids for parent in batch.parents),
            reference_view_ids=tuple(parent.canonical_reference_view_id for parent in batch.parents),
            active_modality_ids=tuple(parent.active_modality_ids for parent in batch.parents),
            excluded_modalities=tuple(parent.excluded_modalities for parent in batch.parents),
            views=tuple(view_states),
            prompt_sha256=tuple(sha256_bytes(prompt.encode("utf-8")) for prompt in prompts),
            model_fingerprint=self.model_fingerprint,
            processor_fingerprint=self.processor_fingerprint,
            qwen_code_revision=self.qwen_code_revision,
            profile=batch.profile.profile,
            dtype=self.config.model.dtype,
            cache_key=cache_key,
            from_cache=False,
        )
        state.validate()
        return state


__all__ = ["QwenBackboneWrapper", "WRAPPER_REVISION", "fingerprint_local_qwen"]
