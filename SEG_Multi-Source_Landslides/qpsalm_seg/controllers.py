#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Mask-query controllers for the v2 Qwen-PSALM architecture.

The production controller feeds task text, active-view tokens, evidence anchors,
and learned mask positions through the Qwen language decoder. Historical
text-only cache formats are intentionally unsupported.
"""

from __future__ import annotations

import hashlib
from contextlib import contextmanager
import copy
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from .config import QPSalmConfig, QWEN_GRADIENT_CHECKPOINTING_MODES
from .paths import resolve_repo_path
from .schema import MODALITY_FAMILIES, MODALITY_FAMILY_IDS, ModalityBatch, SemanticEvidence


EVIDENCE_FAMILIES = MODALITY_FAMILIES
VISUAL_FAMILY_IDS = MODALITY_FAMILY_IDS
CONTROLLER_SEQUENCE_PROTOCOL = "qwen_mask_query_interleaved_views_v2"


def pad_qwen_sequences(
    sequences: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Differentiably pad variable-length Qwen embeddings and build attention masks."""
    if not sequences:
        raise ValueError("Qwen sequence batch cannot be empty")
    device = sequences[0].device
    lengths = torch.tensor([value.shape[0] for value in sequences], device=device)
    inputs = pad_sequence(list(sequences), batch_first=True)
    attention = (
        torch.arange(inputs.shape[1], device=device)[None] < lengths[:, None]
    ).to(torch.long)
    return inputs, attention, lengths


def qwen_gradient_checkpointing_kwargs(mode: str) -> dict[str, Any] | None:
    """Resolve the only supported Qwen activation-checkpoint protocols."""
    if mode == "reentrant":
        return {"use_reentrant": True, "preserve_rng_state": True}
    if mode == "disabled":
        return None
    raise ValueError(
        f"未知 qwen_gradient_checkpointing={mode!r}; "
        f"expected one of {QWEN_GRADIENT_CHECKPOINTING_MODES}"
    )


def configure_qwen_gradient_checkpointing(model: nn.Module, mode: str) -> dict[str, Any] | None:
    """Apply one explicit checkpoint protocol after PEFT has injected LoRA."""
    checkpoint_kwargs = qwen_gradient_checkpointing_kwargs(mode)
    if checkpoint_kwargs is None:
        model.gradient_checkpointing_disable()
        return None
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs=checkpoint_kwargs
    )
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    return checkpoint_kwargs


def anchor_availability(batch: ModalityBatch, index: int, *, use_full: bool, device: torch.device) -> torch.Tensor:
    instances = batch.full_instances[index] if use_full else batch.instances[index]
    active = {item.family for item in instances}
    return torch.tensor([1] + [int(name in active) for name in EVIDENCE_FAMILIES], device=device)


def resolve_local_path(path_ref: str | Path) -> Path:
    return resolve_repo_path(path_ref) or Path(path_ref)


def validate_qwen_model_dir(model_path: str | Path) -> Path:
    path = resolve_local_path(model_path)
    if not path.is_dir():
        raise FileNotFoundError(f"Qwen3-VL 模型目录不存在: {path}")
    missing = [name for name in ("config.json", "tokenizer_config.json") if not (path / name).exists()]
    has_weights = (
        any(path.glob("*.safetensors"))
        or any(path.glob("pytorch_model*.bin"))
        or (path / "model.safetensors.index.json").exists()
    )
    if not has_weights:
        missing.append("model weights")
    if missing:
        raise FileNotFoundError(f"Qwen3-VL 模型目录不完整: {path}; missing={missing}")
    return path


def select_qwen_model_class() -> type[Any]:
    errors = []
    for class_name in ("Qwen3VLForConditionalGeneration", "AutoModelForImageTextToText"):
        try:
            module = __import__("transformers", fromlist=[class_name])
            return getattr(module, class_name)
        except Exception as exc:  # pragma: no cover - optional dependency
            errors.append(f"{class_name}: {exc}")
    raise RuntimeError("transformers 不支持 Qwen3-VL: " + "; ".join(errors))


def _update_digest_from_file(digest, path: Path) -> None:
    digest.update(path.name.encode())
    digest.update(str(path.stat().st_size).encode())
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)


@lru_cache(maxsize=8)
def local_model_revision(model_dir: Path) -> str:
    model_dir = model_dir.resolve()
    digest = hashlib.sha256()
    for name in ("config.json", "model.safetensors.index.json"):
        path = model_dir / name
        if path.exists():
            _update_digest_from_file(digest, path)
    weight_paths = sorted({
        *model_dir.glob("*.safetensors"),
        *model_dir.glob("pytorch_model*.bin"),
    })
    if not weight_paths:
        raise FileNotFoundError(f"Qwen 权重文件不存在: {model_dir}")
    for path in weight_paths:
        _update_digest_from_file(digest, path)
    return digest.hexdigest()


@lru_cache(maxsize=8)
def local_processor_revision(model_dir: Path) -> str:
    model_dir = model_dir.resolve()
    digest = hashlib.sha256()
    observed = 0
    for name in (
        "preprocessor_config.json", "video_preprocessor_config.json", "processor_config.json",
        "tokenizer_config.json", "special_tokens_map.json",
        "tokenizer.json", "vocab.json", "merges.txt", "chat_template.json",
    ):
        path = model_dir / name
        if path.exists():
            digest.update(name.encode())
            digest.update(path.read_bytes())
            observed += 1
    if observed == 0:
        raise FileNotFoundError(f"Qwen processor 配置不存在: {model_dir}")
    return digest.hexdigest()


class TextProbeMaskController(nn.Module):
    """Development-only controller with the same mask-state contract as Qwen."""

    def __init__(self, dim: int, num_queries: int, num_heads: int, vocab_size: int = 4096) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.embedding = nn.Embedding(vocab_size, dim)
        self.type_embedding = nn.Parameter(torch.randn(4, dim) * 0.02)
        self.evidence_anchors = nn.Parameter(torch.randn(1 + len(EVIDENCE_FAMILIES), dim) * 0.02)
        self.anchor_availability = nn.Embedding(2, dim)
        nn.init.normal_(self.anchor_availability.weight, std=0.02)
        self.mask_embeddings = nn.Parameter(torch.randn(num_queries, dim) * 0.02)
        layer = nn.TransformerEncoderLayer(dim, num_heads, dim * 4, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=2,
            enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(dim)

    def _text(self, values: Sequence[str], device: torch.device) -> torch.Tensor:
        pooled = []
        for text in values:
            words = text.lower().replace(".", " ").replace(",", " ").split() or ["landslide"]
            ids = [int(hashlib.sha256(word.encode()).hexdigest()[:8], 16) % self.vocab_size for word in words[:96]]
            pooled.append(self.embedding(torch.tensor(ids, device=device)).mean(0))
        return torch.stack(pooled)

    def encode_batch(self, batch: ModalityBatch, *, use_full: bool = False) -> SemanticEvidence:
        device = self.embedding.weight.device
        task_text = batch.full_proposal_context_text if use_full else batch.proposal_context_text
        condition_text = batch.full_condition_prompt_text if use_full else batch.condition_prompt_text
        reasoning_text = batch.full_evidence_reasoning_text if use_full else batch.evidence_reasoning_text
        task = self._text(task_text, device)
        condition = self._text(condition_text, device)
        reasoning = self._text(reasoning_text, device)
        b = task.shape[0]
        prefix = torch.stack(
            [task + self.type_embedding[0], condition + self.type_embedding[1], reasoning + self.type_embedding[2]], 1
        )
        anchor_states = []
        for index in range(b):
            availability = anchor_availability(batch, index, use_full=use_full, device=device)
            anchor_states.append(self.evidence_anchors + self.anchor_availability(availability))
        anchors = torch.stack(anchor_states) + self.type_embedding[2]
        masks = self.mask_embeddings[None].expand(b, -1, -1) + self.type_embedding[3]
        hidden = self.norm(self.encoder(torch.cat([prefix, anchors, masks], 1)))
        anchor_start = 3
        mask_start = anchor_start + anchors.shape[1]
        return SemanticEvidence(
            tokens=hidden[:, :mask_start],
            token_mask=torch.ones((b, mask_start), dtype=torch.bool, device=device),
            task_token=hidden[:, 0],
            condition_token=hidden[:, 1],
            global_token=hidden[:, anchor_start],
            mask_query_states=hidden[:, mask_start:],
            evidence_anchors=hidden[:, anchor_start:mask_start],
            visual_token_count=0,
            sequence_lengths=(int(hidden.shape[1]),) * b,
            visual_token_counts=(0,) * b,
        )


class QwenMaskQueryController(nn.Module):
    """Online NF4+QLoRA Qwen language decoder that produces PMRD queries."""

    def __init__(self, config: QPSalmConfig, device: torch.device, vision_bank: nn.Module | None) -> None:
        super().__init__()
        if device.type != "cuda" and not config.allow_qwen_cpu:
            raise RuntimeError("qwen_mask_query 需要 CUDA；本地结构测试请使用 controller=text_probe")
        try:
            from transformers import AutoTokenizer, BitsAndBytesConfig
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        except Exception as exc:  # pragma: no cover - optional production stack
            raise RuntimeError("qwen_mask_query 需要 transformers、peft 和 bitsandbytes") from exc

        checkpoint_mode = str(config.qwen_gradient_checkpointing)
        qwen_gradient_checkpointing_kwargs(checkpoint_mode)
        model_dir = validate_qwen_model_dir(config.qwen_model_path)
        if vision_bank is not None:
            cached_revision = str(vision_bank.manifest.get("model_revision") or "")
            expected_revision = local_model_revision(model_dir)
            if cached_revision not in {expected_revision, "hash-smoke"}:
                raise ValueError(
                    f"vision cache/model revision 不匹配: cache={cached_revision} model={expected_revision}"
                )
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
        model_cls = select_qwen_model_class()
        compute_dtype = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }.get(str(config.amp_dtype))
        if compute_dtype is None:
            raise ValueError(f"未知 amp_dtype={config.amp_dtype!r}")
        load_args: dict[str, Any] = {
            "trust_remote_code": True,
            "torch_dtype": compute_dtype,
            "attn_implementation": str(config.qwen_attn_implementation),
        }
        if config.qwen_4bit:
            load_args["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
            )
            load_args["device_map"] = {"": device.index or 0}
        model = model_cls.from_pretrained(str(model_dir), **load_args)
        self.vision_start_token_id = int(getattr(model.config, "vision_start_token_id", -1))
        self.vision_end_token_id = int(getattr(model.config, "vision_end_token_id", -1))
        if min(self.vision_start_token_id, self.vision_end_token_id) < 0:
            raise RuntimeError("Qwen config 缺少 vision_start_token_id/vision_end_token_id")
        if not config.qwen_4bit:
            model.to(device)
        if hasattr(model, "model") and hasattr(model.model, "visual"):
            model.model.visual = None
            if device.type == "cuda":
                torch.cuda.empty_cache()
        if config.qwen_4bit:
            # Quantization preparation must not also configure checkpointing.
            # The protocol is applied exactly once after LoRA injection below.
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        selected_layers = self._last_language_layer_indices(
            model, int(config.qwen_lora_last_n_layers)
        )
        if config.qwen_lora_trainable:
            lora = LoraConfig(
                r=int(config.qwen_lora_rank),
                lora_alpha=int(config.qwen_lora_alpha),
                lora_dropout=float(config.qwen_lora_dropout),
                bias="none",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                layers_to_transform=list(selected_layers),
                layers_pattern="layers",
                task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(model, lora)
            self.lora_config = copy.deepcopy(lora)
            # FP16 autocast still needs FP32 adapter master weights/gradients.
            # Otherwise GradScaler unscale can erase very small LoRA updates.
            for name, parameter in self.model.named_parameters():
                if "lora_" in name:
                    parameter.data = parameter.data.float()
        else:
            self.model = model
            self.lora_config = None
        self.gradient_checkpointing_mode = checkpoint_mode
        self.gradient_checkpointing_kwargs = configure_qwen_gradient_checkpointing(
            self.model, checkpoint_mode
        )
        self.model.config.use_cache = False
        text_config = getattr(model.config, "text_config", None) or getattr(model.config, "llm_config", None)
        hidden = int(getattr(text_config, "hidden_size", getattr(model.config, "hidden_size", 2048)))
        self.hidden_size = hidden
        self.dim = int(config.decoder_dim)
        self.num_queries = int(config.num_mask_tokens)
        self.max_text_tokens = int(config.qwen_max_text_tokens)
        self.view_tokens_per_view = int(config.qwen_view_tokens_per_view)
        self.view_pooling = str(config.qwen_view_pooling)
        if self.view_pooling not in {"tokens", "image-end", "attention"}:
            raise ValueError(f"未知 qwen_view_pooling={self.view_pooling!r}")
        self.visual_ablation = str(config.visual_ablation)
        self.vision_bank = vision_bank
        self.text_type = nn.Parameter(torch.randn(3, hidden, device=device, dtype=torch.float32) * 0.01)
        self.view_description_type = nn.Parameter(torch.randn(hidden, device=device, dtype=torch.float32) * 0.01)
        self.view_attention_query = nn.Parameter(torch.randn(hidden, device=device, dtype=torch.float32) * 0.02)
        self.evidence_anchors = nn.Parameter(
            torch.randn(1 + len(EVIDENCE_FAMILIES), hidden, device=device, dtype=torch.float32) * 0.02
        )
        self.anchor_availability = nn.Embedding(2, hidden).to(device=device, dtype=torch.float32)
        nn.init.normal_(self.anchor_availability.weight, std=0.02)
        self.mask_embeddings = nn.Parameter(
            torch.randn(self.num_queries, hidden, device=device, dtype=torch.float32) * 0.02
        )
        view_dim = int(getattr(vision_bank, "token_dim", self.dim))
        self.view_to_hidden = nn.Linear(view_dim, hidden).to(device=device, dtype=torch.float32)
        self._initialize_view_projection(self.view_to_hidden)
        self.visual_family_embedding = nn.Embedding(len(VISUAL_FAMILY_IDS), hidden).to(
            device=device, dtype=torch.float32
        )
        nn.init.normal_(self.visual_family_embedding.weight, std=0.02)
        self.output_projection = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, self.dim)).to(
            device=device, dtype=torch.float32
        )
        unexpected = [
            name for name, parameter in self.model.named_parameters()
            if parameter.requires_grad and "lora_" not in name
        ]
        if unexpected:
            raise RuntimeError(f"Qwen base 参数意外可训练，QLoRA 隔离失败: {unexpected[:8]}")
        non_fp32_lora = [
            (name, str(parameter.dtype))
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad and "lora_" in name and parameter.dtype != torch.float32
        ]
        if non_fp32_lora:
            raise RuntimeError(f"QLoRA adapter 必须保持 FP32 master weights: {non_fp32_lora[:8]}")
        self.lora_layer_indices = tuple(selected_layers) if config.qwen_lora_trainable else ()
        self.lora_module_names = (
            self._validate_lora_injection(selected_layers)
            if config.qwen_lora_trainable
            else ()
        )

    def ensure_named_adapter(self, adapter_name: str) -> None:
        """Create one explicit PEFT adapter with the segmentation LoRA protocol."""
        if not adapter_name or adapter_name == "default":
            return
        peft_config = getattr(self.model, "peft_config", None)
        if not isinstance(peft_config, dict) or "default" not in peft_config:
            raise RuntimeError("命名 description adapter 需要已启用的 Qwen PEFT default adapter")
        if adapter_name not in peft_config:
            self.model.add_adapter(adapter_name, copy.deepcopy(peft_config["default"]))
            for name, parameter in self.model.named_parameters():
                if f".{adapter_name}." in name and "lora_" in name:
                    parameter.data = parameter.data.float()
        if adapter_name not in self.model.peft_config:
            raise RuntimeError(f"PEFT adapter 创建失败: {adapter_name}")

    @contextmanager
    def adapter_scope(self, adapter_name: str):
        """Activate one adapter without letting PEFT rewrite optimizer ownership.

        ``PeftModel.set_adapter`` also toggles ``requires_grad``.  A description
        forward exits this context before ``loss.backward()``, so merely
        restoring the previous adapter can silently freeze the description
        leaves that already participate in the graph.  Preserve and restore
        the optimizer-selected flags independently from adapter activation.
        """
        self.ensure_named_adapter(adapter_name)
        active = tuple(getattr(self.model, "active_adapters", ("default",)))
        previous = active[0] if active else "default"
        trainability = {
            name: bool(parameter.requires_grad)
            for name, parameter in self.model.named_parameters()
            if "lora_" in name
        }
        self.model.set_adapter(adapter_name)
        # PEFT activates an adapter by toggling its leaves trainable. Reapply
        # the optimizer-owned flags before forward so D2 can consume the
        # D0/D1 desc_adapter representation without updating that adapter.
        for name, parameter in self.model.named_parameters():
            if name in trainability:
                parameter.requires_grad_(trainability[name])
        try:
            observed = tuple(getattr(self.model, "active_adapters", ()))
            if observed != (adapter_name,):
                raise RuntimeError(f"PEFT adapter 激活失败: expected={adapter_name} observed={observed}")
            yield
        finally:
            self.model.set_adapter(previous)
            for name, parameter in self.model.named_parameters():
                if name in trainability:
                    parameter.requires_grad_(trainability[name])

    @staticmethod
    def _initialize_view_projection(projection: nn.Linear) -> None:
        if projection.in_features == projection.out_features:
            nn.init.eye_(projection.weight)
        else:
            nn.init.xavier_uniform_(projection.weight)
        if projection.bias is not None:
            nn.init.zeros_(projection.bias)

    @staticmethod
    def _last_language_layer_indices(model: nn.Module, last_n: int) -> tuple[int, ...]:
        container = getattr(model, "model", None)
        language = getattr(container, "language_model", None)
        layers = getattr(language, "layers", None)
        if layers is None:
            raise RuntimeError("当前 Qwen3-VL 结构缺少 model.language_model.layers")
        count = len(layers)
        if count <= 0:
            raise RuntimeError("Qwen language decoder 不包含 transformer layers")
        width = min(max(1, int(last_n)), count)
        return tuple(range(count - width, count))

    def _lora_projection_modules(self) -> dict[str, nn.Module]:
        return {
            name: module
            for name, module in self.model.named_modules()
            if hasattr(module, "lora_A") and hasattr(module, "lora_B")
        }

    def _validate_lora_injection(self, selected_layers: Sequence[int]) -> tuple[str, ...]:
        modules = self._lora_projection_modules()
        expected = len(tuple(selected_layers)) * 4
        if len(modules) != expected:
            raise RuntimeError(
                "Qwen LoRA projection 数量不符合预期: "
                f"observed={len(modules)} expected={expected} names={list(modules)[:8]}"
            )
        selected = set(int(value) for value in selected_layers)
        for name, module in modules.items():
            if not any(f"layers.{index}." in name for index in selected):
                raise RuntimeError(f"LoRA 注入到非目标 language layer: {name}")
            active = tuple(getattr(module, "active_adapters", ()))
            if not active:
                raise RuntimeError(f"LoRA projection 没有 active adapter: {name}")
            if bool(getattr(module, "disable_adapters", False)):
                raise RuntimeError(f"LoRA projection 被禁用: {name}")
            if tuple(getattr(module, "merged_adapters", ())) or bool(getattr(module, "merged", False)):
                raise RuntimeError(f"LoRA projection 已 merged，无法进行 QLoRA 训练: {name}")
        return tuple(sorted(modules))

    def lora_runtime_status(self) -> dict[str, Any]:
        modules = self._lora_projection_modules()
        return {
            "selected_layers": list(self.lora_layer_indices),
            "module_names": sorted(modules),
            "module_count": len(modules),
            "active_adapters": {
                name: list(getattr(module, "active_adapters", ()))
                for name, module in modules.items()
            },
            "disabled_modules": sorted(
                name for name, module in modules.items()
                if bool(getattr(module, "disable_adapters", False))
            ),
            "merged_modules": sorted(
                name for name, module in modules.items()
                if tuple(getattr(module, "merged_adapters", ())) or bool(getattr(module, "merged", False))
            ),
        }

    @contextmanager
    def trace_lora_execution(self):
        counts = {name: 0 for name in self.lora_module_names}
        handles = []
        for name, module in self._lora_projection_modules().items():
            def record(_module, _inputs, _output, *, module_name=name):
                counts[module_name] += 1

            handles.append(module.register_forward_hook(record))
        try:
            yield counts
        finally:
            for handle in handles:
                handle.remove()

    @lru_cache(maxsize=32768)
    def _cached_segment_ids(self, text: str) -> tuple[int, ...]:
        encoded = self.tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_text_tokens,
        )["input_ids"]
        if torch.is_tensor(encoded):
            if encoded.ndim > 1:
                encoded = encoded[0]
            encoded = encoded.tolist()
        if encoded and isinstance(encoded[0], list):
            encoded = encoded[0]
        if not encoded:
            encoded = [self.tokenizer.eos_token_id]
        return tuple(int(value) for value in encoded)

    def _segment_ids(self, text: str, device: torch.device) -> torch.Tensor:
        return torch.tensor(self._cached_segment_ids(text), dtype=torch.long, device=device)

    def _visual_tokens(self, batch: ModalityBatch, use_full: bool, device: torch.device):
        if self.vision_bank is None:
            return (
                torch.zeros((batch.batch_size, 1, self.view_to_hidden.in_features), device=device),
                torch.zeros((batch.batch_size, 1), dtype=torch.bool, device=device),
                torch.zeros((batch.batch_size, 1), dtype=torch.long, device=device),
                [[] for _ in range(batch.batch_size)],
            )
        if use_full:
            subsets = [
                type(subset)(
                    active_names=tuple(item.name for item in instances),
                    dropped_names=(),
                    signature="teacher-full",
                    is_full=True,
                )
                for subset, instances in zip(batch.active_subsets, batch.full_instances)
            ]
        else:
            subsets = batch.active_subsets
        tokens, mask, _, family_ids, segments = self.vision_bank.tokens_for(
            batch.visual_evidence_key, subsets, device, self.view_tokens_per_view
        )
        return tokens, mask, family_ids, segments

    def _pool_view_chunk(self, chunk: torch.Tensor) -> torch.Tensor:
        if not chunk.numel() or self.view_pooling == "tokens":
            return chunk
        if self.view_pooling == "image-end":
            return chunk[-1:]
        query = self.view_attention_query.to(device=chunk.device, dtype=chunk.dtype)
        score = (chunk * query).sum(-1).float() / max(float(chunk.shape[-1]) ** 0.5, 1.0)
        return (torch.softmax(score, 0).to(chunk.dtype)[:, None] * chunk).sum(0, keepdim=True)

    def _encode_once(
        self,
        batch: ModalityBatch,
        *,
        use_full: bool,
        include_visual: bool,
    ) -> SemanticEvidence:
        device = next(self.model.parameters()).device
        tasks = batch.full_proposal_context_text if use_full else batch.proposal_context_text
        conditions = batch.full_condition_prompt_text if use_full else batch.condition_prompt_text
        reasons = batch.full_evidence_reasoning_text if use_full else batch.evidence_reasoning_text
        visual, visual_mask, visual_family_ids, visual_segments = self._visual_tokens(batch, use_full, device)
        if not include_visual:
            visual_mask = torch.zeros_like(visual_mask)
            visual_segments = [[] for _ in range(batch.batch_size)]
        embedding = self.model.get_input_embeddings()
        vocabulary_size = int(embedding.weight.shape[0])
        if max(self.vision_start_token_id, self.vision_end_token_id) >= vocabulary_size:
            raise RuntimeError(
                "Qwen vision delimiter token 超出 embedding vocabulary: "
                f"start={self.vision_start_token_id} end={self.vision_end_token_id} vocab={vocabulary_size}"
            )
        vision_start = embedding(torch.tensor([self.vision_start_token_id], device=device))
        vision_end = embedding(torch.tensor([self.vision_end_token_id], device=device))
        sequences = []
        positions = []
        for index, (task, condition, reason) in enumerate(zip(tasks, conditions, reasons)):
            task_embed = embedding(self._segment_ids(task, device))
            condition_embed = embedding(self._segment_ids(condition, device))
            reason_embed = embedding(self._segment_ids(reason, device))
            input_dtype = task_embed.dtype
            task_embed = task_embed + self.text_type[0].to(input_dtype)
            condition_embed = condition_embed + self.text_type[1].to(input_dtype)
            reason_embed = reason_embed + self.text_type[2].to(input_dtype)
            active_types = visual_family_ids[index, visual_mask[index]]
            projected_visual = (
                self.view_to_hidden(visual[index, visual_mask[index]].float())
                + self.visual_family_embedding(active_types)
            ).to(input_dtype)
            visual_parts = []
            visual_offset = 0
            pooled_visual_count = 0
            for description, token_count in visual_segments[index]:
                description_ids = self._segment_ids(description, device)[: min(48, self.max_text_tokens)]
                description_embed = (
                    embedding(description_ids) + self.view_description_type.to(input_dtype)
                )
                view_chunk = self._pool_view_chunk(
                    projected_visual[visual_offset:visual_offset + token_count]
                )
                visual_parts.extend([
                    description_embed,
                    vision_start.to(input_dtype),
                    view_chunk,
                    vision_end.to(input_dtype),
                ])
                pooled_visual_count += int(view_chunk.shape[0])
                visual_offset += token_count
            if visual_offset != projected_visual.shape[0]:
                raise RuntimeError(
                    f"Qwen view segment/token 数量不一致: segments={visual_offset} "
                    f"tokens={projected_visual.shape[0]}"
                )
            active_visual = (
                torch.cat(visual_parts, 0)
                if visual_parts
                else projected_visual.new_zeros((0, self.hidden_size))
            )
            availability = anchor_availability(batch, index, use_full=use_full, device=device)
            anchors = (self.evidence_anchors + self.anchor_availability(availability)).to(input_dtype)
            masks = self.mask_embeddings.to(input_dtype)
            task_pos = task_embed.shape[0] - 1
            condition_pos = task_embed.shape[0] + condition_embed.shape[0] - 1
            anchor_start = task_embed.shape[0] + condition_embed.shape[0] + reason_embed.shape[0] + active_visual.shape[0]
            mask_start = anchor_start + anchors.shape[0]
            sequence = torch.cat([task_embed, condition_embed, reason_embed, active_visual, anchors, masks], 0)
            sequences.append(sequence)
            positions.append((task_pos, condition_pos, anchor_start, mask_start, pooled_visual_count))
        inputs, attention, _lengths = pad_qwen_sequences(sequences)
        if self.training and not inputs.requires_grad:
            raise RuntimeError(
                "Qwen inputs_embeds 缺少梯度链路；请检查动态 padding、controller embeddings 和 LoRA 初始化"
            )
        # BitsAndBytes already owns the NF4 base compute dtype. Keep QLoRA
        # master weights and controller projections outside the dense
        # segmentation autocast context so their autograd path is identical
        # in controller-only probes and end-to-end training.
        with torch.amp.autocast(device_type=device.type, enabled=False):
            outputs = self.model(
                inputs_embeds=inputs,
                attention_mask=attention,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
                logits_to_keep=1,
            )
            hidden_states = getattr(outputs, "hidden_states", None)
            if not hidden_states:
                raise RuntimeError("PEFT Qwen forward 未返回 language hidden states")
            hidden = hidden_states[-1]
            task_states, condition_states, anchors, masks = [], [], [], []
            for index, (task_pos, condition_pos, anchor_start, mask_start, _) in enumerate(positions):
                task_states.append(hidden[index, task_pos])
                condition_states.append(hidden[index, condition_pos])
                anchors.append(hidden[index, anchor_start:mask_start])
                masks.append(hidden[index, mask_start:mask_start + self.num_queries])
            task_out = self.output_projection(torch.stack(task_states).float())
            condition_out = self.output_projection(torch.stack(condition_states).float())
            anchor_out = self.output_projection(torch.stack(anchors).float())
            mask_out = self.output_projection(torch.stack(masks).float())
        if self.training and not mask_out.requires_grad:
            raise RuntimeError("Qwen mask-query states 缺少梯度链路")
        token_out = torch.cat([task_out[:, None], condition_out[:, None], anchor_out], 1)
        return SemanticEvidence(
            tokens=token_out,
            token_mask=torch.ones(token_out.shape[:2], dtype=torch.bool, device=device),
            task_token=task_out,
            condition_token=condition_out,
            global_token=anchor_out[:, 0],
            mask_query_states=mask_out,
            evidence_anchors=anchor_out,
            visual_token_count=sum(value[-1] for value in positions),
            sequence_lengths=tuple(int(value.shape[0]) for value in sequences),
            visual_token_counts=tuple(int(value[-1]) for value in positions),
        )

    def encode_batch(self, batch: ModalityBatch, *, use_full: bool = False) -> SemanticEvidence:
        include_visual = self.visual_ablation != "text-only"
        semantic = self._encode_once(batch, use_full=use_full, include_visual=include_visual)
        if self.visual_ablation != "image-text-delta":
            return semantic

        # The delta branch isolates the incremental post-context evidence
        # contributed by visual view tokens. PMRD queries remain the full
        # image-text states, while QMEF/verifier consume delta anchors.
        text_only = self._encode_once(batch, use_full=use_full, include_visual=False)
        if semantic.evidence_anchors is None or text_only.evidence_anchors is None:
            raise RuntimeError("image-text-delta 需要 Qwen evidence anchors")
        delta = semantic.evidence_anchors - text_only.evidence_anchors
        tokens = torch.cat([semantic.task_token[:, None], semantic.condition_token[:, None], delta], 1)
        return SemanticEvidence(
            tokens=tokens,
            token_mask=torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device),
            task_token=semantic.task_token,
            condition_token=semantic.condition_token,
            global_token=delta[:, 0],
            mask_query_states=semantic.mask_query_states,
            evidence_anchors=delta,
            visual_token_count=semantic.visual_token_count,
            sequence_lengths=semantic.sequence_lengths,
            visual_token_counts=semantic.visual_token_counts,
            visual_delta_norm=delta.float().norm(dim=-1).mean(dim=1),
        )


def build_controller(config: QPSalmConfig, device: torch.device, vision_bank: nn.Module | None = None) -> nn.Module:
    if config.controller == "text_probe":
        return TextProbeMaskController(config.decoder_dim, config.num_mask_tokens, config.num_heads)
    if config.controller == "qwen_mask_query":
        return QwenMaskQueryController(config, device, vision_bank)
    raise ValueError(f"v2 不支持 controller={config.controller!r}; expected text_probe/qwen_mask_query")
