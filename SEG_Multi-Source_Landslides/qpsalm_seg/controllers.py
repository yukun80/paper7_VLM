#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Mask-query controllers for the v2 Qwen-PSALM architecture.

The production controller feeds task text, active-view tokens, evidence anchors,
and learned mask positions through the Qwen language decoder. Historical
text-only cache formats are intentionally unsupported.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
import re
from typing import Any, Sequence

import torch
from torch import nn

from .config import QPSalmConfig
from .paths import resolve_repo_path
from .schema import MODALITY_FAMILIES, MODALITY_FAMILY_IDS, ModalityBatch, SemanticEvidence


EVIDENCE_FAMILIES = MODALITY_FAMILIES
VISUAL_FAMILY_IDS = MODALITY_FAMILY_IDS
CONTROLLER_SEQUENCE_PROTOCOL = "qwen_mask_query_interleaved_views_v2"


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
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
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
        load_args: dict[str, Any] = {"trust_remote_code": True, "torch_dtype": torch.bfloat16}
        if config.qwen_4bit:
            load_args["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
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
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        target_regex = self._last_layer_projection_regex(model, int(config.qwen_lora_last_n_layers))
        lora = LoraConfig(
            r=int(config.qwen_lora_rank),
            lora_alpha=int(config.qwen_lora_alpha),
            lora_dropout=float(config.qwen_lora_dropout),
            bias="none",
            target_modules=target_regex,
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(model, lora)
        self.model.gradient_checkpointing_enable()
        if hasattr(self.model, "enable_input_require_grads"):
            self.model.enable_input_require_grads()
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

    def _language_model(self) -> nn.Module:
        base = self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model
        container = getattr(base, "model", None)
        language = getattr(container, "language_model", None)
        if language is None:
            raise RuntimeError("当前 Qwen3-VL 结构缺少 model.language_model")
        return language

    @staticmethod
    def _initialize_view_projection(projection: nn.Linear) -> None:
        if projection.in_features == projection.out_features:
            nn.init.eye_(projection.weight)
        else:
            nn.init.xavier_uniform_(projection.weight)
        if projection.bias is not None:
            nn.init.zeros_(projection.bias)

    @staticmethod
    def _last_layer_projection_regex(model: nn.Module, last_n: int) -> str:
        candidates: list[tuple[int, str]] = []
        pattern = re.compile(r"(?:language_model|text_model|model)\.layers\.(\d+)\..*(q_proj|k_proj|v_proj|o_proj)$")
        for name, _ in model.named_modules():
            if "visual" in name or "vision" in name:
                continue
            match = pattern.search(name)
            if match:
                candidates.append((int(match.group(1)), name))
        if not candidates:
            raise RuntimeError("无法定位 Qwen language decoder 的 q/k/v/o projection")
        indices = sorted({index for index, _ in candidates})
        selected = set(indices[-max(1, last_n):])
        names = [re.escape(name) for index, name in candidates if index in selected]
        return "(?:" + "|".join(names) + ")"

    def _segment_ids(self, text: str, device: torch.device) -> torch.Tensor:
        encoded = self.tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_text_tokens,
            return_tensors="pt",
        )["input_ids"][0].to(device)
        if encoded.numel() == 0:
            encoded = torch.tensor([self.tokenizer.eos_token_id], device=device)
        return encoded

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
        length = max(value.shape[0] for value in sequences)
        inputs = sequences[0].new_zeros((len(sequences), length, self.hidden_size))
        attention = torch.zeros((len(sequences), length), dtype=torch.long, device=device)
        for index, sequence in enumerate(sequences):
            inputs[index, :sequence.shape[0]] = sequence
            attention[index, :sequence.shape[0]] = 1
        outputs = self._language_model()(
            inputs_embeds=inputs,
            attention_mask=attention,
            output_hidden_states=False,
            return_dict=True,
            use_cache=False,
        )
        hidden = outputs.last_hidden_state
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
            visual_delta_norm=delta.float().norm(dim=-1).mean(dim=1),
        )


def build_controller(config: QPSalmConfig, device: torch.device, vision_bank: nn.Module | None = None) -> nn.Module:
    if config.controller == "text_probe":
        return TextProbeMaskController(config.decoder_dim, config.num_mask_tokens, config.num_heads)
    if config.controller == "qwen_mask_query":
        return QwenMaskQueryController(config, device, vision_bank)
    raise ValueError(f"v2 不支持 controller={config.controller!r}; expected text_probe/qwen_mask_query")
