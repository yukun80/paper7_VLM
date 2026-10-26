"""Qwen3.5-4B full-attention LoRA 模型合同。"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import unittest

import torch
from torch import nn

from oa_groundrag.vlm.backends.qwen3_5.model import (
    EXPECTED_QWEN35_4B_CHECKPOINT_PARAMETERS,
    EXPECTED_QWEN35_4B_RUNTIME_UNIQUE_PARAMETERS,
    EXPECTED_QWEN35_FULL_ATTENTION_LORA_R8_PARAMETERS,
    EXPECTED_QWEN35_LORA_TENSORS,
    QWEN35_FULL_ATTENTION_LAYERS,
    Qwen35ModelAdapter,
    Qwen35ModelIdentity,
    QWEN35_UNUSED_MTP_TENSORS,
    _official_config_contract,
)
from oa_groundrag.vlm.config import load_config
from oa_groundrag.vlm.errors import ModelError, ReasonCode


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = REPOSITORY_ROOT / "models_zoo/Qwen3.5-4B-r851bf6e8"
CONFIG_ROOT = REPOSITORY_ROOT / "configs/vlm/rs_general"


def _identity(*, backend: str = "qwen3_5", official: bool = False):
    return Qwen35ModelIdentity(
        backend=backend,
        model_path=str(MODEL_ROOT),
        hub_repo_id="Qwen/Qwen3.5-4B",
        hub_revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        asset_ledger_path="fixture-ledger.json",
        asset_ledger_file_sha256="a" * 64,
        asset_ledger_payload_sha256="b" * 64,
        model_type="qwen3_5",
        architecture="Qwen3_5ForConditionalGeneration",
        config_sha256="c" * 64,
        weights_index_sha256="d" * 64,
        base_parameter_count=(
            EXPECTED_QWEN35_4B_RUNTIME_UNIQUE_PARAMETERS if official else 1
        ),
        checkpoint_parameter_count=EXPECTED_QWEN35_4B_CHECKPOINT_PARAMETERS,
        unused_mtp_parameter_count=120_599_552,
        full_attention_layers=QWEN35_FULL_ATTENTION_LAYERS,
    )


class _LoRABranch(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones((1, 1)))


class _LoRAProjection(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lora_A = nn.ModuleDict({"default": _LoRABranch()})
        self.lora_B = nn.ModuleDict({"default": _LoRABranch()})


class _Attention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(self, name, _LoRAProjection())


class _Layer(nn.Module):
    def __init__(self, full: bool) -> None:
        super().__init__()
        if full:
            self.self_attn = _Attention()
        else:
            self.linear_attn = nn.Identity()


class _TinyHybridModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        model = nn.Module()
        language_model = nn.Module()
        language_model.layers = nn.ModuleList(
            [_Layer(index in QWEN35_FULL_ATTENTION_LAYERS) for index in range(32)]
        )
        model.language_model = language_model
        model.visual = nn.Module()
        model.visual.merger = nn.Linear(1, 1)
        model.visual.requires_grad_(False)
        self.model = model
        self.last_use_cache = None

    def forward(
        self,
        input_ids,
        attention_mask=None,
        mm_token_type_ids=None,
        use_cache=None,
    ):
        del attention_mask, mm_token_type_ids
        self.last_use_cache = use_cache
        trainable = sum(
            parameter.sum()
            for parameter in self.parameters()
            if parameter.requires_grad
        )
        leading = trainable.expand((*input_ids.shape, 1))
        trailing = torch.zeros(
            (*input_ids.shape, 7),
            dtype=leading.dtype,
            device=leading.device,
        )
        return type("Output", (), {"logits": torch.cat((leading, trailing), -1)})()


class Qwen35ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.lora = load_config(
            CONFIG_ROOT / "rs_generaldesc_lora_qwen3vl_2b.yaml"
        ).adaptation

    def test_official_meta_topology_has_only_eight_full_attention_layers(self) -> None:
        from accelerate import init_empty_weights
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoConfig, AutoModelForMultimodalLM

        _official_config_contract(MODEL_ROOT)
        config = AutoConfig.from_pretrained(MODEL_ROOT, local_files_only=True)
        with init_empty_weights():
            model = AutoModelForMultimodalLM.from_config(
                config,
                attn_implementation="sdpa",
            )
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            model = get_peft_model(
                model,
                LoraConfig(
                    r=8,
                    lora_alpha=16,
                    lora_dropout=0.05,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                    bias="none",
                    task_type=TaskType.CAUSAL_LM,
                ),
            )
        adapter = Qwen35ModelAdapter(
            model,
            identity=_identity(official=True),
            adaptation=self.lora,
        )
        self.assertEqual(len(adapter.trainable_names), EXPECTED_QWEN35_LORA_TENSORS)
        self.assertEqual(
            adapter.trainable_parameter_count,
            EXPECTED_QWEN35_FULL_ATTENTION_LORA_R8_PARAMETERS,
        )
        self.assertEqual(len(QWEN35_UNUSED_MTP_TENSORS), 15)
        self.assertEqual(
            EXPECTED_QWEN35_4B_CHECKPOINT_PARAMETERS
            - EXPECTED_QWEN35_4B_RUNTIME_UNIQUE_PARAMETERS,
            120_599_552,
        )

    def test_tiny_forward_backward_and_state_round_trip(self) -> None:
        model = _TinyHybridModel()
        adapter = Qwen35ModelAdapter(
            model,
            identity=_identity(),
            adaptation=self.lora,
        )
        output = adapter.forward(
            {
                "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
                "attention_mask": torch.ones((1, 3), dtype=torch.long),
                "mm_token_type_ids": torch.zeros((1, 3), dtype=torch.long),
                "labels": torch.tensor([[-100, 0, 1]], dtype=torch.long),
            }
        )
        output.loss.backward()
        self.assertTrue(torch.isfinite(output.loss))
        self.assertIs(model.last_use_cache, False)
        self.assertTrue(
            all(
                parameter.grad is not None
                for parameter in model.parameters()
                if parameter.requires_grad
            )
        )
        state = adapter.trainable_state_dict()
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.data.zero_()
        adapter.load_trainable_state_dict(state)
        for name, value in adapter.trainable_state_dict().items():
            torch.testing.assert_close(value, state[name])

    def test_cross_family_identity_and_adapter_state_are_rejected(self) -> None:
        with self.assertRaises(ModelError) as identity_error:
            Qwen35ModelAdapter(
                _TinyHybridModel(),
                identity=_identity(backend="qwen3_vl"),
                adaptation=self.lora,
            )
        self.assertEqual(
            identity_error.exception.code,
            ReasonCode.MODEL_IDENTITY_MISMATCH,
        )
        adapter = Qwen35ModelAdapter(
            _TinyHybridModel(),
            identity=_identity(),
            adaptation=self.lora,
        )
        wrong = adapter.trainable_state_dict()
        wrong["qwen3_vl.adapter.weight"] = torch.ones(1)
        with self.assertRaises(ModelError) as state_error:
            adapter.load_trainable_state_dict(wrong)
        self.assertEqual(
            state_error.exception.code,
            ReasonCode.CHECKPOINT_INCOMPATIBLE,
        )

    def test_vision_trainable_and_hybrid_lora_are_rejected(self) -> None:
        model = _TinyHybridModel()
        model.model.visual.merger.weight.requires_grad_(True)
        with self.assertRaises(ModelError) as vision_error:
            Qwen35ModelAdapter(
                model,
                identity=_identity(),
                adaptation=self.lora,
            )
        self.assertEqual(
            vision_error.exception.code,
            ReasonCode.MODEL_IDENTITY_MISMATCH,
        )
        hybrid = replace(
            self.lora,
            target_modules=(*self.lora.target_modules, "in_proj_qkv"),
        )
        with self.assertRaises(ModelError) as hybrid_error:
            Qwen35ModelAdapter(
                _TinyHybridModel(),
                identity=_identity(),
                adaptation=hybrid,
            )
        self.assertEqual(
            hybrid_error.exception.code,
            ReasonCode.MODEL_IDENTITY_MISMATCH,
        )


if __name__ == "__main__":
    unittest.main()
