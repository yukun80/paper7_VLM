"""Qwen3.5 训练专属监督位置 logits 与数值等价测试。"""

from __future__ import annotations

import copy
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from oa_groundrag.training.vlm.qwen35_supervised import (
    Qwen35SupervisedPositionAdapter,
    causal_supervised_positions,
    supervised_position_causal_loss,
)
from oa_groundrag.vlm.backends.modeling import assistant_sample_mean_causal_loss
from oa_groundrag.vlm.errors import ModelError


class ToyQwen35(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(17, 7)
        self.lm_head = nn.Linear(7, 11, bias=False)
        self.last_logits_to_keep = None

    def forward(self, input_ids, logits_to_keep=0, **kwargs):
        hidden = self.embedding(input_ids)
        indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        self.last_logits_to_keep = logits_to_keep
        return SimpleNamespace(logits=self.lm_head(hidden[:, indices, :]))


class Identity:
    def __init__(self, backend="qwen3_5", model_type="qwen3_5") -> None:
        self.backend = backend
        self.model_type = model_type

    def to_dict(self):
        return {"backend": self.backend, "model_type": self.model_type}


class ToyAdapter:
    def __init__(self, model=None, *, backend="qwen3_5") -> None:
        self.model = model or ToyQwen35()
        self.identity = Identity(backend=backend)
        self.trainable_names = tuple(name for name, _ in self.model.named_parameters())
        self.trainable_parameter_count = sum(
            parameter.numel() for parameter in self.model.parameters()
        )
        self.loaded = None

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def trainable_state_dict(self):
        return self.model.state_dict()

    def load_trainable_state_dict(self, state):
        self.loaded = dict(state)

    def generate_text(self, *args, **kwargs):
        return ["delegated"]


class Qwen35SupervisedPositionTests(unittest.TestCase):
    def test_first_label_is_not_a_prediction_target(self) -> None:
        labels = torch.tensor([[4, 5, -100]])
        shifted, positions = causal_supervised_positions(labels)
        self.assertEqual(shifted.tolist(), [[5, -100, -100]])
        self.assertEqual(positions.tolist(), [0])

    def test_contiguous_and_noncontiguous_positions(self) -> None:
        contiguous = torch.tensor([[-100, 1, 2, 3]])
        _, positions = causal_supervised_positions(contiguous)
        self.assertEqual(positions.tolist(), [0, 1, 2])
        noncontiguous = torch.tensor([[-100, 1, -100, 2, -100, 3]])
        _, positions = causal_supervised_positions(noncontiguous)
        self.assertEqual(positions.tolist(), [0, 2, 4])

    def test_empty_supervision_and_batch_greater_than_one_fail_closed(self) -> None:
        with self.assertRaises(ModelError):
            causal_supervised_positions(torch.full((1, 4), -100))
        with self.assertRaises(ModelError):
            causal_supervised_positions(torch.ones((2, 4), dtype=torch.long))

    def test_wrong_backend_fails_closed(self) -> None:
        with self.assertRaises(ModelError):
            Qwen35SupervisedPositionAdapter(ToyAdapter(backend="qwen3_vl"))

    def test_wrapper_passes_one_dimensional_positions_and_delegates(self) -> None:
        base = ToyAdapter()
        wrapper = Qwen35SupervisedPositionAdapter(base)
        labels = torch.tensor([[-100, 1, -100, 2, 3]])
        result = wrapper.forward(
            {
                "input_ids": torch.tensor([[2, 3, 4, 5, 6]]),
                "attention_mask": torch.ones((1, 5), dtype=torch.long),
                "labels": labels,
            }
        )
        self.assertEqual(base.model.last_logits_to_keep.tolist(), [0, 2, 3])
        self.assertEqual(result.logits.shape, (1, 3, 11))
        self.assertEqual(result.loss.dtype, torch.float32)
        self.assertEqual(wrapper.generate_text(), ["delegated"])
        wrapper.load_trainable_state_dict({"fixture": torch.ones(1)})
        self.assertEqual(set(base.loaded), {"fixture"})

    def test_loss_and_all_trainable_gradients_match_full_logits_reference(self) -> None:
        torch.manual_seed(20260902)
        full_model = ToyQwen35()
        projected_model = copy.deepcopy(full_model)
        input_ids = torch.tensor([[2, 3, 4, 5, 6, 7]])
        labels = torch.tensor([[-100, -100, 1, -100, 2, 3]])

        full_logits = full_model(input_ids=input_ids).logits
        full_loss = assistant_sample_mean_causal_loss(full_logits, labels)
        full_loss.backward()

        wrapper = Qwen35SupervisedPositionAdapter(ToyAdapter(projected_model))
        projected = wrapper.forward(
            {
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids),
                "labels": labels,
            }
        )
        projected.loss.backward()

        self.assertTrue(
            torch.allclose(full_loss, projected.loss, atol=1e-4, rtol=1e-3)
        )
        full_gradients = []
        projected_gradients = []
        for (_, full_parameter), (_, projected_parameter) in zip(
            full_model.named_parameters(),
            projected_model.named_parameters(),
            strict=True,
        ):
            self.assertIsNotNone(full_parameter.grad)
            self.assertIsNotNone(projected_parameter.grad)
            full_gradients.append(full_parameter.grad.flatten())
            projected_gradients.append(projected_parameter.grad.flatten())
        full_vector = torch.cat(full_gradients)
        projected_vector = torch.cat(projected_gradients)
        relative_l2 = torch.linalg.vector_norm(projected_vector - full_vector) / torch.linalg.vector_norm(full_vector)
        cosine = torch.nn.functional.cosine_similarity(
            full_vector,
            projected_vector,
            dim=0,
        )
        self.assertLessEqual(float(relative_l2), 1e-3)
        self.assertGreaterEqual(float(cosine), 0.9999)

    def test_selected_loss_rejects_unsorted_positions(self) -> None:
        logits = torch.randn(1, 2, 11)
        shifted = torch.tensor([[1, 2, -100]])
        with self.assertRaises(ModelError):
            supervised_position_causal_loss(
                logits,
                shifted,
                torch.tensor([1, 0]),
            )


if __name__ == "__main__":
    unittest.main()

