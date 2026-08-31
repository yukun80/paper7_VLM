"""Shared MLLM 后端的模型无关合同。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol, Sequence

from torch import Tensor

class VLMProcessorAdapter(Protocol):
    """训练、视觉推理与 text-only 推理共用的 processor 合同。"""

    pad_token_id: int

    def clone_for_worker(self) -> "VLMProcessorAdapter": ...

    def identity(self) -> Mapping[str, Any]: ...

    def encode_training(
        self,
        messages: Sequence[Mapping[str, Any]],
    ) -> Any: ...

    def encode_inference(
        self,
        messages: Sequence[Mapping[str, Any]],
    ) -> Any: ...

    def encode_text_inference(
        self,
        messages: Sequence[Mapping[str, Any]],
    ) -> Any: ...


class VLMModelAdapter(Protocol):
    """训练、checkpoint 与生成共用的模型合同。"""

    identity: Any
    trainable_parameter_count: int

    @property
    def trainable_names(self) -> tuple[str, ...]: ...

    def train(self) -> None: ...

    def eval(self) -> None: ...

    def forward(self, batch: Mapping[str, Any]) -> Any: ...

    def generate_text(
        self,
        batch: Mapping[str, Any],
        *,
        processor: Any,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        top_p: float,
    ) -> list[str]: ...

    def trainable_state_dict(self) -> dict[str, Tensor]: ...

    def load_trainable_state_dict(
        self,
        state: Mapping[str, Tensor],
    ) -> None: ...


ProcessorBuilder = Callable[[Any], VLMProcessorAdapter]
ModelBuilder = Callable[[Any, Any, bool], VLMModelAdapter]


@dataclass(frozen=True)
class VLMBackendSpec:
    """一个模型家族的不可变构造与身份规格。"""

    name: str
    model_types: tuple[str, ...]
    architectures: tuple[str, ...]
    processor_classes: tuple[str, ...]
    supports_thinking: bool
    processor_builder: ProcessorBuilder
    model_builder: ModelBuilder

    def __post_init__(self) -> None:
        if (
            not self.name
            or not self.name.replace("_", "a").isalnum()
            or not self.name[0].isalpha()
            or self.name != self.name.lower()
        ):
            raise ValueError(f"非法 backend name：{self.name!r}")
        for label, values in (
            ("model_types", self.model_types),
            ("architectures", self.architectures),
            ("processor_classes", self.processor_classes),
        ):
            if (
                not values
                or len(values) != len(set(values))
                or any(not isinstance(value, str) or not value for value in values)
            ):
                raise ValueError(f"{self.name}.{label} 必须是非空无重复字符串元组")
        if not isinstance(self.supports_thinking, bool):
            raise ValueError("supports_thinking 必须是 bool")
        if not callable(self.processor_builder) or not callable(self.model_builder):
            raise ValueError("backend builders 必须可调用")
