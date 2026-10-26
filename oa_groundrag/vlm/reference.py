"""Phase 4 主参考项目的可审计身份与本地设计映射。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ReferenceProject:
    name: str
    repository_url: str
    paper_url: str
    revision: str
    revision_date: str
    inspected_date: str
    license: str
    inspected_files: tuple[str, ...]
    traced_sample_path: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "repository_url": self.repository_url,
            "paper_url": self.paper_url,
            "revision": self.revision,
            "revision_date": self.revision_date,
            "inspected_date": self.inspected_date,
            "license": self.license,
            "inspected_files": list(self.inspected_files),
            "traced_sample_path": list(self.traced_sample_path),
        }


MAIN_REFERENCE = ReferenceProject(
    name="Qwen3-VL",
    repository_url="https://github.com/QwenLM/Qwen3-VL",
    paper_url="https://arxiv.org/abs/2511.21631",
    revision="96588727e44c78b25ba03ea03b8e12f7e64fd0da",
    revision_date="2026-01-30",
    inspected_date="2026-07-30",
    license="Apache-2.0",
    inspected_files=(
        "LICENSE",
        "qwen-vl-finetune/README.md",
        "qwen-vl-finetune/qwenvl/train/argument.py",
        "qwen-vl-finetune/qwenvl/data/__init__.py",
        "qwen-vl-finetune/qwenvl/data/data_processor.py",
        "qwen-vl-finetune/qwenvl/train/train_qwen.py",
        "qwen-vl-finetune/qwenvl/train/trainer.py",
        "README.md (Transformers quickstart)",
        "evaluation/RealWorldQA/README.md",
        "evaluation/RealWorldQA/run_realworldqa.py",
    ),
    traced_sample_path=(
        "ModelArguments/DataArguments/TrainingArguments",
        "data_list",
        "LazySupervisedDataset",
        "_build_messages",
        "preprocess_qwen_visual/apply_chat_template",
        "SupervisedDatasetCollator",
        "Qwen3VL forward/causal loss",
        "Trainer inference checkpoint",
        "Transformers quickstart or RealWorldQA vLLM generate",
        "RealWorldQA prediction JSONL",
        "RealWorldQA rule/optional-judge evaluator",
    ),
)
