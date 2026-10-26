"""Shared RS-Geohazard MLLM 的惰性公共接口。"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    "CheckpointManager": (".checkpoint", "CheckpointManager"),
    "VLMConfig": (".config", "VLMConfig"),
    "load_config": (".config", "load_config"),
    "run_inference": (".inference", "run_inference"),
    "resolve_vlm_backend": (".backends", "resolve_vlm_backend"),
    "build_processor_adapter": (".backends", "build_processor_adapter"),
    "build_model_adapter": (".backends", "build_model_adapter"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(name) from error
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value
