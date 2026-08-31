"""Shared MLLM backend 的惰性公共入口。"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    "BackendRegistryError": (".registry", "BackendRegistryError"),
    "VLMBackendRegistry": (".registry", "VLMBackendRegistry"),
    "VLMBackendSpec": (".contracts", "VLMBackendSpec"),
    "VLMModelAdapter": (".contracts", "VLMModelAdapter"),
    "VLMProcessorAdapter": (".contracts", "VLMProcessorAdapter"),
    "build_model_adapter": (".factory", "build_model_adapter"),
    "build_processor_adapter": (".factory", "build_processor_adapter"),
    "resolve_vlm_backend": (".factory", "resolve_vlm_backend"),
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
