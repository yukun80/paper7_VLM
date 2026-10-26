"""VLM backend registry；未知与重复注册均 fail closed。"""

from __future__ import annotations

from importlib import import_module
from threading import RLock
from types import MappingProxyType
from typing import Callable, Mapping

from oa_groundrag.vlm.errors import ReasonCode, VLMError

from .contracts import VLMBackendSpec


class BackendRegistryError(VLMError):
    """后端发现或注册合同失败。"""


_BUILTIN_MODULES: Mapping[str, str] = MappingProxyType(
    {
        "qwen3_vl": "oa_groundrag.vlm.backends.qwen3_vl",
        "qwen3_5": "oa_groundrag.vlm.backends.qwen3_5",
    }
)
BUILTIN_BACKEND_NAMES = frozenset(_BUILTIN_MODULES)


class VLMBackendRegistry:
    """线程安全的显式后端表；不允许覆盖已注册规格。"""

    def __init__(
        self,
        *,
        builtin_modules: Mapping[str, str] | None = None,
        importer: Callable[[str], object] = import_module,
    ) -> None:
        self._specs: dict[str, VLMBackendSpec] = {}
        self._builtin_modules = dict(builtin_modules or {})
        self._importer = importer
        self._lock = RLock()

    def register(self, spec: VLMBackendSpec) -> None:
        if not isinstance(spec, VLMBackendSpec):
            raise BackendRegistryError(
                ReasonCode.BACKEND_CONTRACT_INVALID,
                "只能注册 VLMBackendSpec",
            )
        with self._lock:
            if spec.name in self._specs:
                raise BackendRegistryError(
                    ReasonCode.BACKEND_DUPLICATE,
                    f"backend 重复注册：{spec.name}",
                )
            self._specs[spec.name] = spec

    def resolve(self, name: str) -> VLMBackendSpec:
        if not isinstance(name, str) or not name:
            raise BackendRegistryError(
                ReasonCode.BACKEND_UNKNOWN,
                "backend name 必须是非空字符串",
            )
        with self._lock:
            existing = self._specs.get(name)
            if existing is not None:
                return existing
            module_name = self._builtin_modules.get(name)
            if module_name is None:
                raise BackendRegistryError(
                    ReasonCode.BACKEND_UNKNOWN,
                    f"未知 VLM backend：{name}",
                    details={"available": sorted(self.names())},
                )
            try:
                module = self._importer(module_name)
                spec = getattr(module, "BACKEND_SPEC")
            except (ImportError, AttributeError) as error:
                raise BackendRegistryError(
                    ReasonCode.BACKEND_CONTRACT_INVALID,
                    f"无法加载 backend 规格：{name}",
                    details={"module": module_name, "error": str(error)},
                ) from error
            if not isinstance(spec, VLMBackendSpec) or spec.name != name:
                raise BackendRegistryError(
                    ReasonCode.BACKEND_CONTRACT_INVALID,
                    f"backend 模块规格不匹配：{name}",
                    details={"module": module_name},
                )
            self.register(spec)
            return spec

    def names(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(
                sorted(set(self._builtin_modules).union(self._specs))
            )


DEFAULT_VLM_BACKEND_REGISTRY = VLMBackendRegistry(
    builtin_modules=_BUILTIN_MODULES,
)

