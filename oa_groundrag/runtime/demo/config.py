"""Unified Demo Workbench v1 的严格配置合同。"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from oa_groundrag.artifacts.identity import sha256_file
from oa_groundrag.artifacts.io import first_symlink_component
from oa_groundrag.runtime.config import UnifiedInferenceConfig, load_unified_config
from oa_groundrag.runtime.contracts import UnifiedTask
from oa_groundrag.vlm.config import _load_yaml


DEMO_CONFIG_SCHEMA = "oa_groundrag.unified_demo.config.v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class DemoConfigError(ValueError):
    """Demo 配置或只读资产身份不满足严格合同。"""


@dataclass(frozen=True)
class BenchmarkBinding:
    root: Path
    schema_version: str
    manifest_sha256: str
    index_sha256: str

    @property
    def identity(self) -> dict[str, str]:
        return {
            "root": str(self.root),
            "schema_version": self.schema_version,
            "manifest_sha256": self.manifest_sha256,
            "index_sha256": self.index_sha256,
        }


@dataclass(frozen=True)
class FrozenEvaluationBinding:
    name: str
    root: Path
    manifest_sha256: str


@dataclass(frozen=True)
class DemoDefaults:
    mode: str
    split: str
    suite_tasks: tuple[UnifiedTask, ...]
    prompts: Mapping[UnifiedTask, str]


@dataclass(frozen=True)
class DemoConfig:
    config_path: Path
    config_sha256: str
    unified: UnifiedInferenceConfig
    benchmark: BenchmarkBinding
    frozen_evaluations: tuple[FrozenEvaluationBinding, ...]
    demo_root: Path
    allow_test_demo: bool
    defaults: DemoDefaults
    schema_version: str = DEMO_CONFIG_SCHEMA


def _mapping(value: Any, *, location: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise DemoConfigError(f"{location} 必须是字符串键对象")
    return dict(value)


def _exact(row: Mapping[str, Any], fields: Sequence[str], *, location: str) -> None:
    expected = set(fields)
    if set(row) != expected:
        raise DemoConfigError(
            f"{location} 字段不匹配：missing={sorted(expected - set(row))}, "
            f"unknown={sorted(set(row) - expected)}"
        )


def _string(value: Any, *, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DemoConfigError(f"{location} 必须是非空字符串")
    return value.strip()


def _sha(value: Any, *, location: str) -> str:
    text = _string(value, location=location)
    if _SHA256.fullmatch(text) is None:
        raise DemoConfigError(f"{location} 必须是小写 SHA-256")
    return text


def _path(base: Path, value: Any, *, location: str) -> Path:
    text = _string(value, location=location)
    path = Path(text)
    return Path(os.path.abspath(path if path.is_absolute() else base / path))


def _regular(path: Path, *, location: str, directory: bool = False) -> None:
    linked = first_symlink_component(path)
    exists = path.is_dir() if directory else path.is_file()
    if linked is not None or not exists or path.is_symlink():
        kind = "目录" if directory else "文件"
        raise DemoConfigError(f"{location} 必须是普通{kind}：{path}")


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def load_demo_config(path: Path | str) -> DemoConfig:
    """加载并验证 Demo 配置；不构造 provider，也不打开 Benchmark shard。"""

    config_path = Path(os.path.abspath(Path(path)))
    _regular(config_path, location="Demo config")
    row = _mapping(_load_yaml(config_path), location="$")
    _exact(
        row,
        (
            "schema_version",
            "unified_config",
            "benchmark",
            "frozen_evaluations",
            "demo_root",
            "allow_test_demo",
            "defaults",
        ),
        location="$",
    )
    if row["schema_version"] != DEMO_CONFIG_SCHEMA:
        raise DemoConfigError(f"只支持 {DEMO_CONFIG_SCHEMA}")
    base = config_path.parent
    unified_path = _path(base, row["unified_config"], location="$.unified_config")
    unified = load_unified_config(unified_path)

    benchmark_row = _mapping(row["benchmark"], location="$.benchmark")
    _exact(
        benchmark_row,
        ("root", "schema_version", "manifest_sha256", "index_sha256"),
        location="$.benchmark",
    )
    benchmark = BenchmarkBinding(
        root=_path(base, benchmark_row["root"], location="$.benchmark.root"),
        schema_version=_string(
            benchmark_row["schema_version"], location="$.benchmark.schema_version"
        ),
        manifest_sha256=_sha(
            benchmark_row["manifest_sha256"], location="$.benchmark.manifest_sha256"
        ),
        index_sha256=_sha(
            benchmark_row["index_sha256"], location="$.benchmark.index_sha256"
        ),
    )
    _regular(benchmark.root, location="$.benchmark.root", directory=True)
    _regular(benchmark.root / "manifest.json", location="Benchmark manifest")
    _regular(benchmark.root / "index.jsonl", location="Benchmark index")
    if sha256_file(benchmark.root / "manifest.json") != benchmark.manifest_sha256:
        raise DemoConfigError("Benchmark manifest SHA-256 漂移")
    if sha256_file(benchmark.root / "index.jsonl") != benchmark.index_sha256:
        raise DemoConfigError("Benchmark index SHA-256 漂移")

    frozen_value = row["frozen_evaluations"]
    if not isinstance(frozen_value, list):
        raise DemoConfigError("$.frozen_evaluations 必须是列表")
    frozen: list[FrozenEvaluationBinding] = []
    for index, value in enumerate(frozen_value):
        location = f"$.frozen_evaluations[{index}]"
        item = _mapping(value, location=location)
        _exact(item, ("name", "root", "manifest_sha256"), location=location)
        binding = FrozenEvaluationBinding(
            name=_string(item["name"], location=f"{location}.name"),
            root=_path(base, item["root"], location=f"{location}.root"),
            manifest_sha256=_sha(
                item["manifest_sha256"], location=f"{location}.manifest_sha256"
            ),
        )
        _regular(binding.root, location=f"{location}.root", directory=True)
        _regular(binding.root / "manifest.json", location=f"{location}.manifest")
        if sha256_file(binding.root / "manifest.json") != binding.manifest_sha256:
            raise DemoConfigError(f"{binding.name} manifest SHA-256 漂移")
        frozen.append(binding)
    names = [item.name for item in frozen]
    if len(names) != len(set(names)):
        raise DemoConfigError("Frozen evaluation name 不允许重复")

    demo_root = _path(base, row["demo_root"], location="$.demo_root")
    linked = first_symlink_component(demo_root)
    if linked is not None or demo_root.is_symlink():
        raise DemoConfigError(f"$.demo_root 含 symlink：{linked or demo_root}")
    demo_outputs_root = Path(os.path.abspath(
        unified.repository_root / "outputs" / "demo"
    ))
    if demo_root == demo_outputs_root or not _is_within(demo_root, demo_outputs_root):
        raise DemoConfigError(
            "$.demo_root 必须是 repository outputs/demo 下的独立版本目录"
        )
    for protected in (benchmark.root, *(item.root for item in frozen)):
        if _is_within(demo_root, protected) or _is_within(protected, demo_root):
            raise DemoConfigError("Demo root 必须与 Benchmark/Frozen Eval 完全分离")
    allow_test = row["allow_test_demo"]
    if not isinstance(allow_test, bool):
        raise DemoConfigError("$.allow_test_demo 必须是 bool")

    defaults_row = _mapping(row["defaults"], location="$.defaults")
    _exact(defaults_row, ("mode", "split", "suite_tasks", "prompts"), location="$.defaults")
    mode = _string(defaults_row["mode"], location="$.defaults.mode")
    if mode != "Demo / Qualitative Exploration":
        raise DemoConfigError("默认 mode 必须是 Demo / Qualitative Exploration")
    split = _string(defaults_row["split"], location="$.defaults.split")
    if split not in {"train", "val"}:
        raise DemoConfigError("默认 split 只允许 train/val")
    suite_value = defaults_row["suite_tasks"]
    if not isinstance(suite_value, list) or not suite_value:
        raise DemoConfigError("$.defaults.suite_tasks 必须是非空列表")
    try:
        suite_tasks = tuple(UnifiedTask(value) for value in suite_value)
    except (TypeError, ValueError) as error:
        raise DemoConfigError("默认 suite 含未知 UnifiedTask") from error
    if len(suite_tasks) != len(set(suite_tasks)):
        raise DemoConfigError("默认 suite task 不允许重复")
    prompts_row = _mapping(defaults_row["prompts"], location="$.defaults.prompts")
    _exact(prompts_row, tuple(task.value for task in UnifiedTask), location="$.defaults.prompts")
    prompts = {
        task: _string(prompts_row[task.value], location=f"$.defaults.prompts.{task.value}")
        for task in UnifiedTask
    }
    return DemoConfig(
        config_path=config_path,
        config_sha256=sha256_file(config_path),
        unified=unified,
        benchmark=benchmark,
        frozen_evaluations=tuple(frozen),
        demo_root=demo_root,
        allow_test_demo=allow_test,
        defaults=DemoDefaults(
            mode=mode,
            split=split,
            suite_tasks=suite_tasks,
            prompts=prompts,
        ),
    )
