# 工程约定

[返回本节目录](README.md) · [文档总目录](../README.md)

- Python 3.11，四空格缩进，公共合同使用类型标注。
- 优先使用 `pathlib`、严格 JSON/JSONL、原子写入和 SHA-256。
- 稳定能力位置为 `segmentation`、`vlm`、`grounding`、`retrieval`、`runtime`；数据生产、
  训练和评价分别进入 `data`、`training`、`evaluation`。
- `configs/` 按能力组织；`scripts/` 按 `data/train/infer/evaluate` 组织；`tests/` 镜像能力
  边界。历史 Stage 可保留在配置名、schema、输出根和 provenance 中。
- 新可执行脚本必须有简短中文头部，说明用途、命令、输入、输出、写入行为和所属能力；
  如 Stage 对 provenance 有意义，可同时说明 Stage。
- 算法不得写在 CLI 中。
- 生产 package 不得反向导入 `scripts`，不得创建 phase/stage compatibility alias。
- 不从文件名猜测通道科学含义。
- 不在模型 `forward` 中读取 HDF5。
- 保留用户已有改动；禁止 `git reset --hard`、`git checkout --` 和广泛清理。
- 未经明确请求不得 commit 或 push。
