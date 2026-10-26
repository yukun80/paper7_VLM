# 环境与受保护资产

[返回本节目录](README.md) · [文档总目录](../README.md)

仓库使用 Python 3.11。推荐环境：

```bash
cd <bundle-root>/paper7_VLM
conda activate qwen3vl
python --version
```

开发环境通常还包含原始数据和第三方参考代码；2B 运行交付只要求以下固定关系：

```text
<bundle-root>/
├── benchmark/   2B 运行绑定的 OA-AuxSeg Benchmark
└── paper7_VLM/  代码、配置、模型和只读运行产物
```

- `../datasets` 和 `../external` 默认只读。
- `../benchmark` 只有在相应数据生产任务获明确授权时才能写入，并拒绝覆盖已有根。
- `models_zoo/`、`outputs/`、`docs/RAG_knowledge/`、`参考文献/` 和 `docs/archive/`
  必须保留。
- 输出根、schema 字符串和 checkpoint metadata 中的历史 phase/stage 名称属于 scientific
  provenance，不随 package 迁移改写。
- `yukun80/RAG_tmp` 只作外部原型参考，不复制进仓库，也不是运行时依赖。
