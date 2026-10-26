# 配置

[返回本节目录](README.md) · [文档总目录](../README.md)

配置按使用者任务组织：

```text
configs/
├── segmentation/              OA-AuxSeg 训练、评价和推理
├── data/rs_general/           RS-GeneralDesc 构建、验证和导出
├── vlm/
│   ├── rs_general/            Shared MLLM / RS-General Adapter
│   └── grounded/              Grounded runtime 与 train-only curriculum
├── grounding/                 Corpus、annotation 和监督数据
│   └── prompts/
├── retrieval/                 Text Bank runtime、构建与评价配方
└── runtime/                   Unified Inference
```

统一推理使用 `oa_groundrag.unified_inference.config.v2`。配置显式绑定 `spatial`、
`semantic_core` 和 `retrieval`；Shared MLLM 不再通过 retrieval 配置间接定位，retrieval
provider 也不负责模型 loader。当前稳定 runtime 配置为：

- `configs/vlm/grounded/runtime_v1.yaml`：只验证已发布 workflow state、best
  pointer、checkpoint manifest、Adapter 与当前 model/processor/LoRA topology；推理不读
  compact training 或 OA-GroundedEval payload。
- `configs/retrieval/runtime_v1.yaml`：只绑定 source registry、Text Evidence Bank、BGE-M3
  和 retrieval 参数；运行时不依赖任何 development prediction/evaluation 输出。
- `configs/vlm/grounded/train_v2.yaml`：新的 train-only curriculum，不含 Eval-dev 或基于
  Eval-dev 的 baseline 生成/评价阶段。

已退役 Eval-dev 和 development outputs 的旧默认配置不再随仓库发布。构建与评价
实现仍按职责保留，但若要开启新的 development evaluation，必须在新授权下显式
提供新配置，不会因 CLI 默认值读取已删除路径。
