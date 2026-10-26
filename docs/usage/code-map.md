# 能力与代码位置

[返回本节目录](README.md) · [文档总目录](../README.md)

| 能力 | 稳定源码 | 主要职责 |
|---|---|---|
| Spatial Perception / OA-AuxSeg | `oa_groundrag/segmentation/` | 模型数学、稀疏辅助模态合同、checkpoint、区域提取与只读推理 |
| Shared RS-Geohazard MLLM | `oa_groundrag/vlm/` | Qwen processor/model、LoRA loader、生成、输出与 grounded adapter runtime |
| Grounded Evidence Interface | `oa_groundrag/grounding/` | RegionSelector、EvidenceBuilder、消息与严格输出合同 |
| Knowledge Augmentation | `oa_groundrag/retrieval/` | Text Evidence Bank、混合检索、Evidence Packet 与 Pass-2 |
| Unified Inference | `oa_groundrag/runtime/` | `UnifiedTask`、确定性 router、lazy provider 与执行编排 |

训练、数据生产和评价不进入 runtime 主链：

```text
oa_groundrag/
├── artifacts/                 通用身份、原子写入和路径安全
├── data/
│   ├── oa_auxseg/             OA-AuxSeg Benchmark producer/validator
│   ├── rs_general/            RS-GeneralDesc Benchmark producer/consumer
│   └── grounded/              Corpus、annotation、supervision 与 workflow
├── segmentation/              OA-AuxSeg 稳定模型与推理接口
├── vlm/                       Shared MLLM 稳定模型与推理接口
├── grounding/                 Grounded Evidence Interface
├── retrieval/                 Evidence-Constrained Text RAG
├── runtime/                   Instruction-Routed Unified Inference
├── training/
│   ├── segmentation/
│   ├── vlm/
│   └── grounding/
└── evaluation/
    ├── segmentation.py        OA-AuxSeg 评价
    ├── vlm.py                 Shared MLLM 评价
    ├── rs_general/            Gate B
    ├── grounding/
    └── retrieval/             Gate D
```
