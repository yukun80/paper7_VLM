# 当前能力状态

[返回本节目录](README.md) · [文档总目录](../README.md)

新增 v4 共享模型的实现与验证范围见[共享分割状态](shared-segmentation.md)。以下保留旧版能力边界。

| 能力 | 稳定代码位置 | 当前结果 | 尚未越过的边界 |
| --- | --- | --- | --- |
| Spatial Perception / OA-AuxSeg | `oa_groundrag/segmentation`；`oa_groundrag/training/segmentation`；`oa_groundrag/evaluation/segmentation.py` | full Benchmark 与负责人定版 checkpoint 可用 | Gate A、正式 fixed predicted masks、sealed test 均未执行 |
| Shared RS-Geohazard MLLM | `oa_groundrag/vlm`；`oa_groundrag/data/rs_general`；`oa_groundrag/training/vlm` | RS-GeneralDesc native v1、冻结 2B step-1000 Adapter/Gate B，以及新 4B step-1000 RS-General Adapter/独立 Gate B 可用；Qwen3.5 监督位置训练包装器仅存在于 training 边界 | 4B Gate B 只接受其 RS-GeneralDesc 冻结作用域；M7 v4 梯度等价门失败，4B Grounded Adapter、Runtime 与 Grounded 绝对验证均未完成 |
| Grounded Multimodal Understanding | `oa_groundrag/grounding`；`oa_groundrag/data/grounded`；`oa_groundrag/training/grounding` | train-only Corpus、compact supervision 与 step-900 2B Region Adapter 保留；2B runtime 已直接绑定便携模型和 LoRA 合同；4B v4 资源画像和失败 parity 证据可审计，但未形成可发布 Adapter | R1 梯度等价阈值失败后未进入 allocator 1/20/100 资源门；当前无 materialized OA-GroundedEval-dev，Gate C、专家共识和正式 OA-GroundedEval 未完成 |
| Knowledge Augmentation | `oa_groundrag/retrieval`；`oa_groundrag/evaluation/retrieval` | Text Bank、BGE-M3 与知识源改用 `<bundle-root>` 相对发布身份；Bank-only runtime retrieval/Pass-2 与评价实现保留 | 当前无 materialized Gate-D development evaluation；无 retrieval Gold、专家盲评或正式阈值；Gate D 未科学通过 |
| Unified Inference | `oa_groundrag/runtime` | 六类显式任务、确定性 router、lazy provider 与双语只读 Demo 工程完成；2B 运行配置可从任意工作目录解析固定兄弟目录包；纯 `KNOWLEDGE_QA` 不消费 Benchmark payload | 新服务器真实 GPU 推理尚未执行；auxiliary preview 不进入当前 P0 MLLM formal grounded input；test 默认锁定；P1 多源 grounded evidence 与统一科学评价均未开始 |

Stage 只保留为 curriculum、schema、output root、checkpoint metadata 和历史 provenance；
活动源码、配置、脚本和测试按能力与工程职责组织。
