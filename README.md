# OA-AuxSeg + VLM

本项目研究光学锚定、任意辅助模态增强的滑坡分割，以及基于分割区域证据的视觉语言理解。

当前仓库已停止维护 SANE、QMEF、PMRD、MGRR、SegDesc、Bridge 和旧
SAMI-GroundSegDesc 路线。Git 历史和 `docs/archive/` 保存历史证据，活动工程不提供旧接口、
兼容包装或运行时回退。

详细设计见
[`光学锚定任意辅助模态滑坡分割与VLM区域理解_算法构建方案.md`](docs/光学锚定任意辅助模态滑坡分割与VLM区域理解_算法构建方案.md)。

## 研究路线

系统分为两个独立阶段：

1. OA-AuxSeg 滑坡分割
   - 光学影像是必要主模态和空间边界基准；
   - SAR、InSAR、DEM、多光谱等是任意可选辅助模态；
   - 采用 CMNeXt 式辅助注入和简化 MAGIC 式质量选择；
   - 输出概率图、二值 mask、no-target 状态和区域列表。
2. VLM 区域理解
   - 接收用户指令、光学影像、预测 mask 和可用辅助证据；
   - 完成区域选择、证据约束描述和多模态回答；
   - RAG 仅预留接口，当前不实现复杂知识库。

分割和 VLM 不做联合反向传播。分割稳定前不进入复杂 VLM 或 RAG 开发。

## 当前状态

- 当前阶段：阶段 0，旧活动实现清理与最小工程基线。
- 活动 Python 包：无。
- 活动 CLI：无。
- 活动 Benchmark、模型、Trainer、Evaluator、配置或 schema：无。
- 下一阶段：只读审计全部候选 HDF5，基于真实字段、shape、dtype、通道和配准状态重新设计
  统一 Benchmark。

阶段状态只记录在 [`REBUILD_PROGRESS.md`](REBUILD_PROGRESS.md)。

## 数据现场

数据位于 `/home/yukun80/codes/datasets`，当前按 HDF5 文件统计：

| 数据源 | 样本对 | HDF5 文件 | 大小 |
| --- | ---: | ---: | ---: |
| GDCLD | 13,447 | 26,894 | 26.211 GiB |
| LMHLD | 28,185 | 56,370 | 3.072 GiB |
| LandslideBench_agent | 2,130 | 4,260 | 1.569 GiB |
| Landslide4Sense | 3,799 | 7,598 | 1.570 GiB |
| multimodal-landslide-dataset | 6,084 | 12,168 | 0.825 GiB |
| Sen12Landslides | 0 | 0 | 0 GiB |
| 合计 | 53,645 | 107,290 | 33.247 GiB |

这是 2026-07-24 的只读文件级快照，不代表字段、模态、空间配准或科学语义已经验收。
后续参数不得从旧代码或本表推断，必须来自真实 HDF5 审计。

## 保留资产

```text
/home/yukun80/codes/
├── datasets/       只读原始 HDF5 资产
├── benchmark/      当前为空；后续阶段生成
├── external/       第三方参考代码
└── paper7_VLM/
    ├── docs/archive/   历史文档，只读参考
    ├── models_zoo/     本地模型权重和元数据
    └── 参考文献/       论文与研究资料
```

`../external/DELIVER` 提供 CMNeXt 参考实现；Grasp Any Region、PSALM 和
Qwen-VL-Series-Finetune 也保留在 `../external`。这些目录不得被修改、复制进活动代码或
作为隐式运行时依赖。

## 下一阶段

下一阶段名称为“多源 HDF5 数据审计与统一 Benchmark 构建”。执行顺序是：

1. 只读检查真实 HDF5 结构与索引；
2. 确认光学、mask、辅助模态、validity 和空间对应关系；
3. 冻结读取合同、split 原则、目标 patch 参数和重采样规则；
4. 再实现新的 Benchmark builder 与验证器。

当前没有可运行的 builder 或训练命令，也不应恢复旧命令。
