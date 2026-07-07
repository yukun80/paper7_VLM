# Multi-Source Qwen-PSALM-Seg

本仓库用于开展面向任意模态组合的多源遥感滑坡指令分割研究。当前研究目标、技术路线和交付要求以 [docs/Task_Introduction.md](docs/Task_Introduction.md) 为准。

研究主线是构建统一的多源滑坡分割 benchmark，并在此基础上设计 Multi-Source Qwen-PSALM-Seg 原型模型。模型目标是支持高分辨率光学、Sentinel-2 多光谱、Sentinel-1 SAR、DEM、InSAR 形变速率以及灾前灾后影像等不同输入条件，在模态缺失、图像尺寸不一致、空间分辨率不同的情况下完成滑坡 mask 分割。

## Repository Layout

```text
datasets/
benchmark/
Multi-Source_Landslides_seg/
scripts/
external/
models/
docs/
参考文献/
```

### `datasets/`

存放不同论文来源的开源原始数据集。该目录用于保留原始数据、解压后的数据结构和必要的数据来源材料，不作为统一训练格式的输出位置。

当前已整理的数据来源包括但不限于：

```text
datasets/DisasterM3/
datasets/GDCLD/
datasets/LMHLD/
datasets/LandslideBench_agent/
datasets/Sen12Landslides/
datasets/landslide4sense/
datasets/multimodal-landslide-dataset/
```

原则上不要在数据处理脚本中直接改写原始数据；如需清洗、裁剪、重采样或统一索引，应将派生产物输出到 `benchmark/`。

### `benchmark/`

用于存放整合后的目标格式数据，供模型训练、推理、评估和可视化使用。该目录应承载统一样本索引、处理后的 patch、mask、任务指令、数据划分、统计报告和评估结果。

后续统一格式应围绕 instruction segmentation 设计，至少记录图像路径、mask 路径、可用模态、空间分辨率、图像尺寸、任务类型、区域/事件标识和 split 信息。

### `Multi-Source_Landslides_seg/`

用于放置当前多源遥感 VLM 分割模型构建代码，包括 Multi-Source Qwen-PSALM-Seg 的模型结构、训练入口、数据读取器、loss、评估接口和实验配置。

本研究自己的核心模型代码优先放在该目录，而不是混入外部参考代码目录。

### `scripts/`

用于放置数据处理、格式转换、统计和可视化脚本。当前保留的脚本主要来自旧数据管线中仍可复用的数据侧能力：

```text
scripts/1-1_scan_sources.py
scripts/1-2_prepare_sen12_views.py
scripts/1-3_prepare_gdcld_tiles.py
scripts/1-4_merge_annotations.py
scripts/1-6_validate_and_summarize.py
scripts/geohazard_common.py
```

这些脚本后续需要围绕新的多源滑坡 instruction segmentation 格式继续改造。旧的 Qwen 文本 SFT、bbox-only grounding 和临时训练评估入口已经不再作为主线。

### `external/`

存放来自其他研究的开源代码，用于参考、复现实验或局部复用。外部代码应尽量保持原始结构，必要适配应写在本仓库自己的模型代码或脚本中。

当前外部参考代码包括：

```text
external/DisasterM3-master/
external/LandslideAgent-main/
external/PSALM/
external/Qwen-VL-Series-Finetune/
```

### `models/`

存放从 Hugging Face 等来源下载的开源模型权重和处理器文件。模型权重通常体积较大，不应提交到 Git。

示例：

```text
models/Qwen3-VL-2B-Instruct/
```

### `docs/`

存放研究计划、任务说明和后续工程文档。当前核心文档是：

```text
docs/Task_Introduction.md
```

该文档描述当前研究背景、目标、方法路线、实验设计、评价指标和最终交付物。

### `参考文献/`

存放本研究参考的论文、报告和相关材料。可按研究主题继续划分，例如多模态大模型、VLM 分割、benchmark 数据集等。

## Current Research Direction

当前第一阶段聚焦真实 mask 监督的滑坡分割任务，优先完成以下工作：

1. 整理多源滑坡数据清单，明确每个数据集的模态、尺寸、空间分辨率、标签格式和区域来源。
2. 建立统一样本格式，将不同数据集组织为 instruction segmentation 任务。
3. 保留真实 mask 作为主监督信号，bbox 只作为可选派生信息。
4. 设计可处理缺失模态、不同图像尺寸和不同 GSD 的数据读取与模型接口。
5. 实现传统分割 baseline 和 Multi-Source Qwen-PSALM-Seg 原型。
6. 按数据集、模态组合、任务类型、缺失模态条件和跨区域/跨数据集设置报告 IoU、Dice/F1、Precision、Recall 和 Boundary F1。

当前不再沿用旧的纯文本视觉问答微调、普通 LoRA 文本 SFT 或 bbox-only grounding 主线。第一阶段也不把区域描述、灾害报告生成或完整地灾大模型作为主任务。

## Development Principles

- `datasets/` 保留原始数据和论文来源数据结构，不在处理过程中直接覆盖。
- `benchmark/` 存放统一训练、推理和评估格式的派生产物。
- `Multi-Source_Landslides_seg/` 存放本研究核心模型代码。
- `scripts/` 存放可复用的数据处理、统计和可视化工具。
- `external/` 存放外部开源代码，避免把本研究改动直接混入第三方仓库。
- `models/` 存放本地模型权重，不提交大模型文件。
- 新增脚本和模型接口应优先服务真实 mask 分割、任意模态组合、尺度信息记录和跨数据集评估。

## Expected Outputs

后续项目产物应逐步包括：

- 原始数据清单和数据来源说明。
- 统一的 train/val/test 样本索引。
- 数据清洗报告，包括 nodata、空 mask、错位 mask、尺寸差异和缺失模态处理说明。
- 多源 instruction segmentation 数据样例。
- Baseline 分割模型结果。
- Multi-Source Qwen-PSALM-Seg 原型模型、训练脚本和评估脚本。
- 按数据集、模态组合和任务类型汇总的指标表。
- 参数设置表、消融实验表、失败案例分析和可视化图表。
