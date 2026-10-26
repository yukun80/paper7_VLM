# 第03页｜任务二与任务三的六项子任务关系

[返回本节目录](README.md) · [文档总目录](../../README.md)

本页是汇报编排素材；上屏成果与数字必须在制作时核对[当前进度](../../status/README.md)，不在此维护第二份进度。

### 项目对应与研究状态

- 项目对应：T2.1 至 T2.3、T3.1 至 T3.3
- 研究状态：总体任务关系已明确

### 页面标题与核心结论

- 页面标题：**任务二负责模型能力，任务三负责数据准备和持续改进**
- 核心结论：样本库和指令数据支持模型训练，验证中发现的问题再用于补充样本和修订指令。

### 上屏文字

1. T3.1 提供多源样本，T3.2 正在按统一格式建设指令和回答内容
2. T2.1 和 T2.2 负责多源信息处理、通用理解和区域理解训练
3. 模型验证发现的问题进入 T3.3，用于下一轮样本和训练改进

### 主视觉布局

采用上下双层任务关系图：上层为任务二三个模块，下层为任务三三个模块。任务三前两个模块向任务二的
架构和训练模块提供数据；任务二的验证问题通过返回箭头进入任务三的持续改进。任务编号完整可见。

### Imagegen 制图提示词

```text
Create a 16:9 landscape scientific infographic on a pure white background. All visible explanatory text, process labels, status labels, and captions must be in Simplified Chinese. English is allowed only for the task identifiers T2.1, T2.2, T2.3, T3.1, T3.2, and T3.3. Do not render the English layout instructions in the image. If accurate Chinese text cannot be generated, leave a clean blank label area for manual replacement instead of using English. Draw exactly six rounded task modules in two rows. The upper row is labeled 任务二 面向铁路的多源要素遥感大模型构建 and contains T2.1 多模态融合模型架构设计, T2.2 多阶段演进训练策略, and T2.3 风险评估方法构建与集成. The lower row is labeled 任务三 典型地质遥感要素样本库构建与智能识别 and contains T3.1 典型地质要素解译样本库, T3.2 领域专用指令数据集, and T3.3 领域微调与迭代训练. Draw arrows from T3.1 and T3.2 toward model design and training. Draw one return arrow from 模型验证发现的问题 toward T3.3 and label it 补充样本并修订指令. Keep the layout sparse. Do not add algorithm layers, experiment numbers, extra tasks, or invented completion status.
```

### 准确标签清单

- T2.1 多模态融合模型架构设计
- T2.2 多阶段演进训练策略
- T2.3 风险评估方法构建与集成
- T3.1 典型地质要素解译样本库
- T3.2 领域专用指令数据集
- T3.3 领域微调与迭代训练
- 模型验证发现的问题
- 补充样本并修订指令


---
