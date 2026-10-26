# 第16页｜领域微调与错误案例改进

[返回本节目录](README.md) · [文档总目录](../../README.md)

本页是汇报编排素材；上屏成果与数字必须在制作时核对[当前进度](../../status/README.md)，不在此维护第二份进度。

### 项目对应与研究状态

- 项目对应：T3.3 领域微调与迭代训练；衔接 T2.2 多阶段演进训练策略
- 研究状态：4B 通用适配已完成；区域适配待优化验证；领域微调为下一阶段

### 页面标题与核心结论

- 页面标题：**通用适配完成后，区域训练暂缓，错误案例用于后续改进**
- 核心结论：4B 通用遥感训练已经完成，区域训练因计算一致性问题暂停，后续将根据错误案例补充样本和修订指令。

### 上屏文字

1. Qwen3.5-4B 通用遥感适配和 256 条样本的独立对比验证已完成
2. 区域适配的训练计算一致性未达到要求，正式训练尚未启动
3. 后续将针对误报、漏报和依据错位补充样本并修订指令

### 主视觉布局

上半部为四阶段训练时间轴：基础模型、通用遥感 SFT、Mask-Grounded SFT、Geo-Instruct 领域 SFT；
第二阶段为蓝色完成态，第三阶段在入口处设置橙色暂停标记，第四阶段为紫色虚线。下半部将三类错误
连接到补充样本、修订指令和下一轮训练三个步骤。

### Imagegen 制图提示词

```text
Create a 16:9 landscape scientific infographic on a pure white background. All visible explanatory text, process labels, status labels, and captions must be in Simplified Chinese. English is allowed only for Qwen3.5-4B, RS-General, SFT, Mask-Grounded, and Geo-Instruct. Do not render the English layout instructions in the image. If accurate Chinese text cannot be generated, leave a clean blank label area for manual replacement instead of using English. Across the upper half draw exactly four stages: Qwen3.5-4B 基础模型, RS-General 通用遥感 SFT 已完成, Mask-Grounded 区域 SFT 待优化验证, Geo-Instruct 领域 SFT 下一阶段. Add a blue status card labeled 独立对比验证 已完成 after the second stage. Add an orange pause marker labeled 训练计算一致性检查未达到要求 before formal regional training. In the lower half show 误报, 漏报, and 依据错位 leading to 补充样本, 修订指令, and 下一轮训练. Keep the layout sparse. Do not show completed regional training, completed domain training, performance gains, automatic online learning, or additional stages.
```

### 准确标签清单

- Qwen3.5-4B 基础模型
- RS-General 通用遥感 SFT｜已完成
- 独立对比验证｜已完成
- Mask-Grounded 区域 SFT｜待优化验证
- 训练计算一致性检查未达到要求
- Geo-Instruct 领域 SFT｜下一阶段
- 误报｜漏报｜依据错位
- 补充样本｜修订指令｜下一轮训练

### 本地参考图或真实素材

- [当前进度](../../status/README.md)：4B 通用适配、独立对比验证和 Mask-Grounded 训练计算一致性检查的实时状态。
- 本页为新绘训练与错误案例改进图，不使用虚构训练曲线。

### 内部事实边界（不进入 PPT）

- 当前只确认 4B RS-General Adapter 完成并通过固定范围内的独立 Gate B。
- 4B Mask-Grounded 未形成可发布 Adapter；Geo-Instruct 领域 SFT 尚未实施。
- 使用错误案例改进样本和指令是方法设计，不代表已完成线上自动学习系统。

---
