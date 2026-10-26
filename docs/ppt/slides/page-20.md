# 第20页｜下一阶段路线与项目验收目标

[返回本节目录](README.md) · [文档总目录](../../README.md)

本页是汇报编排素材；上屏成果与数字必须在制作时核对[当前进度](../../status/README.md)，不在此维护第二份进度。

### 项目对应与研究状态

- 项目对应：T2.2、T2.3、T3.1、T3.2、T3.3 后续工作与终期验收
- 研究状态：下一阶段；进一步的知识增强研究为探索性展望

这些是项目背景中的候选后续工作，不是本次文档整理新增的研究方向或执行授权；实施顺序以[下一步](../../status/next.md)和负责人新授权为准。

### 页面标题与核心结论

- 页面标题：**下一阶段先恢复 4B 区域训练，再完成专业数据、领域微调和风险评估**
- 核心结论：后续依次处理训练计算一致性问题、建设专业数据、开展领域微调和风险评估，进一步的知识增强研究作为探索方向，不否认已有 Text RAG 工程能力。

### 上屏文字

1. 查明并修正 4B 区域训练的计算一致性问题，完成小规模和中等规模验证
2. 建设 Geo-Instruct 和铁路场景样本，开展领域微调与风险评估
3. 项目验收目标：mIoU≥75%｜分类准确率≥90%｜风险描述 ROUGE-L≥0.4

### 主视觉布局

中央为三阶段右向路线：第一阶段恢复 4B 区域训练，第二阶段建设领域数据并开展微调和风险评估，第三阶段
探索进一步的知识增强验证。第一阶段用橙色转蓝色表示从问题修正进入验证，第二阶段用紫色虚线，最右端 RAG 使用更淡的
紫色虚线并标注探索性展望。底部单独放三个浅灰目标卡，统一标注项目验收目标，不做完成度展示。

### Imagegen 制图提示词

```text
Create a 16:9 landscape scientific infographic on a pure white background. All visible explanatory text, process labels, status labels, and captions must be in Simplified Chinese. English is allowed only for 4B, Geo-Instruct, RAG, mIoU, and ROUGE-L. Do not render the English layout instructions in the image. If accurate Chinese text cannot be generated, leave a clean blank label area for manual replacement instead of using English. Draw exactly three future work areas from left to right. The first is 4B 区域训练恢复与验证 and contains 修正计算一致性问题, 小规模验证, 中等规模验证, 正式训练复核. The second is 专业数据与风险评估 and contains Geo-Instruct 与铁路样本, 领域微调, 风险评估. The third is a light purple dashed area labeled 进一步知识增强研究 and 探索性展望. Use Figure 5 only as a layout reference and do not copy its English labels or current system meaning. Along the bottom place exactly three gray target cards under 项目验收目标: 分割 mIoU≥75%, 要素分类准确率≥90%, 风险描述 ROUGE-L≥0.4. Keep the layout sparse. Do not add achieved metrics, unverified scientific RAG gains, railway field results, or unspecified modules.
```

### 准确标签清单

- 4B 区域训练恢复与验证
- 修正计算一致性问题｜小规模验证｜中等规模验证｜正式训练复核
- 专业数据与风险评估
- Geo-Instruct 与铁路样本｜领域微调｜风险评估集成
- 进一步知识增强研究｜探索性展望
- 项目验收目标
- 分割 mIoU≥75%
- 要素分类准确率≥90%
- 风险描述 ROUGE-L≥0.4

### 本地参考图或真实素材

- [参考图](<../figure/Figure 5 — Evidence-Constrained Geohazard RAG.png>)，仅作最右端展望模块的版式参考。
- [项目背景附件](../../references/实施方案v4.docx)，三个终期指标的项目口径来源。
- [当前进度](../../status/README.md)，确定下一阶段起点和当前未完成边界。

### 内部事实边界（不进入 PPT）

- 专业知识增强的后续研究不能与已有 Text RAG 运行能力混淆；已有能力及科学验证状态统一见[当前能力](../../status/capabilities.md)与[验收边界](../../status/acceptance.md)。
- 三个数值只表示项目验收目标，不表示当前模型已经达到。
- 4B 区域训练、Geo-Instruct 正式数据集、领域微调和铁路风险集成均需后续验证。

---
