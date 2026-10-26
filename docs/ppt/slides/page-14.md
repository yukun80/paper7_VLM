# 第14页｜样本加工与质量控制

[返回本节目录](README.md) · [文档总目录](../../README.md)

本页是汇报编排素材；上屏成果与数字必须在制作时核对[当前进度](../../status/README.md)，不在此维护第二份进度。

### 项目对应与研究状态

- 项目对应：T3.1 典型地质要素解译样本库构建
- 研究状态：基础数据检查已构建；四层专业标签体系建设中

### 页面标题与核心结论

- 页面标题：**先排除损坏和错配样本，再补充影响识别难度的专业标签**
- 核心结论：基础检查保证文件和模态配对正确，后续标签说明样本类别、易混淆程度、图像质量和关键地貌特征。

### 上屏文字

1. 已完成文件损坏、格式转换和多源数据配对检查，排除不可用样本
2. 已检查同一场景是否同时出现在 train、val 和 test，避免评价结果失真
3. 正在补充类别、易混淆样本、图像质量和关键地貌特征标签

### 主视觉布局

采用上下两层结构。上层为蓝色基础数据检查流程，依次放四类检查；下层为橙色
四层专业标签，用四块叠层卡片展示标签维度。两层之间使用向下箭头，表示只有通过基础检查的
样本才进入领域标注。

### Imagegen 制图提示词

```text
Create a 16:9 landscape scientific infographic on a pure white background. All visible explanatory text, process labels, status labels, and captions must be in Simplified Chinese. English is allowed only for train, val, test, manifest, ledger, schema, and SHA-256. Do not render the English layout instructions in the image. If accurate Chinese text cannot be generated, leave a clean blank label area for manual replacement instead of using English. Use a two-level vertical structure. The upper blue area is labeled 基础数据检查 已构建 and contains exactly four checks: 文件可用性, 格式转换, 模态配对, 场景重复. Add manifest, ledger, schema, and SHA-256 below it. Use one downward arrow to an orange four-layer stack labeled 专业标签 建设中. The layers are 类别标签, 易混淆样本, 图像质量, 关键地貌特征. Add one Chinese note 场景重复检查覆盖 train val test. Keep the layout sparse. Do not add quality scores, sample counts, expert consensus, annotation examples, or railway imagery.
```

### 准确标签清单

- 基础数据检查｜已构建
- 文件可用性｜格式转换｜模态配对｜场景重复
- manifest｜ledger｜schema｜SHA-256
- 专业标签｜建设中
- 类别标签｜易混淆样本｜图像质量｜关键地貌特征

### 本地参考图或真实素材

- [参考图](<../figure/Figure 2 — Dataset Construction.png>)，仅参考数据治理语言。
- 实际质量检查要求以仓库中的版本化 manifest、ledger 与 schema 为准。

### 内部事实边界（不进入 PPT）

- 四层统一标签体系尚在建设中，不能与基础工程检查一起标成已完成。
- 不对当前样本做未有证据支持的专家一致性或质量百分比声明。
- 本页不改变、不重建任何现有 Benchmark 数据。

---
