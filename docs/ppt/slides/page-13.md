# 第13页｜样本库加工步骤

[返回本节目录](README.md) · [文档总目录](../../README.md)

本页是汇报编排素材；上屏成果与数字必须在制作时核对[当前进度](../../status/README.md)，不在此维护第二份进度。

### 项目对应与研究状态

- 项目对应：T3.1 典型地质要素解译样本库构建
- 研究状态：基础加工步骤与版本化数据已构建；铁路场景补充为后续工作

### 页面标题与核心结论

- 页面标题：**多源样本经过检查、配准和分组后形成可训练数据**
- 核心结论：每条样本先检查文件和模态配对，再统一空间位置并按场景划分训练集、验证集和测试集。

### 上屏文字

1. 首先检查开源滑坡数据是否完整、可读，并确认不同模态正确配对
2. 以光学影像为位置基准，对齐 DEM、编码 InSAR 和 mask
3. 按场景划分 train、val 和 test，并用 HDF5、manifest、ledger 记录数据版本

### 主视觉布局

使用一条由左至右的五阶段加工流程：多源数据源、质量清洗、以光学影像为基准进行空间对齐、分组划分、版本化数据输出。
在第四阶段画三条互不交叉的分组线表示 train、val、test；末端输出 HDF5、manifest、ledger、schema 四张卡。

### Imagegen 制图提示词

```text
Create a 16:9 landscape scientific infographic on a pure white background. All visible explanatory text, process labels, status labels, and captions must be in Simplified Chinese. English is allowed only for DEM, InSAR, mask, train, val, test, HDF5, manifest, ledger, schema, and SHA-256. Do not render the English layout instructions in the image. If accurate Chinese text cannot be generated, leave a clean blank label area for manual replacement instead of using English. Draw exactly five processing steps from left to right: 开源多模态滑坡数据, 文件和配对检查, 以光学影像为基准进行空间对齐, 按场景分组, 版本化数据输出. The first step shows 光学, DEM, 编码 InSAR, and mask. The second shows 可读性, 尺寸一致性, and 模态配对. The fourth branches into train, val, and test. The final step shows HDF5, manifest, ledger, schema, and SHA-256. Keep all action labels in Chinese and the layout sparse. Do not generate remote sensing images, railway field samples, sample totals, extra modalities, or expert annotation claims.
```

### 准确标签清单

- 开源多模态滑坡数据
- 格式检查与质量清洗
- 以光学影像为基准进行空间对齐
- 按场景分组｜train｜val｜test
- HDF5｜manifest｜ledger｜schema｜SHA-256
- 光学｜DEM｜编码 InSAR｜mask

### 本地参考图或真实素材

- [参考图](<../figure/Figure 2 — Dataset Construction.png>)，参考加工步骤版式。
- `/home/yukun80/codes/datasets/multimodal-landslide-dataset/`，只读数据源；本页不复制素材。

### 内部事实边界（不进入 PPT）

- 当前样本主体来自开源滑坡数据，不能写成铁路沿线实地采集。
- HDF5 格式统一不代表模态语义、数值范围或有效覆盖天然一致。
- train／val／test 的正式身份以相应 manifest 和 ledger 为准，不从文件名推断。

---
