# 第17页｜单条多源样本真实截图

[返回本节目录](README.md) · [文档总目录](../../README.md)

本页是汇报编排素材；上屏成果与数字必须在制作时核对[当前进度](../../status/README.md)，不在此维护第二份进度。

### 项目对应与研究状态

- 项目对应：T3.1 典型地质要素解译样本库构建
- 研究状态：真实同源样本已构建；本页仅展示数据组织

### 页面标题与核心结论

- 页面标题：**同一编号关联光学、DEM、编码 InSAR 和像素标注**
- 核心结论：同一样本编号对应四类空间位置一致的数据，光学影像作为位置基准，其他数据提供地形、形变编码和像素标签。

### 上屏文字

1. 同一样本编号对应光学、DEM、编码 InSAR 和 mask 四类数据
2. 光学影像作为位置基准，四张截图保持相同空间范围和原始比例
3. 编码 InSAR 只作图像输入，不解释具体位移数值和正负方向

### 主视觉布局

采用 2×2 大图矩阵，按左上光学、右上 DEM、左下编码 InSAR、右下 mask 排列；中央交叉位置
放置同一样本编号。四图使用相同尺寸和边界，不裁剪、不重着色；页脚标注真实同源样本和仅重新编排版式。

### 真实截图制作建议

1. 直接读取下列四个真实文件，使用程序或图像查看器分别导出截图，不调用生成式制图工具。
2. 在 PPT 中建立四个等大的图片框，按光学、DEM、编码 InSAR、mask 的顺序完整嵌入，不裁剪、不拉伸、不重着色。
3. 四图上方分别添加短标签，中央标注样本编号 `multimodal_landslide_loess_000002`，页脚标注真实同源样本和仅重新编排版式。
4. 若程序显示带色标的编码 InSAR，只保留原始色标和原始文字，不补写单位、位移方向或定量解释。

### 准确标签清单

- multimodal_landslide_loess_000002
- 光学｜DEM｜编码 InSAR｜mask
- 同一样本编号
- 光学影像作为位置基准
- 真实同源样本｜仅重新编排版式

### 本地参考图或真实素材

- 光学：`/home/yukun80/codes/datasets/multimodal-landslide-dataset/png/rgb/train/multimodal_landslide_loess_000002.png`
- DEM：`/home/yukun80/codes/datasets/multimodal-landslide-dataset/png/dem/train/multimodal_landslide_loess_000002.png`
- 编码 InSAR：`/home/yukun80/codes/datasets/multimodal-landslide-dataset/png/insar_vel/train/multimodal_landslide_loess_000002.png`
- mask：`/home/yukun80/codes/datasets/multimodal-landslide-dataset/png/mask/train/multimodal_landslide_loess_000002.png`

### 内部事实边界（不进入 PPT）

- 四张图必须原样使用，不允许生成式工具重绘或补全像素。
- 编码 InSAR 仅是已有编码图；在单位和符号未核验前不得解释为定量位移。
- 该样本来自开源 Multimodal Landslide 数据，不是铁路项目现场样本。

---
