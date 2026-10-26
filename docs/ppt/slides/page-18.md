# 第18页｜多任务能力与案例矩阵

[返回本节目录](README.md) · [文档总目录](../../README.md)

本页是汇报编排素材；上屏成果与数字必须在制作时核对[当前进度](../../status/README.md)，不在此维护第二份进度。

### 项目对应与研究状态

- 项目对应：T2.1 多模态融合模型架构设计；T2.3 风险评估集成；T3.3 领域微调
- 研究状态：通用描述与 OA-AuxSeg 案例为真实输出；区域与风险两格为流程示意

### 页面标题与核心结论

- 页面标题：**两类真实输出可以复核，区域理解和风险分析仍是方法示意**
- 核心结论：铁路场景描述和分割结果使用真实输出，区域理解与风险分析只展示处理步骤，不作为 4B 实验结果。

### 上屏文字

1. 真实输出包括 Qwen3.5-4B 铁路场景描述和 OA-AuxSeg 分割结果
2. Mask-Grounded 区域理解和风险分析当前只展示处理步骤
3. 两类方法示意均不代表 4B 区域训练或风险评估结果

### 主视觉布局

采用 2×2 等宽案例矩阵。左上放外部验证铁路场景影像与真实 4B prediction；右上放同一次
OA-AuxSeg Demo 的 optical、global mask、overlay。左下使用 PPT 形状绘制区域理解的输入和输出步骤，右下使用
PPT 形状绘制风险分析的信息和说明步骤；下排统一使用橙色边框并标注方法示意和非 4B 实验结果。

### 真实截图制作建议

1. 左上直接使用下列外部验证影像，并从对应 `predictions.jsonl` 核对 record ID、模型角色和原始 prediction；英文输出由制作者原样粘贴，不交给生成式工具改写。
2. 右上直接使用同一 Demo run 下的 `optical_full.png`、`global_mask.png` 和 `predicted_overlay.png`，保持原始像素、颜色和比例。
3. 左下使用 PPT 基本形状绘制全幅光学影像、mask、候选区局部影像进入结构化区域观察的步骤，并标注方法示意和非 4B 实验结果。
4. 右下使用 PPT 基本形状绘制候选区域信息、线路暴露信息进入风险说明的步骤，并标注方法示意和非 4B 实验结果；没有真实结果时不得添加模拟描述或风险等级。

### 准确标签清单

- 通用描述｜真实 4B prediction
- `There are many buildings on both sides of the railway .`
- 像素分割｜真实 Demo 输出
- 区域理解｜流程示意，非 4B 实验结果
- 风险分析｜流程示意，非 4B 实验结果

### 本地参考图或真实素材

- 4B 铁路影像：`/home/yukun80/codes/benchmark/rs_generaldesc_v1/assets/image/90/90dcb4c0f5e9bf41cbb9587d74f67094f9db0ee9be15b8451519156621de6d5a.jpg`
- 4B prediction：`/home/yukun80/codes/paper7_VLM/outputs/phase4_rs_vlm/rs_generaldesc_gate_b_qwen35_4b_r851bf6e8_v1/adapter/predictions.jsonl`（对应 record 的 prediction）
- Demo 光学：`/home/yukun80/codes/paper7_VLM/outputs/demo/unified_workbench_v1/runs/demo_20260828T144934105536Z_b43e5bc9089f48ae8d6fd73f58e60b02/inputs/optical_full.png`
- Demo mask：`/home/yukun80/codes/paper7_VLM/outputs/demo/unified_workbench_v1/runs/demo_20260828T144934105536Z_b43e5bc9089f48ae8d6fd73f58e60b02/viewer_assets/segment_only/global_mask.png`
- Demo overlay：`/home/yukun80/codes/paper7_VLM/outputs/demo/unified_workbench_v1/runs/demo_20260828T144934105536Z_b43e5bc9089f48ae8d6fd73f58e60b02/viewer_assets/segment_only/predicted_overlay.png`

### 内部事实边界（不进入 PPT）

- 左上是 RS-GeneralDesc 外部验证样例，不是本项目铁路现场验证影像。
- 上排只证明对应单条输出可追溯，不外推整体业务精度。
- 下排不得放置伪造的 4B 区域描述、风险等级或实际威胁结论。

---
