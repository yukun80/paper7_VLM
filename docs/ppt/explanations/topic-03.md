# 3. 只需要理解的三个公式

[返回本节目录](README.md) · [文档总目录](../../README.md)

本页用于理解与讲解，不承担设计批准或进度记录；模型和结果状态请查[当前进度](../../status/README.md)。

### 3.1 Attention：当前 token 应该关注哪些信息

$$
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}\left(\frac{QK^\mathsf{T}}{\sqrt{d_k}}\right)V
$$

- <code>Q</code>（Query）表示“我现在想找什么”。
- <code>K</code>（Key）表示“每条已有信息适合被怎样匹配”。
- <code>V</code>（Value）表示“匹配后真正取回什么内容”。
- <code>QKᵀ</code> 得到匹配分数，除以 <code>√d_k</code> 防止维度较大时数值过尖。
- <code>softmax</code> 把分数变成权重，最后对 <code>V</code> 加权汇总。

本项目没有重新发明 Attention；它在 Qwen3-VL 的既有 Attention 投影
<code>q_proj/k_proj/v_proj/o_proj</code> 上加入 LoRA。

### 3.2 LoRA：只学习一个小增量

$$
W^\prime = W + \frac{\alpha}{r}BA
$$

- <code>W</code> 是冻结的基座权重。
- <code>A</code> 和 <code>B</code> 是需要训练的小矩阵。
- <code>r</code> 是低秩维度；项目配置为 8。
- <code>α</code> 控制增量缩放；项目配置为 16。

直觉上，大矩阵 <code>W</code> 不动，只学习“如何稍微调整它”。这使单卡训练可行，也让
基座模型身份与适配器身份可以分别记录。

### 3.3 RRF：不直接比较两种检索分数

$$
\operatorname{RRF}(d)=\sum_i\frac{1}{k+\operatorname{rank}_i(d)}
$$

- <code>d</code> 是一个候选证据单元。
- <code>rank_i(d)</code> 是它在第 <code>i</code> 条检索路径中的名次。
- <code>k</code> 是平滑常数；项目配置为 60。

关键词检索分数和向量相似度不是同一量纲，硬把原始分数相加不可靠。RRF 只融合“名次”，
让一个同时被两路排在前面的证据获得更高融合分。
