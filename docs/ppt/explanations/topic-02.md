# 2. 零基础术语表

[返回本节目录](README.md) · [文档总目录](../../README.md)

本页用于理解与讲解，不承担设计批准或进度记录；模型和结果状态请查[当前进度](../../status/README.md)。

| 术语 | 最直白的解释 | 在本项目里的含义 |
| --- | --- | --- |
| 像素 | 图像中的一个小格子 | 分割模型对每个格子预测属于目标的可能性 |
| mask | 与图像同尺寸的黑白区域图 | 白色表示关注区域，黑色表示其他区域；它不是地物真实颜色 |
| 分割 | 给每个像素分类 | OA-AuxSeg 输出候选滑坡区域的概率图和 mask |
| no-target | 当前样本没有可用目标区域 | 不是报错；它是必须被严格表达的一种合法结果 |
| LLM | 主要处理和生成文本的大语言模型 | Shared MLLM 中负责自回归语言生成的主体 |
| VLM / MLLM | 同时处理图像和文字的视觉语言模型/多模态大模型 | 2B 运行路线使用 Qwen3-VL-2B-Instruct，另有独立 4B 路线 |
| Grounding | 把语言回答约束到明确的图像区域和证据 | 原图、二值 mask、clean crop、程序事实共同限定“正在说哪里” |
| Programmatic Facts | 程序从 mask 确定计算的事实 | bbox、质心、面积、组件数等；模型只能读取，不能改写 |
| LoRA | 冻结大部分基座参数，只训练小型低秩增量 | 项目只在 attention 的 q/k/v/o 投影上训练适配参数 |
| Tokenizer | 把文本转换成模型能处理的 token ID | VLM 输入编码和 RAG 文本长度检查都依赖 tokenizer |
| Embedding | 把一段文字变成语义向量 | BGE-M3 把问题和证据单元映射到可比较的向量空间 |
| RAG | 先检索外部资料，再基于资料生成 | 给专业解释提供可追溯的文本证据，不改变模型权重 |
| Router | 根据显式任务生成执行计划 | 不看图、不推理、不让 LLM 猜任务，只映射 <code>UnifiedTask</code> |
| Provider | 对某项已有能力的稳定运行封装 | spatial、shared_mllm、evidence、text_rag |
| Gate | 预先定义的科学验收关口 | 工程能跑不等于 Gate 已通过 |
| provenance | 一个结果从何而来、由什么配置和资产产生 | Git、配置、manifest、SHA 和 checkpoint 元数据共同追溯 |
