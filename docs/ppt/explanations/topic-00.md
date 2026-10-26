# 0. 先说明这份文档的边界

[返回本节目录](README.md) · [文档总目录](../../README.md)

本页用于理解与讲解，不承担设计批准或进度记录；模型和结果状态请查[当前进度](../../status/README.md)。

这是一份**学习与面试讲解**，不是新的算法协议，也不是实时进度报告。

- 冻结算法定义以 [OA-GroundRAG 算法构建方案](<../../architecture/README.md>) 为准。
- 当前进度、Gate 状态和产物身份只看 [当前进度](<../../status/README.md>)。
- 稳定运行入口看 [README](<../../../README.md>)，协作规则看 [AGENTS](<../../../AGENTS.md>)。
- 本文只解释现有设计，不宣布新的科学结果，不把开发评价写成正式验收。
- 文中的小林面试题链接只用于标明题目来源。回答全部根据本项目重新组织，没有复制原文。

如果只记住一句话，请记住：

> OA-GroundRAG 不是让一个大模型包办所有事情，而是让空间专家、视觉语言模型和知识检索
> 各做自己擅长的事，再由显式任务决定本次需要调用哪些能力。

### 建议阅读路线

- 第一次阅读：[角色直觉与术语](topic-01.md#guide-basics) → [具体案例](topic-04.md#guide-example) →
  [核心模块](topic-06.md#guide-modules) → [科学边界](topic-08.md#guide-science)。
- 准备项目介绍：[Q1–Q7 架构题](q01.md#interview-architecture)。
- 准备算法岗位：[Q8–Q15 LLM/VLM 题](q08.md#interview-vlm)。
- 准备 RAG 岗位：[Q16–Q26 RAG 题](q16.md#interview-rag)。
- 准备系统与科研追问：[Q27–Q30 Runtime/Governance 题](q27.md#interview-runtime)。
- 面试前十分钟：[十句复习卡](topic-15.md#review-cards)。

所有链接使用显式英文锚点，即使阅读器不自动生成中文标题锚点也能跳转。
