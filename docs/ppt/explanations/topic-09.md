# 9. 面试题使用方法

[返回本节目录](README.md) · [文档总目录](../../README.md)

本页用于理解与讲解，不承担设计批准或进度记录；模型和结果状态请查[当前进度](../../status/README.md)。

下面恰好 30 道题。每题都包含：

- **来源**：小林题库原题或本项目追问；
- **口语回答**：先给出可以直接说的 30–60 秒主答；
- **概念拆解**：解释为什么；
- **项目落地**：说明当前系统如何做，并给出证据路径；
- **常见追问**：准备面试官继续深挖；
- **不要这样回答**：避免夸大或混淆。

口语回答不是让你逐字背诵。更好的方式是先记住每题的“问题—选择—理由—边界”四步，
然后用自己的语言回答。

### 小林题库筛选结果

| 小林专题 | 与项目高度相关的题 | 本文对应 | 只作延伸或不作为项目经验 |
| --- | --- | --- | --- |
| [RAG](https://xiaolinnote.com/ai/rag/rag_info.html) | RAG 流程、RAG vs 微调、Chunking、Embedding、关键词/向量检索、Query Rewrite、多路召回、向量库/图数据库选型、幻觉、评价、更新、落地难点 | Q14、Q16–Q26 | Self-RAG 等高级范式未实现；图数据库只讲选型边界 |
| [大模型工程](https://xiaolinnote.com/ai/llm/llm_info.html) | LLM/VLM 基础、Transformer、Tokenizer、微调、LoRA、解码、Prompt、幻觉、评价、模型选型 | Q8–Q15、Q25、Q29–Q30 | RoPE、GQA、FlashAttention、KV Cache、量化、DPO/GRPO、MoE 和部署框架未做专项实现 |
| [Agent](https://xiaolinnote.com/ai/agent/agent_info.html) | Workflow、Agent、Tools 的区别 | Q27–Q28 | ReAct、Reflection、记忆和 Multi-Agent 不在当前主线 |
| [工具调用与框架](https://xiaolinnote.com/ai/) | 只用于帮助区分内部 provider 与外部工具协议 | 第 14 节边界表 | Function Calling、MCP、Skill、LangChain/LangGraph 均不能包装成项目经验 |

Q1–Q7、Q24、Q28 等“项目追问”不是照抄网站题目，而是面试官在听完通用概念后最可能沿着
当前架构继续追问的设计选择。它们的答案以本地实现为准。
