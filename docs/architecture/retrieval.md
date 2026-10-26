# 知识检索与解释

[返回本节目录](README.md) · [文档总目录](../README.md)

以用户问题、区域事实和第一遍视觉观察构造查询，组合关键词与向量检索。证据包覆盖解释、混淆因素和限制，第二遍只使用文本完成有来源的解释。

知识库原始 PDF 保持在 RAG_knowledge；知识问答不要求图像。检索可运行不代表专业准确率已经验证，参见[验收边界](../status/acceptance.md)。

实现入口：[oa_groundrag/retrieval/search.py](../../oa_groundrag/retrieval/search.py)。相关使用入口见[使用说明](../usage/README.md)。
