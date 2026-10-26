# 下一任务与授权

[返回本节目录](README.md) · [文档总目录](../README.md)

本分支新的 v4 后续事项以[共享分割状态](shared-segmentation.md#下一步)为准。
以下仍是保留的 v3.2 恢复条件，不将本次重构解释为旧版资源门通过。

2B 便携改造已完成到 CPU 静态验证。[部署说明](../usage/deployment.md)已给出从 `<bundle-root>` 执行的安装、归档、
SHA-256 校验和统一推理 dry-run 命令；此前实施轮次没有实际创建归档，也没有在新服务器执行真实
GPU 推理。下一步由负责人按[部署说明](../usage/deployment.md) 复制交付目录并在目标服务器验证，若需本仓库继续执行
GPU 或正式评价，必须另行授权。

4B 路线仍没有可自动执行的下一模块。v3.2/R0 已冻结完整 16,000 条资源画像；R1 的监督位置
logits 在同一真实样本上将 CUDA peak 从 `11,204,981,760` 降到 `9,673,601,536` bytes，
且 loss 完全一致，但 64-LoRA gradient aggregate relative-L2 为 `0.009208712726831436`，
超过预注册上限 `0.001`。这正是 v3.2 规定的串行硬停止条件，因此 native/expandable 的
最坏样本与 1/20/100-step R2、1000-step R3、retention、M8 Runtime、M9 100 条 val-only
评价及 M10 默认发布均未启动。

资源门未通过，所以旧 M7 v3 配置、解析分支和 19 MiB 失败根均未删除；不得把“条件清理
未触发”误记为清理遗漏。若要继续，负责人必须重新 grill 并明确一个新实验合同，例如
决定是否接受 BF16 LM-head 不同矩阵形状带来的梯度数值差、改用何种等价计算，或改走
QLoRA、输入缩减、可选 kernel、LoRA 拓扑调整/更大显存设备。任何阈值放宽或数学路径变化
都不得自动采用。在取得新授权前必须保留 R0 profile、R1 failed parity 与旧 step-1
smoke 证据，并继续保持既有 2B/4B RS-General Adapter、Gate B、Benchmark 与正式 outputs
只读；重发布后的 2B Region 和 Text Bank 路径元数据也不得无授权再次改写。仍禁止访问
test/sealed、Gate A/C/D、Case RAG、commit 和 push。
