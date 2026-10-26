# 最近一次有效工程验收

[返回本节目录](README.md) · [文档总目录](../README.md)

以下为**此前运行的验证记录**，不是 2026-10-21 文档整理重新运行的结果。本次检查单独见[文档检查](documentation-check.md)。

能力驱动重构和 M7 失败证据仍保留；当前 2B 便携运行改造的有效工程证据如下。

| 检查 | 最近结果 |
| --- | --- |
| 此前实施轮次相关回归 | 共 267 项完成：263 passed、4 项因缺少 `oa_auxseg_hdf5_v1/small` skipped。分组为 Runtime 36、Retrieval 33、VLM 与 training/vlm 72、Segmentation 43、Grounding 与 Architecture 49、Gate B 34；没有调用 GPU |
| Runtime / 迁移模拟 | `tests/runtime` 36/36；在不同名称的临时 `<bundle-root>` 中复制最小配置、真实 step-900 Adapter 和 Text Bank，从不同工作目录完成配置解析、224 个 Adapter tensor 读取及 Bank manifest/ledger/file hash 验证 |
| 训练与 checkpoint 合同 | OA-AuxSeg、RS-VLM 与 Mask-Grounded 三个训练帮助均无续训参数；非空 `resume_checkpoint` 被拒绝；新 checkpoint 不保存 optimizer、scheduler、RNG 或 sampler；旧 checkpoint 仍可用于只读推理与验证；1、20、100 步检查使用各自全新输出根并从同一基础 Adapter 开始 |
| 路径与资产保护 | 绝对发布路径、`..`、目录越界和符号链接逃逸均被拒绝；2B Adapter SHA 仍为 `858e12ff...1f84d7`，OA-AuxSeg checkpoint SHA 仍为 `672d39ab...84fb038`，Text Bank 向量未重建；冻结 2B 训练配置和 4B/Gate B 资产未修改 |
| Gradio smoke | Blocks 构建由 Runtime/Demo tests 覆盖；最终代码在 `127.0.0.1:8977` 成功启动并关闭，`share=false`，allowed path 仅 `outputs/demo/unified_workbench_v1` |
| compile / CLI / diff | `compileall`、三个训练 CLI help 和 `git diff --check` 通过；最终相关回归使用 Python 标准库 `unittest` 实跑，未安装新依赖 |
| 真实输入预览 | 固定 val `landslide4sense::landslide4sense_000002` 从同一 raw sample 展示 B01–B12、DEM 与 slope；逐通道 raw min/max、valid fraction 和 transform 可审计，且未进入 MLLM formal input |
| CUDA bounded Demo | 固定 val `gdcld::gdcld_val_original_00002`：`VLM_ONLY=SUCCESS`、`SEGMENT_AND_UNDERSTAND=SUCCESS`、首次 RI=`WAITING_FOR_CANDIDATE`；本次真实 candidates `[0,1,2,3]`，candidate 0 replay 未重跑 OA-AuxSeg，Pass-1、6-item retrieval、Pass-2 与 6 citations 全部成功；CUDA peak `4,684,341,760` bytes，`sealed_test_accessed=false` |
| 删除与保护资产 | M7 v4 失败后未触发旧配置/19 MiB 根的条件清理；4B Gate B 和 M5 资产未修改。2B Region 与 Text Bank 只重发布路径元数据，未改模型、Adapter、向量、Benchmark 或历史科学结论 |
| Ruff | 未安装；准确结果为 `No module named ruff` |

Unified Demo 的上述 CPU、UI 与 CUDA smoke 均为 engineering validation，不构成 Gate A/C/D、
正式 test evaluation、scientific evaluation 或 scientific acceptance。`allow_test_demo=false`，
真实 test 未读取，Demo root 中不存在 test access receipt；Frozen Evaluation 页签、selection、
推理入口和文件白名单均已移除，`frozen_evaluations=[]`。Demo run/viewer 使用 Demo-only v2
sidecar，正式 `UnifiedRequest/UnifiedResponse v1` 与底层
`candidate missing → global fallback` 合同未修改。
