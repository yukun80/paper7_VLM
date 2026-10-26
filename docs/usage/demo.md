# Unified Demo Workbench

[返回本节目录](README.md) · [文档总目录](../README.md)


只读 Workbench 复用现有 Benchmark reader、六个 `UnifiedTask`、router、runtime 和 lazy
providers，用于浏览 train/val Benchmark、人工维护 qualitative Demo Gallery，并展示空间结果、
grounded observation、Evidence Packet、Pass-2 与执行轨迹。它不修改 Benchmark、checkpoint、
Adapter、Text Bank 或正式 outputs。冻结评价不再是 Workbench 数据模式；评价 reader
与实现仍保留，但已退役 Eval-dev selection/development outputs 的默认配置已删除。

Workbench 的三个逻辑页签固定为 Benchmark Browser、Demo Gallery 和
Task Runner / Result Viewer / Trace；中文界面分别显示为“基准数据浏览”、
“Demo Gallery”和“任务运行 / 结果查看 / 执行轨迹”。

页面顶部提供 `界面语言 / Interface Language`，默认中文，可在 `中文` 与 `English` 间即时
切换三个页签的标题、字段、按钮、说明、状态消息、Dataframe 表头及 candidate/输入预览说明。
语言切换只更新 Gradio 表示层，不重新读取 Benchmark/HDF5，不加载模型、不执行推理或检索，
也不改变当前 sample、Gallery、Demo run、result task 或 candidate snapshot/token。模型 prompt、
用户输入、模型原始输出、Evidence 内容、科学 JSON、schema、枚举、ID、路径和 reason code 始终
保持原值，不做自动翻译。

推荐交互顺序是：

```text
Benchmark Browser 选择样本
→ 查看 Full Optical、逐通道 Optical/Multispectral 与真实 auxiliary preview
→ 运行 SEGMENT_ONLY 或包含 spatial task 的 Suite
→ 查看同一次 OA-AuxSeg 结果的 Candidate Preview
→ 明确选择 candidate，或人工确认 OA-AuxSeg global mask
→ 点击“运行所选 REGION_INTERPRETATION”
→ 查看 Pass-1、retrieval、citations、Pass-2 与 execution trace
```

首次运行默认 Suite 时，`REGION_INTERPRETATION` 会显示 `WAITING_FOR_CANDIDATE`，不会自动
选择 Top-1，也不会把缺失 candidate 的 global fallback 伪装成具体区域解释。切换样本或产生
新的 spatial run 后，旧 candidate 选择立即失效。逐通道与 auxiliary Gallery 只是
`Spatial Expert Input Preview`；当前 P0 MLLM formal grounded input 仍只有 Full Optical、
Binary Mask 和 Context Crop，preview 不会进入 `UnifiedRequest.auxiliary_views`。

纯 `KNOWLEDGE_QA` 与当前 Browser selection 完全解耦：不会读取 Benchmark payload、
构造 spatial input、运行 OA-AuxSeg 或产生 test receipt。问题留空时使用配置中的默认 prompt；
界面顶部 banner 和只读 `Effective prompts` 区会显示本次实际输入及其来源。其他视觉任务则消费
banner 中明确显示的当前样本；Benchmark Browser 初始化时已经定位一条记录，即使用户没有再次
点击样本按钮，也不是“无输入运行”。

安装可选 UI 依赖并从本地回环地址启动：

```bash
python -m pip install -e '.[vlm,retrieval,demo]'

python scripts/infer/demo.py \
  --config configs/runtime/demo_v1.yaml \
  --port 7860
```

默认 `allow_test_demo: false`：test 只显示锁定的 index metadata，不打开 HDF5 payload，
不能推理或加入 Gallery。只有负责人显式修改独立 Demo 授权后才能去盲化；届时每次访问先
发布 test receipt，并永久标记其不再具有 blind/sealed evaluation 属性。Reference/GT mask
始终只作 `Reference / Audit Only`，不会伪装成 `USER_MASK`。
