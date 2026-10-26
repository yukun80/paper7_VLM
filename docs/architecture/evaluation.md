# 7. 评价与发布状态机

[返回本节目录](README.md) · [文档总目录](../README.md)

本章整理自已批准的 v3.2 设计，描述约定，不代表全部已经实现或通过验证。实际状态见[当前进度](../status/README.md)。

### 7.1 独立 Gate B

4B Gate B 精确复用冻结 2B Gate B 的 256 个 val record ID 与六项预注册判据，但发布新的
protocol ID、schema、implementation identity 与输出根，并绑定 4B backend、revision、
processor、训练 checkpoint 和生成配置。不得读取 test。

4B Gate B 失败即终止后续链。另可对同一 256 条生成 2B/4B paired CI 报告，但必须标记
`report_only=true`；该报告不参与默认升级，也不能支持“4B 优于 2B”的科学结论。

### 7.2 Grounded 绝对验证

在全新、不可覆盖的 Benchmark 根构建 100 条 val-only 评价资产。确定性选择沿用预注册
seed 与规则：5 个来源各 20；每源 16 target、4 no-target；每源 target 的
small/medium/large=`5/6/5`。不得恢复或覆盖已退役 Eval-dev，不得读取 test。

零容忍标准为：

- 100/100 成功生成；
- schema、target status、input identity、no-target empty-region 合同均为 1.0；
- forbidden claim、overlay leakage、unresolved output 均为 0。

### 7.3 发布状态

```text
training_complete
  + Gate B PASS
  + 24GB resource gate PASS
  + six-task runtime smoke PASS
  = candidate

candidate
  + Grounded 100-record zero-tolerance PASS
  = engineering_validated / scientific_acceptance=false
```

默认 runtime/Demo 推荐命令只在 `engineering_validated` 后切换到 4B。升级只依据 4B
自身验收；paired report 不构成门。Gate C、专家评价和 sealed test 仍未完成，因此
`scientific_acceptance` 必须保持 false。
