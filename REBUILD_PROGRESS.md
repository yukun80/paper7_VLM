# REBUILD_PROGRESS

## 当前状态

- program: `OA_AUXSEG_VLM_REBUILD`
- phase: `2`
- phase_name: `OA_AUXSEG_MULTIMODAL`
- phase_status: `phase2_v6_five_source_cpu_contract_passed_gpu_gates_pending`
- execution_date: `2026-07-26`
- branch: `main`
- implementation_baseline_head: `6f799c615076ec233beb5273c064f881b8b76c50`
- benchmark_schema: `oa_auxseg_hdf5_v1`
- model_schema: `oa_auxseg_model_v6`
- checkpoint_schema: `oa_auxseg_checkpoint_v6`
- runtime_config_schema: `oa_auxseg_runtime_config_v5`
- inference_schema: `oa_auxseg_inference_v6`
- benchmark_small_built: `true`
- benchmark_small_access: `read_only`
- benchmark_small_sample_count: `500`
- benchmark_full_built: `false`
- model_implemented: `true`
- trainer_implemented: `true`
- evaluator_implemented: `true`
- inference_implemented: `true`
- v6_cpu_unit_tests_run: `true`
- v6_real_small_batch8_gpu_smoke_run: `false`
- v6_overfit_run: `false`
- v6_uniform_300_step_run: `false`
- v6_balanced_300_step_run: `false`
- v6_optical_only_train_run: `false`
- download_run: `false`
- commit_performed: `false`
- push_performed: `false`

## 当前 Small Benchmark

唯一 Phase 2 数据权威：

```text
/home/yukun80/codes/benchmark/oa_auxseg_hdf5_v1/small
```

本次只读使用，未修改、重建或访问 full。

| source | train | val | test | positive | empty | total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gdcld | 20 | 40 | 40 | 60 | 40 | 100 |
| lmhld | 36 | 35 | 29 | 53 | 47 | 100 |
| landslidebench_agent | 34 | 34 | 32 | 50 | 50 | 100 |
| landslide4sense | 34 | 34 | 32 | 50 | 50 | 100 |
| multimodal_landslide | 34 | 66 | 0 | 67 | 33 | 100 |
| total | 158 | 209 | 133 | 280 | 220 | 500 |

- source_selection: `subset`
- included_sources:
  `gdcld, lmhld, landslidebench_agent, landslide4sense, multimodal_landslide`
- excluded_sources: `sen12landslides`
- optical_channel_counts: `3, 4, 12`
- auxiliary_registry: `dem, insar_velocity, slope`
- auxiliary_combinations: `none`, `dem+slope`, `dem+insar_velocity`
- train positive/empty: `106/52`
- train native auxiliary/none: `68/90`
- val positive/empty: `105/104`
- shard_count: `14`
- directory_size_bytes: `357638820`
- index_sha256:
  `85c86a8d09dc2f602f04dad890ca65dab59e16235415e818427673dc1483f2df`
- manifest_sha256:
  `6f386b274b4fea3272ce065488e327719e57f9a39b025e8e2372cb27a3230ed8`
- deep_validation: `500/500 pass`
- raw_dataloader_smoke: `pass`
- zscore_dataloader_smoke: `pass`

Benchmark builder 仍保留通用 Sen12/SAR 构建能力，但 Phase 2 v6 模型合同明确排除
Sen12、10 通道光学和 SAR。历史六源产物不再是当前训练权威。

## Phase 2 v6 已完成

- 将受审计辅助目录收敛为 `dem, insar_velocity, slope`；未知模态和 SAR 立即报错。
- registry 由当前 Benchmark index 动态生成，只为实际出现的已审计模态建立 adapter；
  模态列顺序按目录确定，不能由输入字典插入顺序改变。
- 删除 Phase 2 的 Sen12 10 通道 RGB 映射；保留 3 通道官方 stem 等价、4 通道 NIR
  和 12 通道 Sentinel-2 extra-band 残差分支。
- 保留 ConvNeXt-Small、四尺度 `96/192/384/768`、深度 `3/3/27/3` 的完整
  validity-aware MSPA、空间 selector、FRM/FFM 和逐阶段双流传播。
- FFM 保持全通道 `KᵀV`；fractional coverage 只乘入 V，全有效时与
  `softmax((KᵀV)×scale)` 一致。
- support/coverage 合同不变；下采样 coverage 使用 area average，无效值在 stem、
  downsample、MSPA、selector 和融合前后均受 mask 约束。
- 六个 variant 仍由同一模型和训练器支持：
  `optical_only`、`direct_concat`、`mean_auxiliary_fusion`、
  `cmnext_injection`、`injection_quality`、`proposed_dropout`。
- 当前 active registry 的输出合同：
  - mask logits/probability: `[B,1,224,224]`
  - no-target score: `[B]`
  - modality weights: `[B,4]`
  - stride 4/8/16/32 weight maps:
    `[B,4,56,56]`、`[B,4,28,28]`、`[B,4,14,14]`、`[B,4,7,7]`
  - weight order: `dem, insar_velocity, slope, __null__`
  - candidate regions: threshold 0.5、8 邻域、最小 16 像素
  - region feature: `128 + 128 + 8 + 4 = 268`
- 动态 registry 子集已验证：仅 `dem` 时输出权重 `[B,2]`、区域特征 266 维。
- 当前五源完整模型参数量为 `152916881`。
- 模型/checkpoint/inference schema 升级为 v6，runtime config 升级为 v5；
  旧 v5 checkpoint 和 v4 runtime config 明确拒绝，不提供兼容或转换包装。
- checkpoint 固化并严格校验：
  - 完整 model registry 与 region feature dim
  - modality order 和四尺度图合同
  - Benchmark index/manifest SHA-256
  - source_selection、included_sources、excluded_sources
  - ConvNeXt-Small SHA-256
  - optimizer、scheduler、AMP、step 和全部 RNG
- evaluate/infer 在加载时同时重验 Benchmark 合同和 model registry；推理 manifest
  使用 v6，并以动态列数写 JSONL/NPZ。

## 训练与采样合同

- Loss 仍只使用 `BCEWithLogits + soft Dice`。
- optimizer 仍分 backbone/new 与 decay/no-decay 四组；extra-band、辅助编码、
  selector、融合和 decoder 使用 new LR。
- train 使用带 floor 的 warmup-cosine、bf16、clip 1.0；capacity overfit 强制
  FP32、无 stochastic regularization、weight decay 0、clip 5.0。
- `StatefulTrainingBatcher` 每步严格返回 8 个样本，跨 permutation 边界补齐；
  permutation、cursor、RNG 和累计计数均可恢复。
- runtime config 已删除 `auxiliary_null_probability`。网络内部空间 null selector 保留。
- `proposed_dropout` 对原生有辅助样本：
  1. 在 `1..N` 上均匀采样 cardinality；
  2. 对选中模态执行 `p=0.2` dropout；
  3. 若全部被丢弃，使用同一可恢复 RNG 从原选择中恢复一个模态。
- 原生有辅助样本因此不会被人为变为空；checkpoint 保存 cardinality/dropout 计数、
  RNG 和 `dropout_restored` 次数。
- 训练报告记录 native-none、single、multi、all、dropout-restored，原生有辅助样本数、
  实际激活数、全局辅助曝光率和 `conditional_active_auxiliary_fraction`。
- 新策略要求 `conditional_active_auxiliary_fraction=1.0`；不再强制全局辅助曝光
  至少 40%，因为当前 train 原生有辅助样本只有 `68/158`。
- uniform 仍为默认。`balanced_target_presence` 仅按 4 positive + 4 empty 采样，
  不使用 source；只有达到既定 Dice/FPR 对照标准后才能成为后续默认。
- `checkpoint_last.pt` 用于恢复，`checkpoint_best.pt` 按 val Dice、val loss、
  no-target FPR 依次选择。
- train/overfit 终端继续使用低噪声 tqdm；JSONL/JSON 报告与简洁 CLI 终态不变。

## 实际检查

| 检查 | Exit | 结果 |
| --- | ---: | --- |
| Phase 2 unittest | 0 | 33/33；内部测试耗时 22.277 s |
| Phase 1B 回归 | 0 | 10/10；内部测试耗时 4.148 s |
| 当前 small deep validator | 0 | 14 shards、500/500、errors/warnings=0 |
| raw DataLoader smoke | 0 | 五源、3/4/12 通道、none/single/all 通过 |
| z-score DataLoader smoke | 0 | 五源、3/4/12 通道、none/single/all 通过 |
| 六 variant 合成 forward/backward/step | 0 | mask、4 列权重、四尺度图、268 维区域合同通过 |
| 动态 registry 子集 | 0 | dem-only 为 2 列权重和 266 维区域特征 |
| SAR/10 通道拒绝 | 0 | 未注册 SAR 与未审计 10 通道签名均明确失败 |
| 非空 modality dropout | 0 | 多 seed/dropout 均不把原生有辅助样本变为空，恢复序列一致 |
| 三个当前辅助 adapter | 0 | DEM/InSAR/slope 首次 backward 均有非零梯度与参数更新 |
| MSPA/FRM/FFM/quality | 0 | 全部 MSPA block、四层 FRM/FFM 和 selector 梯度通过 |
| invalid/zero coverage/null | 0 | 极大无效值不变、零覆盖权重 0、空间 null 合同通过 |
| 无辅助硬旁路 | 0 | logits/probability/global weights/四尺度图与 optical-only 相同 |
| checkpoint v6 | 0 | 三项输出及四尺度图重载误差不超过 `1e-6`，Benchmark 合同错配被拒绝 |
| checkpoint v5 拒绝 | 0 | 先按 schema 明确拒绝 |
| v6 推理合同 | 0 | Landslide4Sense 与 multimodal-landslide JSONL/NPZ 重载及拒绝覆盖通过 |
| 官方 3 通道 stem | 0 | 与 torchvision ConvNeXt-Small stem 逐元素一致 |
| official state_dict | 0 | `strict=True`，SHA-256 `0c510722...bfab9a` |
| Python `py_compile` | 0 | Phase 1B/Phase 2 模块、CLI 与测试通过 |
| 六份 runtime config 严格解析 | 0 | 均为 runtime v5，输出目录均迁移到 v6 |
| CLI `--help` | 0 | train/overfit/evaluate/infer/smoke 入口正常 |
| shell `bash -n` | 0 | small/full builder 入口语法通过 |
| CUDA/NVML probe | 0 | CUDA available=false、device_count=0，NVML 初始化失败 |
| git diff --check | 0 | 通过 |

上述均为 CPU/合同/只读数据验收，不是 GPU 显存或分割精度结果。

## 历史训练结果

旧 Benchmark 上的 v4/v5 训练目录全部保留且只读，但与 v6 schema、五源 registry 和
当前 Benchmark hash 不兼容，不能恢复或冒充当前验收。

- 历史 300-step proposed：EMA 下降 53.25%、val Dice 0.4299、
  IoU 0.2738、峰值 3.81 GiB、重载差异 0；只代表旧合同工程闭环。
- 历史 1000-step capacity overfit：loss 下降 79.36%、train Dice 0.9029，
  52 个空样本中仍有 1 个误报，未通过容量阈值。
- 排除 Sen12 消除了旧训练中的主要拟合短板之一，但不能据此宣称 v6 已提升精度；
  必须重新训练验证。

## 未运行

- full Benchmark 构建、验证、smoke 或任何 full 访问
- v6 六 variant 真实 small batch=8 GPU smoke 和 `<23 GiB` 显存验收
- v6 全部 158 条 train、最多 1000-step capacity overfit
- v6 uniform 与 balanced-target 两次同 seed 300-step proposed 对照
- v6 optical-only 短训练与 val 评价
- 训练后 Landslide4Sense 和 multimodal-landslide 正式推理导出
- full 上的 50,000-step 正式多模态训练
- Region Grounding、VLM Description、RAG 或端到端集成
- 数据、模型或依赖下载
- commit 或 push

## 已知限制

- 当前 train 为 positive/empty=`106/52`，val 为 `105/104`。
- GDCLD train 为 20/20 positive，而 val 为 20 positive + 20 empty；
  multimodal-landslide train 为 34/34 positive，而 val 为 33 positive + 33 empty。
  该分布差异仍可能推高验证集 no-target FPR，不能由 registry 迁移自动解决。
- LandslideBench_agent 保留负责人批准的跨 split location 例外。
- LMHLD 和 Landslide4Sense 缺少可靠地理 parent/group，不伪造空间关系。
- multimodal InSAR 保留 encoded 数值和 validity，不推断未确认物理单位。
- 当前 v6 模型没有 SAR 能力；未来若引入重新审计的高质量 SAR Benchmark，必须显式
  升级 schema 与 registry，不能静默接入。
- full 尚不存在；五源 full 的实际 split、压缩体积、显存和耗时都必须由负责人运行后确认。
- 旧 3.81 GiB 峰值不代表 v6 六 variant 或 capacity overfit 显存。

## 下一步人工 GPU 顺序

```bash
cd /home/yukun80/codes/paper7_VLM
conda activate qwen3vl

python scripts/phase2_oa_auxseg/run_oa_auxseg.py smoke \
  --config configs/phase2_oa_auxseg/small_smoke.json

python scripts/phase2_oa_auxseg/run_oa_auxseg.py overfit \
  --config configs/phase2_oa_auxseg/small_overfit.json

python scripts/phase2_oa_auxseg/run_oa_auxseg.py train \
  --config configs/phase2_oa_auxseg/small_proposed_dropout.json

python scripts/phase2_oa_auxseg/run_oa_auxseg.py train \
  --config configs/phase2_oa_auxseg/small_proposed_dropout_balanced.json

python scripts/phase2_oa_auxseg/run_oa_auxseg.py train \
  --config configs/phase2_oa_auxseg/small_optical_only.json
```

验收标准：

1. 六 variant 均 exit 0，真实 batch=8 峰值显存各自 `<23 GiB`。
2. capacity overfit 使用全部 158 条 train，loss 下降至少 90%、micro Dice ≥0.95、
   positive-only Dice ≥0.90、空 mask FPR=0、空样本平均概率 ≤0.01。
3. 三个辅助 adapter、共享 MSPA、四层 FRM/FFM 和 quality selector 均有非零梯度及更新。
4. uniform/balanced 各运行 300 step、累计 2400 样本；
   `conditional_active_auxiliary_fraction=1.0`，观察到 none/single/all，checkpoint
   严格重载误差 ≤`1e-6`。
5. balanced 仅在 val FPR 至少下降 0.10 且 positive Dice 下降不超过 0.01，或
   overall Dice 至少提高 0.01 且其他指标不退化时采用。
6. optical-only 完成短训练和 val；Landslide4Sense、multimodal-landslide 的全部
   v6 推理输出可重载。

未来 full 只允许同样排除 Sen12 的五源合同，且本阶段不构建或访问：

```bash
bash scripts/phase1_benchmark_build/run_build_full.sh \
  --exclude-source sen12landslides

python scripts/phase2_oa_auxseg/run_oa_auxseg.py train \
  --config configs/phase2_oa_auxseg/full_proposed_dropout.json
```

两条 full 命令均未运行。small GPU 验收完成后仍先停止，由项目负责人决定后续；
不得进入 Region Grounding、VLM Description 或 RAG。
