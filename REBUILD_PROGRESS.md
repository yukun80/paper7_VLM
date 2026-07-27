# REBUILD_PROGRESS

## 当前状态

- program: `OA_AUXSEG_VLM_REBUILD`
- phase: `1`
- phase_name: `OA_AUXSEG`
- phase_status: `v6_cpu_contract_passed_full_b16_e100_training_external`
- next_phase_name: `MASK_GROUNDED_VLM_DESCRIPTION`
- execution_date: `2026-07-27`
- branch: `main`
- implementation_baseline_head: `41e64d7f80bbcca7e2fb352ee36df748c424ef40`
- benchmark_schema: `oa_auxseg_hdf5_v1`
- model_schema: `oa_auxseg_model_v6`
- checkpoint_schema: `oa_auxseg_checkpoint_v6`
- runtime_config_schema: `oa_auxseg_runtime_config_v5`
- inference_schema: `oa_auxseg_inference_v6`
- benchmark_small_built: `true`
- benchmark_small_access: `read_only`
- benchmark_small_sample_count: `500`
- benchmark_full_built: `true`
- benchmark_full_sample_count: `53645`
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
- full_b16_e100_config_prepared: `true`
- full_b16_e100_train_started_external: `true`
- full_b16_e100_train_complete: `false`
- download_run: `false`
- commit_performed: `false`
- push_performed: `false`

## 当前 Small Benchmark

唯一 OA-AuxSeg 数据权威：

```text
/home/yukun80/codes/benchmark/oa_auxseg_hdf5_v1/small
```

本次只读使用，未修改或重建。

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

Benchmark builder 仍保留通用 Sen12/SAR 构建能力，但 OA-AuxSeg v6 模型合同明确排除
Sen12、10 通道光学和 SAR。历史六源产物不再是当前训练权威。

## 当前 Full Benchmark

负责人构建的五源 full 已存在：

```text
/home/yukun80/codes/benchmark/oa_auxseg_hdf5_v1/full
```

| source | train | val | test | total |
| --- | ---: | ---: | ---: | ---: |
| gdcld | 7,897 | 4,459 | 1,091 | 13,447 |
| lmhld | 19,729 | 5,637 | 2,819 | 28,185 |
| landslidebench_agent | 1,701 | 210 | 219 | 2,130 |
| landslide4sense | 3,039 | 380 | 380 | 3,799 |
| multimodal_landslide | 4,395 | 1,689 | 0 | 6,084 |
| total | 36,761 | 12,375 | 4,509 | 53,645 |

- source_selection: `subset`
- included_sources:
  `gdcld, lmhld, landslidebench_agent, landslide4sense, multimodal_landslide`
- excluded_sources: `sen12landslides`
- shard_count: `126`
- directory_size_bytes: `32404761903`
- index_sha256:
  `389877226249d2477bdda62d937950339e9fa60df35558b945d02757e8d0da42`
- manifest_sha256:
  `9a3b1478ed844f234e32b839fded67a937c49d202e3d8f5efd7db52596b5a00a`

本次只读核对 manifest、文件数量和目录大小，没有重建 full，也没有重跑完整 deep
validator。

## OA-AuxSeg v6 已完成

- 将受审计辅助目录收敛为 `dem, insar_velocity, slope`；未知模态和 SAR 立即报错。
- registry 由当前 Benchmark index 动态生成，只为实际出现的已审计模态建立 adapter；
  模态列顺序按目录确定，不能由输入字典插入顺序改变。
- 删除 OA-AuxSeg 的 Sen12 10 通道 RGB 映射；保留 3 通道官方 stem 等价、4 通道 NIR
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
- `StatefulTrainingBatcher` 每步严格返回 runtime config 指定的 device batch，
  跨 permutation 边界补齐；permutation、cursor、RNG 和累计计数均可恢复。
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
- 当前正式 full 配置为物理 batch 16、`max_steps=229757`，累计曝光
  `3,676,112` 条样本，即 `100.0003` 个 train pass；每 `4596` step 评价和保存。
  配置关闭 activation checkpointing、使用 bf16，并将 backbone/new LR 线性放大为
  `6e-5/6e-4`。batch 8 checkpoint 不允许恢复到该配置。

## 路线调整与下一阶段

- 独立 Region Grounding Adapter 不再是研究主线，也不作为论文或阶段验收条件。
- 为避免破坏已验收配置、checkpoint 和命令，OA-AuxSeg 工程路径暂时保留
  `phase2` 历史实现名；算法方案中的当前阶段统一为 Phase 1。
- OA-AuxSeg 继续输出 global mask、candidate regions、no-target，以及可选只读
  region features；候选仍由 threshold 0.5、8 邻域和最小 16 像素的既有规则产生。
- 下一阶段为 Mask-Grounded VLM Description：直接接收 global mask，或由调用方明确
  指定的 candidate-region mask，并结合多模态证据和问题生成结构化事实、描述与回答。
- 首版只支持 global/all regions、region ID、bbox/点击、面积/位置规则和编号 overlay
  等确定性区域输入。可选轻量 region scorer 仅在完整主线之后按需评估。
- 本轮已清理独立 Grounding 的打包入口、依赖、公共 helper、专属 Benchmark 和
  规则基线输出；该独立模块的源码、CLI 与测试目录在清理开始时已不存在。

## 实际检查

| 检查 | Exit | 结果 |
| --- | ---: | --- |
| OA-AuxSeg unittest | 0 | 33/33；本轮内部测试耗时 32.982 s |
| Phase 1B 回归 | 0 | 10/10；本轮内部测试耗时 5.716 s |
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
| Python `py_compile` | 0 | Phase 1B/OA-AuxSeg 模块、CLI 与测试通过 |
| 七份 runtime config 严格解析 | 0 | 均为 runtime v5；新增 full batch16/100-epoch 配置 |
| CLI `--help` | 0 | OA-AuxSeg train/overfit/evaluate/infer/smoke 入口正常 |
| 打包与公共 API 清理 | 0 | 仅保留 `oa-auxseg` 入口，独立候选 helper 不再公开 |
| 三阶段文档合同 | 0 | canonical 共 13 章，只有 OA-AuxSeg/Description/RAG 三个 Phase |
| 独立 Grounding 残留搜索 | 1 | 无匹配；`rg` 的 1 表示活动实现残留为零 |
| shell `bash -n` | 0 | small/full builder 入口语法通过 |
| CUDA/NVML probe | 0 | CUDA available=false、device_count=0，NVML 初始化失败 |
| git diff --check | 0 | 通过 |

上述均为 OA-AuxSeg 的 CPU、合同和只读数据验收，不是 GPU 显存或正式分割精度结果。

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

- 本次没有重跑 full deep validator、summarizer 或 DataLoader smoke
- v6 六 variant 真实 small batch=8 GPU smoke 和 `<23 GiB` 显存验收
- v6 全部 158 条 train、最多 1000-step capacity overfit
- v6 uniform 与 balanced-target 两次同 seed 300-step proposed 对照
- v6 optical-only 短训练与 val 评价
- 训练后 Landslide4Sense 和 multimodal-landslide 正式推理导出
- 外部 Full batch16、229,757-step、100-epoch 训练尚未完成或验收
- Mask-Grounded VLM Description、RAG 或端到端集成
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
- 五源 full 已存在，但本次没有重跑完整 deep validator；正式 batch16 的峰值显存、
  吞吐和总耗时仍需由前50 step及后续训练确认。
- 旧 3.81 GiB 峰值不代表 v6 六 variant 或 capacity overfit 显存。
- 当前没有已验收 Small OA-AuxSeg checkpoint 或正式 inference export，不能将
  Full 训练中的未验收中间状态作为正式 Description 区域证据。
- 当前进程 CUDA 不可用且外部训练占用 GPU，本轮不并行启动 VLM 任务。
- 语义 mask 合并相邻滑坡时，候选提取仍只能产生合并区域；candidate regions
  不宣称为人工实例，region features 也不是 Description 的前置条件。

## 下一步 OA-AuxSeg 正式命令与恢复合同

尚未完成的 Small GPU 验收仍保留，但不能与当前 Full 训练并行：

```bash
python scripts/phase2_oa_auxseg/run_oa_auxseg.py smoke \
  --config configs/phase2_oa_auxseg/small_smoke.json

python scripts/phase2_oa_auxseg/run_oa_auxseg.py overfit \
  --config configs/phase2_oa_auxseg/small_overfit.json
```

本轮未启动、停止、恢复或修改 Full 训练；只读观察其隔离日志正在增长。OA-AuxSeg
验收后直接进入 Mask-Grounded VLM Description，不再提供独立 Grounding 命令。

负责人当前 Full 训练与同配置恢复合同仍为：

```bash
python scripts/phase2_oa_auxseg/run_oa_auxseg.py train \
  --config configs/phase2_oa_auxseg/full_proposed_dropout_b16_nockpt_e100.json

python scripts/phase2_oa_auxseg/run_oa_auxseg.py train \
  --config configs/phase2_oa_auxseg/full_proposed_dropout_b16_nockpt_e100.json \
  --resume outputs/phase2_oa_auxseg/full_proposed_dropout_v6_b16_nockpt_e100/checkpoint_last.pt
```

batch-16 不得从旧 batch-8 checkpoint 恢复；若峰值超过 21.5 GiB 或 OOM，应另建
batch-12 配置，不能降低 224 分辨率或关闭辅助路径。正式评价使用
`checkpoint_best.pt`。
