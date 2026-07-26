# REBUILD_PROGRESS

## 当前状态

- program: `OA_AUXSEG_VLM_REBUILD`
- phase: `2`
- phase_name: `OA_AUXSEG_MULTIMODAL`
- phase_status: `phase2_v5_cpu_contract_passed_gpu_gates_pending`
- execution_date: `2026-07-26`
- branch: `main`
- implementation_baseline_head: `2ea33635dba3246019e66a3a843cd7a4f351e12c`
- benchmark_schema: `oa_auxseg_hdf5_v1`
- benchmark_sen12_adaptation_date: `2026-07-25`
- benchmark_small_refresh_date: `2026-07-26`
- phase2_path_migration_date: `2026-07-26`
- phase2_sar_registry_date: `2026-07-26`
- phase2_v4_full_deliver_fusion_date: `2026-07-26`
- phase2_v5_training_optimization_date: `2026-07-26`
- phase2_terminal_observability_date: `2026-07-26`
- model_schema: `oa_auxseg_model_v5`
- checkpoint_schema: `oa_auxseg_checkpoint_v5`
- runtime_config_schema: `oa_auxseg_runtime_config_v4`
- inference_schema: `oa_auxseg_inference_v5`
- benchmark_small_built: `true`
- benchmark_full_built: `false`
- model_implemented: `true`
- trainer_implemented: `true`
- evaluator_implemented: `true`
- inference_implemented: `true`
- training_run: `true`
- cpu_trainer_smoke_run: `true`
- v4_synthetic_cpu_optimizer_smoke_run: `true`
- v4_real_small_batch8_smoke_run: `false`
- v5_cpu_unit_tests_run: `true`
- v5_real_small_batch8_smoke_run: `false`
- v5_overfit_run: `false`
- v5_uniform_300_step_run: `false`
- v5_balanced_300_step_run: `false`
- official_small_training_run: `true`
- gpu_run: `true`
- download_run: `false`
- commit_performed: `false`
- push_performed: `false`

## 已完成

- OA-AuxSeg 模型相关路径从 Phase 1 完整迁移到 Phase 2；`oa_groundrag` 保留为项目级
  Python 包名，不保留旧 `phase1` 模型 alias 或兼容包装。
- 建立 `oa_groundrag/phase2/` 完整多模态 OA-AuxSeg：
  - 严格 registry、batch、运行配置、输出和 checkpoint schema；
  - 为 3/4/10/12 通道签名建立共享官方 RGB `3→96` 分支和签名专属
    `Cextra→96` 零初始化残差，保留 Sentinel-2 B01–B12、Sen12 B02–B12 和 NIR；
  - 已审计 RGB/B04-B03-B02 位置进入共享官方分支，extra-band 归入 `new_lr`
    参数组；3 通道 stem 与官方输出逐元素一致，direct concat 复用同一分解；
  - DEM、slope、encoded InSAR、升轨 SAR 和降轨 SAR 使用独立
    7×7/stride-4/padding-3 stem，之后共享 3×3/stride-2/padding-1 下采样和
    auxiliary MSPA；
  - validity-aware 全深度 MSPA 使用 `96/192/384/768` 通道、`3/3/27/3`
    深度、`8/8/4/4` MLP ratio、7×7 depthwise context、3/7/11 masked pooling、
    Channel LayerNorm、`1e-2` LayerScale 和 stochastic depth；
  - support 与 fractional coverage 双合同，coverage 下采样使用 area average；
  - stride 4/8/16/32 均在模态仍独立时执行空间选择；quality/proposed 使用
    full-channel local scorer、null auxiliary 和零覆盖/零重叠硬屏蔽；
  - cmnext 使用逐模态 sigmoid score、`(1+score)×feature` 和 masked channel-wise max；
  - 四尺度 FRM 保持 `lambda_channel=lambda_spatial=0.5`，不使用弱残差缩放；
  - 四尺度 FFM 使用完整 stage 通道、reduction=1、`3/6/12/24` heads 的
    `KᵀV` channel-context cross attention 和完整 channel embedding；
  - v5 FFM 对 fractional coverage 只执行 `V×coverage`，不再除以 coverage/token
    总量；全有效时严格等价于 DELIVER 的 `softmax((KᵀV)×scale)`；
  - FRM 校正后的光学和辅助特征分别传播到下一 stage，无辅助或零覆盖时硬旁路；
  - 原生有辅助样本以 0.1 概率显式走 null，否则在 `1..N` 上均匀采样 cardinality，
    再执行 `p=0.2` modality dropout；原生无辅助样本保持 optical-only；
  - 可恢复定长训练 batcher 跨 permutation 边界补齐，固定每步 8 样本并保存
    permutation、cursor、RNG；另提供不使用 source 的 4-positive/4-empty sampler；
  - 共享 ConvNeXt-Small 光学主干和 512 维 SegFormer-style 四尺度 decoder；
  - 四尺度 `[B,6,H_s,W_s]` 空间诊断图及 coverage pooling 的 `[B,6]` 摘要；
  - 两个独立 128 维区域投影，保持 270 维区域接口；
  - BCEWithLogits + soft Dice，以及完整分割/no-target 指标；
  - 阈值 0.5、8 邻域、最小 16 像素区域和 registry 驱动的 270 维 region feature；
  - 原子 checkpoint、严格重载、分层评价、原子 JSONL/NPZ 推理。
- evaluate/infer 同时严格校验 Benchmark index SHA-256 与 checkpoint registry，
  并固化光学/辅助 stem、四尺度 selector、MSPA、FRM/FFM、decoder 和 validity 合同。
- 受支持辅助模态集中为单一 registry catalog；诊断列固定为
  `dem, insar_velocity, sar_ascending, sar_descending, slope, __null__`，
  两条 SAR 保持独立身份，未知传感器不会自动注册。
- region feature 按 `2 × region_projection_dim + 8 + weight_dim` 推导，当前输出
  `128 + 128 + 8 + 6 = 270`；模型、checkpoint 与推理 schema 升级到 v5，
  runtime config 升级到 v4，不提供旧 v4 checkpoint 兼容层。
- 同一训练器实现六个 variant：`optical_only`、`direct_concat`、
  `mean_auxiliary_fusion`、`cmnext_injection`、`injection_quality`、
  `proposed_dropout`。
- 建立 `scripts/phase2_oa_auxseg/run_oa_auxseg.py` 单一
  `train/evaluate/infer/smoke/overfit` CLI 和严格 JSON 配置。
- `train/overfit` 已增加低噪声 tqdm 终端进度：交互终端单行显示 loss、EMA、
  BCE、Dice loss、两组学习率、吞吐、ETA 和显存；验证显示运行 loss/Dice/IoU，
  checkpoint、最终评价和重载显示阶段耗时。非 TTY 按 `log_interval` 输出无控制符
  固定行，完整指标继续写 JSON；默认终态为简洁摘要，`--full-report-json` 可输出
  完整 JSON。
- AdamW 现按 backbone/new 与 decay/no-decay 四组组织：ConvNeXt、共享 RGB stem
  使用 `3e-5`，extra-band、辅助编码、融合、selector 和 decoder 使用 `3e-4`；
  bias、LayerNorm 和 LayerScale 不衰减。production stochastic depth/MSPA
  DropPath/decoder dropout 均为 0.1，cosine floor 为 0.05；capacity overfit 强制
  weight decay/dropout/stochastic depth 为 0、FP32、clip=5、LR floor=0.10。
- train 同时写 `checkpoint_last.pt` 与按 val Dice、val loss、no-target FPR
  依次择优的 `checkpoint_best.pt`；overfit 每 100 step 评价完整 train，全部容量
  阈值通过时允许提前停止，失败异常列出具体阈值。
- 官方 ConvNeXt-Small state_dict、SHA-256 和 3 通道 signature stem 等价已在 CPU
  严格验证；source 仅用于报告，没有进入模型。
- 随机初始化 v5 checkpoint 已在真实 Landslide4Sense、multimodal-landslide 和
  Sen12Landslides val 样本上验证原子推理、拒绝覆盖、四尺度权重图和 270 维区域特征重载；
  这不等同于训练结果。
- 建立 `scripts/phase1_benchmark_build/`：
  - 公共 schema、原子 I/O、重采样、hash、Dataset 和 collate；
  - GDCLD、LMHLD、LandslideBench_agent、Landslide4Sense、multimodal、
    Sen12Landslides 六个显式源适配器；
  - small/full builder；
  - 独立 validator、summarizer 和 DataLoader smoke；
  - small/full shell 入口。
- 更新 `pyproject.toml`，声明 NumPy、h5py、SciPy、PyTorch、torchvision 和 tqdm
  依赖，并注册 OA-AuxSeg CLI。
- 建立 `tests/phase1_benchmark_build/` 六源合成 fixture。
- 采用分片 HDF5 加 JSONL 索引；磁盘 optical/mask 为 `[N,C,H,W]`/
  `[N,1,H,W]`，每条索引通过 `storage.shard + storage.row` 定位。`N` 是 shard
  样本数，不是 DataLoader batch size。
- 分片按 source、split、辅助模态签名隔离；dense dataset 使用 `[1,1,H,W]` chunk，
  并启用 gzip level 4、shuffle 和 Fletcher32。目标存在时拒绝覆盖，临时目录完成后
  原子发布。
- 单样本 mask 为 `[1,H,W]`，batch mask 为 `[B,1,H,W]`。
- 不保存 `mask_validity`；标签无效位置直接置背景 0。
- 保留 image/modality pixel validity 与 channel validity；影像无效值直接置 0。
- 保留 Sentinel-2 B01–B12，不用全零张量伪造缺失辅助模态。
- Landslide4Sense 使用固定 seed 在 positive/background 内确定性分配 80/10/10。
- LandslideBench_agent 全部保留源 split，并在 manifest 中记录 311 个已批准跨 split
  `location_key` 例外。
- Sen12Landslides 使用 post 时相的 10 个 Sentinel-2 波段；pre 仅保留 provenance。
  升轨/降轨 SAR 按源 channel validity 独立存在，DEM 必须存在，不创建缺失模态占位。
- Sen12Landslides 的 15 个 region 作为已知 group，固定 seed 分组得到
  train/val/test = 5,253/759/659，region 不跨 split。

## Small Benchmark

输出：

```text
/home/yukun80/codes/benchmark/oa_auxseg_hdf5_v1/small
```

构建参数：

```text
patch_size=224
small_per_source=100
seed=20260724
split_seed=20260724
shard_target_mib=512
```

实际统计：

| source | train | val | test | positive | background | 合计 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gdcld | 20 | 40 | 40 | 60 | 40 | 100 |
| lmhld | 36 | 35 | 29 | 53 | 47 | 100 |
| landslidebench_agent | 34 | 34 | 32 | 50 | 50 | 100 |
| landslide4sense | 34 | 34 | 32 | 50 | 50 | 100 |
| multimodal_landslide | 34 | 66 | 0 | 67 | 33 | 100 |
| sen12landslides | 46 | 22 | 32 | 100 | 0 | 100 |
| 合计 | 204 | 231 | 165 | 380 | 220 | 600 |

- shard_count: `23`
- output_size_bytes: `546674219`
- index_sha256:
  `d6f68dd92173a5da9bd2917a4c0632fa792a4bb56b044bdd0ee956fdfeebb4ec`
- deep_validation: `pass`
- checked_samples: `600/600`
- validator_errors: `0`
- validator_warnings: `0`
- raw_dataloader_smoke: `pass`
- zscore_dataloader_smoke: `pass`

历史记录：五源 160 条、14 shards 的产物和六源 192 条、23 shards 的 `/tmp`
适配验证是早期阶段 1B 里程碑，均不是当前 `small` 路径中的产物。

## 实际检查

| 检查 | Exit | 结果 |
| --- | ---: | --- |
| Phase 2 Python `py_compile` | 0 | 模型、引擎、CLI 和测试通过 |
| Phase 2 `unittest` | 0 | 33/33 v5 多模态、训练优化、条件诊断与终端可观测性测试通过 |
| Phase 2 测试资源 | 0 | elapsed 27.030 s；CPU 合同测试，不是 GPU 显存或精度结果 |
| tqdm TTY/非 TTY/CLI 测试 | 0 | 3/3；动态条、限流纯文本、异常关闭、简洁/完整报告通过 |
| 当前六源 small Phase 2 registry | 0 | train=204、3/4/10/12 通道及五种辅助模态全部注册 |
| 六 variant 合成 forward/backward/step | 0 | 3/4/10/12 通道与五辅助模态全部完成，loss/gradient 有限 |
| 官方 3 通道 stem 等价 | 0 | signature stem 与 torchvision 官方 stem 逐元素相同 |
| 多光谱额外波段梯度 | 0 | Sentinel-2 非 RGB 初始化通道首次 backward 梯度非零 |
| FFM DELIVER 数值公式 | 0 | 全 coverage 与 `softmax((KᵀV)×scale)` 等价；fractional coverage 仅乘入 V |
| v5 optimizer/LR | 0 | RGB/backbone 与 extra/new LR 分组、bias/norm/LayerScale no-decay、cosine floor 通过 |
| v5 定长 sampler | 0 | batch 恒为 8、跨 204 样本边界补齐、resume 后索引完全一致、4+4 target balance 通过 |
| v5 300-step sampler dry-run | 0 | 2400 样本；none/single/multi/all=1325/664/52/359，有效辅助曝光 44.79%；不是模型训练 |
| 官方 ConvNeXt-Small 严格加载 | 0 | `strict=True`，SHA-256 `0c510722...bfab9a` |
| 当前 CUDA/NVML probe | 0 | torch 2.8.0+cu128；CUDA available=false、device_count=0，NVML 初始化失败 |
| 此前 CUDA/NVML probe | 255 | 历史会话被操作系统阻断；不再代表当前 4090 训练现场 |
| v4 300-step proposed dropout | 未捕获 | 只读产物完整，四项工程 acceptance=true；EMA 下降 53.25%，val Dice 0.4299、IoU 0.2738 |
| v4 proposed checkpoint reload | 未捕获 | logits/probability/weights/四尺度图最大差异均为 0 |
| v4 proposed GPU 峰值 | 未捕获 | `training_report.json` 记录 3.8086 GiB；原 Shell exit code 未单独保存 |
| v4 1000-step overfit | 非零 | 训练完整后主动验收失败；loss 下降 79.36%、train Dice 0.9029、52 个空样本中 1 个误报 |
| SAR 稀疏/顺序/invalid/zero-coverage | 0 | 升轨、降轨、双轨身份及退化、不变性合同通过 |
| fractional validity | 0 | 1/64 coverage、部分通道、全无效及 area 下采样通过 |
| 空间选择与渐进传播 | 0 | masked max、空间 softmax/null 和 stride 4 改变 stride 8 输入通过 |
| 首次辅助梯度 | 0 | 五 adapter、全部 36 个 MSPA block、四层 FRM/FFM、四个 quality selector 非零 |
| 无辅助硬旁路 | 0 | logits/probability、`[B,6]` 摘要和四尺度图均与 optical-only 逐元素相同 |
| checkpoint v5 严格重载 | 0 | logits/probability/weights/四尺度图一致，旧 v4 明确拒绝 |
| 随机权重真实来源推理合同 | 0 | 三源 JSONL/NPZ、四尺度图、拒绝覆盖、270 维重载通过 |
| shell `bash -n` | 0 | small/full 入口通过 |
| 阶段 1B `unittest` | 0 | 6/6 回归通过 |
| 合成 small/full schema | 0 | 索引字段一致 |
| 合成重复 small | 0 | sample ID 顺序和 index 字节一致 |
| 损坏索引检测 | 0 | validator 正确失败 |
| 缺失 shard 检测 | 0 | validator 正确失败 |
| 错误 shape/validity 检测 | 0 | validator 正确失败 |
| 非法 mask 检测 | 0 | validator 正确失败 |
| 当前实际 small 构建 | 0 | 六源 600 条、23 shard |
| 当前实际 small deep validator | 0 | 600/600，通过 |
| 当前实际 small summary | 0 | 六源、split、通道和模态统计完整 |
| DataLoader raw smoke | 0 | none/single/all 通过 |
| DataLoader z-score smoke | 0 | none/single/all 通过 |
| 已存在 small 拒绝覆盖 | 1 | 预期失败，未创建临时输出、未改写 small |
| 适配前 full `--estimate-only` | 0 | 五源 53,645 条，逻辑上界 59.30 GiB |
| `git diff --check` | 0 | 通过 |

Sen12 适配时的历史六源临时验收：

| 检查 | Exit | 结果 |
| --- | ---: | --- |
| 六源 `/tmp` small 构建 | 0 | 192 条、23 shard，未写入 `../benchmark` |
| 六源 small deep validator | 0 | 192/192，无错误和警告 |
| 六源 raw/z-score smoke | 0 | 3/4/10/12 通道及 none/single/all 通过 |
| Sen12 模态组合 | 0 | DEM+升轨、DEM+降轨、DEM+升轨+降轨均覆盖 |
| 六源 index SHA-256 | 0 | `87fc4813b4d4d4eeeb74a6920f3c9306c81e0d4450353c46b20dcb87adb0b4c8` |
| 六源 full `--estimate-only` | 0 | 60,316 条，逻辑上界 82.68 GiB |

指定权重现场检查：

```text
models_zoo/ConvNeXt/convnext_small-0c510722.pth
```

- exists: `true`
- backbone_sha256:
  `0c510722adfd92966a2bd72b92f785ca05966bbac03cafe2f7a90b1f54bfab9a`
- torchvision_strict_load: `pass`
- v5_real_gpu_smoke: `not_run`
- v4_proposed_dropout_peak_cuda_memory: `3.8086109161376953 GiB`
- v4_small_overfit_result: `failed_capacity_thresholds`
- v4_modality_dropout_300_step_result: `engineering_pass`
- v5_small_overfit_result: `not_run`
- v5_uniform_300_step_result: `not_run`
- v5_balanced_300_step_result: `not_run`

因此当前状态是“v5 六源 SAR、fractional validity、完整四尺度 MSPA/FRM/FFM、
DELIVER FFM 语义、训练优化、CPU/真实数据合同、官方权重与 stem 等价通过；
v5 GPU gate 待补齐”，不是 Phase 2 全部训练验收完成。既有 v4 300-step 结果仅作
历史基线，不替代 v5 验收；其原 Shell exit code未单独捕获，因此只按完整产物和
报告内 acceptance 记录。

## Full 空间估计

- full_sample_count: `60316`
- full_split_counts: `train=42014, val=13134, test=5168`
- logical_uncompressed_upper_bound: `82.68 GiB`
- extrapolated_hdf5_data_size: `41.69 GiB`
- recommended_free_space_with_staging_margin: `91.8 GiB`
- available_space_at_estimate_time: `5.4 TiB`

物理空间估计按六源临时 small 中各源实际压缩率分别外推，不是保证值。

## 未运行

- full Benchmark 构建、deep 验证和 smoke
- v5 六 variant 真实 small batch=8 GPU smoke 和 `<23 GiB` 显存验收
- v5 当前 Benchmark 全部 204 条 train、最多 1000 steps 的 proposed 容量过拟合
- v5 uniform 与 balanced-target 两次同 seed 300-step proposed 对照
- optical-only 短训练、真实 val 评价和正式 checkpoint 重载
- 训练后 Landslide4Sense/multimodal-landslide/Sen12Landslides 正式推理导出
- full 上的 50,000-step 正式多模态训练
- VLM、RAG 或端到端集成
- 数据、模型或依赖下载
- commit 或 push

## 已知限制

- LandslideBench_agent 有 311 个已批准的跨 split `location_key`；该源不满足严格
  group isolation，full validator 将其作为已知警告。
- LMHLD 和 Landslide4Sense 缺少可靠地理 parent/group，明确记录
  `group_status=unknown`，未伪造空间关系。
- multimodal InSAR 保留 encoded 数值和 validity，不推断其未确认物理单位。
- full 物理空间与耗时只能在项目负责人实际运行后确认。
- 历史会话曾无法访问 CUDA/NVML，但当前负责人终端已完成 4090 训练；尚未重新运行
  v5 六 variant smoke；本次 Codex 执行环境 `torch.cuda.is_available()==False`，
  因此 3.81 GiB 只代表 v4 proposed 300-step 训练，不代表 v5 或六消融峰值。
- v5 推理集成测试使用随机初始化 backbone 验证三源导出；官方权重仅完成严格加载和
  stem 等价。两者都不代表训练后的分割质量。
- v5 完整模型在当前六源 registry 下为 157,836,705 参数；v4 300-step proposed 的
  batch=8 峰值为 3.81 GiB，不能代替 v5 六 variant smoke 与 overfit 显存测量。
- v4 300-step EMA 工程阈值通过，但 val Dice/FPR 不构成精度验收；v4 1000-step
  capacity overfit 已失败。v5 对应训练尚未运行，不能声称修正已提升精度。
- `full_proposed_dropout.json` 的 50,000-step 日程是供负责人确认的单卡正式配置；
  full 存在并通过 validator、small GPU 验收通过前不得运行。

## 下一步

Sen12 升/降轨 SAR registry、fractional validity、DELIVER-style 四尺度渐进融合、
FFM 语义校正和 v5 训练合同已经完成。保留 v4 300/1000-step 产物，不重复覆盖；
按以下顺序在新的 v5 目录补齐 GPU gate：

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

small 验收标准：

1. 六 variant 均 exit 0，batch=8 峰值显存各自小于 23 GiB；
2. 容量过拟合 loss 下降至少 90%、micro Dice ≥0.95、
   positive-only Dice ≥0.90、空 mask FPR=0、空样本平均概率 ≤0.01；
3. 五个辅助 adapter、共享 MSPA encoder、四层 FRM/FFM 和 quality selector 均有非零梯度及更新；
4. 两次 300-step 均要求工程 acceptance、有效辅助曝光比例 ≥40%、实际 2400
   样本和 checkpoint 重载差异 ≤`1e-6`；balanced 仅在 FPR 至少下降 0.10 且
   positive Dice 下降不超过 0.01，或 overall Dice 至少提高 0.01 且其他指标不退化时采用；
5. optical、cmnext、quality、proposed 完成同配置 val 消融；checkpoint 三项输出和
   四尺度权重图重载差异均 ≤`1e-6`；
6. 三个指定真实来源的全部输出可以重载。

这些验收完成后仍先停止，由项目负责人决定是否构建/验收 full。不得进入 Region
Grounding、VLM Description 或 RAG。
