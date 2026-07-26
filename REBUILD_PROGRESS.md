# REBUILD_PROGRESS

## 当前状态

- program: `OA_AUXSEG_VLM_REBUILD`
- phase: `2`
- phase_name: `OA_AUXSEG_MULTIMODAL`
- phase_status: `phase2_v4_full_deliver_cpu_contract_complete_gpu_blocked`
- execution_date: `2026-07-26`
- branch: `main`
- implementation_baseline_head: `0a9c50d9abc4208e0aaa44218e857d60bd3821f4`
- benchmark_schema: `oa_auxseg_hdf5_v1`
- benchmark_sen12_adaptation_date: `2026-07-25`
- benchmark_small_refresh_date: `2026-07-26`
- phase2_path_migration_date: `2026-07-26`
- phase2_sar_registry_date: `2026-07-26`
- phase2_v4_full_deliver_fusion_date: `2026-07-26`
- model_schema: `oa_auxseg_model_v4`
- checkpoint_schema: `oa_auxseg_checkpoint_v4`
- runtime_config_schema: `oa_auxseg_runtime_config_v3`
- inference_schema: `oa_auxseg_inference_v4`
- benchmark_small_built: `true`
- benchmark_full_built: `false`
- model_implemented: `true`
- trainer_implemented: `true`
- evaluator_implemented: `true`
- inference_implemented: `true`
- training_run: `false`
- cpu_trainer_smoke_run: `true`
- v4_synthetic_cpu_optimizer_smoke_run: `true`
- v4_real_small_batch8_smoke_run: `false`
- official_small_training_run: `false`
- gpu_run: `false`
- download_run: `false`
- commit_performed: `false`
- push_performed: `false`

## 已完成

- OA-AuxSeg 模型相关路径从 Phase 1 完整迁移到 Phase 2；`oa_groundrag` 保留为项目级
  Python 包名，不保留旧 `phase1` 模型 alias 或兼容包装。
- 建立 `oa_groundrag/phase2/` 完整多模态 OA-AuxSeg：
  - 严格 registry、batch、运行配置、输出和 checkpoint schema；
  - 为 3/4/10/12 通道签名建立直接 `C→96, kernel=4, stride=4` 光学 stem，
    保留 Sentinel-2 B01–B12、Sen12 B02–B12 和 NIR；
  - 已审计 RGB/B04-B03-B02 位置复制官方 ConvNeXt stem，额外波段权重从零初始化并
    立即可训练；3 通道 stem 与官方输出逐元素一致；
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
  - FRM 校正后的光学和辅助特征分别传播到下一 stage，无辅助或零覆盖时硬旁路；
  - balanced cardinality subset sampling 与 `p=0.2` modality dropout；
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
  `128 + 128 + 8 + 6 = 270`；模型、checkpoint 与推理 schema 升级到 v4，
  runtime config 升级到 v3，不提供旧 v3 checkpoint 兼容层。
- 同一训练器实现六个 variant：`optical_only`、`direct_concat`、
  `mean_auxiliary_fusion`、`cmnext_injection`、`injection_quality`、
  `proposed_dropout`。
- 建立 `scripts/phase2_oa_auxseg/run_oa_auxseg.py` 单一
  `train/evaluate/infer/smoke/overfit` CLI 和严格 JSON 配置。
- 官方 ConvNeXt-Small state_dict、SHA-256 和 3 通道 signature stem 等价已在 CPU
  严格验证；source 仅用于报告，没有进入模型。
- 随机初始化 v4 checkpoint 已在真实 Landslide4Sense、multimodal-landslide 和
  Sen12Landslides val 样本上验证原子推理、拒绝覆盖、四尺度权重图和 270 维区域特征重载；
  这不等同于训练结果。
- 建立 `scripts/phase1_benchmark_build/`：
  - 公共 schema、原子 I/O、重采样、hash、Dataset 和 collate；
  - GDCLD、LMHLD、LandslideBench_agent、Landslide4Sense、multimodal、
    Sen12Landslides 六个显式源适配器；
  - small/full builder；
  - 独立 validator、summarizer 和 DataLoader smoke；
  - small/full shell 入口。
- 更新 `pyproject.toml`，声明 NumPy、h5py、SciPy、PyTorch 和 torchvision
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
| Phase 2 `unittest` | 0 | 23/23 v4 多模态合同测试通过 |
| Phase 2 测试资源 | 0 | elapsed 约 28 s，peak RSS 4,373,128 KiB；含多模型 checkpoint 重载，不是 GPU 显存 |
| 当前六源 small Phase 2 registry | 0 | train=204、3/4/10/12 通道及五种辅助模态全部注册 |
| 六 variant 合成 forward/backward/step | 0 | 3/4/10/12 通道与五辅助模态全部完成，loss/gradient 有限 |
| 官方 3 通道 stem 等价 | 0 | signature stem 与 torchvision 官方 stem 逐元素相同 |
| 多光谱额外波段梯度 | 0 | Sentinel-2 非 RGB 初始化通道首次 backward 梯度非零 |
| 官方 ConvNeXt-Small 严格加载 | 0 | `strict=True`，SHA-256 `0c510722...bfab9a` |
| CUDA/NVML probe | 255 | `torch.cuda.is_available() == False`，操作系统阻断 NVML |
| SAR 稀疏/顺序/invalid/zero-coverage | 0 | 升轨、降轨、双轨身份及退化、不变性合同通过 |
| fractional validity | 0 | 1/64 coverage、部分通道、全无效及 area 下采样通过 |
| 空间选择与渐进传播 | 0 | masked max、空间 softmax/null 和 stride 4 改变 stride 8 输入通过 |
| 首次辅助梯度 | 0 | 五 adapter、全部 36 个 MSPA block、四层 FRM/FFM、四个 quality selector 非零 |
| 无辅助硬旁路 | 0 | logits/probability、`[B,6]` 摘要和四尺度图均与 optical-only 逐元素相同 |
| checkpoint v4 严格重载 | 0 | logits/probability/weights/四尺度图一致，旧 v3 明确拒绝 |
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
- real_gpu_smoke: `not_run`
- peak_cuda_memory: `not_measured`
- small_overfit_result: `not_run`
- modality_dropout_300_step_result: `not_run`

因此当前状态是“六源 SAR、fractional validity、完整四尺度 MSPA/FRM/FFM、
CPU/真实数据合同、官方权重与 stem 等价通过，GPU 训练验收受 CUDA/NVML 访问阻塞”，
不是 Phase 2 训练验收完成。

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
- 六 variant 真实 small batch=8 GPU smoke 和 `<23 GiB` 显存验收
- 当前 Benchmark 全部 204 条 train、最多 1000 steps 的 proposed 容量过拟合
- proposed balanced subset + `p=0.2` dropout 的 300-step 短训练
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
- 当前执行环境 `torch.cuda.is_available() == False` 且 NVML 被操作系统阻断，
  CPU RSS 不能替代 GPU 峰值显存。
- v4 推理集成测试使用随机初始化 backbone 验证三源导出；官方权重仅完成严格加载和
  stem 等价。两者都不代表训练后的分割质量。
- v4 完整模型在当前六源 registry 下约 157.92M 参数；24 GiB 4090 上的真实
  batch=8 峰值尚未测量，若超过 23 GiB 只报告实际值，不自动缩维或降低物理 batch。
- GPU 显存、1000-step 过拟合阈值和 300-step EMA 阈值均没有结果。
- `full_proposed_dropout.json` 的 50,000-step 日程是供负责人确认的单卡正式配置；
  full 存在并通过 validator、small GPU 验收通过前不得运行。

## 下一步

Sen12 升/降轨 SAR registry、fractional validity、DELIVER-style 四尺度渐进融合和
270 维 region feature 合同已经完成。CUDA 可用后严格按以下顺序人工运行：

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
  --config configs/phase2_oa_auxseg/small_optical_only.json
```

small 验收标准：

1. 六 variant 均 exit 0，batch=8 峰值显存各自小于 23 GiB；
2. 容量过拟合 loss 下降至少 90%、micro Dice ≥0.95、
   positive-only Dice ≥0.90、空 mask FPR=0、空样本平均概率 ≤0.01；
3. 五个辅助 adapter、共享 MSPA encoder、四层 FRM/FFM 和 quality selector 均有非零梯度及更新；
4. 300-step EMA 下降至少 50%，实际观察 none/single/all-or-multi；
5. optical、cmnext、quality、proposed 完成同配置 val 消融；checkpoint 三项输出和
   四尺度权重图重载差异均 ≤`1e-6`；
6. 三个指定真实来源的全部输出可以重载。

这些验收完成后仍先停止，由项目负责人决定是否构建/验收 full。不得进入 Region
Grounding、VLM Description 或 RAG。
