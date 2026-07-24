# REBUILD_PROGRESS

## 当前状态

- program: `OA_AUXSEG_VLM_REBUILD`
- phase: `1B`
- phase_name: `UNIFIED_BENCHMARK_BUILD`
- phase_status: `implementation_complete_small_validated_full_not_run`
- execution_date: `2026-07-24`
- branch: `main`
- implementation_baseline_head: `534fbefc4819fda529975d045a3236eeb591ec84`
- benchmark_schema: `oa_auxseg_hdf5_v1`
- benchmark_small_built: `true`
- benchmark_full_built: `false`
- model_implemented: `false`
- training_run: `false`
- gpu_run: `false`
- download_run: `false`
- commit_performed: `false`
- push_performed: `false`

## 已完成

- 建立 `scripts/phase1_benchmark_build/`：
  - 公共 schema、原子 I/O、重采样、hash、Dataset 和 collate；
  - GDCLD、LMHLD、LandslideBench_agent、Landslide4Sense、multimodal 五个显式源适配器；
  - small/full builder；
  - 独立 validator、summarizer 和 DataLoader smoke；
  - small/full shell 入口。
- 更新 `pyproject.toml`，声明 NumPy、h5py 和 PyTorch 依赖。
- 建立 `tests/phase1_benchmark_build/` 五源合成 fixture。
- 采用分片 HDF5 加 JSONL 索引；目标存在时拒绝覆盖，临时目录完成后原子发布。
- 单样本 mask 为 `[1,H,W]`，batch mask 为 `[B,1,H,W]`。
- 不保存 `mask_validity`；标签无效位置直接置背景 0。
- 保留 image/modality pixel validity 与 channel validity；影像无效值直接置 0。
- 保留 Sentinel-2 B01–B12，不用全零张量伪造缺失辅助模态。
- Landslide4Sense 使用固定 seed 在 positive/background 内确定性分配 80/10/10。
- LandslideBench_agent 全部保留源 split，并在 manifest 中记录 311 个已批准跨 split
  `location_key` 例外。

## Small Benchmark

输出：

```text
/home/yukun80/codes/benchmark/oa_auxseg_hdf5_v1/small
```

构建参数：

```text
patch_size=224
small_per_source=32
seed=20260724
split_seed=20260724
shard_target_mib=512
```

实际统计：

| source | train | val | test | positive | background | 合计 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gdcld | 7 | 13 | 12 | 20 | 12 | 32 |
| lmhld | 12 | 10 | 10 | 16 | 16 | 32 |
| landslidebench_agent | 12 | 10 | 10 | 16 | 16 | 32 |
| landslide4sense | 12 | 10 | 10 | 16 | 16 | 32 |
| multimodal_landslide | 11 | 21 | 0 | 22 | 10 | 32 |
| 合计 | 54 | 64 | 42 | 90 | 70 | 160 |

- shard_count: `14`
- output_size_bytes: `113832170`
- index_sha256:
  `822d7b361b9e05a4b8b5d47beeea752d51dfbd056de6a7216004b903aebf3fe1`
- deep_validation: `pass`
- checked_samples: `160/160`
- validator_errors: `0`
- validator_warnings: `0`
- raw_dataloader_smoke: `pass`
- zscore_dataloader_smoke: `pass`
- real_small_repeat_index_byte_identical: `true`

## 实际检查

| 检查 | Exit | 结果 |
| --- | ---: | --- |
| Python `py_compile` | 0 | 所有阶段 1B 程序和测试通过 |
| shell `bash -n` | 0 | small/full 入口通过 |
| `unittest` | 0 | 6/6 通过 |
| 合成 small/full schema | 0 | 索引字段一致 |
| 合成重复 small | 0 | sample ID 顺序和 index 字节一致 |
| 损坏索引检测 | 0 | validator 正确失败 |
| 缺失 shard 检测 | 0 | validator 正确失败 |
| 错误 shape/validity 检测 | 0 | validator 正确失败 |
| 非法 mask 检测 | 0 | validator 正确失败 |
| 实际 small 构建 | 0 | 160 条、14 shard |
| 实际 small deep validator | 0 | 160/160，通过 |
| 实际 small summary | 0 | 五源、split、通道和模态统计完整 |
| DataLoader raw smoke | 0 | none/single/all 通过 |
| DataLoader z-score smoke | 0 | none/single/all 通过 |
| 第二次实际 small 临时构建 | 0 | index 逐字节一致，临时目录已清理 |
| 已存在 small 拒绝覆盖 | 1 | 预期失败，未创建临时输出、未改写 small |
| full `--estimate-only` | 0 | 53,645 条，逻辑上界 59.30 GiB |
| `git diff --check` | 0 | 通过 |

## Full 空间估计

- full_sample_count: `53645`
- full_split_counts: `train=36761, val=12375, test=4509`
- logical_uncompressed_upper_bound: `59.30 GiB`
- extrapolated_hdf5_data_size: `29.71 GiB`
- recommended_free_space_with_staging_margin: `65.4 GiB`
- available_space_at_estimate_time: `5.4 TiB`

物理空间估计按 small 中五个数据源各自的实际压缩率分别外推，不是保证值。

## 未运行

- full Benchmark 构建、deep 验证和 smoke
- 模型、Trainer、Evaluator
- GPU/CUDA、训练或正式评价
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

## 下一步

只读重新估算：

```bash
cd /home/yukun80/codes/paper7_VLM
conda activate qwen3vl
python scripts/phase1_benchmark_build/1_1_build_benchmark.py \
  --mode full \
  --datasets-root ../datasets \
  --output-root ../benchmark/oa_auxseg_hdf5_v1 \
  --patch-size 224 \
  --seed 20260724 \
  --split-seed 20260724 \
  --shard-target-mib 512 \
  --estimate-only
```

项目负责人确认后运行 full：

```bash
cd /home/yukun80/codes/paper7_VLM
conda activate qwen3vl
bash scripts/phase1_benchmark_build/run_build_full.sh
```

full 验收完成前停止，不进入光学分割模型阶段。
