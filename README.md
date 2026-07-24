# OA-AuxSeg + VLM

本项目研究光学锚定、任意辅助模态增强的滑坡分割，以及基于分割区域证据的视觉语言理解。

当前只完成阶段 1B 的统一 Benchmark 数据管线和 small Benchmark 验收。尚未实现分割模型、
Trainer、Evaluator、VLM 或 RAG，也未运行 full Benchmark。

详细设计见
[`光学锚定任意辅助模态滑坡分割与VLM区域理解_算法构建方案.md`](docs/光学锚定任意辅助模态滑坡分割与VLM区域理解_算法构建方案.md)，
活动进度见 [`REBUILD_PROGRESS.md`](REBUILD_PROGRESS.md)。

## 环境

当前已验证环境：

```bash
cd /home/yukun80/codes/paper7_VLM
conda activate qwen3vl
python --version
```

实际验证版本为 Python 3.11.15、NumPy 2.1.2、h5py 3.16.0 和
PyTorch 2.8.0+cu128。本阶段不需要 GPU。

## 阶段 1B 程序

```text
scripts/phase1_benchmark_build/
├── benchmark_common.py
├── benchmark_sources.py
├── 1_1_build_benchmark.py
├── 1_2_validate_benchmark.py
├── 1_3_summarize_benchmark.py
├── 1_4_smoke_dataloader.py
├── run_build_small.sh
└── run_build_full.sh
```

- builder、validator、summarizer 和 DataLoader smoke 相互独立；
- `../datasets` 始终只读；
- 输出目标存在时立即停止，不覆盖；
- small 和 full 使用同一 `oa_auxseg_hdf5_v1` schema；
- full 入口只供项目负责人手动运行。

## Benchmark 输出

默认输出：

```text
/home/yukun80/codes/benchmark/oa_auxseg_hdf5_v1/
├── small/
└── full/       # 当前不存在
```

每个模式内部结构：

```text
{mode}/
├── manifest.json
├── build_config.json
├── index.jsonl
├── source_statistics.json
├── SHA256SUMS.jsonl
└── data/{source}/{split}/shard-*.h5
```

分片按 source、split 和模态签名隔离。连续影像为 float32；mask、pixel validity 和
channel validity 为 uint8。不存在 `mask_validity`；源标签无效位置直接置为背景 0。

单条训练样本提供：

- `optical: [C,H,W]`，保留各源真实光学通道；
- `mask: [1,H,W]`；
- 可选辅助模态映射；
- image/modality pixel validity 与 channel validity；
- source、split、通道名、原始尺寸、resize 参数、foreground ratio 和 provenance。

DataLoader 形成 B 条样本后，mask 为 `[B,1,H,W]`。可变通道 optical 保持 tensor
列表；辅助模态按名称和样本下标打包，不用全零张量伪造缺失模态。source 仅存在于
metadata，不作为模型输入。

无效、NaN、Inf 和 nodata 影像像素先直接置 0；连续影像使用双线性插值，pixel validity
使用最近邻插值，resize 后按 validity 再次清零。mask 使用最近邻并严格保持 `{0,1}`。

## 数据源合同

| source | 光学通道 | 辅助模态 | full 记录 |
| --- | --- | --- | ---: |
| GDCLD | Red, Green, Blue | 无 | 13,447 |
| LMHLD | Blue, Green, Red, NIR | 无 | 28,185 |
| LandslideBench_agent | R, G, B | 无 | 2,130 |
| Landslide4Sense | Sentinel-2 B01–B12 | slope, DEM | 3,799 |
| multimodal-landslide | Red, Green, Blue | DEM, encoded InSAR | 6,084 |
| 合计 |  |  | 53,645 |

GDCLD、LMHLD、LandslideBench_agent 和 multimodal 保留源 split。Landslide4Sense
没有源 split，使用固定 `split_seed=20260724` 在 positive/background 内确定性分配
80/10/10，得到 train/val/test = 3,039/380/380。

LandslideBench_agent 的 311 个 `location_key` 已知跨 split。按项目负责人要求，全部
2,130 条仍保留源 split；manifest 记录该例外，严格 group-isolation 验收不适用于该源。

## 已构建的 small

当前 small 参数：

```text
patch_size=224
small_per_source=32
seed=20260724
split_seed=20260724
shard_target_mib=512
```

实际结果：

| source | train | val | test | positive | background | 合计 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gdcld | 7 | 13 | 12 | 20 | 12 | 32 |
| lmhld | 12 | 10 | 10 | 16 | 16 | 32 |
| landslidebench_agent | 12 | 10 | 10 | 16 | 16 | 32 |
| landslide4sense | 12 | 10 | 10 | 16 | 16 | 32 |
| multimodal_landslide | 11 | 21 | 0 | 22 | 10 | 32 |
| 合计 | 54 | 64 | 42 | 90 | 70 | 160 |

- shard：14 个；
- 磁盘占用：113,832,170 bytes，约 109 MiB；
- index SHA-256：
  `822d7b361b9e05a4b8b5d47beeea752d51dfbd056de6a7216004b903aebf3fe1`；
- deep validator：160/160 通过，无错误、无 small 内跨 split group 警告；
- DataLoader：raw 与 z-score 下的 `none/single/all` 均通过；
- 真实数据第二次临时构建的 index 与正式 small 逐字节一致。

输出已存在，以下构建入口现在会按设计拒绝覆盖：

```bash
bash scripts/phase1_benchmark_build/run_build_small.sh
```

可以随时只读重新验收：

```bash
python scripts/phase1_benchmark_build/1_2_validate_benchmark.py \
  --benchmark-root ../benchmark/oa_auxseg_hdf5_v1/small \
  --deep

python scripts/phase1_benchmark_build/1_3_summarize_benchmark.py \
  --benchmark-root ../benchmark/oa_auxseg_hdf5_v1/small

python scripts/phase1_benchmark_build/1_4_smoke_dataloader.py \
  --benchmark-root ../benchmark/oa_auxseg_hdf5_v1/small
```

## Full 空间估计与运行

只读估计命令：

```bash
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

估计结果：

- full：53,645 条；
- train/val/test：36,761/12,375/4,509；
- 未压缩逻辑上界：59.30 GiB；
- 按 small 的各源实际压缩率外推，HDF5 分片约 29.71 GiB；
- 为临时目录、最终发布和余量，建议至少预留 65.4 GiB 可用空间。

项目负责人运行 full 的精确入口：

```bash
cd /home/yukun80/codes/paper7_VLM
conda activate qwen3vl
bash scripts/phase1_benchmark_build/run_build_full.sh
```

full 验收标准：

1. 构建、deep validator、summarizer 和 DataLoader smoke 均 exit 0；
2. manifest `sample_count=53645`；
3. split 为 36,761/12,375/4,509；
4. 所有输出 patch 为 224×224，mask 仅含 0/1；
5. image/modality validity shape 正确，无效影像像素为 0；
6. RGB、NIR/多光谱、optical-only、单辅助和多辅助 batch 均可形成；
7. 除已批准的 311 个 LandslideBench location 例外外，不得出现新的已知 group 跨 split；
8. 不覆盖 small 或任何已有 full 输出。

## 测试

```bash
python -m unittest discover -s tests/phase1_benchmark_build -v
python -m py_compile scripts/phase1_benchmark_build/*.py \
  tests/phase1_benchmark_build/test_benchmark_pipeline.py
bash -n scripts/phase1_benchmark_build/run_build_small.sh \
  scripts/phase1_benchmark_build/run_build_full.sh
git diff --check
```

合成测试验证五源合同、small/full schema 一致、确定性、无效像素清零、二值 mask、
可变通道 collate、拒绝覆盖，以及 validator 对损坏索引、缺失 shard、错误 shape、
非法 validity 和非法 mask 的检测。

## 当前边界

- full 尚未运行；
- 未实现模型、Trainer、Evaluator、训练、正式评价、VLM 或 RAG；
- 未下载数据、模型或依赖；
- 未 commit、未 push；
- 下一步仅是项目负责人决定是否运行并验收 full，不进入光学分割模型阶段。
