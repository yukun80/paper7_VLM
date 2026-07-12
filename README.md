# Multi-Source Qwen-PSALM-Seg

本仓库构建多源遥感滑坡 instruction-segmentation benchmark，并实现面向单时相或同期多源证据的
**SANE -> QMEF -> PMRD** 研究模型。当前主协议为 benchmark v2；v1 benchmark、旧 checkpoint、
text cache v1 和 visual cache v2 均不兼容。

## 目录约定

默认从 `paper7_VLM` 根目录运行命令，并使用同级大数据目录：

```text
/home/yukun80/codes/
├── datasets/
├── benchmark/
└── paper7_VLM/
```

可用 `PAPER7_DATASETS_ROOT` 和 `PAPER7_BENCHMARK_ROOT` 覆盖物理位置。JSONL 始终保存
`datasets/...`、`benchmark/...` 逻辑路径，不写机器绑定的绝对路径。

推荐环境：

```bash
conda activate qwen3vl
export PYTHONPATH=SEG_Multi-Source_Landslides${PYTHONPATH:+:${PYTHONPATH}}
```

也可安装命令别名：

```bash
python -m pip install -e SEG_Multi-Source_Landslides
```

## 构建 Benchmark V2

构建 small：

```bash
SMALL_LIMIT=500 \
bash scripts/run_1_build_benchmark.sh small
bash scripts/run_2_build_instruction_dataset.sh small
```

`SMALL_LIMIT` 表示每个 `dataset_name + split` 的父样本上限，不是整个 split 的
总上限。instruction 构建还会从父样本派生 global、referring 和 no-target 任务，
因此 instruction 行数通常明显大于父样本数。

构建 full：

```bash
bash scripts/run_1_build_benchmark.sh full
bash scripts/run_2_build_instruction_dataset.sh full
```

输出分别位于同级：

```text
../benchmark/multisource_landslide_v2_small
../benchmark/multisource_landslide_v2_full
```

质量门要求 source、final、referring-target 和 instruction validation 的 `errors == []`。
v2 每个模态必须显式包含 `family`、`sensor`、`product_type`、band metadata、GSD、units、
signed、quality、结构化 normalization 和归一化前物化的 valid mask。

独立阶段入口：

```bash
python scripts/1-benchmark/1-1_scan_sources.py --datasets-root datasets --out-dir benchmark/multisource_landslide_v2_small
python scripts/1-benchmark/1-2_build_index.py --mode small --datasets-root datasets --out-dir benchmark/multisource_landslide_v2_small
python scripts/1-benchmark/1-3_validate_index.py --benchmark-dir benchmark/multisource_landslide_v2_small --stage source
python scripts/1-benchmark/1-4_preprocess_samples.py --benchmark-dir benchmark/multisource_landslide_v2_small --strategy materialize
python scripts/1-benchmark/1-3_validate_index.py --benchmark-dir benchmark/multisource_landslide_v2_small --stage final
python scripts/1-benchmark/1-5_build_splits.py --benchmark-dir benchmark/multisource_landslide_v2_small
python scripts/1-benchmark/1-6_build_referring_targets.py --benchmark-dir benchmark/multisource_landslide_v2_small
python scripts/1-benchmark/1-3_validate_index.py --benchmark-dir benchmark/multisource_landslide_v2_small --stage referring_target
python scripts/1-benchmark/1-7_summarize_benchmark.py --benchmark-dir benchmark/multisource_landslide_v2_small
python scripts/2-instruction/2-1_build_instruction_templates.py --benchmark-dir benchmark/multisource_landslide_v2_small
python scripts/2-instruction/2-2_apply_instruction_templates.py --benchmark-dir benchmark/multisource_landslide_v2_small
python scripts/2-instruction/2-3_validate_instruction_index.py --benchmark-dir benchmark/multisource_landslide_v2_small
```

## 模型 Preset

主 preset 由 `qpsalm_seg/presets.py` 定义：

| Preset | 作用 |
|---|---|
| `raw_sane_baseline` | 单 query、无语义 reliability/null gate 的均匀多源 SANE 基线 |
| `raw_sane_qmef` | 增加 null-aware、query-conditioned QMEF |
| `raw_sane_qmef_pmrd` | 增加 proposal set 与两轮 PMRD |
| `pretrained_sane_qmef_pmrd` | 使用 Qwen-ViT cache v3 的中间空间特征 |
| `qwen_psalm_full` | 在线 4-bit Qwen language decoder + QLoRA mask-query states |
| `qwen_mask_query_frozen` | 冻结 Qwen language decoder，仅训练软提示、SANE/QMEF/PMRD 的消融基线 |

正式 Qwen 路线固定使用离线视觉塔和在线语言 decoder。Qwen 不生成 bbox；它负责语义条件、
多视图证据 token、evidence anchors 和 mask-query hidden states。

## Smoke 回归

先重建 small v2，再运行：

```bash
bash SEG_Multi-Source_Landslides/scripts/run_qpsalm_smoke.sh
```

该入口使用 development-only `text_probe`，执行 5-step forward/backward、validation、
checkpoint reload 和可视化，不加载 Qwen 权重。

## 正式训练

small：

```bash
BENCHMARK_SIZE=small \
PRESET=qwen_psalm_full \
SEED=42 \
RUN_NAME=small_qwen_b6_bf16_nockpt \
RUN_CONTROL=--overwrite \
CACHE_CONTROL=reuse \
bash SEG_Multi-Source_Landslides/scripts/run_qpsalm_experiment.sh
```

full：

```bash
BENCHMARK_SIZE=full \
PRESET=qwen_psalm_full \
SEED=42 \
RUN_NAME=full_qwen_b6_bf16_nockpt \
RUN_CONTROL=--overwrite \
CACHE_CONTROL=reuse \
bash SEG_Multi-Source_Landslides/scripts/run_qpsalm_experiment.sh
```

24GB 单卡参数直接定义在 small/full YAML：BF16、`batch_size=4`、
`grad_accum_steps=1`、`query_chunk_size=16`，并关闭 Qwen gradient checkpoint。
脚本不再接受隐藏的精度、batch或checkpoint覆盖。正式训练先进行 450-step decoder warmup，
随后以 `0.2 × lr` 启用最后四层 QLoRA。首次运行使用 YAML 中的 batch 规模执行代表性反向门禁，
峰值上限为22.5 GiB；可用
`MEMORY_GATE=0` 显式跳过，但正式实验不建议关闭。

周期验证使用固定的 parent-aware monitor subset：small 为 512 条，full 为 1024 条；
训练结束后脚本再用 best checkpoint 完整评估 val。

## Vision Cache V3

单独准备 small cache：

```bash
python -m qpsalm_seg.cli.cache_qwen_vision_features \
  --config SEG_Multi-Source_Landslides/configs/qpsalm_v2_small.yaml \
  --preset qwen_psalm_full \
  --output-dir outputs/qpsalm_v2/cache/small_qwen_psalm_full_qwen_vision_v3 \
  --backend qwen --device cuda --overwrite
```

cache 按 parent sample 和 view 分片，默认将物理视图渲染为 256，并以 `16/8/6/4`
保存 ViT layers 5/11/17/23
空间特征，使浅层保留更多边界、深层压缩语义上下文；同时保存原生 view tokens、
grid/padding transform、content hash、renderer/model/processor/prompt revision、pooling method、
full-subset signature 和 preset/尺寸 input protocol。训练时按
`ActiveModalitySubset` 动态选择，多个 instruction 不重复编码同一父图像。
cache 构建采用流式 parent 编码：Qwen 视觉塔只加载一次，内存中最多保留
`--shard-size` 个已编码父样本，写出 shard 后立即释放。`manifest.json` 中的
`peak_buffer_records` 可用于核对实际缓存上界；full 数据不再先把所有渲染视图驻留内存。
manifest 同时绑定 train/val/test instruction index 的 SHA-256；benchmark 或 instruction
索引重建后，`--verify-only` 会拒绝旧 cache。`RUN_CONTROL` 只控制训练目录；视觉 cache
由 `CACHE_CONTROL=reuse|verify|overwrite` 独立控制，默认校验并复用。
本地 Qwen revision 对配置和全部权重文件计算完整 SHA-256；每个进程只计算一次，因此首次
启动会多一次顺序读盘，但不会用仅哈希 `config.json` 的弱 revision 误判权重一致。

开发结构测试可使用：

```bash
python -m qpsalm_seg.cli.cache_qwen_vision_features \
  --config SEG_Multi-Source_Landslides/configs/qpsalm_v2_smoke.yaml \
  --output-dir /tmp/qpsalm_vision_v3_smoke --backend hash-smoke --max-samples 4 --overwrite
```

校验已有 cache 的 renderer/prompt/pooling/revision/subset 协议：

```bash
python -m qpsalm_seg.cli.cache_qwen_vision_features \
  --config SEG_Multi-Source_Landslides/configs/qpsalm_v2_small.yaml \
  --preset qwen_psalm_full \
  --output-dir outputs/qpsalm_v2/cache/small_qwen_psalm_full_qwen_vision_v3 --verify-only
```

## 独立训练与评估

```bash
python -m qpsalm_seg.cli.train \
  --config SEG_Multi-Source_Landslides/configs/qpsalm_v2_small.yaml \
  --preset qwen_psalm_full --device cuda \
  --vision-feature-cache outputs/qpsalm_v2/cache/small_qwen_psalm_full_qwen_vision_v3 \
  --output-dir outputs/qpsalm_v2/manual_run --skip-torch-preflight
```

验证集：

```bash
python -m qpsalm_seg.cli.eval \
  --config SEG_Multi-Source_Landslides/configs/qpsalm_v2_small.yaml \
  --preset qwen_psalm_full --checkpoint outputs/qpsalm_v2/manual_run/checkpoint_best.pt \
  --vision-feature-cache outputs/qpsalm_v2/cache/small_qwen_psalm_full_qwen_vision_v3 \
  --split val --device cuda --output-dir outputs/qpsalm_v2/manual_run/eval_val \
  --export-multimodal-overview --skip-torch-preflight
```

测试集需先生成 test 专用 cache：

```bash
python -m qpsalm_seg.cli.cache_qwen_vision_features \
  --config SEG_Multi-Source_Landslides/configs/qpsalm_v2_full.yaml \
  --preset qwen_psalm_full \
  --eval-index indexes/instruction_test.jsonl --eval-split test \
  --output-dir outputs/qpsalm_v2/cache/full_qwen_psalm_full_test_vision_v3 \
  --backend qwen --device cuda --overwrite
```

然后将 eval 命令改为 `--split test --vision-feature-cache outputs/qpsalm_v2/cache/full_qwen_psalm_full_test_vision_v3`。

## 消融与真实性测试

instruction 消融：

```bash
python -m qpsalm_seg.cli.eval ... --instruction-ablation shuffled
python -m qpsalm_seg.cli.eval ... --instruction-ablation fixed-generic
python -m qpsalm_seg.cli.eval ... --instruction-ablation no-semantic
```

`shuffled` instruction 同样要求至少两个不同 parent 和不同文本；不能构造有效反事实时会
直接报错。

视觉真实性消融：

```bash
python -m qpsalm_seg.cli.eval ... --visual-ablation shuffled
python -m qpsalm_seg.cli.eval ... --visual-ablation text-only
python -m qpsalm_seg.cli.eval ... --visual-ablation image-text-delta
python -m qpsalm_seg.cli.eval ... --visual-ablation remove:deformation
python -m qpsalm_seg.cli.eval ... --visual-ablation remove:sar
```

`visual_ablation` 只改变送入 Qwen language decoder 的语义 view tokens，不改变 SANE
读取的当前样本 Qwen-ViT 空间特征。因此这些实验衡量的是 Qwen 多视图 evidence 的作用，
不会同时替换 dense visual backbone。`image-text-delta` 使用完整图文上下文与 text-only
上下文的 post-context evidence anchor 差值进行 QMEF/verifier 消融，PMRD mask query 仍取
完整图文序列的 Qwen hidden states。
`shuffled` 要求 cache 中每种 raw modality 或 family 组合至少有两个不同 parent；否则评估
会明确报错，不会静默使用原图冒充 shuffle。

推荐使用单进程 suite，只加载一次 Qwen/checkpoint 并自动生成全部评估和证据报告：

```bash
PYTHONPATH=SEG_Multi-Source_Landslides \
python -m qpsalm_seg.cli.eval_ablation_suite \
  --config SEG_Multi-Source_Landslides/configs/qpsalm_v2_small.yaml \
  --preset qwen_psalm_full --checkpoint outputs/RUN/checkpoint_best.pt \
  --vision-feature-cache outputs/qpsalm_v2/cache/small_qwen_psalm_full_qwen_vision_v3 \
  --split val --device cuda \
  --visual-remove terrain --visual-remove sar --visual-remove deformation \
  --include-image-text-delta --min-delta 0 \
  --output-dir outputs/RUN/ablation_suite --overwrite-output --skip-torch-preflight
```

suite 依次切换 Dataset instruction 和 Qwen token-only visual evidence，SANE dense features
始终不变。每个条件仍生成标准 `eval_report.json/eval_manifest.json`，最后自动写
`ablation_evidence.json`；任一必需消融未出现性能退化时非零退出。

已有独立 eval 目录时，也可以单独生成严格成对证据报告：

```bash
PYTHONPATH=SEG_Multi-Source_Landslides \
python -m qpsalm_seg.cli.ablation_report \
  --normal outputs/ablations/normal \
  --instruction-shuffled outputs/ablations/instruction_shuffled \
  --instruction-fixed-generic outputs/ablations/instruction_fixed \
  --instruction-no-semantic outputs/ablations/instruction_no_semantic \
  --visual-shuffled outputs/ablations/visual_shuffled \
  --visual-text-only outputs/ablations/visual_text_only \
  --visual-remove terrain=outputs/ablations/remove_terrain \
  --visual-remove sar=outputs/ablations/remove_sar \
  --image-text-delta outputs/ablations/image_text_delta \
  --min-delta 0 --output outputs/ablations/ablation_evidence.json
```

汇总器要求所有目录包含 `eval_report.json` 和 `eval_manifest.json`，并严格检查 checkpoint、
step、split、preset 与 sample IDs 完全相同。Instruction 比较联合逐样本 proposal/final-mask
退化与 paired/no-target sensitivity；`remove:<family>` 只在确实包含该 family 的样本上比较。
normal 未优于任意必需消融时命令非零退出，不能据此声称模型使用了对应语义或视觉证据。

Qwen view token pooling 需要分别训练，可通过
`--qwen-view-pooling tokens|image-end|attention` 选择。`tokens` 是主路线；`image-end`
仅保留每个 view 的最后一个视觉 token；`attention` 使用可学习查询池化。该选项属于
checkpoint architecture protocol，不能在加载同一权重时临时切换。

报告包含 positive-only IoU/Dice、negative accuracy、empty false-positive rate、component
recall/precision、relevance AP/AUC、unmatched rejection、merge/duplicate/missed-component rate、
proposal-union Dice、同 parent paired target/prediction IoU、instruction contrast ratio 和
no-target rejection。proposal CSV、mask export 和可视化同时保留 verifier 实际选择的
`selected_proposal` 与由 GT assignment 得到的 `oracle_matched_proposal`；后者只用于测量
proposal capacity 与 selection gap，不是推理时可用的模型输出。

## 真实集成门槛

重建 small-v2 并生成正式 vision cache 后，在三 seed 实验前运行一次严格单卡检查：

```bash
PYTHONPATH=SEG_Multi-Source_Landslides \
python -m qpsalm_seg.cli.integration_check \
  --config SEG_Multi-Source_Landslides/configs/qpsalm_v2_small.yaml \
  --mode all --device cuda \
  --vision-feature-cache outputs/qpsalm_v2/cache/small_qwen_psalm_full_qwen_vision_v3 \
  --max-memory-gib 22.5 \
  --output outputs/qpsalm_v2/real_integration_report.json
```

`raw` 检查会从真实 train split 各选择 global、referring 和 no-target 样本并完成一次
optimizer step。`qwen` 检查只运行一个 batch：从同一空间桶、Qwen sequence-load 桶和 task
group 中选择六个不同 parent 的多源正样本，并交替使用 full/dropped evidence。该 batch 完成
一次 forward、backward 和 optimizer step；只有聚合 LoRA 梯度有限非零、LoRA 参数实际更新、
teacher consistency 生效且峰值 reserved memory 不超过 22.5 GiB 时才通过。单个样本 LoRA
梯度为零不作为失败条件。结果写入
`qpsalm_real_integration_v2` JSON，任一检查失败时命令非零退出。

需要定位 Qwen/PEFT 梯度链路时运行深度诊断；它额外执行 controller-only 两个优化步骤，
第一步检查 `lora_B`，第二步检查 `lora_A`；随后分别检查 student-only segmentation 和
full/dropped consistency 的 Qwen hidden、mask/coarse/refined query 梯度：

```bash
PYTHONPATH=SEG_Multi-Source_Landslides \
python -m qpsalm_seg.cli.integration_check \
  --config SEG_Multi-Source_Landslides/configs/qpsalm_v2_small.yaml \
  --mode qwen --qwen-check diagnostic --device cuda \
  --vision-feature-cache outputs/qpsalm_v2/cache/small_qwen_psalm_full_qwen_vision_v3 \
  --max-memory-gib 22.5 \
  --output outputs/qpsalm_v2/qwen_trainability_diagnostic.json
```

门禁通过后，先运行 5-step 阶段切换 smoke；它在 step 2 启用 QLoRA，并要求 trainer 同时
观测到非零 LoRA 梯度和真实参数更新。正式 YAML 仍保持 step 450：

```bash
PYTHONPATH=SEG_Multi-Source_Landslides \
python -m qpsalm_seg.cli.train \
  --config SEG_Multi-Source_Landslides/configs/qpsalm_v2_small.yaml \
  --preset qwen_psalm_full --device cuda \
  --vision-feature-cache outputs/qpsalm_v2/cache/small_qwen_psalm_full_qwen_vision_v3 \
  --qwen-lora-start-step 2 --max-steps 5 \
  --max-train-samples 24 --max-val-samples 12 --monitor-val-samples 12 \
  --num-workers 0 --val-interval 5 --save-interval 5 --num-visualizations 4 \
  --output-dir outputs/qpsalm_v2/qwen_stage_smoke \
  --overwrite-output --skip-torch-preflight
```

成功时终端会出现一次 `[QLORA]`，详细阶段证据写入 `stage_events.jsonl`，并生成
`checkpoint_best.pt`、`checkpoint_last.pt`、validation report 和可视化。

Qwen 主训练默认关闭 activation checkpoint，以增加激活显存换取更高吞吐；显存不足时可显式
传入 `--qwen-gradient-checkpointing reentrant`。运行时不会自动回退，实际模式会写入 resolved
config、checkpoint protocol、训练启动日志和 integration report。当前配置还会使用 SDPA、
序列负载分桶和 dropped-only teacher batch，减少 padding 与重复 teacher forward。

完成门禁后可用独立 20-step batch 4 运行检查稳定吞吐，并关闭周期验证与可视化：

```bash
PYTHONPATH=SEG_Multi-Source_Landslides python -m qpsalm_seg.cli.train \
  --config SEG_Multi-Source_Landslides/configs/qpsalm_v2_small.yaml \
  --preset qwen_psalm_full --device cuda --batch-size 4 --max-steps 20 \
  --qwen-lora-start-step 0 \
  --val-interval 20 --max-val-batches 1 --num-visualizations 0 \
  --vision-feature-cache outputs/qpsalm_v2/cache/small_qwen_psalm_full_qwen_vision_v3 \
  --output-dir outputs/qpsalm_v2/throughput_b4_nf4 --overwrite-output --skip-torch-preflight
```

`train_history.jsonl` 会记录 `samples_per_sec`、`qwen_tokens_per_sec`、峰值显存、Qwen padding
比例和 teacher 样本比例。冻结 BF16 Qwen 对照可在相同命令中增加 `--no-qwen-4bit`；只有吞吐
更高且峰值不超过 22.5 GiB 时才应修改正式 YAML。

```bash
PYTHONPATH=SEG_Multi-Source_Landslides python -m qpsalm_seg.cli.summarize_run \
  --run-dir outputs/qpsalm_v2/throughput_b6_nf4 --no-export-tables
PYTHONPATH=SEG_Multi-Source_Landslides python -m qpsalm_seg.cli.summarize_run \
  --run-dir outputs/qpsalm_v2/throughput_b8_nf4 --no-export-tables
```

终端 `train_performance.steady_state_last_window` 用于比较稳定吞吐，
`weighted_mean` 用于查看包含冷启动在内的整体效率。

## 数据与结果工具

```bash
python -m qpsalm_seg.cli.inspect_data --config SEG_Multi-Source_Landslides/configs/qpsalm_v2_small.yaml --split train
python -m qpsalm_seg.cli.cache_index --config SEG_Multi-Source_Landslides/configs/qpsalm_v2_small.yaml --output-dir outputs/qpsalm_v2/index_cache --split both --strategy round-robin-family
python -m qpsalm_seg.cli.summarize_run --run-dir outputs/qpsalm_v2/RUN --eval-dir outputs/qpsalm_v2/RUN/eval_val
python -m qpsalm_seg.cli.compare_runs --help
python -m qpsalm_seg.cli.diagnose_run --help
python -m qpsalm_seg.cli.recommend_threshold --help
python -m qpsalm_seg.cli.export_tables --help
```

三组固定 seed 的模块准入检查可重复传入成对 summary；只有至少 2/3 seed 的 candidate
pipeline ready，且 positive-only、instruction sensitivity 或 component-set 主指标超过
`--min-delta`，报告中的 `passed_2_of_3_gate` 才为 true：

```bash
python -m qpsalm_seg.cli.compare_runs \
  --baseline-summary outputs/base_s42/run_summary.json \
  --candidate-summary outputs/candidate_s42/run_summary.json \
  --baseline-summary outputs/base_s123/run_summary.json \
  --candidate-summary outputs/candidate_s123/run_summary.json \
  --baseline-summary outputs/base_s3407/run_summary.json \
  --candidate-summary outputs/candidate_s3407/run_summary.json \
  --min-delta 0 --output outputs/seed_gate.json
```

可编辑安装后的命令别名：

| 命令 | 模块 |
|---|---|
| `qpsalm-inspect-data` | `qpsalm_seg.cli.inspect_data` |
| `qpsalm-cache-index` | `qpsalm_seg.cli.cache_index` |
| `qpsalm-check-env` | `qpsalm_seg.cli.check_env` |
| `qpsalm-cache-qwen-vision-features` | `qpsalm_seg.cli.cache_qwen_vision_features` |
| `qpsalm-integration-check` | `qpsalm_seg.cli.integration_check` |
| `qpsalm-ablation-report` | `qpsalm_seg.cli.ablation_report` |
| `qpsalm-eval-ablation-suite` | `qpsalm_seg.cli.eval_ablation_suite` |
| `qpsalm-train` | `qpsalm_seg.cli.train` |
| `qpsalm-eval` | `qpsalm_seg.cli.eval` |
| `qpsalm-summarize-run` | `qpsalm_seg.cli.summarize_run` |
| `qpsalm-compare-runs` | `qpsalm_seg.cli.compare_runs` |
| `qpsalm-diagnose-run` | `qpsalm_seg.cli.diagnose_run` |
| `qpsalm-recommend-threshold` | `qpsalm_seg.cli.recommend_threshold` |
| `qpsalm-export-tables` | `qpsalm_seg.cli.export_tables` |

## 静态检查与单元测试

```bash
bash -n scripts/run_1_build_benchmark.sh scripts/run_2_build_instruction_dataset.sh \
  SEG_Multi-Source_Landslides/scripts/run_qpsalm_experiment.sh \
  SEG_Multi-Source_Landslides/scripts/run_qpsalm_smoke.sh

python -B -m py_compile $(find scripts/1-benchmark scripts/2-instruction \
  SEG_Multi-Source_Landslides/qpsalm_seg -name '*.py' -type f)

PYTHONPATH=SEG_Multi-Source_Landslides python -B -m unittest \
  SEG_Multi-Source_Landslides/tests/test_benchmark_v2.py \
  SEG_Multi-Source_Landslides/tests/test_refactor_core.py \
  SEG_Multi-Source_Landslides/tests/test_renderer.py \
  SEG_Multi-Source_Landslides/tests/test_v2_integration.py -v
```

算法设计与阶段门见 [docs/opt_refactor_algo.md](docs/opt_refactor_algo.md)。
