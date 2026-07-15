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

## 构建 Description Benchmark M0/M1

`docs/benchmark_GAR.md` 的下一阶段先构建遥感全图描述与区域对齐 benchmark。
脚本只从原始数据中选择所需 parent，并将对应图片原样复制到 benchmark；不会修改
现有 Landslide Benchmark V2。默认读取：

```text
../datasets/MMRS-1M
../datasets/RSGPT/dataset/RSICap
../datasets/RSGPT/dataset/RSIEval
```

同时兼容旧的 `external/RSGPT/dataset` 布局；可用 `PAPER7_RSGPT_DATA_ROOT` 覆盖。
构建并验证 small：

```bash
PYTHON_BIN=/home/yukun80/miniconda3/envs/qwen3vl/bin/python \
bash scripts/run_3_build_description_benchmark.sh small
```

如需重建已有派生产物：

```bash
RUN_CONTROL=--overwrite \
PYTHON_BIN=/home/yukun80/miniconda3/envs/qwen3vl/bin/python \
bash scripts/run_3_build_description_benchmark.sh small
```

默认 small 保留全部 RSICap/RSIEval，分层选择 12,000 个 MMRS Caption canonical parent 和
5,000 个 DIOR-RSVG parent。dHash 只用于召回候选；统一 RGB 64x64 后 MAE 不超过 3.0 的
同图重编码会在 split 前合并为一个 canonical parent，并合并多来源 caption。输出位于：

```text
../benchmark/qpsalm_description_v2_small/
├── data/
├── indexes/
├── manifests/
└── reports/
```

其中 `3-4_deduplicate_and_split.py` 只冻结 selected source records，
`3-5_materialize_description_images.py` 才复制图片并发布最终索引；验证与汇总依次为
`3-6_validate_description_benchmark.py` 和 `3-7_summarize_description_benchmark.py`。
可用 `DESCRIPTION_COPY_WORKERS=8` 调整本地复制并行度；研究复现时应保持默认
`DESCRIPTION_PERCEPTUAL_MAE_THRESHOLD=3.0`，修改门槛必须形成新的 split protocol。
canonical 合并后的每条 caption answer 都保留 source answer index、原文 hash 和来源记录；
`verified_perceptual_duplicates.jsonl` 同时记录 canonical 选择与每个成员的 split action。

在 `/tmp` 手动执行小规模闭环而不覆盖正式 benchmark：

```bash
PAPER7_BENCHMARK_ROOT=/tmp \
MAX_SAMPLES=8 \
RUN_CONTROL=--overwrite \
DESCRIPTION_COPY_WORKERS=4 \
PYTHON_BIN=/home/yukun80/miniconda3/envs/qwen3vl/bin/python \
bash scripts/run_3_build_description_benchmark.sh small
```

最终 `all/train/dev/test` 索引只引用 `data/` 中的图片；`*_source.jsonl` 和
provenance 保留 `datasets/...` 路径用于审计。质量门为
`reports/validation_report.json` 中 `errors == []`，并要求
`verified_perceptual_duplicate_cross_split_groups == 0`。训练应读取
`indexes/train_eligible.jsonl`，完整 `train.jsonl` 仍保留零权重审计答案。DIOR 只提供
box-to-phrase 和 phrase-to-candidate-region 对齐监督，不作为详细区域 caption 真值。
物化图片仅用于本地研究；未经各源数据许可审核不得公开重新分发。
本阶段通过后再进入 Landslide Bridge、MGRR 和描述 Adapter 训练。

## Landslide Bridge Pilot

M2 从 `multisource_landslide_v2_<mode>` 构建区域清单、三级多源证据、规则候选文本和
双人专家审核包。`pseudo_instance_component` 仅表示 8 邻域伪实例组件，不等同人工实例。
正式 Pilot 必须精确包含 300 个不同 parent，并满足 train/val/test = 180/60/60；候选不足会
作为验证错误报告。使用 `MAX_SAMPLES` 的缩小构建只用于 smoke，不能进入专家 gate。
准备阶段不会生成专家标签：

```bash
BRIDGE_STAGE=prepare \
BRIDGE_PILOT_PARENTS=300 \
RUN_CONTROL=--overwrite \
PYTHON_BIN=/home/yukun80/miniconda3/envs/qwen3vl/bin/python \
bash scripts/run_4_build_landslide_bridge.sh small
```

审核者分别填写 `review_package/reviewer_1_template.*` 和
`review_package/reviewer_2_template.*` 的副本。`decision` 只允许 `accept/revise/reject`；
双人分歧必须提供仲裁文件。Pilot 分析后，还需人工复制并填写
`manifests/evaluation_gate_manifest.template.json`，将状态显式冻结，再执行合并：
模板已绑定本轮 Pilot parent、review selection 和 candidate index 的 SHA-256；不得从其他
run 复制 gate，也不得修改 `bindings`。

```bash
BRIDGE_STAGE=merge \
REVIEWER_1=/path/to/reviewer_1_completed.jsonl \
REVIEWER_2=/path/to/reviewer_2_completed.jsonl \
ARBITRATION_FILE=/path/to/arbitration_completed.jsonl \
EVALUATION_GATE=/path/to/evaluation_gate_frozen.json \
RUN_CONTROL=--overwrite \
PYTHON_BIN=/home/yukun80/miniconda3/envs/qwen3vl/bin/python \
bash scripts/run_4_build_landslide_bridge.sh small
```

正式 prepare 验收状态为 `awaiting_expert_review`、`pilot_protocol_complete == true` 且
`errors == []`；带 `MAX_SAMPLES` 的构建状态为 `smoke_only`。只有仲裁清零、三个 split
均有审核通过记录且 gate 已冻结时，状态才会变为 `expert_pilot_frozen`。Bridge 只引用
Landslide V2 已物化模态，不读取原始 `datasets/`，也不重复复制多源数组。

M2 prepare 有效后，可构建独立的 task-neutral Description Vision Cache v1。它按 parent
缓存，不包含 instruction、region geometry 或 segmentation state；现有 segmentation
Vision Cache v3 只读复用，不会被覆盖：

```bash
PYTHONPATH=SEG_Multi-Source_Landslides \
python -m qpsalm_seg.cli.cache_description_vision_features \
  --config SEG_Multi-Source_Landslides/configs/qpsalm_v2_small.yaml \
  --description-benchmark benchmark/qpsalm_description_v2_small \
  --bridge-benchmark benchmark/landslide_region_description_v1_small \
  --segmentation-vision-cache outputs/qpsalm_v2/cache/small_qwen_psalm_full_qwen_vision_v3 \
  --output-dir outputs/qpsalm_description/cache/small_vision_v1 \
  --device cuda --backend qwen --overwrite
```

## Segmentation-Grounded Description M3-M7

以下命令均从仓库根目录手动运行。`--resume` 只用于同一 stage 中断续训，会恢复优化器和
scheduler；`--initialize-from` 用于进入下一个 stage，只加载模型权重并重置该 stage 的优化状态。
当消融实验显式改变 `region_encoder` 时，初始化器只重新初始化该 encoder，并严格迁移其余
共享参数；迁移报告会记录跳过的 region keys。其他架构差异仍会直接失败。
不得用 `--resume` 跨越 D0-D4。M2 专家数据未冻结前，只能运行 M3、D-1 和 D3a 工程验证，
不能把 D3b、D4、M7 的输出作为正式科学结果。

当前 M4 使用 `qpsalm_mgrr_v2_multiscale_grid_replay`：四尺度 RoI 网格为
`7×7/7×7/4×4/2×2`，由两个 learnable queries 压缩，并单独记录 component coverage 与
residual ratio。该协议改变了 MGRR 参数形状和描述序列协议；此前生成的实验性 segdesc
checkpoint 不兼容，需要从分割 checkpoint 重新开始 D-1/D0，而 segmentation checkpoint 和
两类 vision cache 无需因此重建。

先发布只含 component 引用、hash 和精确 JSONL 行号的统一索引。它不复制 M1/M2 图片或 mask，
并绑定三个 component validation report。Bridge 尚在专家审核时只发布自动描述；目录中残留的
expert index 或旧 gate 会被明确忽略。只有冻结后的 v2 evaluation gate 才能启用专家监督：

```bash
RUN_CONTROL=--overwrite \
PYTHON_BIN=/home/yukun80/miniconda3/envs/qwen3vl/bin/python \
bash scripts/run_5_build_segdesc_dataset.sh small
```

### D-1 基线与过拟合

原生 Qwen3-VL 全图描述 zero-shot 基线：

```bash
PYTHONPATH=SEG_Multi-Source_Landslides \
python -m qpsalm_seg.cli.eval_description_zero_shot \
  --model models_zoo/Qwen3-VL-2B-Instruct \
  --benchmark benchmark/qpsalm_description_v2_small \
  --split dev --device cuda --max-samples 64 \
  --output-dir outputs/qpsalm_description/d_minus_1_zero_shot_dev \
  --overwrite-output
```

32-64 条 Bridge 样本过拟合，用于检查 `desc_adapter`、MGRR、causal labels 和 checkpoint：

```bash
PYTHONPATH=SEG_Multi-Source_Landslides \
python -m qpsalm_seg.cli.train_description \
  --config SEG_Multi-Source_Landslides/configs/qpsalm_segdesc_small.yaml \
  --stage overfit --region-encoder mgrr --device cuda \
  --max-steps 100 --max-train-samples 64 --max-val-samples 64 \
  --val-interval 25 --save-interval 50 \
  --output-dir outputs/qpsalm_description/d_minus_1_overfit \
  --overwrite-output
```

### D0-D3 课程训练

D0 使用 MMRS Caption 做遥感场景预适配：

```bash
PYTHONPATH=SEG_Multi-Source_Landslides \
python -m qpsalm_seg.cli.train_description \
  --config SEG_Multi-Source_Landslides/configs/qpsalm_segdesc_small.yaml \
  --stage mmrs_caption --device cuda \
  --output-dir outputs/qpsalm_description/d0_mmrs_seed42 --overwrite-output
```

D0/D1 只训练 `desc_adapter`、task-neutral visual projection 和 instruction/visual special
embeddings；全图 caption 序列不注入 MGRR region tokens。MGRR、description spatial backbone、
region projector 和 region special embedding 从 D2 才进入训练，跨 stage 必须使用
`--initialize-from` 重建 optimizer。D0/D1 还会跳过四尺度 spatial cache 投影；实际参数集合写入
运行目录的 `trainable_parameter_manifest.json`。该路径使用
`qpsalm_description_causal_v4_stage_separated`；旧的 v3 实验描述 checkpoint 需要舍弃并从 D0
重训，现有分割 checkpoint 和 vision cache 不需要重建。

D1 使用 RSICap 校准详细描述，并按配置回放 30% MMRS：

```bash
PYTHONPATH=SEG_Multi-Source_Landslides \
python -m qpsalm_seg.cli.train_description \
  --config SEG_Multi-Source_Landslides/configs/qpsalm_segdesc_small.yaml \
  --stage rsicap_caption --device cuda \
  --initialize-from outputs/qpsalm_description/d0_mmrs_seed42/checkpoint_best.pt \
  --output-dir outputs/qpsalm_description/d1_rsicap_seed42 --overwrite-output
```

D2 只做 DIOR 同图候选区域对齐；batch 必须至少为 2：

```bash
PYTHONPATH=SEG_Multi-Source_Landslides \
python -m qpsalm_seg.cli.train_description \
  --config SEG_Multi-Source_Landslides/configs/qpsalm_segdesc_small.yaml \
  --stage dior_alignment --batch-size 4 --device cuda \
  --initialize-from outputs/qpsalm_description/d1_rsicap_seed42/checkpoint_best.pt \
  --output-dir outputs/qpsalm_description/d2_dior_seed42 --overwrite-output
```

同一 parent 内文本完全相同的区域使用 multi-positive contrastive target，不会互相充当
假负样本；训练 loss 与 same-image R@1 采用同一正样本定义。D2 训练使用 parent-grouped batch
sampler，使同图不同区域稳定成为 hard negatives，而不是依赖普通随机 batch 偶然相遇。

D3a 使用全部合法 train mask 和规则化结构事实。该 stage 没有人工 val，因此后续初始化使用
`checkpoint_last.pt`，不能按自动 candidate 指标选择“科学 best”：

```bash
PYTHONPATH=SEG_Multi-Source_Landslides \
python -m qpsalm_seg.cli.train_description \
  --config SEG_Multi-Source_Landslides/configs/qpsalm_segdesc_small.yaml \
  --stage bridge_auto --region-protocol vision_only --region-encoder mgrr --device cuda \
  --initialize-from outputs/qpsalm_description/d2_dior_seed42/checkpoint_best.pt \
  --output-dir outputs/qpsalm_description/d3a_bridge_auto_seed42 --overwrite-output
```

只有 M2 双人审核、仲裁和 gate 冻结后才能运行 D3b。D3b 使用独立 Bridge、DIOR 和 global-caption
DataLoader，默认按 3:1:1 交替：

```bash
PYTHONPATH=SEG_Multi-Source_Landslides \
python -m qpsalm_seg.cli.train_description \
  --config SEG_Multi-Source_Landslides/configs/qpsalm_segdesc_small.yaml \
  --stage bridge_expert --region-protocol vision_only --region-encoder mgrr --device cuda \
  --initialize-from outputs/qpsalm_description/d3a_bridge_auto_seed42/checkpoint_last.pt \
  --output-dir outputs/qpsalm_description/d3b_bridge_expert_seed42 --overwrite-output
```

将 `--region-encoder` 分别设为 `crop_only`、`full_image_box`、`masked_pooling`、
`roi_replay_only`、`mgrr_no_context`、`mgrr`，并从相同 D2 checkpoint 初始化，才能形成受控的
M4 消融。Assisted 和 Vision-only 也必须分开训练、分开报告。

### GT、固定预测与端到端评价

GT-mask oracle：

```bash
PYTHONPATH=SEG_Multi-Source_Landslides \
python -m qpsalm_seg.cli.eval_description \
  --config SEG_Multi-Source_Landslides/configs/qpsalm_segdesc_small.yaml \
  --stage bridge_expert \
  --checkpoint outputs/qpsalm_description/d3b_bridge_expert_seed42/checkpoint_best.pt \
  --split val --evaluation-mode gt_mask --region-encoder mgrr --device cuda \
  --max-val-samples 0 --max-generate-samples 0 --counterfactual-samples 128 \
  --output-dir outputs/qpsalm_description/d3b_bridge_expert_seed42/eval_gt_val \
  --overwrite-output
```

固定 val prediction 先由同一冻结分割 checkpoint 离线导出：

```bash
PYTHONPATH=SEG_Multi-Source_Landslides \
python -m qpsalm_seg.cli.export_predicted_regions \
  --segmentation-config SEG_Multi-Source_Landslides/configs/qpsalm_v2_small.yaml \
  --preset qwen_psalm_full \
  --checkpoint outputs/qpsalm_v2/small_qwen_b4_bf16_nockpt/checkpoint_best.pt \
  --source-index benchmark/landslide_region_description_v1_small/indexes/expert_val.jsonl \
  --split val \
  --vision-feature-cache outputs/qpsalm_v2/cache/small_qwen_psalm_full_qwen_vision_v3 \
  --output-dir outputs/qpsalm_description/predicted_val --device cuda --overwrite-output
```

固定预测评价必须同时使用 `stage=predicted_mask` 和 `evaluation_mode=fixed_prediction`：

```bash
PYTHONPATH=SEG_Multi-Source_Landslides \
python -m qpsalm_seg.cli.eval_description \
  --config SEG_Multi-Source_Landslides/configs/qpsalm_segdesc_small.yaml \
  --stage predicted_mask \
  --checkpoint outputs/qpsalm_description/d4_predicted_seed42/checkpoint_best.pt \
  --split val --evaluation-mode fixed_prediction \
  --predicted-index outputs/qpsalm_description/predicted_val/predicted_val.jsonl \
  --max-val-samples 0 --max-generate-samples 0 --counterfactual-samples 128 \
  --device cuda --output-dir outputs/qpsalm_description/d4_predicted_seed42/eval_fixed_val \
  --overwrite-output
```

端到端使用 `--stage bridge_expert --evaluation-mode end_to_end`。评估器按 Bridge region 身份
严格选择 segmentation instruction：`gt_global_mask` 使用全图指令，`gt_referring_mask`、
no-target 以及带 referring alias 的伪 component 使用对应的 referring/no-target 指令。
没有语言可识别 alias 的纯 `pseudo_instance_component` 不会回退为全图 mask，而是从端到端集合
排除并记录在 `end_to_end_coverage.excluded_by_reason`。逐条映射写入
`end_to_end_target_audit.jsonl`；默认以 `segmentation_mask_threshold=0.5` 二值化分割输出。
三种评价必须生成独立目录，不能混合统计。原始生成保存在
`raw_generations.jsonl`，主指标只读取未修复 JSON；deterministic repair 仅作错误分析。
独立 `eval_description` 默认完整评估并完整生成；只有 smoke 才显式传正整数上限。
端到端正式命令为：

```bash
PYTHONPATH=SEG_Multi-Source_Landslides \
python -m qpsalm_seg.cli.eval_description \
  --config SEG_Multi-Source_Landslides/configs/qpsalm_segdesc_small.yaml \
  --stage bridge_expert \
  --checkpoint outputs/qpsalm_description/d4_predicted_seed42/checkpoint_best.pt \
  --split val --evaluation-mode end_to_end --region-encoder mgrr --device cuda \
  --max-val-samples 0 --max-generate-samples 0 --counterfactual-samples 128 \
  --output-dir outputs/qpsalm_description/d4_predicted_seed42/eval_end_to_end_val \
  --overwrite-output
```

端到端 mask 会先按 segmentation resize transform 恢复到原图，再按 Description Cache
render transform 投影，禁止直接在两个 padded canvas 之间插值。

### D4 Out-of-Fold predicted-mask curriculum

先建立 parent-level 三折索引。OOF v2 只接受内容全部为 `split=train` 的 segmentation index：

```bash
PYTHONPATH=SEG_Multi-Source_Landslides \
python -m qpsalm_seg.cli.build_oof_folds \
  --segmentation-index benchmark/multisource_landslide_v2_small/indexes/instruction_train.jsonl \
  --bridge-index benchmark/landslide_region_description_v1_small/indexes/expert_train.jsonl \
  --num-folds 3 --seed 42 \
  --output-dir outputs/qpsalm_description/oof_folds_small --overwrite-output
```

对 fold 0、1、2 分别构建与其 train/holdout 指纹绑定的 segmentation Vision Cache v3，随后从头
训练一个排除该 fold 的 segmentation checkpoint。以 fold 0 为例：

```bash
PYTHONPATH=SEG_Multi-Source_Landslides \
python -m qpsalm_seg.cli.cache_qwen_vision_features \
  --config SEG_Multi-Source_Landslides/configs/qpsalm_v2_small.yaml \
  --preset qwen_psalm_full \
  --train-index outputs/qpsalm_description/oof_folds_small/fold_0_train.jsonl \
  --val-index outputs/qpsalm_description/oof_folds_small/fold_0_holdout.jsonl \
  --output-dir outputs/qpsalm_description/oof_folds_small/cache_fold_0 \
  --device cuda --backend qwen --overwrite

PYTHONPATH=SEG_Multi-Source_Landslides \
python -m qpsalm_seg.cli.train \
  --config SEG_Multi-Source_Landslides/configs/qpsalm_v2_small.yaml \
  --preset qwen_psalm_full --device cuda \
  --train-index outputs/qpsalm_description/oof_folds_small/fold_0_train.jsonl \
  --val-index outputs/qpsalm_description/oof_folds_small/fold_0_holdout.jsonl \
  --vision-feature-cache outputs/qpsalm_description/oof_folds_small/cache_fold_0 \
  --output-dir outputs/qpsalm_description/oof_folds_small/seg_fold_0 \
  --overwrite-output --skip-torch-preflight
```

用 fold-specific checkpoint 只预测对应 holdout。导出器会同时核验 checkpoint 中的
`config.train_index`、fold train hash 和 prediction holdout hash：

```bash
PYTHONPATH=SEG_Multi-Source_Landslides \
python -m qpsalm_seg.cli.export_predicted_regions \
  --segmentation-config SEG_Multi-Source_Landslides/configs/qpsalm_v2_small.yaml \
  --preset qwen_psalm_full \
  --checkpoint outputs/qpsalm_description/oof_folds_small/seg_fold_0/checkpoint_best.pt \
  --source-index benchmark/landslide_region_description_v1_small/indexes/expert_train.jsonl \
  --split train --checkpoint-fold 0 \
  --fold-manifest outputs/qpsalm_description/oof_folds_small/fold_manifest.json \
  --train-index outputs/qpsalm_description/oof_folds_small/fold_0_train.jsonl \
  --val-index outputs/qpsalm_description/oof_folds_small/fold_0_holdout.jsonl \
  --prediction-index outputs/qpsalm_description/oof_folds_small/fold_0_holdout.jsonl \
  --vision-feature-cache outputs/qpsalm_description/oof_folds_small/cache_fold_0 \
  --output-dir outputs/qpsalm_description/predicted_fold_0 --device cuda \
  --overwrite-output
```

三个 fold 全部导出后合并；缺失、重复、in-fold、错误 checkpoint、mask hash/shape 错误或
fold 覆盖不完整都会直接失败：

```bash
PYTHONPATH=SEG_Multi-Source_Landslides \
python -m qpsalm_seg.cli.merge_oof_predictions \
  --fold-manifest outputs/qpsalm_description/oof_folds_small/fold_manifest.json \
  --input outputs/qpsalm_description/predicted_fold_0/predicted_train_0.jsonl \
  --input outputs/qpsalm_description/predicted_fold_1/predicted_train_1.jsonl \
  --input outputs/qpsalm_description/predicted_fold_2/predicted_train_2.jsonl \
  --output outputs/qpsalm_description/predicted_train_oof.jsonl
```

D4 从通过 D3b 的权重开始，默认混入 25% OOF predicted masks，其余仍为 expert GT regions：

```bash
PYTHONPATH=SEG_Multi-Source_Landslides \
python -m qpsalm_seg.cli.train_description \
  --config SEG_Multi-Source_Landslides/configs/qpsalm_segdesc_small.yaml \
  --stage predicted_mask \
  --predicted-index outputs/qpsalm_description/predicted_train_oof.jsonl \
  --initialize-from outputs/qpsalm_description/d3b_bridge_expert_seed42/checkpoint_best.pt \
  --device cuda --output-dir outputs/qpsalm_description/d4_predicted_seed42 \
  --overwrite-output
```

### 专家事实性与三 seed MGRR 门槛

先冻结模型生成，再为同一批 parent 生成两份独立审核文件：

```bash
PYTHONPATH=SEG_Multi-Source_Landslides \
python -m qpsalm_seg.cli.score_expert_factuality \
  --eval-dir outputs/qpsalm_description/d3b_bridge_expert_seed42/eval_gt_val \
  --write-template \
  --output outputs/qpsalm_description/d3b_bridge_expert_seed42/eval_gt_val/expert_review_template.jsonl
```

模板包含冻结 instruction、model generation、区域/多模态面板路径和自动枚举的 factual
claims。审核者只能填写 `reviewer_id`、family score、每条 claim 的 support 和 notes；改写
generation、claim ID、来源字段或文本会被汇总器拒绝。模板不显示 reference target，避免
用标签措辞代替视觉事实判断。

两名审核者填写后汇总 parent-level ERFS：

```bash
PYTHONPATH=SEG_Multi-Source_Landslides \
python -m qpsalm_seg.cli.score_expert_factuality \
  --eval-dir outputs/qpsalm_description/d3b_bridge_expert_seed42/eval_gt_val \
  --review /path/to/reviewer_1.jsonl --review /path/to/reviewer_2.jsonl \
  --output outputs/qpsalm_description/d3b_bridge_expert_seed42/eval_gt_val/expert_factuality_report.json
```

`qpsalm-compare-description-runs` 需要三个 seed 各自成对的 generation、DIOR retrieval 和
expert factuality 报告；正式准入同时要求 ERFS 的 paired bootstrap CI 下界大于 0、R@1 提升，
专家 unsupported claim rate 不越过预注册非劣界限，并要求 shuffled-mask、region-swap、
cross-parent modality swap 的 paired target-score CI 上界小于 0，以及 modality removal 后
factual-claim count 的 paired CI 上界小于 0。四种正式反事实都必须完成配置指定的全部
有效样本数；覆盖不足时即使 CI 数值看似有利也不能通过门槛。
其中 region-swap 仅使用同一 parent 中另一个真实区域；跨 parent mask 和几何翻转不属于
same-image region-swap，不能用于补齐正式评估覆盖率。

```bash
PYTHONPATH=SEG_Multi-Source_Landslides \
python -m qpsalm_seg.cli.compare_description_runs \
  --baseline outputs/crop_s42/eval --candidate outputs/mgrr_s42/eval --seed 42 \
  --baseline outputs/crop_s123/eval --candidate outputs/mgrr_s123/eval --seed 123 \
  --baseline outputs/crop_s3407/eval --candidate outputs/mgrr_s3407/eval --seed 3407 \
  --baseline-retrieval outputs/crop_s42/dior_eval \
  --candidate-retrieval outputs/mgrr_s42/dior_eval \
  --baseline-retrieval outputs/crop_s123/dior_eval \
  --candidate-retrieval outputs/mgrr_s123/dior_eval \
  --baseline-retrieval outputs/crop_s3407/dior_eval \
  --candidate-retrieval outputs/mgrr_s3407/dior_eval \
  --baseline-expert outputs/crop_s42/expert_factuality_report.json \
  --candidate-expert outputs/mgrr_s42/expert_factuality_report.json \
  --baseline-expert outputs/crop_s123/expert_factuality_report.json \
  --candidate-expert outputs/mgrr_s123/expert_factuality_report.json \
  --baseline-expert outputs/crop_s3407/expert_factuality_report.json \
  --candidate-expert outputs/mgrr_s3407/expert_factuality_report.json \
  --output outputs/qpsalm_description/mgrr_seed_gate.json
```

### M7 联合训练与分割保持

M7 新运行必须用 `--initialize-from` 加载通过 M6 的 checkpoint，不能随机初始化描述头：

```bash
PYTHONPATH=SEG_Multi-Source_Landslides \
python -m qpsalm_seg.cli.train_segdesc_joint \
  --config SEG_Multi-Source_Landslides/configs/qpsalm_segdesc_small.yaml \
  --initialize-from outputs/qpsalm_description/d4_predicted_seed42/checkpoint_best.pt \
  --region-stage predicted_mask \
  --predicted-index outputs/qpsalm_description/predicted_train_oof.jsonl \
  --device cuda --output-dir outputs/qpsalm_description/m7_joint_seed42 \
  --overwrite-output
```

主路线默认令 `joint_train_shared_segmentation_dense=false`：SANE/QMEF/PMRD 与 controller
dense projection 保持冻结，segmentation batch 只更新 `default` adapter，description batch
只更新 `desc_adapter`、MGRR 和描述投影。`--train-shared-segmentation-dense` 仅用于已经通过
主路线 retention 后的独立消融，不能与默认结果混报。

`grad_accum_steps` 表示一个 optimizer step 内、对当前选中任务连续累积的 microbatch 数；
任务不会在同一次梯度累积中切换。默认 task pattern 为
`segmentation, global_caption, segmentation, region_description`，即 50/25/25。

联合运行使用 `qpsalm_segdesc_joint_v3_task_isolated`：首次遇到三类任务时分别检查目标参数有
非零有限梯度，并检查 inactive Adapter/模块保持零梯度。`joint_manifest.json` 保存 optimizer
逐参数清单和 parent population，`joint_coverage_latest.json` 保存各任务真实步数、样本数与
parent 覆盖；旧的二分类 description 梯度门禁报告不能作为 M7 验收依据。

新 run 会在联合优化前一次性冻结 `segmentation_monitor_baseline.json`。续训必须保留原输出
目录，并校验 checkpoint 中的 baseline identity、progress step 和三类 parent population；不能
把已联合训练的模型重新当作 baseline。因此 `--resume` 禁止与 `--overwrite-output` 同时使用：

```bash
PYTHONPATH=SEG_Multi-Source_Landslides \
python -m qpsalm_seg.cli.train_segdesc_joint \
  --config SEG_Multi-Source_Landslides/configs/qpsalm_segdesc_small.yaml \
  --resume outputs/qpsalm_description/m7_joint_seed42/checkpoint_last.pt \
  --device cuda --output-dir outputs/qpsalm_description/m7_joint_seed42
```

训练结束必须在与分割基线完全相同的完整 val 上执行 retention，而不是只看 monitor subset。
当前评估器会在 `coverage.sample_population` 中写入样本 ID、任务、目标引用、空间变换和
prompt 协议的确定性 SHA-256；
retention 同时要求两份报告样本数相同、身份指纹相同、阈值相同。旧版不含该字段的 baseline
报告不能通过正式门禁，需要先用当前 `qpsalm-eval` 对原分割 checkpoint 重新执行一次 full-val：

```bash
PYTHONPATH=SEG_Multi-Source_Landslides \
python -m qpsalm_seg.cli.eval \
  --config SEG_Multi-Source_Landslides/configs/qpsalm_v2_small.yaml \
  --preset qwen_psalm_full \
  --checkpoint outputs/qpsalm_v2/small_qwen_b4_bf16_nockpt/checkpoint_best.pt \
  --split val --device cuda --max-val-samples 0 \
  --vision-feature-cache outputs/qpsalm_v2/cache/small_qwen_psalm_full_qwen_vision_v3 \
  --output-dir outputs/qpsalm_v2/small_qwen_b4_bf16_nockpt/eval_val_population_v1 \
  --overwrite-output --skip-torch-preflight
```

然后使用新报告执行 retention：

```bash
PYTHONPATH=SEG_Multi-Source_Landslides \
python -m qpsalm_seg.cli.eval_segdesc_retention \
  --config SEG_Multi-Source_Landslides/configs/qpsalm_segdesc_small.yaml \
  --checkpoint outputs/qpsalm_description/m7_joint_seed42/checkpoint_best.pt \
  --baseline-eval-report outputs/qpsalm_v2/small_qwen_b4_bf16_nockpt/eval_val_population_v1/eval_report.json \
  --device cuda --output-dir outputs/qpsalm_description/m7_joint_seed42/retention_full_val \
  --overwrite-output
```

交互质检使用 `qpsalm-demo-description`，默认监听 `127.0.0.1:7861`。所有 M3-M7
工程入口完成并不等于 Full 准入；只有专家 Pilot、三 seed 和 retention 门槛均通过后才能构建 Full。

```bash
PYTHONPATH=SEG_Multi-Source_Landslides \
python -m qpsalm_seg.cli.demo_description \
  --config SEG_Multi-Source_Landslides/configs/qpsalm_segdesc_small.yaml \
  --stage bridge_expert \
  --checkpoint outputs/qpsalm_description/d3b_bridge_expert_seed42/checkpoint_best.pt \
  --split val --device cuda
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
RUN_NAME=small_qwen_b4_bf16_nockpt \
RUN_CONTROL=--overwrite \
CACHE_CONTROL=reuse \
bash SEG_Multi-Source_Landslides/scripts/run_qpsalm_experiment.sh
```

full：

```bash
BENCHMARK_SIZE=full \
PRESET=qwen_psalm_full \
SEED=42 \
RUN_NAME=full_qwen_b4_bf16_nockpt \
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

## 交互推理与 PPT 图库

在浏览器中选择 benchmark 样本、活动模态并输入指令。模型和 Qwen cache 在进程内只加载一次：

```bash
PYTHONPATH=SEG_Multi-Source_Landslides \
python -m qpsalm_seg.cli.demo \
  --config SEG_Multi-Source_Landslides/configs/qpsalm_v2_small.yaml \
  --preset qwen_psalm_full \
  --checkpoint outputs/qpsalm_v2/small_qwen_b4_bf16_nockpt/checkpoint_best.pt \
  --vision-feature-cache outputs/qpsalm_v2/cache/small_qwen_psalm_full_qwen_vision_v3 \
  --split val --device cuda --inbrowser
```

默认地址为 `http://127.0.0.1:7860`。原始 benchmark 指令对应正式 GT 指标；修改 instruction
或 condition 后，页面会把 Dice/IoU 标记为参考指标。关闭模态会同时更新像素输入、Qwen view、
prompt 和 QMEF availability，不会只改变界面文字。

从已有完整 val 报告生成强、典型、失败、指令对、小目标和 Sen12 专题图库：

```bash
PYTHONPATH=SEG_Multi-Source_Landslides \
python -m qpsalm_seg.cli.curate_gallery \
  --config SEG_Multi-Source_Landslides/configs/qpsalm_v2_small.yaml \
  --preset qwen_psalm_full \
  --checkpoint outputs/qpsalm_v2/small_qwen_b4_bf16_nockpt/checkpoint_best.pt \
  --eval-report outputs/qpsalm_v2/small_qwen_b4_bf16_nockpt/eval_val/eval_report.json \
  --vision-feature-cache outputs/qpsalm_v2/cache/small_qwen_psalm_full_qwen_vision_v3 \
  --split val --device cuda \
  --output-dir outputs/qpsalm_v2/small_qwen_b4_bf16_nockpt/ppt_gallery \
  --overwrite-output
```

入口只重新推理入选样本，输出 `gallery_index.html`、presentation overview、独立 mask、
`gallery_manifest.jsonl` 和 `gallery_summary.json`。PPT 图不导出 GT-only oracle proposal。

只生成局部/指代分割的 PPT 图库：

```bash
PYTHONPATH=SEG_Multi-Source_Landslides \
python -m qpsalm_seg.cli.curate_gallery \
  --config SEG_Multi-Source_Landslides/configs/qpsalm_v2_small.yaml \
  --preset qwen_psalm_full \
  --checkpoint outputs/qpsalm_v2/small_qwen_b4_bf16_nockpt/checkpoint_best.pt \
  --eval-report outputs/qpsalm_v2/small_qwen_b4_bf16_nockpt/eval_val/eval_report.json \
  --vision-feature-cache outputs/qpsalm_v2/cache/small_qwen_psalm_full_qwen_vision_v3 \
  --split val --task-family referring_landslide_segmentation --device cuda \
  --max-items 120 \
  --output-dir outputs/qpsalm_v2/small_qwen_b4_bf16_nockpt/ppt_gallery_referring \
  --overwrite-output
```

`--task-family` 在选样前过滤评估记录；可重复传入以组合多个任务族。该命令会从同一
parent 的不同位置、尺度、形态和数量指令中优先保留对照 pair，并同时抽取强、典型和失败案例。

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
  --run-dir outputs/qpsalm_v2/throughput_b4_nf4 --no-export-tables
PYTHONPATH=SEG_Multi-Source_Landslides python -m qpsalm_seg.cli.summarize_run \
  --run-dir outputs/qpsalm_v2/throughput_b4_frozen_bf16 --no-export-tables
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
| `qpsalm-cache-description-vision-features` | `qpsalm_seg.cli.cache_description_vision_features` |
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
| `qpsalm-curate-gallery` | `qpsalm_seg.cli.curate_gallery` |
| `qpsalm-demo` | `qpsalm_seg.cli.demo` |
| `qpsalm-train-description` | `qpsalm_seg.cli.train_description` |
| `qpsalm-eval-description` | `qpsalm_seg.cli.eval_description` |
| `qpsalm-build-oof-folds` | `qpsalm_seg.cli.build_oof_folds` |
| `qpsalm-export-predicted-regions` | `qpsalm_seg.cli.export_predicted_regions` |
| `qpsalm-merge-oof-predictions` | `qpsalm_seg.cli.merge_oof_predictions` |
| `qpsalm-train-segdesc-joint` | `qpsalm_seg.cli.train_segdesc_joint` |
| `qpsalm-eval-segdesc-retention` | `qpsalm_seg.cli.eval_segdesc_retention` |
| `qpsalm-compare-description-runs` | `qpsalm_seg.cli.compare_description_runs` |
| `qpsalm-eval-description-zero-shot` | `qpsalm_seg.cli.eval_description_zero_shot` |
| `qpsalm-demo-description` | `qpsalm_seg.cli.demo_description` |
| `qpsalm-score-expert-factuality` | `qpsalm_seg.cli.score_expert_factuality` |

## 静态检查与单元测试

```bash
bash -n scripts/run_1_build_benchmark.sh scripts/run_2_build_instruction_dataset.sh \
  scripts/run_3_build_description_benchmark.sh \
  scripts/run_4_build_landslide_bridge.sh \
  scripts/run_5_build_segdesc_dataset.sh \
  SEG_Multi-Source_Landslides/scripts/run_qpsalm_experiment.sh \
  SEG_Multi-Source_Landslides/scripts/run_qpsalm_smoke.sh

python -B -m py_compile $(find scripts/1-benchmark scripts/2-instruction \
  scripts/3-description scripts/4-landslide-bridge scripts/5-segdesc \
  SEG_Multi-Source_Landslides/qpsalm_seg -name '*.py' -type f)

PYTHONPATH=SEG_Multi-Source_Landslides python -B -m unittest \
  SEG_Multi-Source_Landslides/tests/test_benchmark_v2.py \
  SEG_Multi-Source_Landslides/tests/test_refactor_core.py \
  SEG_Multi-Source_Landslides/tests/test_inference_gallery.py \
  SEG_Multi-Source_Landslides/tests/test_renderer.py \
  SEG_Multi-Source_Landslides/tests/test_v2_integration.py \
  SEG_Multi-Source_Landslides/tests/test_description_benchmark.py \
  SEG_Multi-Source_Landslides/tests/test_landslide_bridge.py \
  SEG_Multi-Source_Landslides/tests/test_segdesc_unified_index.py \
  SEG_Multi-Source_Landslides/tests/test_segdesc_protocol.py -v
```

分割算法设计见 [docs/opt_refactor_algo.md](docs/opt_refactor_algo.md)，描述 benchmark、MGRR、
双 Adapter、训练课程和科学门槛见 [docs/benchmark_GAR.md](docs/benchmark_GAR.md)。
