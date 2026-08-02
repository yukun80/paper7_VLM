# OA-GroundRAG

本项目研究光学锚定、任意辅助模态增强的滑坡分割，以及基于候选区域证据的遥感视觉
观察、专业知识检索和证据受限生成。当前唯一算法依据是
[`docs/OA-GroundRAG_算法构建方案.md`](docs/OA-GroundRAG_算法构建方案.md)，活动状态只在
[`REBUILD_PROGRESS.md`](REBUILD_PROGRESS.md) 维护。

2026-08-01 已完成新路线 Stage 0 的权威迁移、Stage 1 工程定版和 Stage 2
RS-GeneralDesc Benchmark native v1 验收。仓库中的 `phase2/phase3/phase4` Python 包
分别承载 OA-AuxSeg、RS-GeneralDesc Benchmark 和 RS-VLM；活动配置、脚本与测试目录
已经使用各自的原生产品名。

当前统一状态：`Stage 2 RS-GeneralDesc Benchmark native v1 acceptance completed / Stage 3 RS-General Adapter retraining and Base-vs-Adapter Gate B pending; Gate A, ablations, sealed test and formal fixed masks remain deferred`。

当前依赖分两条支线：OA-AuxSeg 工程定版 → 未来 Gate A → formal fixed masks；
RS-GeneralDesc Stage 2 → Adapter 重训 → Gate B。两条支线在 Mask-Grounded 阶段汇合；
Gate A 延后不等于通过，也不阻止独立的 Stage 3 准备。

- **Stage 1 / OA-AuxSeg：** 五源 full Benchmark 有 53,645 条样本。batch-16 proposed
  训练由项目负责人在日志 step `213200` 主动停止，不再续训；step `206820` 的
  `checkpoint_best.pt` 已冻结为最终权重。离线 `training_report.json` 已完成，
  train/val Dice 分别为 `0.9753235/0.8254105`；Gate A 和 sealed test 尚未执行。
- **Stage 2 / RS-GeneralDesc：** native v1 Benchmark 有 274,693 records、104,954
  parents，manifest 原生 eligible、blockers 为空，saved full deep validation 为
  0 error / 0 warning，不使用额外验收报告。历史 `release_equivalence.json` 记录了当时
  的迁移结论，但旧实现未对所有前代 ledger entry 做严格逐文件验证，不能把它扩大为
  可追溯的逐项证明；当前没有证据表明 native payload 已损坏。
- **Stage 3 / RS-General Adapter：** 身份迁移前的候选 LoRA/checkpoint/training evidence
  已删除，不能用于 native v1 Gate B。下一步先按新 Benchmark 身份重新训练 Adapter，
  再冻结 Base-vs-Adapter 固定生成集合和判据。
- **Stage 4–5 / 专业证据与区域理解：** 现有 `phase4` 合同、RegionSelector、
  EvidenceBuilder、Qwen、checkpoint、推理和反事实评价继续复用；OA-GroundedEval、
  两遍式生成和扩展证据合同尚未实施。
- **Stage 6–9 / RAG 与集成：** `yukun80/RAG_tmp` commit
  `4241140a8005bb79b8d8ebce982c645b096b7aca` 仅作外部工程原型。它不复制进仓库、
  不作为运行时依赖，也不在 Gate C 前接入。

早先的 native 身份迁移曾对既有发布 payload 做一次字节复制与 metadata/ID 重写，并对
新树执行一次 full deep validation；没有重新读取三套源数据或重新编码图像。本次身份
绑定加固只修改代码、配置、测试和文档，没有重跑 repackage/deep validation，也未运行
optimizer/backward、GPU、test、Gate A 或 Gate B，未修改 OA-AuxSeg 权重或
`docs/RAG_knowledge/`。

## 环境

当前已验证环境：

```bash
cd /home/yukun80/codes/paper7_VLM
conda activate qwen3vl
python --version
```

实际验证版本为 Python 3.11.15、NumPy 2.1.2、h5py 3.16.0、
SciPy 1.16.2、Pillow 11.3.0、PyYAML 6.0.2、PyTorch 2.8.0+cu128、
torchvision 0.23.0+cu128 和 tqdm 4.67.1。
合同测试、官方 3 通道 stem 等价和官方权重严格加载已在 CPU 完成；真实训练配置使用
单卡 bf16。

## OA-AuxSeg（仓库 phase2；新路线 Stage 1）

为保持已验收配置、checkpoint 和命令兼容，现有工程路径继续使用
`oa_groundrag/phase2/`、`scripts/phase2_oa_auxseg/` 等历史实现名；这些名称不表示
算法主线中仍存在独立的 Region Grounding Phase 2。

实现路径：

```text
oa_groundrag/phase2/
├── contracts.py       # 严格模型、batch、配置与输出合同
├── data.py            # 复用 BenchmarkDataset/collate，稀疏子集采样
├── validity.py        # support、fractional coverage 与 masked 统计
├── fusion.py          # 全深度 MSPA、四尺度 selector、全宽 FRM/FFM
├── model.py           # ConvNeXt-Small、稀疏回填和逐阶段 forward
├── losses.py          # BCEWithLogits + soft Dice
├── metrics.py         # IoU/Dice/Precision/Recall/F1/no-target FPR
├── regions.py         # 确定性连通域与 registry 驱动区域特征
├── checkpoint.py      # 当前 schema 的原子保存与严格恢复
├── progress.py        # tqdm 训练/验证进度与简洁终态报告
├── engine.py          # 训练、评价、smoke、过拟合和推理
└── cli.py             # 单一 CLI
```

训练数据流保持阶段 1B 合同不变：

```text
StatefulTrainingBatcher / BenchmarkDataset
  -> collate_benchmark_samples
  -> optical tensor list + 按名称/sample_indices 稀疏辅助模态
  -> 共享官方 RGB stem + 签名专属 extra-band 残差 + fractional validity
  -> 共享 ConvNeXt-Small 光学主干
  -> 模态独立辅助 stem + 共享下采样
  -> 每个 stage 在模态仍独立时做空间选择
  -> 96/192/384/768、3/3/27/3 的共享 validity-aware MSPA
  -> stride 4/8/16/32 全宽 FRM/FFM 与光学/辅助双流传播
  -> 512 维四尺度 decoder -> BCE + Dice
```

- 光学 stem 按精确通道签名注册：所有签名共享官方 `3→96` RGB 分支，
  非 RGB 通道进入签名专属 `Cextra→96` 零初始化残差分支；额外光谱参数使用
  `new_lr=3e-4`，3 通道输出与官方 stem 完全一致；`direct_concat` 复用同一分解；
- 缺失辅助模态不会生成零占位，`source` 只用于日志和分组评价；
- DEM、slope 和 encoded InSAR 各自使用独立
  7×7/stride-4/padding-3 stem，
  后续共享下采样和完整 MSPA stage；
- validity 同时保留 bool support 和逐通道有效比例 coverage；下采样使用 area average，
  不再用 max pooling 把单个有效像素扩张成完整有效格；
- `cmnext_injection` 在每个 stage 使用 depthwise 3×3 + pointwise sigmoid score，
  执行 `(1+score)×feature` 后的 masked channel-wise max；
- `injection_quality` 与 `proposed_dropout` 在每个 stage 使用全通道局部 scorer 和
  null auxiliary 空间 softmax；缺失、零覆盖和零光学重叠位置在 softmax 前排除；
- FRM 保持 `lambda_channel=lambda_spatial=0.5`；FFM 不降维，使用完整 stage
  通道和 `3/6/12/24` heads 的 `KᵀV` channel-context cross attention；
  fractional coverage 只乘入 `V`，不再除以 coverage/token 总量，全有效时与
  DELIVER 的 `softmax((KᵀV)×scale)` 数值公式一致；
  FRM 校正后的光学和辅助特征都进入下一 stage；
- proposed 对原生有辅助样本在 `1..N` 上均匀采样 cardinality，再执行
  `p=0.2` modality dropout；若全部被丢弃，就用同一可恢复 RNG 从原选择中恢复一个
  模态，因此原生有辅助样本不会被采样成人为空辅助；网络内部空间 null selector 保留；
  原生无辅助样本保持 optical-only；
- 训练 batcher 跨 permutation 边界补齐，每步严格返回 runtime config 指定的
  device batch，并将 permutation、cursor 和 RNG 写入 checkpoint；small smoke 固定
  batch 8，正式 full 配置使用 batch 16；可选 `balanced_target_presence` 按一半
  positive、一半 empty 采样，不使用 source；
- 所有辅助缺失或覆盖为零时显式硬旁路，退化为完全相同的光学路径。

统一支持六个 `variant`：

1. `optical_only`
2. `direct_concat`
3. `mean_auxiliary_fusion`
4. `cmnext_injection`
5. `injection_quality`
6. `proposed_dropout`

模型输出：

```text
mask_logits, mask_probability  [B,1,224,224]
no_target_score                [B]
modality_weights               [B,4]
modality_weight_maps           [B,4,56,56], [B,4,28,28],
                               [B,4,14,14], [B,4,7,7]
candidate_regions              每样本确定性区域列表
region_features                每区域 [268]
```

诊断权重固定列顺序为
`dem, insar_velocity, slope, __null__`，
四尺度空间图对应 stride 4/8/16/32；`modality_weights` 是各尺度在光学 coverage
内池化后等权平均并归一化的摘要，不解释为地学证据。区域使用阈值 0.5、8 邻域连通并过滤小于 16 像素的组件；
268 维特征由 128 维光学 pooling、128 维 fused pooling、8 维几何和 4 维权重组成。
模态列数和区域维度由当前活跃 registry 推导；若未来经审计的 Benchmark 只保留
`dem / insar_velocity / slope` 的子集，接口会相应收缩。模型、checkpoint 和推理
schema 均为 v6，runtime config 为 v5；旧 v5 checkpoint 和 v4 runtime config 会被
明确拒绝，不提供兼容包装。

### 权重门槛

真实训练只接受以下本地 torchvision 官方 state_dict，严格加载并记录 SHA-256：

```bash
test -f models_zoo/ConvNeXt/convnext_small-0c510722.pth
sha256sum models_zoo/ConvNeXt/convnext_small-0c510722.pth
```

当前权重已就位，SHA-256 为
`0c510722adfd92966a2bd72b92f785ca05966bbac03cafe2f7a90b1f54bfab9a`，并已通过
torchvision ConvNeXt-Small `strict=True` 加载。ConvNeXt-Small 光学主干约
50.2M 参数；完整 v6 模型在当前五源 registry 下为 152,916,881 个参数。
光学和辅助阶段通道均为 `96/192/384/768`，深度为 `3/3/27/3`。程序不会联网，也不会使用 `timm`；
权重缺失时真实训练入口仍会明确失败。

### Small 人工运行顺序

以下是当前 500 条五源 small 的 v6 人工验收顺序；这些命令均写入新的 v6 目录，
不覆盖既有训练产物：

```bash
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

`smoke` 对六个 variant 各完成一次真实异构 batch=8 的 forward、BCE+Dice、
backward、optimizer step 和临时 checkpoint 重载。`overfit` 使用配置所指 Benchmark
的全部 train，
保持 proposed 结构但固定全部可用辅助模态，强制 FP32、weight decay/dropout/
stochastic depth 为 0、clip=5，并每 100 step 评价完整 train，全部阈值通过时提前停止。
两个 300-step 配置分别运行 uniform 与 4-positive/4-empty sampler；两者都使用
非空 cardinality 采样、`p=0.2` dropout 和非空恢复。balanced 只有在既定
Dice/FPR 条件优于 uniform 时才可成为后续默认，当前默认仍为 uniform。

训练同时原子保存 `checkpoint_last.pt` 和按 val Dice、val loss、no-target FPR
依次择优的 `checkpoint_best.pt`。checkpoint 保存模型、optimizer、scheduler、AMP、
step、Python/NumPy/Torch/CUDA RNG、训练 permutation/cursor、subset sampler、模型
registry、Benchmark index/manifest SHA-256、included/excluded sources 和 backbone
SHA-256。只恢复当前 schema。评价和推理同时重验完整 Benchmark 身份、registry、
模态列顺序、四尺度融合合同和区域维度，不接受仅伪造 index hash 但其他合同不一致的
checkpoint。`training_report.json` 额外记录 native-none、dropout-restored、
single/multi/all、原生/实际激活辅助样本、全局与条件辅助曝光、实际 batch/累计样本、
梯度裁剪比例与缩放、extra-band 梯度和更新、
有覆盖辅助样本条件下的全局/分尺度/分 source 权重，以及完整 best/last 验证轨迹。

负责人主动停止训练时使用 `finalize`，它不创建 optimizer、不更新权重、不裁剪日志，
只严格核对 best/last/log/Benchmark 身份，重新评价 train/val、检查 checkpoint
严格重载并原子写入 `training_report.json`。计划 `max_steps` 是预算上限，不是要求
续训的充分理由。报告分别记录 last logged、last checkpoint 和 selected best step；
Gate A、test 和 formal acceptance 状态保持未执行。

### 训练终端进度

`train` 和 `overfit` 在交互终端使用 tqdm 单行进度条，刷新频率不高于每秒一次。
训练条显示 loss、EMA loss、BCE、Dice loss、主干/新模块学习率、吞吐、ETA 和
CUDA 峰值；验证条显示 batch 进度、运行 loss、Dice 和 IoU。每次验证结束只保留
overall 的 IoU、Dice、Precision、Recall、F1、positive-only Dice 和 no-target FPR，
分 source 与模态组合的详细结果仍写入 JSON。

非交互终端或输出重定向时不写 ANSI 控制字符，只在 step 1、每个 `log_interval`
和最后一步打印固定训练行。`train_log.jsonl` 继续按 `log_interval` 落盘，并在新训练
中额外记录 step 耗时、吞吐、ETA、wall time 和显存；checkpoint 保存、最终评价及
严格重载都会打印阶段状态和耗时。

训练结束默认只打印关键指标和产物路径。需要完整 JSON 输出到 stdout 时使用：

```bash
python scripts/phase2_oa_auxseg/run_oa_auxseg.py train \
  --config configs/phase2_oa_auxseg/small_proposed_dropout.json \
  --full-report-json
```

进度始终写 stderr，因此 `--full-report-json` 的 stdout 可以直接交给 JSON 工具。
完整报告无论是否使用该参数都会写入 `training_report.json`。

评价和两个正式来源的推理命令：

```bash
python scripts/phase2_oa_auxseg/run_oa_auxseg.py evaluate \
  --config configs/phase2_oa_auxseg/small_proposed_dropout.json \
  --checkpoint outputs/phase2_oa_auxseg/small_proposed_dropout_v6_uniform/checkpoint_best.pt \
  --split val \
  --output outputs/phase2_oa_auxseg/small_proposed_dropout_v6_uniform/val_metrics.json

python scripts/phase2_oa_auxseg/run_oa_auxseg.py infer \
  --config configs/phase2_oa_auxseg/small_proposed_dropout.json \
  --checkpoint outputs/phase2_oa_auxseg/small_proposed_dropout_v6_uniform/checkpoint_best.pt \
  --split val --source landslide4sense \
  --output-dir outputs/phase2_oa_auxseg/infer_landslide4sense

python scripts/phase2_oa_auxseg/run_oa_auxseg.py infer \
  --config configs/phase2_oa_auxseg/small_proposed_dropout.json \
  --checkpoint outputs/phase2_oa_auxseg/small_proposed_dropout_v6_uniform/checkpoint_best.pt \
  --split val --source multimodal_landslide \
  --output-dir outputs/phase2_oa_auxseg/infer_multimodal_landslide
```

推理目录原子发布并默认拒绝覆盖，包含 `predictions.jsonl`、`predictions.npz`
和 `manifest.json`。NPZ 保存全局 probability/mask、区域 masks/features、全局模态
权重和四尺度 float32 空间权重图；JSONL 只保存空间图对应的 NPZ key。

## RS-GeneralDesc Benchmark（仓库 phase3；新路线 Stage 2）

RS-GeneralDesc 的库实现继续位于仓库 `phase3` 包，活动入口均使用原生产品名：

```text
oa_groundrag/phase3/             # native canonical、builder、repackage、validator、Dataset、exporter
scripts/phase3_rs_generaldesc/   # audit/build/repackage/validate/export 薄 CLI
configs/phase3_rs_generaldesc/   # smoke/full/train/val 严格 YAML
tests/phase3_rs_generaldesc/     # 单元与合成端到端测试
```

公共 API 为 `load_build_config`、`audit_sources`、`build_benchmark`、
`validate_benchmark`、`repackage_benchmark`、`RSGeneralDescDataset`、
`HashLedgerVerifier`、`ParentBalancedSampler` 和 `export_qwen`。canonical schema 为
`rs_generaldesc.canonical.v1`，manifest schema 为 `rs_generaldesc.manifest.v1`，
scope 为 `rs_generaldesc_external_train_val`；
manifest 是物理布局唯一入口。Qwen messages 独立导出，不进入 canonical 真值。

Builder 只加载 RSGPT、MMRS-1M 和 DisasterM3，只能发布
`external_train/external_val` 与七类 RS-GeneralDesc 文本任务。full build 在 deep
validation 完成后必须 `formal_acceptance_eligible=true` 且 blockers 为空；smoke 与
deep-pending 仍保留真实工程 blocker。配置、canonical、Dataset、exporter 和 validator
均没有 mask target、Gold/Silver、混合 build、旧 schema 兼容或 alias。

`repackage` 是一次性确定性重发布入口：它锁定前代 manifest/build/payload/hash identity，
严格解析 hash ledger，并在读取 record/metadata 时核验实际 bytes；复制 asset 时在同一
次流式复制中核验 size/SHA，不增加第二遍扫描。manifest layout、asset inventory、ledger
与实际 payload 文件集合必须一致，全部 entry 完成验证后才能写等价结论；随后才重写
canonical/provenance/ID/metadata、运行 full deep validation 并以 sibling staging 原子
发布。该加固只用 `/tmp` fixture 验证，未对真实 40 GB Benchmark 重跑 repackage。

三库统一保存“视觉证据 → 文本输出”，但保留独立任务语义：

- 全图 caption、视觉/属性/场景 QA、数量问答；
- bbox 指定区域描述、红蓝框对象空间关系描述；
- 明确灾前/灾后顺序的可见变化报告。

DisasterM3 RefSeg 的输出是 pixel mask，而不是人工区域描述，因此不进入该统一文本
Benchmark；phrase→bbox、detection、classification、灾害类型、恢复建议、SAR 和
非视觉风险/灾因结论也不采用。统一训练只共享 autoregressive text loss，训练 sampler
先平衡 task family、再平衡 parent；验证必须按 caption/report、QA/scene、count、
relation 和 region caption 分任务报告，不能用一个混合 validation loss 代表全部能力。

设计参考锁定 Hugging Face Datasets `4.8.5`、commit
`a015b2fa5c1a6cda677fa46f20a54773258553ac`，仅借鉴严格配置、Features、
确定性样本生成、ImageFolder metadata 和小 fixture 测试范式；没有引入
`datasets`、Hub、Viewer、联网缓存或发布运行时。

2026-07-30 已完成构建验证快照的口径如下。本轮只核对 manifest/schema/saved
validation 中记录的 identity，不重新统计这些全量数字：

- RSGPT：2,681 条 caption、764 条 visual QA、119 条 count、45 条 scene，
  合计 3,609；另记录 1 条重复 QA、13 条禁止 claim、5 条不采用方向题和
  415 张未引用图像；
- MMRS-1M：46,275 条多参考 caption records、141,154 条 VQA、
  30,809 条 bbox→phrase；另记录 9 条 VQA duplicate、3 条重复参考、
  30,820 条反向 grounding、11 个零面积 bbox，并明确不读取 `total.json`
  和 7 个 classification metadata；
- DisasterM3：采用 scene 18,184、count 22,912、boxed relation 2,661 和 visible
  report 9,089，合计 52,846；明确排除 49,552 条 RefSeg，不读取 mask archive；
  另记录 8,430 条非光学记录、1 个被 2 条候选引用的零字节图像和 3 条 schema
  错误；
- 三库合计 274,693 records、104,954 parents；full deep audit 未拒绝记录，并发现
  3,955 个精确内容重复组件，合并后为 100,054 components。防泄漏调整后的角色为
  train 261,646 / val 13,047 records。源 `train/eval/benchmark` 只保存在
  provenance；该重划分不能复现 RSGPT RSIEval 或 DisasterM3 Bench 官方结果。

phase3 builder 的合成有界 smoke 使用 native v1 临时根，只作为可重复工程证据：
RSGPT/MMRS-1M/DisasterM3 分别为 `17/6/12` 条，共 35 records、32 parents、
34 个唯一内容资产、8,944,309 bytes。7 个 text task family 全部覆盖，deep
validator 为 0 error / 0 warning；不传 source 配置时仍可遍历 37 个 image views、
4 个 bbox records、2 个 pre/post records，并运行 task-parent sampler、
DataLoader 和独立 train/val Qwen export。第二个独立构建树的 47 个文件逐项
SHA-256 一致。smoke manifest 明确记录 `formal_acceptance_eligible=false`，blocker 仅为
`bounded_smoke_profile`。下文 phase4 真实 Qwen bounded smoke 是身份迁移前的历史连通性
记录，不是当前 native v1 Adapter 或 Gate B 证据。

入口示例：

```bash
/home/yukun80/miniconda3/envs/qwen3vl/bin/python \
  scripts/phase3_rs_generaldesc/run_rs_generaldesc.py audit \
  --config configs/phase3_rs_generaldesc/full.yaml --deep \
  --report /tmp/rs_generaldesc_full_audit.json

/home/yukun80/miniconda3/envs/qwen3vl/bin/python \
  scripts/phase3_rs_generaldesc/run_rs_generaldesc.py build \
  --config configs/phase3_rs_generaldesc/full.yaml

/home/yukun80/miniconda3/envs/qwen3vl/bin/python \
  scripts/phase3_rs_generaldesc/run_rs_generaldesc.py export \
  --config configs/phase3_rs_generaldesc/qwen_train.yaml

/home/yukun80/miniconda3/envs/qwen3vl/bin/python \
  scripts/phase3_rs_generaldesc/run_rs_generaldesc.py export \
  --config configs/phase3_rs_generaldesc/qwen_val.yaml
```

`build`、`repackage` 和 `export` 的目标根必须预先不存在。正式 native 发布根是
`/home/yukun80/codes/benchmark/rs_generaldesc_v1`，其 build/payload/hash identity 由
manifest 唯一给出，saved deep validation 必须为 0 error / 0 warning。Stage 2 验收
不表示 OA-Grounded 数据通过，也不表示 Gate B 通过；`external_val` 仍只用于训练监控。

当前 native identity：build ID 为
`build_3ebc09a4daad10e121fc14c2727d9896e10371a95bbaf6b780d15aa42eaf3c03`，payload
SHA-256 为 `549281f296b357bce256e6af71cec7412fe17e36052d6a8674f4876ae2d06e0b`，
hash-manifest SHA-256 为
`55ac26d9771ce8385318fbd23a10b999afb754ac195e823be659f4e49b0a7090`。

## RS-VLM（仓库 phase4；新路线 Stage 3–9 复用）

唯一实现树为：

```text
oa_groundrag/phase4/                         # 严格库实现
scripts/phase4_rs_vlm/    # 薄 CLI
configs/phase4_rs_vlm/    # 人工 YAML
tests/phase4_rs_vlm/      # 小 fixture 与合成端到端
```

主参考锁定 Qwen3-VL revision
`96588727e44c78b25ba03ea03b8e12f7e64fd0da`（Apache-2.0，revision date
2026-01-30，查阅日期 2026-07-30）。本地借鉴其配置、Dataset/预处理、原生多图、
processor/model、训练和 checkpoint 的职责拆分，重写严格配置、assistant-only loss
mask、resume 身份、结构化输出、失败 artifact 和测试；不采用换邻样本重试、
FlashAttention、DeepSpeed、packing/video、vLLM、自动下载/外部 judge、视觉解冻、
segmentation token 或 pixel decoder。除 fine-tune 路径外，还核对了官方
Transformers quickstart 和 `evaluation/RealWorldQA/{README.md,run_realworldqa.py}`：
上游的生成→JSONL→规则/可选 LLM judge 只作为职责参考，本地重写为离线、严格、
分任务且带 evidence/counterfactual 约束的 evaluator。

两条数据流共享 processor、collator、trainer、checkpoint、inference 和 evaluator：

```text
External canonical record
→ RS-GeneralDesc 公共 task-aware renderer
→ Qwen processor/collator
→ generic loss/inference
→ per-task metrics

新 Stage 5 mask-grounded 数据合同或 oa_auxseg_inference_v6
→ RegionSelector
→ EvidenceBuilder
→ mask-grounded structured messages
→ Qwen processor/collator
→ loss/checkpoint/inference
→ evidence/counterfactual metrics
```

External 记录不会进入 `RegionSelector`，也不会由 bbox/caption/QA 制造 OA mask。
`RegionSelector` 只能返回已有 global/candidate mask 或显式 empty/no-target；bbox、
click、region ID、面积排名、3×3 位置和冻结 Qwen 编号 overlay 仅用于匹配已有候选。
编号模式必须显式声明 selector 冻结，且只接受严格 JSON 的已有 `region_id`。未匹配是
selection failure，不等价于语义 no-target。`EvidenceBuilder` 生成光学全图、mask
overlay、15% context crop，并程序计算半开 bbox、centroid、area/ratio、位置、8 连通
fragment、4 邻域 perimeter、compactness 和 elongation。no-target 不生成 bbox、
overlay、crop 或形态事实。辅助证据必须声明配准、coverage、unit 和 sign convention；
不足或未知会生成限制 reason code。attention、quality weight 和 region feature 只记
provenance，不作为地学证据。当前 v1 Evidence 合同仍要求 RAG 输入为空；该约束用于
保持 Stage 5 无 RAG 基线，只有 Gate C 通过后才在 Stage 6 扩展。

版本化合同为：

- `rs_vlm.config.v2`
- `rs_vlm.evidence.v1`
- `rs_vlm.model_output.v1`
- `rs_vlm.prediction.v1`
- `rs_vlm.failure.v1`
- `rs_vlm.checkpoint.v1`
- `rs_vlm.run_manifest.v1`

YAML 只保存人工配置；JSON 保存 manifest/config snapshot/metrics/hash；JSONL 保存逐条
prediction/failure/provenance；mask、图像和 tensor 使用独立二进制资产。所有输出根
原子发布并拒绝覆盖、链接和路径逃逸。GT、fixed predicted 和 end-to-end predicted
mask 必须使用不同运行根和报告。

phase4 data config 直接固定实际 manifest 与 saved validation 文件 SHA，并绑定 native
canonical schema、build ID、payload SHA-256 和 hash-manifest SHA-256。preflight 严格解析
ledger，按需核验 canonical schema、build config、statistics，检查 shard layout、role
映射和 ledger entry，但不读取 record/asset 内容；Dataset 在解析 shard 前核验其 bytes，
并在首次返回或解码 asset 前核验实际 bytes。进程内缓存以 root、相对路径、预期身份和
stat fingerprint 为键，不会跨文件误命中。当前 config v2 只允许 `external_generic`，
已经移除未被活动路径消费的 External evidence/evaluation 配置段；其他产物 schema 仍为
v1，不提供 config v1 alias。临时 `MaskGroundedExample/Dataset` 也已撤回，
mask-grounded messages、EvidenceBuilder、RegionSelector、AuxSeg inference 和反事实评价
核心继续保留，Stage 4/5 必须先冻结新数据 schema，再增加活动入口。

Stage 3 的 Base 配置为
`configs/phase4_rs_vlm/rs_generaldesc_prompt_only_qwen3vl_2b.yaml`，只选择
`external_val` 且 `prompt_only` 训练参数为 0。LoRA 配置为
`rs_generaldesc_lora_qwen3vl_2b.yaml`；native v1 身份下尚未训练 Adapter，也未运行 Gate B。

模型路径配置化为本地 `models_zoo/Qwen3-VL-2B-Instruct`，强制
`local_files_only=true`、bf16、SDPA。冻结 prompt-only baseline 的训练参数为 0；
计划的 RS-General Adapter 使用 LLM attention `q/k/v/o_proj` LoRA，`r=8`、
`alpha=16`、dropout `0.05`，视觉 encoder 与 merger 冻结。锁定模型为
2,127,532,032 parameters，LoRA 为 3,211,264 parameters（约 0.151%）。重新训练仍采用
有界 `external_val` teacher-forced loss：每 100 个
optimizer step 验证一次，从 `external_val` 确定性选取至多 128 个不同 parent，
覆盖 3 个 source 和 7 个 task family，使用与训练一致的 assistant-only label
masking。主选择指标为七类任务 macro-average loss，并列时依次选择 overall loss
更低、step 更早的 checkpoint。验证运行于 `eval()`/`inference_mode()`，结束后恢复
训练状态且不改变 sampler、RNG 或多参考轮换；训练期间不生成文本。teacher-forced
loss 只用于 checkpoint 选择，不能代替 Stage 3 的 Base-vs-Adapter 固定生成 Gate B。

训练根内固定写入 `train_log.jsonl`、`sample_trace.jsonl`、
`validation_selection.json`、`validation_results.jsonl`、
`best_checkpoint.json` 和 `training_report.json`。前两者分别保存结构化 step 日志
和不刷屏的完整样本轨迹；首次验证前不会创建 validation results 或 best pointer。
TTY 使用进度条，非 TTY 每 10 step 打印 loss/EMA/LR/梯度范数、吞吐、ETA、显存和
输入等待时间/比例。`train_log.jsonl` 使用 `rs_vlm.train_log.v1` 保存这些输入侧诊断。
每 100 step 保存不可变 checkpoint，resume 严格核验配置、Benchmark、验证子集、
模型/processor、LoRA、sampler、RNG 和多参考状态。若异常退出发生在两个 checkpoint
之间，checkpoint 之后的日志/trace 会先保存到 `resume_recoveries/`，再原子回滚活动
artifact 到显式 checkpoint；不会静默丢弃，也不会把未保存权重的 step 当成已完成。
`sample_trace.jsonl` 使用 `rs_vlm.sample_trace.v1`，逐样本记录
`micro_step` 和 `batch_slot=0..3`。checkpoint、run manifest 和 training report
显式锁定 physical batch、accumulation、effective batch、输入 pipeline、worker、
prefetch、pin-memory 与 trace schema；缺少任一字段的 checkpoint 严格拒绝。

入口：

```bash
python scripts/phase4_rs_vlm/run_rs_vlm.py \
  preflight --config configs/phase4_rs_vlm/bounded_smoke.yaml

python scripts/phase4_rs_vlm/run_rs_vlm.py \
  smoke --config configs/phase4_rs_vlm/bounded_smoke.yaml \
  --output-root /tmp/rs_vlm_smoke_NEW

python scripts/phase4_rs_vlm/run_rs_vlm.py \
  preflight \
  --config configs/phase4_rs_vlm/rs_generaldesc_prompt_only_qwen3vl_2b.yaml

python -m unittest discover \
  -s tests/phase4_rs_vlm -p 'test_*.py' -v
```

身份迁移前完成的历史真实有界 smoke 最多探测 19 个确定性分片、每片前 256 条，共 4,864
records；最终选取 7 records/7 parents、8 unique assets、2,430,960 asset bytes，
覆盖 `external_train/external_val`、RSGPT/MMRS-1M/DisasterM3 和全部 7 task
family。真实 Qwen processor 处理 2,155 input tokens、9 images、219 supervised
tokens；copied asset bytes 为 0。其 `/tmp` 过程产物已按负责人要求清理，统计来自
已记录结果，未重新扫描 Benchmark。合成 tiny model 覆盖
loss→optimizer→checkpoint→resume→inference→evaluation；该 smoke 只证明当时的通用
遥感描述入口连通，不是当前 native v1 Adapter checkpoint 或 Gate B 证据。

身份迁移前的 phase4 Adapter 输出已经删除且不保留 alias 或备份。下一步必须先在
native v1 Benchmark 身份上重新训练，再冻结 Gate B；不得从已删除产物推断结论。

## 历史 Benchmark 构建程序（仓库 phase1）

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
├── small/      # 历史 500 条五源产物；当前现场缺失
└── full/       # 当前 53,645 条五源只读产物
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

### 磁盘、单样本与 batch

| 层级 | optical | mask | 第一维含义 |
| --- | --- | --- | --- |
| 磁盘 shard | `[N,C,H,W]` | `[N,1,H,W]` | `N` 是该 shard 的样本数 |
| Dataset 单样本 | `[C,H,W]` | `[1,H,W]` | 通过 index 读取一个 row |
| DataLoader batch | `List[Tensor[C_i,H,W]]` | `[B,1,H,W]` | `B` 是运行时 batch size |

磁盘中的 `N` 不是训练 batch 的 `B`。`index.jsonl` 中每条记录使用
`storage.shard + storage.row` 定位样本；Dataset 只读取指定 row。不同光学通道签名
不能直接堆叠，因此 optical 保持 tensor 列表，mask 因 shape 固定而直接堆叠。

shard 内部合同：

```text
/optical                         float32 [N,C_o,H,W]
/optical_pixel_valid             uint8   [N,C_o,H,W]
/optical_channel_valid           uint8   [N,C_o]
/mask                            uint8   [N,1,H,W]
/auxiliary/{name}/values         float32 [N,C_m,H,W]
/auxiliary/{name}/pixel_valid    uint8   [N,C_m,H,W]
/auxiliary/{name}/channel_valid  uint8   [N,C_m]
```

dense 数组使用 `[1,1,H,W]` chunk，channel validity 使用
`[min(N,256),C]` chunk；统一启用 gzip level 4、shuffle 和 Fletcher32。默认
`shard_target_mib=512`，builder 按未压缩逻辑量计算 shard 样本上限。当前 small 的
每个 source/split/模态签名分组都小于该上限，因此多数目录只有一个 HDF5；
Sen12Landslides 因存在三种 SAR 模态签名，每个 split 有三个 HDF5。这不代表样本
缺失，一个 HDF5 的第一维可以包含多条样本。

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
| Sen12Landslides | Sentinel-2 B02–B08, B8A, B11, B12 | ascending SAR, descending SAR, DEM | 6,671 |
| 合计 |  |  | 60,316 |

GDCLD、LMHLD、LandslideBench_agent 和 multimodal 保留源 split。Landslide4Sense
没有源 split，使用固定 `split_seed=20260724` 在 positive/background 内确定性分配
80/10/10，得到 train/val/test = 3,039/380/380。

Sen12Landslides 使用 post 时相作为当前分割输入，pre 文件只写入 provenance，不引入
变化检测语义。15 个 region 作为已知 source group，使用固定 split seed 确定性分组，
得到 train/val/test = 5,253/759/659，region 不跨 split。10 个 Sentinel-2 光学波段
全部保留；升轨和降轨 SAR 根据源 `channel_valid` 独立存在，完整缺失的轨道不创建
辅助模态，DEM 必须存在。

LandslideBench_agent 的 311 个 `location_key` 已知跨 split。按项目负责人要求，全部
2,130 条仍保留源 split；manifest 记录该例外，严格 group-isolation 验收不适用于该源。

### 构建数据源子集

small/full 共用 builder 支持可重复的 `--exclude-source`。允许值只使用以下 canonical
source ID，不接受目录名 alias：

```text
gdcld
lmhld
landslidebench_agent
landslide4sense
multimodal_landslide
sen12landslides
```

例如，构建不含 Sen12Landslides 的 five-source small：

```bash
OUTPUT_ROOT=../benchmark/oa_auxseg_hdf5_v1_no_sen12 \
bash scripts/phase1_benchmark_build/run_build_small.sh \
  --exclude-source sen12landslides
```

同时排除多个 source：

```bash
OUTPUT_ROOT=../benchmark/oa_auxseg_hdf5_v1_optical_only \
bash scripts/phase1_benchmark_build/run_build_small.sh \
  --exclude-source landslide4sense \
  --exclude-source multimodal_landslide \
  --exclude-source sen12landslides
```

排除发生在数据发现之前，被排除目录不会被扫描或读取。参数重复时自动去重；未知
source、缺少参数值或排除全部 source 会立即失败。默认不传参数时仍构建全部六源。
subset 必须使用新的 `OUTPUT_ROOT`，不得与当前 canonical Benchmark 共用输出目录。
新产物的 config/manifest 显式记录 `source_selection`、`included_sources` 和
`excluded_sources`；index、HDF5 schema、split 和确定性抽样规则不变。

## 历史已验收的 small（当前现场缺失）

历史 OA-AuxSeg v6 small 路径为
`/home/yukun80/codes/benchmark/oa_auxseg_hdf5_v1/small`。它曾是负责人重构后的五源
只读产物并明确排除 `sen12landslides`；当前现场路径缺失，本次没有修改或重建它。

当前 small 参数：

```text
patch_size=224
small_per_source=100
seed=20260724
split_seed=20260724
shard_target_mib=512
```

实际结果：

| source | train | val | test | positive | background | 合计 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gdcld | 20 | 40 | 40 | 60 | 40 | 100 |
| lmhld | 36 | 35 | 29 | 53 | 47 | 100 |
| landslidebench_agent | 34 | 34 | 32 | 50 | 50 | 100 |
| landslide4sense | 34 | 34 | 32 | 50 | 50 | 100 |
| multimodal_landslide | 34 | 66 | 0 | 67 | 33 | 100 |
| 合计 | 158 | 209 | 133 | 280 | 220 | 500 |

- shard：14 个；
- 目录占用：357,638,820 bytes，约 341.1 MiB；
- index SHA-256：
  `85c86a8d09dc2f602f04dad890ca65dab59e16235415e818427673dc1483f2df`；
- manifest SHA-256：
  `6f386b274b4fea3272ce065488e327719e57f9a39b025e8e2372cb27a3230ed8`；
- source selection：included 为
  `gdcld, lmhld, landslidebench_agent, landslide4sense, multimodal_landslide`，
  excluded 为 `sen12landslides`；
- deep validator：500/500 通过，无错误；
- DataLoader：raw 与 z-score 下的 `none/single/all` smoke 均通过；
- optical 覆盖 3/4/12 通道；
- train 为 158 条，其中 positive/empty=`106/52`、原生有/无辅助=`68/90`；
- val 为 209 条，其中 positive/empty=`105/104`；
- 辅助组合为 `DEM+slope` 与 `DEM+encoded InSAR`，不存在 SAR。

历史六源 600 条和 Sen12 适配产物已不是当前 OA-AuxSeg v6 权威，不能用于训练或
checkpoint 恢复。

目标已存在，构建入口会按设计拒绝覆盖；OA-AuxSeg 只读使用，不自动重建。

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

## Full Benchmark 与正式训练

只读估计命令：

```bash
python scripts/phase1_benchmark_build/1_1_build_benchmark.py \
  --mode full \
  --datasets-root ../datasets \
  --output-root ../benchmark/oa_auxseg_hdf5_v1 \
  --exclude-source sen12landslides \
  --patch-size 224 \
  --seed 20260724 \
  --split-seed 20260724 \
  --shard-target-mib 512 \
  --estimate-only
```

当前负责人构建的五源 full 已存在，OA-AuxSeg 只读使用：

```text
sample_count=53,645
train/val/test=36,761/12,375/4,509
shard_count=126
directory_size=32,404,761,903 bytes
index_sha256=389877226249d2477bdda62d937950339e9fa60df35558b945d02757e8d0da42
manifest_sha256=9a3b1478ed844f234e32b839fded67a937c49d202e3d8f5efd7db52596b5a00a
included_sources=gdcld,lmhld,landslidebench_agent,landslide4sense,multimodal_landslide
excluded_sources=sen12landslides
```

本次正式训练配置变更只读取 manifest 核对上述身份，没有重新构建 full，也没有重跑
53,645 条 deep validator。

项目负责人运行 full 的精确入口：

```bash
cd /home/yukun80/codes/paper7_VLM
conda activate qwen3vl
bash scripts/phase1_benchmark_build/run_build_full.sh \
  --exclude-source sen12landslides
```

full 验收标准：

1. 构建、deep validator、summarizer 和 DataLoader smoke 均 exit 0；
2. manifest `sample_count=53645`，`source_selection=subset`，
   included 为固定五源且 excluded 仅为 `sen12landslides`；
3. split 数量与本次五源 estimate/build 报告一致；
4. 所有输出 patch 为 224×224，mask 仅含 0/1；
5. image/modality validity shape 正确，无效影像像素为 0；
6. 3/4/12 通道光学、optical-only、单辅助和多辅助 batch 均可形成，不能出现
   10 通道或 SAR；
7. 除已批准的 311 个 LandslideBench location 例外外，不得出现新的已知 group 跨 split；
8. 不覆盖 small 或任何已有 full 输出。

full 完成上述 Benchmark 验收后，本次多模态训练原计划采用物理 batch 16 和最多约
100 个等价 epoch：

```bash
python scripts/phase2_oa_auxseg/run_oa_auxseg.py train \
  --config configs/phase2_oa_auxseg/full_proposed_dropout_b16_nockpt_e100.json
```

配置的预算上限为 `max_steps=229757`，若跑满会累计曝光 `3,676,112` 条样本，即
`100.0003` 个 train pass；每 `4596` step（约 2 epoch）评价并保存，warmup 为
`int(229757×0.05)=11487` steps。为提高 24 GiB 4090 的利用率，配置使用 bf16、
关闭 activation checkpointing，并将两档 LR 相对 batch 8 线性放大 2 倍。
原 `full_proposed_dropout.json` 及其 batch 8 日志不覆盖、不用于跨 batch 恢复。

项目负责人在日志 step `213200` 主动停止本次训练，并确认不再续训。已有
`checkpoint_best.pt` step `206820` 按既定 validation Dice、loss、no-target FPR
顺序选出并冻结为最终权重；`checkpoint_last.pt` 为 step `211416`，其后的 1784 个
已记录 step 没有进入任何 checkpoint，也不进入最终权重。离线定版命令为：

```bash
python scripts/phase2_oa_auxseg/run_oa_auxseg.py finalize \
  --config configs/phase2_oa_auxseg/full_proposed_dropout_b16_nockpt_e100.json \
  --checkpoint outputs/phase2_oa_auxseg/full_proposed_dropout_v6_b16_nockpt_e100/checkpoint_best.pt \
  --termination-reason project_owner_manual_stop
```

该命令不执行训练或 test，只生成 schema `oa_auxseg.training_report.v1` 报告。当前
实测工程结果为：

1. final checkpoint SHA-256：
   `672d39ab4220d8e1b4f949ca8d1d5dcd34f58898cecd1553dd56cdd9d84fb038`；
2. train：36,761 samples，Dice `0.9753234911`，IoU `0.9518355135`，
   no-target FPR `0.0011661808`；
3. val：12,375 samples，Dice `0.8254104938`，IoU `0.7027225166`，
   positive-only Dice `0.8343047072`，no-target FPR `0.4239963915`；
4. fresh val 与训练期 step-206820 selection snapshot 的指标差值为 0；
5. mask logits、probability、模态权重和四尺度权重图严格重载差值均为 0；
6. train/val、身份、日志证据和输入资产不变等工程检查全部通过；
7. `gate_a_evaluated=false`、`formal_acceptance=false`、`test_evaluated=false`。

后续评价、推理和 Mask-Grounded VLM Description 使用上述 `checkpoint_best.pt`，不
使用 `checkpoint_last.pt`，也不复制一份 `checkpoint_final.pt`。

## 测试

```bash
python -m unittest tests.phase2_oa_auxseg.test_oa_auxseg -v
python -m unittest discover -s tests/phase1_benchmark_build -v
python -m unittest discover -s tests/phase3_rs_generaldesc -p 'test_*.py' -v
python -m unittest discover \
  -s tests/phase4_rs_vlm -p 'test_*.py' -v
python -m py_compile oa_groundrag/*.py oa_groundrag/phase2/*.py \
  scripts/phase2_oa_auxseg/*.py tests/phase2_oa_auxseg/*.py \
  scripts/phase1_benchmark_build/*.py \
  tests/phase1_benchmark_build/test_benchmark_pipeline.py
bash -n scripts/phase1_benchmark_build/run_build_small.sh \
  scripts/phase1_benchmark_build/run_build_full.sh

python scripts/phase1_benchmark_build/1_2_validate_benchmark.py \
  --benchmark-root ../benchmark/oa_auxseg_hdf5_v1/small --deep
python scripts/phase1_benchmark_build/1_4_smoke_dataloader.py \
  --benchmark-root ../benchmark/oa_auxseg_hdf5_v1/small --normalization none
python scripts/phase1_benchmark_build/1_4_smoke_dataloader.py \
  --benchmark-root ../benchmark/oa_auxseg_hdf5_v1/small --normalization zscore
git diff --check
```

OA-AuxSeg 测试覆盖六 variant、3/4/12 通道、当前 small 五源稀疏子集、SAR/10 通道
明确拒绝、动态 registry 子集维度、官方 3 通道 stem 等价、额外波段梯度、缺失辅助退化、字典与稀疏索引
顺序不变、invalid-value 不变、fractional coverage、masked max、空间 softmax/null、
stride-4→8 渐进传播、四尺度权重图、DELIVER FFM 公式、定长 sampler 精确恢复、
extra-band 快学习参数组、LR floor、best/last 选择、首次 BCE+Dice backward 时全部
MSPA block 及四层 FRM/FFM 的辅助梯度、checkpoint v6 严格重载、旧 v5 拒绝、两个正式来源
的原子推理、8 邻域区域和 268 维特征。阶段 1B
回归继续覆盖六源合同、确定性、无效像素清零、二值 mask、拒绝覆盖，以及 validator
的损坏检测。另用 `/tmp` 中随机但严格匹配 torchvision 结构的临时 state_dict
曾在历史五源 small 完成 1-step CPU trainer、train 54/val 64 评价和差异 0.0 的重载；
临时文件已清理，该检查不属于官方权重或训练质量验收。

当前 Phase 2 共发现 40 个单元测试；其中 36 个不依赖历史 small 资产的测试全部通过，
包括 7 个新增 finalization 测试。另 4 个测试因当前不存在
`../benchmark/oa_auxseg_hdf5_v1/small` 而未能运行，不冒充通过；它们不是本次代码失败。
阶段 1B 历史回归结果仍为 10/10。实际命令和边界见 `REBUILD_PROGRESS.md`。

## 当前边界

- 当前统一状态是 `Stage 2 RS-GeneralDesc Benchmark native v1 acceptance completed / Stage 3 RS-General Adapter retraining and Base-vs-Adapter Gate B pending; Gate A, ablations, sealed test and formal fixed masks remain deferred`。
- 五源 OA-AuxSeg Benchmark、final checkpoint、训练日志、模型权重和
  `docs/RAG_knowledge/` 均未重建或覆盖；RS-GeneralDesc 已原生重发布到 native v1。
- OA-AuxSeg proposed 主模型已由项目负责人定版：final 权重为
  `checkpoint_best.pt` step `206820`，不再续训，工程报告已完成。
- Gate A、分割消融、多随机种子、sealed test 和正式 fixed predicted masks 仍未执行；
  权重工程定版不等于科学正式验收。
- 当前 OA-AuxSeg registry 只接受 `dem / insar_velocity / slope`；10 通道光学和 SAR
  明确拒绝。encoded InSAR 无可靠物理单位时不得输出定量物理结论。
- RS-GeneralDesc native v1 在早先迁移中进行过一次 full deep validation，manifest
  eligible、blockers 为空；前代 root 与额外 scope report 已删除。历史等价报告不能由
  本次修复追溯性地升级为严格前代逐文件证明，但目前没有 native payload 损坏证据。
- phase3 活动 builder/Dataset/exporter 严格 native External-only；RS-VLM config v2 只允许
  External，并直接绑定 manifest/validation/build/payload/ledger 与运行时消费文件。
- 前代 Adapter 产物已删除；Stage 3 必须重新训练，teacher-forced validation loss
  仍不等于 Gate B。
- OA-GroundedEval、Landslide Evidence Corpus、正式 mask-grounded 评价、RAG 和统一
  推理仍未实施。
- Gate C 通过前不接入 RAG；`RAG_tmp` 不作为当前算法组件或运行时依赖。
- 本次身份加固未运行 GPU、训练、test、Gate A/B、repackage 或 deep validation，也未
  下载、commit 或 push；早先 native 迁移执行过一次复制和全量 deep validation。
