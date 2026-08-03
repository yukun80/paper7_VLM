# REBUILD_PROGRESS

## 当前状态

- program: `OA_GROUNDRAG_V2`
- authority: `docs/OA-GroundRAG_算法构建方案.md`
- stage: `4`
- stage_name: `LANDSLIDE_EVIDENCE_CORPUS_AND_OA_GROUNDED_EVAL`
- stage_status: `in_progress`
- current_task: `STAGE4B_LOCAL_SILVER_PROVIDER_HARDENING`
- current_task_status: `complete`
- next_gate: `A (OA-AuxSeg branch) / C (after Stage 4–5)`
- scientific_status: `Stage 4B local Silver provider hardened / clean-HEAD preflight, GPU smoke, formal generation, automatic filtering, expert review and OA-GroundedEval pending`
- execution_date: `2026-08-03`
- branch: `main`
- stage0_baseline_head: `1436c9dab5121f8d766bb939d6812334d2ca6409`
- stage1_finalization_baseline_head: `88fec508048b1a8b3bc8dc8085396ba64449d33b`
- stage2_native_migration_baseline_head: `c198f0eb89148032f86c47e5163ac2a05498118d`
- stage3_gate_b_evidence_baseline_head: `2ad01f0723eaf698c0cbaff9bb3e993122bd87e0`
- stage4a_auto_pilot_baseline_head: `ee71e02127476da8e75b6bc9f2ce007fc38f77e5`
- active_training_process_found: `false`
- gpu_inference_only_run_performed: `true`
- stage3_training_performed: `true`
- stage3_gate_b_generation_performed: `true`
- stage3_gate_b_evaluated: `true`
- stage3_gate_b_passed: `true`
- stage3_adapter_formal_acceptance: `true`
- stage4a_auto_pilot_built: `true`
- stage4a_auto_pilot_validated: `true`
- stage4b_local_provider_implemented: `true`
- stage4b_formal_generation_performed: `false`
- stage4b_automatic_filter_performed: `false`
- stage4b_review_queue_prepared: `false`
- stage4b_gpu_run_performed: `false`
- stage4_silver_generated: `false`
- stage4_expert_review_completed: `false`
- oa_grounded_eval_completed: `false`
- stage2_gpu_run_performed: `false`
- stage2_native_full_deep_validation_performed: `true`
- identity_hardening_repackage_performed: `false`
- identity_hardening_deep_validation_performed: `false`
- identity_hardening_gpu_run_performed: `false`
- current_task_training_or_optimizer_step_performed: `false`
- formal_evaluation_performed: `true`
- test_split_evaluated: `false`
- commit_performed: `false`
- push_performed: `false`

Stage 0 权威迁移、Stage 1 工程定版、Stage 2 RS-GeneralDesc Benchmark native v1
重发布与验收、Stage 3 Adapter 重训和 Gate B、Stage 4A deterministic Auto Pilot，
以及 Stage 4B 本地 Silver provider 正式运行前加固均已完成。Stage 4B clean-HEAD
preflight、GPU smoke、正式 generation、自动过滤、专家审核和 OA-GroundedEval 尚未完成。
依赖分为 OA-AuxSeg
工程定版 → 未来 Gate A → formal fixed masks，以及 RS-GeneralDesc Stage 2 → Adapter
重训 → Gate B；两者在 Mask-Grounded 阶段汇合。Gate A 延后不等于通过，也不推翻
独立完成的 Stage 3 Gate B。Gate B 不表示 OA-Grounded、mask-grounded 或系统验收。

## Stage 0 已完成

- 重构版已接管 canonical 路径 `docs/OA-GroundRAG_算法构建方案.md`，成为唯一活动
  算法方案。
- 旧版已移至 `docs/archive/OA-GroundRAG_算法构建方案_v1.md`，并明确标记为历史资料。
- `AGENTS.md`、`README.md` 和本文件已统一为新方案 Stage 0–9。
- 历史工程路径 `oa_groundrag/phase2/phase3/phase4` 保持不变，只更新语义映射。
- 未修改模型、Trainer、Evaluator、Benchmark builder、RAG、测试或配置代码。
- 未修改 `../datasets`、`../benchmark`、`../external`、`models_zoo/`、`outputs/`、
  `docs/RAG_knowledge/` 或任何既有 archive 文件。

## 冻结资产

### OA-AuxSeg Benchmark

- root:
  `/home/yukun80/codes/benchmark/oa_auxseg_hdf5_v1/full`
- schema: `oa_auxseg_hdf5_v1`
- sample_count: `53645`
- split_counts: `train=36761 / val=12375 / test=4509`
- source_counts:
  `gdcld=13447 / lmhld=28185 / landslidebench_agent=2130 / landslide4sense=3799 / multimodal_landslide=6084`
- included_sources:
  `gdcld / lmhld / landslidebench_agent / landslide4sense / multimodal_landslide`
- excluded_sources: `sen12landslides`
- multimodal_landslide test count: `0`

这些值来自已发布 manifest 的轻量读取。本次未重跑 full validator、样本遍历、逐文件
hash 或数据源扫描。

### OA-AuxSeg 实现与训练

- repository path: `oa_groundrag/phase2`
- model schema: `oa_auxseg_model_v6`
- checkpoint schema: `oa_auxseg_checkpoint_v6`
- runtime config:
  `configs/phase2_oa_auxseg/full_proposed_dropout_b16_nockpt_e100.json`
- output root:
  `outputs/phase2_oa_auxseg/full_proposed_dropout_v6_b16_nockpt_e100`
- configured max steps: `229757`
- last logged training step: `213200`
- `checkpoint_best.pt`: final, step `206820`
- `checkpoint_last.pt`: trace only, step `211416`
- logged but not checkpointed: `211417–213200` (`1784` steps)
- final `training_report.json`: present
- report schema: `oa_auxseg.training_report.v1`
- completion mode: `project_owner_manual_stop`
- resume required: `false`
- final checkpoint SHA-256:
  `672d39ab4220d8e1b4f949ca8d1d5dcd34f58898cecd1553dd56cdd9d84fb038`
- visible training process on 2026-07-31: absent

项目负责人确认本次训练是主动手动停止，计划 `max_steps` 仅为预算上限，不要求续训。
既有选择规则冻结的 step `206820` best validation 为：

- Dice: `0.8254104937535791`
- IoU: `0.7027225165592563`
- positive-only Dice: `0.8343047072386582`
- no-target FPR: `0.4239963915200722`
- sample_count: `12375`

离线 fresh val 与上述训练期 selection snapshot 的指标差值为 0。完整 train 工程评价为
36,761 samples、Dice `0.9753234910697791`、IoU `0.9518355135202943`、no-target FPR
`0.0011661807580174927`。严格重载的 mask logits、probability、modality weights 和
四尺度 weight maps 差值均为 0。

该 `checkpoint_best.pt` 是项目负责人定版的当前最终权重，但不是 Gate-A-accepted
checkpoint。Gate A、消融、sealed test 和正式 fixed predicted masks 仍未执行。

### RS-GeneralDesc Benchmark native v1

- root:
  `/home/yukun80/codes/benchmark/rs_generaldesc_v1`
- manifest schema: `rs_generaldesc.manifest.v1`
- canonical schema: `rs_generaldesc.canonical.v1`
- benchmark scope: `rs_generaldesc_external_train_val`
- build_id:
  `build_3ebc09a4daad10e121fc14c2727d9896e10371a95bbaf6b780d15aa42eaf3c03`
- payload SHA-256:
  `549281f296b357bce256e6af71cec7412fe17e36052d6a8674f4876ae2d06e0b`
- semantic config SHA-256:
  `bb9b00ea44fb1c79e9efdfa00fbf73e8d8e9e5c416b8e9ff48d0c0b550e0d162`
- hash manifest SHA-256:
  `55ac26d9771ce8385318fbd23a10b999afb754ac195e823be659f4e49b0a7090`
- records / parents: `274693 / 104954`
- external_train / external_val: `261646 / 13047`
- saved deep validation: `errors=0 / warnings=0`
- source roots embedded: `false`
- formal acceptance eligible: `true`
- formal acceptance blockers: `[]`
- release equivalence schema: `rs_generaldesc.release_equivalence.v1`
- OA-Grounded acceptance: `false`
- Gate B evaluated: `true`（由独立 Stage 3 protocol 完成）
- external_val usage: `Stage 2 training monitoring; Stage 3 independent frozen Gate B selection`

该资产由冻结 payload 确定性重发布；canonical/provenance/ID/SHA/build identity 全部按
native v1 重算，且新树自身具备固定 manifest/hash identity、eligible/空 blockers 和
saved clean deep validation。历史 `release_equivalence.json` 记录了当时的 record、parent、
role 和 asset 等价结论，但旧 repackage 未逐项核验前代 ledger 的全部 record/metadata，
前代 root 又已删除，因此不能追溯性地把该报告升级为严格逐文件证明。目前没有证据表明
native payload 损坏；本次修复后的未来 repackage 才强制完整 ledger 验证。

活动 phase3 builder 只装载 RSGPT、MMRS-1M、DisasterM3，只生成
`rs_generaldesc_external_train_val`；Dataset/exporter 仅接受两类 role 和七类任务。
额外作用域报告、混合分支、旧 validator 与兼容 API 已删除。

### RS-General Adapter

身份迁移前的 phase4 Adapter/checkpoint/training evidence 已永久删除，不保留备份、链接
或 alias。首次 native v1 `batch=4/accumulation=4` 运行在 24 GB 目标显存边界下停止，
不得跨布局恢复；随后使用 physical batch 1、gradient accumulation 16、effective
batch 16 在全新 `rs_vlm_lora_qwen3vl_2b_b1a16` 根完成到 step 1000。best pointer 按
`macro_task_loss` 指向 step 1000（`0.8427927826882716`）。training report 保留
`formal_acceptance=false`，因为它只证明训练闭环；独立 Gate B report 随后以
`formal_acceptance=true` 接受该 Adapter。

最终 Adapter 身份：training report SHA-256
`a4f42e777eaab6e444f04d63b89f482ee31a077bf13006d587863bfa4fb1eb1e`，best pointer
`4d93e2c6c34fe01a10db373c00946166238b8d132bb32b9863b52e305b6f4db6`，step-1000
checkpoint manifest
`aa279659a4c563536f1d7554ed9e51643398365ce11539eb9865f77c1d3a621f`，Adapter 权重
`a367e39c626338a151dad33e6f7a7f9cc9887206dbcd261d147837e6408becc1`。

### Mask-Grounded VLM 基础

- repository path: `oa_groundrag/phase4`
- 已有能力:
  `RegionSelector / EvidenceBuilder / Qwen processor-model / prompt-only / LoRA / checkpoint-resume / inference / evaluation / counterfactual`
- 当前 evidence schema:
  `rs_vlm.evidence.v1`
- 当前合同仍要求 `rag_context=[]`
- 当前 evaluator 仍拒绝 formal evaluation
- 当前消息为单遍 user 生成合同

这些是 Stage 5 的可复用无 RAG 基线，不是 OA-GroundedEval、两遍式生成或完整
RAG 系统。当前 `rs_vlm.config.v2` 只运行 RS-GeneralDesc External；GT/fixed/end-to-end
mask、RegionSelector、EvidenceBuilder、mask-grounded messages、AuxSeg inference 和
反事实评价核心继续保留。临时 Mask-Grounded Dataset 合同已撤回，等待 Stage 4/5 冻结
新数据 schema 后再接入。

### RAG_tmp 与知识文档

- repository: `https://github.com/yukun80/RAG_tmp`
- inspected baseline commit:
  `4241140a8005bb79b8d8ebce982c645b096b7aca`
- local knowledge root: `docs/RAG_knowledge`
- local PDF count: `12`
- relationship: `external engineering prototype only`

根据项目负责人说明，RAG_tmp 使用的文档资料与本地 PDF 一致；远程 Git 仓库不包含
PDF 实体，因此 Stage 0 不重新下载或做逐文件 hash 对照。后续只借鉴 PDF/OCR、最小
知识单元、FTS5、dense retrieval、RRF、reranker、authority boost、引用和拒答范式。
不得直接复制代码、导入包、复用其中的 Ollama 最终生成器或把它设为运行时依赖。

## Stage 1 工程定版边界

Stage 1 的 proposed 主模型训练已经由负责人定版完成，不再续训。本轮新增的
`finalize` 入口不创建 optimizer、不执行 backward 或 scheduler step，只核对
best/last/log/Benchmark 身份，运行 train/val inference-only 评价、严格重载并原子
生成报告：

```bash
cd /home/yukun80/codes/paper7_VLM

/home/yukun80/miniconda3/envs/qwen3vl/bin/python \
  scripts/phase2_oa_auxseg/run_oa_auxseg.py finalize \
  --config configs/phase2_oa_auxseg/full_proposed_dropout_b16_nockpt_e100.json \
  --checkpoint outputs/phase2_oa_auxseg/full_proposed_dropout_v6_b16_nockpt_e100/checkpoint_best.pt \
  --termination-reason project_owner_manual_stop
```

该命令已经成功完成，不得重复覆盖报告。`checkpoint_best.pt`、`checkpoint_last.pt` 和
`train_log.jsonl` 的 size/mtime 在运行前后完全一致，未创建 `checkpoint_final.pt`。

Stage 1 剩余科学任务不是恢复训练。进入首次正式 test 前，必须只使用 train/val
预注册并冻结：

- checkpoint 选择规则；
- aggregate 与 positive-only 分割指标门槛；
- no-target FPR 门槛；
- auxiliary 非系统性退化判据；
- source、模态组合和低质量子组报告规则；
- 多随机种子与统计汇总规则。

不得根据 test 或当前 best validation 快照反推门槛。分割消融和多随机种子实验按项目
负责人指令延后到完整框架搭建后；只有未来 Gate A 通过，才运行一次 sealed test 并
导出供 Stage 5 使用的正式 fixed predicted masks。

## 当前科学任务：Stage 4A deterministic Auto Pilot 已完成

Stage 4A 已从冻结 OA-AuxSeg train split 构建并正式验证 500 条 Landslide Evidence
Corpus Auto records。五源各 100 条，总计 400 target / 100 no-target；没有打开 val/test
HDF5 内容，没有使用 predicted mask。发布根为
`outputs/stage4_landslide_evidence/landslide_evidence_corpus_v1_pilot_500`。

| 身份或规模 | 冻结值 |
| --- | --- |
| manifest SHA-256 | `37aebb9c5f8ceb720e0a1a3c8621212d44562fa6b6786d145c31e11ffa94f9bb` |
| corpus ID | `f4dd4c4e124e112154145a207d67f209ba9a3a1e7cfe56c35ab2f732624951d0` |
| ordered sample IDs SHA-256 | `84e1801fe9ec37284d7bf02b663153d057145726fc78c6d54e87ded11e185936` |
| records | `500`；`records.jsonl` SHA-256 `36f5dcbf8fe6008be09ccb291eb64a8a72398f01337ebb84ce13063d3a870c5d` |
| assets | `2200`；`115528315` bytes；500 optical / 500 mask / 400 overlay / 400 crop / 400 auxiliary |
| ledger | `2201` entries；file SHA-256 `0b0f7a1488e2069ce5d5ed03365d756905f81e742f12d2848dfb9b81de3ef430`；root `bf1f0446e17e6e94ea04859e4e6311226a5684690f44b6426ac64759e32b64c1` |

五种实际 modality signature 各覆盖 100 条。source/target/foreground-ratio quota 全部
达到；唯一组件覆盖不足为 `landslide4sense × large`，18 条均为多连通、单连通为 0，
manifest 记录 `both_covered=false`，未补造样本。所有 `silver_generated`、
`expert_review_completed`、`oa_grounded_eval`、`adapter_training_eligible` 和
`case_kb_eligible` 均为 false。

该 Corpus 是结构化 Auto evidence，不是分割 Benchmark、完整 RAG 知识库或人工测试集。
Silver 仍是假标签，正式生成和专家审核均未执行；OA-GroundedEval 继续作为独立的设计、
人工标注和冻结任务。Landslide-Evidence Adapter 只在后续 Gate E 失败时考虑。
Stage 4A v1 的辅助模态合同只包含 `dem / slope / insar_velocity`；当前 Corpus 不含 SAR，
SAR 是未来显式扩展，不能从现有通道或文件名推断补造。

### Stage 4A 验证记录

| 检查 | Exit | 结果 |
| --- | ---: | --- |
| `/tmp/oa_groundrag_stage4_tests` synthetic unittest | 0 | 11/11；覆盖 train-only/leakage、500 确定性去重、target/no-target、资产路径、科学 claims、manifest/record/ledger 篡改、原子发布、fake provider、禁止结论、review queue、mask-insensitive、symlink/path escape |
| Phase 3 单元测试 | 0 | 45/45 |
| 当时的 Phase 4 单元测试 | 0 | 72/72；Stage 4A 发布时的历史回归，当前套件规模见 Stage 4B 加固记录 |
| 三份活动 RS-VLM preflight | 0 | 3/3 metadata-only；临时 output roots 未创建 |
| 只读 Gate B verifier | 0 | accepted；report SHA-256 仍为 `b150de8eeed07c5cb3e9c808e7cec5c32f29c23fca9dd82bf7842786d89eb165` |
| 真实 `build-auto` | 0 | staging 内重算通过后原子发布 500 records / 2200 assets |
| 真实 CLI `validate` | 0 | `valid=true`、`source_verified=true`、400/100、五源各 100 |
| Python compile / `git diff --check` | 0 | Stage 4A、phase4、CLI 与临时测试 compile 通过；无 whitespace 错误 |
| 新增核心 Python 规模 | 0 | 指定五模块加薄 CLI 1516 physical / 1352 nonblank 行；既有 `phase4/evidence.py` 通用扩展净增 135 physical / 118 nonblank 行；合计净增 nonblank 1470 行 |
| 临时测试清理 | 0 | 已精确删除 `/tmp/oa_groundrag_stage4_tests`；未创建 `tests/stage4_landslide_evidence` |

临时测试命令为：

```bash
/home/yukun80/miniconda3/envs/qwen3vl/bin/python -m unittest discover \
  -s /tmp/oa_groundrag_stage4_tests -p 'test_stage4_landslide_evidence.py' -v
```

真实构建和验证命令为：

```bash
python scripts/stage4_landslide_evidence/run_landslide_evidence.py build-auto \
  --config configs/stage4_landslide_evidence/pilot_500.yaml
python scripts/stage4_landslide_evidence/run_landslide_evidence.py validate \
  --root outputs/stage4_landslide_evidence/landslide_evidence_corpus_v1_pilot_500
```

本任务未运行 GPU、外部 API、正式 Silver、专家审核、OA-GroundedEval、训练、test、
Gate A/C/D/E、predicted mask、RAG、下载、commit 或 push；未修改 Benchmark、checkpoint、
Gate B 正式产物和既有训练 outputs。

## 当前科学任务：Stage 4B local Silver provider 已加固

新增配置 `configs/stage4_landslide_evidence/silver_local_qwen3vl_2b_v1.yaml`，显式冻结
Stage 4A manifest/Corpus/records/ledger、Gate B frozen protocol/report、training report、
best pointer、checkpoint manifest、Adapter bytes、本地模型/processor identity、prompt
template SHA、生成参数、seed 和关键 Python/package 版本。Stage 4B 独立冻结
`max_input_tokens=4096`、最多 5 张图和 `max_new_tokens=256`，不修改 Gate B 的 2,048-token
配置或产物。Provider 复用现有
`Qwen3VLProcessorAdapter`、`Qwen3VLModelAdapter` 和 `CheckpointManager`；没有修改冻结的
Gate B 实现或正式产物。

正式计划固定为 500 records、1,000 regular candidates、25 wrong-mask 和 25
modality-removal attempts。每个 attempt 原子保存 raw output、严格七字段解析、request/
asset/prompt identity、seed 和模型身份；partial run 恢复时重新验证已有 attempt 与 raw、
request、资产和候选的绑定。生成、filter 和 review 分别发布到新 sibling root 下的独立
目录，均拒绝覆盖。自动过滤固定 `silver_accepted=false`，150 条 review queue 固定为
`prepared_not_reviewed`；该流程不表示 Gate C 或 OA-GroundedEval。

preflight 和 generate 都必须先用真实本地 processor 在 CPU 上编码全部唯一请求；相同输入
的双候选按 request SHA 去重，但预算报告仍须覆盖全部 1,050 attempts。超过 4,096 tokens
或 5 张图时，在正式 output root 创建和 GPU 模型加载前失败。25 条 wrong-mask 只从
`gdcld / landslidebench_agent / lmhld` 的纯光学 target-present 样本中确定性平衡选择，
原样本与 donor 都必须无辅助资产且 DEM/slope/InSAR 全部 unavailable；25 条
modality-removal 从 `landslide4sense / multimodal_landslide` 平衡选择。该 50 条只是 Silver
provider 反事实审计，不是 Gate C。

科学规则将滑坡表述解析为确定、谨慎和否定语境：target-present 允许“疑似/可能/候选/
尚不能确认”，拒绝确定性确认；no-target 拒绝确定和谨慎的存在声明，允许明确否定或证据
不足。任何失败继续保留 raw output 和明确 reason code，不静默改写。

### Stage 4B 实现验证记录

| 检查 | Exit | 结果 |
| --- | ---: | --- |
| 初始 `/tmp/oa_groundrag_stage4b_tests` temporary unittest | 0 | 14/14；初始 provider/恢复/filter/review/CLI 合同通过，路径已清理 |
| `/tmp/oa_groundrag_stage4b_repair_tests` temporary unittest | 0 | 8/8；覆盖 4,096-token 语义合同、三态滑坡表述、纯光学 wrong-mask/donor、五源 1,050-attempt 计划、预算去重/汇总及预算失败先于输出和模型加载 |
| 20-record 真实 processor CPU 预算 | 0 | 50 attempts 全覆盖、30 unique requests；最大 2,116 input tokens、5 images |
| 正式计划真实 processor CPU 预算 | 0 | 1,050 attempts 全覆盖、550 unique requests；最大 2,199 input tokens、5 images；未冻结 development protocol SHA |
| Stage 4A 只读 validator | 0 | `valid=true`、`source_verified=true`；500 records、2,200 assets、115,528,315 bytes，manifest/ordered IDs SHA 未变 |
| 正式 CLI preflight（当前未提交工作区） | 1 | 预期 `DIRTY_WORKTREE`；必须在提交后的 clean HEAD 重跑，不能冒充正式通过 |
| Phase 3 / Phase 4 单元测试 | 0 | 45/45；82/82；Phase 4 含 `gate-b-locate-media` 10 项严格永久测试 |
| 三份活动 RS-VLM preflight | 0 | 3/3；LoRA 已有正式 output，使用未创建的 `/tmp` override 完成只读检查 |
| 只读 Gate B verifier | 0 | `accepted`；正式 report SHA-256 仍为 `b150de8eeed07c5cb3e9c808e7cec5c32f29c23fca9dd82bf7842786d89eb165` |
| 冻结资产修复前后 SHA | 0 | Stage 4A manifest/records/ledger、Gate B report/paired scores、best pointer/training report、step-1000 manifest/Adapter 共 9 项完全不变 |
| Python compile / `git diff --check` | 0 | Stage 4B 核心与 CLI compile 通过；无 whitespace 错误 |
| 临时测试清理 | 0 | 已精确删除初始与 repair 两个 `/tmp` 测试路径；未向仓库新增 Stage 4B 永久测试 |

本次只完成 CPU 代码、合同和只读身份检查；没有运行 GPU、模型生成、外部 API、正式
Silver、自动过滤、专家审核、OA-GroundedEval、训练、test、Gate A/C/D/E、RAG、下载、
commit 或 push，也没有重跑 Stage 4A `build-auto`，没有修改 Corpus、Benchmark、
checkpoint、模型权重、训练输出或 Gate B 正式产物。

`gate-b-locate-media` 继续作为人工检查 Gate B prediction 的严格只读工具，验证 manifest、
ledger、record 与 asset identity；它不属于 Stage 4B Silver 运行时依赖，也不替代 Gate B
verifier。其 10 项永久测试保留在 Phase 4 回归中。

## 后续冻结顺序

1. **Stage 4B 正式运行。** 先在提交后的 clean HEAD 执行全计划 CPU preflight，再由负责人执行
   20-record GPU smoke；检查并冻结 prompt 后才运行 500×2 候选、50 条反事实、自动过滤
   和 150 条待审核队列。自动过滤不等于 Silver accepted。
2. **Stage 4 后续人工资产。** 专家审核、必要 Gold 和 OA-GroundedEval 须分别授权、
   人工标注并冻结。
3. **Stage 5：Mask-Grounded Baseline。** 比较 full/crop/overlay/multimodal 与
   GT/fixed/wrong/empty mask；Gate C 失败时先修 Evidence Representation。
4. **Stage 6–7：文本与案例 RAG。** 重新实现 Evidence Retrieval Provider，先文本，
   再正案例/困难负样本/分模态索引；RAG_tmp 不直接集成。
5. **Stage 8：可选 Landslide-Evidence Adapter。** 仅 Gate E 失败时训练，并执行
   RS-General retention Gate F。
6. **Stage 9：统一推理与报告。** 最后实现 Task Controller、两遍式生成、Evidence
   Cards、引用、failure artifact 和端到端评价。

## 已知科学与数据边界

- 当前 OA-AuxSeg 只支持 `dem / insar_velocity / slope` 辅助 registry；SAR 和未审计
  10 通道光学被明确拒绝。
- multimodal_landslide 当前没有 test 样本，不能宣称该 source 的独立 held-out test。
- encoded InSAR 的物理单位和 sign convention 未确认时，只能作为 encoded evidence；
  不得生成定量位移或物理方向结论。
- LMHLD 和 Landslide4Sense 缺少可靠地理 group；不得从文件名或 sample ID 伪造空间
  身份。
- 当前有负责人定版的 OA-AuxSeg final checkpoint 和已验证的 Stage 4A Auto Corpus，
  且 Stage 4B provider 已实现，但没有 Gate-A-accepted checkpoint、fixed predicted mask、
  正式 Silver/Gold、OA-GroundedEval 或正式 mask-grounded test。
- Gate C 通过前不得接入 RAG；RAG 不能为候选 mask 直接寻找支持理由，必须同时检索
  反对证据、混淆对象、困难负样本和传感器限制。

## Stage 0 验收

| 检查 | Exit | 结果 |
| --- | ---: | --- |
| `git diff --check` | 0 | 无 whitespace 错误 |
| canonical/archive 路径 | 0 | canonical 与 v1 archive 均存在，`_重构版` 临时名已消失 |
| v1 归档正文完整性 | 0 | 去除新增历史警告后 SHA-256 仍为 `cbbe3f6f7b964f559429c88e44d397f0a809718bbc85a20e3d7560f857abd4a9` |
| 活动文档旧路径扫描 | 0 | 无 `_重构版` 或已归档旧方案路径引用 |
| Markdown 本地链接 | 0 | `AGENTS.md`、README、进度和 canonical 方案的本地链接目标均存在 |
| OA Benchmark 轻量身份 | 0 | schema 与 `sample_count=53645` 匹配 |
| RS-GeneralDesc 轻量身份 | 0 | schema/scope/build/payload、saved validation `0/0` 匹配 |
| External LoRA 轻量身份 | 0 | `completed`、step 1000、16,000 samples、formal false 匹配 |
| OA-AuxSeg 训练状态（Stage 0 快照） | 0 | 当时日志末步 213200，training report 尚不存在 |
| Git 变化范围 | 0 | 仅 AGENTS、README、进度、canonical 方案和 v1 archive |

以上是 Stage 0 当时的验收记录；此后 Stage 1 已新增代码、测试和人工定版报告。

## Stage 1 工程定版记录（Gate A 未执行）

| 检查 | Exit | 结果 |
| --- | ---: | --- |
| 新增 finalization 定向测试 | 0 | 7/7 |
| 不依赖历史 small 的 Phase 2 回归 | 0 | 36/36 |
| Phase 2 全量发现 | 1 | 40 个中 36 通过；4 个仅因 `../benchmark/oa_auxseg_hdf5_v1/small` 不存在而失败 |
| 真实 `finalize` train/val | 0 | train 36,761 / val 12,375，工程检查全部通过 |
| fresh val replay | 0 | selection loss 与 overall 全指标差值均为 0 |
| checkpoint 严格重载 | 0 | 四类输出最大绝对差值均为 0 |
| best/last/log 资产不变 | 0 | size 与 mtime 前后完全一致 |
| sealed test | 未运行 | `test_evaluated=false` |
| formal Gate A | 未运行 | `gate_a_evaluated=false`、`formal_acceptance=false` |

缺失的 historical small 资产不是本次改动造成，也不为通过测试而重建、复制或修改
Benchmark。当前 full Benchmark 和真实 final checkpoint 已完成本任务所需验证。

## Stage 2 native v1 重发布与验收

| 检查 | Exit | 结果 |
| --- | ---: | --- |
| Phase 3 全量单元测试 | 0 | 当时 39/39；含 native builder/Dataset/exporter/repackage/validator |
| Phase 4 全量单元测试 | 0 | 当时 48/48；mask/evidence/region/AuxSeg 核心保留 |
| 真实 `repackage` | 0 | 历史报告记录 split/group/content/asset 等价；旧实现未逐项验证全部前代 ledger |
| 新树 full deep validation | 0 | 单次运行；0 error / 0 warning |
| CPU RS-VLM preflight | 0 | schema/build/payload/hash identity 匹配；未创建 output root |
| phase4 活动配置 preflight | 0 | 3/3 直接绑定 native identity |
| 前代 Benchmark 与 Adapter outputs | 已删除 | 验收完成后永久删除；无备份、链接或 alias |
| 源数据读取 / 图像重编码 | 未运行 | 只复制已发布图像字节并重写文本元数据 |
| GPU / training / test / Gate A / Gate B | 未运行 | native 身份迁移不扩张为科学评价 |

## 本次身份绑定加固

| 检查 | Exit | 结果 |
| --- | ---: | --- |
| Phase 3 全量单元测试 | 0 | 45/45；新增 ledger、record/metadata/asset 漂移与等价写入时序保护 |
| Phase 4 全量单元测试 | 0 | 53/53；新增 manifest/validation/metadata/shard/asset 运行时绑定保护 |
| 三份真实配置 preflight | 0 | 3/3 metadata-only；manifest/validation/build/payload/ledger identity 匹配 |
| 真实 Benchmark repackage/deep validation | 未运行 | 只使用 `/tmp` 小 fixture，不扫描 40 GB payload |
| Benchmark/checkpoint/outputs 写入 | 未运行 | size/mtime/SHA 与目录清单前后复核 |

## Stage 3 Gate B 正式验收记录

正式输出根为
`outputs/phase4_rs_vlm/rs_generaldesc_gate_b_qwen3vl_2b_v1`。只读证据复核重新验证了
training root、确定性 selection、Base/Adapter manifests 与 predictions、canonical
records、paired scores、全部指标、10,000 次 bootstrap 和六项判据；未读取图像资产。

| 正式 artifact | SHA-256 |
| --- | --- |
| frozen protocol file | `8378f6f107849439be3b402b0014df4007b10589e93602c1afb99173c2fb2c54` |
| selection file | `98290aaa585b798dcc5a30b9a4d47083e778aa4d48980e24bbce647705b915bd` |
| Base generation manifest | `bad680291426f65fd51dfaa35eca649968e478af963d3c5009cdbce733a699fb` |
| Base predictions | `862759c44400552f40f5211a38ceafd8d1f4712c7d6f870e3a2f0676d8ce8bd6` |
| Adapter generation manifest | `c69314727ba71ef712fd7bbde1990ebd610e6759714641103916adf901787168` |
| Adapter predictions | `a7c791fcfe8f3b94f6b188780bc0aed46dbf2fcf92f1bcca3b6617ca1c4bd98a` |
| paired scores | `64d6802e2b2438305fa7ba560bc4d90ebb4537d1048dc281333c707fb4d3975f` |
| Gate B report | `b150de8eeed07c5cb3e9c808e7cec5c32f29c23fca9dd82bf7842786d89eb165` |

protocol canonical SHA 为
`05ffb5ddf1940fb2474f13daf9ad7beec844f20d3da70a13c6c1c9f20a6eef0d`，selection
canonical SHA 为
`75285af6d8cee21c84760edcc2f20d71beb283bca8d340d6d3243e22c8537119`。selection
重算与发布 items 逐项一致，256 个 parent 与 training monitoring 的 128 个 parent
交集为 0。Base/Adapter 均为 256 predictions、0 failures，输入顺序、69,798 tokens
和 338 images 一致，paired count 为 256。

- primary task-macro：Base `0.2197314184`，Adapter `0.4559849197`，delta
  `0.2362535013`；95% paired-bootstrap CI 为
  `[0.2061703267, 0.2670255801]`；
- task 数量依次为 `42/41/41/41/11/40/40`；primary delta 依次为
  `0.4429630184 / 0.1316141258 / 0.2377844412 / 0.2792818227 /
  0.0753093476 / 0.2834546518 / 0.2033671016`，七类均为正；
- source 数量为 `disasterm3=128 / mmrs1m=101 / rsgpt=27`；source macro delta
  分别为 `0.2266998826 / 0.2959882661 / 0.0341324147`；
- 六项判据 observed 值分别为 `0.2061703267 / 7 / 0.0753093476 /
  0.0341324147 / 0.2743480215 / 0.0804878049`，全部 PASS；
- report 为 `status=completed`、`gate_b_evaluated=true`、`gate_b_passed=true`、
  `formal_acceptance=true`、`adapter_status=accepted`。

当前 PASS 只证明 RS-GeneralDesc native v1 固定 lexical protocol 下的相对提升，不表示
Gate A、OA-Grounded、mask-grounded、sealed test 或最终系统验收。v1 的 selection loader
本身不重推确定性唯一输出，模型完整权重、tokenizer/generation config、传递实现依赖和
运行环境也未形成完整 ledger；当前复核未发现实际 selection、输入或产物漂移。training
monitoring 与 Gate 共享一个 asset SHA，涉及两条 Gate `spatial_relation` records；删除
它们的非门控事后敏感性分析仍满足六项判据，delta 为 `0.2340555947`，CI 下界为
`0.2038789283`。Gate 内 11 条 spatial records 对应 4 个重复 asset 组；task/source
配额是 capacity-constrained，lexical metric 同时反映答案风格和词面重合，现有产物也
没有真实 finish reason。这些限制不事后改写 v1 判据；未来重跑须升级 v2，预注册完整
模型/tokenizer/runtime ledger、asset-component selection、task-aware metric、grouped
bootstrap 和 finish reason。

### 本次证据闭环验证

| 检查 | Exit | 结果 |
| --- | ---: | --- |
| 只读 `gate-b-verify` | 0 | selection、training、256 配对、指标、bootstrap、六判据和 8 个 artifact SHA 全部一致 |
| Phase 3 全量单元测试 | 0 | 45/45 |
| Phase 4 全量单元测试 | 0 | 72/72；含 verifier、public completed pass/fail、真实 exclusion、临时 training tamper 和 CUDA 错误窄分类 |
| 三份 `rs_vlm.config.v2` preflight | 0 | 3/3 metadata-only；临时 output roots 均未创建 |
| Python compile | 0 | `oa_groundrag/phase4/*.py`、phase4 CLI 和测试全部通过 |
| `git diff --check` | 0 | 无 whitespace 错误 |
| 正式 Gate B/训练产物不变 | 0 | 12 个文件的 size、mtime、SHA-256 前后完全一致 |

## Stage 0 当时未运行

- GPU、训练、正式评价或长时间任务
- OA-AuxSeg 恢复、test 或 predicted-mask 导出
- External LoRA 重训或 Base-vs-Adapter Gate B
- 源数据扫描、图像重编码或第二次 full deep validation
- OA-GroundedEval、Silver/Gold、RAG 或端到端集成
- 数据、模型、依赖或 PDF 下载
- commit 或 push

## Stage 1 人工定版当时未运行

- resume、optimizer step、backward、scheduler step 或任何训练
- checkpoint 保存、复制、重命名或 train log 裁剪
- test、Gate A、分割消融、多随机种子或正式 predicted-mask 导出
- Benchmark build、deep validation、payload 重算或源数据扫描
- 当时的 Stage 2–9 算法开发、外部下载、commit 或 push

## 本次身份绑定加固未运行

- 真实 Benchmark build、repackage、export、复制、records/assets 遍历或 payload validator
- GPU、训练、inference、生成评价、test、Gate A 或 Gate B
- OA-GroundedEval、Evidence Corpus、Silver/Gold、RAG 或 Stage 3–9 实施
- 既有 LoRA、checkpoint、training report、validation selection 或模型权重写入
- 数据、模型、依赖下载、commit 或 push

## 本次 Gate B 证据闭环未运行

- Base/Adapter 重新生成、GPU、训练、optimizer/backward 或 checkpoint 写入
- test、Gate A、deep validation、Benchmark payload/asset 重扫或 repackage
- Stage 4 Evidence Corpus/OA-GroundedEval 构建、Mask-Grounded 实验或 RAG
- 正式 Gate B artifact、training report、best pointer、checkpoint、Adapter 权重或
  Benchmark 修改
- 数据、模型、依赖下载、commit 或 push
