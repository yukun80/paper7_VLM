# REBUILD_PROGRESS

本文件是 OA-GroundRAG 的唯一实时状态源，只记录当前游标、冻结资产身份、科学边界、
最近一次有效验收和下一任务。稳定接口见 `README.md`，操作规则见 `AGENTS.md`，详细
算法见冻结设计 `docs/OA-GroundRAG_算法构建方案_0811.md`。

日期化实施日志、已解决的失败尝试、临时目录、旧命令和重复的“未运行”清单不再保留在
活动文档中。能力重构发布时的完整历史可从
`ac94fc1107b524f37dfbcf529cf4dc09bde27405:REBUILD_PROGRESS.md` 读取；该快照只用于
provenance，不代表当前入口或状态。

## 当前游标

| 字段 | 当前值 |
| --- | --- |
| 更新时间 | `2026-08-13` |
| program | `OA_GROUNDRAG_V3` |
| 当前工程状态 | `P0 / INSTRUCTION_ROUTED_UNIFIED_INFERENCE_CORE + BILINGUAL_UNIFIED_DEMO_WORKBENCH + INFERENCE_ONLY_PROVIDER_LOADING / engineering_complete` |
| 当前授权任务 | `FROZEN_EVAL_RUNTIME_DECOUPLING / engineering_complete` |
| 下一任务 | `P1 Multi-Source Grounded Evidence` |
| Git 基线 | 本轮开始于 `main@594ee334a4d73149c555fa9ce9d097ccc7f393d4`；当前为负责人授权的未提交工作树 |
| upstream 基线 | 本地 `origin/main` 与上述 HEAD 一致，`ahead=0 / behind=0` |
| 发行接口 | `oa-groundrag==0.2.0`；`oa-groundrag = oa_groundrag.runtime.cli:main` |
| 总体科学状态 | P0 与能力驱动重构仅为工程完成；Frozen Eval-dev 及其下游开发评价产物已退役，不升级 Gate A/C/D 或系统科学验收 |
| Benchmark test / sealed test | 未评价 / 未访问 |

本轮按负责人明确授权修复 Frozen Eval-dev 退役后的运行时间接依赖。Shared MLLM
现在只按 workflow state、best pointer、checkpoint manifest、Adapter 与当前
model/processor/LoRA topology 加载已发布权重，不再构造 compact training dataset、读 Eval
selection/HDF5 或恢复优化器状态。Text RAG 改用 Bank-only runtime 配置，不再绑定已删
development prediction/evaluation 根。Stage-5 后续配方改为 train-only；现有 compact/
collection 允许按退役 identity 浅验，要求重算历史 parent exclusion 时 fail closed。本轮
不恢复 Frozen 产物，不改模型数学、checkpoint、Adapter、Bank 或 Unified wire schema；未运行训练、
Gate、正式评价或 test/sealed 访问，未 commit/push。

## 负责人授权的评价链退役

负责人已明确授权从 Unified Demo 退役 Frozen Evaluation，并删除可由保留代码、配置与上游数据
重建的对应开发评价链。下表是删除前完成的精确路径、普通目录、无 symlink、大小与 manifest
身份核验。所有列出的目录现均为 `DELETED / ABSENT_VERIFIED`；删除不可直接恢复，只能用保留的
代码、配置与上游数据重建。该授权未扩张到未列出的 Benchmark、训练、模型、Text Bank 或正式代码资产。

| 待退役目录 | 删除前大小 | 删除前 identity | 状态 |
| --- | ---: | --- | --- |
| `../benchmark/oa_grounded_stage4_v1/eval_dev/oa_grounded_eval_dev_v1_100` | 28M | manifest `fed7d8b99e4482da1a9e8553c2779cd64007a710fa10404bf2985e96f1ce7492` | `DELETED / ABSENT_VERIFIED` |
| Region Adapter 根下 `base_gt_mask_baseline` | 148K | prediction manifest `6216332b3fef86b0dab751e5237d3524db7927cd5836e8f7c991080b003233af`；evaluation manifest `e4029d21372251766d616c512b3d06c75eb356df7b83e04e445a392493eb0c93` | `DELETED / ABSENT_VERIFIED` |
| Region Adapter 根下 `rs_general_adapter_gt_mask_baseline` | 152K | prediction manifest `dfab33e7b53bb8e7039128cd971c3ee19cc072e211715b5ef2355a753616f7e1`；evaluation manifest `696b369fe1840eea2ec172d259ab7f85649fef65a921decc1c02c047b8ab31c0` | `DELETED / ABSENT_VERIFIED` |
| Region Adapter 根下 `mask_grounded_region_adapter_gt_mask` | 1020K | prediction manifest `4b090b4392a906817379357d1a8295f8b5eea10339eeb610847bc9bd2ef26a6b`；evaluation manifest `8629114aa22fb8b39b9346943f6100b157428c903e295bb67dfb4dfb603e8448` | `DELETED / ABSENT_VERIFIED` |
| `outputs/stage6_text_rag/dev_retrieval_v1` | 2.0M | manifest `da0e207b284bee81824b3542efb8a4b19138c92f07ba09e0afc3f11c4c0b9e7c` | `DELETED / ABSENT_VERIFIED` |
| `outputs/stage6_text_rag/pass2_gpu_smoke_v1` | 56K | manifest `987b6f7601e4f16632f9c07d8901242717a6551ed3b6758819eb44d0b097a6eb` | `DELETED / ABSENT_VERIFIED` |
| `outputs/stage6_text_rag/gate_d_dev_protocol_v1` | 56K | manifest `8ddf9829c38ae8ab7d583a47f7ba1e4bdf5be0b4dc200618df63669159fa3cd5` | `DELETED / ABSENT_VERIFIED` |
| `outputs/stage6_text_rag/gate_d_dev_25pairs_v1` | 200K | manifest `170ad8acaadc626e224b3a93e9a7e1758f78d07438b9adcf9ad60027f49673b5` | `DELETED / ABSENT_VERIFIED` |
| `outputs/stage6_text_rag/gate_d_dev_auto_eval_v1` | 136K | manifest `85a0efdda16e582b768128a16348bda22448639cf1c4694dbe37d64e53fc22f6` | `DELETED / ABSENT_VERIFIED` |
| Demo run `demo_20260813T040242784502Z_0fcb1828fc9e488bb2098758da894bc2` | 1.9M | run manifest `9440abb176c6e0bbc7348525af85dd287e90433e4b9d811bc9e118ce19c3f049`；重新枚举确认唯一 Frozen run | `DELETED / ABSENT_VERIFIED` |

## 权威与发布基线

| 项目 | 冻结值 |
| --- | --- |
| 冻结算法设计 | `docs/OA-GroundRAG_算法构建方案_0811.md` |
| 设计基线提交 | `087ae4b438a26cc0bdcd3c453b339bccadcc9e85` |
| 设计 SHA-256 | `fd088b0a25b3fc8888e7b4c07971ef36858c784ffcfaa735219e8e8514251243` |
| P0 unified runtime 提交 | `6d9cd816495f79bf9b13263d9725d6e159fe448b` |
| 能力路径迁移提交 | `b784c746c7749783739f21e3e810012ac493bd6b` |
| 能力重构发布提交 | `ac94fc1107b524f37dfbcf529cf4dc09bde27405` |
| Unified runtime config | `configs/runtime/inference_v2.yaml`；SHA-256 `2cca2a111249be033385348d7e2faf6278e0d39f0ac497e185bbcd3097215979` |
| Unified Demo config | `configs/runtime/demo_v1.yaml`；SHA-256 `540ee12c38130ea20513af61da76cf33b9292f62a7c7ee272f261aab64b5e573`；`frozen_evaluations=[]` |
| Grounded runtime config | `configs/vlm/grounded/runtime_v1.yaml`；SHA-256 `329d1cc57443174dc989a63d58c9613ab79532704c9e73ec0457309a022ab48e` |
| Grounded train-only config | `configs/vlm/grounded/train_v2.yaml`；SHA-256 `16fcfff8ba8ebd8becca48fa57d5711e58a20de25a83f04d1cb92aa6920a52d4` |
| Text RAG runtime config | `configs/retrieval/runtime_v1.yaml`；SHA-256 `7e732f24d4eb9061465e4ad9ed866ef88869a30943001df9499aea4c65367f33` |

冻结设计和 `docs/archive/` 只读。README 不保存动态 SHA、checkpoint 或完成状态；
AGENTS 不保存动态运行结果。

## 当前能力状态

| 能力 | 稳定代码位置 | 当前结果 | 尚未越过的边界 |
| --- | --- | --- | --- |
| Spatial Perception / OA-AuxSeg | `oa_groundrag/segmentation`；`oa_groundrag/training/segmentation`；`oa_groundrag/evaluation/segmentation.py` | full Benchmark 与负责人定版 checkpoint 可用 | Gate A、正式 fixed predicted masks、sealed test 均未执行 |
| Shared RS-Geohazard MLLM | `oa_groundrag/vlm`；`oa_groundrag/data/rs_general`；`oa_groundrag/training/vlm` | RS-GeneralDesc native v1、step-1000 Adapter 与 Gate B 冻结证据可用 | Gate B 只接受其冻结 RS-GeneralDesc 作用域 |
| Grounded Multimodal Understanding | `oa_groundrag/grounding`；`oa_groundrag/data/grounded`；`oa_groundrag/training/grounding` | train-only Corpus、compact supervision、step-900 Region Adapter 及评价 reader/实现保留；推理只验证已发布 checkpoint/Adapter 身份；Eval-dev 实例与默认重建配方已退役 | 当前无 materialized OA-GroundedEval-dev；Gate C、专家共识和正式 OA-GroundedEval 未完成 |
| Knowledge Augmentation | `oa_groundrag/retrieval`；`oa_groundrag/evaluation/retrieval` | Text Bank、Bank-only runtime retrieval/Pass-2 与评价实现保留；旧 development retrieval/Pass-2/Gate-D artifacts 与默认 dev 配置已退役 | 当前无 materialized Gate-D development evaluation；无 retrieval Gold、专家盲评或正式阈值；Gate D 未科学通过 |
| Unified Inference | `oa_groundrag/runtime` | 六类显式任务、确定性 router、lazy provider 与双语只读 Demo 工程完成；Shared MLLM/Text RAG provider 已与退役 Eval/dev outputs 解耦；纯 `KNOWLEDGE_QA` 不消费 Benchmark payload，candidate 解释需人工显式选择 | auxiliary preview 不进入当前 P0 MLLM formal grounded input；Demo selection 仅作 qualitative 展示；test 默认锁定；P1 多源 grounded evidence 与统一科学评价均未开始 |

Stage 只保留为 curriculum、schema、output root、checkpoint metadata 和历史 provenance；
活动源码、配置、脚本和测试按能力与工程职责组织。

## 冻结与发布资产身份

下表只保留消费方定位资产所需的根和 identity anchor。更细的统计、ledger、环境和指标以
各正式 manifest/report 为准，不在本文件重复抄录。

### Spatial Perception

| 资产 | 根或文件 | Identity |
| --- | --- | --- |
| OA-AuxSeg full Benchmark | `../benchmark/oa_auxseg_hdf5_v1/full` | manifest `9a3b1478ed844f234e32b839fded67a937c49d202e3d8f5efd7db52596b5a00a`；index `389877226249d2477bdda62d937950339e9fa60df35558b945d02757e8d0da42` |
| OA-AuxSeg final checkpoint | `outputs/phase2_oa_auxseg/full_proposed_dropout_v6_b16_nockpt_e100/checkpoint_best.pt` | SHA-256 `672d39ab4220d8e1b4f949ca8d1d5dcd34f58898cecd1553dd56cdd9d84fb038`；step `206820` |
| OA-AuxSeg finalization report | 同一输出根的 `training_report.json` | SHA-256 `c92ea5e96152f3bc4f67399396a2f1495ce917968d3b8b6c5a516cf377de42b5`；`project_owner_manual_stop` |

Benchmark 共 `53,645` 条（train/val/test 为 `36,761/12,375/4,509`）。checkpoint 是
负责人定版权重，但 `formal_acceptance=false`、`gate_a_evaluated=false`，不得称为
Gate-A-accepted checkpoint。

### Shared MLLM 与 Gate B

| 资产 | 根 | Identity |
| --- | --- | --- |
| RS-GeneralDesc native v1 | `../benchmark/rs_generaldesc_v1` | manifest `36875efb59619cf10fa614ce5995afb88f7d7888624267f7ed216f29ae385832`；payload `549281f296b357bce256e6af71cec7412fe17e36052d6a8674f4876ae2d06e0b`；hash manifest `55ac26d9771ce8385318fbd23a10b999afb754ac195e823be659f4e49b0a7090` |
| RS-General Adapter | `outputs/phase4_rs_vlm/rs_vlm_lora_qwen3vl_2b_b1a16` | report `a4f42e777eaab6e444f04d63b89f482ee31a077bf13006d587863bfa4fb1eb1e`；best pointer `4d93e2c6c34fe01a10db373c00946166238b8d132bb32b9863b52e305b6f4db6`；step-1000 manifest `aa279659a4c563536f1d7554ed9e51643398365ce11539eb9865f77c1d3a621f`；Adapter `a367e39c626338a151dad33e6f7a7f9cc9887206dbcd261d147837e6408becc1` |

RS-GeneralDesc 包含 `274,693` records / `104,954` parents，保存的 deep validation 为
`0 error / 0 warning`。Adapter training report 本身不作科学接受；Gate B 由独立冻结协议
接受。

Gate B 正式根：
`outputs/phase4_rs_vlm/rs_generaldesc_gate_b_qwen3vl_2b_v1`。

| Gate B 文件 | SHA-256 |
| --- | --- |
| `selection/gate_b_protocol.json` | `8378f6f107849439be3b402b0014df4007b10589e93602c1afb99173c2fb2c54` |
| `selection/gate_b_selection.json` | `98290aaa585b798dcc5a30b9a4d47083e778aa4d48980e24bbce647705b915bd` |
| `base/generation_manifest.json` | `bad680291426f65fd51dfaa35eca649968e478af963d3c5009cdbce733a699fb` |
| `base/predictions.jsonl` | `862759c44400552f40f5211a38ceafd8d1f4712c7d6f870e3a2f0676d8ce8bd6` |
| `adapter/generation_manifest.json` | `c69314727ba71ef712fd7bbde1990ebd610e6759714641103916adf901787168` |
| `adapter/predictions.jsonl` | `a7c791fcfe8f3b94f6b188780bc0aed46dbf2fcf92f1bcca3b6617ca1c4bd98a` |
| `evaluation/paired_scores.jsonl` | `64d6802e2b2438305fa7ba560bc4d90ebb4537d1048dc281333c707fb4d3975f` |
| `evaluation/gate_b_report.json` | `b150de8eeed07c5cb3e9c808e7cec5c32f29c23fca9dd82bf7842786d89eb165` |

Gate B 的 `formal_acceptance=true` 只绑定原协议、产物和验收提交。能力重构后的严格 verifier
因 implementation SHA 漂移如实返回 `GATE_B_PROTOCOL_INVALID`；不建立兼容伪协议，也不
据此抹除原提交上的历史 PASS。

### Grounded data 与 Region Adapter

| 资产 | 根 | Manifest SHA-256 / 状态 |
| --- | --- | --- |
| deterministic Auto Pilot | `outputs/stage4_landslide_evidence/landslide_evidence_corpus_v1_pilot_500` | `37aebb9c5f8ceb720e0a1a3c8621212d44562fa6b6786d145c31e11ffa94f9bb`；500 train-only records |
| Region Corpus base | `../benchmark/oa_grounded_stage4_v1/region_corpus/mask_grounded_region_corpus_train_v1_500` | `d18e6b4f3ab566447131ecd6fa45eb21b7675a582e85579ef8320289093ec32e` |
| Region Corpus extension | `../benchmark/oa_grounded_stage4_v2/region_corpus/mask_grounded_region_corpus_train_extension_v2_7950` | `a26f4267ba12fad8ac39481dcd16dd40a65dacdec7133453847cb2e2c71d43fe` |
| Region collection | `../benchmark/oa_grounded_stage4_v2/region_collection/mask_grounded_region_train_collection_v2_8450` | `cd2b86f6244f4f5f42d846166f11a34efdb9edd636239039b42444c453e435d2` |
| compact training messages | `../benchmark/oa_grounded_stage4_v2/training_messages/mask_grounded_region_compact_training_messages_train_v3_6974` | `746f641f1fbe48f4301ffc0c52b586437a1dc0b68a5add4be1e3db50d69a1184`；6,974 mixed-supervision records |

Region Adapter 正式根：
`outputs/phase4_rs_vlm/mask_grounded_region_lora_qwen3vl_2b_rsinit_v1`。

| Region Adapter anchor | SHA-256 |
| --- | --- |
| `training/training_report.json` | `c0751415a2da4ec72c92892975ab713d688aa316a1480b7a82b8ce8e9d5916ab` |
| `training/best_checkpoint.json` | `368bc48fec6e5303c80e8b2c0d397f4d55eb8c6d51fe70d9da417833ac8a2c1b` |
| step-900 checkpoint manifest | `c203823597c7b3ecf7f1bce9b3030efe7f5ef2dc5c1ae58bbc58e0982aae5c30` |
| step-900 Adapter | `858e12ff7e902ce0a3fdfb1a3dfbc2e58ad0892dec870a73fa4fc0a3411f84d7` |
| `workflow_state.json` | `9c5a90743e26576f55a83157e8a1bd3fcf28c1005cc97ff098be1a3b02a62efa` |

该 Adapter、训练状态和 retention 资产保持不变；旧 OA-GroundedEval-dev prediction/evaluation
实例已在上方退役表记录身份后删除，不再作为当前可消费产物。workflow 固定
`formal_acceptance=false / scientific_acceptance=false / sealed_test_evaluated=false`。

### Knowledge Augmentation

| 资产 | 根 | Identity anchor |
| --- | --- | --- |
| Text Evidence Bank v1 | `outputs/stage6_text_rag/text_evidence_bank_v1` | Bank ID `9322a9139d04be7665feb154153b7dc1c2d35b0871fc32bbd6a6daa942fabb28`；manifest `9b891e191581746173a27b80356caa18ec9be5d3c36eaee67444b05a070f0bcc`；ledger `1c73fbd6135daaaf4767f8427e7c9e1ba69f09e0223aaad2c3e151a013c1e650` |

Text Bank 固定消费 `docs/RAG_knowledge/` 的 12 个 PDF；正式索引 1,283 units
（interpretation/confounder/limitation 为 `872/138/273`）。在线 Text RAG 能力和 Bank 身份保留，
旧 development retrieval、Pass-2 smoke 和 Gate-D artifacts 已删除；当前没有 Gate-D 科学结果，
unsupported-claim 人工率、专家相关性、Recall@K、MRR、nDCG 和 `gate_d_pass` 均未建立。

## 科学验收与不可扩张边界

| Gate / 边界 | 当前状态 | 允许的结论 |
| --- | --- | --- |
| Gate A | 未执行 | OA-AuxSeg 只有工程定版 checkpoint；不得导出或宣称正式 fixed predicted masks |
| Gate B | 原冻结作用域正式通过 | 只证明 RS-GeneralDesc native v1 固定协议下 Adapter 相对提升 |
| Gate C | 未执行 | Region Adapter checkpoint 可用，但已无当前 Eval-dev artifact；不能证明模型正确依赖 mask |
| Gate D | 未执行；旧 development artifacts 已退役 | 只能报告在线 RAG 工程合同；不能声称 RAG 有科学增益 |
| Gate E / F | 条件未触发 | 不启动 Landslide-Evidence Adapter；不宣称 retention 通过 |
| sealed test | 未访问 | 任何阈值、协议或实现均不得从 test 反推 |

持续有效的数据与事实边界：

- Corpus 和 supervision 只使用 train；如按保留配方重建 OA-GroundedEval-dev，只能使用 val；
  test/sealed test 不得进入 Corpus、开发阈值或当前报告。
- 6,974 条 compact supervision 是 `mixed_model_and_single_expert`，不是 Gold 或专家共识。
- OA-AuxSeg 当前辅助 registry 只接受 `dem / insar_velocity / slope`；SAR 尚未接入。
- encoded InSAR 的物理单位与 sign convention 未确认时，只能作为 encoded evidence；不得
  生成定量位移或物理方向结论。
- LMHLD 和 Landslide4Sense 缺少可靠地理 group；不得从文件名或 sample ID 伪造空间身份。
- Text RAG 不生成或修改 mask，不改写 Programmatic Facts 或 Pass-1 observation，不把
  候选区域升级为确认滑坡。
- Stage 7 Case RAG 只在正式 Gate D 证明稳定增益后考虑；Stage 8
  Landslide-Evidence Adapter 只在 Gate E 失败时考虑。

## 最近一次有效工程验收

能力驱动重构基线仍保留；Unified Demo Workbench 的当前有效工程证据如下。

| 检查 | 最近结果 |
| --- | --- |
| 全量回归基线 | 能力重构时收集 339 项：331 passed，4 项因缺少 `oa_auxseg_hdf5_v1/small` skip，4 项含 backward/optimizer step 仅收集未执行；本轮未重跑训练相关全集 |
| Runtime / Demo CPU | `tests/runtime` 63/63；新增实测 inference-only Shared MLLM loader 不调用 compact/Eval reader/Benchmark `__getitem__`/HDF5，checkpoint/Adapter/Bank SHA 漂移 fail closed，六任务配置在 Eval-dev 缺失时仍可预检；既有 Demo、i18n、candidate、test lock 和 user-mask 合同继续通过 |
| Stage-5 / retrieval / data | `tests/training/grounding` 11/11、`tests/retrieval` 24/24、`tests/data/grounded` 71/71；train-only config/workflow 无 Eval 阶段，compact/collection 退役 identity 浅验成功，要求重算历史 exclusion 时明确失败 |
| Architecture | 8/8；活动配置可解析，已删 Eval/dev 路径不再出现于活动 YAML/JSON，package import graph 与薄 CLI 保持稳定 |
| Gradio smoke | Blocks 构建由 Runtime/Demo tests 覆盖；最终代码在 `127.0.0.1:8977` 成功启动并关闭，`share=false`，allowed path 仅 `outputs/demo/unified_workbench_v1` |
| compile / CLI / diff | `compileall`、Demo/Unified/Text-RAG/Grounded/Gate-D CLI help、`git diff --check` 通过；development retrieval/Gate-D CLI 要求显式 `--config`，不再带退役默认值 |
| 真实输入预览 | 固定 val `landslide4sense::landslide4sense_000002` 从同一 raw sample 展示 B01–B12、DEM 与 slope；逐通道 raw min/max、valid fraction 和 transform 可审计，且未进入 MLLM formal input |
| CUDA bounded Demo | 固定 val `gdcld::gdcld_val_original_00002`：`VLM_ONLY=SUCCESS`、`SEGMENT_AND_UNDERSTAND=SUCCESS`、首次 RI=`WAITING_FOR_CANDIDATE`；本次真实 candidates `[0,1,2,3]`，candidate 0 replay 未重跑 OA-AuxSeg，Pass-1、6-item retrieval、Pass-2 与 6 citations 全部成功；CUDA peak `4,684,341,760` bytes，`sealed_test_accessed=false` |
| 删除与保护资产 | 退役配置 5 份已删，新增 Grounded runtime/train-only 与 Text RAG runtime 配置；Benchmark manifest/index、OA-AuxSeg checkpoint、Region Adapter checkpoint manifest/Adapter/workflow、Text Bank manifest/ledger 实施前后 SHA-256 不变；不修改历史 output provenance |
| Ruff | 未安装；准确结果为 `No module named ruff` |

Unified Demo 的上述 CPU、UI 与 CUDA smoke 均为 engineering validation，不构成 Gate A/C/D、
正式 test evaluation、scientific evaluation 或 scientific acceptance。`allow_test_demo=false`，
真实 test 未读取，Demo root 中不存在 test access receipt；Frozen Evaluation 页签、selection、
推理入口和文件白名单均已移除，`frozen_evaluations=[]`。Demo run/viewer 使用 Demo-only v2
sidecar，正式 `UnifiedRequest/UnifiedResponse v1` 与底层
`candidate missing → global fallback` 合同未修改。

## 下一任务与授权

下一任务仍为 `P1 Multi-Source Grounded Evidence`；本轮 auxiliary 可视化只展示现有 Spatial
Expert input，没有让当前 MLLM 消费多源输入，也没有开始 P1。

开始前必须：

1. 重新读取本文件、AGENTS 和冻结设计，并核对现场 Git、活动进程与受保护资产。
2. 由项目负责人明确授权本次写入、GPU inference 或评价边界。
3. 保持现有模型数学、checkpoint、Benchmark、Text Evidence Bank 和正式 outputs 只读。
4. 不访问 sealed test，不运行 Gate A/C/D，不启动 Case RAG 或新增训练。

## 维护规则

- 新状态直接更新现有游标、能力表、资产表和边界表，不再按日期追加实施日记。
- 新正式资产只记录稳定根、manifest/report identity 和科学身份；细节留在资产自身
  manifest、ledger、report 和 Git 提交中。
- 已解决的失败、临时 PID、scratch 路径和一次性命令不进入活动文档；仍构成当前 blocker
  时才保留最短必要说明。
- 历史 Stage/phase 名可以保留在 schema、output root、checkpoint metadata 和 Git
  provenance 中，但不得重新成为活动源码、配置、脚本或测试的组织方式。
- 不新增重复的 archive、handoff、audit 或 worklog 文档。
