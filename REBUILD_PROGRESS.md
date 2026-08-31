# REBUILD_PROGRESS

本文件是 OA-GroundRAG 的唯一实时状态源，只记录当前游标、冻结资产身份、科学边界、
最近一次有效验收和下一任务。稳定接口见 `README.md`，操作规则见 `AGENTS.md`，详细
算法见冻结设计 `docs/OA-GroundRAG_算法构建方案_0829.md`。

日期化实施日志、已解决的失败尝试、临时目录、旧命令和重复的“未运行”清单不再保留在
活动文档中。能力重构发布时的完整历史可从
`ac94fc1107b524f37dfbcf529cf4dc09bde27405:REBUILD_PROGRESS.md` 读取；该快照只用于
provenance，不代表当前入口或状态。

## 当前游标

| 字段 | 当前值 |
| --- | --- |
| 更新时间 | `2026-08-30` |
| program | `OA_GROUNDRAG_V3` |
| 当前工程状态 | `QWEN3_5_4B_DUAL_BACKEND / M7_RESOURCE_GATE_FAILED / SERIAL_STOPPED` |
| 当前授权任务 | `QWEN3_5_4B_DUAL_BACKEND_SERIAL_M0_M10 / authorized` |
| 下一任务 | `无自动下一模块；M7 的 20-step smoke 已真实 OOM，等待负责人重新 grill 并明确新的实验授权` |
| Git 基线 | 本轮开始于 `main@95fff178fd976aff53589688f171a8ee2a695362`；当前为负责人授权的未提交工作树 |
| upstream 基线 | 本轮开始时本地 `origin/main` 与上述 HEAD 一致，`ahead=0 / behind=0` |
| 发行接口 | `oa-groundrag==0.2.0`；`oa-groundrag = oa_groundrag.runtime.cli:main` |
| 总体科学状态 | P0 与能力驱动重构仅为工程完成；Frozen Eval-dev 及其下游开发评价产物已退役，不升级 Gate A/C/D 或系统科学验收 |
| Benchmark test / sealed test | 未评价 / 未访问 |

本轮负责人已明确授权按新冻结设计串行实施 Qwen3.5-4B 双后端链，包括新设计/代码、
固定 revision 模型下载、GPU smoke、两段 1000-step 训练、独立 Gate B、全新 100 条
val-only Grounded 评价以及 README/本文件更新。每个模块必须通过测试后才进入下一模块；
任何资源、合同、Gate 或零容忍评价失败立即停止，不自动采用 QLoRA、缩减输入、安装
kernel 或 hybrid LoRA。既有 2B、Benchmark、checkpoint、outputs 和 models_zoo 资产保持
只读，不访问 test/sealed，不 commit/push。

此前负责人授权修复 Frozen Eval-dev 退役后的运行时间接依赖。Shared MLLM
现在只按 workflow state、best pointer、checkpoint manifest、Adapter 与当前
model/processor/LoRA topology 加载已发布权重，不再构造 compact training dataset、读 Eval
selection/HDF5 或恢复优化器状态。Text RAG 改用 Bank-only runtime 配置，不再绑定已删
development prediction/evaluation 根。Stage-5 后续配方改为 train-only；现有 compact/
collection 允许按退役 identity 浅验，要求重算历史 parent exclusion 时 fail closed。本轮
该此前修复任务不恢复 Frozen 产物，也未改模型数学、checkpoint、Adapter、Bank 或 Unified
wire schema；其边界不用于描述本轮已授权的 Qwen3.5 训练。当前仍未访问 test/sealed，
未 commit/push。

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
| 冻结算法设计 | `docs/OA-GroundRAG_算法构建方案_0829.md`（v3.1，SHA-256 `1eea63b99b840847440df667899ab26b85faf9f0ea05a947dbcd6a5b67cdacc9`） |
| 历史冻结设计 | `docs/OA-GroundRAG_算法构建方案_0811.md`；只读 provenance |
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

## Qwen3.5-4B 双后端串行实施

| 模块 | 当前状态 | 已验证证据 |
| --- | --- | --- |
| M0 设计与保护基线 | `COMPLETE` | 新 v3.1 设计已发布；0811 SHA 保持 `fd088b0...1243`；本轮起点 `main@95fff178...5362` 与 `origin/main` 一致、工作树起始干净；未发现训练/评价/下载进程；保护根与既有 2B/Gate B 资产存在；冻结 runtime/config SHA 复算一致；`git diff --check` 通过 |
| M1 通用后端合同 | `COMPLETE` | `VLMBackendSpec`、registry/factory 与 `rs_vlm.config.v3` 已实现；未知/重复 backend fail closed；v2 三个语义 SHA 不漂移。新增/相关回归：VLM 46、training/vlm 6、training/grounding 11、evaluation/rs_general 29、runtime 63、architecture 8，全部通过；compileall 与 diff check 通过。静态导入环在本模块门内被发现并消除后才验收 |
| M2 2B 后端迁移 | `COMPLETE` | 2B model/processor/constants 已物理分离；活动 CLI、RS-General/Gate B、Grounded train/runtime、VLM smoke 与 Gate D processor 均经 factory，公共 facade 无旧具体类 alias。冻结 step-1000 inference-only Adapter 实读：基模 `2,127,532,032` 参数、LoRA `3,211,264` 参数/224 tensors；既有配置、模型、Adapter、Gate B 与 Grounded outputs 无 diff。相关回归共 198 项通过，另有 compileall、架构、配置 SHA 与 diff check 通过 |
| M3 4B 资产与 processor | `COMPLETE` | 官方 `Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` 的 14 个文件已下载到全新普通目录并经 Hub strict verify；生产 ledger file/payload SHA 为 `ca48af...ab2f` / `233557...b1ce`，总大小 `9,342,907,469` bytes、2 个完整分片。真实 offline factory 验证文本、单图、三图、assistant-only labels、Jinja thinking 切换及 `mm_token_type_ids` 等 5 个 tensor keys；篡改、缺失、额外文件、symlink、分片索引及覆盖拒绝测试通过。相关回归 204 项通过；2B processor/checkpoint 身份和 `3,211,264` 参数保持不变；compileall 与 diff check 通过 |
| M4 4B model 与资源门 | `COMPLETE` | 使用 `AutoModelForMultimodalLM`，严格锁定 full-attention 层 `3/7/11/15/19/23/27/31` 的 q/k/v/o LoRA：64 tensors、`1,572,864` 参数；vision/merger 冻结，跨家族与 hybrid LoRA 拒绝。官方磁盘完整参数 `4,659,865,088`，标准 AutoModel 运行时唯一参数 `4,539,265,536`；差异精确绑定 15 个 `mtp.*` tensors / `120,599,552` 参数及 tied `lm_head` 视图，任何其他映射 fail closed。RTX 4090 D（`25,756,696,576` bytes）真实 base 推理峰值 `9,161,038,848` bytes；LoRA step-1/step-20 峰值 `10,082,741,760` / `10,103,844,352` bytes，20-step 无 OOM。相关回归 209 项及显式 CUDA gate 1 项通过；compileall/diff check 通过；缺少可选 FLA/causal-conv kernel 时按方案使用原生 PyTorch fallback，未安装 kernel |
| M5 4B RS-General 训练 | `COMPLETE` | prompt-only、bounded LoRA 与正式 LoRA 三份 v3 配置已严格解析，2B v2 semantic SHA 不变；bounded/formal Benchmark preflight 通过。bounded smoke 完成 batch1×acc16 的 1 个 optimizer step。正式 batch1×acc16、BF16/SDPA、non-thinking LoRA 训练在全新根完成 `1000/1000` steps，16,000 samples / 4,974,722 input tokens / 20,571 images，CUDA peak `11.020907878875732 GiB`；10 次同模型 val monitor 后 step 1000 为 best，macro/overall loss=`0.8256946369626602 / 1.0491456486446964`。训练 report/run manifest/best pointer/step-1000 manifest/Adapter SHA 分别为 `cf3f13...fec0` / `d03ccf...9de` / `9ee78b...a5e` / `22cc39...f98c` / `0be1b6...f6d5`；64 LoRA tensors、`1,572,864` 参数，`formal_acceptance=false`。真实产物严格 JSON/JSONL、选择规则、逐文件 SHA 与 symlink 审计通过；相关回归 210 项、compileall 和 diff check 通过 |
| M6 4B 独立 Gate B | `COMPLETE` | 新 v2 协议精确复用原 2B 的 256 个 val record，逐项 identity 与 ordered record SHA 一致，并绑定 4B backend/revision/ledger/config/step-1000 Adapter。RTX 4090 D 单记录 base/best-Adapter CUDA 预检通过，峰值分别为 `9,282,672,640 / 9,287,720,960` bytes；正式 base 与 Adapter 均为 `256/256`、0 failures、70,875 input tokens / 338 images。六项判据全部 PASS：primary macro bootstrap 95% CI 下界 `0.23548574735894734`、7 个任务改善、最差 task/source primary delta=`0.0564974355534128 / 0.19484760058784772`、open-generation Rouge-L delta=`0.3085196745470501`、short-answer EM delta=`0.09923780487804876`；严格只读 verifier 返回 `accepted / formal_acceptance=true`。2B/4B 256 条配对报告固定 `report_only=true / promotion_criterion_used=false / scientific_superiority_claim_supported=false`。相关回归 215 项及显式 CUDA smoke 1 项通过；compileall、架构与 diff check 通过；旧 2B Gate B 八个冻结 SHA 不变 |
| M7 4B Grounded 训练 | `STOPPED / RESOURCE_GATE_FAILED` | v3 配置、同家族 warm-start/retention 合同、跨家族拒绝和 bounded 1/20-step 严格恢复已实现；M7 合同 16/16、VLM 58/58、架构 9/9、Grounded evaluator 1/1、compileall/CLI/diff check 通过。RTX 4090 D 上 1-step smoke 完成：loss `0.8616053573787212`、16 samples / 23,785 input tokens / 46 images、64 tensors / `1,572,864` LoRA 参数、allocator peak `14.034749984741211 GiB`。从唯一 step-1 checkpoint 恢复到 step 20 时，设备巡检已达 `23,978 / 24,564 MiB`，随后在 backward 报 `CUDA out of memory` 并以 exit `134` 终止；没有任何 step 2+ 日志、trace 或 checkpoint，正式训练根未创建。按冻结方案立即停止，未改 QLoRA、输入、kernel 或 LoRA 拓扑 |
| M8–M10 | `NOT_STARTED / BLOCKED_BY_M7` | 未创建 4B Grounded Runtime/Demo、100 条 val-only Grounded 资产或默认发布；未访问 test/sealed |

## 当前能力状态

| 能力 | 稳定代码位置 | 当前结果 | 尚未越过的边界 |
| --- | --- | --- | --- |
| Spatial Perception / OA-AuxSeg | `oa_groundrag/segmentation`；`oa_groundrag/training/segmentation`；`oa_groundrag/evaluation/segmentation.py` | full Benchmark 与负责人定版 checkpoint 可用 | Gate A、正式 fixed predicted masks、sealed test 均未执行 |
| Shared RS-Geohazard MLLM | `oa_groundrag/vlm`；`oa_groundrag/data/rs_general`；`oa_groundrag/training/vlm` | RS-GeneralDesc native v1、冻结 2B step-1000 Adapter/Gate B，以及新 4B step-1000 RS-General Adapter/独立 Gate B 可用 | 4B Gate B 只接受其 RS-GeneralDesc 冻结作用域；4B Mask-Grounded 20-step 资源门失败，Adapter、Runtime 与 Grounded 绝对验证均未完成 |
| Grounded Multimodal Understanding | `oa_groundrag/grounding`；`oa_groundrag/data/grounded`；`oa_groundrag/training/grounding` | train-only Corpus、compact supervision、step-900 2B Region Adapter 及评价 reader/实现保留；4B v3 训练合同与 step-1 smoke 证据存在，但未形成可发布 Adapter；Eval-dev 实例与默认重建配方已退役 | 4B 20-step smoke 在 backward OOM 后串行停止；当前无 materialized OA-GroundedEval-dev，Gate C、专家共识和正式 OA-GroundedEval 未完成 |
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
| Qwen3.5-4B base candidate | `models_zoo/Qwen3.5-4B-r851bf6e8` | Hub commit `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`；asset ledger `configs/vlm/assets/qwen3_5_4b_r851bf6e8.asset_ledger.json` SHA-256 `ca48af5142c25796898f0f9c44a6720d0656747714565544341f1733ea9dab2f`；payload `23355763f89e7f03e6dc756fede244c5d6761360b065308b2adcbffeb1e5b1ce`；M4 24 GB 工程资源门通过；基模自身不作 Adapter/Gate B/科学接受声明 |
| Qwen3.5-4B RS-General Adapter | `outputs/phase4_rs_vlm/rs_vlm_lora_qwen35_4b_r851bf6e8_b1a16_v1` | config semantic `f2e3a3dc05b5152ddda64e1e88d765c4825b128021d355b9425b35d53220b843`；report `cf3f135bf452a365907c3a4bd3f4d9d07a0f0031be5cab7c01d0ed93401bfec0`；best pointer `9ee78b2c239321ac983ea94e76d254f083e093aebe84928c55438baf8d106a5e`；step-1000 manifest `22cc396e89a8ba84843657b8a9272acc8bcaced7de475c2a831370580cccf98c`；Adapter `0be1b656d5646073c54183837e43cbf0c2c45f24d1c9652f4b7a9ccd9d89f6d5`；training report 自身 `formal_acceptance=false`，独立 4B Gate B 已通过 |

RS-GeneralDesc 包含 `274,693` records / `104,954` parents，保存的 deep validation 为
`0 error / 0 warning`。Adapter training report 本身不作科学接受；Gate B 由独立冻结协议
接受。

Qwen3-VL-2B Gate B 正式根：
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

Qwen3.5-4B Gate B 正式根：
`outputs/phase4_rs_vlm/rs_generaldesc_gate_b_qwen35_4b_r851bf6e8_v1`。

| Qwen3.5-4B Gate B 文件 | SHA-256 |
| --- | --- |
| `selection/gate_b_protocol.json` | `7c3b94ffc473460f1a1d6652d05eb448d3a5e14ba7ed4eea96a5df423e9f9ba5` |
| `selection/gate_b_selection.json` | `f67b169c1d40cd42ac8591909e1eee49a92ff7179325daf52a67866386c637b1` |
| `base/generation_manifest.json` | `6be25639646fb2aaa2a066dd30f6018ae4351b8845171d5ac8ee39bdfda64fc1` |
| `base/predictions.jsonl` | `87e94fdece610706bb59975c245372885df6251f1f0b615d208fce96bac22687` |
| `adapter/generation_manifest.json` | `cbdaf59efd38e3427c4f2df31e96c18ebf1dea0e65f392ecb7690708c2ad7b1b` |
| `adapter/predictions.jsonl` | `70d730f42fbe8a772b5a6af7e53fdcd2e8ae9e9e0fdceb2bcb41578e972d00be` |
| `evaluation/paired_scores.jsonl` | `c728fd7f2bc64d2783c197c5bc28aea8899e2bc8ba4432a40d1a52668c0d9fc9` |
| `evaluation/gate_b_report.json` | `b7c144eee000222b0556838e5bc17ce14dc409ec7704fa51fb0a4e23ad013cd9` |
| `family_comparison/paired_scores.jsonl` | `cd35ef0161545a606420c04cac8d32efb957e81607f37d8e3f5a1cfc87368fca` |
| `family_comparison/family_comparison_report.json` | `7322eb107b3b1b8fc0f5c713fc9eff6d050fb914d8d6ee9052f0ef1eb0287da5` |

4B Gate B 的 protocol/selection canonical SHA 分别为
`424a3f223a8d30db8138a188c09c731b992ae444cd4e573401ca75443c3eb318` /
`9a16bf06462879549b4e278f652da1aaf55b80908ad8c9db4a857fa6bfee281a`；其
`formal_acceptance=true` 只证明该 4B RS-General Adapter 在独立冻结协议下相对自身 base
通过六项判据。跨家族报告不参与 Gate，不支持 4B 相对 2B 的科学优越性结论。

### Grounded data 与 Region Adapter

| 资产 | 根 | Manifest SHA-256 / 状态 |
| --- | --- | --- |
| deterministic Auto Pilot | `outputs/stage4_landslide_evidence/landslide_evidence_corpus_v1_pilot_500` | `37aebb9c5f8ceb720e0a1a3c8621212d44562fa6b6786d145c31e11ffa94f9bb`；500 train-only records |
| Region Corpus base | `../benchmark/oa_grounded_stage4_v1/region_corpus/mask_grounded_region_corpus_train_v1_500` | `d18e6b4f3ab566447131ecd6fa45eb21b7675a582e85579ef8320289093ec32e` |
| Region Corpus extension | `../benchmark/oa_grounded_stage4_v2/region_corpus/mask_grounded_region_corpus_train_extension_v2_7950` | `a26f4267ba12fad8ac39481dcd16dd40a65dacdec7133453847cb2e2c71d43fe` |
| Region collection | `../benchmark/oa_grounded_stage4_v2/region_collection/mask_grounded_region_train_collection_v2_8450` | `cd2b86f6244f4f5f42d846166f11a34efdb9edd636239039b42444c453e435d2` |
| compact training messages | `../benchmark/oa_grounded_stage4_v2/training_messages/mask_grounded_region_compact_training_messages_train_v3_6974` | `746f641f1fbe48f4301ffc0c52b586437a1dc0b68a5add4be1e3db50d69a1184`；6,974 mixed-supervision records |
| Qwen3.5-4B Grounded bounded smoke | `outputs/smoke/mask_grounded_region_lora_qwen35_4b_rsinit_r851bf6e8_smoke_v1` | step-1 workflow state/report/checkpoint manifest/Adapter SHA 为 `5181f05233740457c5f3177beee5cbaed1af985016da2113839dbf3660d0c4c9` / `158f48a5ac96b29c6ccd2582f28aa199681ca5a662d2ad66542ebe66d21af755` / `c1fea784bf6e949628286f04da14d402ba6a3f84c14084d0790d0f6abe620532` / `41c6182fbbe7abb8cf533b9a2f6295ad0feda259a3426317b1c54fb12612e040`；step-20 resume 在生成新持久化 step 前 OOM，故这里只登记失败门证据，不是可发布 Adapter |

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
| Gate B | 2B 与 4B 各自冻结作用域正式通过 | 分别只证明同一模型家族中 RS-General Adapter 相对自身 base 的提升；跨模型报告不作升级判据或优越性结论 |
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
| 全量回归基线 | 能力重构时收集 339 项：331 passed，4 项因缺少 `oa_auxseg_hdf5_v1/small` skip，4 项含 backward/optimizer step 仅收集未执行；M6 完成后双后端/VLM/训练/Gate B/Runtime/Retrieval/架构相关回归 215/215 通过，另有 Qwen3.5 Gate B 单记录真实 CUDA smoke 1/1 通过；M7 变更后本模块/VLM/架构/Grounded evaluator 分别 16/16、58/58、9/9、1/1 通过，但 20-step CUDA smoke OOM，故不得把静态通过升级为 M7 工程通过 |
| Runtime / Demo CPU | `tests/runtime` 63/63；新增实测 inference-only Shared MLLM loader 不调用 compact/Eval reader/Benchmark `__getitem__`/HDF5，checkpoint/Adapter/Bank SHA 漂移 fail closed，六任务配置在 Eval-dev 缺失时仍可预检；既有 Demo、i18n、candidate、test lock 和 user-mask 合同继续通过 |
| Stage-5 / retrieval / data | 当前 `tests/training/grounding` 16/16；此前 `tests/retrieval` 24/24、`tests/data/grounded` 71/71。M7 v3 同家族 warm-start、固定 retention identity、独立正式/smoke 根及 1/20-step 恢复合同通过静态测试；20-step 真实 CUDA 资源门失败 |
| Architecture | 9/9；活动配置可解析，双后端保持物理分离，已删 Eval/dev 路径不再出现于活动 YAML/JSON，package import graph 与薄 CLI 保持稳定 |
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

当前没有可自动执行的下一模块。M7 的 1-step smoke 已通过，但从唯一 step-1 checkpoint
恢复的 20-step smoke 在 RTX 4090 D 上真实 OOM；这正是冻结方案规定的资源硬停止条件。
因此 1000-step Grounded 正式训练、M8 Runtime、M9 100 条 val-only 评价与 M10 默认发布均
未启动。

若要继续，负责人必须重新 grill 并明确授权一个新的实验方案；QLoRA、缩减输入/图像、
安装可选 kernel、改变 LoRA 层或换用更大显存设备都属于新方案，不得在本轮自动尝试。
在取得新方案前必须保留 step-1 smoke 证据和正式根缺失状态，并继续保持既有 2B/4B
RS-General Adapter、Gate B、Benchmark、Text Evidence Bank 与正式 outputs 只读；仍禁止
访问 test/sealed、Gate A/C/D、Case RAG、commit 和 push。

## 维护规则

- 新状态直接更新现有游标、能力表、资产表和边界表，不再按日期追加实施日记。
- 新正式资产只记录稳定根、manifest/report identity 和科学身份；细节留在资产自身
  manifest、ledger、report 和 Git 提交中。
- 已解决的失败、临时 PID、scratch 路径和一次性命令不进入活动文档；仍构成当前 blocker
  时才保留最短必要说明。
- 历史 Stage/phase 名可以保留在 schema、output root、checkpoint metadata 和 Git
  provenance 中，但不得重新成为活动源码、配置、脚本或测试的组织方式。
- 不新增重复的 archive、handoff、audit 或 worklog 文档。
