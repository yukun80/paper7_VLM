# P0R Handoff: HDF5 Rebaseline and Clean P3 Reset

- Date: 2026-07-24
- Phase: P0R
- Branch: `refactor/sami-groundsegdesc`
- Reset-start HEAD: `58813468d2cf3be715e399f5904a8d9f7f5c880d`
- Commit/push: not performed
- Next phase: P1, not authorized by this handoff

## Outcome

P0R replaces the old current-route assumptions with one HDF5-first design and removes the entire
uncommitted P3. The two authorities were rewritten before cleanup. No new algorithm was
implemented, and P1 was not entered.

Training usability is now independent of evaluation credibility. All five non-empty HDF5 sources
are admitted to the training candidate population with native split policy preserved; incomplete
group/location/duplicate evidence lowers assurance instead of excluding image/mask pairs.

## Data readiness

| Source | Pairs | Positive | No-target/background | Split policy | Evaluation |
|---|---:|---:|---:|---|---|
| GDCLD | 13,447 | 11,593 | 1,854 | native train/val/test | exploratory |
| LMHLD | 28,185 | 28,008 | 177 | native train/val/test | exploratory |
| LandslideBench_agent | 2,130 | 1,819 | 311 | native train/val/test; report 311 location conflicts | exploratory |
| Landslide4Sense | 3,799 | 2,231 | 1,568 | all train-only | train-only |
| Multimodal | 6,084 | 5,495 | 589 | native train/val | exploratory |
| Sen12Landslides | 0 | 0 | 0 | excluded as not-ready | unavailable |

Expected complete v4 source-index population is 53,645 pairs:

- train: 37,521;
- val: 11,995;
- test: 4,129;
- strict evaluation: 0 until new verified group-isolation evidence exists.

These are P0R audit expectations, not a claim that a P1 builder has run.

## Changed governance

Rewritten from zero:

- `docs/CODEX_REFACTOR_PROMPT.md`
- `docs/REFACTOR_TASK_SPEC.md`
- `AGENTS.md`
- `README.md`
- `REFACTOR_PROGRESS.md`
- `docs/audits/deletion_plan.yaml`
- `docs/audits/reuse_matrix.md`

Added:

- `docs/adr/ADR-0004-hdf5-rebaseline-and-clean-p3-reset.md`
- `docs/audits/hdf5_source_audit.md`
- `docs/audits/hdf5_source_inventory.json`
- `docs/audits/hdf5_source_contract.yaml`
- `docs/audits/greenfield_reset_reuse_matrix.md`
- `docs/audits/p0r_dirty_p3_reset_manifest.json`
- `docs/handoffs/P0R.md`

The new stage graph is:

```text
P0R -> P1 HDF5 Benchmark v4 -> P2 direct-dense sanity
    -> P3 unique kernel -> P4 training -> P5 evaluation
```

Description/Bridge work is provisional. Deletions are replacement-owned; P8 receives residual
assets only.

## Reset record

Tracked dirty files other than the two authorities were restored to HEAD by explicit path. The
reset includes governance/history edits, implementation/config/dependency edits, and the two P1
tests modified by the old P3.

The untracked reset removed:

- the old colour-view and candidate configs;
- five unaccepted draft ADRs, the P3 continuation worklog, and the baseline replay helper;
- 19 resolved candidate schemas;
- candidate contracts/runtime/model/test modules;
- all current `reports/`, including 2,861 report/cache/checkpoint files totaling
  16,267,634,136 bytes.
- three ignored bytecode remnants found by the post-status scan:
  `src/sami_gsd/contracts/__pycache__/g0.cpython-311.pyc`,
  `src/sami_gsd/contracts/__pycache__/g0_supervised.cpython-311.pyc`, and
  `src/sami_gsd/utilities/__pycache__/schema.cpython-311.pyc`.

No patch backup was created. Those untracked bytes cannot be recovered from the current worktree.
Exact paths, sizes, and key hashes are in
`docs/audits/p0r_dirty_p3_reset_manifest.json`. `deleted_commit` remains `null`.

## Reuse/reset decisions

All 70 committed paths in `src/sami_gsd`, `configs`, `schemas`, `tests/p1`, and `tests/p2` were
classified:

| Decision | Count |
|---|---:|
| KEEP_GENERIC | 5 |
| REBIND_AND_TEST | 16 |
| REWRITE | 21 |
| DELETE_AFTER_REPLACEMENT | 28 |

See `docs/audits/greenfield_reset_reuse_matrix.md` for every path. Existing language-model P2 code
is not the new P2; it remains only until the minimal direct-dense replacement passes its later gate.

## Commands and exit codes

Pre-reset and cleanup:

| Command/check | Exit | Result |
|---|---:|---|
| branch/HEAD/status/diff and external-root inventory | 0 | reset-start state captured |
| authority UTF-8/BOM/fence parse | 0 | passed |
| old active-route symbol scan over both authorities | 1 | expected no matches |
| first authority `git diff --check` | 2 | trailing blockquote whitespace found |
| corrected authority `git diff --check` | 0 | passed |
| normal-sandbox exact `git restore` | 128 | `.git/index.lock` was read-only; no restore occurred |
| approved exact-path `git restore --source=HEAD` | 0 | tracked P3 restored |
| text-file deletion through `apply_patch` | 0 | exact untracked text targets deleted |
| exact `reports`/bytecode removal | 0 | generated binary/cache targets deleted |
| empty-directory cleanup | 0 | exact reset directories removed |
| post-reset target absence check | 0 | only authorities and reset manifest remained dirty |

Two harmless inspection failures were also recorded:

- an initial shell command containing literal Markdown backticks was malformed and exited 2; it
  changed no file;
- `jq` was unavailable and exited 127; the same read-only JSON inspection was completed with the
  Python standard library;
- `git -C ../datasets status --short` exited 128 because that directory is not a valid Git
  worktree. No command in P0R wrote to it; the external non-modification claim is based on the
  executed command set, not an external Git diff.

Final static acceptance commands and exit codes are filled below after the last pass:

| Check | Exit | Result |
|---|---:|---|
| JSON parse (inventory + reset manifest) | 0 | 2 files passed |
| YAML parse (source contract + deletion plan) | 0 | 2 files passed |
| HDF5 contract absolute-machine-path scan | 1 | expected no matches |
| Markdown UTF-8/fence/local-link check | 0 | 10 files passed |
| changed-Python AST parse | 0 | no Python file changed |
| HDF5 inventory invariant replay | 0 | 5 ready sources; 53,645 pairs; 37,521/11,995/4,129; strict 0 |
| authority old-route symbol scan | 1 | expected no matches |
| active runtime/config/test old-P3 scan | 1 | expected no matches |
| old-P3 filename/ignored-bytecode scan | 0 | no output |
| final `git status` old-P3 path scan | 1 | expected no matches |
| reset target absence replay | 0 | all 37 resolved targets absent |
| 70-path reset matrix coverage | 0 | every scoped committed path appears exactly once |
| deletion target resolution | 0 | all non-placeholder targets exist |
| `git diff --check` | 0 | passed |
| untracked-file whitespace/final-newline check | 0 | 7 files passed |
| final `git status --short` | 0 | only P0R documentation/audit changes present |
| `../benchmark` top-level inspection | 0 | empty |

No builder, validator, unit/smoke test, training, evaluation, CUDA, formal, commit, or push command
was run.

## Unresolved risks

- Current native development/evaluation splits are not independently group-isolated; no strict
  generalization conclusion is available.
- LandslideBench_agent retains 311 reported location-level conflicts.
- LMHLD lacks per-sample geospatial/sensor/group evidence and has notable mask-coverage warnings.
- Landslide4Sense has no trustworthy val/test and 114 all-zero slope channels.
- Multimodal has known spatial-neighbour leakage and unconfirmed DEM/InSAR unit/scale metadata.
- Source-side conversion metadata contains absolute paths; P1 must exclude them from identity.
- Hashing and reopening 53,645 image/mask pairs is owner-run work and may be I/O intensive.
- Committed v3/P2 code still exists until the corresponding replacement-owned gates pass.

## Complete proposed P1 task prompt

```text
PLEASE IMPLEMENT THIS PLAN:

# P1 — HDF5-First Canonical Benchmark v4 Small

仓库根目录为 /home/yukun80/codes/paper7_VLM。

CURRENT_PHASE = P1
CURRENT_PHASE_NAME = HDF5_CANONICAL_BENCHMARK_V4_SMALL
EXECUTION_SCOPE = COMPLETE_PHASE_IMPLEMENTATION_TO_HUMAN_RUN_GATE
AUTO_CONTINUE_WITHIN_PHASE = TRUE
ALLOW_BUILDER_RUN = FALSE
ALLOW_VALIDATOR_RUN = FALSE
ALLOW_UNIT_TEST_RUN = FALSE
ALLOW_GPU = FALSE
ALLOW_TRAINING = FALSE
ALLOW_COMMIT = FALSE
ALLOW_PUSH = FALSE

在分析或修改前完整读取：

1. AGENTS.md
2. docs/REFACTOR_TASK_SPEC.md
3. docs/CODEX_REFACTOR_PROMPT.md
4. REFACTOR_PROGRESS.md
5. docs/adr/ADR-0004-hdf5-rebaseline-and-clean-p3-reset.md
6. docs/audits/hdf5_source_audit.md
7. docs/audits/hdf5_source_inventory.json
8. docs/audits/hdf5_source_contract.yaml
9. docs/audits/greenfield_reset_reuse_matrix.md
10. docs/audits/deletion_plan.yaml
11. docs/handoffs/P0R.md

先只读记录 branch、HEAD、git status、git diff、../benchmark 顶层状态，并廉价复核
../datasets 的六个 source 目录、authoritative index、channel schema、conversion summary 和
conversion errors。不得假设 P0R 计数仍未变化。

本阶段只实现 reference-first HDF5 Canonical Benchmark v4 Small，不实现模型、不进入 P2。

## 冻结数据决定

五个非空 source 全部进入 v4 source index，并在其允许的 train population 中参与训练：

- GDCLD：13,447；保留 train 7,897 / val 4,459 / test 1,091；
  split_assurance=source_declared_unverified；
  evaluation_eligibility=exploratory。
- LMHLD：28,185；保留 train 19,729 / val 5,637 / test 2,819；Blue/Green/Red/NIR
  四通道全部绑定；
  split_assurance=source_declared_unverified；
  evaluation_eligibility=exploratory。
- LandslideBench_agent：2,130；保留 train 1,701 / val 210 / test 219；311 个
  location-level conflict 必须原样进入风险报告；对话文本只能作为未验证 provenance，
  不得成为事实/专家/因果监督；
  split_assurance=source_declared_unverified；
  evaluation_eligibility=exploratory。
- Landslide4Sense：3,799；2,231 positive 与 1,568 background 全部进入 train；
  14 通道 B01..B12/slope/DEM 全部绑定；不得创建 val/test；
  split_assurance=train_only；
  evaluation_eligibility=train_only。
- Multimodal：6,084；保留 train 4,395 / val 1,689；RGB/DEM/InSAR 全部绑定，
  pixel_valid、channel_valid、valid_mask 全部传播；空间邻接风险进入报告；
  split_assurance=source_declared_unverified；
  evaluation_eligibility=exploratory。
- Sen12Landslides：空目录，ingestion_status=not_ready，不生成 canonical record，不伪造
  第六 source。

预期完整 source-record 总数为 53,645，split 计数预期为 train 37,521 / val 11,995 /
test 4,129；必须由现场 builder/validator 重算，不得把预期值硬编码为验收真值。

group_id、location_key、canonical parent/group 和 duplicate component 允许为空或不完备；
这不阻止训练。不得随机重分任何 source，不得把 source_declared_unverified 或 train_only
提升为 strict。当前 strict cohort 预期为 0。

## 实现范围

1. 从零定义 Benchmark v4 的 source-record、canonical-parent、manifest、validation 和
   statistics schema/config；不得保留 v3 schema alias 或 compatibility reader。
2. Canonical identity 只含 datasets/... 逻辑路径、image/mask 文件 SHA-256、HDF5 dataset
   key、shape、dtype、layout、channel schema、validity schema、source split、assurance 和
   evaluation eligibility。/home/...、mtime 和 runtime resolved path 不得进入 identity
   或 aggregate hash。
3. 每条记录独立绑定 image 与 mask HDF5；validator 必须重新打开文件，校验 hash、/image、
   /mask、shape/dtype/layout、channel order、valid_mask/pixel_valid/channel_valid。
4. 采用 reference-first index，不复制源 HDF5。只允许当前验收必需的小型审计/manifest/
   index；不要批量物化 render、cache、checkpoint 或 description asset。
5. parent_id 是 canonical 稳定标识，不得冒充 verified location/scene/group。
6. strict、exploratory、train_only 必须分别统计；train_only 不得进入 val/test；Sen12
   不得产生记录。
7. 输出必须是新的、不可覆盖、原子写入的
   ../benchmark/sami_landslide_hdf5_v4/small；builder 发现非空目标时必须停止。
8. 只实现 P1 所需 CLI、配置、schema、builder、validator、统计和 focused synthetic tests。
   不实现模型输入加载、direct-dense、Qwen、bbox、candidate registry、Description、
   Bridge、SegDesc、Full 或未来阶段抽象。
9. 按 greenfield_reset_reuse_matrix 处理 committed 文件：
   KEEP_GENERIC 可保留；REBIND_AND_TEST 必须真正绑定 v4；REWRITE 从零替换；
   DELETE_AFTER_REPLACEMENT 在项目负责人跑完 P1 并接受前不得物理删除。
10. 更新 README 的唯一 P1 运行命令、REFACTOR_PROGRESS，并写 docs/handoffs/P1.md。
    handoff 必须区分“Codex 静态实现完成”和“等待项目负责人运行/返回报告”。

## Codex 验收边界

Codex 只执行 JSON/YAML/Markdown 静态解析、修改 Python 的 ast.parse、引用扫描和
git diff --check。不得运行 builder、validator、unit tests、GPU/CUDA、训练或 formal。

完成实现后给项目负责人精确命令，顺序至少包括：

1. focused synthetic tests；
2. v4 Small builder；
3. 独立 validator；
4. validation/statistics/manifest hash 摘要检查。

不得自动执行这些命令。项目负责人返回 errors=[] 且 binding/split/status 重放通过前，
不得把 P1 标为 accepted，不得执行 replacement-owned 删除，不得进入 P2。

最终报告实际修改路径、静态命令与 exit code、未运行命令、待人工命令、已知风险和 P2
阻塞条件。
```
