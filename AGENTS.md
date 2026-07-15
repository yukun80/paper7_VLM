# Repository Agent Guide

## 1. Repository Mission

This repository is no longer the old GeoHazard-HalluGround/Qwen-LLaVA export project.
Its active research program has two connected tracks:

1. Build and evaluate **Multi-Source Qwen-PSALM-Seg** for instruction-driven landslide
   segmentation from heterogeneous remote-sensing evidence.
2. Extend the validated segmentation system into **segmentation-grounded remote-sensing
   description**, covering global captioning, region alignment, auditable landslide-region
   descriptions, predicted-mask curricula, and segmentation-retention-aware joint training.

The active segmentation architecture is:

```text
SANE -> QMEF -> PMRD -> semantic-evidence verifier
```

- **SANE**: sensor-aware native-scale encoder with per-modality multiscale features.
- **QMEF**: query-conditioned multimodal evidence fusion with reliability and null evidence.
- **PMRD**: PSALM-style proposal mask generation, matching, relevance verification, and
  iterative refinement.
- **Qwen3-VL-2B**: frozen visual tower used through offline caches plus an online language
  decoder whose mask-query states are trained with staged QLoRA. Qwen is a semantic/evidence
  controller and proposal verifier, not a bbox generator.

The first description architecture is sequential rather than a single monolithic forward:

```text
encode_multisource -> segment -> build region evidence -> describe
```

The governing description plan is `docs/benchmark_GAR.md`. The older segmentation refactor
rationale is in `docs/opt_refactor_algo.md`. Read both before changing model contracts.

## 2. Current Research Status

The codebase is an **M0-M7 engineering-complete candidate**, not a scientifically validated
final system. Code existence must never be reported as experimental success.

| Stage | Engineering state | Scientific/manual gate still required |
| --- | --- | --- |
| M0/M1/M1.1 | Description audit, image materialization, canonical near-duplicate merge, validation and summaries implemented | Rebuild Small with current protocol; require `errors == []` and zero verified cross-split clusters |
| M2 | Region inventory, three-level evidence, rule candidates, two-reviewer package, arbitration merge and validation implemented | Two independent reviews, arbitration, and a frozen Pilot gate |
| M3 | Task-neutral backbone state and Description Vision Cache v1 implemented | Manual cache build, state-equivalence and cache-isolation validation |
| M4 | Region encoder ablations and MGRR v2 implemented | Three-seed Small comparison, retrieval, ERFS, UFCR and counterfactual gates |
| M5 | `desc_adapter`, causal JSON generation, raw parse/repair split and explicit checkpoint migration implemented | Overfit, reload and 24GB smoke tests |
| M6 | D-1 and D0-D4 curricula, GT/fixed/end-to-end evaluation, OOF v2, factuality and paired CI implemented | Run the curriculum and complete expert/scientific evaluation |
| M7 | Task-isolated alternating training and strict segmentation-retention gate implemented | Initialize from accepted M6 weights and pass exact-population full-val retention |

Only after every Small gate passes may the project build and train the Full description system.

### Current artifact snapshot

Treat this as a dated orientation note and re-check files before relying on it:

- `../benchmark/multisource_landslide_v2_small` exists and its final validation report has
  `errors == []`.
- `../benchmark/qpsalm_description_v2_small` was built with the older
  `description_benchmark_m1_v2_materialized` protocol. Current code uses
  `description_benchmark_m1_v4_answer_trace`, so Description Small must be rebuilt before M1.1
  can be accepted.
- `../benchmark/landslide_region_description_v1_small` had not yet been built at the last
  inspection. M2 therefore remains a manual next step.
- The completed segmentation run is
  `outputs/qpsalm_v2/small_qwen_b4_bf16_nockpt`. It contains best/last checkpoints, full-val
  evaluation, galleries, and training manifests. The last reported full-val metrics were
  approximately IoU `0.3975`, Dice `0.4758`, positive IoU `0.3055`, and positive Dice `0.3959`.
  Verify the current `eval_report.json` before quoting them.
- The worktree intentionally contains a large ongoing M0-M7 change set. Do not revert or
  normalize unrelated changes, and do not restore user-deleted files under `external/`.

## 3. Collaboration Rules

The user manually runs all project programs. Unless the user explicitly asks otherwise:

- Do **not** run benchmark builders, validators, unit tests, smoke tests, training, evaluation,
  CUDA programs, environment probes, or web servers.
- Implement and statically inspect code, then give exact commands for the user to run.
- Read-only inspection commands such as `rg`, `sed`, `find`, `git status`, `git diff`, and
  `git diff --check` are acceptable.
- Do not spend time diagnosing Torch/CUDA availability. The user manages the `qwen3vl`
  environment and GPU.
- When the user returns logs, diagnose from the actual stack trace, config, report, or artifact
  before editing.
- Never claim a stage passed until the user has run its command and supplied a valid report.

This is a dirty worktree. Preserve user work:

- Never use `git reset --hard`, `git checkout --`, or destructive cleanup.
- Do not revert changes you did not make.
- Do not rewrite external benchmarks, source datasets, checkpoints, or caches unless explicitly
  requested.
- Use `apply_patch` for manual edits.

## 4. Filesystem and Path Protocol

Large data live beside the repository, not inside it:

```text
/home/yukun80/codes/
├── datasets/
├── benchmark/
└── paper7_VLM/
    ├── SEG_Multi-Source_Landslides/
    ├── models_zoo/
    ├── outputs/
    ├── scripts/
    └── docs/
```

Default overrides:

```text
PAPER7_DATASETS_ROOT=/home/yukun80/codes/datasets
PAPER7_BENCHMARK_ROOT=/home/yukun80/codes/benchmark
```

Legacy `DATASETS_ROOT` and `BENCHMARK_PREFIX` may still be recognized where documented.

Indexes use portable logical references:

```text
datasets/...
benchmark/...
```

`outputs/`, `models_zoo/`, `external/`, and configuration paths remain repository-relative.
Use the shared resolvers; do not reintroduce ad hoc `REPO_ROOT / value` path handling.

### Storage semantics

- Landslide Benchmark V2 materializes modalities, masks, valid masks and previews in the
  benchmark package.
- Description Benchmark V2 materializes each selected parent image exactly once under
  `data/<split>/<source>/<parent>.<suffix>`. Final model indexes must not depend on `datasets/`.
- Description source indexes keep original `datasets/...` references for provenance only.
- Landslide Bridge materializes region masks and review panels but references Landslide V2
  modalities instead of copying large arrays again.
- The unified SegDesc benchmark is intentionally reference-only. It binds component indexes,
  exact line numbers, record IDs, hashes, validation reports, and Bridge publication status.

## 5. Repository Layout

### Benchmark pipelines

```text
scripts/1-benchmark/          Landslide Benchmark V2 source/final/referring pipeline
scripts/2-instruction/        global/referring/no-target instruction expansion
scripts/3-description/        Description M0/M1/M1.1 audit, dedup, split, materialization
scripts/4-landslide-bridge/   M2 region facts, candidates, review and expert freeze
scripts/5-segdesc/            unified component-reference index
```

Shell entrypoints:

```text
scripts/run_1_build_benchmark.sh
scripts/run_2_build_instruction_dataset.sh
scripts/run_3_build_description_benchmark.sh
scripts/run_4_build_landslide_bridge.sh
scripts/run_5_build_segdesc_dataset.sh
```

### Model package

```text
SEG_Multi-Source_Landslides/qpsalm_seg/
├── data/ or data modules        physical modalities, subsets, transforms and prompts
├── models/                      SANE, QMEF, PMRD and total model assembly
├── description/                 M3-M7 states, MGRR, adapters, training and evaluation
├── engine/                      segmentation trainer/evaluator/checkpoint plumbing
└── cli/                         executable command entrypoints only
```

`cli/` follows standard Python application organization: command parsing and orchestration live
there; reusable algorithm logic belongs in `models/`, `description/`, `engine/`, or data modules.

Key configs:

```text
SEG_Multi-Source_Landslides/configs/qpsalm_v2_small.yaml
SEG_Multi-Source_Landslides/configs/qpsalm_v2_full.yaml
SEG_Multi-Source_Landslides/configs/qpsalm_v2_smoke.yaml
SEG_Multi-Source_Landslides/configs/qpsalm_segdesc_small.yaml
configs/description_ontology_v1.yaml
configs/qpsalm_description_record_v2.schema.json
configs/qpsalm_description_output_v1.schema.json
configs/qpsalm_landslide_region_description_v1.schema.json
configs/landslide_bridge_v1.yaml
```

## 6. Segmentation Protocol Invariants

The active benchmark is `multisource_landslide_v2_{small,full}`. V1 indexes, caches and
checkpoints are not compatible.

Each physical modality must carry explicit metadata instead of being inferred from a canonical
slot: family, sensor, product type, band/polarization names, orbit, units, signedness, native and
aligned GSD, quality, normalization and valid mask.

`ActiveModalitySubset` is the single source of truth after modality dropout. Inactive modalities
must not leak into:

- SANE instances;
- availability or prompts;
- visual cache tokens;
- QMEF reliability;
- teacher/student consistency inputs.

Valid masks must be propagated through resize/pad, SANE/QMEF attention, proposal matching,
losses and metrics. Metrics distinguish overall, positive-only, negative accuracy and empty
false-positive rate.

The main path excludes:

- bbox priors or Qwen box generation;
- disaster pre/post change detection;
- hidden fixed five-slot channel concatenation;
- old local-preview dense branches and old multi-gate/scorer stacks.

Current Qwen training defaults in Small are BF16, NF4, batch size 4, gradient accumulation 1,
query chunk 16, SDPA and disabled gradient checkpointing. QLoRA activates after a decoder warmup
at step 450. Do not add runtime-profile indirection; YAML is the runtime source of truth.

Qwen Vision Cache v3 is parent-level and stores frozen multiview visual features. It is not a
description cache and must not be silently rebuilt when only training settings change.

## 7. Description and Bridge Protocol Invariants

### M1.1 canonical image policy

- dHash equality is candidate recall only.
- Candidate images are normalized to RGB 64x64; `MAE <= 3.0` verifies near-duplicates.
- Verified edges form connected canonical clusters before split and sampling.
- RSIEval/source official test status has priority for the whole cluster.
- One visual canonical parent remains; all source captions are merged with answer-level
  provenance.
- `indexes/train_eligible.jsonl` excludes zero-weight answers, while full audit indexes retain
  them.
- `verified_perceptual_duplicate_cross_split_groups` must equal zero.

Do not treat DIOR-RSVG referring phrases as detailed captions. It supplies region alignment and
same-image candidate retrieval only.

### M2 Bridge evidence and review

Region sources are `gt_global_mask`, `pseudo_instance_component`, deduplicated
`gt_referring_mask`, and `no_target`. Connected components are pseudo instances, not human
instance labels.

Evidence levels are strict:

- Level A: physical values only when units, sign and GSD are trustworthy.
- Level B: normalized relative region/context anomalies only.
- Level C: unavailable or insufficient; never fabricate physical claims.

Rule or teacher text is a review candidate, never expert truth. The formal Pilot uses 300 parent
samples split 180/60/60 across train/val/test. Two reviewers independently accept/revise/reject;
disagreements require arbitration.

The evaluation gate starts as
`manifests/evaluation_gate_manifest.template.json` with `frozen=false`. A human must fill Pilot
thresholds and explicitly set the v2 gate to `frozen_after_pilot`. The gate is bound to current
Pilot, selection and candidate hashes.

### Unified index publication

Current protocols:

```text
builder:    qpsalm_segdesc_index_builder_v2_expert_gate_bound
schema:     qpsalm_segdesc_index_v1
validation: qpsalm_segdesc_index_validation_v2
statistics: qpsalm_segdesc_index_statistics_v2
```

Top-level task groups are:

```text
segmentation
global_caption
region_alignment
region_description_auto
region_description_expert
```

When Bridge status is `awaiting_expert_review`, only auto descriptions may be published. Stale
`expert_all.jsonl` or final gate files are recorded and ignored. Expert rows may be published only
when Bridge validation is `expert_pilot_frozen` and the current v2 gate path, hash and protocol
all match.

### M3-M7 model protocols

Important current versions:

```text
Description Vision Cache: task_neutral_parent_visual_features_v1
MGRR:                    qpsalm_mgrr_v2_multiscale_grid_replay
Description sequence:    qpsalm_description_causal_v4_stage_separated
SegDesc checkpoint:      qpsalm_segdesc_v1
Joint training:          qpsalm_segdesc_joint_v3_task_isolated
```

- Keep `MultisourceBackboneState`, `SegmentationState`, and `RegionEvidenceState` distinct.
- Global caption stages must not inject MGRR region tokens.
- D0/D1 train global caption components; D2 starts region alignment/MGRR; D3a is auto Bridge;
  D3b requires frozen expert Bridge; D4 uses out-of-fold predicted masks.
- `--resume` is only for the same stage/run. Use `--initialize-from` between D0-D4 stages.
- Never silently use `strict=False` to hide checkpoint incompatibility.
- `default` is the segmentation adapter and `desc_adapter` is the description adapter. Only the
  adapter belonging to the current task may be active/trainable.
- Main structured metrics use raw, unrepaired JSON. Deterministic repair is analysis-only.
- GT-mask, fixed-prediction and end-to-end results must be reported separately.
- OOF predicted masks must be generated by checkpoints that excluded the target parent fold.
- M7 uses separate task DataLoaders and same-task gradient accumulation. Do not mix task types
  inside one optimizer accumulation window.
- Full-val retention requires the exact same sample population identity, threshold and prompt/
  transform protocol as the segmentation baseline. Maximum allowed positive Dice drop is one
  absolute percentage point.

`jsonschema>=4.20` remains a declared dependency. `output_protocol.py` also has an internal
validator for the exact keyword subset used by the fixed output schema so package import and
protocol tests remain functional when the environment has not yet been reinstalled.

## 8. Manual Workflows

Run all commands from the repository root with the `qwen3vl` environment activated. The agent
should provide these commands but not execute them unless explicitly asked.

### Landslide Benchmark V2 Small

```bash
SMALL_LIMIT=500 bash scripts/run_1_build_benchmark.sh small
bash scripts/run_2_build_instruction_dataset.sh small
```

`SMALL_LIMIT` means a parent limit per `dataset_name + split`, not a global split limit.

### Description M1.1 Small rebuild

```bash
RUN_CONTROL=--overwrite \
DESCRIPTION_COPY_WORKERS=8 \
PYTHON_BIN=/home/yukun80/miniconda3/envs/qwen3vl/bin/python \
bash scripts/run_3_build_description_benchmark.sh small
```

### M2 prepare

```bash
BRIDGE_STAGE=prepare \
BRIDGE_PILOT_PARENTS=300 \
RUN_CONTROL=--overwrite \
PYTHON_BIN=/home/yukun80/miniconda3/envs/qwen3vl/bin/python \
bash scripts/run_4_build_landslide_bridge.sh small
```

Do not invent completed reviewer files or a frozen gate. Merge only after the user supplies real
reviews, arbitration where needed, and a filled gate.

### Unified index

```bash
RUN_CONTROL=--overwrite \
PYTHON_BIN=/home/yukun80/miniconda3/envs/qwen3vl/bin/python \
bash scripts/run_5_build_segdesc_dataset.sh small
```

### Segmentation training

```bash
BENCHMARK_SIZE=small \
PRESET=qwen_psalm_full \
SEED=42 \
RUN_NAME=small_qwen_b4_bf16_nockpt \
RUN_CONTROL=--overwrite \
CACHE_CONTROL=reuse \
bash SEG_Multi-Source_Landslides/scripts/run_qpsalm_experiment.sh
```

Do not overwrite the existing completed run unless the user explicitly requests retraining with
that same name.

### Protocol tests

```bash
PYTHONPATH=SEG_Multi-Source_Landslides \
python -B -m unittest \
  SEG_Multi-Source_Landslides/tests/test_description_benchmark.py \
  SEG_Multi-Source_Landslides/tests/test_landslide_bridge.py \
  SEG_Multi-Source_Landslides/tests/test_segdesc_unified_index.py \
  SEG_Multi-Source_Landslides/tests/test_segdesc_protocol.py -v
```

For complete D0-D4, OOF, expert factuality, three-seed comparison, M7 and demo commands, use the
root `README.md`; do not duplicate or improvise abbreviated scientific workflows.

## 9. Validation and Scientific Gates

Engineering acceptance always requires the relevant `validation_report.json` to contain
`errors == []`. Warnings must be interpreted, not silently discarded.

Minimum stage gates:

1. Landslide V2 source/final/referring/instruction validation passes.
2. Description M1.1 has no cross-split exact or verified canonical cluster, no unregistered image,
   no `.part` file, and complete answer provenance.
3. Bridge prepare is `awaiting_expert_review`, covers the requested Pilot quota and has no expert
   labels. Frozen Bridge requires complete reviews, zero pending arbitration, all three expert
   splits, and a valid bound gate.
4. Unified index line number, record ID, component hash, component validation hash, task mapping,
   split partition and expert publication state all match.
5. MGRR enters the main model only after fixed-parent three-seed gains in expert factuality and
   same-image retrieval without worsening unsupported claims, plus required mask/modality
   counterfactual sensitivity.
6. M7 must pass exact-population full-val segmentation retention; monitor subsets are not final
   evidence.

Keep zero-shot, GT-mask oracle, fixed-prediction and end-to-end results separate. Do not use
caption overlap metrics alone as evidence of grounded regional understanding.

## 10. Coding and Documentation Style

- Python 3.11, four-space indentation and type hints for data contracts.
- Prefer `pathlib`, `argparse`, dataclasses and structured JSON/JSONL APIs.
- Use `rg`/`rg --files` for searches.
- New executable scripts need a short Chinese header containing purpose, recommended command,
  inputs, outputs, write behavior and workflow stage.
- Add concise Chinese comments at scientifically important or non-obvious logic; avoid comments
  that merely restate code.
- Keep algorithms out of CLI modules.
- Use atomic writes for benchmark manifests and indexes.
- Preserve deterministic seeds, parent-level split isolation and explicit protocol versions.
- Do not add backward-compatibility shims for old V1 benchmarks, old experimental SegDesc
  checkpoints, text cache v1 or visual cache v2 unless the user explicitly changes this policy.
- Update root `README.md` for runnable commands and `docs/benchmark_GAR.md` for scientific design;
  avoid creating a second competing run manual.

## 11. New-Session Handoff Checklist

At the beginning of a new task:

1. Read this file, `docs/benchmark_GAR.md`, and the relevant README section.
2. Run only read-only inspection: `git status --short`, inspect current validation reports, and
   identify whether the user has rebuilt M1.1 or M2 since this snapshot.
3. Do not assume generated benchmark counts or statuses from an older conversation.
4. Preserve the current dirty worktree and inspect nearby edits before changing a file.
5. Ask for or read the user's latest command log; do not rerun their expensive workflow.
6. Make the smallest protocol-correct change, add focused synthetic tests, and provide manual
   commands for verification.

The immediate continuation at the time of this update is:

```text
rerun the four protocol test modules
-> rebuild Description Small under M1.1 v4
-> inspect validation and canonical merge reports
-> run M2 Bridge prepare
-> publish auto-only unified index
-> complete real expert review before D3b/D4/M7
```
