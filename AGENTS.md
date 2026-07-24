# Repository Agent Guide

## 1. Active program

This repository is executing an incompatible HDF5-first rewrite of single-time landslide
segmentation. The active runtime package is `src/sami_gsd/`, but location inside that package does
not imply that an existing file is approved for the new line.

The current authorities, in order, are:

1. the project owner's current phase and explicit corrections;
2. `docs/REFACTOR_TASK_SPEC.md` — the only detailed scientific and interface design;
3. `docs/CODEX_REFACTOR_PROMPT.md` — execution boundaries and prohibited actions;
4. `REFACTOR_PROGRESS.md` — current cursor only;
5. the latest applicable handoff under `docs/handoffs/`;
6. accepted ADRs under `docs/adr/`.

Old reports, handoffs, archived documents, checkpoints, and Git history are historical evidence.
They do not prove the current HDF5 Benchmark or model route has passed.

For every refactor task, read both authorities before planning or editing. The active
`SEGMENTATION_MODEL_CONTINUATION` Goal is authorized through P4, but real owner-run gates still
control every phase transition.

## 2. Stage map

```text
P0R  governance, HDF5 audit, dirty-P3 reset
  -> P1  self-contained materialized HDF5 Canonical Benchmark v4 Small
  -> P2  minimal convolutional direct-dense sanity
  -> P3  independent kernel candidates and one frozen winner
  -> P4  selected-model training engineering
  -> P5  strict/exploratory-separated evaluation
  -> P6  robustness and necessary ablations
  -> P7  reproducibility and export packaging
  -> P8  residual deletion only
```

Internal work packages are not additional approval points within an explicitly authorized complete
phase. Continue within the active Goal, but stop code progression at a phase artifact, destructive
gate, or documented owner/scientific gate.

## 3. Current cursor

P0R is the completed governance baseline. The P3-P8 authority restoration and continuous Goal were
authorized on 2026-07-24. P1 materialized-copy implementation and focused checks are complete; the
current cursor is the owner builder and independent-validator gate. Benchmark v4 has not been built
or accepted. The benchmark root was empty at the continuation start.

The dirty uncommitted P3 was intentionally discarded under the project owner's explicit
authorization. Do not reconstruct it, its class/config/schema names, or its checkpoints.

Before new work, reopen:

- `docs/adr/ADR-0004-hdf5-rebaseline-and-clean-p3-reset.md`;
- `docs/adr/ADR-0005-segmentation-only-continuation.md`;
- `docs/audits/hdf5_source_audit.md`;
- `docs/audits/hdf5_source_contract.yaml`;
- `docs/audits/greenfield_reset_reuse_matrix.md`;
- `docs/audits/deletion_plan.yaml`;
- `docs/handoffs/P0R.md`.
- `docs/handoffs/P1.md`.

## 4. Data governance

Training admission and evaluation credibility are separate. Every new source/canonical record
must carry:

- `ingestion_status`;
- `canonical_split`;
- `split_assurance`;
- `evaluation_eligibility`.

Missing per-sample location, parent/group, canonical index, or complete duplicate-component
evidence does not exclude readable image/mask pairs from training. It prevents strict
group-isolation claims.

P1 must preserve native splits for GDCLD, LMHLD, LandslideBench_agent, and Multimodal with
`source_declared_unverified` assurance. Landslide4Sense is entirely `train_only`. Empty Sen12 is
`not_ready`. Do not randomly re-split any of them.

Canonical identities bind portable `datasets/...` source paths and
`benchmark/sami_landslide_hdf5_v4/small/assets/...` copy paths to the same content hashes. Do not
serialize machine absolute paths. HDF5 files under the datasets root are read-only construction
assets; downstream model loading uses only immutable Benchmark copies and the bound channel catalog.

Canonical builders perform technical and scientific validation, not legal review. Do not create
raw-data license or source-permission runtime gates. Preserve non-gating scientific provenance;
future public release remains a separate human review.

## 5. Model boundaries

P1 contains no model algorithm.

P2 implements only a small convolutional direct-dense segmentation baseline and 1/4/8/32-parent
memory tests. It must not load a language model, generate boxes, build a candidate registry, or
pre-implement P3.

P3 may then implement independent:

1. the Channel-Set Dense direct multimodal kernel;
2. a lightweight prompt-conditioned derivative after the direct kernel passes.

No candidate may auto-switch or silently fall back. Larger language-driven pixel decoders and
box/proposal paths are outside the current program. The old runtime architecture, previous
uncommitted candidate framework, old schemas, and old checkpoint formats are forbidden from the
new main line. Compatibility shims are forbidden.

## 6. Collaboration and runtime rules

The current program explicitly authorizes focused CPU/unit tests and a synthetic or
accepted-tiny-fixture GPU smoke with batch size 1 and at most two optimizer steps. These checks do
not establish phase acceptance.

- do not run builders, independent validators, owner micro-overfit, frozen multi-seed comparisons,
  formal memory gates, long training, formal evaluation, web servers, or paid APIs;
- provide exact owner commands for every phase gate;
- never claim a phase passed because code or a report path exists;
- diagnose returned logs from the actual stack trace, config, report, or artifact before editing.

Read-only `rg`, `sed`, `find`, Git inspection, JSON/YAML/Markdown parsing, Python `ast.parse`, and
`git diff --check` are allowed. Run permitted Python checks through
`/home/yukun80/miniconda3/envs/qwen3vl/bin/python`.

Do not commit or push unless the project owner explicitly requests it.

## 7. Filesystem and paths

```text
codes/
├── datasets/       read-only HDF5 source assets
├── benchmark/      generated artifacts; empty at P0R
└── paper7_VLM/     this repository
```

Default runtime roots:

```text
PAPER7_DATASETS_ROOT=/home/yukun80/codes/datasets
PAPER7_BENCHMARK_ROOT=/home/yukun80/codes/benchmark
```

These runtime values must never enter canonical identity. New indexes store portable
`datasets/...` and `benchmark/...` references and use shared resolvers.

Do not modify `../datasets`, invent assets under `../benchmark`, or overwrite existing outputs
without exact authorization.

## 8. Dirty worktree and deletion

Preserve user changes by default. Never use `git reset --hard` or `git checkout --`. Use
`apply_patch` for text edits.

The completed P0R dirty reset was a one-time, exact-path authorization recorded in
`docs/audits/p0r_dirty_p3_reset_manifest.json`; it does not authorize future broad cleanup.

Committed deletion is replacement-owned:

- the owning replacement must be accepted;
- exact targets must be resolved;
- references/imports, tests, and docs must be scanned;
- human approval for that deletion entry must be present;
- raw datasets, models, accepted external artifacts, and unresolved legacy paths are excluded;
- `deleted_commit` stays `null` until the owner creates a later deletion commit.

P8 is not a universal waiting room; it receives only assets whose earlier replacement gates were
never satisfied.

## 9. Coding and documentation style

- Python 3.11, four-space indentation, and type hints for contracts.
- Prefer `pathlib`, `argparse`, dataclasses/Pydantic, strict JSON/JSONL, atomic writes, and SHA-256.
- New executable scripts need a short Chinese header describing purpose, recommended command,
  inputs, outputs, write behavior, and stage.
- Use concise Chinese comments for scientific or non-obvious logic.
- Keep algorithms out of CLI modules.
- Add only the modules, schemas, configs, and entrypoints required by the current phase.
- README is the sole runbook; Task Spec owns design; AGENTS owns agent behavior; Progress owns the
  cursor; handoffs own command evidence.

## 10. New-session checklist

1. Read the two authorities, current progress, applicable handoff, accepted ADRs, source contract,
   reset matrix, and deletion plan.
2. Run read-only `git branch --show-current`, `git rev-parse HEAD`, `git status --short`, and
   `git diff --stat`.
3. Inspect current `../benchmark` and relevant source/report files; do not trust dated counts
   without cheap verification.
4. Confirm the user's exact phase and deletion authority.
5. Make the smallest phase-complete change and stop at the phase/human gate.
6. Update Progress and write a handoff before declaring phase completion.
7. Report changed/restored/deleted paths, actual commands and exit codes, unrun programs, risks,
   and required human actions.
