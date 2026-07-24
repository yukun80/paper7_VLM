# REFACTOR_PROGRESS

## Current status

- program: `SEGMENTATION_MODEL_CONTINUATION`
- goal_status: `resumed_after_interrupted_turn`
- terminal_scope: `P4_MODEL_ENGINEERING_ACCEPTED`
- phase: `P1`
- phase_name: `HDF5_CANONICAL_BENCHMARK_V4_SMALL`
- phase_status: `awaiting_owner_p1_build_and_validation`
- execution_date: `2026-07-24`
- current_branch: `refactor/sami-groundsegdesc`
- reset_start_head: `58813468d2cf3be715e399f5904a8d9f7f5c880d`
- push_performed: `false`
- commit_performed: `false`
- builder_run: `false`
- validator_run: `false`
- focused_unit_tests_run: `true`
- bounded_gpu_smoke_run: `false`
- training_or_formal_cuda_run: `false`
- benchmark_root_at_audit: `empty`
- current_owner_gate: `P1_BUILDER_AND_INDEPENDENT_VALIDATOR`
- next_phase: `P2`
- next_phase_authorized: `false_until_live_p1_acceptance`

## Active authorities

- `docs/REFACTOR_TASK_SPEC.md`
- `docs/CODEX_REFACTOR_PROMPT.md`
- `docs/adr/ADR-0004-hdf5-rebaseline-and-clean-p3-reset.md`
- `docs/adr/ADR-0005-segmentation-only-continuation.md`
- `docs/adr/ADR-0006-self-contained-hdf5-benchmark.md`

ADR-0001 and ADR-0003 remain historical accepted decisions only where they do not conflict with
the current authorities, ADR-0004, ADR-0005, or ADR-0006.

## Segmentation continuation

- P3-P8 authority restoration completed before model-code changes.
- Task Spec is the only detailed design; Prompt contains execution and routing only.
- The continuous Goal is active across P1-P4 but cannot cross owner-run phase gates.
- P5-P8 are authority-only under the active Goal.
- The default P3 candidate is Channel-Set Dense; prompt conditioning is evidence-gated.
- Larger language-driven pixel decoders and box/proposal paths are outside the active program.
- No earlier removed P3 file, schema, config, class, report, or checkpoint was restored.

Authority checks:

| Check | Exit | Result |
| --- | ---: | --- |
| UTF-8/BOM | 0 | passed |
| Markdown fences | 0 | passed |
| relative links | 0 | passed |
| removed P3 symbol scan | 0 | no matches |
| authority `git diff --check` | 0 | passed |

The first fence-check command exited 1 because the shell interpreted literal Markdown backticks.
It had no write effect; the corrected check passed.

## P0R decisions

- Training admission and evaluation credibility are independent.
- GDCLD, LMHLD, LandslideBench_agent, and Multimodal retain native splits as
  `source_declared_unverified` / `exploratory`.
- Landslide4Sense contributes all 3,799 samples as `train_only`.
- Sen12Landslides is empty and remains `not_ready`.
- No current cohort is certified strict.
- Benchmark v4 byte-copies the selected HDF5 pairs into an immutable, source-organized `assets/`
  tree and binds source/copy paths to identical hashes.
- Stored channel order remains source-native; per-record descriptors and a 19-key channel catalog
  carry physical meaning and known/unknown metadata into later model loading.
- P2 begins with minimal direct-dense segmentation; no language model or box path.
- P3 candidates are rebuilt from zero.
- Deletion is replacement-owned; P8 receives residual assets only.

## Dirty reset

The two authorities were rewritten first. All other tracked dirty P3 edits were restored to HEAD
using exact paths. All inventoried untracked P3 configs, draft ADRs, worklog, schemas, modules,
tests, reports, caches, checkpoints, and bytecode were removed without a patch backup.

Pre-delete evidence and exact targets are recorded in
`docs/audits/p0r_dirty_p3_reset_manifest.json`.

## P0R artifacts

- `docs/adr/ADR-0004-hdf5-rebaseline-and-clean-p3-reset.md`
- `docs/audits/hdf5_source_audit.md`
- `docs/audits/hdf5_source_inventory.json`
- `docs/audits/hdf5_source_contract.yaml`
- `docs/audits/greenfield_reset_reuse_matrix.md`
- `docs/audits/p0r_dirty_p3_reset_manifest.json`
- `docs/audits/deletion_plan.yaml`
- `docs/audits/reuse_matrix.md`
- `docs/handoffs/P0R.md`

## Static acceptance

- JSON parse: passed, 2 files.
- YAML parse: passed, 2 files.
- Markdown UTF-8/fence/local-link check: passed, 10 files.
- Changed Python AST parse: passed, 0 changed Python files.
- HDF5 inventory counts/splits/strict-status invariants: passed.
- HDF5 contract absolute-machine-path scan: no matches.
- Authority and active-runtime old-P3 symbol scan: no matches.
- Reset manifest target-absence replay: passed, 37 resolved targets.
- Greenfield matrix coverage: passed, all 70 scoped committed paths exactly once.
- Deletion target resolution: passed.
- `git diff --check`: passed.
- `../benchmark` top level: empty.

Exact commands, exit codes, expected no-match exit codes, and non-effect failures are recorded in
`docs/handoffs/P0R.md`.

## P1 implementation checks

| Check | Exit | Result |
| --- | ---: | --- |
| changed Python AST | 0 | materialized-copy P1 modules parsed |
| strict JSON + Schema meta-check | 0 | P1 JSON plus 7 P1 schemas |
| YAML parse | 0 | 4 active config/contract/state files |
| modified Markdown UTF-8/BOM/fence/link check | 0 | 13 files |
| expanded production config invariants | 0 | 53,645 pairs; 37,521/11,995/4,129; 311 conflicts |
| selected HDF5 stat preflight | 0 | 107,290 files; 35,698,325,607 bytes |
| focused `unittest` | 0 | 19 tests passed |
| five-source source/parent Schema replay | 0 | one live read-only pair per ready source |
| active P1 old-P3 symbol scan | 0 | no matches |
| P1 target/report absence | 0 | both owner output paths absent |
| `git diff --check` | 0 | passed |

The broader old-symbol scan found committed P2 Qwen paths. This is expected: they are
`DELETE_AFTER_REPLACEMENT` assets and cannot be removed before the new P2 gate. No removed dirty-P3
path was restored.

## Current cursor

P0R remains complete. The corrected P1 v4 config, strict contracts, HDF5 source reader,
self-contained byte-copy builder, independent source/copy validator, CLI, seven JSON Schemas, and
focused synthetic tests are implemented. Nineteen focused `unittest` cases pass in `qwen3vl`.
Synthetic evidence confirms independent file copying, materialization-ledger replay, preservation of
valid-zero versus missing-zero semantics, and an explicit 19-key production channel catalog. These
checks do not create Benchmark v4 acceptance.

The project owner must now run the README P1 commands. Codex did not run the full builder or
independent validator and did not write `../benchmark`. Do not enter P2 until the owner returns the
live immutable benchmark plus validation with `errors == []`, matching source/copy/materialization/
channel-catalog/manifest/normalization bindings, exact native split projection, five ready sources,
zero strict population, and the separately reported 311 LandslideBench_agent location conflicts.
