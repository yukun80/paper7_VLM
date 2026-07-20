# REFACTOR_PROGRESS

## Current status

- phase: P1
- phase_status: `in_progress`
- completed_subtask: `P1.2` (`engineering_passed`)
- current_branch: `refactor/sami-groundsegdesc`
- p1_2_implementation_commit: `ac4be61e6fd994408934885095563311b9e43ebe`
- p1_2_handoff_commit: `5ad6af944199e99c9815f3d1df5b1d62565767ba`
- p1_1_implementation_commit: `898f5b83820760ca86d1d488fc4cee0e8fa5cc9e`
- p1_1_handoff_commit: `eb7b5feabeb6b2209195ed4e42beb893ee3a2a9f`
- p0_acceptance_commit: `52c93b3a77635c82eb591850e758d3333482d4b1`
- baseline_tag: verified `pre-sami-rewrite-2026-07-20` -> `0c53624dd93159f78acd6d39a579b100d7e3255f`
- baseline_branch: verified `baseline/sane-qmef-pmrd-mgrr` -> `0c53624dd93159f78acd6d39a579b100d7e3255f`
- dirty_worktree: no after the P1.2 final progress-cursor commit; verify live status before P1.3
- task_spec_version: SHA-256 `ad3f40ef1c4c06b17d97b68523aadbe00ccc1659a56ffa96b2f9ff2fcb34802b`
- active_adr: `docs/adr/ADR-0001-greenfield-rewrite.md` (`accepted`)

P1 remains `in_progress`. P1.1 established the package/contracts/audit boundary and P1.2 established
deterministic reference-canvas and spatial primitives. No real Small benchmark, source-specific
materializer, split, duplicate grouping, task expansion, language subset, validator or summary exists.
P1.2 performed zero physical deletions.

## Objective

Freeze a deterministic and auditable spatial contract for Canonical Benchmark v3: choose one
reference grid from explicit evidence, record crop/resize/pad transitions, preserve mask/valid
semantics, expose coordinate inverse availability, and isolate Qwen `[0,1000]` conversion at the
serialization boundary.

## Scope for P1.2

### Allowed

- strict reference-canvas candidate and decision contracts
- deterministic native-mask/registered-mask/single-language reference selection
- crop/resize/pad coordinate and CPU raster reference primitives
- fixed bilinear image and nearest-only mask/valid resampling
- padding/nodata exclusion and global-only support constraints
- reference half-open/Qwen-1000 bbox conversion
- synthetic CPU tests, P1.2 report, README, progress, handoff and deletion-gate evidence

### Explicitly excluded

- real Small/Full benchmark construction or raw-source mutation
- source-specific materializers
- split, duplicate grouping, task expansion and language-subset materialization
- P2 model/training/evaluation/CUDA work
- old runtime compatibility, physical legacy deletion, push, paid API or expert action

## Changes

### Files added

- `src/sami_gsd/contracts/spatial.py`
- `src/sami_gsd/data/reference_canvas.py`
- `src/sami_gsd/data/transforms.py`
- `tests/p1/test_spatial.py`
- `docs/reports/p1/p1_2_spatial_report.json`

### Files modified

- `schemas/canonical_parent_v3.schema.json`
- `src/sami_gsd/contracts/__init__.py`
- `src/sami_gsd/contracts/canonical.py`
- `src/sami_gsd/data/__init__.py`
- `tests/p1/test_contracts.py`
- `README.md`
- `docs/audits/deletion_plan.yaml`
- `docs/handoffs/P1.md`
- `REFACTOR_PROGRESS.md`

### Files deleted

- None.

## Commands executed

| command | exit code | result |
|---|---:|---|
| branch/status/HEAD and baseline-ref checks | 0 | clean P1.1 start at `dc02e05...`; both baseline refs still resolve to approved SHA |
| complete authority/current-contract reads | 0 | no governing-document conflict found for P1.2 |
| `conda run -n qwen3vl env PYTHONPATH=src python -m unittest discover -s tests/p1 -v` | 0 | final run: 32/32 passed |
| two independent synthetic spatial trace constructions | 0 | identical aggregate SHA-256 `d5bcd5e9ed6c9a2b5b93ec1d96b038bb893b033e56ac19dbbbf8ddb5f0e92d09` |
| `conda run -n qwen3vl env PYTHONPATH=src python -m compileall -q src/sami_gsd tests/p1` | 0 | import/bytecode compilation passed |
| draft-2020-12 canonical schema check | 0 | schema valid |
| old-runtime/prohibited-temporal-task/machine-path scans | 1 | expected no-match exit; no forbidden greenfield runtime match |
| deletion-manifest 36-entry null approval/deleted-commit assertion | 0 | every entry remains unapproved and undeleted |
| first local commit attempt inside read-only Git sandbox | 128 | `.git/index.lock` could not be created; no repository mutation |
| approved escalated local implementation commit | 0 | `ac4be61e6fd994408934885095563311b9e43ebe`; no push |
| `git diff --check` | 0 | passed before implementation commit and during documentation update |

## Tests

| contract/gate | status | evidence |
|---|---|---|
| reference selection determinism | passed | input permutations produce equal decision and candidate-set hash |
| reference priority | passed | authoritative native mask wins; registered grids prefer complete coverage then finest GSD |
| ambiguous reference fail-closed | passed | incomplete/incomparable multi-grid cases raise explicit errors |
| T1-T4 eligibility | passed | missing coordinate inverse is recorded and spatial task derivation is rejected |
| TransformChain continuity/endpoints | passed | discontinuity, endpoint mismatch and ambiguous interpolation are rejected |
| transform round-trip | passed | retained-content points and half-open boxes round-trip through crop/resize/pad |
| padding inverse domain | passed | padded coordinates have no fabricated source-space inverse |
| image interpolation | passed | explicit HWC image uses bilinear half-pixel-center sampling |
| mask/valid interpolation | passed | explicit HW binary masks remain binary under nearest-only sampling |
| padding/nodata exclusion | passed | zero padding and nodata are absent from the effective target population |
| global-only support | passed | unregistered support cannot expose pixel-level transforms |
| Qwen-1000 bbox boundary | passed | coverage preserved with per-edge error bounded by `ceil(axis/1000)` across representative scales |
| schema and deterministic regression | passed | strict schema accepts audited chain; two trace hashes are identical |
| complete P1 regression | passed | 32/32 unittest-compatible tests |

## Smoke / micro-overfit

- config: synthetic spatial fixtures only
- device: CPU
- steps: not applicable
- peak_vram: not measured
- result: no model exists in P1.2; GPU smoke and micro-overfit were not run

## Data and artifact bindings

- report: `docs/reports/p1/p1_2_spatial_report.json`
- implementation commit: `ac4be61e6fd994408934885095563311b9e43ebe`
- implementation aggregate SHA-256: `e9bdefbd9de3c4f46bfb1919ff8eb38c2d308d88da0f718322dca8590748e761`
- synthetic spatial aggregate SHA-256: `d5bcd5e9ed6c9a2b5b93ec1d96b038bb893b033e56ac19dbbbf8ddb5f0e92d09`
- real raw source audit: not run
- canonical Small benchmark: not built
- checkpoint/config for model training: not applicable

## Blockers

- P1.3 is not yet authorized; raw-source structures and source-specific materializers may not be changed.
- Source-specific training/evaluation eligibility remains closed pending exact license evidence and human decisions.
- `pytest` remains absent from `qwen3vl`; the complete unittest-compatible suite passed without installing dependencies.
- P1 acceptance remains blocked on materialization, split, duplicates, task/language views, validation and summary.

## Human action required

- Review and confirm or revise the proposed P1.3 scope in `docs/handoffs/P1.md`.
- Do not approve a raw source for training merely because an adapter can inventory it.
- No P1.2 deletion or push action is required.

## Next exact command

Optional P1.2 reproduction from the repository root:

```bash
conda run -n qwen3vl env PYTHONPATH=src python -m unittest discover -s tests/p1 -v
```

The next implementation action is the proposed P1.3 confirmation bundle in `docs/handoffs/P1.md`;
it must be accepted before editing P1.3 files or reading raw source contents beyond scoped inspection.

## Next phase scope

- proposed subtask: P1.3 raw-source structure audit and source-adapter/materializer boundary
- inspect source structures read-only, define strict adapter outputs, bind source grouping/modality/mask metadata,
  and implement only audit-mode canonical candidate generation with synthetic/source-minimal tests
- keep every source fail-closed for training until exact license eligibility is separately accepted
- do not split, deduplicate, expand tasks, build full Small/Full, start P2 or delete legacy paths

## Known technical debt

- The dependency-free nested-sequence raster implementation is the CPU policy oracle, not a high-throughput materializer.
- Static JSON Schema cannot compare transform-chain endpoints or array member IDs; Pydantic owns those cross-field invariants.
- Qwen-1000 integer conversion is coverage-preserving and may expand each edge by the declared quantization bound.
- Parent/task JSONL materializers, source grouping, duplicate clustering and formal validators do not yet exist.
