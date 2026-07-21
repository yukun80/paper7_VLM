# REFACTOR_PROGRESS

## Current status

- phase: P1
- phase_status: `blocked_on_human_data_license_gate`
- completed_internal_checkpoint: complete P1 engineering pipeline including component-license-bound language parents, independent report replay and synthetic Small acceptance (`engineering_passed`)
- active_internal_work_package: real Small license preflight and canonical build
- current_branch: `refactor/sami-groundsegdesc`
- p1_component_license_commit: `ea185efa6dc8a89bc7ade1dd5ae5c52264f578b0`
- p1_language_parent_commit: `487b309d7b99f120367e2cf5b137c3e4b92f2e98`
- p1_continuous_engineering_commit: `c83c11a833f8fec12c8dbc46fbc54ee0fdff7c2c`
- p1_human_license_gate_docs_commit: `430a9cc70f3ae23256a43e4d4ea6eb8ef79c825d`
- p1_3_implementation_commit: `de64ddf33474d59e796831d1f2b6d7b0abd09e46`
- p1_3_handoff_commit: `273a4a03294338a7e4a382b89f2bab1b0361dff2`
- p1_2_implementation_commit: `ac4be61e6fd994408934885095563311b9e43ebe`
- p1_2_handoff_commit: `5ad6af944199e99c9815f3d1df5b1d62565767ba`
- p1_1_implementation_commit: `898f5b83820760ca86d1d488fc4cee0e8fa5cc9e`
- p1_1_handoff_commit: `eb7b5feabeb6b2209195ed4e42beb893ee3a2a9f`
- p0_acceptance_commit: `52c93b3a77635c82eb591850e758d3333482d4b1`
- baseline_tag: verified `pre-sami-rewrite-2026-07-20` -> `0c53624dd93159f78acd6d39a579b100d7e3255f`
- baseline_branch: verified `baseline/sane-qmef-pmrd-mgrr` -> `0c53624dd93159f78acd6d39a579b100d7e3255f`
- dirty_worktree_after_gate_docs_commit: no at `430a9cc70f3ae23256a43e4d4ea6eb8ef79c825d`
- task_spec_version: SHA-256 `ad3f40ef1c4c06b17d97b68523aadbe00ccc1659a56ffa96b2f9ff2fcb34802b`
- active_adr: `docs/adr/ADR-0001-greenfield-rewrite.md` (`accepted`)

P1 remains the only formal execution unit. Internal labels previously written as P1.1--P1.3 are
implementation checkpoints only: they are not phases, acceptance points, handoffs or reasons to
wait for another user task. All planned engineering work packages now have implementations and focused
tests, including an atomic synthetic Small build whose independent replay has `errors=[]`. The real
Small benchmark was not built: all nine live source registry rows remain
`allowed_for_training=false`, so the formal build preflight stops before raw decode or output writes.
This is the explicit `license unknown for requested training source` human stop condition, not an
internal checkpoint stop and not P1 completion.

## Current continuous-P1 evidence

- complete focused regression: 57/57 passed in `qwen3vl`;
- live bounded registry audit: 9 present, 7 sampled, 2 blocked, 14 audit candidates, 0 eligible,
  `errors=[]`;
- repeated live aggregate SHA-256:
  `4e2edbe2549313db49bb8e97144f0d6f2429d2aa0dadaddcdb1977f4f44c54fc`;
- synthetic end-to-end Small: two independent builds have identical aggregate hashes; independent
  validator replay has `errors=[]`, cross-split verified duplicates 0 and training-eligible unknown 0;
- licensed-language synthetic build: every selected row is bound to one exact component license;
  aggregate MMRS/RSGPT permission cannot promote siblings; exact-image rows with one license snapshot
  share a visual parent, RSIEval forces its verified duplicate component to test, canonical training rows
  use only `assets/...`, and DIOR remains box/short phrase without a maskless T2;
- independent replay reconstructs duplicate clusters, group split, evaluation conditions, component
  license report, summary pixel counts and all canonical-description provenance; clean replay has `errors=[]`;
- real preflight: `SourceLoadingError: no training-eligible spatial source; approve exact source
  license evidence before a formal Small build`;
- status report: `docs/reports/p1/p1_continuous_engineering_status.json`.
- exact human decision form: `docs/audits/p1_human_data_license_decision_request.md`.

## Active objective

Continue the frozen P1 sequence from the completed bounded source audit through source metadata and
grouping closure, materialization, valid-mask propagation, parent split, duplicate clustering,
T1--T4 expansion, the frozen description subset, validation, deterministic rebuild hashing and
Small acceptance. Internal checkpoints are recorded in `docs/worklogs/P1_CONTINUATION.md` and do
not interrupt execution.

## Continuous P1 scope

### Allowed

- complete P1 contracts, raw-source readers, materialization, split, duplicates, tasks and language subset
- synthetic CPU integration, independent validation replay and deterministic rebuild checks
- bounded live read-only source inspection and fail-closed real-build preflight
- README, P1 progress/worklog/report and deletion-gate evidence

### Explicitly excluded

- source license approval
- real Small/Full construction before the exact source license gate is approved
- P2 model/training/evaluation/CUDA work
- compatibility shims, physical deletion, push, paid API or expert action

## Changes

### Files added/replaced by the latest internal work package

- `schemas/canonical_description_v2.schema.json` replaces the unaccepted v1 draft with the
  component-license-bound runtime contract.

### Files modified by the latest internal work package

- `src/sami_gsd/contracts/__init__.py`
- `src/sami_gsd/contracts/language.py`
- `src/sami_gsd/contracts/config.py`
- `src/sami_gsd/data/audit.py`
- `src/sami_gsd/data/materialize.py`
- `src/sami_gsd/data/builder.py`
- `src/sami_gsd/data/language_subset.py`
- `src/sami_gsd/data/tasks.py`
- `src/sami_gsd/data/validation.py`
- `src/sami_gsd/cli.py`
- `configs/benchmark_v3_small.yaml`
- `configs/benchmark_v3_full.yaml`
- `tests/p1/test_builder_validation.py`
- `tests/p1/test_configs.py`
- `tests/p1/test_contracts.py`
- `tests/p1/test_language_canonical_build.py`
- `tests/p1/test_language_subset.py`
- `docs/audits/p1_human_data_license_decision_request.md`
- `README.md`
- `REFACTOR_PROGRESS.md`
- `docs/worklogs/P1_CONTINUATION.md`
- `docs/reports/p1/p1_continuous_engineering_status.json`

### Replaced path

- `schemas/canonical_description_v1.schema.json` was replaced by v2 before any real benchmark
  acceptance; this is not a deletion-manifest legacy removal.

## Commands executed

| command | exit code | result |
|---|---:|---|
| complete governing-document and current-state reread | 0 | no ADR/spec conflict; P1 and license stop remain authoritative |
| focused component-license and canonical-language build tests | 0 | partial approval, aggregate override, registry tamper and report replay passed |
| `conda run -n qwen3vl env PYTHONPATH=src python -B -m unittest discover -s tests/p1 -t . -v` | 0 | final run 57/57 passed |
| `conda run -n qwen3vl env PYTHONPATH=src python -B -m compileall -q src/sami_gsd tests/p1` | 0 | source and P1 tests compiled |
| Draft 2020-12 check of `canonical_description_v2.schema.json` | 0 | schema valid |
| two bounded live audit runs (`limit_per_source=2`) | 0 | identical aggregate; 9 sources, 7 sampled, 2 blocked, 14 candidates, 0 eligible |
| fail-closed spatial source preflight | 0 | expected `SourceLoadingError`; no raw decode/output write |
| deletion-manifest assertion | 0 | 36 entries remain unapproved and undeleted |
| `git diff --check` | 0 | passed before local implementation commit |
| local component-license implementation commit | 0 | `ea185efa6dc8a89bc7ade1dd5ae5c52264f578b0`; no push |

## Tests

| contract/gate | status | evidence |
|---|---|---|
| exact registry coverage | passed | all nine config keys match one unique registry key |
| no fallback | passed | unknown keys raise; blocked adapters refuse extraction |
| strict audit-only records | passed | training promotion and extra fields rejected |
| bounded format probes | passed | signature-based PNG/JPEG and header-only NPY paths |
| supported synthetic layouts | passed | GDCLD, LMHLD, LandslideBench, MMRS and RSGPT |
| explicit blockers | passed | Sen12, Landslide4Sense, multimodal and DisasterM3 remain closed |
| raw immutability | passed | synthetic bytes exact; live sampled assets rehashed unchanged |
| path portability | passed | published artifacts contain logical `datasets/...` paths only |
| repeat hash | passed | exact live aggregate repeated |
| canonical language parent | passed | exact rows sharing one license snapshot share a visual parent; runtime refs and answer provenance replay |
| RSIEval test isolation | passed | verified duplicate connected component is forced to test |
| DIOR role isolation | passed | reference box/short phrase only; no fabricated T2 mask |
| denied language source | passed | audit row retained without raw decode or materialization |
| component-license isolation | passed | aggregate override rejected; one-component approval promotes only that component |
| independent report replay | passed | duplicate/split/conditions/license/summary are recomputed; tampering is rejected |
| complete P1 regression | passed | 57/57 tests |

## Smoke / micro-overfit

- config: synthetic source layouts plus bounded live source samples
- device: CPU
- steps: not applicable
- peak_vram: not measured
- result: no P2 model exists; GPU smoke and micro-overfit were not run

## Data and artifact bindings

- live structure audit: `docs/audits/p1_source_structure_audit.json`
- P1.3 report: `docs/reports/p1/p1_3_source_adapter_report.json`
- continuous engineering base: `c83c11a833f8fec12c8dbc46fbc54ee0fdff7c2c`
- canonical language-parent implementation: `487b309d7b99f120367e2cf5b137c3e4b92f2e98`
- component-license and independent replay implementation: `ea185efa6dc8a89bc7ade1dd5ae5c52264f578b0`
- live adapter aggregate SHA-256: `3335535bc7e8fc3ba337511081dc5acd9d83129859095f46d4c017116a9eaf5a`
- blocked sample file-set aggregate SHA-256: `9d11988bce7b4436e405a1302f386537385c62a2707dcd9e36cbb17b5c6f615d`
- canonical Small benchmark: not built
- checkpoint/model config: not applicable

## Blockers

- All nine live source licenses remain `allowed_for_training=false`; Codex made no eligibility decision.
- Sen12 engineering is now resolved to annotated S2/ASC/DSC triplets and one event-nearest acquisition per
  modality within a 30-day window, with no paired change input. Human approval of its CC-BY-4.0 plus
  Sentinel/Copernicus/DEM obligations is still required before training use.
- All eight MMRS/RSGPT component policies remain unapproved. Their aggregate containers are now
  contractually incapable of granting training/evaluation/redistribution permission.
- GDCLD, LMHLD, Landslide4Sense and derived LandslideBench also retain grouping/provenance and/or license blockers.
- P1 lacks only a licensed real Small build, its second deterministic build, and final real acceptance reports.

## Human gate required before execution can continue

- Approve exact source-by-source license evidence and permitted task roles before any training-eligible record.
- The narrowest ready path is an explicit owner decision on Sen12Landslides research training/evaluation and
  redistribution policy, including required Sentinel, Copernicus and DEM attribution.
- Do not treat audit candidate availability as data-use authorization.
- No deletion or push action is required.

## Resume command after the human license decision is recorded

```bash
sami-gsd data build --config configs/benchmark_v3_small.yaml
```

Do not run it against the unchanged registry: it is expected to fail closed. The next formal phase and
formal P1 handoff remain unset until two real Small builds and all P1 gates pass.

## Known technical debt

- Live audit adapters intentionally remain sample-bounded; formal spatial materialization is currently connected
  to the resolved Sen12 source loader.
- Canonical language materialization and per-component replay are implemented, but live MMRS/RSGPT rows
  correctly remain audit-only while their eight component licenses are unapproved.
- Unknown/group-ambiguous candidates remain audit-only and cannot enter training.
- P1 acceptance remains blocked by the human data-license decision and the resulting real Small construction.
