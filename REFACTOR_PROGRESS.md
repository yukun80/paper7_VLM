# REFACTOR_PROGRESS

## Current status

- phase: P1
- phase_status: `blocked_on_human_data_license_gate`
- completed_internal_checkpoint: complete P1 engineering pipeline through synthetic Small acceptance (`engineering_passed`)
- active_internal_work_package: real Small license preflight and canonical build
- current_branch: `refactor/sami-groundsegdesc`
- p1_3_implementation_commit: `de64ddf33474d59e796831d1f2b6d7b0abd09e46`
- p1_3_handoff_commit: `273a4a03294338a7e4a382b89f2bab1b0361dff2`
- p1_2_implementation_commit: `ac4be61e6fd994408934885095563311b9e43ebe`
- p1_2_handoff_commit: `5ad6af944199e99c9815f3d1df5b1d62565767ba`
- p1_1_implementation_commit: `898f5b83820760ca86d1d488fc4cee0e8fa5cc9e`
- p1_1_handoff_commit: `eb7b5feabeb6b2209195ed4e42beb893ee3a2a9f`
- p0_acceptance_commit: `52c93b3a77635c82eb591850e758d3333482d4b1`
- baseline_tag: verified `pre-sami-rewrite-2026-07-20` -> `0c53624dd93159f78acd6d39a579b100d7e3255f`
- baseline_branch: verified `baseline/sane-qmef-pmrd-mgrr` -> `0c53624dd93159f78acd6d39a579b100d7e3255f`
- dirty_worktree_at_resume: no at `88b75098bbd5ffa37507bc8dfec0215af713348b`
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

- complete focused regression: 52/52 passed in `qwen3vl`;
- live bounded registry audit: 9 present, 7 sampled, 2 blocked, 14 audit candidates, 0 eligible,
  `errors=[]`;
- repeated live aggregate SHA-256:
  `4e2edbe2549313db49bb8e97144f0d6f2429d2aa0dadaddcdb1977f4f44c54fc`;
- synthetic end-to-end Small: two independent builds have identical aggregate hashes; independent
  validator replay has `errors=[]`, cross-split verified duplicates 0 and training-eligible unknown 0;
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

## Scope for P1.3

### Allowed

- bounded reads of raw directory metadata, annotation/index samples and minimal sample assets
- old source readers only as structural cross-check evidence
- strict raw source, canonical candidate and projection contracts
- source adapter protocol/registry and deterministic sample extraction
- fail-closed license, temporal, grouping, raster metadata and scope blockers
- synthetic CPU and bounded live read-only integration tests
- P1.3 audit/report, README, progress, handoff and deletion-gate evidence

### Explicitly excluded

- full recursive source hashing or full Small/Full benchmark construction
- bulk asset materialization or raw-source mutation
- source license approval
- group split, duplicate clustering, task expansion or language-subset build
- P2 model/training/evaluation/CUDA work
- compatibility shims, physical deletion, push, paid API or expert action

## Changes

### Files added

- `src/sami_gsd/contracts/sources.py`
- `src/sami_gsd/data/adapters/__init__.py`
- `src/sami_gsd/data/adapters/base.py`
- `src/sami_gsd/data/adapters/formats.py`
- `src/sami_gsd/data/adapters/implemented.py`
- `src/sami_gsd/data/adapters/registry.py`
- `src/sami_gsd/data/adapters/audit.py`
- `tests/p1/test_source_adapters.py`
- `docs/audits/p1_source_structure_audit.json`
- `docs/reports/p1/p1_3_source_adapter_report.json`

### Files modified

- `src/sami_gsd/contracts/__init__.py`
- `src/sami_gsd/data/__init__.py`
- `README.md`
- `docs/audits/deletion_plan.yaml`
- `docs/handoffs/P1.md`
- `REFACTOR_PROGRESS.md`

### Files deleted

- None.

## Commands executed

| command | exit code | result |
|---|---:|---|
| authority/current-state/legacy-reader reads and bounded raw tree probes | 0 | no governing conflict; nine sources inspected read-only |
| bounded `file`, HDF5/NPY/NetCDF/GeoTIFF/JSON sample metadata probes | 0 | live structure evidence recorded without raw writes |
| base `PYTHONPATH=src python -B -m unittest discover -s tests/p1 -v` | 1 | wrong base Python 3.13 lacked PyYAML/jsonschema; test bodies did not run |
| `conda run -n qwen3vl env PYTHONPATH=src python -B -m unittest discover -s tests/p1 -v` | 0 | final run 37/37 passed |
| two live `audit_source_samples(..., limit_per_source=8)` runs | 0 | identical aggregate SHA-256; 5 sampled, 4 blocked, `errors=[]` |
| two in-memory SHA passes over 11 blocked-source sample files | 0 | all sample bytes equal |
| `git diff --check` before implementation commit | 0 | passed |
| local implementation commit | 0 | `de64ddf33474d59e796831d1f2b6d7b0abd09e46`; no push |
| local documentation/handoff commit | 0 | `273a4a03294338a7e4a382b89f2bab1b0361dff2`; no push |
| final complete P1 unittest regression | 0 | 37/37 passed after documentation commit |
| `python -B -m compileall -q src/sami_gsd tests/p1` in `qwen3vl` | 0 | source and P1 tests compiled |
| P1.3 JSON/hash/deletion-manifest assertions | 0 | both reports valid; 36 entries remain unapproved and undeleted |

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
| complete P1 regression | passed | 37/37 tests |

## Smoke / micro-overfit

- config: synthetic source layouts plus bounded live source samples
- device: CPU
- steps: not applicable
- peak_vram: not measured
- result: no P2 model exists; GPU smoke and micro-overfit were not run

## Data and artifact bindings

- live structure audit: `docs/audits/p1_source_structure_audit.json`
- P1.3 report: `docs/reports/p1/p1_3_source_adapter_report.json`
- implementation commit: `de64ddf33474d59e796831d1f2b6d7b0abd09e46`
- documentation/handoff commit: `273a4a03294338a7e4a382b89f2bab1b0361dff2`
- live adapter aggregate SHA-256: `3335535bc7e8fc3ba337511081dc5acd9d83129859095f46d4c017116a9eaf5a`
- blocked sample file-set aggregate SHA-256: `9d11988bce7b4436e405a1302f386537385c62a2707dcd9e36cbb17b5c6f615d`
- canonical Small benchmark: not built
- checkpoint/model config: not applicable

## Blockers

- All nine live source licenses remain `allowed_for_training=false`; Codex made no eligibility decision.
- Sen12 engineering is now resolved to annotated S2/ASC/DSC triplets and one event-nearest acquisition per
  modality within a 30-day window, with no paired change input. Human approval of its CC-BY-4.0 plus
  Sentinel/Copernicus/DEM obligations is still required before training use.
- MMRS components and restricted RSGPT remain audit-only until separate component/owner decisions.
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

- Live audit adapters intentionally remain sample-bounded; formal materialization is currently connected only
  to the resolved Sen12 source loader.
- Training-language image materialization remains fail-closed; selected MMRS/RSGPT records are audit-only while
  their component licenses are unapproved.
- Unknown/group-ambiguous candidates remain audit-only and cannot enter training.
- P1 acceptance remains blocked by the human data-license decision and the resulting real Small construction.
