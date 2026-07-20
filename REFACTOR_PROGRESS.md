# REFACTOR_PROGRESS

## Current status

- phase: P1
- phase_status: `in_progress`
- completed_subtask: `P1.1` (`engineering_passed`)
- current_branch: `refactor/sami-groundsegdesc`
- implementation_commit: `898f5b83820760ca86d1d488fc4cee0e8fa5cc9e`
- p0_acceptance_commit: `52c93b3a77635c82eb591850e758d3333482d4b1`
- baseline_tag: verified `pre-sami-rewrite-2026-07-20` -> `0c53624dd93159f78acd6d39a579b100d7e3255f`
- baseline_branch: verified `baseline/sane-qmef-pmrd-mgrr` -> `0c53624dd93159f78acd6d39a579b100d7e3255f`
- dirty_worktree: yes; this progress update and `docs/handoffs/P1.md` await the P1.1 handoff commit
- task_spec_version: SHA-256 `ad3f40ef1c4c06b17d97b68523aadbe00ccc1659a56ffa96b2f9ff2fcb34802b`
- active_adr: `docs/adr/ADR-0001-greenfield-rewrite.md` (`accepted`)

P1 remains `in_progress`: P1.1 introduced the greenfield contracts and read-only audit, but no real
Small benchmark exists. P1.2 and later P1 subtasks require a new explicit confirmation. P1.1
performed zero physical deletions.

## Objective

Establish the Canonical Benchmark v3 package boundary, strict parent/task contracts, portable and
fail-closed source/license audit, deterministic strict artifacts, and a single `sami-gsd` CLI before
any source-specific materialization or model implementation.

## Scope for P1.1

### Allowed

- root `pyproject.toml` and the single `sami-gsd` CLI under `src/sami_gsd/`
- `canonical_parent_v3` and `task_view_v3` schemas and Python contracts
- Small/Full audit configs and `scene_region_ontology_v2.yaml`
- read-only raw scanner, fail-closed license registry and atomic audit artifacts
- CPU/synthetic tests, P1.1 report, README, progress, handoff and deletion-gate evidence

### Explicitly excluded

- real Small/Full benchmark construction or source mutation
- reference-canvas materialization and transform engine (P1.2)
- split, duplicates, task expansion and language-subset materialization
- any P2 model/training/evaluation/CUDA work
- old runtime compatibility, physical legacy deletion, push, paid API or expert action

## Changes

### Files added

- `pyproject.toml`
- `configs/benchmark_v3_small.yaml`
- `configs/benchmark_v3_full.yaml`
- `configs/scene_region_ontology_v2.yaml`
- `schemas/canonical_parent_v3.schema.json`
- `schemas/task_view_v3.schema.json`
- `src/sami_gsd/__init__.py`
- `src/sami_gsd/cli.py`
- `src/sami_gsd/contracts/{__init__.py,canonical.py,config.py}`
- `src/sami_gsd/data/{__init__.py,audit.py}`
- `src/sami_gsd/utilities/{__init__.py,artifacts.py}`
- `tests/p1/{__init__.py,conftest.py,test_audit.py,test_configs.py,test_contracts.py}`
- `docs/reports/p1/p1_1_contract_report.json`
- `docs/handoffs/P1.md`

### Files modified

- `README.md`
- `docs/audits/deletion_plan.yaml`
- `REFACTOR_PROGRESS.md`

### Files deleted

- None.

## Commands executed

| command | exit code | result |
|---|---:|---|
| `git status --short --branch`; HEAD/ref checks | 0 | clean accepted P0 at `52c93b3...`; verified both baseline refs |
| `git switch -c refactor/sami-groundsegdesc` | 0 | new local refactor branch; no push |
| `conda run -n qwen3vl env PYTHONPATH=src python -m pytest -q tests/p1` | 1 | collection did not start because `pytest` is absent from `qwen3vl` |
| Pydantic/jsonschema/PyYAML import/version probe | 0 | Pydantic 2.11.3, jsonschema 4.26.0, PyYAML 6.0.2 |
| `conda run -n qwen3vl env PYTHONPATH=src python -m unittest discover -s tests/p1 -v` | 0 | final run: 15/15 passed |
| `conda run -n qwen3vl env PYTHONPATH=src python -m sami_gsd.cli data audit --help` | 0 | P1.1 CLI help passed |
| `python -m json.tool` for both schemas and the P1.1 report | 0 | strict JSON syntax passed |
| two independent `sami_gsd.cli data audit` synthetic runs | 0 | identical aggregate SHA-256 `523757400620d814aa32e289a867124b83758f24134be2a95c06af103d39e491` |
| runtime legacy/prohibited-task/machine-path scans and single-CLI assertion | 0 | no forbidden runtime matches; only `sami-gsd` is declared |
| P1.1 report hash replay and deletion-manifest PyYAML assertions | 0 | report bindings matched; 36 entries remain unapproved/undeleted |
| `git diff --check` | 0 | passed before implementation commit |
| local implementation commit | 0 | `898f5b83820760ca86d1d488fc4cee0e8fa5cc9e`; no push |

## Tests

| contract/gate | status | evidence |
|---|---|---|
| JSON Schema draft-2020-12 validation | passed | canonical parent and task fixtures validate; extras and training-eligible unknown are rejected |
| Pydantic strict contracts | passed | extra fields, invalid half-open boxes, invalid reference modality and nonportable paths rejected |
| missing versus zero-valid state | passed | state/coverage/asset invariants enforced |
| source license fail-closed | passed | unknown/unreviewed training eligibility rejected at config validation |
| read-only source scan | passed | source bytes identical before/after synthetic runs |
| stable ordering and repeat hash | passed | independent CLI runs produced the same aggregate SHA-256 |
| atomic/no-overwrite/finite JSON | passed | `.part` absent, existing targets rejected, NaN rejected |
| symlink boundary | passed | symlinked source root rejected; nested links are never followed |
| live configs/ontology/single CLI | passed | two nine-source configs fail closed; 25 ontology fields complete; one console script |
| pytest collection | environment warning | `pytest` is declared in test extras but not installed in the current environment |

## Smoke / micro-overfit

- config: synthetic audit fixture only
- device: CPU
- steps: not applicable
- peak_vram: not measured
- result: no model exists in P1.1; GPU smoke and micro-overfit were not run

## Data and artifact bindings

- report: `docs/reports/p1/p1_1_contract_report.json`
- implementation aggregate SHA-256: `a77514764930dd9ff07c297eff98812998b70b17a944e433cd9f6e49b7c73470`
- synthetic audit aggregate SHA-256: `523757400620d814aa32e289a867124b83758f24134be2a95c06af103d39e491`
- real raw source audit: not run
- canonical Small benchmark: not built
- training-eligible sources in committed configs: 0/9
- checkpoint/config for model training: not applicable

## Blockers

- P1.2 is not yet authorized; no reference-canvas assets or transforms may be materialized.
- Source-specific training/evaluation eligibility remains closed pending exact license evidence and human decisions.
- `pytest` is absent from `qwen3vl`; the full unittest-compatible suite passed without installing dependencies.
- P1 acceptance remains blocked on the remaining materializer, transform, split, duplicate, task, language, validator and summary subtasks.

## Human action required

- Review and confirm or revise the proposed P1.2 scope in the P1 handoff/final response.
- Do not approve a raw source for training merely because it appears in the audit inventory.
- No P1.1 deletion or push action is required.

## Next exact command

For optional reproduction after installing the declared test extra:

```bash
conda run -n qwen3vl env PYTHONPATH=src python -m pytest -q tests/p1
```

The next implementation command is the proposed P1.2 confirmation bundle in `docs/handoffs/P1.md`;
it must be accepted before editing P1.2 files.

## Next phase scope

- proposed subtask: P1.2 reference-canvas and spatial transform primitives
- implement deterministic reference selection, crop/resize/pad `TransformChain`, inverse metadata,
  nearest-only mask/valid propagation, and half-open/Qwen-1000 bbox boundary conversion
- use synthetic fixtures only; do not build the real benchmark
- do not implement split, duplicates, task expansion, language subsets, P2 code or deletion

## Known technical debt

- Static schemas and Pydantic contracts are strict, but parent/task JSONL materializers do not exist yet.
- The scanner hashes every visible file and has no incremental cache by design; a real full audit may be expensive.
- `benchmark_root` is validated for future builders but the P1.1 audit uses its explicit `--output-dir` only.
- Source registry rows remain conservative templates, not final legal determinations.
