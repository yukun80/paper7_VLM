# REFACTOR_PROGRESS

## Current status

- phase: P0
- phase_status: human_accepted
- current_branch: `master`
- current_commit: `fab0ae7ce4ca17715d3fb52e5834b5110f2094d9`
- baseline_tag: verified `pre-sami-rewrite-2026-07-20` -> `0c53624dd93159f78acd6d39a579b100d7e3255f`
- baseline_branch: verified `baseline/sane-qmef-pmrd-mgrr` -> `0c53624dd93159f78acd6d39a579b100d7e3255f`
- dirty_worktree: yes; P0 acceptance/license evidence is awaiting the authorized local acceptance commit
- task_spec_version: SHA-256 `ad3f40ef1c4c06b17d97b68523aadbe00ccc1659a56ffa96b2f9ff2fcb34802b` at current commit `fab0ae7ce4ca17715d3fb52e5834b5110f2094d9`
- active_adr: `docs/adr/ADR-0001-greenfield-rewrite.md` (`accepted`)

`human_accepted` means the owner approved ADR-0001, the legacy baseline, Apache-2.0 for greenfield project code, gated phase-by-phase deletion, and the P1.1 implementation scope. The root license does not authorize datasets, weights, legacy code, or third-party assets. P1.1 performs no physical deletion.

## Objective

Audit the live repository, adjacent raw data, licenses, legacy code/config/CLI/tests, preserved artifacts, and deletion gates; freeze the proposed SAMI-GroundSegDesc greenfield decision without changing or deleting implementation code.

## Scope for this session

### Allowed

- `docs/audits/**`
- `docs/adr/**`
- `docs/handoffs/P0.md`
- `REFACTOR_PROGRESS.md`

### Explicitly excluded

- Any source, schema, config, test, script, README, or AGENTS change outside the allowed paths.
- Any deletion.
- P1 Benchmark v3 implementation.
- Benchmark builders, validators, tests, smoke runs, CUDA/environment probes, training, evaluation, paid APIs, expert review/merge, or full benchmark construction.
- Tag, branch, commit, or push without human approval.

## Changes

### Files added

- `docs/audits/repo_inventory.json`
- `docs/audits/reuse_matrix.md`
- `docs/audits/license_matrix.md`
- `docs/audits/deletion_plan.yaml`
- `docs/adr/ADR-0001-greenfield-rewrite.md`
- `REFACTOR_PROGRESS.md`
- `docs/handoffs/P0.md`

### Files modified

- None; every P0 output was absent at audit start.

### Files deleted

- None.

## Commands executed

The audit used read-only inspection except for creating the authorized documentation directories/files.

| command | start/end | exit code | result artifact |
|---|---|---:|---|
| `git status --short --branch` | 2026-07-20 / time not instrumented | 0 | clean `master...origin/master` at audit start |
| `git rev-parse HEAD` | 2026-07-20 / time not instrumented | 0 | `0c53624dd93159f78acd6d39a579b100d7e3255f` |
| `git ls-files | wc -l` | 2026-07-20 / time not instrumented | 0 | 268 tracked baseline files |
| `git ls-files -z | sort -z | xargs -0 sha256sum | sha256sum` | 2026-07-20 / time not instrumented | 0 | `d835fa1b52feb0da825fa92f1c44eb7815e1036f6f900f0f15ea5b10ff9f399c` |
| `git status --short --ignored` | 2026-07-20 / time not instrumented | 0 | relevant ignored outputs, weights, logs, PDFs, and bytecode inventoried |
| `rg --files` and focused `rg -n` scans over package/scripts/configs/tests/docs | 2026-07-20 / time not instrumented | 0 | class/function/CLI/config/test/deletion inventory |
| `find external -maxdepth 2 ...` guarded by directory existence | 2026-07-20 / time not instrumented | 0 | `external/` absent |
| `find /home/yukun80/codes/benchmark ...` | 2026-07-20 / time not instrumented | 0 | benchmark root exists and is empty |
| per-source `find ... -type f | wc -l` and `du -sh` under `/home/yukun80/codes/datasets` | 2026-07-20 / time not instrumented | 0 | nine local source visibility counts/sizes |
| top-level raw-source layout scan and local license-document `sha256sum` | 2026-07-20 / time not instrumented | 0 | nine source roots bound; Sen12Landslides and DisasterM3 license evidence hashes recorded |
| focused `sha256sum` over preserved reports, manifests, checkpoints, and local model cards/configs | 2026-07-20 / time not instrumented | 0 | hashes recorded in `repo_inventory.json` |
| read-only official upstream license-page audit | 2026-07-20 / time not instrumented | 0 | evidence links recorded in `license_matrix.md` |
| `mkdir -p docs/audits docs/adr docs/handoffs` | 2026-07-20 / time not instrumented | 0 | authorized P0 document directories |
| `python3 -m json.tool docs/audits/repo_inventory.json` | 2026-07-20 / time not instrumented | 0 | inventory JSON syntax valid |
| `python3 -c "... import yaml ..."` | 2026-07-20 / time not instrumented | 1 | default shell lacks PyYAML; no dependency was installed |
| `ruby -e 'require "yaml"; ...'` | 2026-07-20 / time not instrumented | 127 | Ruby is absent; no dependency was installed |
| standard-library structural assertions over `deletion_plan.yaml` | 2026-07-20 / time not instrumented | 0 | 36 entries; required fields present; approvals/deleted commits null; even indentation; no tabs |
| standard-library inventory binding verifier | 2026-07-20 / time not instrumented | 0 | all 16 recorded report/checkpoint/model-card/config paths exist and match SHA-256 |
| standard-library deletion-target coverage replay against `git ls-files` | 2026-07-20 / time not instrumented | 0 | 36 targets cover all 242 tracked paths under the audited legacy roots; 0 missing and 0 nonexistent targets |
| `rg -n '[[:blank:]]+$' ...` | 2026-07-20 / time not instrumented | 1 | expected no-match exit; no trailing whitespace found in P0 outputs |
| `git diff --check` | 2026-07-20 / time not instrumented | 0 | tracked diff whitespace check passed; new P0 files separately covered by the trailing-whitespace scan |
| user: `conda run -n qwen3vl python -c "... yaml.safe_load(...); ..."` | 2026-07-20 / user-reported | 0 | full PyYAML parse succeeded with `deletion manifest valid: 36` |
| user Git synchronization | 2026-07-20 / verified from repository | 0 | P0 outputs committed and pushed as `fab0ae7ce4ca17715d3fb52e5834b5110f2094d9`; worktree was clean before this evidence update |

## Tests

| test | status | evidence |
|---|---|---|
| JSON syntax for `repo_inventory.json` | passed | `python3 -m json.tool`, exit 0 |
| Inventory path/hash replay | passed | 16 current files reopened and SHA-256 matched, exit 0 |
| YAML syntax and structure for `deletion_plan.yaml` | passed | user-provided `qwen3vl` PyYAML parse succeeded with 36 entries and all approval/deletion fields null; prior structural assertions also passed |
| Deletion-manifest tracked-path coverage | passed | 242/242 legacy tracked paths covered; zero missing and zero nonexistent current targets; this is coverage evidence, not approval |
| New-file trailing whitespace scan | passed | `rg` found no matches; exit 1 is the expected no-match status |
| `git diff --check` | passed | exit 0; tracked diff only because P0 outputs are untracked |
| Unit/integration/smoke/model tests | not run by policy | P0 is documentation-only and AGENTS delegates program execution to the user |

## Smoke / micro-overfit

- config: not applicable
- device: not probed
- steps: 0
- peak_vram: not measured
- result: not run; outside P0 scope

## Data and artifact bindings

- benchmark: `/home/yukun80/codes/benchmark` exists but has zero entries; no live v3 benchmark exists
- manifest_sha256: legacy report/manifest hashes are listed individually in `docs/audits/repo_inventory.json`; no v3 manifest exists
- config_sha256: no v3 config exists; local upstream model config hashes are recorded in the inventory
- checkpoint: preserved legacy segmentation, D-1, and D0 checkpoints only
- checkpoint_sha256: `9ec3c766e6ec9d9475c3128e615e69f8b8e0d7ed376d86d1ed74b31c744f65e2`, `c6f1db5bd97b4c96ec171066012ad201b1bc782ff084daa834ed5d1ed55c6246`, `4d88bc2aa26a583c0b6d02b2eb8a4d23229873a3d7cd4129e1e68c5db6c121ce`, `8ffd1c342ff471687d01e2c5167ff3a0aa74690078432cd6cdc2cd7f1601585f`
- interpretation: preserved legacy bytes only; not a v3 initialization or scientific-success claim

## Blockers

- Most raw-source licenses are unresolved; every source remains non-training-eligible until P1 closes an exact registry entry.
- The benchmark root is empty, so preserved legacy reports cannot currently be replayed against their bound inputs.
- Weight and benchmark publication remain blocked by asset-specific terms even though the greenfield root code license is accepted.

## Human action required

- No additional P0 decision is required.
- Later, approve or reject source-by-source restricted/unknown data only after P1 produces exact evidence; no approval is inferred now.

## Next exact command

```bash
git switch -c refactor/sami-groundsegdesc
```

Run only after the authorized local P0 acceptance commit is clean; Codex will perform and verify it in this task.

## Next phase scope

- P1.1 is authorized: create the greenfield package/CLI, canonical contracts/schemas/configs, and deterministic read-only source audit with fail-closed licenses.
- P1 remains `in_progress`; P1.2 and later work require a separate task confirmation.
- P1.1 may add replacements but must physically delete zero legacy files.

## Known technical debt

- No root license or notice.
- No root `pyproject.toml`, `src/sami_gsd/`, v3 schemas/configs, or `sami-gsd` CLI yet; these are future-phase work, not P0 omissions.
- Local raw source licenses and component provenance are incomplete.
- Legacy report inputs are absent from the adjacent benchmark root.
- Legacy code/config/CLI/tests/docs remain intentionally present until their manifest gates open.
