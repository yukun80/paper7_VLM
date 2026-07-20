# ADR-0001: Adopt the incompatible SAMI-GroundSegDesc greenfield rewrite

- Status: accepted
- Date: 2026-07-20
- Owners: project maintainer
- Phase: P0
- Commit: P0 documentation base `fab0ae7ce4ca17715d3fb52e5834b5110f2094d9`; this accepted decision is recorded by the local P0 acceptance commit

## Context

The repository currently contains an engineering-complete candidate built around SANE, QMEF, PMRD, MGRR, Landslide Benchmark V2, Description M1.1, Bridge/Unified protocols, offline vision caches, and 36 `qpsalm-*` commands. Those assets are scientifically and operationally coupled to old schemas, configurations, checkpoints, caches, and class names.

The governing task specification freezes a different research question: same-area, single-time/contemporaneous multi-sensor observations must produce a landslide mask on one reference canvas and a region description limited to facts supported by the active modalities. It explicitly excludes pre/post change detection, video/tracking, old fixed-slot fusion, compatibility shims, and whole third-party repository embedding.

The live P0 audit found:

- a clean `master` worktree at `0c53624dd93159f78acd6d39a579b100d7e3255f` before P0 document creation;
- 268 tracked files bound by aggregate SHA-256 `d835fa1b52feb0da825fa92f1c44eb7815e1036f6f900f0f15ea5b10ff9f399c`;
- no local `external/` repositories;
- an empty adjacent `/home/yukun80/codes/benchmark` root, while preserved legacy reports reference now-missing benchmark packages;
- raw datasets and legacy output/model artifacts present beside or inside the repository;
- no root project license or notice;
- unresolved source licenses for most raw datasets and aggregate MMRS data.

The historical reports and checkpoints remain valuable for provenance and fair comparison, but their missing bound benchmark inputs mean P0 cannot promote them to current replay-verified evidence.

## Decision

The project owner accepted this ADR on 2026-07-20 and the baseline tag/branch were verified. The active development branch will implement an incompatible greenfield system named **SAMI-GroundSegDesc**.

The new main line has exactly three top-level model modules:

1. **Sensor-Aware Multi-Image Adapter** for arbitrary active optical, multispectral, SAR, DEM, slope, InSAR, or explicitly documented deformation observations represented through official Qwen3-VL native multi-image inputs and minimal auditable spatial support.
2. **Unified Grounded Segmentation**, with the final kernel selected only by G0/ADR-0002 between a Qwen3-VL-Seg-style candidate and PSALM-Lite.
3. **Mask-Grounded Multi-Source Region Reader**, using exact masks, full-image context, RoI-aligned feature replay, coverage, and explicit null states without restoring MGRR.

The implementation boundaries are frozen as follows:

- Qwen3-VL-2B is used through an official online native multi-image forward wrapper.
- The project uses a self-contained minimal Trainer, not ms-swift plus a second internal Trainer.
- The canonical data package is `SAMI Landslide Grounded Benchmark v3` under a new name and protocol.
- New code uses a root `src/sami_gsd/` layout and exposes only the `sami-gsd` CLI.
- Old Benchmark, reader, cache, config, checkpoint, CLI, protocol, and class names are incompatible and receive no adapters, aliases, migrations, or silent fallbacks.
- Legacy code is preserved through Git history, a baseline tag, and a baseline branch, not through a `legacy/` directory.
- Raw sources, accepted/historical artifacts, model cards/weights, source registries, licenses, expert-review provenance, and experiment logs remain read-only preservation assets.
- Unknown-license data may be inventoried but must fail closed for training eligibility.
- Deletion is controlled exclusively by `docs/audits/deletion_plan.yaml`; no P0 deletion is authorized.

## Alternatives considered

1. **Incrementally refactor SANE/QMEF/PMRD/MGRR in place.** Rejected because it would preserve old serialization and architectural coupling, encourage compatibility shims, and make scientific attribution and ablation boundaries ambiguous.
2. **Keep two active systems behind compatibility adapters.** Rejected because dual readers/trainers/protocols create silent fallback risk and make the new benchmark non-canonical.
3. **Vendor PSALM, GAR, Mask2Former, SAM2, or RSGPT repositories.** Rejected because whole-repository embedding adds dependency and license conflicts and violates the minimal greenfield boundary.
4. **Delete the legacy system immediately.** Rejected because replacement gates, human approval, a verified baseline tag/branch, reference scans, replacement tests, and replacement documentation do not yet exist.
5. **Reuse the preserved legacy benchmark/cache/checkpoints as v3 inputs.** Rejected because the new protocol must scan raw sources, and the current legacy reports bind benchmark paths that are absent.

## Evidence

- experiment/report: `docs/audits/repo_inventory.json` records current paths and hashes; preserved legacy reports are explicitly historical engineering evidence, not scientific success.
- license/dependency: `docs/audits/license_matrix.md` records verified upstream terms and all unresolved project/data decisions.
- reuse/deletion: `docs/audits/reuse_matrix.md` and `docs/audits/deletion_plan.yaml` define preservation, independent reimplementation, and gated removal.
- hardware: no GPU or environment probe was run in P0. The later implementation target remains one approximately 24 GiB GPU, subject to P7 measurement.
- governing design: `docs/REFACTOR_TASK_SPEC.md` and `docs/CODEX_REFACTOR_PROMPT.md`.

## Consequences

### Positive

- One canonical benchmark, package, Trainer, CLI, and lineage model replaces overlapping legacy protocols.
- Scientific novelty is easier to attribute because mature upstream ideas are reused narrowly and explicitly.
- Data-license, sensor, spatial-transform, valid-mask, and task-view provenance become first-class rather than inherited from old derived packages.
- The baseline remains reproducible from Git and preserved artifacts without burdening the active runtime with compatibility code.

### Negative

- Old checkpoints, caches, indexes, configuration files, and commands will not load in the new system.
- P1 must rebuild from raw data and close source licenses before training indexes exist.
- Historical reports whose bound inputs are missing cannot be replayed without a separately authorized restoration.
- The project cannot publish greenfield code, benchmarks, or weights until the root license and data-use decisions are approved.
- G0 still requires a separate ADR to select exactly one segmentation kernel and supported runtime profile.

## Implementation constraints

- Execute only P0-P8 in order and do not advance a phase without its stated acceptance evidence and human gates.
- Do not run formal training, paid APIs, expert review/merge, or full benchmark construction automatically.
- Do not modify raw data or overwrite accepted artifacts.
- Do not use old V1/V2/M1.1/Bridge/Unified/cache/checkpoint protocols as new runtime inputs.
- Do not add pre/post/change, recovery advice, video, tracking, bbox generation by Qwen, fixed five-slot concatenation, compatibility shims, silent fallback, or oracle outputs.
- P1 must reject `license_status=unknown` from every training-eligible index.
- G0 decisions are deferred to ADR-0002; this ADR does not choose Qwen3-VL-Seg-style versus PSALM-Lite, Profile S/M, or aligned support residual.
- Every deletion requires all manifest gates, resolved targets, reference scans, replacement tests/docs, a verified baseline, and named human approval.

## Backup and branch plan

The following local backup references were created and verified on 2026-07-20; they were not pushed by Codex:

```bash
git tag -a pre-sami-rewrite-2026-07-20 0c53624dd93159f78acd6d39a579b100d7e3255f -m "Freeze pre-SAMI greenfield legacy baseline"
git branch baseline/sane-qmef-pmrd-mgrr 0c53624dd93159f78acd6d39a579b100d7e3255f
git switch -c refactor/sami-groundsegdesc
```

The annotated tag and baseline branch both resolve to the owner-approved legacy SHA `0c53624dd93159f78acd6d39a579b100d7e3255f`. The refactor branch is created from the local P0 acceptance commit after it is written. Codex does not push these references automatically.

## Rollback

- tag/branch: verified tag `pre-sami-rewrite-2026-07-20` and branch `baseline/sane-qmef-pmrd-mgrr`, both resolving to `0c53624dd93159f78acd6d39a579b100d7e3255f`.
- trigger: frozen scientific scope changes; required source licenses cannot be obtained; G0 yields no viable 24 GiB candidate; replacement gates fail; deletion occurred without complete manifest approval; or greenfield work must be abandoned.
- procedure: stop mutation, preserve the failing refactor commit and reports, verify the baseline tag resolves to the approved SHA, switch to the baseline branch in a clean worktree, and restore any large accepted artifact only from its separately verified read-only path. Never use destructive reset or overwrite raw/accepted assets.

## Human approval

- approver: project owner (written confirmation in the active task)
- date: 2026-07-20
- approved baseline SHA: `0c53624dd93159f78acd6d39a579b100d7e3255f`
- baseline tag verified: `pre-sami-rewrite-2026-07-20` -> approved baseline SHA
- baseline branch verified: `baseline/sane-qmef-pmrd-mgrr` -> approved baseline SHA
- root license decision: Apache License 2.0 for greenfield project code and project-authored documentation; see root `LICENSE` and `NOTICE`
- restricted-data decision: not granted; every data/model/third-party asset remains separately licensed and fail-closed
- notes: Deletion is authorized only phase by phase through `docs/audits/deletion_plan.yaml`. P1.1 performs no physical deletion, and every entry-level `approved_by` and `deleted_commit` remains null.
