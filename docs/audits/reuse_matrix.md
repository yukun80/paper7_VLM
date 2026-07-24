# Greenfield Reuse Policy

- Generation: P0R v2
- Date: 2026-07-24
- Governing ADR: `docs/adr/ADR-0004-hdf5-rebaseline-and-clean-p3-reset.md`
- Exact current-code classification: `docs/audits/greenfield_reset_reuse_matrix.md`
- Deletion gates: `docs/audits/deletion_plan.yaml`

This document defines the reuse vocabulary and repository-level boundaries. It authorizes no
deletion and no compatibility path.

## Decisions

| Decision | Meaning |
|---|---|
| `KEEP_GENERIC` | Source-agnostic utility may remain, subject to static review. |
| `REBIND_AND_TEST` | Useful interface/logic must be rebound to Benchmark v4 and tested in its owning phase before use. |
| `REWRITE` | Existing implementation encodes the superseded v3/source design and must be replaced from zero. |
| `DELETE_AFTER_REPLACEMENT` | Asset stays until its named replacement is accepted and deletion gates pass. |
| `HISTORICAL_EVIDENCE_ONLY` | May be read for provenance; must not be imported or consumed by the new runtime. |
| `SEPARATE_LEGACY_GATE` | Legacy package remains outside the greenfield runtime and has its own replacement-owned deletion entry. |
| `DIRTY_P3_DISCARDED` | Uncommitted P3 bytes were inventoried and intentionally deleted in P0R. |

## Repository-level decisions

| Asset | Decision | Rule |
|---|---|---|
| `../datasets` | read-only source assets | Never modify; P1 binds HDF5 by logical path and hash. |
| `../benchmark` | generated output root | Empty at P0R; P1 creates a new immutable v4 package only after authorization. |
| `src/sami_gsd` committed P1/P2 code | mixed | Use the exact reset matrix; package location does not imply approval. |
| `scripts/1-benchmark` | `DELETE_AFTER_REPLACEMENT` | P1 v4 builder/validator must be accepted first. |
| `scripts/2-instruction` through `scripts/5-segdesc` | `DELETE_AFTER_REPLACEMENT` | Description/Bridge replacements are provisional; P1 segmentation does not satisfy these gates. |
| `SEG_Multi-Source_Landslides` | `SEPARATE_LEGACY_GATE` | No imports/copies/shims; delete components only when their owning segmentation/training/evaluation replacement passes. |
| old committed reports/handoffs/ADRs | `HISTORICAL_EVIDENCE_ONLY` | Do not use as current Benchmark v4 or model evidence. |
| P0R reset targets | `DIRTY_P3_DISCARDED` | See exact path/hash inventory; do not reconstruct. |
| `models_zoo` and reference papers | read-only external/reference assets | Not current runtime approval; any later use needs a phase-owned narrow interface. |

## Scientific invariants that may be reimplemented

- explicit channel order, meaning, unit/unknown-unit, normalization, and validity;
- strict separation of source sample, canonical parent, and verified group/location identity;
- valid-mask propagation through render, loss, and metric;
- no-target samples retained;
- source byte/hash binding and atomic artifacts;
- strict/exploratory/train-only evaluation separation;
- deterministic record identities and non-overwriting outputs.

These are requirements, not authorization to copy old implementations or serialized formats.

## Prohibited carry-over

- old benchmark/cache/checkpoint/schema readers;
- compatibility aliases or migration shims;
- hidden fixed-slot multimodal concatenation;
- automatic candidate switching or fallback;
- forced box-first segmentation;
- description/Bridge code in P1;
- future-stage framework added “for later”;
- machine absolute paths in canonical identity.

## Deletion policy

Every committed deletion requires:

1. the owning replacement is accepted;
2. exact targets are resolved;
3. reference/import scans are clean;
4. tests are replaced or explicitly retired;
5. live docs no longer depend on the target;
6. the project owner approves that entry;
7. `deleted_commit` remains `null` until a later real commit exists.

P8 receives only residual assets without an earlier satisfied replacement gate.
