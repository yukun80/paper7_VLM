# ADR-0004: HDF5 Rebaseline and Clean P3 Reset

- Status: accepted
- Date: 2026-07-24
- Owner: project owner
- Scope: P0R governance, data eligibility, phase reset, and replacement-owned deletion

## Context

The committed P1/P2 line and the uncommitted P3 work were designed around an earlier benchmark
generation. The current benchmark root is empty, while five locally converted HDF5 source cohorts
are non-empty. Continuing to patch the old P3 would preserve assumptions that have not been
validated against those assets and would grow an inactive candidate/config/schema framework.

Three sources also lack complete per-sample location, canonical parent/group, or duplicate-component
evidence. Those gaps reduce the credibility of evaluation splits, but do not invalidate the image
and mask pairs as segmentation training data.

## Decision

1. Establish a new incompatible HDF5-first line:

   ```text
   P0R -> P1 HDF5 Benchmark v4 -> P2 direct-dense sanity
       -> P3 kernel selection -> P4 training -> P5 evaluation
   ```

   Description, Bridge, SegDesc, and joint training remain provisional until the P3 kernel is
   frozen.

2. Separate four data decisions:

   - `ingestion_status`;
   - `canonical_split`;
   - `split_assurance`;
   - `evaluation_eligibility`.

   Missing group/location/duplicate evidence does not block training. It prevents a cohort from
   being reported as strictly group-isolated.

3. Preserve native source splits without random re-splitting:

   - GDCLD, LMHLD, and LandslideBench_agent keep train/val/test as
     `source_declared_unverified`;
   - Multimodal keeps train/val as `source_declared_unverified`;
   - Landslide4Sense contributes all 3,799 samples as `train_only`;
   - empty Sen12 remains `not_ready`.

4. Reset the uncommitted P3 completely. The two authorities are rewritten first; all other tracked
   P3 edits are restored to HEAD by exact path; all inventoried untracked P3 configs, schemas,
   modules, tests, ADR drafts, worklog, reports, caches, and checkpoints are deleted. No P3 patch
   backup is created.

5. The discarded ADR-0004 through ADR-0008 files were untracked drafts and never accepted project
   decisions. This accepted decision therefore uses the next committed number, ADR-0004, rather
   than inventing an ADR-0009 succession.

6. P1 implements only a reference-first HDF5 Canonical Benchmark v4. P2 implements only a minimal
   convolutional direct-dense sanity baseline. P3 candidates are written from zero and may not
   reuse discarded P3 names, schemas, configs, or checkpoints.

7. Deletion is replacement-owned. Once the replacement owned by a phase is accepted and its
   reference/test/doc gates pass, the corresponding old implementation becomes eligible for an
   explicit human-approved deletion. P8 receives only residual assets.

## Consequences

- Existing committed P1/P2 code and reports are historical evidence for the previous line, not
  evidence that Benchmark v4 or the new P2 has passed.
- P5 must publish strict and exploratory aggregates separately. If no strict cohort exists,
  `strict_generalization_status` is `unavailable`.
- Canonical identity contains only `datasets/...` logical paths and content hashes. Absolute
  machine paths observed in source-side conversion metadata remain audit findings only.
- The removed untracked P3 evidence is not recoverable from the current worktree. Its pre-delete
  inventory is preserved in `docs/audits/p0r_dirty_p3_reset_manifest.json`.
- No compatibility shim, candidate auto-switch, or pre-implementation of a future phase is
  permitted.

## Supersession

This ADR supersedes conflicting current-phase, dataset-admission, stage-order, and deletion-timing
clauses in older governance documents. ADR-0001 and ADR-0003 remain historical accepted decisions
only where they do not conflict with the two current refactor authorities and this ADR.

## Evidence

- `docs/audits/hdf5_source_audit.md`
- `docs/audits/hdf5_source_inventory.json`
- `docs/audits/hdf5_source_contract.yaml`
- `docs/audits/greenfield_reset_reuse_matrix.md`
- `docs/audits/p0r_dirty_p3_reset_manifest.json`
- `docs/handoffs/P0R.md`
