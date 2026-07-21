# P1 continuous worklog

## Phase-control contract

- Formal phase: `P1`
- Status: `blocked_on_human_data_license_gate` after all independently executable engineering work
- Internal work-package names are implementation/test organization only.
- No internal checkpoint is a user acceptance point, stopping point, independent handoff or next task.
- Formal `docs/handoffs/P1.md`, the P1 completion report and the next accepted task are finalized only
  after every P1 acceptance criterion passes.

## Resume binding

- Branch: `refactor/sami-groundsegdesc`
- Resume implementation HEAD for the latest continuation: `ea185efa6dc8a89bc7ade1dd5ae5c52264f578b0`
- Worktree at resume: clean
- Continuous engineering commit: `c83c11a833f8fec12c8dbc46fbc54ee0fdff7c2c`
- Human license-gate documentation commit: `430a9cc70f3ae23256a43e4d4ea6eb8ef79c825d`
- Canonical language-parent implementation commit: `487b309d7b99f120367e2cf5b137c3e4b92f2e98`
- Component-license and independent-report-replay commit: `ea185efa6dc8a89bc7ade1dd5ae5c52264f578b0`
- Governing task-spec SHA-256:
  `ad3f40ef1c4c06b17d97b68523aadbe00ccc1659a56ffa96b2f9ff2fcb34802b`
- P1.3-named historical report (internal checkpoint only):
  `docs/reports/p1/p1_3_source_adapter_report.json`
- No physical deletion and no push are authorized.

## Completed internal work packages

1. Canonical Parent v3 and task-view schema contracts.
2. Read-only raw scanner and strict audit artifacts.
3. Deterministic reference-canvas selection.
4. Reversible crop/resize/pad transform chain and coordinate round trips.
5. Exact nine-source adapter registry with seven bounded projections and two fail-closed adapters.
6. Explicit HDF5/NetCDF/GeoTIFF data extra and header/grid/nodata readers.
7. Atomic preprocessing/materialization with fit-resize-pad, valid/nodata propagation and transformed affine.
8. SHA exact plus dHash recall plus RGB64/MAE verified duplicate connected components.
9. Scene/event/region/source-group/duplicate union followed by deterministic parent-level split.
10. T1--T4 expansion with real answer/OOF requirements and non-duplicated modality conditions.
11. Frozen MMRS/RSGPT description-source subset with answer-level provenance and exact exclusions.
12. Independent validator, summary, manifest hashing and two-build deterministic synthetic Small acceptance.
13. Sen12 single-time loader for annotated S2/ASC/DSC triplets using one event-nearest acquisition per modality.
14. Canonical language-parent closure:
    - licensed MMRS/RSGPT image rows sharing one license snapshot group by
      `source_key + exact image SHA + component-license SHA` and materialize once;
    - raw source rows retain `datasets/...` provenance while canonical description indexes bind only
      Benchmark `assets/...` image/valid references;
    - answer text, answer ID, source index SHA and source-record SHA remain replayable;
    - RSIEval permanent-test priority propagates through verified duplicate connected components;
    - DIOR-RSVG remains box/short phrase only and cannot fabricate a maskless T2 view;
    - denied language rows remain audit-only and are not decoded or copied.
15. Exact language component-license and report-replay closure:
    - MMRS-1M/RSGPT aggregate containers are inventory-only and cannot authorize component use;
    - eight frozen component keys independently bind role, split policy and full license snapshot;
    - one approved component cannot promote a sibling, and RSIEval can never enter training;
    - source rows, visual parents, canonical descriptions, registry provenance and license hashes replay exactly;
    - validator independently rebuilds duplicate clusters, group split, evaluation conditions, component
      license report and summary pixel counts instead of trusting builder reports.

Last verified regression: 57/57 focused P1 unit tests passed in `qwen3vl`. The licensed-language synthetic
Small was built twice from reversed source-record order with identical manifest/output hashes; independent
validation replay returned `errors=[]`. Aggregate override, one-component approval, registry tampering and
summary-report tampering have focused negative coverage. The live bounded adapter audit
had `errors=[]`, 14 audit candidates, zero materialization-eligible candidates and repeated aggregate SHA-256
`4e2edbe2549313db49bb8e97144f0d6f2429d2aa0dadaddcdb1977f4f44c54fc`.

## Active formal acceptance gate

Real Small preflight is fail-closed because every live source remains `allowed_for_training=false`.
No raw decode or output write occurs. This matches the governing manual stop condition
`license unknown for requested training source`.

- Exact preflight result: `SourceLoadingError: no training-eligible spatial source; approve exact
  source license evidence before a formal Small build`.
- Narrowest technically ready source: Sen12Landslides.
- Required human decision: training/evaluation/redistribution permissions and full Sentinel,
  Copernicus and DEM attribution obligations; update both Small and Full registry rows consistently.
- Decision form: `docs/audits/p1_human_data_license_decision_request.md`.
- MMRS components and RSGPT require eight separate license decisions and remain audit-only. Their aggregate
  containers are contractually inventory-only; the canonical materializer is covered by synthetic replay and
  no live language image was promoted.

## Remaining work after the human gate

1. Record the approved license fields without changing any unrelated source row.
2. Run the real Small build once into the configured new directory.
3. Reopen `reports/validation_report.json`; localize and fix any engineering error.
4. Build the identical input/config into a separate new verification directory and compare aggregate hashes.
5. Confirm cross-split verified duplicates = 0, no forbidden fields/legacy dependency, and license unknown = 0.
6. Only then finalize the P1 completion report, `docs/handoffs/P1.md` and next accepted task.

## Next exact command

```bash
sami-gsd data build --config configs/benchmark_v3_small.yaml
```

Do not execute this command against the unchanged registry; it is recorded as the exact resume command
after the required owner decision. No formal P1 handoff or next accepted phase exists yet.
