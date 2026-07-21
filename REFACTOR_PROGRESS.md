# REFACTOR_PROGRESS

## Current status

- phase: `P2`
- phase_status: `engineering_accepted`
- phase_completion_date: `2026-07-21`
- current_branch: `refactor/sami-groundsegdesc`
- completion_implementation_report_commit: `pending_local_commit`
- push_performed: `false`
- active_adrs:
  - `docs/adr/ADR-0001-greenfield-rewrite.md` (`accepted`, raw-data clauses amended)
  - `docs/adr/ADR-0003-raw-data-provenance-not-runtime-license-gate.md` (`accepted`)
- completion_report: `docs/reports/p2/p2_completion_report.json`
- handoff: `docs/handoffs/P2.md`
- next_accepted_task: `P3 G0 segmentation-kernel selection`

P2 was executed and accepted as one formal phase. Its design, implementation, synthetic matrix,
official processor probe, real Profile S forward, regression, report and handoff were continuous
internal work packages rather than separate user gates.

## Accepted P2 evidence

- official local backend: `Qwen3VLProcessor` plus `Qwen3VLForConditionalGeneration` from
  Transformers 5.3.0;
- accepted input: P1 Small parent `sen12-chimanimani-1001` bound to the accepted manifest and
  validation aggregate hashes;
- one official native multi-image forward over `s2_optical`, `s1_ascending`, `s1_descending`, and
  `dem`;
- reconstructable merged grids: reference 16x16; each support 12x12; four DeepStack/final spatial
  levels per view;
- Profile S peak allocated 4.1048808097839355 GiB and peak reserved 4.205078125 GiB, below the
  frozen 22 GiB limit;
- cache equivalence: exact metadata/shape/dtype, 24 tensors, minimum cosine
  0.9999999999999998, maximum absolute difference 0;
- P2 focused tests: 16/16 passed; P1 regression: 62/62 passed; final combined standard-library
  discovery: 78/78 passed. The optional `pytest` command collected nothing because the current
  environment has not installed the declared test extra.

## Accepted P1 evidence

- primary Small: `../benchmark/sami_landslide_v3/small`;
- deterministic repeat: `../benchmark_repeat/sami_landslide_v3/small`;
- manifest aggregate SHA-256:
  `5bd5f4dbd97f41b8276acd3f5c2a1953d6d41de0f3e509cf3dff6689ec321d54`;
- validation aggregate SHA-256:
  `c7532c6ba4b00d2c503117ad34df3e2725948700cfc7a639ae26c5485ccae8dd`;
- complete output hashes: 4501 entries, exactly equal across both builds;
- parents: 1064 total, 128 spatial, 936 language;
- parent split: train 686, val 121, test 257;
- descriptions: 996; tasks: 128 T1;
- validation: both `errors=[]`, provenance binding errors 0, verified duplicate cross-split 0;
- final regression: 62/62 passed.

## Governance boundary

Raw-data licenses and human source authorization are not P0-P7 runtime/build gates. Sources and
components retain only minimal scientific provenance. Public data/package/checkpoint/weight release
review is a separate future human process. Code/dependency license and notice obligations remain.

## Deletion state

No deletion-manifest legacy path was deleted in P2. Three explicit old Qwen wrapper/cache/CLI
entries were added, bringing the manifest to 39 entries. Every `approved_by` and `deleted_commit`
remains null.

## Stop cursor

Do not auto-advance from this completed task. P3 requires a new explicit `CURRENT_PHASE=P3` task.
Do not overwrite accepted Small/model smoke artifacts, build Full, start formal training, choose a
G0 segmentation kernel, or execute legacy deletion under this P2 authorization.
