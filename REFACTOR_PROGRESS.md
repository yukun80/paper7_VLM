# REFACTOR_PROGRESS

## Current status

- phase: `P1`
- phase_status: `engineering_accepted`
- phase_completion_date: `2026-07-21`
- current_branch: `refactor/sami-groundsegdesc`
- completion_commit: `not_created`
- push_performed: `false`
- active_adrs:
  - `docs/adr/ADR-0001-greenfield-rewrite.md` (`accepted`, raw-data clauses amended)
  - `docs/adr/ADR-0003-raw-data-provenance-not-runtime-license-gate.md` (`accepted`)
- completion_report: `docs/reports/p1/p1_completion_report.json`
- handoff: `docs/handoffs/P1.md`
- next_accepted_task: `P2 model minimum skeleton`

P1 was executed and accepted as one formal phase. Historical P1.1-P1.3 labels were internal work
packages only.

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

No deletion-manifest legacy path was deleted in P1. All 36 entry approvals and deleted commits
remain null. Obsolete greenfield authorization/schema/internal-report drafts were removed or
replaced as recorded in the P1 handoff.

## Stop cursor

Do not auto-advance from this completed task. P2 requires a new explicit `CURRENT_PHASE=P2` task.
Do not overwrite the accepted Small artifacts, build Full, start model training, or execute legacy
deletion under this P1 authorization.
