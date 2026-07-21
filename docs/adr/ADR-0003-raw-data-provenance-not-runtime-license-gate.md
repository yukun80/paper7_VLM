# ADR-0003: Raw-data provenance is not a runtime license gate

- Status: accepted
- Date: 2026-07-21
- Owners: project maintainer
- Phase: P1
- Commit: pending P1 completion commit
- Supersedes: the raw-data license/authorization gate clauses of ADR-0001 only

## Context

The initial P0/P1 implementation treated the Canonical Benchmark builder as a legal approval system. It added
source/component permission booleans, review identities/dates, license reports, human decision requests and a
pre-decode construction stop. That mechanism blocked the primary P1 objective even though the project owner had
already placed the intended research data under the adjacent local `datasets/` root.

The current objective is an internal-research Canonical Benchmark v3 for algorithm development, training and
evaluation. Public redistribution of raw images, a materialized benchmark or a derived data package is not part
of P1 and should not be conflated with technical dataset construction.

## Decision

1. Benchmark builders and validators perform technical/scientific checks, not legal review.
2. P0-P7 do not query, infer, verify or compare raw-data licenses and do not generate source authorization requests.
3. Dataset inclusion depends on local existence/readability, parseability, frozen research scope, required task
   supervision, reliable parent/scene/event/group identity, leakage-safe splits, coordinates, valid regions and
   exact/perceptual duplicate validation.
4. Runtime contracts, configs, schemas, reports and indexes contain no `allowed_for_*`, `academic_only`, reviewer,
   approval, license-status or redistribution fields.
5. Source/component records retain at most these scientific provenance fields:

   ```text
   source_key, source_name, source_root, source_document,
   citation_key, upstream_url, provenance_notes
   ```

   They never decide whether a row is built, trained or evaluated.
6. RSIEval permanent-test status, DIOR-RSVG region-only use, task-role selection and split isolation remain
   scientific policies and are not permission gates.
7. If the project later publishes raw images, a materialized Benchmark or derived data package, a distinct human
   publication/release review must be performed. That future review is outside P0-P7 and does not block them.
8. Code/dependency `LICENSE` and `NOTICE` obligations are unchanged.

## Alternatives considered

1. **Set every `allowed_for_training` flag to true.** Rejected because it preserves the incorrect approval system
   and creates a false legal conclusion.
2. **Keep permission metadata but make it warnings-only.** Rejected because the schema and reports would still
   present the builder as an authorization authority and invite future gate regression.
3. **Delay P1 until a source-by-source legal review.** Rejected because publication review is not the current
   internal-research construction objective.

## Evidence

- owner decision: written P1 design correction in the active task on 2026-07-21;
- implementation: strict provenance-only config/contracts, provenance report replay and negative tests rejecting
  removed permission fields;
- scientific controls retained: source roles, parent/event grouping, split replay, duplicate clustering, valid-mask
  propagation and RSIEval permanent-test forcing.

## Consequences

### Positive

- P1 can pursue its scientific and technical acceptance criteria without a pseudo-legal runtime subsystem.
- Provenance remains auditable while permission claims are not fabricated.
- A future release decision has a clear, explicit boundary.

### Negative

- P1 artifacts do not answer whether raw or derived data may be publicly redistributed.
- Historical P0/P1 license-gate reports and commits must be marked superseded/non-gating.

## Implementation constraints

- Do not add compatibility readers for license-bound P1 schemas or reports.
- Rename the canonical description protocol to v3 provenance-bound and replace `license_report.json` with
  `provenance_report.json`.
- Language parents group by physical `source_key + exact image SHA`; component identity remains on description rows.
- Technical failures remain hard errors; ordinary implementation/test failures are repaired within P1.

## Rollback

- trigger: the project owner explicitly starts a separate publication/release phase or changes the internal-research scope.
- procedure: create a new ADR for that release scope. Do not reintroduce legal approval into the Benchmark builder.

## Human approval

- approver: project owner
- date: 2026-07-21
- decision: accepted in the active P1 task
