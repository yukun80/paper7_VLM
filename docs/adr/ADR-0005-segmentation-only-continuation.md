# ADR-0005: Segmentation-Only Continuation

- Status: accepted
- Date: 2026-07-24
- Owner: project owner
- Scope: P3-P8 governance and the continuous P1-P4 engineering program

## Context

P0R established a clean HDF5-first data baseline, but its phase authority did not yet define P6/P7
or the complete P3/P4 model engineering route. The earlier model-selection narrative was coupled to
an incompatible benchmark generation and cannot govern the new data contract.

## Decision

1. Adopt the P0R -> P1 -> P2 -> P3 -> P4 -> P5 -> P6 -> P7 -> P8 segmentation phase graph
   defined by `docs/REFACTOR_TASK_SPEC.md`.
2. Authorize one continuous Goal across authority work and P1-P4, subject to the real owner-run
   gates at every phase boundary.
3. Make the variable-channel Channel-Set Dense kernel the default P3 candidate. A lightweight
   prompt-conditioned candidate is eligible only after the direct candidate passes and prompt
   information is demonstrably non-redundant.
4. Exclude larger language-driven pixel decoders and all box/proposal-based segmentation from the
   current program.
5. Freeze exactly one kernel in a later accepted ADR backed by live P3 artifacts.
6. Keep P5-P8 as restored authority only; the current Goal ends at
   `P4_MODEL_ENGINEERING_ACCEPTED`.

## Supersession

This ADR supersedes pre-rebaseline model-selection and phase-routing clauses wherever they conflict
with the current Task Spec. It does not alter historical records or restore removed files.

## Consequences

- P1 and P2 remain mandatory prerequisites.
- Language model size is not evidence of task information or scientific value.
- Unsupported complexity is rejected or placed behind a new owner decision.
- Human-run builder, validation, overfit, comparison, and memory artifacts remain mandatory.
- No commit, push, repository download, or long training is authorized.
