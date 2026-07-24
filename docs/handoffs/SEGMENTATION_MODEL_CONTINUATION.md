# Segmentation Model Continuation Handoff

- Date: 2026-07-24
- Program: `SEGMENTATION_MODEL_CONTINUATION`
- Goal status: resumed by owner after an interrupted blocked tool state
- Terminal scope: `P4_MODEL_ENGINEERING_ACCEPTED`
- Current phase: P1 owner gate
- Commit/push: not performed

## Authority outcome

The two authority files were rewritten before any model code:

- `docs/REFACTOR_TASK_SPEC.md` now owns the complete P0R-P8 segmentation design.
- `docs/CODEX_REFACTOR_PROMPT.md` contains execution, routing, permissions, and human gates only.

ADR-0005 records the project owner's segmentation-only continuation decision without selecting a
P3 kernel in advance.

ADR-0006 records the later owner correction from reference-only source paths to a self-contained,
byte-copied HDF5 Benchmark with explicit channel semantics.

## Static authority gate

| Check | Exit | Result |
| --- | ---: | --- |
| UTF-8 and BOM | 0 | passed |
| Markdown fences | 0 | passed after correcting the inspection command |
| relative Markdown links | 0 | passed |
| removed P3 symbol scan | 0 | no matches |
| `git diff --check` for both authorities | 0 | passed after removing trailing spaces |

The first fence inspection command exited 1 because literal Markdown backticks were interpreted by
the shell. It wrote no file. The replacement check uses `chr(96)` and passed.

## Current evidence

- Branch: `refactor/sami-groundsegdesc`
- Reset-start/current HEAD: `58813468d2cf3be715e399f5904a8d9f7f5c880d`
- Benchmark top level: empty at the continuation start
- Five HDF5 sources are ready for the P1 training population
- Sen12 is empty and remains not-ready
- Strict evaluation population remains zero
- No builder, validator, model overfit, formal comparison, or training result exists for v4
- P1 materialized-copy config/contracts/reader/builder/validator/CLI/schemas are implemented
- Nineteen focused `unittest` cases pass in `qwen3vl`
- Synthetic copy/ledger validation and the 19-key production channel catalog pass
- One live read-only HDF5 pair from each ready source was reopened and projected successfully

## Permissions

Codex may run focused tests and a synthetic or accepted-tiny-fixture GPU smoke of at most two
optimizer steps. The project owner still runs the Benchmark builder, independent validator,
micro-overfit, frozen multi-seed comparisons, formal memory gate, and any long training.

## Next gate

Run the exact P1 owner commands in the root README and return their exit codes, builder manifest
stdout, and the independent validation report. Do not enter P2 until that live report has
`errors == []` and matching source/copy/materialization/channel/split/manifest/normalization replay.
Detailed implementation evidence and the same commands are in `docs/handoffs/P1.md`.
