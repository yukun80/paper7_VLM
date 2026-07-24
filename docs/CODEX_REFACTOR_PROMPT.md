# SAMI-GroundSegDesc Codex Execution Rules

> Status: active
> Role: execution boundary, phase routing, and human gates
> Detailed design: [`REFACTOR_TASK_SPEC.md`](REFACTOR_TASK_SPEC.md)

## 1. Authority order

1. the project owner's current explicit phase, scope, and corrections;
2. `docs/REFACTOR_TASK_SPEC.md`;
3. this execution file;
4. `REFACTOR_PROGRESS.md`;
5. the latest applicable handoff;
6. accepted ADRs.

Historical handoffs, reports, archived documents, and Git history are evidence only for their original
generation. They cannot establish current acceptance.

## 2. Active program

```text
PROGRAM_NAME = SEGMENTATION_MODEL_CONTINUATION
INITIAL_SUBTASK = RESTORE_P3_P8_AUTHORITIES
TERMINAL_SCOPE = P4_MODEL_ENGINEERING_ACCEPTED
AUTO_CONTINUE_WITHIN_AUTHORIZED_PROGRAM = TRUE
```

The program may continue through authority work, P1, P2, P3, and P4. It must not cross a real phase artifact
or owner-run gate. P5-P8 are authority-only in this program.

Description, Bridge, SegDesc, and joint training are excluded. They must not enter code, configs, commands,
acceptance, or future scaffolding under this Goal.

## 3. Stage routing

```text
P0R governance baseline
  -> P1 HDF5 Benchmark v4 Small
  -> owner P1 builder and validator gate
  -> P2 direct-dense prerequisite
  -> owner P2 overfit and reload gate
  -> P3 multimodal candidate protocols and implementation
  -> owner P3 frozen comparison gate
  -> accepted unique-kernel ADR
  -> P4 selected-model engineering
  -> owner P4 engineering gate
  -> stop
```

Do not implement P2 before live P1 acceptance, P3 before live P2 acceptance, or P4 before the unique P3
kernel is frozen. Work that does not depend on a pending result may continue only when it belongs to the
current or an earlier authorized phase.

## 4. Continuous Goal rules

After the two authority files pass their static gate:

1. call `get_goal`;
2. create the owner-specified Goal only when no unfinished Goal exists;
3. continue the identical active Goal instead of creating another;
4. stop rather than replace a different unfinished Goal;
5. never set a token budget unless the owner requests one.

A pending human gate does not complete the Goal. The first wait does not make it blocked. Leave it active,
publish the exact command, and inspect the returned live artifacts before advancing. Only the Goal tool's
repeated-blocker rule can justify `blocked`.

Call `update_goal(status=complete)` only when P4 is genuinely engineering-accepted and no required work
remains. Do not complete the Goal merely because P5 is out of scope.

## 5. Execution permissions

Codex may run:

- read-only inspection;
- UTF-8, Markdown, JSON, YAML, Python AST, and Git diff checks;
- focused CPU/unit tests in the `qwen3vl` environment;
- a synthetic or accepted-tiny-fixture GPU smoke with batch size 1 and at most two optimizer steps.

Codex must not run:

- the P1 builder or independent validator;
- owner micro-overfit, frozen multi-seed comparison, or formal memory gates;
- long training or formal evaluation;
- repository downloads;
- commits or pushes.

The bounded GPU smoke is diagnostic. It cannot establish phase acceptance and must not write to
`../benchmark`.

## 6. Data and artifact boundaries

- `../datasets` is read-only.
- A source HDF5 is a read-only construction input, not a model-ready RGB image.
- P1 byte-copies every selected image/mask HDF5 into the immutable Benchmark package; it must not
  decode/re-encode, reorder channels, use symlinks/hard links, or fall back to `../datasets` during
  downstream loading.
- Canonical identities bind portable source and Benchmark logical paths, exact byte hashes, and explicit
  per-channel physical descriptors. Source channel order is preserved and interpreted through the bound
  channel catalog rather than a fixed global tensor layout.
- Missing group/location/duplicate evidence lowers evaluation assurance but does not block training.
- Native source splits are retained with their assurance labels; no random split is introduced.
- A not-ready source produces no canonical row.
- Builders are fail-on-existing; no accepted benchmark, report, cache, or checkpoint is overwritten.
- Strict, exploratory, and train-only evidence are never merged.

## 7. Architecture prohibitions

- do not restore any removed pre-rebaseline P3 implementation or artifact;
- do not copy or import the legacy package;
- do not add a compatibility shim, candidate registry, runtime switch, or automatic fallback;
- do not use fixed source slots or a fixed input channel count in the selected multimodal kernel;
- do not infer wavelength, GSD, physical units, or sign from names;
- do not use a language model for spatial encoding, coordinates, proposals, or mask generation;
- do not add an external segmentation foundation model;
- do not implement a future phase while waiting for an owner result.

## 8. Research loop

Use only the project-safe form of the research loop:

```text
hypothesis
  -> frozen protocol
  -> minimum implementation
  -> static and permitted bounded checks
  -> exact owner command
  -> live artifact inspection
  -> findings/state update
  -> keep, simplify, reject, or next hypothesis
```

There is no wall-clock scheduler, automatic commit, autonomous formal experiment, or unregistered
hyperparameter sweep. Run an outer synthesis after three to five real experiments or contradictory evidence.

## 9. Code and documentation discipline

- Python 3.11, four spaces, typed data contracts.
- Prefer `pathlib`, dataclasses/Pydantic, strict JSON/JSONL, atomic writes, and SHA-256.
- CLI modules parse arguments and orchestrate; reusable algorithms live outside CLI.
- New executable scripts include a short Chinese header with purpose, command, inputs, outputs, write
  behaviour, and phase.
- Add only files required by the current consumer.
- README is the sole runbook; Task Spec is the sole detailed design; Progress records only current evidence.
- Protect unrelated dirty changes. Do not reset, checkout, stash, or broaden a deletion target.

## 10. Gate reporting

Every Goal turn reports:

1. Goal status and current phase;
2. completed hypothesis or work package;
3. modified files;
4. actual commands and exit codes;
5. commands not run;
6. current evidence and conclusion;
7. rejected complexity;
8. exact owner command;
9. next action;
10. whether the Goal completion definition is satisfied.

Never translate static implementation, file presence, or an older report into scientific acceptance.
