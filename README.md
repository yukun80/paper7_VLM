# SAMI-GroundSegDesc

SAMI-GroundSegDesc is undergoing an incompatible HDF5-first rewrite for single-time landslide
segmentation. The current line starts from local HDF5 image/mask assets, proves them learnable with
a minimal dense model, and only then compares a small number of segmentation kernels.

This README is the sole runbook. Detailed design is in
[`docs/REFACTOR_TASK_SPEC.md`](docs/REFACTOR_TASK_SPEC.md); Codex execution rules are in
[`docs/CODEX_REFACTOR_PROMPT.md`](docs/CODEX_REFACTOR_PROMPT.md).

## Current status

- Active program: `SEGMENTATION_MODEL_CONTINUATION`.
- Goal status: resumed after an interrupted turn; terminal scope is
  `P4_MODEL_ENGINEERING_ACCEPTED`.
- Active cursor: authority restoration and P1 implementation/focused checks are complete; P1 is
  awaiting the project-owner builder and independent-validator gate.
- Branch audited: `refactor/sami-groundsegdesc`.
- Reset-start HEAD: `58813468d2cf3be715e399f5904a8d9f7f5c880d`.
- `../benchmark` was empty at continuation start; there is no live accepted Benchmark v4.
- No v4 builder, independent validator, model overfit, frozen comparison, or training workflow has
  passed.
- The previous uncommitted P3, including reports/checkpoints/cache, was deliberately discarded.

Do not use committed old P1/P2 reports as evidence that the HDF5-first line passed. Their code is
classified for keep/rebind/rewrite/replacement in
[`docs/audits/greenfield_reset_reuse_matrix.md`](docs/audits/greenfield_reset_reuse_matrix.md).

## Stage map

```text
P0R  HDF5 audit, governance, and dirty reset                 complete
P1   self-contained HDF5 Canonical Benchmark v4 Small       awaiting owner gate
P2   minimal convolutional direct-dense sanity              not started
P3   independent kernel comparison and one frozen winner    not started
P4   selected-model training engineering                    not started
P5   strict/exploratory-separated evaluation                not started
P6   robustness and necessary ablations                     not started
P7   reproducibility and export packaging                   not started
P8   residual replacement-owned deletion                    not started
```

Description, Bridge, SegDesc, and joint training are excluded from the active program.

## Data baseline

P0R audited six local source directories:

| Source | Pairs | Training policy | Evaluation policy |
|---|---:|---|---|
| GDCLD | 13,447 | native train/val/test retained | exploratory |
| LMHLD | 28,185 | native train/val/test; B/G/R/NIR | exploratory |
| LandslideBench_agent | 2,130 | native train/val/test; 311 conflicts reported | exploratory |
| Landslide4Sense | 3,799 | all positive/background samples train-only | train-only |
| Multimodal | 6,084 | native train/val; RGB/DEM/InSAR | exploratory |
| Sen12Landslides | 0 | excluded as not-ready | unavailable |

All five non-empty sources must contribute to the P1 training candidate population. Missing
location/group/canonical-index/duplicate evidence reduces split assurance; it does not make the
image/mask pairs unusable for training.

The current snapshot contains no verified group-isolated cohort. Strict generalization claims are
therefore unavailable unless a later benchmark generation adds independently verified evidence.

Authoritative P0R records:

- [`hdf5_source_audit.md`](docs/audits/hdf5_source_audit.md)
- [`hdf5_source_inventory.json`](docs/audits/hdf5_source_inventory.json)
- [`hdf5_source_contract.yaml`](docs/audits/hdf5_source_contract.yaml)
- [`ADR-0004`](docs/adr/ADR-0004-hdf5-rebaseline-and-clean-p3-reset.md)

The continuation decision is recorded in
[`ADR-0005`](docs/adr/ADR-0005-segmentation-only-continuation.md).
The self-contained Benchmark correction is recorded in
[`ADR-0006`](docs/adr/ADR-0006-self-contained-hdf5-benchmark.md).

## Filesystem

Large assets live beside the repository:

```text
/home/yukun80/codes/
├── datasets/
├── benchmark/
└── paper7_VLM/
```

Runtime roots:

```bash
export PAPER7_DATASETS_ROOT=/home/yukun80/codes/datasets
export PAPER7_BENCHMARK_ROOT=/home/yukun80/codes/benchmark
```

Runtime resolvers may use these absolute roots, but Canonical Benchmark identities may contain only
portable `datasets/...` or `benchmark/...` logical paths plus content hashes.

`../datasets` is read-only. The P1 builder copies every selected image/mask HDF5 byte-for-byte into
`../benchmark/sami_landslide_hdf5_v4/small/assets/<source_key>/...`, preserving its source-relative
hierarchy. It does not decode, reorder, recompress, symlink, or hard-link source files. Downstream
model loading uses only the Benchmark copies; `../datasets` is needed again only for independent
provenance replay. Benchmark outputs must be new, immutable, and non-overwriting.

## P1 owner gate

The P1 contracts, HDF5 reader, builder, independent validator, CLI, schemas, and focused tests are
implemented. This is implementation evidence only: Benchmark v4 has not been built or accepted.

Run every command from the repository root. First rerun the focused tests:

```bash
cd /home/yukun80/codes/paper7_VLM

PYTHONPATH=src:. /home/yukun80/miniconda3/envs/qwen3vl/bin/python -m unittest -v \
  tests.p1.test_v4_contracts_config \
  tests.p1.test_v4_source_records \
  tests.p1.test_v4_builder_validation \
  tests.p1.test_v4_cli
```

The builder is deliberately fail-on-existing. Confirm the exact target is absent, then build:

```bash
test ! -e /home/yukun80/codes/benchmark/sami_landslide_hdf5_v4/small

PYTHONPATH=src:. /home/yukun80/miniconda3/envs/qwen3vl/bin/python -m sami_gsd.cli \
  benchmark build \
  --config /home/yukun80/codes/paper7_VLM/configs/benchmark_v4_small.yaml \
  --datasets-root /home/yukun80/codes/datasets \
  --benchmark-root /home/yukun80/codes/benchmark
```

The validator reopens both source and copied HDF5 files, verifies their hashes/bytes and channel
catalog, and writes outside the immutable benchmark. Confirm the report path is absent, then validate:

```bash
test ! -e /home/yukun80/codes/paper7_VLM/outputs/sami_gsd/p1/hdf5_v4_small_independent_validation.json

PYTHONPATH=src:. /home/yukun80/miniconda3/envs/qwen3vl/bin/python -m sami_gsd.cli \
  benchmark validate \
  --benchmark /home/yukun80/codes/benchmark/sami_landslide_hdf5_v4/small \
  --datasets-root /home/yukun80/codes/datasets \
  --schemas-root /home/yukun80/codes/paper7_VLM/schemas \
  --output /home/yukun80/codes/paper7_VLM/outputs/sami_gsd/p1/hdf5_v4_small_independent_validation.json
```

Finally, require an empty error list and print the binding/population summary:

```bash
PYTHONPATH=src:. /home/yukun80/miniconda3/envs/qwen3vl/bin/python -c 'import json; from pathlib import Path; p=Path("/home/yukun80/codes/paper7_VLM/outputs/sami_gsd/p1/hdf5_v4_small_independent_validation.json"); d=json.loads(p.read_text(encoding="utf-8"), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x))); assert d["errors"] == [], d["errors"]; expected={"materialized_asset_count":107290,"source_record_count":53645,"parent_count":53645,"split_counts":{"train":37521,"val":11995,"test":4129},"assurance_counts":{"verified_group_isolated":0,"source_declared_unverified":49846,"train_only":3799},"eligibility_counts":{"strict":0,"exploratory":49846,"train_only":3799}}; assert all(d[k] == v for k,v in expected.items()), {k:d[k] for k in expected}; assert d["materialized_size_bytes"] > 0; print(json.dumps({k:d[k] for k in ("benchmark_manifest_sha256","normalization_binding_sha256","materialized_asset_count","materialized_size_bytes","source_record_count","parent_count","split_counts","assurance_counts","eligibility_counts")}, ensure_ascii=False, sort_keys=True, indent=2))'
```

Then verify that the catalog exposes the 19 currently audited semantic channel identities:

```bash
PYTHONPATH=src:. /home/yukun80/miniconda3/envs/qwen3vl/bin/python -c 'import json; from pathlib import Path; p=Path("/home/yukun80/codes/benchmark/sami_landslide_hdf5_v4/small/manifests/channel_catalog.json"); d=json.loads(p.read_text(encoding="utf-8")); keys=[x["channel_key"] for x in d["entries"]]; assert len(keys)==19 and len(keys)==len(set(keys)); print(json.dumps(keys, ensure_ascii=False, indent=2))'
```

The expected counts are frozen audit expectations and a drift detector, not a substitute for
validator replay. The selected index projection currently contains 107,290 HDF5 files and
35,698,325,607 bytes (about 33.25 GiB) by read-only stat replay; the builder repeats an exact
file-size/free-space preflight and reports exact materialized bytes in its manifest. Return all exit
codes, builder stdout, the independent report, and both summaries. Do not enter P2 until the report
is reopened and accepted with `errors == []`.

The live continuation cursor is in
[`docs/handoffs/SEGMENTATION_MODEL_CONTINUATION.md`](docs/handoffs/SEGMENTATION_MODEL_CONTINUATION.md).
P1 implementation evidence is recorded in [`docs/handoffs/P1.md`](docs/handoffs/P1.md).

## P1 contract preview

P1 is limited to HDF5 Canonical Benchmark v4:

- source records bind source and Benchmark copy paths, identical file hashes, byte sizes, HDF5
  dataset keys, shapes, dtypes, layouts, channel descriptors, and validity schema;
- the materialization ledger accounts for every copied HDF5 and rejects symlinks, hard links,
  missing copies, partial files, hash drift, or unregistered extras;
- the channel catalog assigns a stable semantic token to each `channel_key`; stored tensors retain
  their source order, and every per-record descriptor carries modality, unit, wavelength/GSD known
  state, and validity semantics;
- native source split and its assurance label are both retained;
- strict, exploratory, and train-only populations are disjoint;
- Landslide4Sense never enters val/test;
- empty Sen12 produces no record;
- source HDF5 is read-only; copied HDF5 is the only downstream data path;
- outputs are atomic and immutable;
- validator reopens source bytes and recomputes manifest bindings.

P1 contains no model, language-model loading, bounding box generation, candidate registry,
description data, or future-phase compatibility layer.

## Model boundaries

P2 first implements a small convolutional direct-dense model and 1/4/8/32-parent memory tests. It
does not load a language model or generate boxes.

P3 first evaluates the Channel-Set Dense multimodal kernel. A lightweight prompt-conditioned
derivative is eligible only after the direct kernel passes and prompt information is non-redundant.
There is no runtime auto-switch. Exactly one accepted kernel may enter P4.

P5 reports strict, exploratory, and train-only evidence separately. Without a strict cohort it must
publish `strict_generalization_status: unavailable`.

Research sources, hypotheses, and evidence state live under `docs/research/`. Literature does not
select a model by itself; it only contributes mechanisms to frozen, falsifiable protocols.

## Legacy and deletion policy

The legacy package and old benchmark scripts remain historical/reference assets until their
replacement-owned deletion entries pass. They are not dependencies of the new main line.

Deletion rules are in [`docs/audits/deletion_plan.yaml`](docs/audits/deletion_plan.yaml). A
replacement must pass first, then exact references/tests/docs must be checked and the project owner
must approve that entry. P8 handles residual assets only.

The P0R one-time dirty reset is recorded in
[`p0r_dirty_p3_reset_manifest.json`](docs/audits/p0r_dirty_p3_reset_manifest.json). It does not grant
permission for future cleanup.

## Agent and owner boundary

Codex may run focused CPU/unit tests and a synthetic or accepted-tiny-fixture GPU smoke with batch
size 1 and at most two optimizer steps. The project owner runs the Benchmark builder, independent
validator, micro-overfit gates, frozen multi-seed comparison, formal memory gate, long training, and
formal evaluation.

No phase is scientifically accepted merely because code exists. Acceptance requires the owner-run
artifact and the phase-specific replay defined in the Task Spec.
