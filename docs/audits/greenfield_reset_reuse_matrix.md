# P0R Greenfield Reset Reuse Matrix

- Date: 2026-07-24
- Git baseline: `58813468d2cf3be715e399f5904a8d9f7f5c880d`
- Scope: all 70 committed paths under `src/sami_gsd`, `configs`, `schemas`, `tests/p1`, and
  `tests/p2` after the tracked P3 reset
- Rule: classification is not acceptance and authorizes no committed deletion

Every scoped committed path appears exactly once below. Package location is not evidence that a
file belongs to the new HDF5-first design.

## KEEP_GENERIC

These five files are source-agnostic enough to remain. Their current APIs still receive normal
phase-local review before use.

- `src/sami_gsd/__init__.py`
- `src/sami_gsd/contracts/spatial.py`
- `src/sami_gsd/data/adapters/formats.py`
- `src/sami_gsd/utilities/__init__.py`
- `src/sami_gsd/utilities/artifacts.py`

## REBIND_AND_TEST

These 16 files contain useful package, coordinate, transform, split, or model-input boundaries, but
must be rebound to Canonical Parent v4, the new assurance fields, and current validity semantics.
They are not accepted unchanged.

- `src/sami_gsd/contracts/__init__.py`
- `src/sami_gsd/contracts/model.py`
- `src/sami_gsd/data/__init__.py`
- `src/sami_gsd/data/adapters/__init__.py`
- `src/sami_gsd/data/duplicates.py`
- `src/sami_gsd/data/reference_canvas.py`
- `src/sami_gsd/data/split.py`
- `src/sami_gsd/data/transforms.py`
- `src/sami_gsd/model/__init__.py`
- `src/sami_gsd/model/input_loader.py`
- `src/sami_gsd/model/rendering.py`
- `src/sami_gsd/model/states.py`
- `tests/p1/__init__.py`
- `tests/p1/conftest.py`
- `tests/p1/test_duplicates_split.py`
- `tests/p1/test_spatial.py`

P1 owns the data/coordinate items. P2 may consume the model-input items only after P1 acceptance.
In particular, existing split/duplicate logic must stop treating incomplete group evidence as a
training rejection or permission to rewrite native splits.

## REWRITE

These 21 files encode the superseded Benchmark v3, raw/source adapters, or v3 validation
assumptions. P1 replaces them from zero under the HDF5 contract; old public names and schemas are
not preserved.

- `configs/benchmark_v3_full.yaml`
- `configs/benchmark_v3_small.yaml`
- `schemas/canonical_parent_v3.schema.json`
- `src/sami_gsd/cli.py`
- `src/sami_gsd/contracts/canonical.py`
- `src/sami_gsd/contracts/config.py`
- `src/sami_gsd/contracts/sources.py`
- `src/sami_gsd/data/adapters/audit.py`
- `src/sami_gsd/data/adapters/base.py`
- `src/sami_gsd/data/adapters/implemented.py`
- `src/sami_gsd/data/adapters/registry.py`
- `src/sami_gsd/data/audit.py`
- `src/sami_gsd/data/builder.py`
- `src/sami_gsd/data/materialize.py`
- `src/sami_gsd/data/validation.py`
- `tests/p1/test_audit.py`
- `tests/p1/test_builder_validation.py`
- `tests/p1/test_configs.py`
- `tests/p1/test_contracts.py`
- `tests/p1/test_materialization.py`
- `tests/p1/test_source_adapters.py`

## DELETE_AFTER_REPLACEMENT

### P1-owned obsolete source-loader paths

These four paths remain until P1 v4 accepts a source registry with empty Sen12 marked `not_ready`
and replacement tests. P1 must not port the old Sen12-only loader.

- `src/sami_gsd/data/source_loaders/__init__.py`
- `src/sami_gsd/data/source_loaders/sen12.py`
- `tests/p1/test_sen12_loader.py`
- `tests/p1/test_tasks.py`

The task test in this subsection is retired with the old task builder rather than converted into
model or description work.

### Provisional description/Bridge-owned paths

These 14 paths are outside P1-P5 segmentation implementation. They remain until a post-P3
description scope supplies accepted replacements; P1 does not satisfy their deletion gate.

- `configs/description_ontology_v1.yaml`
- `configs/instruction_templates/multisource_landslide_v2.yaml`
- `configs/landslide_bridge_v1.yaml`
- `configs/qpsalm_description_output_v1.schema.json`
- `configs/qpsalm_description_record_v2.schema.json`
- `configs/qpsalm_landslide_region_description_v1.schema.json`
- `configs/scene_region_ontology_v2.yaml`
- `schemas/canonical_description_v3.schema.json`
- `schemas/task_view_v3.schema.json`
- `src/sami_gsd/contracts/language.py`
- `src/sami_gsd/data/language_subset.py`
- `src/sami_gsd/data/tasks.py`
- `tests/p1/test_language_canonical_build.py`
- `tests/p1/test_language_subset.py`

### P2-owned obsolete model line

These ten paths implement or test the committed language-model/sensor-adapter P2 line. The new P2
is a minimal convolutional direct-dense sanity baseline, so these paths are not reused. They remain
until the new P2 replacement passes owner-run 1/4/8/32-parent and strict reload gates.

- `configs/model_sami.yaml`
- `src/sami_gsd/model/cache.py`
- `src/sami_gsd/model/qwen_backbone.py`
- `src/sami_gsd/model/sensor_adapter.py`
- `src/sami_gsd/model/smoke.py`
- `tests/p2/__init__.py`
- `tests/p2/conftest.py`
- `tests/p2/test_config_adapter.py`
- `tests/p2/test_qwen_cache.py`
- `tests/p2/test_rendering_cli_boundaries.py`

## DIRTY_P3_DISCARDED

The following resolved classes were uncommitted and were deleted in P0R after exact inventory:

- `configs/benchmark_v3_small_truecolor_v1.yaml`
- `configs/g0/`
- untracked ADR draft numbers 0004 through 0008
- `docs/baseline_patches/`
- `docs/worklogs/P3_CONTINUATION.md`
- all 19 resolved `schemas/g0_*.schema.json` files
- `src/sami_gsd/contracts/g0.py`
- `src/sami_gsd/contracts/g0_supervised.py`
- `src/sami_gsd/data/truecolor.py`
- `src/sami_gsd/g0/`
- `src/sami_gsd/model/segmentation/`
- `src/sami_gsd/utilities/schema.py`
- `tests/p1/test_truecolor.py`
- `tests/p3/`
- `reports/`

The exact files, counts, sizes, and key hashes are in
`docs/audits/p0r_dirty_p3_reset_manifest.json`. None may be restored as a compatibility layer or
used as the starting point for new P3.

## HISTORICAL_EVIDENCE_ONLY

- committed P1/P2 completion reports and handoffs;
- earlier accepted ADR text where not superseded;
- Git commits preceding the HDF5 rebaseline;
- legacy reports and artifacts outside the one-time dirty reset.

These can explain prior work but cannot pass a current phase.

## SEPARATE_LEGACY_GATE

- `SEG_Multi-Source_Landslides/`
- `scripts/1-benchmark/` through `scripts/5-segdesc/`
- their run wrappers

They are never imported, copied, or exposed through shims. Exact replacement owners and deletion
gates are in `docs/audits/deletion_plan.yaml`.

## Coverage check

| Class | Committed scoped paths |
|---|---:|
| KEEP_GENERIC | 5 |
| REBIND_AND_TEST | 16 |
| REWRITE | 21 |
| DELETE_AFTER_REPLACEMENT | 28 |
| Total | 70 |

## P1 implementation delta

This section records the post-P0R replacement without changing the 70-path baseline classification.
Implementation is complete, including the later self-contained materialization correction, but the
replacement is not accepted until the owner builder and independent validator pass.

New P1 replacement paths:

- `configs/benchmark_v4_small.yaml`
- `schemas/hdf5_source_record_v1.schema.json`
- `schemas/canonical_parent_v4.schema.json`
- `schemas/benchmark_manifest_v4.schema.json`
- `schemas/benchmark_statistics_v4.schema.json`
- `schemas/benchmark_validation_report_v4.schema.json`
- `schemas/channel_catalog_v1.schema.json`
- `schemas/materialized_hdf5_asset_v1.schema.json`
- `src/sami_gsd/contracts/benchmark_v4.py`
- `src/sami_gsd/contracts/benchmark_v4_config.py`
- `src/sami_gsd/data/hdf5_sources_v4.py`
- `src/sami_gsd/data/benchmark_v4.py`
- `src/sami_gsd/data/benchmark_v4_validation.py`
- `tests/p1/v4_test_support.py`
- `tests/p1/test_v4_contracts_config.py`
- `tests/p1/test_v4_source_records.py`
- `tests/p1/test_v4_builder_validation.py`
- `tests/p1/test_v4_cli.py`

Rebound paths:

- `src/sami_gsd/__init__.py`
- `src/sami_gsd/cli.py`
- `src/sami_gsd/contracts/__init__.py`
- `src/sami_gsd/data/__init__.py`

No P1-owned obsolete committed path has been deleted. Its deletion-plan entries remain unapproved and
blocked on live P1 acceptance plus exact reference/test/doc scans.
