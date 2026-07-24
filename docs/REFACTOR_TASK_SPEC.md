# SAMI-GroundSegDesc HDF5-First Segmentation Refactor Task Specification

> Status: active
> Design generation: HDF5 Benchmark v4 segmentation continuation
> Scientific scope: single-time landslide semantic segmentation
> Runtime package: `src/sami_gsd/`
> Terminal scope of the active program: `P4_MODEL_ENGINEERING_ACCEPTED`

## 1. Authority and program objective

This file is the only detailed design authority for the current incompatible refactor.
`docs/CODEX_REFACTOR_PROMPT.md` defines execution boundaries and phase routing only.

The active program rebuilds the segmentation main line from the audited local HDF5 assets:

1. construct a self-contained Canonical Benchmark by byte-copying the selected HDF5 pairs;
2. prove the image, mask, validity, and registered-view path with a minimal dense model;
3. compare the smallest evidence-backed multimodal kernels;
4. freeze exactly one kernel;
5. build the complete training, evaluation, checkpoint, resume, and export engineering around it.

Historical implementations and reports can establish provenance, but they cannot establish acceptance for
this HDF5-first generation. No compatibility layer may make an earlier benchmark, config, class, cache, or
checkpoint appear current.

## 2. Frozen scientific scope

### 2.1 Included

- single-time binary landslide semantic segmentation;
- positive, no-target, single-component, and multi-component masks;
- heterogeneous optical, multispectral, elevation, slope, radar, and interferometric channels;
- variable channel count and missing modalities;
- explicit channel identity, modality family, validity, and known/unknown physical metadata;
- per-source and unified training;
- strict, exploratory, and train-only evidence kept separate;
- a final engineering configuration for a single GPU with approximately 24 GiB memory.

### 2.2 Excluded

- pre/post change detection;
- generic detection, visual question answering, video, or tracking;
- Description, Bridge, SegDesc, or joint training;
- box-first, proposal-cascade, oracle-region, or autoregressive mask generation;
- external segmentation foundation models in the active path;
- source slots, a fixed five-channel layout, runtime candidate switching, or compatibility shims;
- runtime raw-data licence gates;
- repository downloads, long training, commits, or pushes by Codex.

The excluded language and description programs require a separate future authority. They cannot block or
silently enter P1-P4.

## 3. Phase graph and stopping rules

```text
P0R  HDF5 rebaseline, governance, and clean reset
  -> P1  HDF5-first Canonical Benchmark v4 Small
  -> P2  registered-view direct-dense learnability prerequisite
  -> P3  multimodal kernel research, comparison, and unique freeze
  -> P4  selected-model training engineering
  -> P5  formal evaluation
  -> P6  robustness and necessary ablations
  -> P7  reproducibility, export, and formal run packaging
  -> P8  residual replacement-owned deletion
```

The active continuous program may cross documentation work, P1, P2, P3, and P4, but it must stop at every
real owner-run gate. P5-P8 are specified here but are outside the active program's implementation scope.

Phase status is evidence-based:

- source code or a config is not acceptance;
- a static check is not a builder, model, or scientific result;
- an earlier-generation report is not evidence for Benchmark v4;
- an owner-run gate is accepted only after its live artifact and bindings are independently reopened;
- the continuous Goal remains active while a required owner result is pending.

## 4. P0R data baseline

P0R confirmed the following local source population. Counts are audit evidence, not runtime constants.

| source_key | pairs | positive | no-target | ingestion | split policy |
| --- | ---: | ---: | ---: | --- | --- |
| `gdcld` | 13,447 | 11,593 | 1,854 | `ready` | retain source train/val/test |
| `lmhld` | 28,185 | 28,008 | 177 | `ready` | retain source train/val/test |
| `landslidebench_agent` | 2,130 | 1,819 | 311 | `ready` | retain source train/val/test |
| `landslide4sense` | 3,799 | 2,231 | 1,568 | `ready` | all train-only |
| `multimodal_landslide` | 6,084 | 5,495 | 589 | `ready` | retain source train/val |
| `sen12_landslides` | 0 | 0 | 0 | `not_ready` | excluded |

All five non-empty sources participate in the training population. Missing per-sample coordinates,
verified geographic parents, complete duplicate components, or reliable group identifiers lower evaluation
assurance but do not make readable image/mask pairs untrainable.

### 4.1 Independent qualification fields

Each source and canonical record declares:

```yaml
ingestion_status: ready | not_ready
canonical_split: train | val | test
split_assurance: verified_group_isolated | source_declared_unverified | train_only
evaluation_eligibility: strict | exploratory | train_only
```

Allowed combinations:

| split_assurance | evaluation_eligibility | allowed use |
| --- | --- | --- |
| `verified_group_isolated` | `strict` | train, selection, strict evaluation |
| `source_declared_unverified` | `exploratory` | train, development, exploratory evaluation |
| `train_only` | `train_only` | train and overfit diagnostics only |

No random split may be introduced to manufacture an evaluation cohort. A future verified group graph
requires a new immutable Benchmark generation; it cannot upgrade an existing artifact in place.

### 4.2 Current source decisions

- GDCLD remains `source_declared_unverified` until source grouping can be independently replayed.
- LMHLD retains B/G/R/NIR and its declared splits; missing location and group fields affect evaluation only.
- LandslideBench_agent retains its declared splits and reports all known location-level conflicts. Its
  dialogue text is provenance or an unverified candidate, never factual supervision.
- Landslide4Sense contributes all positive and background rows to train-only.
- Multimodal retains RGB, DEM, and InSAR-like inputs and its declared train/val split. Known spatial
  adjacency risk keeps evaluation exploratory.
- Sen12 remains not-ready while empty and cannot be represented by placeholder rows.

## 5. HDF5 source contract

### 5.1 Identity

The source-to-benchmark protocol is:

```yaml
schema_version: sami_hdf5_source_contract_v1
protocol: sami_hdf5_materialized_copy_v1
source_path_namespace: datasets
benchmark_path_namespace: benchmark
hash_algorithm: sha256
```

Canonical identity binds both the read-only `datasets/...` source path and the immutable
`benchmark/.../assets/...` copy, plus their identical content hash, dataset keys, scientific metadata,
and channel schema. Machine absolute paths are runtime resolver inputs and must not affect record or
aggregate identity.

### 5.2 Source record

Every record binds:

```yaml
source_key: string
source_sample_id: string
ingestion_status: ready | not_ready
source_declared_split: train | val | test | null
canonical_split: train | val | test
split_assurance: verified_group_isolated | source_declared_unverified | train_only
evaluation_eligibility: strict | exploratory | train_only
group:
  group_id: string | null
  group_kind: scene | event | location | source_sample | unknown
  evidence: [string, ...]
  completeness: verified | partial | unavailable
image:
  source_logical_path: datasets/<source>/<relative>.h5
  benchmark_logical_path: benchmark/sami_landslide_hdf5_v4/small/assets/<source_key>/<relative>.h5
  sha256: <64 lowercase hexadecimal>
  size_bytes: integer
  dataset_key: string
  shape: [C, H, W]
  dtype: string
  layout: CHW
mask:
  source_logical_path: datasets/<source>/<relative>.h5
  benchmark_logical_path: benchmark/sami_landslide_hdf5_v4/small/assets/<source_key>/<relative>.h5
  sha256: <64 lowercase hexadecimal>
  size_bytes: integer
  dataset_key: string
  shape: [H, W]
  dtype: string
channels: [ChannelDescriptorV1, ...]
validity: {...}
record_sha256: <64 lowercase hexadecimal>
```

Changing source bytes, copied bytes, dataset keys, shapes, dtypes, layout, channel metadata, or validity
semantics invalidates the record and every downstream binding. A build is invalid unless source and copied
SHA-256 values match.

### 5.3 `ChannelDescriptorV1`

Every scalar channel declares:

```yaml
schema_version: sami_channel_descriptor_v1
index: integer
channel_key: string
display_name: string
modality_family: optical | multispectral | dem | slope | sar | insar | other
physical_unit: string | null
wavelength_nm: number | null
wavelength_known: boolean
gsd_m: number | null
gsd_known: boolean
normalization: none | divide_255 | zscore_valid_pixels | source_preprocessed
validity_source: channel_valid | pixel_valid | valid_mask | implicit_present
```

Rules:

- `wavelength_known=false` requires `wavelength_nm=null`;
- `gsd_known=false` requires `gsd_m=null`;
- no numeric wavelength, GSD, unit, sign, scale, or offset may be inferred from a source or sensor name;
- an approximate source-level statement is provenance, not a numeric model condition;
- DEM, slope, radar, and interferometric channels never receive spectral wavelength conditioning;
- zero-filled missing channels require `channel_valid=false`; a present valid zero remains an observation.

### 5.4 Validity

The source record declares the actual keys and semantics for:

```yaml
validity:
  valid_mask_key: string | null
  pixel_valid_key: string | null
  channel_valid_key: string | null
  valid_mask_semantics: string
  pixel_valid_semantics: string | null
  channel_valid_semantics: string | null
```

Label-valid pixels alone enter loss and metrics. Channel-invalid inputs never enter aggregation. Padding
is invalid. Rendering or resizing must transform values and validity together and must preserve the
transform description.

## 6. P1: Canonical Benchmark v4 Small

### 6.1 Storage and immutability

P1 creates `benchmark/sami_landslide_hdf5_v4/small` under the configured benchmark root.

- The output path must not exist; build is fail-on-existing.
- Source HDF5 files remain read-only.
- Every selected image/mask HDF5 is byte-copied into `assets/<source_key>/...` while preserving its
  source-relative hierarchy; no HDF5 dataset is decoded, reordered, recompressed, or rewritten for storage.
- Canonical indexes use the benchmark copy for downstream loading and retain the source path only for
  provenance and independent replay.
- A materialization ledger binds every copied path, source path, role, byte size, and SHA-256. Symlinks,
  hard links, partial files, missing copies, and unregistered extra assets are invalid.
- The builder performs an exact source-size/free-space preflight before staging and copies through a
  temporary file followed by atomic replacement inside the private staging directory.
- Every JSON/JSONL artifact is written atomically with finite metadata.
- A manifest binds the source inventory, contract, copied-asset ledger, channel catalog, indexes, derived
  artifacts, and reports.

Minimum responsibilities:

```text
indexes/    source records, parents, and split projections
manifests/  benchmark, materialization, channel, source, split-assurance, and normalization bindings
assets/     immutable byte-identical image/mask HDF5 copies grouped by source
derived/    registered task assets only
reports/    validation, statistics, duplicate risk, and eligibility
```

Internal filenames may change, but responsibilities may not be merged in a way that hides lineage.

### 6.2 `CanonicalParentV4`

```yaml
schema_version: sami_canonical_parent_v4
parent_id: string
source_key: string
source_sample_id: string
canonical_split: train | val | test
split_assurance: verified_group_isolated | source_declared_unverified | train_only
evaluation_eligibility: strict | exploratory | train_only
channels: [ChannelDescriptorV1, ...]
image_ref: {...}
mask_ref: {...}
validity_ref: {...}
registered_views: [...]
group: {...}
source_record_sha256: sha256
record_sha256: sha256
```

`parent_id` is a stable canonical sample identity, not a claim of geographic grouping.

### 6.3 Registered views and normalization

- P1 registers only views supported by declared channels and alignment.
- P1 publishes a finite `ChannelCatalogV1` containing the global `channel_key` vocabulary and every
  source-index binding. This catalog is a model-input vocabulary, not a request to reorder stored tensors.
- The P2 diagnostic RGB view is constructed only when red, green, and blue identities are explicit.
- No source tensor is sent directly to an RGB vision-language tower.
- Source/channel normalization statistics use canonical train rows and valid pixels only.
- Normalization artifacts bind population, channel descriptors, computation protocol, and hash.
- Validation and test pixels never contribute to statistics.

`RegisteredRGBViewV1` is metadata, not an independently materialized image requirement:

```yaml
schema_version: sami_registered_rgb_view_v1
view_id: registered_rgb
role: rgb
source_indices: [red_index, green_index, blue_index]
channel_keys: [red_key, green_key, blue_key]
normalization_binding: sha256
mapping_evidence: string
```

The same channel selection and ordering apply to values and input pixel validity. Source
visualization stretches never enter this view. When per-channel `pixel_valid` exists it is selected
with the RGB channels; otherwise label validity is broadcast to present channels when available,
and only then may a full-grid present-channel mask be used.

### 6.4 P1 validator and acceptance

The independent validator reopens and replays:

- logical path resolution and datasets-root containment;
- source file hashes, copied file hashes, exact source/copy byte equality, and HDF5 dataset metadata;
- materialization-ledger completeness, benchmark-root containment, byte counts, and absence of links or
  unregistered assets;
- image/mask pairing and record hashes;
- channel descriptors, global channel-catalog bindings, source order, and known/unknown consistency;
- validity semantics;
- source split preservation and assurance/eligibility combinations;
- train-only exclusion from val/test;
- absence of not-ready rows;
- canonical index line hashes and aggregate manifests;
- absence of unbound or partial files;
- strict, exploratory, and train-only populations separately.

P1 acceptance requires an owner-run builder and an independently invoked owner-run validator whose live
report has `errors == []`. P1 acceptance does not create strict generalization evidence.

## 7. Model input contract

### 7.1 `DenseSampleV1`

The data layer consumes `CanonicalParentV4`; it never scans source directories and does not require
`../datasets` after Benchmark construction. It resolves only the parent record's
`benchmark_logical_path` under the immutable Benchmark package.

```yaml
schema_version: sami_dense_sample_v1
parent_id: string
source_key: string
canonical_split: train | val | test
split_assurance: string
evaluation_eligibility: string
channel_values: float32[C,H,W]
channel_descriptors: [ChannelDescriptorV1, ...]
channel_valid: bool[C]
pixel_valid: bool[C,H,W]
target: float32[1,H,W]
target_valid: bool[1,H,W]
transform_record: {...}
```

### 7.2 `DenseBatchV1`

```yaml
schema_version: sami_dense_batch_v1
channel_values: float32[N,Cmax,H,W]
channel_metadata: {...}
channel_valid: bool[N,Cmax]
pixel_valid: bool[N,Cmax,H,W]
target: float32[N,1,H,W]
target_valid: bool[N,1,H,W]
parent_ids: [string, ...]
source_keys: [string, ...]
evaluation_eligibility: [string, ...]
```

Only the batch collator may pad the channel dimension. Padded entries are invalid in both validity tensors.
Reordering channels together with their descriptors and validity must not change evaluation-mode logits
beyond the frozen numeric tolerance.

The model may not consume `source_key`; it is retained for sampling, reporting, and normalization lookup.

## 8. P2: registered-view direct-dense prerequisite

P2 implements K0, a diagnostic model that is never eligible to become the P3/P4 multimodal kernel.

- Input is the registered RGB view and its validity.
- The model is a convolutional encoder-decoder with at most 5 million trainable parameters.
- It outputs one binary logit map.
- It uses valid-pixel BCE-with-logits, positive soft Dice, and explicit no-target reporting.
- It contains no language model, multimodal fusion, candidate framework, or future-phase abstraction.

P2 freezes five per-source protocols and one unified protocol. Each covers 1, 4, 8, and 32 parents,
including positive and no-target rows where the source population permits.

Default engineering thresholds:

| population | positive Dice |
| --- | ---: |
| 1 parent | >= 0.99 |
| 4 parents | >= 0.99 |
| 8 parents | >= 0.98 |
| 32 parents | >= 0.95 |

Additional gates:

- no-target false-positive valid-pixel rate <= 0.005;
- invalid target pixels make exactly zero loss/metric contribution;
- FP32 checkpoint reload logits use `atol=1e-6`, `rtol=1e-5`;
- benchmark, population, config, and code bindings match after reload.

Owner-run evidence for all required populations is mandatory before P3 implementation begins. Failure
returns to P1 data, label, view, transform, or validity audit.

## 9. Research discipline

### 9.1 Evidence files

The segmentation research program maintains:

- `docs/research/segmentation_literature_matrix.md`;
- `docs/research/segmentation_first_principles.md`;
- `docs/research/segmentation_findings.md`;
- `docs/research/segmentation_research_state.yaml`.

The state file records program, cursor, hypotheses, immutable protocol bindings, owner gates, evidence,
decisions, rejected complexity, and next action. It contains no fabricated result placeholders.

### 9.2 Experiment protocol

Before implementation or execution, every experiment freezes:

```yaml
experiment_id: string
hypothesis_id: H1 | H2 | H3 | H4 | H5 | H6
status: proposed | implementation_ready | awaiting_owner_run | accepted | rejected | inconclusive
population:
  index_sha256: sha256
  split: train | val | test
  assurance: string
seeds: [integer, ...]
candidate:
  config_sha256: sha256
  code_identity: string
metrics: [...]
thresholds: {...}
budget:
  trainable_parameters_max: integer
  peak_memory_gib_max: number
  relative_flops_max: number | null
prediction: string
falsifier: string
owner_command: string
artifact_paths: [...]
```

No unexplained sweep is allowed. A result discovered outside the frozen protocol is exploratory and cannot
alone freeze the kernel.

### 9.3 Inner and outer loops

The inner loop is:

1. select the highest-priority hypothesis;
2. freeze its protocol;
3. implement the minimum change;
4. run static and permitted bounded checks;
5. provide the exact owner command;
6. inspect returned logs and artifacts;
7. update findings and state;
8. decide keep, simplify, reject, or the next experiment.

An outer synthesis runs after three to five real experiments or immediately after contradictory evidence.
It removes unsupported complexity, revisits primary literature, and freezes the next falsifiable questions.
There is no wall-clock job, automatic commit, or autonomous formal run.

## 10. P3: multimodal kernel research and unique freeze

### 10.1 Frozen hypotheses

- H1: explicit channel identity plus masked set aggregation is sufficient for variable channel combinations.
- H2: source-balanced sampling and channel/modality dropout reduce dominant-source collapse.
- H3: GSD conditioning helps only where reliable physical scale exists.
- H4: prompt conditioning helps only where prompt text changes target information.
- H5: one fusion layer is sufficient; deeper fusion does not justify its complexity.
- H6: heterogeneous non-RGB channels are more reliable through the channel-set encoder than through an RGB
  rendering path.

Formal comparison seeds are `3407`, `3408`, and `3409`.

### 10.2 K1: Channel-Set Dense Kernel

K1 is the default P3 candidate:

1. each scalar channel passes through a shared lightweight convolutional stem;
2. channel identity, modality family, and known/unknown physical metadata are embedded;
3. metadata is added to the channel feature at each pyramid level;
4. `channel_valid * pixel_valid` masks every feature;
5. sum-normalized masked mean aggregates channels at each level;
6. a shared hierarchical CNN processes the fused pyramid;
7. a simple FPN/U-Net-style decoder emits one binary logit map.

The initial K1 has no source embedding, attention, dynamic spectral weights, or fixed channel slot. A single
set/cross-modal attention layer is eligible only after a frozen experiment reproduces a specific failure of
masked mean.

Spectral dynamic weights are eligible only for channels with reliable numeric wavelengths and never apply
to elevation, slope, radar, or interferometric modalities. With no eligible cohort, the mechanism remains
unimplemented.

### 10.3 Losses

For target-valid pixels:

```text
L = L_bce + lambda_dice * L_positive_dice + lambda_empty * L_no_target_probability
```

- `L_positive_dice` is evaluated on positive parents;
- `L_no_target_probability` is the mean predicted foreground probability on valid pixels of no-target
  parents;
- invalid pixels contribute zero to loss and metrics;
- the protocol freezes all coefficients before an owner run.

### 10.4 K2: lightweight prompt conditioning

K2 may be implemented only after K1 passes its engineering prerequisite.

- It reuses K1 for all spatial encoding.
- A frozen local language interface produces a pooled query embedding.
- The visual tower is not invoked.
- One FiLM block conditions the bottleneck or decoder.
- The conditioning module adds at most 2 million trainable parameters.
- Text never emits boxes, coordinates, proposals, or masks.

Prompt tests include correct, null, random, semantically equivalent, and wrong-object text. Wrong-object
text without a corresponding target annotation is a sensitivity test, not supervised evidence that the
mask should be empty.

If all canonical prompts denote the same landslide target, the research state records
`prompt_information_status: redundant`. K2 may receive bounded engineering coverage but is ineligible for
formal three-seed selection.

### 10.5 Complexity ceiling

A larger language-driven pixel decoder is not implemented in the active program. Adding one requires a new
owner-approved ADR after K2 has eligible, positive evidence; that decision is outside the current scope.

Budgets:

- K1: at most 20 million trainable parameters;
- K2: at most 2 million additional trainable parameters, with frozen parameters reported separately;
- P4 target: at most 22 GiB peak allocated/reserved memory in the frozen owner-run configuration;
- a complex candidate cannot exceed 1.5 times K1 FLOPs without satisfying the replacement criterion.

### 10.6 Metrics and selection

Every eligible candidate reports:

- unified and per-source positive Dice and IoU;
- no-target false-positive valid-pixel rate;
- component recall at IoU 0.25;
- valid-pixel population;
- channel-subset and missing-modality results;
- channel permutation equivalence;
- known/unknown metadata cohorts;
- parameters, FLOPs, throughput, peak memory, and reload equivalence.

A more complex candidate can replace K1 only if, across the frozen seeds and population, it:

- improves unified positive Dice by at least 1.0 percentage point;
- improves at least three of five sources;
- improves median per-source Dice by at least 0.5 percentage point;
- worsens no source by more than 1.0 percentage point;
- worsens component recall by no more than 1.0 percentage point;
- produces an improvement larger than twice the pooled seed standard deviation;
- passes every validity, no-target, memory, lineage, and reload gate.

Among all qualifying candidates, P3 freezes the Pareto-simplest one in one accepted ADR. The ADR binds its
class, config, code identity, population, reports, and rejected alternatives. No selection field may be
filled before the owner artifacts exist.

Unselected candidates enter a replacement-owned deletion gate immediately after acceptance. Physical
deletion still requires exact targets, reference scans, and human approval.

## 11. P4: selected-model training engineering

P4 contains one selected model and no runtime candidate switch.

### 11.1 Assembly and sampler

- model assembly exposes encoder, fusion, decoder, and loss without CLI-owned algorithm logic;
- the source-balanced sampler chooses each ready source with equal probability;
- within a source it deterministically cycles canonical train parents;
- sampler seed, epoch, cursor, and source queues are checkpointed;
- positive/no-target composition is measured and reported, not silently rewritten.

### 11.2 Channel and modality dropout

- dropout applies only to present valid channels;
- it can drop a channel or a modality family according to frozen probabilities;
- at least one valid channel remains for every sample;
- the actual dropout mask is retained for reproducibility;
- target and validity tensors are never altered to hide a dropped input.

### 11.3 Metrics

Training and evaluation share one validity-aware implementation for:

- BCE and positive Dice loss;
- positive Dice/IoU;
- no-target specificity and false-positive area;
- component recall;
- valid-pixel counts;
- per-source and unified aggregates;
- strict, exploratory, and train-only partitions.

### 11.4 Checkpoint and code identity

`CheckpointEnvelopeV1` atomically stores:

- model, optimizer, scheduler, scaler, and gradient-accumulation state;
- RNG states;
- sampler state and next-sample cursor;
- benchmark manifest, split index, source binding, and normalization hashes;
- canonical config hash;
- model schema and selected-kernel identity;
- Git HEAD, canonical dirty-diff hash, and source-tree manifest hash;
- step, epoch, best metric, and finite publishable metadata.

Recursive finite checks reject NaN and Inf before artifact replacement. Strict resume rejects every identity
mismatch. Strict reload must reproduce FP32 logits within the frozen tolerance, and resume must reproduce
the next sampler sequence and optimizer transition.

### 11.5 CLI and export

The package exposes separate commands for:

- `train`: start or strictly resume the selected model;
- `evaluate`: evaluate a bound checkpoint without changing it;
- `export`: produce a lineage-bound inference package;
- bounded overfit and owner smoke protocols.

Export contains the selected model, channel vocabulary, metadata encoding, normalization binding, inference
config, model/data/code identity, and a strict loader. It contains no training optimizer state.

Visualization uses a registered RGB view when available; otherwise it renders an explicitly labelled
single channel. Every panel shows prediction, target, and valid region without inventing unavailable
physical colour.

### 11.6 P4 acceptance

Codex may perform static checks, focused tests, and the explicitly authorized bounded smoke, but cannot
produce P4 acceptance.

The owner must return:

- 1/4/8/32-parent and approved extended overfit reports;
- no-target and validity reports;
- strict reload and resume reports;
- export reload evidence;
- finite metadata and lineage replay;
- peak memory for the frozen single-GPU configuration.

Only after reopening those artifacts, updating README, Progress, handoff, and deletion gates may the phase
be labelled `P4_MODEL_ENGINEERING_ACCEPTED`.

## 12. P5: formal evaluation

P5 is outside the active continuous Goal. Its authority is frozen now to prevent evidence mixing.

Reports have three disjoint sections:

1. `strict`: verified group-isolated cohorts only;
2. `exploratory`: source-declared unverified cohorts;
3. `train_only`: overfit/training behaviour only, never a generalization metric.

Required metrics are per-source and unified positive Dice/IoU, no-target specificity and false-positive
rate, component recall, target coverage, valid-pixel population, parameters, memory, and throughput.

When no strict cohort exists:

```yaml
strict_generalization_status: unavailable
strict_population: 0
```

No prose may imply strict generalization from exploratory results.

## 13. P6: robustness and necessary ablations

P6 freezes and runs only ablations needed to explain the selected model:

- channel subsets and single-channel removals;
- whole-modality absence;
- channel enumeration permutation;
- known versus unknown GSD conditioning;
- source holdout;
- correct, null, randomized, equivalent, and wrong-object prompts where applicable;
- channel-valid and pixel-valid counterfactuals;
- masked mean versus an accepted additional fusion layer;
- accuracy/parameters/FLOPs/memory/latency Pareto analysis.

Each ablation preserves population, seed, metric, and budget. P6 cannot add a new production architecture
without returning to an owner architecture decision.

## 14. P7: reproducibility, export, and formal run package

P7 produces:

- multi-seed summaries with mean, standard deviation, and exact run bindings;
- config/data/model/code lineage;
- checkpoint and inference exports with strict reload;
- one README-owned formal training command and one formal evaluation command;
- machine-readable environment and dependency information;
- a reproducible package containing protocols, configs, manifests, and report schemas.

It contains no excluded description or joint-training functionality and does not duplicate the README
runbook in policy documents.

## 15. P8: residual replacement-owned deletion

P8 handles only assets whose replacement gate did not mature in P1-P7. It is not a universal waiting point.

Every deletion entry records:

```yaml
owner_phase: P1 | P2 | P3 | P4 | P5 | P6 | P7 | P8
replacement: string
delete_after: [condition, ...]
references_checked: boolean
tests_removed_or_replaced: [path, ...]
docs_removed_or_replaced: [path, ...]
approved_by: string | null
deleted_commit: string | null
```

Before physical deletion:

- resolve exact paths and hashes;
- verify the accepted replacement;
- scan imports, configs, tests, docs, entry points, and generated references;
- obtain explicit human approval;
- use `apply_patch` for repository text;
- leave `deleted_commit=null` until the owner creates a commit.

Raw datasets, accepted external artifacts, unresolved legacy paths, and historical audit records are not
implicitly deletable.

## 16. Phase gates

### P1

- five ready sources appear in their allowed training population;
- the empty source produces no row;
- native splits and assurance labels replay;
- every selected HDF5 pair is present below Benchmark `assets/`, byte-identical to the bound source, and
  independently replayable without source-path fallback;
- channel catalog and per-record descriptors preserve source channel order while exposing physical meaning;
- source bytes and HDF5 metadata replay;
- owner builder and independent validator return `errors == []`.

### P2

- all per-source and unified 1/4/8/32 protocols meet their frozen thresholds;
- no-target, target validity, and reload pass;
- K0 contains no P3 abstraction.

### P3

- every eligible candidate has a frozen, replayable protocol;
- populations, seeds, and budgets are fair;
- one accepted ADR freezes one kernel;
- unselected candidates enter deletion gates;
- no runtime switch remains.

### P4

- selected-model assembly and all training engineering exist;
- owner overfit, reload/resume, no-target, validity, export, lineage, and memory gates pass;
- documentation and handoff match the live artifacts.

### P5-P8

Their gates are owned by their sections and cannot be inferred from P4.

## 17. Documentation and evidence ownership

- `README.md`: the only run manual.
- `AGENTS.md`: repository agent rules.
- `REFACTOR_PROGRESS.md`: the current evidence cursor.
- `docs/handoffs/`: commands, exit codes, artifacts, risks, and next action.
- `docs/adr/`: accepted decisions, not experiment logs.
- `docs/audits/`: source, contract, reuse, reset, and deletion evidence.
- `docs/research/`: literature, first principles, protocols, state, and findings.

Authority documents contain no transient run hashes, checkpoint inventories, or failure logs.

## 18. Execution permissions for the active program

Codex may run:

- read-only repository and HDF5 inspection;
- UTF-8, Markdown, JSON, YAML, AST, and diff checks;
- focused CPU/unit tests;
- synthetic or accepted-tiny-fixture GPU smoke with batch size 1 and at most two optimizer steps.

Codex may not run:

- the Benchmark builder or independent validator;
- owner micro-overfit or three-seed comparisons;
- formal memory gates, long training, or formal evaluation;
- repository downloads, commits, or pushes;
- writes to `../datasets`;
- overwrites of any existing `../benchmark` artifact.

The bounded smoke is diagnostic only, writes no benchmark artifact, and cannot establish phase acceptance.
