# P0R HDF5 Source Audit

- Audit date: 2026-07-24
- Repository HEAD at reset start: `58813468d2cf3be715e399f5904a8d9f7f5c880d`
- Method: read-only inspection of HDF5-side indexes, channel schemas, conversion manifests,
  conversion summaries, statistics, error reports, and leakage reports
- Runtime actions: no converter, builder, validator, unit test, training, CUDA, or formal run
- Machine-readable snapshot: `docs/audits/hdf5_source_inventory.json`
- Frozen interface: `docs/audits/hdf5_source_contract.yaml`

## Conclusion

Five non-empty sources are technically ready for segmentation training. Their training admission
does not depend on complete location, group, canonical-index, or duplicate-component evidence.
Those evidence gaps instead determine evaluation credibility.

No current source is certified `verified_group_isolated`. Consequently, this P0R snapshot has no
strict evaluation cohort: four sources retain their native development/evaluation splits as
exploratory, Landslide4Sense is train-only, and Sen12 is not ready.

| Source | Pairs | Positive | No-target/background | Native split decision | Assurance / evaluation |
|---|---:|---:|---:|---|---|
| GDCLD | 13,447 | 11,593 | 1,854 | keep train 7,897 / val 4,459 / test 1,091 | source-declared unverified / exploratory |
| LMHLD | 28,185 | 28,008 | 177 | keep train 19,729 / val 5,637 / test 2,819 | source-declared unverified / exploratory |
| LandslideBench_agent | 2,130 | 1,819 | 311 | keep train 1,701 / val 210 / test 219 | source-declared unverified / exploratory |
| Landslide4Sense | 3,799 | 2,231 | 1,568 | all samples train-only | train-only / train-only |
| Multimodal | 6,084 | 5,495 | 589 | keep train 4,395 / val 1,689 | source-declared unverified / exploratory |
| Sen12Landslides | 0 | 0 | 0 | none | not-ready |

## Shared findings

- All five non-empty conversion error JSONL files contain zero rows.
- Source-side summaries/manifests include machine-specific absolute paths. They are useful for this
  local audit but are forbidden from Canonical Benchmark identity and aggregate hashes.
- HDF5 unifies storage, not science: channel order, units, preprocessing, validity, and grouping
  remain source-specific and must be bound per record.
- Image and mask are stored as separate HDF5 files for the audited cohorts. P1 must bind both file
  hashes, dataset keys, shapes, dtypes, layouts, and validity references.
- `group_id`, `location_key`, and duplicate-component identity can be null or incomplete. P1 must
  report that state and must not promote the split to strict.
- Native splits must not be randomly rewritten in P1.
- No audited channel schema provides a numeric centre wavelength. Every P1
  `wavelength_nm` is therefore null with `wavelength_known=false`.
- No audited source provides a reliable per-sample metre-scale GSD for the current tensor. Dataset
  ranges, native band resolutions, approximate target grids, and geographic-degree transforms
  remain provenance; every current model `gsd_m` is null with `gsd_known=false`.
- Model normalization is recomputed from canonical-train valid pixels. Source visualization
  percentile stretches and PNG policies are not model normalization.

## Source findings

### GDCLD

- Authoritative candidates:
  `datasets/GDCLD/jsonl/sample_index_{train,val,test}.jsonl`,
  `datasets/GDCLD/hdf5/conversion_manifest.jsonl`, and
  `datasets/GDCLD/hdf5/channel_schema.json`.
- 13,447 image HDF5 files pair with 13,447 mask HDF5 files.
- `/image` is float32 CHW `[3,1024,1024]` in Red/Green/Blue order.
- `/mask` is uint8 binary background/landslide.
- `/valid_mask` distinguishes label-valid pixels from nodata/padding; `/channel_valid` records
  channel presence.
- Native train/val/test is retained. Group isolation has not been independently replayed, so all
  evaluation is exploratory.
- The registered RGB channel indices are `[0,1,2]`; the schema explicitly declares
  Red/Green/Blue.

### LMHLD

- Authoritative candidates:
  `datasets/LMHLD/jsonl/sample_index_{train,val,test}.jsonl`,
  `datasets/LMHLD/hdf5/conversion_manifest.jsonl`, and
  `datasets/LMHLD/hdf5/channel_schema.json`.
- 28,185 image HDF5 files pair with 28,185 mask HDF5 files.
- `/image` is float32 CHW `[4,128,128]` in Blue/Green/Red/NIR order; all four channels participate
  in training.
- Source values are dataset-provided preprocessed values. Per-sample sensor, coordinates, CRS,
  acquisition date, physical unit, and reliable group identity are unavailable.
- The source report flags 177 zero masks, 39 full-one masks, and 18,100 masks below 10% coverage.
  These are training/statistics concerns, not automatic exclusion gates.
- Native train/val/test is retained as exploratory.
- The registered RGB channel indices are `[2,1,0]`, directly bound to the schema's declared
  Red/Green/Blue visualization order. The dataset-level 0.8-10 m range is not a per-sample GSD.

### LandslideBench_agent

- Authoritative candidates:
  `datasets/LandslideBench_agent/jsonl/sample_index_{train,val,test}.jsonl`,
  `datasets/LandslideBench_agent/reports/conversion_manifest.jsonl`, and
  `datasets/LandslideBench_agent/hdf5/channel_schema.json`.
- 2,130 image HDF5 files pair with 2,130 mask HDF5 files.
- `/image` is float32 CHW `[3,512,512]` in R/G/B order; `/mask` is uint8 binary.
- Image HDF5 files contain `/channel_valid`; the registered RGB indices are `[0,1,2]`.
- The leakage report contains 311 location-level cross-split conflicts. P1 must preserve and
  surface them; it must not claim strict location isolation or silently move samples.
- The 311 non-landslide records provide no-target masks.
- Dialogue text is unverified provenance/auto-candidate material only. It is not factual truth,
  expert description, causal supervision, or a P1 segmentation target.

### Landslide4Sense

- Authoritative candidates:
  `datasets/landslide4sense/hdf5/conversion_manifest.jsonl` and
  `datasets/landslide4sense/hdf5/channel_schema.json`.
- 3,799 image HDF5 files pair with 3,799 mask HDF5 files.
- `/image` is float32 CHW `[14,128,128]` in
  B01/B02/B03/B04/B05/B06/B07/B08/B09/B10/B11/B12/slope/DEM order.
- B8A is absent by source design. Source values are dataset-provided preprocessed values; physical
  units are not locally available.
- 114 samples have an all-zero slope channel and must remain explicitly visible to channel/data
  quality reporting. The channel is still present and valid; zero values must not be rewritten as
  a missing channel.
- The registered RGB indices are `[3,2,1]`, using the schema's explicit red B04, green B03, and
  blue B02 meanings. Reported native 10/20/60 m resolutions and the approximately 10 m target grid
  are not promoted to a precise model GSD.
- The available asset has no trustworthy canonical val/test index. All 2,231 positive and 1,568
  background samples enter training as `train_only`; P1 must not fabricate val/test.

### Multimodal

- Authoritative candidates:
  `datasets/multimodal-landslide-dataset/jsonl/sample_index_{train,val}.jsonl`,
  `datasets/multimodal-landslide-dataset/hdf5/conversion_manifest.jsonl`, and
  `datasets/multimodal-landslide-dataset/hdf5/channel_schema.json`.
- 6,084 image HDF5 files pair with 6,084 mask HDF5 files.
- `/image` is float32 CHW `[5,128,128]` in
  Red/Green/Blue/DEM/InSAR-encoded order.
- `/pixel_valid` is per-channel/per-pixel, `/channel_valid` is per-channel, and `/valid_mask`
  covers labels. The audit reports 9,580 invalid InSAR pixels.
- DEM and InSAR physical units/scale are not confirmed by local files; P1 may not infer them.
- The registered RGB indices are `[0,1,2]`, directly bound to Red/Green/Blue channel declarations.
- Native train/val is retained, but known spatial-neighbour leakage keeps evaluation exploratory.

### Sen12Landslides

- The directory exists but contains no auditable HDF5/index cohort.
- `ingestion_status` is `not_ready`; no Canonical records or synthetic split may be produced.
- Its absence does not block the other five sources or P0R acceptance.

## P1 obligations

P1 must generate per-record source and Benchmark-copy logical paths and file hashes, byte-copy every
selected image/mask HDF5 into the source-organized Benchmark `assets/` tree, independently reopen
both sides, preserve source split plus assurance labels, and report strict/exploratory/train-only
populations separately. It must publish explicit per-channel descriptors and a global channel
catalog, include all five non-empty sources in the training candidate population, and must not
upgrade any cohort to strict without new verified evidence.
