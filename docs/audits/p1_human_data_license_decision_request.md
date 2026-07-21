# P1 human data-license decision request

## Why this decision is required

P1 engineering is complete through a deterministic synthetic Small build, but a real Small build is
not authorized. All nine live registry rows remain `allowed_for_training=false`; the builder therefore
stops before raw decode and before creating an output directory. This is the governing manual stop
condition `license unknown for requested training source`.

Codex has not made and will not infer a legal/data-use decision. Public availability, a paper, the
project Apache-2.0 code license, or an audit candidate is not permission to train or redistribute data.

## Narrowest technically ready decision: Sen12Landslides

Exact local evidence:

- document: `datasets/Sen12Landslides/README.md`;
- SHA-256: `77a5f542e7d57ff89ec62da8bf05d51c257700507bb24a04a3e6cffe1d5404a2`;
- declared dataset license: `CC-BY-4.0`;
- named underlying sources/obligations: Sentinel-1, Sentinel-2/Copernicus and Copernicus
  WorldDEM-30/DLR/Airbus attribution statements in the same README;
- local source layout: 39,556 NetCDF files under `s2/`, `s1asc/` and `s1dsc/`;
- frozen loader policy: only triplets with `annotated=True`; one acquisition nearest the event date
  per modality; maximum event offset 30 days; maximum selected cross-modality span 30 days; no
  pre/post pair, change field or change target; official binary mask; SCL cloud/nodata exclusion.

The owner must provide a written decision for every field below:

```yaml
source_key: sen12_landslides
decision: approve | reject
allowed_for_training: true | false
allowed_for_evaluation: true | false
allowed_for_redistribution: true | false
academic_only: true | false
allowed_task_roles: [inventory, t1]  # revise explicitly if needed
attribution: "<owner-approved exact attribution/NOTICE text>"
reviewed_by: "<human name or accountable role>"
review_date: YYYY-MM-DD
notes: "<restrictions on benchmark assets, checkpoints, weights or publication>"
```

If approved, update the `sen12_landslides` row identically in both
`configs/benchmark_v3_small.yaml` and `configs/benchmark_v3_full.yaml`. Do not change another source
row and do not set redistribution permission unless that permission was explicitly decided.

## Separate decisions not bundled with Sen12

- GDCLD, LMHLD, Landslide4Sense, multimodal-landslide-dataset and LandslideBench derived assets still
  require their own source/ownership/grouping and data-license decisions.
- MMRS-1M requires one decision per selected component and separate image/annotation terms; the
  aggregate must not override those terms.
- RSGPT/RSICap/RSIEval is DOTA-derived and restricted academic-only in the audited evidence. Owner
  approval, noncommercial restrictions and redistribution policy must be explicit. RSIEval remains
  permanent test-only regardless of approval.
- DisasterM3 remains outside the frozen model-input scope because it is a pre/post change source.

## Resume boundary

After the human decision is recorded and both registry configs are updated consistently, resume P1
with the root README command:

```bash
sami-gsd data build --config configs/benchmark_v3_small.yaml
```

The command must target a new directory. P1 is still incomplete until a second independent real
Small build has the same aggregate hash and all real validation/license/duplicate gates pass.
