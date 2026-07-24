# ADR-0006: Self-Contained HDF5 Benchmark and Explicit Channel Semantics

- Status: accepted
- Date: 2026-07-24
- Owner: project owner
- Scope: P1 storage, lineage, and later model-input semantics
- Supersedes: the reference-first/no-copy clauses of ADR-0004

## Context

The first Benchmark v4 implementation kept all image/mask HDF5 files under `../datasets` and wrote
only references into `../benchmark`. That did not produce an independently organized training
package. The five ready sources also have different channel counts and source orders, so a fixed
global channel tensor would either discard modalities or silently change their physical meaning.

## Decision

1. P1 byte-copies every selected image/mask HDF5 into
   `benchmark/sami_landslide_hdf5_v4/small/assets/<source_key>/...`, preserving the path relative to
   its source root.
2. HDF5 bytes and internal datasets are not decoded, reordered, recompressed, or rewritten.
   Symlinks, hard links, source fallback, and overwrite are forbidden.
3. Each array reference binds its portable source path, portable Benchmark copy path, SHA-256, byte
   size, HDF5 dataset key, shape, dtype, and layout.
4. A materialization ledger accounts for every copy. The independent validator hashes both source
   and copied bytes and rejects missing, extra, linked, partial, or mismatched assets.
5. Source channel order remains storage order. Each record carries `ChannelDescriptorV1`, and a
   Benchmark-owned channel catalog binds the global `channel_key` vocabulary to source indices,
   modality family, known/unknown wavelength and GSD, unit evidence, normalization, and validity.
6. Later models consume the descriptors/catalog and validity masks; they do not infer physical
   meaning from tensor position or receive `source_key` as a feature.

## Consequences

- Benchmark v4 is self-contained for downstream training and evaluation.
- Construction and independent provenance validation require the read-only source root; downstream
  model loading does not.
- P1 requires enough free space for an atomic staging copy. The builder performs an exact file-size
  preflight before copying.
- Physical copying increases storage and hashing time, but removes runtime dependence on mutable
  source paths and makes model lineage auditable.
- Unknown wavelength, GSD, unit, scale, or sign remains unknown. The catalog does not invent sensor
  metadata.

## Evidence required

- owner-run builder manifest with non-zero `materialized_asset_count` and
  `materialized_size_bytes`;
- owner-run independent validator with `errors=[]`;
- exact materialization-ledger/source-record projection;
- source/copy hash equality for every HDF5;
- channel-catalog semantic replay.
