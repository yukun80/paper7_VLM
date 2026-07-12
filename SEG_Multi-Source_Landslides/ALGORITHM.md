# SANE-QMEF-PMRD Algorithm

## Problem Scope

Multi-Source Qwen-PSALM-Seg solves single-temporal or contemporaneous multi-source landslide instruction segmentation. A sample may contain any number of optical, multispectral, SAR, terrain, and deformation products with different channel counts, native H/W, GSD, validity, and quality. The model predicts a semantic landslide mask through a set of mask proposals; it does not use a bounding-box branch.

## Typed Input

Each `ModalityInstance` stores

```text
(image, family, sensor, product_type, band_names, band_metadata,
 orbit, units, signed, native_gsd_m, aligned_gsd_m, valid_mask, quality)
```

`ModalityBatch` keeps a variable-length instance list per sample and a shared segmentation canvas. An `ActiveModalitySubset` is sampled before prompts, cached-view selection, SANE, or QMEF. Resize/pad produces a canvas `valid_mask`; invalid pixels are excluded from losses, connected-component matching, instruction-sensitivity signatures, metrics, and restored-size comparison.

## SANE

The Sensor-Aware Native-Scale Encoder applies a shared single-band stem to every physical band. Materialized modality validity is applied before the first convolution, and band-attention pooling is validity weighted, so finite nodata values cannot leak through the stem or alter band weights. A band token combines registered family, sensor, product, band, orbit, units, measurement geometry, sign convention, continuous native/aligned GSD, spectral physics, and quality. Learned band attention aggregates all bands without truncating Sentinel-2 or fixed-channel packing:

```text
f_m^1/4 = sum_b softmax(g(pool(s(x_mb)), e_mb)) * (s(x_mb) + e_mb)
```

The production pretrained path renders 256-pixel views and caches Qwen-ViT layers 5/11/17/23 at `16/8/6/4` spatial sizes, then adapts them into per-modality `1/2`, `1/4`, `1/8`, and `1/16` maps. This retains the shallow patch grid for boundary refinement without storing every deeper layer at the largest resolution. The raw physical encoder is retained as a near-zero initialized residual at every scale, including the shallow `1/2` boundary feature. Cache manifest `spatial_channels` determines adapter input dimensions; it is not hard-coded to a particular Qwen revision.

When native arrays or the reference mask are resized into a bucket, SANE uses the effective encoder/canvas GSD rather than the source-file GSD. Original GSD remains in evaluation metadata for ground-area recovery.

## Semantic Evidence

The production controller loads the Qwen language decoder in NF4 4-bit and trains QLoRA only on q/k/v/o projections in the last four language blocks. The sequence order is task/condition text, interleaved physical view descriptions and active-view tokens, six post-context evidence anchors, and learned `<MASK_i>` embeddings. Every compressed physical view is wrapped by Qwen's pretrained vision-start and vision-end token embeddings, reducing the distribution shift caused by injecting cached visual embeddings directly into the language decoder. Hidden states at the mask positions are the only PMRD query initialization; there is no parallel PMRD mask-token path.

The anchors represent global, optical, multispectral, SAR, terrain, and deformation evidence. They occur after all active views, so they can summarize cross-source context. Cache v3 stores parent-level per-view tokens and Qwen-ViT layers 5/11/17/23; the active subset dynamically removes unavailable views before the language decoder.

The default view strategy retains multiple adaptive vision tokens. `image-end` and learnable `attention` pooling are separate training ablations and are bound into the checkpoint architecture protocol.

The multi-view renderer produces sensor-specific views: optical true color, strict-band S2 true/false color, S1 VV/VH/difference by orbit, product-aware terrain views, and signed zero-centered InSAR with a dataset-fixed scale. Visual cache keys include rendered content hashes, renderer revision, prompt revision, model/processor revision, pooling method, and full-subset signature. `shuffled`, `text-only`, `image-text-delta`, and `remove:<family>` alter only Qwen evidence tokens and never replace SANE dense features.

Cache construction is parent-streaming rather than corpus-buffered. The Qwen vision tower is loaded once, the unused language decoder is released immediately, each parent is rendered and encoded once, and at most one shard of serialized CPU tensors is retained before atomic shard emission. Every view records its processor `grid_thw`, merger grid, render padding transform, per-layer tensor shapes, validity map, and source modality. These spatial fields participate in the record fingerprint. The manifest records `peak_buffer_records`, full local Qwen weight/config SHA-256 revisions, and train/val/test instruction-index content fingerprints, so scalability and benchmark freshness are explicit, testable protocol properties rather than assumptions derived from shard filenames.

## QMEF

Qwen-Guided Multi-Source Evidence Fusion aligns each native pyramid with a pure-PyTorch deformable sampler. Learned offsets are bounded by the native/aligned GSD ratio, and sampled validity masks suppress invalid keys.

A valid-weighted reliability prior is computed from pooled modality features, physical metadata, coverage, quality, and the matching Qwen family-evidence anchor. A learnable null slot conditioned by the global anchor lets the model reject all real evidence; real weights are not renormalized to one. Semantic FiLM is hard-masked after conditioning, so its shift cannot recreate padding or nodata. The fused valid mask is the geometric union of active valid supports and is independent of learned reliability: reliability controls evidence amplitude, not whether a pixel exists. Each mask query subsequently samples four points per scale over the unified `1/4`, `1/8`, and `1/16` modality memory, with the corresponding family anchor added to its modality score:

```text
A_qlmp = softmax(q_q^T k_lmp / sqrt(d) + log r_m)
z_q    = sum_lmp A_qlmp v_lmp
```

Sampling references come from coarse-mask centroids and learned offsets. A single softmax is applied jointly over scale, modality, and sampling point; scales are not normalized independently or averaged with fixed weights. Invalid keys are removed before this joint normalization. Reports export reliability, null/real evidence mass, query-modality mass, query-scale mass, sampling references, and sampling grids.

## PMRD

The Proposal-Set Mask Refinement Decoder consumes Qwen mask-position hidden states. It first predicts coarse proposals, then combines query modality mass with each sigmoid coarse spatial gate to construct query-specific `1/2` detail features in bounded query chunks. The gate directly suppresses out-of-proposal detail rather than retaining a fixed background floor. Coarse and refined dynamic mask heads include a learnable query bias initialized to `-2`, providing an explicit sparse-background prior. Updated queries and dynamic kernels produce refined proposal masks; no bounding box is used.

One semantic-evidence verifier predicts relevance logits for all proposals. The final semantic mask is a relevance-gated noisy union rather than a top-k softmax average. Neutral relevance is calibrated by query count:

```text
gate_q = sigmoid(relevance_q - log(max(Q - 1, 1)))
p(y=1) = 1 - product_q (1 - gate_q * sigmoid(mask_q))
```

This avoids the initialization failure where `Q` half-active proposals saturate the union as foreground.

## Proposal Supervision

The parent semantic mask is split with 8-neighborhood connected components. Components smaller than `max(4 px, valid_area * 5e-5)` are filtered.

- If component count is at most query count, Hungarian matching uses proposal BCE + Dice cost.
- If components exceed available queries, Hungarian subset loss is disabled and coverage-set supervision minimizes the best refined and coarse proposal BCE + Dice cost for every component. A query may win multiple components; only queries that win at least one component receive positive verifier targets.
- The same assignment supplies a low-weight coarse-mask auxiliary term.
- Relevance targets mark matched proposals positive and all proposals negative for empty masks.

The main objective is

```text
L = L_final_BCE + L_final_Dice
  + lambda_set * (L_refined_set + lambda_coarse * L_coarse_set + L_coverage)
  + lambda_verifier * L_relevance
  + lambda_consistency * L_missing_modality
```

Boundary loss is disabled by default. Diversity, gate entropy, query-usage balance, hard-combo weighting, and separate condition/evidence/visual ranking losses are not part of the main objective.

## Evaluation Protocol

Reports separate overall, positive-only IoU/Dice, negative accuracy, and empty-mask false-positive rate. Proposal-set reports include matched Dice, component recall/precision, relevance AP/AUC, matched-proposal mean relevance rank and normalized rank score, unmatched rejection, merge/duplicate/missed-component rates, and proposal-union Dice. Diagnostics preserve both the verifier-selected proposal and the highest-Dice query among GT-matched proposals. The latter is explicitly named `oracle_matched_proposal`, is derived from training/evaluation assignment, and must never be reported as deployable model output. Paired targets from one parent measure instruction contrast, while no-target rows measure empty prediction and rejection. The best checkpoint is selected by positive-only Dice by default. Metrics are computed both on the valid target canvas and after restoration to original H/W; malformed resize transforms are rejected instead of silently falling back to canvas metrics, and the canvas/original difference is recorded.

The task-balanced sampler allocates the epoch-wide batch quota before assigning each task group to its observed size buckets. This preserves the configured global 40/40/20 global/referring/no-target ratio while keeping every batch on one reference-grid size; missing task groups are renormalized only when absent from the whole split, not separately inside each bucket.

Instruction and visual truthfulness are evaluated with paired reports from the same checkpoint, split, preset, and exact sample IDs. Normal evidence must outperform each of shuffled/fixed-generic/no-semantic instructions and shuffled/text-only/at-least-one-family-removed visual evidence on a composite of final-mask, proposal-selection, component, and paired/no-target sensitivity metrics. Image-text delta is reported as a strategy comparison rather than assumed to be a degradation baseline.

The staged presets are `raw_sane_baseline`, `raw_sane_qmef`, `raw_sane_qmef_pmrd`, `pretrained_sane_qmef_pmrd`, `qwen_mask_query_frozen`, and `qwen_psalm_full`. The full Qwen preset first warms up the dense segmentation path and controller-side prompts, then activates the final four language-block LoRA adapters at a lower learning rate. A module should be treated as a validated contribution only after improving positive-only IoU/Dice, instruction sensitivity, or component-set metrics in at least two of three fixed-small-v2 seeds.
