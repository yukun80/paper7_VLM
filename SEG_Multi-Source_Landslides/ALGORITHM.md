# SANE-QMEF-PMRD Algorithm

## Problem Scope

Multi-Source Qwen-PSALM-Seg solves single-temporal or contemporaneous multi-source landslide instruction segmentation. A sample may contain any number of optical, multispectral, SAR, terrain, and deformation products with different channel counts, native H/W, GSD, validity, and quality. The model predicts a semantic landslide mask through a set of mask proposals; it does not use a bounding-box branch.

## Typed Input

Each `ModalityInstance` stores

```text
(image, family, sensor, band_names, orbit,
 native_gsd_m, aligned_gsd_m, valid_mask, quality)
```

`ModalityBatch` keeps a variable-length instance list per sample and a shared segmentation canvas. Resize/pad produces a canvas `valid_mask`; invalid pixels are excluded from losses, connected-component matching, metrics, and restored-size comparison.

## SANE

The Sensor-Aware Native-Scale Encoder applies a shared single-band stem to every physical band. A band token combines family, sensor, band identity, orbit, continuous native/aligned GSD, and quality. Learned band attention aggregates all bands without truncating Sentinel-2 or fixed-channel packing:

```text
f_m^1/4 = sum_b softmax(g(pool(s(x_mb)), e_mb)) * (s(x_mb) + e_mb)
```

Family-specific blocks produce `f_m^1/4`, `f_m^1/8`, and `f_m^1/16`. SAR, terrain, and deformation products receive an auxiliary gradient band for shallow physical structure. Modality dropout removes complete instances before encoding and always retains at least one source.

## Semantic Evidence

The frozen Qwen text controller separately encodes task context, condition prompt, and evidence reasoning. Optional Qwen visual cache v2 contributes one token per sensor-aware rendered view. A lightweight attention pool creates a unified `SemanticEvidence`; proposal generation receives the task token, while condition/reasoning/visual tokens are consumed by evidence fusion and one verifier.

The multi-view renderer produces sensor-specific views: optical true color, S2 true/false color, S1 VV/VH/ratio by orbit, terrain value/hillshade/slope, and signed zero-centered InSAR. Visual cache keys include rendered content hashes, renderer revision, model/processor revision, pooling method, and authenticity-ablation settings.

## QMEF

Qwen-Guided Multi-Source Evidence Fusion aligns each native pyramid with a pure-PyTorch deformable sampler. Learned offsets are bounded by the native/aligned GSD ratio, and sampled validity masks suppress invalid keys.

A sample-level reliability prior is computed from pooled modality features, physical metadata, quality, and global semantic evidence. It only supplies a prior. Each mask query subsequently attends over modality-spatial tokens:

```text
A_qmxy = softmax(q_q^T k_mxy / sqrt(d) + log r_m)
z_q    = sum_mxy A_qmxy v_mxy
```

This allows different proposals and regions to use optical, SAR, terrain, or deformation evidence differently. Reports export both reliability priors and query-level modality attention.

## PMRD

The Proposal-Set Mask Refinement Decoder uses learnable mask tokens. A task-conditioned transformer first generates coarse queries and masks. Coarse soft masks pool high-resolution multi-source region evidence; the pooled evidence updates each query and produces refined proposal masks.

One semantic-evidence verifier predicts relevance logits for all proposals. The final semantic mask is a relevance-gated noisy union rather than a top-k softmax average. Neutral relevance is calibrated by query count:

```text
gate_q = sigmoid(relevance_q - log(max(Q - 1, 1)))
p(y=1) = 1 - product_q (1 - gate_q * sigmoid(mask_q))
```

This avoids the initialization failure where `Q` half-active proposals saturate the union as foreground.

## Proposal Supervision

The parent semantic mask is split with 8-neighborhood connected components. Components smaller than `max(4 px, valid_area * 5e-5)` are filtered.

- If component count is at most query count, Hungarian matching uses proposal BCE + Dice cost.
- If components exceed available queries, coverage-set supervision minimizes the best proposal cost for every component.
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

Reports separate overall, positive-only IoU/Dice, negative accuracy, and empty-mask false-positive rate. The best checkpoint is selected by positive-only Dice by default, preventing negative samples from rewarding an empty predictor. Metrics are computed both on the valid target canvas and after restoration to original H/W; their difference is recorded.

The staged presets are `sane_baseline`, `sane_qmef`, `sane_qmef_pmrd`, and `full_multiview`. A module should be treated as a validated contribution only after improving positive-only IoU/Dice or verifier best-query accuracy in at least two of three fixed-small-split seeds.
