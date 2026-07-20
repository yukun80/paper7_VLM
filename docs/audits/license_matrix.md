# P0 License Matrix

- Audit date: 2026-07-20
- Purpose: record evidence and hard eligibility gates for code, weights, datasets, and publication.
- This is an engineering provenance audit, not legal advice. The project owner approved Apache-2.0 for greenfield project code; restricted-data use remains a separate decision.
- No row marked `unknown`, `unverified`, or `human decision pending` is training-eligible.

## Evidence policy

1. Prefer a license file or model/data card in the exact local asset.
2. Otherwise use the official upstream repository or official dataset page.
3. A paper saying that data are public or downloadable is not a data license.
4. An aggregate dataset license does not override the licenses of its component imagery or annotations.
5. If code and weights/data have different terms, record them separately.

## Project publication gate

| Asset | Evidence | Status | Allowed now | Required action |
|---|---|---|---|---|
| SAMI-GroundSegDesc greenfield repository code and documentation | Root `LICENSE` and `NOTICE`, accepted by the project owner on 2026-07-20 | Apache-2.0 | Greenfield implementation and documentation may be distributed under Apache-2.0 | This decision does not license legacy code, datasets, model weights, checkpoints, generated benchmark content, or third-party assets; retain applicable notices and close each asset separately. |
| Legacy repository code | Git history only; no root license found | `unlicensed for redistribution` | Preserve locally and rewrite greenfield code | Do not publish or copy legacy code outside this repository without owner/legal review. |

## Upstream code and model weights

| Project/asset | Exact evidence | Observed terms | P0 use decision | Obligations and unresolved points |
|---|---|---|---|---|
| Qwen3-VL code | [Official Qwen3-VL repository](https://github.com/QwenLM/Qwen3-VL) includes a `LICENSE` and identifies Apache-2.0 | Apache-2.0 | Declared dependency/wrapper allowed in P2 | Pin versions; retain license/notice; do not vendor the repository. |
| Local Qwen3-VL-2B-Instruct weights | `models_zoo/Qwen3-VL-2B-Instruct/README.md`, SHA-256 `5fc5be1ca9a3910399bd6239ee5086ab5d82a2a59c5d2b00e887a8835cc110e4`, declares `license: apache-2.0` | Apache-2.0 in the local model card | Preserve read-only; use only after P2 records exact model identity and card hash | The local directory is not tied to an upstream Git commit. Keep the model card with any permitted redistribution. |
| PSALM code | [Official PSALM repository](https://github.com/zamling/PSALM) identifies Apache-2.0; local `models_zoo/PSALM/README.md` also declares Apache-2.0 | Apache-2.0 | Reference or minimal G0 fallback only | Attribute PSALM and its upstream bases; do not copy LLaVA/Swin/full repository. Review any checkpoint-specific model card before redistribution. |
| Detectron2 code | [Official Detectron2 repository](https://github.com/facebookresearch/detectron2) states Apache-2.0 | Apache-2.0 | Architecture/evaluator reference; no whole-repo dependency planned | If a small utility is copied, preserve copyright/license notice and record exact source revision. Detectron2 model-zoo weights have separate terms and are not approved by the code license alone. |
| Mask2Former code | [Official archived repository](https://github.com/facebookresearch/Mask2Former) states the majority is MIT, with MIT and Apache-2.0 subcomponents | Mixed: MIT plus identified Apache-2.0 portions | Reference and minimal G0 fallback only | Track provenance per copied file; do not infer that all dependencies or model weights are MIT. |
| SAM2 code/checkpoints | [Official SAM2 repository](https://github.com/facebookresearch/sam2) states its model checkpoints and code are Apache-2.0; optional `cc_torch` code has BSD-3-Clause terms | Apache-2.0 plus BSD-3-Clause optional component | Optional isolated G0 baseline allowed | Preserve both notices if the optional component is used; video/tracking remains out of scope. |
| ms-swift | [Official ms-swift repository](https://github.com/modelscope/ms-swift) states Apache-2.0 | Apache-2.0 | Reference only | No runtime code reuse and no second Trainer. Recheck the pinned release if future code reuse is proposed. |
| Grasp Any Region code | [Official GAR repository](https://github.com/Haochen-Wang409/Grasp-Any-Region) states Apache-2.0 | Apache-2.0 | Prefer independent GAR-lite implementation; small adaptation only with attribution | Do not copy AnyRes, PerceptionLM, XTuner, or its full data pipeline. Record exact file/revision for any adapted code. |
| Grasp Any Region dataset/card | [Official Hugging Face dataset card](https://huggingface.co/datasets/HaochenWang/Grasp-Any-Region-Dataset/blob/main/README.md) declares Apache-2.0 | Apache-2.0 at the aggregate card | Not a P0/P1 training source; later evaluation requires a separate source-content audit | Verify that underlying images/annotations permit the intended use and redistribution; the aggregate label alone is not assumed to override component terms. |
| MIGRANT | Task specification records MIT, but P0 did not resolve an official code repository/revision; only the official paper/dataset pages were discoverable | `unverified for code reuse` | Research taxonomy reference only | No code copying. A later proposal must identify the official repository, exact revision, and license file first. |
| RSGPT code | [Official RSGPT repository](https://github.com/Lavender105/RSGPT) has no root license file in the audited listing | No code license found | Code copying prohibited | Repository visibility is not redistribution permission. |
| EarthGPT code | [Official EarthGPT repository](https://github.com/wivizhang/EarthGPT) did not expose a root license in the P0 audit | No code license found | Code copying prohibited | Use only data records whose individual source licenses are closed in the P1 registry. |
| Qwen3-VL-Seg | Task specification/paper reference; no confirmed official implementation was established | `unknown` | Paper-derived independent implementation candidate only | Third-party reproductions must not be represented as official and require their own license review. |

## Local raw and derived data

`training-eligible now` is deliberately false for every source at P0. P1 must create a record-level license registry and may change eligibility only with exact evidence and human approval where noted.

| Source | Local evidence | Observed terms | Training-eligible now | P1 gate |
|---|---|---|---:|---|
| GDCLD | No local README/LICENSE located under `/home/yukun80/codes/datasets/GDCLD` | Unknown | No | Identify official source, exact version, image/mask ownership, research/redistribution terms, and attribution. |
| LMHLD | No local README/LICENSE located; content is largely packaged archives | Unknown | No | Identify official source/version/license before extraction is indexed; record archive hash and downstream redistribution terms. |
| Sen12Landslides | Local `/home/yukun80/codes/datasets/Sen12Landslides/README.md`, SHA-256 `77a5f542e7d57ff89ec62da8bf05d51c257700507bb24a04a3e6cffe1d5404a2`, declares CC-BY-4.0 and names Sentinel/Copernicus/DEM provenance | CC-BY-4.0 plus underlying product attribution/terms | No | Register the required attribution, product provenance, units, and the single/contemporaneous temporal slice permitted by the task. |
| Landslide4Sense | No local README/LICENSE found; the [dataset paper](https://arxiv.org/abs/2206.00515) establishes availability but not a conclusive data license | Unknown | No | Obtain the official dataset license/terms. Do not equate paper copyright or download access with data permission. |
| multimodal-landslide-dataset | No local README/LICENSE found | Unknown | No | Resolve official source, exact version, modality provenance, and license for images and masks. |
| LandslideBench_agent | No local README/LICENSE; directory contains materialized images/masks and Qwen JSONL derivatives | Unknown/derived | No | Trace every row to an authorized raw source. It cannot substitute for the canonical raw-source scan. |
| MMRS-1M aggregate | No local README/LICENSE at `/home/yukun80/codes/datasets/MMRS-1M`; [EarthGPT](https://github.com/wivizhang/EarthGPT) describes an aggregate built from many datasets | Per-component license required | No | Audit only RSICD, UCM-Captions, Sydney-Captions, NWPU-Captions, RSITMD, and DIOR-RSVG individually. Exclude aggregate `total.json`, classification, ordinary detection, VQA, and unrelated infrared tasks. |
| RSGPT RSICap/RSIEval | Local data has no license file; [official RSGPT README](https://github.com/Lavender105/RSGPT) states that DOTA images/annotations are academic-only and commercial use is prohibited | Academic use only; commercial use prohibited | No | Owner explicitly approves restricted academic use; registry records DOTA provenance, acquisition evidence, split role, noncommercial restriction, and redistribution limits. |
| DisasterM3 | Local `/home/yukun80/codes/datasets/DisasterM3/README.md`, SHA-256 `5b941adb1bbf27303aa92d97a7a1e1526582e91cc11dd80e739c58782325c1b4`, declares CC-BY-NC-SA-4.0 and academic-only use | CC-BY-NC-SA-4.0/noncommercial | No | Excluded from model inputs because the frozen task forbids bi-temporal change/recovery tasks. It may inform taxonomy with attribution only. |

## Caption and grounding subsource requirements

Before any MMRS/RSGPT record becomes eligible, P1 must have one registry row per underlying source with at least:

- `source_name`, `source_version`, `source_url`, local path and content hash;
- `license_name`, exact license document/URL and evidence hash;
- owner/copyright and required attribution;
- research, commercial, derivative, and redistribution permissions;
- image license separated from annotation/caption license;
- allowed task roles and prohibited task roles;
- reviewer, decision date, and human approval when terms are restrictive or ambiguous.

DIOR-RSVG is eligible only for region alignment after its image and expression terms are closed; it is never promoted to detailed global-caption supervision by license approval alone.

## Dependency-license constraints for the eventual root project

- Apache-2.0 and MIT components can generally be combined when their attribution and notice obligations are preserved, but P0 does not make a legal compatibility ruling for the final distribution.
- Model code, model weights, datasets, output annotations, and generated benchmark packages require separate provenance records.
- Optional baselines must be isolated so their dependencies and licenses do not silently become requirements of the main Qwen stack.
- The root `NOTICE` must be updated before distribution whenever a copied or materially adapted upstream file is introduced. Pure paper-derived reimplementation still requires scientific citation.
- The Apache-2.0 project license does not authorize publication of weights or benchmark packages until their data, base-model, and third-party terms are accepted separately.

## Open human decisions

1. Decide whether academic-only RSGPT/DOTA material may be used and whether resulting artifacts may be redistributed.
2. Approve source-by-source data eligibility after P1 produces exact license records.
3. Decide whether any trained weights can be published after reconciling all training-source and base-model terms.

## P0 license gate result

`root code license accepted; data and weight publication remain blocked`. P1 must fail closed: `license_status=unknown` must never appear in a training-eligible index, and Apache-2.0 for greenfield code must not be presented as permission for legacy or third-party assets.
