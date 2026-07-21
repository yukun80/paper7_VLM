# P0 Code and Dependency License Audit

- Audit date: 2026-07-20
- Governance correction: 2026-07-21
- Status: historical P0 evidence amended by accepted `ADR-0003`
- Scope: greenfield project code, copied code, declared dependencies, and model-weight publication provenance.

This audit is not a raw-data eligibility matrix. Under `ADR-0003`, P0-P7 builders do not query,
infer, compare, or approve raw-data licenses. Locally provided research data enter Canonical
Benchmark v3 only through scientific and technical checks. Public redistribution of raw images,
materialized benchmark packages, derived data, checkpoints, or weights is a separate future
publication/release review and is not a P0-P7 construction gate.

## Project code boundary

| Asset | Evidence | Engineering decision |
|---|---|---|
| SAMI-GroundSegDesc greenfield code and documentation | Root `LICENSE` and `NOTICE`, accepted by the project owner on 2026-07-20 | New project code and documentation use Apache-2.0; retain applicable notices. |
| Legacy repository code | No root license existed at the audited baseline | Preserve locally and independently rewrite; do not copy legacy implementations into greenfield code. |

The root project license does not itself grant permission to redistribute third-party code,
model weights, raw datasets, materialized benchmark data, or generated checkpoints.

## Upstream code and model-weight evidence

| Project/asset | Audited evidence | P0 engineering decision | Obligation |
|---|---|---|---|
| Qwen3-VL code | Official repository identifies Apache-2.0 | Declared dependency/wrapper in P2 | Pin the dependency and retain notices; do not vendor the repository. |
| Local Qwen3-VL-2B-Instruct weights | Local model card declares Apache-2.0 | Preserve read-only; P2 records exact identity | Keep the model card and separately review any future weight redistribution. |
| PSALM code | Official repository and local model card identify Apache-2.0 | Reference or minimal G0 fallback | Attribute the exact reused source; do not copy the complete LLaVA/Swin stack. |
| Detectron2 | Official repository identifies Apache-2.0 | Architecture/evaluator reference | Record exact source and notice for any copied utility. |
| Mask2Former | Official repository contains MIT and identified Apache-2.0 portions | Reference/minimal fallback | Track provenance per copied file. |
| SAM2 | Official code/checkpoint terms identify Apache-2.0; optional component is BSD-3-Clause | Optional isolated G0 baseline | Preserve applicable notices; video/tracking stays out of scope. |
| ms-swift | Official repository identifies Apache-2.0 | Reference only | Do not add a second Trainer or vendor the repository. |
| Grasp Any Region | Official repository identifies Apache-2.0 | Prefer independent GAR-lite implementation | Attribute any small adapted section; do not copy its full stack. |
| MIGRANT | No exact official repository/revision was resolved in P0 | Scientific taxonomy reference only | No code copying without later exact code provenance. |
| RSGPT code | No root code license found in the audited repository listing | Data-format reference only; no code copying | Repository visibility is not code redistribution permission. |
| EarthGPT code | No root code license found in the audited repository listing | Data-format reference only; no code copying | Independently implement required parsers. |
| Qwen3-VL-Seg | Paper reference; no confirmed official implementation | Paper-derived independent candidate only | Do not represent third-party reproductions as official. |

## Raw-data governance correction

The original P0 audit contained source-by-source permission columns and human authorization
requests. Those clauses are superseded by `ADR-0003` and have no runtime or acceptance effect.
P1 records only the following provenance fields for each source/component:

```yaml
source_key: sen12_landslides
source_name: Sen12Landslides
source_root: datasets/Sen12Landslides
source_document: datasets/Sen12Landslides/README.md
citation_key: sen12_landslides
upstream_url: null
provenance_notes: local research copy provided by project owner
```

No permission booleans, review status, reviewer, decision date, academic-use flag, or runtime
license error belongs in the source registry, canonical records, build manifest, validation
report, or CLI preflight.

## Remaining human decisions

No raw-data authorization decision is required for P1-P7 construction, training, or evaluation.
Before a future public release, the project owner must independently define the release payload
and review the redistribution terms of every included third-party asset. That later review must
not be retrofitted as a Benchmark v3 builder gate.

## Corrected P0 result

The Apache-2.0 greenfield code boundary is accepted. Code-copying and dependency notice
obligations remain active. Raw-data selection is governed by scientific provenance, readability,
schema/coordinate/valid-region validation, grouping, split isolation, and duplicate detection—not
by this audit.
