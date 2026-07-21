# P0 Reuse Matrix

- Audit date: 2026-07-20
- Baseline candidate: `0c53624dd93159f78acd6d39a579b100d7e3255f`
- Scope: decide what may be preserved, reimplemented, referenced, or ultimately deleted during the SAMI-GroundSegDesc greenfield rewrite.
- Rule: this matrix authorizes no deletion and no legacy compatibility path.

## Decision vocabulary

| Decision | Meaning |
|---|---|
| Preserve read-only | Keep the current bytes for provenance, comparison, or human review; new runtime code must not load them. |
| Reimplement idea | Retain a scientifically necessary contract or generic engineering pattern, but write a new implementation and new public names under `src/sami_gsd/`. |
| Declared dependency | Use an upstream package/model through an explicit versioned dependency and wrapper; do not vendor its repository. |
| Reference only | Use the paper/repository for attribution or experimental design, with no copied runtime code. |
| Delete after gate | Remove from the active branch only after all `delete_after` conditions in `deletion_plan.yaml` and human approval are satisfied. |

## Repository assets

| Current asset | Current responsibility | P0 decision | Greenfield destination or gate | Constraints |
|---|---|---|---|---|
| Raw `/home/yukun80/codes/datasets/*` | Source imagery, masks, captions, and provenance | Preserve read-only | P1 technical scan and minimal provenance registry | Never modify raw data. Inclusion is decided only by scientific scope, readability, parseability, task evidence, grouping, leakage control, coordinates, valid regions, and duplicate validation. |
| `/home/yukun80/codes/benchmark/` | Intended generated benchmark root | Preserve path; currently empty | P1 creates a new, separately named Benchmark v3 package | Never reconstruct old V2/M1.1/Bridge/Unified implicitly. |
| `outputs/` | Legacy reports, caches, checkpoints, logs, metrics, visualizations | Preserve read-only | Source crosswalk and fair-comparison evidence only | Historical reports bind missing benchmark directories and are not current replay evidence. |
| `models_zoo/Qwen3-VL-2B-Instruct/` | Frozen local Qwen3-VL-2B model | Preserve read-only; declared upstream model dependency in P2 | Qwen wrapper under `src/sami_gsd/model/` | Verify exact model-card revision/license; no copied Qwen repository. |
| `models_zoo/PSALM/` | Legacy PSALM model/config | Preserve read-only; G0 reference/fallback only | P3 may implement a minimal PSALM-Lite candidate | Do not import the legacy model package or LLaVA/Swin stack. |
| `参考文献/` | Local research PDFs | Preserve read-only | Research attribution | Not a runtime or training input. |
| `external/` | Expected nested upstream repositories | Absent | None | Do not recreate whole-repository vendoring. |
| Root `.gitignore` | Protects data, outputs, weights, PDFs, and logs | Preserve until P7/P8 packaging review | Root packaging policy | New v3 outputs must use distinct names and remain ignored as appropriate. |

## Data contracts and engineering ideas

| Legacy contract/idea | Scientific value retained | P0 decision | New contract | Prohibited carry-over |
|---|---|---|---|---|
| `ModalityInstance` physical metadata | Sensor, product, bands/polarizations, orbit, units, signedness, GSD, quality, valid mask | Reimplement idea | `ModalityRecord` in canonical Benchmark v3 | No old class alias, loader, or serialized schema. |
| `ActiveModalitySubset` | One authoritative post-dropout modality set | Reimplement invariant | New typed batch/view contract | No fixed slot assumptions or inactive-modality leakage. |
| `MultisourceBackboneState` separation | Task-neutral backbone state | Reimplement idea | `QwenBackboneState` | No old cache tensors or state migration shim. |
| `SegmentationState` | Segmentation output kept distinct from language state | Reimplement idea | `GroundingState` | No QMEF semantic state or old checkpoint keys. |
| `RegionEvidenceState` | Region evidence kept distinct from segmentation | Reimplement idea | `RegionReadout` | No MGRR context ring, component replay, or old reliability stack. |
| Atomic JSON/JSONL writes | Crash-safe artifacts | Reimplement idea | Shared strict utilities | No direct import from legacy benchmark modules. |
| SHA-256 provenance binding | Reproducible lineage | Reimplement idea | v3 manifest/report bindings | No acceptance of missing bound files or copied hash fields without replay. |
| Parent-level split isolation | Leakage prevention | Reimplement invariant | v3 split and duplicate reports | No reuse of old split indexes. |
| Valid-mask propagation | Correct losses and metrics | Reimplement invariant | v3 transforms/model/evaluation | No implicit all-valid fallback in formal paths. |
| Checkpoint lineage and strict restore | Auditable training continuation | Reimplement idea | P7 checkpoint protocol | No legacy checkpoint compatibility loader and no `strict=False` fallback. |
| Ontology separation of deterministic geometry and modality evidence | Avoid unsupported physical claims | Reimplement semantics | P1/P6 v3 ontology/schema | Do not reuse old schema identifiers or Bridge publication protocol. |

## Model and runtime code

| Legacy asset | P0 decision | Replacement owner | Deletion gate summary |
|---|---|---|---|
| `models/sane.py::SensorAwareNativeScaleEncoder` | Delete after gate | P2 Sensor Adapter plus optional support residual; P4 accepted segmentation | P2 adapter and P4 segmentation accepted; baseline verified; references/tests/docs replaced; human approval. |
| `models/qmef.py::QwenGuidedEvidenceFusion` | Delete after gate | Native Qwen multi-image forward and lightweight segmentation path | P2 backbone and P4 segmentation accepted; full reference scan; human approval. |
| `models/qmef.py::ScaleAwareDeformableAggregator` | Delete after gate | Selected P3/P4 segmentation kernel | Same as QMEF. |
| `controllers.py::QwenMaskQueryController` | Delete after gate | Official online Qwen3-VL wrapper | P2 native forward accepted; P4 segmentation accepted; human approval. |
| `models/pmrd.py::ProposalSetMaskRefinementDecoder` | Delete after gate | Single G0-selected kernel | ADR-0002 accepted and P4 accepted; human approval. |
| `models/qpsalm.py::MultiSourceQwenPSALMSeg` | Delete after gate | `SAMI-GroundSegDesc` assembly | P4 replacement accepted; CLI/tests/docs replaced; human approval. |
| `description/modeling/mgrr.py::MultiGranularityRegionReplay` | Delete after gate | P6 GAR-lite exact-mask/RoI reader | P6 reader accepted including counterfactual tests; human approval. |
| `description/modeling/region_baselines.py` | Delete after gate | P6 formal baselines/reports outside runtime | P6 accepted and evidence preserved; human approval. |
| Legacy `data/`, `engine/`, `description/training`, `description/evaluation`, `description/protocols` | Delete after gate | P1/P4/P5/P7 greenfield modules | Owning replacement accepted; P7 entrypoints accepted; tests/docs replaced; human approval. |
| 36 `qpsalm-*` console scripts | Delete after gate | Single `sami-gsd` CLI with approved subcommands | Corresponding P1-P7 commands accepted and documented; reference scan empty; human approval. |
| `SEG_Multi-Source_Landslides/pyproject.toml` | Delete after gate | Root `pyproject.toml` | P7 install/CLI/runtime accepted; license/NOTICE resolved; human approval. |
| `qpsalm_v2_*.yaml`, `qpsalm_segdesc_small.yaml`, old root configs/schemas | Delete after gate | v3 benchmark/model/train/eval configs and schemas | Owning phase accepted; no old configuration reader remains; human approval. |
| Ten legacy tests | Delete after gate, never before replacement | Focused P1-P7 unit/integration/smoke tests | Each covered behavior mapped to a new test or explicitly retired; human approval. |
| `SEG_Multi-Source_Landslides/ALGORITHM.md` and archived legacy design docs | Preserve as history until P8, then remove from active tree after gate | README, accepted ADRs, live v3 reports | Current documentation complete, reference scan performed, human approval. Git history remains the archive. |

## Benchmark pipelines

| Legacy pipeline | Allowed P0/P1 use | New runtime use | Retirement condition |
|---|---|---|---|
| `scripts/1-benchmark/` | Audit source mappings and invariants only | Forbidden | P1 Benchmark v3 accepted, replacement tests/docs present, human approval. |
| `scripts/2-instruction/` | Audit prompt/task coverage only | Forbidden | P1 task views/instructions accepted, human approval. |
| `scripts/3-description/` | Audit caption sources, dedup rules, and provenance only | Forbidden | P1 language subset accepted, human approval. |
| `scripts/4-landslide-bridge/` | Preserve existing reviewer/arbitration materials and audit concepts | Forbidden | P5 region-description dataset replacement accepted; human-reviewed assets preserved separately; human approval. |
| `scripts/5-segdesc/` | Audit component lineage only | Forbidden | P5 unified greenfield task views accepted; human approval. |
| Root `run_1` through `run_5` shell scripts | Historical README commands only | Forbidden | Single `sami-gsd` replacements accepted and documented; human approval. |

## External projects

| Project | Reusable contribution | Allowed mechanism | P0 decision |
|---|---|---|---|
| Qwen3-VL | Native multi-image understanding, processor/chat template, official model loading | Declared Transformers/model dependency behind a narrow wrapper | Use in P2; do not copy the repository or invent a parallel visual cache protocol. |
| Detectron2 | Dataset/model/loss/evaluator separation and evaluator contract | Reference architecture; exceptionally small licensed utility only if attribution is recorded | Do not make it a main runtime dependency. |
| Mask2Former | Masked attention, matching, proposal decoder | Reference and minimal independently integrated G0/PSALM fallback | Archived repository; do not embed it wholesale. |
| SAM2 | Box/point prompt segmentation baseline | Optional isolated G0 baseline | Do not include video/tracking or make it the main model by default. |
| ms-swift | PEFT/QLoRA and run-lineage ideas | Reference only | Do not add a second Trainer or vendor ms-swift. |
| PSALM | LMM-updated mask tokens and proposal/classification decoupling | Minimal G0 fallback with attribution | Do not copy LLaVA/Swin/full PSALM stack. |
| Grasp Any Region | Mask prompts, global context, RoI-aligned replay | Prefer independent GAR-lite implementation; attribute any adapted small section | Do not copy AnyRes/PerceptionLM/XTuner/data pipeline. |
| MIGRANT | Task taxonomy and data-centric curriculum | Reference only | Do not copy vendor dependencies or its task-specific runtime. |
| RSGPT | RSICap/RSIEval roles | Parse locally provided scientific data with minimal provenance | Do not copy upstream code; preserve official-test semantics for RSIEval. |
| EarthGPT/MMRS-1M | Source-specific language subset and task format | Parse only the technically selected source components | Do not inherit the general-assistant objective, aggregate `total.json`, classification, ordinary detection, VQA, or unrelated subsets. |
| Qwen3-VL-Seg | Box-guided compact decoder and GT/predicted-box evaluation split | Paper-derived independent P3 candidate | No official implementation was established; third-party code is not an official source. |

## Assets explicitly outside the new task

- Any pre/post change-detection field, recovery advice, disaster comparison, video, or tracking task.
- MMRS classification, ordinary detection, ordinary VQA, infrared-general tasks, and aggregate `total.json`.
- DIOR-RSVG phrases as global detailed captions; they are region-alignment evidence only.
- Old V1/V2 benchmark indexes, old Bridge/Unified indexes, old caches, old checkpoints, and old class/config names as runtime inputs.
- Fixed five-slot modality concatenation, bbox generation by Qwen, SANE/QMEF/PMRD/MGRR compatibility shims, and silent fallbacks.

## P0 conclusion

The reusable core is a small set of data semantics, scientific invariants, engineering patterns,
locally provided raw sources with sufficient technical evidence, and declared upstream
dependencies. No old package path is approved as a greenfield runtime dependency. Raw-data
license approval is not a P0-P7 builder gate; future public release review remains separate. All
legacy-code removal remains closed until the manifest gates and human deletion approvals are
satisfied.
