# Spatial Memory Benchmark (SMB)

**Benchmarking long-horizon spatiotemporal memory for embodied agents.**

SMB is introduced in [Linguistic Trajectory Encoding for Efficient Long-Horizon
Spatial Memory in Embodied Agents](https://arxiv.org/abs/2609.04802),
**accepted at NeurIPS 2026**. It tests
whether an embodied memory system can retrieve objects by their state history
and recover their last occurrence over hours-long observations.

[Benchmark page](https://sealical.github.io/st-mem/benchmark/) ·
[LTE / ST-Mem project](https://sealical.github.io/st-mem/) ·
[Paper](https://arxiv.org/abs/2609.04802v1) · [BibTeX](../citation.bib)

**Description only.** Task definitions and paper-reported results are public.
SMB annotations and evaluation code are **not released here**. Source video,
model weights, and an executable benchmark are not included, and there is no
announced dataset-release date.

## Data source and task taxonomy

SMB is constructed from [EgoLife](https://egolife-ai.github.io/) multi-day
egocentric recordings. The paper describes EgoLife's 300 hours across six
participants over seven days. This is the source collection, not a claim that
every SMB query spans the entire collection. Individual sessions reach 50 hours.
SMB's LOR lookback windows are 2, 6, 12, and 24 hours.

| Task                                | Queries                           | What is retrieved?                                                                               | Query constraints                                                                         |
| ----------------------------------- | --------------------------------- | ------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------- |
| Semantic Trajectory Retrieval (STR) | 300, over 232 object instances    | Objects matching a described motion or state history, with supporting frames and bounding boxes. | Object state description with optional semantic spatial regions and temporal constraints. |
| Long-Horizon Object Retrieval (LOR) | 300, with 75 per lookback horizon | The last occurrence of a described object within the requested lookback window.                  | Object description, lookback window, and optional spatial constraints.                    |

STR examples involve one or two objects, such as a book being put into a
backpack. LOR is about the latest observed occurrence, not a claim to know an
object's current position when it is off camera. This description summarizes
the published task definitions. It does not define a new annotation-file schema.

## Evaluation protocol reported in the paper

- Both tasks report **success rate**: a retrieved frame must be within the
  object's ground-truth temporal span, with bounding-box IoU ≥ 0.3.
- STR annotations contain verified object instances, valid temporal spans,
  and representative-frame boxes. Ground truth is constructed with tracking
  assistance and manual verification. Instances with failed tracking are skipped.
- LOR annotation checks observations in reverse chronological order to identify
  the last occurrence and its box inside each lookback window.
- The paper also reports IoU ≥ 0.5 results and per-horizon analysis. Consult
  the paper for those settings. This repository does not offer a scoring implementation.

The tracking-assisted annotation process and skipped instances matter when
interpreting the benchmark. See §4.1 and Appendix A of the
[paper](https://arxiv.org/abs/2609.04802v1) for construction and annotation details.

## SMB is not Ego4D NLQ or VQ2D

The same paper also evaluates two **established Ego4D tasks**, but they are not
SMB tasks: Natural Language Queries (NLQ) localizes an interval described in
language, while Visual Queries 2D (VQ2D) uses a visual crop to retrieve an object's
recent occurrence. SMB contributes STR and LOR over EgoLife recordings.

This makes SMB relevant to long-term embodied agent memory, object-centric
spatiotemporal memory, object state history, and natural-language video retrieval.
It is not a benchmark of general lifelong learning or robot navigation.

## Results reported in the paper

Success rate (%), 300 queries per task, bounding-box IoU ≥ 0.3 within the
ground-truth temporal span:

| Method                         |  STR |  LOR |
| ------------------------------ | ---: | ---: |
| Qwen3-VL-8B + Grounding-DINO   | 21.5 | 25.1 |
| Qwen3-VL-235B + Grounding-DINO | 31.9 | 34.4 |
| KFMem (3D-Mem-style)           | 19.8 | 33.8 |
| VideoAgent                     | 24.7 | 30.5 |
| LTE-based system (ours)        | 45.3 | 48.7 |

Source: §4 of [arXiv:2609.04802v1](https://arxiv.org/abs/2609.04802v1).
These are reported research results, **not an independently reproduced
leaderboard** or measurements from this website.

## Availability, source attribution, and citation

| Resource                                     | Status                                                                        |
| -------------------------------------------- | ----------------------------------------------------------------------------- |
| Paper and benchmark description              | Public.                                                                       |
| ST-Mem memory-core implementation            | Code coming soon. Separate from benchmark evaluation code.                    |
| SMB query annotations and evaluation scripts | Not released here.                                                            |
| EgoLife source recordings                    | Not redistributed. Consult the original EgoLife project for access and terms. |

EgoLife is the work of Jingkang Yang and collaborators, _EgoLife: Towards
Egocentric Life Assistant_, CVPR 2025. Ego4D is the work of Kristen Grauman
and collaborators, _Ego4D: Around the World in 3,000 Hours of Egocentric Video_,
CVPR 2022. This website's software license does not grant access or redistribution
rights to either source dataset or to unpublished SMB annotations.

For SMB, cite the LTE paper using [the shared BibTeX](../citation.bib) or
[preferred citation metadata](../CITATION.cff). No separate SMB DOI is asserted.
