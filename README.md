# Linguistic Trajectory Encoding (LTE) — ST-Mem

Official project page for **Linguistic Trajectory Encoding for Efficient
Long-Horizon Spatial Memory in Embodied Agents**.

LTE is an **object-centric spatiotemporal memory representation** for
long-horizon embodied agents. It compresses dynamic object motion histories
into natural-language descriptions, sparse spatial anchors, and visual anchors.
The LTE-based ST-Mem system links object state history to spatial and visual
evidence for semantic trajectory retrieval and long-horizon object retrieval.

[Paper](https://arxiv.org/abs/2609.04802) ·
[Project page](https://sealical.github.io/st-mem/) ·
[Spatial Memory Benchmark](https://sealical.github.io/st-mem/benchmark/) ·
[BibTeX](citation.bib) · [Citation metadata](CITATION.cff)

- **Paper:** arXiv:2609.04802 · arXiv preprint · 2026.
- **Benchmark:** Spatial Memory Benchmark (SMB), constructed from EgoLife recordings.
- **Evaluation:** SMB (STR, LOR) and Ego4D (NLQ, VQ2D).
- **Code coming soon.** This repository contains the website and benchmark
  description, not the ST-Mem implementation, annotations, model weights, or
  evaluation pipelines. No benchmark download is currently offered here.

## What problem does LTE address?

Long-term embodied agent memory needs to retain what happened to an object,
where it happened, and when. Clip-level video memory and raw geometric
trajectories expose different parts of this evidence. LTE connects them in a
language-queryable per-object history, supporting natural-language spatial
retrieval over hours-to-days observations without scanning every clip at query
time. The research focuses on spatiotemporal memory, object state history,
and trajectory compression; it does not claim a general lifelong-learning agent.

## Method, system, and benchmark

| Name                                 | Role                                                                                                             |
| ------------------------------------ | ---------------------------------------------------------------------------------------------------------------- |
| Linguistic Trajectory Encoding (LTE) | The hybrid trajectory representation: language, sparse spatial anchors, and visual anchors.                      |
| ST-Mem                               | This research project and its LTE-based memory system, with five linked views and spatial indexing.              |
| Spatial Memory Benchmark (SMB)       | The paper's long-horizon benchmark: Semantic Trajectory Retrieval (STR) and Long-Horizon Object Retrieval (LOR). |

Read the [method and original framework figure](https://sealical.github.io/st-mem/#method).
The [task illustrations](https://sealical.github.io/st-mem/#queries) are explanatory
examples, **not live inference**.

## Spatial Memory Benchmark

SMB contains **600 queries** constructed from EgoLife multi-day recordings:
300 Semantic Trajectory Retrieval queries and 300 Long-Horizon Object Retrieval
queries. LOR uses lookback windows from 2 to 24 hours. SMB is distinct from the
established Ego4D NLQ and VQ2D tasks.

See the [benchmark page](https://sealical.github.io/st-mem/benchmark/) or
[benchmark README](benchmark/README.md) for task definitions, the paper-reported
evaluation protocol, results, source attribution, and release status.

## Results reported in the paper

| Measurement                 | Reported result | Scope                                                                                          |
| --------------------------- | --------------- | ---------------------------------------------------------------------------------------------- |
| SMB STR success             | 45.3%           | 300 semantic trajectory queries; bounding-box IoU ≥ 0.3 within the ground-truth temporal span. |
| SMB LOR success             | 48.7%           | 300 long-horizon object queries; the same success criterion.                                   |
| LTE trajectory compression  | 8.7–26.1×       | LTE trajectory storage relative to dense trajectories, not total system storage.               |
| Query latency on 24 h video | 0.43 s          | Single A800, after memory construction; not end-to-end video processing.                       |

These are [paper-reported results](https://arxiv.org/abs/2609.04802v1), not new
measurements or an independent reproduction by this repository.

## Code availability

**Code coming soon.** The planned release focuses on the memory core: LTE,
five linked views, portable memory storage, query APIs, and offline perception
adapters. Full benchmark reproduction, evaluation/training pipelines, audio,
and navigation are outside that planned release. Public installation instructions
will be added only when the implementation is available.

## Citation

Please cite the paper rather than the website:

```bibtex
@misc{xie2026linguistictrajectory,
  title = {Linguistic Trajectory Encoding for Efficient Long-Horizon Spatial Memory in Embodied Agents},
  author = {Tianyidan Xie and Shenyi Wang and Qiang Tang and Mingjie Wang and Zhicheng Qiu and Xuanfu Li and Zhan Xu and Jian Yang and Lanjun Wang and Zili Yi},
  year = {2026},
  eprint = {2609.04802},
  archivePrefix = {arXiv},
  primaryClass = {cs.CV},
  doi = {10.48550/arXiv.2609.04802},
  url = {https://arxiv.org/abs/2609.04802}
}
```

## License and figure attribution

Original website code is available under the [MIT License](LICENSE).
Figures 1 and 2 are unmodified author-supplied images from the paper, under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), not MIT. Preserve the
[figure attribution](assets/README.md) when redistributing them.

For website maintainers: [maintenance documentation](docs/website-maintenance.md).
