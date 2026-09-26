# ST-Mem core demo

A standalone command-line demo of Linguistic Trajectory Encoding (LTE), five
linked memory views, and four query APIs. Python 3.11 or newer is required.
No GPU, Torch, model weights, external checkout, or API key is needed.

For SAM3 tracking, ViPE geometry, DINOv2 features, and Qwen3-VL descriptions on
your own video, use the separate [real-video guide](video-demo.md).

## Install and run

```bash
git clone https://github.com/Sealical/st-mem.git
cd st-mem
python -m venv .venv
source .venv/bin/activate
python -m pip install -c constraints-demo.txt .
stmem demo --output outputs/demo
stmem verify --memory outputs/demo/memory/memory.json
```

On Windows, activate with `.venv\Scripts\activate` instead. You can also use
`python -m stmem` in place of `stmem`.
The first installation downloads Python dependencies. After installation, the
demo and queries run offline. The package is installed from this repository,
not from a separately published PyPI release.

Choose a new output directory on each run. Existing output directories are
refused, even if empty, so earlier results cannot be overwritten.

The demo creates nine synthetic observations of a red cup and a blue book. It
includes motion, a stop, an observation gap, and two scenes. Positions, captions,
RGB arrays, and identity vectors are generated fixtures, not model predictions.
The memory construction and queries execute the actual core implementation.

## Outputs

```text
outputs/demo/
  observations/       Synthetic RGB arrays and observations.json
  memory/
    memory.json       Versioned LTE records and five linked views
    crops/            Retained visual evidence as non-pickled uint8 NPY files
  query_feature.json  Synthetic visual query vector and feature-space identifier
  queries.json        Results from NLQ, VQ2D, STR, and LOR
```

Expected view counts are 2 objects, 2 scenes, 8 text entries, 5 events, and
7 image anchors. The four demo queries return 5, 4, 4, and 1 results respectively.
The last observed cup occurrence is at 10 seconds in the office.

`stmem verify` copies only the memory directory to a temporary location and
executes the four query commands in fresh processes. Its NLQ query is filtered
to the cup, so its query counts are 4, 4, 4, and 1. This is a functional check,
not an accuracy evaluation.

## Query the saved memory

```bash
stmem inspect --memory outputs/demo/memory/memory.json

stmem query --memory outputs/demo/memory/memory.json \
  --type nlq --text 'red cup moved sink kitchen' --object-id cup_1

stmem query --memory outputs/demo/memory/memory.json \
  --type vq2d --feature-file outputs/demo/query_feature.json

stmem query --memory outputs/demo/memory/memory.json \
  --type str --object-id cup_1 --scene kitchen --start 0 --end 2

stmem query --memory outputs/demo/memory/memory.json \
  --type lor --label cup --as-of 7 --lookback 3
```

The last command returns the cup's actual observation at 5 seconds. The gap at
6 to 7 seconds does not create a new observation. Copy the whole `memory/`
directory when moving a memory, including `crops/`. Original observations and
model files are not needed to query the saved memory.

## Build from your own observations

```bash
stmem build --observations outputs/demo/observations/observations.json \
  --output outputs/rebuilt-memory
```

Use the generated `observations.json` as a small example of the input contract.
The data models are defined in `src/stmem/observations.py`.

- Frame IDs and timestamps must increase strictly. Timestamps are in seconds.
- Each object needs a stable ID, label, positive-area pixel box, and visual
  feature vector from the declared feature space.
- Supply world positions in meters, or aligned depth, camera intrinsics, and a
  right-handed camera-to-world transform (`T_world_camera`). Missing geometry
  raises an error.
- Supply interval captions for observed object histories. `--captions template`
  is an explicit debugging option, not language-model inference.
- RGB and depth paths must stay inside the observation bundle. NPY inputs use
  `allow_pickle=False`. Other RGB formats are read with Pillow.

The Python API exposes `STMemBuilder`, `STMemParams`, `MemoryViews`,
`QueryEngine`, `load_memory`, and `save_memory` from `stmem`.
`stmem.pipeline.build_memory` accepts `STMemParams` for configuring LTE thresholds.

## Scope

This CPU example exercises the offline memory core with synthetic observations.
The repository also provides optional [video perception integrations](video-demo.md).
Neither path includes a browser UI, benchmark annotations, evaluation drivers,
training, audio, or navigation. The full research framework figure contains
components outside these demos.

NLQ uses BM25 over stored text. Spatial and temporal filters are explicit, not
automatically parsed from a question. VQ2D compares supplied features in the same
feature space. STR returns sparse retained trajectory evidence. LOR returns the
last actual retained observation, not an inferred current location. Paper results
are separate from the demo's functional checks.

Do not publish memory crops or captions derived from private data. The bundled
generator creates synthetic fixtures only. Original software is under the root
[MIT License](../LICENSE). Paper figures retain their [separate attribution](../assets/README.md).

## Development checks

```bash
python -m pip install -c constraints-demo.txt '.[dev]'
python -m pytest
python -m ruff check src tests
python -m build
```

See [validation coverage](demo-validation.md) for the release checks.
