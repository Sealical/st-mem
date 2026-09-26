# Real-video demo

The optional pipeline integrates SAM3 tracking, ViPE geometry, DINOv2 visual
features, and Qwen3-VL interval descriptions with LTE and the five memory views.
The [CPU demo](demo.md) still runs independently without any model dependencies.

```text
Your video
  -> bounded, frame-aligned clip
  -> SAM3 tracks + ViPE poses, intrinsics, and depth
  -> DINOv2 object features + world positions
  -> Qwen3-VL interval descriptions + LTE
  -> portable memory + NLQ / VQ2D / STR / LOR
```

This is an offline, prompt-guided short-clip integration, not a browser demo or
paper-results reproduction. The source code is public. Model weights, footage,
SMB annotations, and evaluation drivers are not included.

## Environments

Run commands from the repository root. The validated setup is Linux x86_64,
Python 3.11, and an NVIDIA RTX A6000. The model stages require CUDA. A CPU-only
environment can query the saved memory using supplied feature vectors.

Use separate environments for ViPE and the perception models. The pinned ViPE
revision requires Transformers 4, while this Qwen3-VL adapter was tested with
Transformers 5. Do not install both into the same environment. Do not use the
CPU demo's `constraints-demo.txt` for these model environments.

The commands below use `uv`. Constraint files record tested direct dependencies,
not complete transitive locks. Choose fresh environment and checkout directories.
Do not overwrite or reset existing source checkouts.

### External source checkouts

```bash
git clone https://github.com/facebookresearch/sam3.git references/sam3
git -C references/sam3 checkout --detach 660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7
git clone https://github.com/facebookresearch/dinov2.git references/dinov2
git -C references/dinov2 checkout --detach 7764ea0f912e53c92e82eb78a2a1631e92725fc8
git clone https://github.com/nv-tlabs/vipe.git references/vipe
git -C references/vipe checkout --detach 95a8816947602ddc26fcb7a80bea4f9313059578
```

These are ignored local checkouts, not vendored source in the ST-Mem package.
Review [third-party terms](../THIRD_PARTY_NOTICES.md) before installing models.

### SAM3, DINOv2, and Qwen3-VL

```bash
uv venv .venv-models --python 3.11
uv pip install --python .venv-models/bin/python \
  torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
uv pip install --python .venv-models/bin/python \
  -c environment/perception-constraints.txt '.[perception]' -e references/sam3
uv pip check --python .venv-models/bin/python
```

The SAM3 integration uses the native video predictor with the OpenCV loader.
It does not require the image-only Transformers SAM adapter, decord, Flash
Attention, or xFormers. The tested setuptools pin preserves SAM3's use of
`pkg_resources`.

Prepare authorized local model files before running inference:

| Component | Example local path | Official source |
| --- | --- | --- |
| SAM3 native video checkpoint | `checkpoints/sam3/sam3.pt` | [facebook/sam3](https://huggingface.co/facebook/sam3) |
| DINOv2 ViT-B/14 | `checkpoints/dinov2/dinov2_vitb14_pretrain.pth` | [DINOv2 models](https://github.com/facebookresearch/dinov2#pretrained-models) |
| Qwen3-VL complete snapshot | `checkpoints/Qwen3-VL-2B-Instruct/` | [Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) |

SAM3 requires the official native `.pt` file. Qwen requires the tokenizer and
processor files as well as the weights. Its adapter uses `local_files_only=True`.
Do not put access tokens into scripts or configuration files.

### ViPE

ViPE needs a working C++ compiler, CUDA toolkit with `nvcc`, and video decoding
support. Compile its extensions for your own GPU architecture.

```bash
uv venv .venv-vipe --python 3.11
uv pip install --python .venv-vipe/bin/python \
  torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
uv pip install --python .venv-vipe/bin/python 'setuptools>=77' wheel ninja
CUDA_VISIBLE_DEVICES=0 MAX_JOBS=4 uv pip install \
  --python .venv-vipe/bin/python --no-build-isolation \
  -c environment/vipe-constraints.txt -e references/vipe
uv pip install --python .venv-vipe/bin/python -c environment/vipe-constraints.txt .
uv pip check --python .venv-vipe/bin/python
```

Set `CUDA_HOME` to your installed toolkit if automatic discovery fails. The
validation reused an extension built with CUDA toolkit 12.4 and cu126 Torch.
That is one tested machine, not a guarantee for every driver or compiler.

The runner selects Metric3D-small, saves depth artifacts, and disables ViPE's
separate instance-mask prior. Upstream ViPE may download DROID-SLAM, GeoCalib,
and Metric3D weights on first use. For an offline run, populate its authorized
`TORCH_HOME` cache beforehand:

```text
<TORCH_HOME>/hub/droid_slam/droid.pth
<TORCH_HOME>/hub/geocalib/pinhole.tar
<TORCH_HOME>/hub/checkpoints/metric_depth_vit_small_800k.pth
```

## Run a bounded video

The example selects physical GPU 0 with `CUDA_VISIBLE_DEVICES=0`. Change that
value to the GPU you intend to use. Inside the isolated process it is `cuda:0`.
Choose a new output directory for each stage because existing outputs are not
overwritten. Replace the source video, prompt, and scene label with your own.

### 1. Sample the video

```bash
.venv-models/bin/stmem video sample \
  --video /path/to/your/video.mp4 --output outputs/video-input \
  --fps 2 --max-frames 16 --max-size 480
```

The sampler supports constant-frame-rate source timing. It records actual source
frame timestamps and decodes the newly encoded clip to produce aligned RGB
frames. Variable-frame-rate source timing is not supported.

### 2. Estimate geometry

```bash
CUDA_VISIBLE_DEVICES=0 .venv-vipe/bin/stmem video vipe \
  --video outputs/video-input/sampled.mp4 --output outputs/video-geometry
```

### 3. Track prompted objects

```bash
CUDA_VISIBLE_DEVICES=0 .venv-models/bin/stmem video track \
  --video outputs/video-input/sampled.mp4 --output outputs/video-tracks \
  --checkpoint checkpoints/sam3/sam3.pt --prompts laptop --device cuda:0
```

### 4. Create the observation bundle

```bash
CUDA_VISIBLE_DEVICES=0 .venv-models/bin/stmem video prepare \
  --video-bundle outputs/video-input --vipe-root outputs/video-geometry \
  --tracking-cache outputs/video-tracks/tracking.json \
  --output outputs/video-observations --prompts laptop --scene-label office \
  --dinov2-checkpoint checkpoints/dinov2/dinov2_vitb14_pretrain.pth \
  --dinov2-repo references/dinov2 --device cuda:0
```

Alternatively, omit `--tracking-cache` and supply `--sam3-checkpoint` to run
tracking inside preparation. Separate tracking lets you reuse expensive SAM3
outputs when retrying later stages.

SAM3, ViPE, and the frame bundle must correspond to the same sampled clip.
SHA-256 provenance rejects mismatched video artifacts or tracking prompts.
Frame IDs, mask/depth dimensions, and geometry are checked before fusion.
Missing geometry is an error, not an invented object position.

### 5. Build memory with Qwen3-VL descriptions

```bash
CUDA_VISIBLE_DEVICES=0 .venv-models/bin/stmem build \
  --observations outputs/video-observations/observations.json \
  --output outputs/video-memory --captions vlm \
  --vlm-checkpoint checkpoints/Qwen3-VL-2B-Instruct --device cuda:0
```

Qwen sees selected RGB frames, target pixel boxes, and actual timestamps for
each observed LTE interval. `provided` still requires supplied interval captions.
`template` is an explicit debugging mode, not a fallback for failed model inference.

### 6. Query and verify

```bash
.venv-models/bin/stmem inspect --memory outputs/video-memory/memory.json
.venv-models/bin/stmem query --memory outputs/video-memory/memory.json \
  --type nlq --text laptop
.venv-models/bin/stmem query --memory outputs/video-memory/memory.json \
  --type lor --label laptop
.venv-models/bin/stmem verify --memory outputs/video-memory/memory.json \
  --object-id 'prompt_0/track_0' --text laptop

CUDA_VISIBLE_DEVICES=0 .venv-models/bin/stmem query \
  --memory outputs/video-memory/memory.json --type vq2d \
  --image /path/to/your/object_crop.png \
  --dinov2-checkpoint checkpoints/dinov2/dinov2_vitb14_pretrain.pth \
  --dinov2-repo references/dinov2 --device cuda:0
```

Use an object ID returned by the LOR query, not an assumed ID from this example.
The verifier runs all four query types in fresh processes using a retained visual
anchor as an identity check. It tests functional consistency, not retrieval accuracy.
Without DINOv2, VQ2D can use the CPU `--feature-file` path described in the core guide.
Copy the entire memory directory, including `crops/`, when moving the result.

## Boundaries

- Prompts and the scene label are user supplied. Categories and rooms are not
  automatically discovered. Independent prompts are namespaced and not deduplicated.
- ViPE input must use a pinhole camera model. Rectify calibrated fisheye input
  before making geometric accuracy claims. Monocular geometry is an estimate.
- This is not long-video streaming, cross-clip re-identification, or benchmark
  evaluation. Upstream tracking errors, geometry noise, and VLM hallucinations
  can propagate into memory.
- A generated memory contains real image crops and descriptions. Treat it as
  derived private data until you have permission to share it.

See the [validation record](video-validation.md) and
[dependency notices](../THIRD_PARTY_NOTICES.md).
