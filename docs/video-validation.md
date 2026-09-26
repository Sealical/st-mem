# Real-video validation

## Scope

On 2026-09-26, the public `stmem` implementation reran all four model-backed
stages on one private egocentric clip. This was a fresh perception run, not a
replay of previously computed SAM3 or ViPE outputs.

The clip was sampled to 16 frames at 2 FPS and 480 by 480 pixels, spanning
0 to 7.5 seconds. The object prompt was `laptop`, with a user-provided `office`
scene label. The video, crops, observations, captions, and model files are not
distributed. Only the integration source, setup instructions, and tests are public.

## Model-backed checks

All GPU stages used physical GPU 3, an NVIDIA RTX A6000 with 48 GiB memory,
isolated as `cuda:0` through `CUDA_VISIBLE_DEVICES=3`. This is a tested machine,
not a minimum VRAM claim. The existing local model runtimes and authorized
weights were reused. The ViPE extension was not rebuilt for this check.

| Stage | Result |
| --- | --- |
| Public video sampler | Frame-aligned clip and RGB bundle generated with video SHA-256 provenance |
| SAM3 native video tracking | One persistent laptop ID across all 16 sampled frames |
| ViPE with Metric3D-small | Indexed camera poses, intrinsics, and aligned metric-depth artifacts for all frames |
| DINOv2 ViT-B/14 | 768-dimensional object features generated from tracked RGB crops |
| Qwen3-VL-2B-Instruct | Actual model-generated interval descriptions used in the LTE build |
| Saved five-view memory | 1 object, 1 scene, 3 text entries, 2 events, and 3 visual anchors |
| CPU-only portable replay | NLQ 2, VQ2D 3, STR 2, and LOR 1 results in fresh processes |
| Last-observation query | Returned the actual observation at 7.5 seconds |
| Image-based VQ2D | A retained crop re-encoded with DINOv2 ranked its matching anchor first with cosine approximately 1 |

External source revisions and runtime constraints are in
[environment/](../environment/). Both existing model environments passed
`uv pip check`. The perception environment uses Torch 2.7.1, Transformers 5.16.1,
and NumPy 1.26.4. The separate ViPE environment uses Transformers 4.57.6.
The VLM check used the 2B snapshot, not the 8B model or a hosted API.

## CPU regression coverage

All 55 core, CLI, and adapter-contract tests passed in the model-free Python
3.11 environment. Tests cover import isolation, video-bound caches, explicit
ViPE frame indices, timestamp and target-box propagation into Qwen prompts,
mask/depth fusion, cleanup after tracking errors, and the original core invariants.
Adapter unit tests use controlled fake runtimes. They complement the real GPU
checks above and do not substitute for model inference.

Separate fresh wheel and source-distribution environments each passed the same
55 tests from outside the source checkout. Both ran the synthetic demo and
portable four-query verification without Torch or the original integration
repository. The wheel includes all 12 perception modules but no weights or data.

The public GitHub workflow runs CPU tests, lint, package builds, and a wheel-only
smoke check on Python 3.11 and 3.12. It does not run GPU inference or download weights.

## Limits

This is a bounded desk-scene functional check. It does not establish robustness
to occlusion, cross-clip re-identification, long-duration videos, or dynamic SLAM.
The source is egocentric/fisheye, while the smoke run uses pinhole geometry
without calibration-based rectification. Its world coordinates must not be
presented as calibrated ground truth.

Camera motion and geometry noise can be confused with object motion. VLM
descriptions are not verified annotations. The same-crop VQ2D check validates
encoding and reload consistency, not retrieval generalization. No benchmark
scores, throughput target, full paper reproduction, or clean-machine model
installation guarantee is claimed.
