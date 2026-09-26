# Third-party dependencies

The original ST-Mem code and its inference adapters use the root [MIT license](LICENSE).
This does not relicense external implementations, model weights, datasets, or
paper figures. No upstream model source, weights, private footage, generated
real-video crops, or benchmark annotations are distributed in this repository.

The optional video path calls separately installed upstream software. Source
revisions are recorded in [environment/sources.json](environment/sources.json).
Use only source checkouts and checkpoints you are authorized to access.

- **SAM3** uses Meta's native video predictor. Its code and weights are subject
  to the [SAM License](https://github.com/facebookresearch/sam3/blob/660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7/LICENSE),
  not this repository's MIT license. Obtain the native checkpoint through the
  [official model page](https://huggingface.co/facebook/sam3) and follow its access terms.
- **ViPE** is installed from the [official source](https://github.com/nv-tlabs/vipe/tree/95a8816947602ddc26fcb7a80bea4f9313059578).
  Its [license](https://github.com/nv-tlabs/vipe/blob/95a8816947602ddc26fcb7a80bea4f9313059578/LICENSE)
  and [third-party notices](https://github.com/nv-tlabs/vipe/blob/95a8816947602ddc26fcb7a80bea4f9313059578/THIRD_PARTY_LICENSES.md)
  describe separate component terms. This integration selects Metric3D-small,
  not Unik3D. DROID-SLAM, GeoCalib, and Metric3D remain external dependencies.
- **DINOv2 ViT-B/14** uses the standard
  [DINOv2 source and pretrained model](https://github.com/facebookresearch/dinov2/tree/7764ea0f912e53c92e82eb78a2a1631e92725fc8)
  under its [Apache-2.0 license](https://github.com/facebookresearch/dinov2/blob/7764ea0f912e53c92e82eb78a2a1631e92725fc8/LICENSE).
  Different model families in that repository can have different terms.
- **Qwen3-VL** loads a complete local Hugging Face snapshot. Follow the
  [Qwen3-VL-2B-Instruct model card and license](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct).
  The validation uses this 2B model, not the 8B model or a hosted API.

Original teaser and framework figures retain their separate
[CC BY 4.0 attribution](assets/README.md). For your own videos, check rights and
privacy before sharing the source or generated crops and descriptions.

Dependency pointers and install recipes are not permission to redistribute
upstream models or data. Keep local model caches and credentials out of commits.
