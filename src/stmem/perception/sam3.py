"""Official SAM3 video propagation with persistent, prompt-namespaced object IDs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class Sam3VideoTracker:
    def __init__(self, checkpoint: str, device="cuda:0", *, threshold=0.5):
        self.checkpoint = checkpoint
        self.device = device
        self.threshold = threshold
        self.predictor: Any = None

    def load(self):
        if not Path(self.checkpoint).is_file():
            raise FileNotFoundError(self.checkpoint)
        import torch
        from sam3.model_builder import build_sam3_video_predictor

        device = torch.device(self.device)
        if device.type != "cuda":
            raise ValueError("SAM3 video inference requires a CUDA device")
        index = device.index if device.index is not None else 0
        torch.cuda.set_device(index)
        self.predictor = build_sam3_video_predictor(
            checkpoint_path=self.checkpoint,
            gpus_to_use=[index],
            compile=False,
            video_loader_type="cv2",
        )

    def forward(self, inputs: dict) -> dict:
        video = Path(inputs["video"])
        prompts = inputs["prompts"]
        if not video.exists() or not prompts or not all(isinstance(p, str) and p for p in prompts):
            raise ValueError("provide an existing video and nonempty text prompts")
        if len(set(prompts)) != len(prompts):
            raise ValueError("SAM3 prompts must be unique")
        if self.predictor is None:
            self.load()
        outputs: dict[int, list[dict]] = {}
        for prompt_index, prompt in enumerate(prompts):
            response = self.predictor.handle_request(
                {"type": "start_session", "resource_path": str(video), "offload_video_to_cpu": True}
            )
            session_id = response["session_id"]
            try:
                self.predictor.handle_request(
                    {
                        "type": "add_prompt",
                        "session_id": session_id,
                        "frame_index": 0,
                        "text": prompt,
                    }
                )
                for response in self.predictor.handle_stream_request(
                    {
                        "type": "propagate_in_video",
                        "session_id": session_id,
                        "propagation_direction": "forward",
                        "start_frame_index": 0,
                    }
                ):
                    frame_index = int(response["frame_index"])
                    raw = response["outputs"]
                    ids = raw["out_obj_ids"]
                    masks = raw["out_binary_masks"]
                    scores = raw.get("out_probs", np.ones(len(ids)))
                    for object_id, mask, confidence in zip(ids, masks, scores, strict=True):
                        if hasattr(mask, "detach"):
                            mask = mask.detach().cpu().numpy()
                        mask = np.asarray(mask).squeeze().astype(bool)
                        if mask.ndim != 2:
                            raise ValueError("SAM3 video returned a non-2D object mask")
                        if float(confidence) < self.threshold or not mask.any():
                            continue
                        ys, xs = np.nonzero(mask)
                        outputs.setdefault(frame_index, []).append(
                            {
                                "object_id": f"prompt_{prompt_index}/track_{int(object_id)}",
                                "label": prompt,
                                "bbox_xyxy": [
                                    int(xs.min()),
                                    int(ys.min()),
                                    int(xs.max()) + 1,
                                    int(ys.max()) + 1,
                                ],
                                "confidence": float(confidence),
                                "mask": mask,
                            }
                        )
            finally:
                self.predictor.handle_request({"type": "close_session", "session_id": session_id})
        if not outputs:
            raise ValueError("SAM3 returned no objects; check prompts/video/checkpoint")
        return {"frames": outputs}

    def close(self):
        if self.predictor is not None:
            self.predictor.shutdown()
            self.predictor = None
