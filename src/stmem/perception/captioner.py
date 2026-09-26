"""Adapt local Qwen3-VL inference to timestamped LTE interval captions."""

from stmem.perception.qwen import Qwen3VLModel


class MotionCaptioner:
    def __init__(self, checkpoint: str, device="cuda:0", max_new_tokens=100):
        self.model = Qwen3VLModel(checkpoint, device, max_new_tokens=max_new_tokens)

    def __call__(self, label, kind, samples, image_getter):
        from PIL import Image

        if image_getter is None:
            raise ValueError("VLM captions require source images")
        if not samples:
            raise ValueError("VLM captions require observed samples")
        selected = sorted({0, len(samples) // 2, len(samples) - 1})
        content = [
            {
                "type": "text",
                "text": (
                    f"These frames show the same tracked {label} over time. Geometry classifies this interval as {kind}. "
                    "Describe only its visible motion, state and location in one concise English sentence. "
                    "Use the timestamps, do not invent actions or unseen causes. The target box is supplied in pixel xyxy coordinates."
                ),
            }
        ]
        for index in selected:
            sample = samples[index]
            rgb = image_getter(sample.frame_id)
            if rgb is None or sample.observation is None:
                raise ValueError("missing observed frame for VLM caption")
            content.extend(
                [
                    {
                        "type": "text",
                        "text": f"t={sample.timestamp_s:g}s, target bbox={sample.observation.bbox_xyxy}",
                    },
                    {"type": "image", "image": Image.fromarray(rgb)},
                ]
            )
        return self.model.forward({"messages": [{"role": "user", "content": content}]})["text"]

    def close(self):
        self.model.close()
