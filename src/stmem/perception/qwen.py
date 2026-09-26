"""Qwen3-VL atom: I/O adapter over HuggingFace transformers (local checkpoint)."""

from __future__ import annotations

from typing import Any

_OPS = frozenset({"generate"})


def _validate_inputs(inputs: dict[str, Any]) -> str:
    op = inputs.get("op", "generate")
    if not isinstance(op, str) or op not in _OPS:
        raise KeyError(f"unknown vlm op: {op}")
    if inputs.get("messages") is None and inputs.get("image") is None:
        raise KeyError("Qwen3-VL generate requires 'messages' or 'image'")
    return op


def _to_pil(image: Any):
    import numpy as np
    from PIL import Image

    if isinstance(image, Image.Image):
        return image.convert("RGB")
    array = np.asarray(image)
    return Image.fromarray(array).convert("RGB")


def _messages_from_inputs(inputs: dict[str, Any]) -> list[dict[str, Any]]:
    if inputs.get("messages") is not None:
        return list(inputs["messages"])
    image = _to_pil(inputs["image"])
    prompt = inputs.get("prompt", "Describe this image.")
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]


class Qwen3VLModel:
    """Conventional keys. In: ``op`` + ``messages`` or ``image``/``prompt``; out: ``text``.

    Ops: ``generate`` (default).
    """

    def __init__(self, checkpoint: str, device: str, *, max_new_tokens: int = 128) -> None:
        self.checkpoint = checkpoint
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.network = None
        self.processor = None

    def load(self):
        from pathlib import Path

        if not Path(self.checkpoint).is_dir():
            raise FileNotFoundError("Qwen3-VL requires a complete local model snapshot")
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(self.checkpoint, local_files_only=True)
        self.network = Qwen3VLForConditionalGeneration.from_pretrained(
            self.checkpoint,
            dtype="auto",
            local_files_only=True,
        )
        self.network.to(self.device)
        self.network.eval()
        return self.network

    def _ensure(self):
        if self.network is None or self.processor is None:
            self.load()
        return self.network, self.processor

    def forward(self, inputs: dict[str, Any]) -> dict[str, Any]:
        _validate_inputs(inputs)

        messages = _messages_from_inputs(inputs)
        model, processor = self._ensure()
        batch = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        if hasattr(batch, "to"):
            batch = batch.to(self.device)
        else:
            batch = {
                key: value.to(self.device) if hasattr(value, "to") else value
                for key, value in batch.items()
            }
        max_new_tokens = int(inputs.get("max_new_tokens", self.max_new_tokens))
        import torch

        with torch.inference_mode():
            generated = model.generate(**batch, max_new_tokens=max_new_tokens)
        trimmed = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(batch["input_ids"], generated, strict=True)
        ]
        decoded = processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return {"text": decoded[0] if len(decoded) == 1 else decoded}

    def close(self):
        self.network = None
        self.processor = None
