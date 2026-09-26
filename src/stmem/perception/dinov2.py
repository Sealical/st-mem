"""Lazy, explicit embedding adapters for ST-Mem. No pixel/statistic fallback."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np


class DINOv2Encoder:
    def __init__(
        self,
        checkpoint: str,
        device="cuda:0",
        *,
        repo: str | None = None,
        model_name="dinov2_vitb14",
    ):
        self.checkpoint = Path(checkpoint).resolve()
        self.device = device
        self.repo = repo
        self.model_name = model_name
        self.network: Any = None
        self.transform: Any = None
        if not self.checkpoint.is_file():
            raise FileNotFoundError("DINOv2 expects an official .pth checkpoint")
        digest = hashlib.sha256()
        with self.checkpoint.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        self.feature_space = f"{model_name}:{digest.hexdigest()[:16]}:rgb224-imagenet-v1"

    def load(self):
        import torch
        from torchvision import transforms

        if self.repo is None or not (Path(self.repo) / "hubconf.py").is_file():
            raise FileNotFoundError("Set dinov2 repo to a local clone of facebookresearch/dinov2")
        self.network = torch.hub.load(self.repo, self.model_name, source="local", pretrained=False)
        state = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
        self.network.load_state_dict(state, strict=True)
        self.network.to(self.device).eval()
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def forward(self, inputs: dict) -> dict:
        image = np.asarray(inputs["image"])
        if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8 or not image.size:
            raise ValueError("DINOv2 expects a nonempty uint8 RGB crop")
        if self.network is None:
            self.load()
        import torch

        with torch.inference_mode():
            vector = self.network(self.transform(image).unsqueeze(0).to(self.device))[0]
            vector = torch.nn.functional.normalize(vector.float(), dim=0)
        return {"embedding": vector.cpu().numpy().tolist(), "feature_space": self.feature_space}

    def close(self):
        self.network = None
