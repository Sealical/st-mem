"""Read offline perception bundles without importing any memory or model implementation."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from pydantic import Field

from stmem.observations import FiniteModel, ObservationSequence
from stmem.types import DatasetMetadata, FramePacket


class ObservationDatasetParams(FiniteModel):
    root: str = Field(min_length=1)
    manifest: str = "observations.json"


def bundle_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"bundle asset must stay inside its root: {relative}")
    return path


def load_rgb(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        rgb = np.load(path, allow_pickle=False)
    else:
        from PIL import Image

        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB")).copy()
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError(f"RGB image must have uint8 (H, W, 3) shape: {path}")
    return np.asarray(rgb)


class ObservationDataset:
    def __init__(self, root: str, manifest: str = "observations.json"):
        self.root = Path(root).resolve()
        self.manifest_path = bundle_path(self.root, manifest)
        self.sequence = ObservationSequence.model_validate_json(self.manifest_path.read_text())
        self._frames = {f.frame_id: f for f in self.sequence.frames}

    def __len__(self):
        return len(self.sequence.frames)

    def __iter__(self):
        for index in range(len(self)):
            yield self.get_frame(index)

    @property
    def metadata(self):
        return DatasetMetadata(
            name=self.sequence.video_id,
            num_frames=len(self),
            has_depth=any(f.depth_path is not None for f in self.sequence.frames),
            has_raw_poses=any(f.T_world_camera is not None for f in self.sequence.frames),
            camera=self.sequence.frames[0].camera,
        )

    def get_frame(self, index):
        frame = self.sequence.frames[index]
        depth = None
        if frame.depth_path:
            depth = np.load(bundle_path(self.root, frame.depth_path), allow_pickle=False)
            if depth.ndim != 2:
                raise ValueError(f"depth must be (H, W) at frame {frame.frame_id}")
        return FramePacket(
            frame_id=frame.frame_id,
            timestamp_s=frame.timestamp_s,
            rgb=self.image_by_id(frame.frame_id),
            camera=frame.camera,
            depth=depth,
            raw_pose_world_camera=np.asarray(frame.T_world_camera)
            if frame.T_world_camera
            else None,
            extras={"stmem": frame},
        )

    def image_by_id(self, frame_id):
        path = self._frames[frame_id].rgb_path
        return load_rgb(bundle_path(self.root, path)) if path else None

    def fingerprint(self):
        digest = hashlib.sha256(self.manifest_path.read_bytes())
        assets = sorted(
            {
                path
                for frame in self.sequence.frames
                for path in (frame.rgb_path, frame.depth_path)
                if path
            }
        )
        for relative in assets:
            path = bundle_path(self.root, relative)
            digest.update(relative.encode())
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
        return digest.hexdigest()
