"""Read ViPE's pose/intrinsics/depth outputs without importing the ViPE runtime.

ViPE saves OpenCV camera-to-world transforms and metric Z-depth. Match its explicit
frame indices; never align these files by array order or guess missing poses.
"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np


class ViPEArtifacts:
    def __init__(self, root: str, video_name: str):
        self.root = Path(root)
        if Path(video_name).name != video_name:
            raise ValueError("video_name must be a filename stem")
        self.video_name = video_name
        self.poses = self._indexed(self.root / "pose" / f"{video_name}.npz")
        self.intrinsics = self._indexed(self.root / "intrinsics" / f"{video_name}.npz")
        self.depth_path = self.root / "depth" / f"{video_name}.zip"
        if not self.depth_path.is_file():
            raise FileNotFoundError(
                "ViPE metric depth artifacts are missing; pose-only output is insufficient"
            )
        camera_path = self.root / "intrinsics" / f"{video_name}_camera.txt"
        if camera_path.exists():
            for line in camera_path.read_text().splitlines():
                if line.strip() and line.split(":", 1)[-1].strip() != "PINHOLE":
                    raise ValueError(
                        "ST-Mem v0.1 requires pinhole ViPE artifacts; rectify other camera models first"
                    )

    def validate_video(self, video: str | Path):
        """Bind offline geometry to the exact clip, not just a matching resolution."""
        path = self.root / "stmem_vipe.json"
        if not path.is_file():
            raise ValueError("missing ViPE provenance; generate artifacts with stmem video vipe")
        provenance = json.loads(path.read_text())
        with Path(video).open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        if (
            provenance.get("schema_version") != 1
            or provenance.get("video_sha256") != digest
            or provenance.get("video_name") != self.video_name
        ):
            raise ValueError("ViPE artifacts do not match the sampled video")

    @staticmethod
    def _indexed(path):
        with np.load(path, allow_pickle=False) as data:
            indices, values = data["inds"], data["data"]
            if (
                indices.ndim != 1
                or not np.issubdtype(indices.dtype, np.integer)
                or np.any(indices < 0)
                or len(indices) != len(values)
                or len(set(indices.tolist())) != len(indices)
            ):
                raise ValueError(f"invalid ViPE frame index: {path}")
            return {int(index): value.copy() for index, value in zip(indices, values, strict=True)}

    def forward(self, inputs: dict) -> dict:
        import Imath
        import OpenEXR

        frame_id = int(inputs["frame_id"])
        if frame_id not in self.poses or frame_id not in self.intrinsics:
            raise ValueError(f"ViPE pose/intrinsics missing for frame {frame_id}")
        with zipfile.ZipFile(self.depth_path) as archive:
            with archive.open(f"{frame_id:05d}.exr") as entry:
                exr = OpenEXR.InputFile(io.BytesIO(entry.read()))
                try:
                    window = exr.header()["dataWindow"]
                    shape = (window.max.y - window.min.y + 1, window.max.x - window.min.x + 1)
                    depth = (
                        np.frombuffer(
                            exr.channel("Z", Imath.PixelType(Imath.PixelType.FLOAT)),
                            dtype=np.float32,
                        )
                        .reshape(shape)
                        .copy()
                    )
                finally:
                    exr.close()
        pose = self.poses[frame_id]
        intrinsics = self.intrinsics[frame_id]
        if (
            pose.shape != (4, 4)
            or intrinsics.shape != (4,)
            or not np.isfinite(pose).all()
            or not np.isfinite(intrinsics).all()
        ):
            raise ValueError("invalid ViPE pose/intrinsics shape or values")
        if np.any(intrinsics[:2] <= 0) or not np.any(np.isfinite(depth) & (depth > 0)):
            raise ValueError("ViPE returned invalid camera or empty depth")
        return {"depth": depth, "intrinsics": intrinsics, "T_world_camera": pose}
