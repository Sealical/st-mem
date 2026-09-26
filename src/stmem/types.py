"""Minimal camera and frame contracts for the standalone offline demo."""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class ContractModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")


class CameraModel(ContractModel):
    schema_version: int = 1
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    distortion_model: str = "none"
    distortion_params: tuple[float, ...] = ()
    frame_id: str = "camera"

    def K(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )


class FramePacket(ContractModel):
    """Observed RGB, metric depth and camera-to-world pose at one timestamp."""

    frame_id: int
    timestamp_s: float
    rgb: np.ndarray | None = None
    camera: CameraModel | None = None
    depth: np.ndarray | None = None
    raw_pose_world_camera: np.ndarray | None = None
    extras: dict[str, Any] = Field(default_factory=dict)


class DatasetMetadata(ContractModel):
    name: str
    num_frames: int | None = None
    has_depth: bool = False
    has_raw_poses: bool = False
    camera: CameraModel | None = None
