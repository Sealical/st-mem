"""Portable ST-Mem observations. RGB, seconds, meters and T_world_camera throughout.

This is the boundary between offline perception and memory, not a model runtime.
Missing objects mean unobserved; missing geometry never becomes a fabricated position.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from stmem.types import CameraModel


class FiniteModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class ObjectObservation(FiniteModel):
    object_id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    bbox_xyxy: tuple[float, float, float, float]
    position_world: tuple[float, float, float] | None = None
    visual_feature: list[float] | None = None
    confidence: float = Field(default=1.0, ge=0, le=1)

    @field_validator("bbox_xyxy")
    @classmethod
    def valid_box(cls, box):
        x1, y1, x2, y2 = box
        if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
            raise ValueError("bbox_xyxy must be a positive-area pixel box with nonnegative origin")
        return box

    @field_validator("visual_feature")
    @classmethod
    def valid_feature(cls, feature):
        if feature is not None and (not feature or np.linalg.norm(feature) <= 1e-12):
            raise ValueError("visual_feature must be a nonzero vector")
        return feature


class ObservationFrame(FiniteModel):
    frame_id: int = Field(ge=0)
    timestamp_s: float = Field(ge=0)
    rgb_path: str | None = None
    depth_path: str | None = None
    camera: CameraModel | None = None
    T_world_camera: list[list[float]] | None = None
    scene_id: str = Field(default="scene_0", min_length=1)
    scene_label: str = Field(default="unknown", min_length=1)
    objects: list[ObjectObservation] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_frame(self):
        ids = [obj.object_id for obj in self.objects]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate object ID in frame {self.frame_id}")
        if self.T_world_camera is not None:
            pose = np.asarray(self.T_world_camera)
            if pose.shape != (4, 4) or not np.isfinite(pose).all():
                raise ValueError("T_world_camera must be finite 4x4")
            if not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-6):
                raise ValueError("T_world_camera must have homogeneous last row")
            rotation = pose[:3, :3]
            if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3) or not np.isclose(
                np.linalg.det(rotation), 1, atol=1e-3
            ):
                raise ValueError("T_world_camera rotation must be right-handed and orthonormal")
        return self


class CaptionObservation(FiniteModel):
    object_id: str
    t_start: float = Field(ge=0)
    t_end: float = Field(ge=0)
    text: str = Field(min_length=1)

    @model_validator(mode="after")
    def valid_interval(self):
        if self.t_end < self.t_start:
            raise ValueError("caption ends before it starts")
        return self


class ObservationSequence(FiniteModel):
    schema_version: Literal[1] = 1
    video_id: str = Field(min_length=1)
    units: Literal["meters"] = "meters"
    pose_convention: Literal["T_world_camera"] = "T_world_camera"
    feature_space: str | None = None
    provenance: dict[str, str] = Field(default_factory=dict)
    frames: list[ObservationFrame] = Field(min_length=1)
    captions: list[CaptionObservation] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_sequence(self):
        previous = None
        labels: dict[str, str] = {}
        dimensions = set()
        for frame in self.frames:
            if previous and (
                frame.frame_id <= previous.frame_id or frame.timestamp_s <= previous.timestamp_s
            ):
                raise ValueError("frame IDs and timestamps must be strictly increasing")
            previous = frame
            for obj in frame.objects:
                if obj.object_id in labels and labels[obj.object_id] != obj.label:
                    raise ValueError(f"label changed for object {obj.object_id}")
                labels[obj.object_id] = obj.label
                if obj.visual_feature is not None:
                    dimensions.add(len(obj.visual_feature))
        if len(dimensions) > 1 or (dimensions and not self.feature_space):
            raise ValueError("visual features require one declared feature_space and dimension")
        for caption in self.captions:
            if caption.object_id not in labels:
                raise ValueError(f"caption references unknown object {caption.object_id}")
        return self
