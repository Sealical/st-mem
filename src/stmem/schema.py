"""LTE records and five linked views, ported from spatial-temporal-memory.

All views reference the same records. Images are portable paths relative to memory.json.
An unobserved interval holds the last known position, never evidence of stationarity.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from stmem.observations import FiniteModel


class SpatialAnchor(FiniteModel):
    frame_id: int
    timestamp_s: float
    position_world: tuple[float, float, float]
    observed: bool = True


class VisualAnchor(FiniteModel):
    anchor_id: str
    object_id: str
    scene_id: str
    frame_id: int
    timestamp_s: float
    bbox_xyxy: tuple[float, float, float, float]
    position_world: tuple[float, float, float]
    crop_path: str | None = None
    feature: list[float] | None = None


class MotionInterval(FiniteModel):
    interval_id: str
    object_id: str
    t_start: float
    t_end: float
    kind: Literal["motion", "static", "unobserved"]
    caption: str
    scene_ids: list[str]
    spatial_anchors: list[SpatialAnchor]
    visual_anchor_ids: list[str] = Field(default_factory=list)


class ObjectMemory(FiniteModel):
    object_id: str
    label: str
    first_seen: float
    last_seen: float
    intervals: list[MotionInterval]
    visual_anchors: list[VisualAnchor]


class SceneMemory(FiniteModel):
    scene_id: str
    label: str
    t_start: float
    t_end: float
    keyframe_ids: list[int]
    object_ids: list[str]


class MemoryEvent(FiniteModel):
    event_id: str
    object_id: str
    timestamp_s: float
    kind: Literal["appear", "disappear", "motion_transition"]
    interval_id: str | None = None
    description: str


class STMemory(FiniteModel):
    schema_version: Literal[1] = 1
    video_id: str
    t_start: float
    t_end: float
    feature_space: str | None = None
    provenance: dict[str, str] = Field(default_factory=dict)
    objects: dict[str, ObjectMemory]
    scenes: dict[str, SceneMemory]
    events: list[MemoryEvent]

    @model_validator(mode="after")
    def validate_links(self):
        anchor_ids = set()
        interval_ids = set()
        dimensions = set()
        for oid, obj in self.objects.items():
            if oid != obj.object_id:
                raise ValueError("object dictionary key mismatch")
            own_anchors = {a.anchor_id for a in obj.visual_anchors}
            if len(own_anchors) != len(obj.visual_anchors) or own_anchors & anchor_ids:
                raise ValueError("duplicate visual anchor IDs")
            anchor_ids.update(own_anchors)
            for anchor in obj.visual_anchors:
                if anchor.object_id != oid or anchor.scene_id not in self.scenes:
                    raise ValueError("invalid visual anchor reference")
                if anchor.feature is not None:
                    dimensions.add(len(anchor.feature))
            for interval in obj.intervals:
                if interval.interval_id in interval_ids or interval.object_id != oid:
                    raise ValueError("invalid interval ID")
                interval_ids.add(interval.interval_id)
                if interval.t_end < interval.t_start:
                    raise ValueError("interval ends before it starts")
                if not set(interval.visual_anchor_ids) <= own_anchors:
                    raise ValueError("interval references unknown visual anchor")
                if not set(interval.scene_ids) <= self.scenes.keys():
                    raise ValueError("interval references unknown scene")
        if len(dimensions) > 1 or (dimensions and not self.feature_space):
            raise ValueError("visual features require one feature space and dimension")
        for sid, scene in self.scenes.items():
            if sid != scene.scene_id or not set(scene.object_ids) <= self.objects.keys():
                raise ValueError("invalid scene reference")
        for event in self.events:
            if event.object_id not in self.objects or (
                event.interval_id is not None and event.interval_id not in interval_ids
            ):
                raise ValueError("invalid event reference")
        return self


class MemoryViews:
    def __init__(self, memory: STMemory):
        self.memory = memory

    @property
    def objects(self):
        return self.memory.objects

    @property
    def scenes(self):
        return self.memory.scenes

    @property
    def events(self):
        return self.memory.events

    @property
    def images(self) -> list[VisualAnchor]:
        return [a for obj in self.objects.values() for a in obj.visual_anchors]

    @property
    def texts(self) -> list[dict]:
        entries = [
            {
                "text": iv.caption,
                "object_id": oid,
                "interval_id": iv.interval_id,
                "t_start": iv.t_start,
                "t_end": iv.t_end,
                "scene_ids": iv.scene_ids,
            }
            for oid, obj in self.objects.items()
            for iv in obj.intervals
        ]
        entries.extend(
            {
                "text": f"Scene: {scene.label}",
                "object_id": None,
                "interval_id": None,
                "t_start": scene.t_start,
                "t_end": scene.t_end,
                "scene_ids": [sid],
            }
            for sid, scene in self.scenes.items()
        )
        return entries
