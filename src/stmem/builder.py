"""Build LTE and linked memory views from observations; model access is injected."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import Field

from stmem.lte import (
    LTEParams,
    TrackSample,
    douglas_peucker_3d,
    segment_trajectory,
)
from stmem.observations import (
    CaptionObservation,
    FiniteModel,
    ObservationFrame,
)
from stmem.schema import (
    MemoryEvent,
    MotionInterval,
    ObjectMemory,
    SceneMemory,
    SpatialAnchor,
    STMemory,
    VisualAnchor,
)
from stmem.types import FramePacket


class STMemParams(FiniteModel):
    lte: LTEParams = Field(default_factory=LTEParams)
    captions: Literal["provided", "template", "vlm"] = "provided"
    require_visual_features: bool = True
    require_crops: bool = True


def position_from_depth(packet: FramePacket, box) -> tuple[float, float, float]:
    if packet.depth is None or packet.camera is None or packet.raw_pose_world_camera is None:
        raise ValueError(
            "3D memory needs position_world or aligned depth + camera + T_world_camera"
        )
    depth = np.asarray(packet.depth)
    camera = packet.camera
    if depth.shape != (camera.height, camera.width):
        raise ValueError(
            "depth and camera resolution differ; rescale depth/intrinsics in preprocessing"
        )
    x1, y1, x2, y2 = box
    left, top = max(0, int(x1)), max(0, int(y1))
    right, bottom = min(camera.width, int(np.ceil(x2))), min(camera.height, int(np.ceil(y2)))
    patch = depth[top:bottom, left:right]
    valid = patch[np.isfinite(patch) & (patch > 0)]
    if not len(valid):
        raise ValueError(f"no valid depth for object at frame {packet.frame_id}")
    z = float(np.median(valid))
    if camera.fx <= 0 or camera.fy <= 0:
        raise ValueError("camera focal lengths must be positive")
    camera_point = np.array(
        [
            ((x1 + x2) / 2 - camera.cx) * z / camera.fx,
            ((y1 + y2) / 2 - camera.cy) * z / camera.fy,
            z,
            1,
        ]
    )
    position = (packet.raw_pose_world_camera @ camera_point)[:3]
    if not np.isfinite(position).all():
        raise ValueError("projected world position is not finite")
    return float(position[0]), float(position[1]), float(position[2])


class STMemBuilder:
    def __init__(
        self,
        video_id: str,
        *,
        params: STMemParams | None = None,
        feature_space: str | None = None,
        captions: Sequence[CaptionObservation] = (),
        image_getter: Callable | None = None,
        captioner: Callable | None = None,
        provenance: dict[str, str] | None = None,
    ):
        self.video_id = video_id
        self.params = params or STMemParams()
        self.feature_space = feature_space
        self.captions = captions
        self.image_getter = image_getter
        self.captioner = captioner
        self.provenance = provenance or {}
        self._tracks: dict[str, list[TrackSample]] = {}
        self._labels: dict[str, str] = {}
        self._scenes: dict[str, SceneMemory] = {}
        self._last_time: float | None = None
        self._last_frame: int | None = None
        self._start_time = 0.0
        self._feature_dimension: int | None = None

    def forward(self, inputs: dict) -> dict:
        packet: FramePacket = inputs["packet"]
        frame = packet.extras.get("stmem")
        if not isinstance(frame, ObservationFrame):
            raise TypeError("packet.extras['stmem'] must be an ObservationFrame")
        if frame.frame_id != packet.frame_id or frame.timestamp_s != packet.timestamp_s:
            raise ValueError("observation and packet frame/time mismatch")
        if self._last_time is not None and (
            frame.timestamp_s <= self._last_time
            or (self._last_frame is not None and frame.frame_id <= self._last_frame)
        ):
            raise ValueError("memory observations must be strictly ordered by frame and time")
        resolved = []
        for obj in frame.objects:
            if obj.object_id in self._labels and self._labels[obj.object_id] != obj.label:
                raise ValueError(f"label changed for {obj.object_id}")
            if obj.position_world is None:
                obj = obj.model_copy(
                    update={"position_world": position_from_depth(packet, obj.bbox_xyxy)}
                )
            if obj.visual_feature is not None:
                if not self.feature_space:
                    raise ValueError("declare feature_space for visual features")
                dimension = len(obj.visual_feature)
                if self._feature_dimension not in (None, dimension):
                    raise ValueError("visual feature dimension changed")
                self._feature_dimension = dimension
            elif self.params.require_visual_features:
                raise ValueError(
                    f"missing visual features for {obj.object_id} at frame {frame.frame_id}"
                )
            resolved.append(obj)
        if self._last_time is None:
            self._start_time = frame.timestamp_s
        self._last_time, self._last_frame = frame.timestamp_s, frame.frame_id
        scene = self._scenes.get(frame.scene_id)
        if scene is None:
            scene = SceneMemory(
                scene_id=frame.scene_id,
                label=frame.scene_label,
                t_start=frame.timestamp_s,
                t_end=frame.timestamp_s,
                keyframe_ids=[frame.frame_id],
                object_ids=[],
            )
            self._scenes[frame.scene_id] = scene
        elif scene.label != frame.scene_label:
            raise ValueError("a scene_id must have a stable label; use a new ID for a scene change")
        scene.t_end = frame.timestamp_s
        present = {obj.object_id for obj in resolved}
        for oid, track in self._tracks.items():
            if oid not in present:
                track.append(TrackSample(frame.frame_id, frame.timestamp_s, frame.scene_id, None))
        for obj in resolved:
            self._labels[obj.object_id] = obj.label
            self._tracks.setdefault(obj.object_id, []).append(
                TrackSample(frame.frame_id, frame.timestamp_s, frame.scene_id, obj)
            )
            if obj.object_id not in scene.object_ids:
                scene.object_ids.append(obj.object_id)
        return {"objects_updated": len(resolved)}

    def _caption(self, oid, kind, samples):
        label = self._labels[oid]
        start, end = samples[0].timestamp_s, samples[-1].timestamp_s
        if kind == "unobserved":
            return f"{label} was not observed from {start:g}s to {end:g}s; position is last seen."
        if self.params.captions == "template":
            verb = "moved" if kind == "motion" else "remained stationary"
            return f"{label} {verb} in {self._scenes[samples[0].scene_id].label} from {start:g}s to {end:g}s."
        if self.params.captions == "vlm":
            if self.captioner is None:
                raise ValueError("captions=vlm requires a captioner")
            caption = self.captioner(label, kind, samples, self.image_getter)
            if not isinstance(caption, str) or not caption.strip():
                raise ValueError("VLM returned an empty caption")
            return caption.strip()
        candidates = [
            c for c in self.captions if c.object_id == oid and c.t_start <= end and c.t_end >= start
        ]
        if not candidates:
            raise ValueError(f"missing provided caption for {oid} [{start}, {end}]")
        best = max(
            candidates,
            key=lambda c: (min(c.t_end, end) - max(c.t_start, start), -abs(c.t_start - start)),
        )
        return best.text

    def finalize(self, asset_dir: Path) -> STMemory:
        if self._last_time is None or not self._tracks:
            raise ValueError("cannot build ST-Mem from empty observations/no detected objects")
        asset_dir.mkdir(parents=True, exist_ok=True)
        objects = {}
        events: list[MemoryEvent] = []
        for object_index, (oid, samples) in enumerate(self._tracks.items()):
            intervals: list[MotionInterval] = []
            visual_samples: dict[int, TrackSample] = {}
            observed = [s for s in samples if s.observation is not None]
            # First/last observations must survive even when stationary intervals compress to one position.
            for sample in (observed[0], observed[-1]):
                visual_samples[sample.frame_id] = sample
            for sample_index, sample in enumerate(samples):
                previous = samples[sample_index - 1] if sample_index else None
                if sample.observation is not None and (
                    previous is None or previous.observation is None
                ):
                    events.append(
                        MemoryEvent(
                            event_id=f"event_{len(events)}",
                            object_id=oid,
                            timestamp_s=sample.timestamp_s,
                            kind="appear",
                            description=f"{self._labels[oid]} observed",
                        )
                    )
                elif (
                    sample.observation is None
                    and previous is not None
                    and previous.observation is not None
                ):
                    visual_samples[previous.frame_id] = previous
                    events.append(
                        MemoryEvent(
                            event_id=f"event_{len(events)}",
                            object_id=oid,
                            timestamp_s=sample.timestamp_s,
                            kind="disappear",
                            description=f"{self._labels[oid]} no longer observed",
                        )
                    )
            last_observed = None
            for kind, run in segment_trajectory(samples, self.params.lte):
                if kind == "unobserved":
                    # Anchor time is the actual observation time, not the end of an inferred gap.
                    selected = [last_observed] if last_observed is not None else []
                else:
                    positions = np.asarray([s.position_world for s in run])
                    indices = (
                        [len(run) - 1]
                        if kind == "static"
                        else douglas_peucker_3d(positions, self.params.lte.dp_tolerance_m)
                    )
                    selected = [run[i] for i in indices]
                    last_observed = run[-1]
                    for sample in [run[0], run[-1], *selected]:
                        visual_samples[sample.frame_id] = sample
                interval = MotionInterval(
                    interval_id=f"object_{object_index}/interval_{len(intervals)}",
                    object_id=oid,
                    t_start=run[0].timestamp_s,
                    t_end=run[-1].timestamp_s,
                    kind=kind,
                    caption=self._caption(oid, kind, run),
                    scene_ids=(
                        [last_observed.scene_id]
                        if kind == "unobserved" and last_observed is not None
                        else sorted({s.scene_id for s in run})
                    ),
                    spatial_anchors=[
                        SpatialAnchor(
                            frame_id=s.frame_id,
                            timestamp_s=s.timestamp_s,
                            position_world=s.position_world,
                            observed=(kind != "unobserved"),
                        )
                        for s in selected
                    ],
                )
                if (
                    intervals
                    and kind != "unobserved"
                    and intervals[-1].kind in {"static", "motion"}
                    and intervals[-1].kind != kind
                ):
                    events.append(
                        MemoryEvent(
                            event_id=f"event_{len(events)}",
                            object_id=oid,
                            timestamp_s=interval.t_start,
                            kind="motion_transition",
                            interval_id=interval.interval_id,
                            description=f"{intervals[-1].kind} -> {kind}: {interval.caption}",
                        )
                    )
                intervals.append(interval)
            anchors = []
            for frame_id, sample in sorted(visual_samples.items()):
                obj = sample.observed_object
                image = self.image_getter(frame_id) if self.image_getter else None
                crop_path = None
                if image is not None:
                    h, w = image.shape[:2]
                    x1, y1, x2, y2 = obj.bbox_xyxy
                    crop = image[
                        max(0, int(y1)) : min(h, int(np.ceil(y2))),
                        max(0, int(x1)) : min(w, int(np.ceil(x2))),
                    ]
                    if not crop.size:
                        raise ValueError(f"empty visual crop: {oid}, frame {frame_id}")
                    filename = f"object_{object_index}_frame_{frame_id}.npy"
                    np.save(asset_dir / filename, crop, allow_pickle=False)
                    crop_path = f"{asset_dir.name}/{filename}"
                elif self.params.require_crops:
                    raise ValueError(f"missing RGB for visual anchor: {oid}, frame {frame_id}")
                feature = obj.visual_feature
                if feature is not None:
                    vector = np.asarray(feature, dtype=np.float64)
                    feature = (vector / np.linalg.norm(vector)).tolist()
                anchors.append(
                    VisualAnchor(
                        anchor_id=f"object_{object_index}/frame_{frame_id}",
                        object_id=oid,
                        scene_id=sample.scene_id,
                        frame_id=frame_id,
                        timestamp_s=sample.timestamp_s,
                        bbox_xyxy=obj.bbox_xyxy,
                        position_world=sample.position_world,
                        crop_path=crop_path,
                        feature=feature,
                    )
                )
            for interval in intervals:
                interval.visual_anchor_ids = [
                    a.anchor_id
                    for a in anchors
                    if interval.t_start <= a.timestamp_s <= interval.t_end
                ]
            objects[oid] = ObjectMemory(
                object_id=oid,
                label=self._labels[oid],
                first_seen=observed[0].timestamp_s,
                last_seen=observed[-1].timestamp_s,
                intervals=intervals,
                visual_anchors=anchors,
            )
        provenance = {**self.provenance, "caption_backend": self.params.captions}
        return STMemory(
            video_id=self.video_id,
            t_start=self._start_time,
            t_end=self._last_time,
            feature_space=self.feature_space,
            provenance=provenance,
            objects=objects,
            scenes=self._scenes,
            events=sorted(events, key=lambda event: event.timestamp_s),
        )
