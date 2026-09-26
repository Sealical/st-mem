"""Linguistic Trajectory Encoding, adapted from the original LTEEncoder.

Segmentation uses timestamps, keeps motion/static transitions within visible runs,
and does not connect trajectories through missing observations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from pydantic import Field

from stmem.observations import FiniteModel, ObjectObservation


class LTEParams(FiniteModel):
    dp_tolerance_m: float = Field(default=0.15, gt=0)
    stationary_speed_m_s: float = Field(default=0.05, ge=0)
    theta_static_s: float = Field(default=2.0, ge=0)


@dataclass
class TrackSample:
    frame_id: int
    timestamp_s: float
    scene_id: str
    observation: ObjectObservation | None

    @property
    def observed_object(self) -> ObjectObservation:
        if self.observation is None:
            raise ValueError("sample is unobserved")
        return self.observation

    @property
    def position_world(self) -> tuple[float, float, float]:
        position = self.observed_object.position_world
        if position is None:
            raise ValueError("sample has no world position")
        return position


def douglas_peucker_3d(points: np.ndarray, tolerance: float = 0.15) -> list[int]:
    """Indices retained by segment-distance RDP; iterative to handle long tracks."""
    points = np.asarray(points, dtype=np.float64)
    if tolerance <= 0 or not np.isfinite(tolerance):
        raise ValueError("tolerance must be finite and positive")
    if points.size == 0:
        return []
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("points must be finite (N, 3)")
    if len(points) <= 2:
        return list(range(len(points)))
    kept = {0, len(points) - 1}
    pending = [(0, len(points) - 1)]
    while pending:
        start, end = pending.pop()
        if end <= start + 1:
            continue
        segment = points[end] - points[start]
        norm_sq = float(segment @ segment)
        middle = points[start + 1 : end]
        if norm_sq < 1e-12:
            distances = np.linalg.norm(middle - points[start], axis=1)
        else:
            fractions = np.clip((middle - points[start]) @ segment / norm_sq, 0, 1)
            distances = np.linalg.norm(
                middle - points[start] - fractions[:, None] * segment, axis=1
            )
        split = int(np.argmax(distances))
        if distances[split] > tolerance:
            index = start + 1 + split
            kept.add(index)
            pending.extend([(start, index), (index, end)])
    return sorted(kept)


def segment_trajectory(
    samples: list[TrackSample], params: LTEParams
) -> list[tuple[Literal["motion", "static", "unobserved"], list[TrackSample]]]:
    """Return visible motion/static runs and explicit unobserved spans."""
    result: list[tuple[Literal["motion", "static", "unobserved"], list[TrackSample]]] = []
    start = 0
    while start < len(samples):
        visible = samples[start].observation is not None
        end = start + 1
        while end < len(samples) and (
            (samples[end].observation is not None) == visible
            and (not visible or samples[end].scene_id == samples[start].scene_id)
        ):
            end += 1
        run = samples[start:end]
        if not visible:
            result.append(("unobserved", run))
        elif len(run) == 1:
            result.append(("static", run))
        else:
            positions = np.asarray([s.position_world for s in run])
            times = np.asarray([s.timestamp_s for s in run])
            speeds = np.linalg.norm(np.diff(positions, axis=0), axis=1) / np.diff(times)
            static = speeds <= params.stationary_speed_m_s
            edge = 0
            while edge < len(static):
                stop = edge + 1
                while stop < len(static) and static[stop] == static[edge]:
                    stop += 1
                if static[edge] and times[stop] - times[edge] < params.theta_static_s:
                    static[edge:stop] = False
                edge = stop
            edge = 0
            while edge < len(static):
                stop = edge + 1
                while stop < len(static) and static[stop] == static[edge]:
                    stop += 1
                result.append(("static" if static[edge] else "motion", run[edge : stop + 1]))
                edge = stop
        start = end
    return result
