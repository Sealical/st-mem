"""Four functional query paths over LTE, without evaluation dependencies.

Default text retrieval is explicit BM25. An optional text encoder can be injected
for dense retrieval. Visual retrieval always requires a matching feature space.
"""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

from stmem.schema import MemoryViews, STMemory
from stmem.spatial_index import SpatialOctree

_STOP_WORDS = set(
    "a an the is was were of to in on at and or did i my when where what show me from for during".split()
)


def _tokens(text):
    return [word for word in re.findall(r"[^\W_]+", text.lower()) if word not in _STOP_WORDS]


class QueryEngine:
    def __init__(self, memory: STMemory, *, text_encoder=None):
        self.memory = memory
        self.views = MemoryViews(memory)
        self.spatial = SpatialOctree(self.views.images)
        self.text_encoder = text_encoder
        self._entries = self.views.texts
        self._counts = [Counter(_tokens(entry["text"])) for entry in self._entries]
        self._df = Counter(word for counts in self._counts for word in counts)
        self._mean_length = sum(sum(counts.values()) for counts in self._counts) / max(
            len(self._counts), 1
        )
        self._embeddings = None

    def _scores(self, text):
        if not text.strip():
            raise ValueError("query text must not be empty")
        if not self._entries:
            return []
        if self.text_encoder is not None:
            if self._embeddings is None:
                self._embeddings = self._normalize(
                    self.text_encoder([e["text"] for e in self._entries])
                )
            query = self._normalize(self.text_encoder([text]))
            if query.shape[1] != self._embeddings.shape[1]:
                raise ValueError("text encoder dimensions do not match")
            return (self._embeddings @ query[0]).tolist()
        scores = []
        for counts in self._counts:
            score = 0.0
            for word in set(_tokens(text)):
                frequency = counts[word]
                if frequency:
                    inverse_df = math.log(
                        1 + (len(self._counts) - self._df[word] + 0.5) / (self._df[word] + 0.5)
                    )
                    denominator = frequency + 1.2 * (
                        0.25 + 0.75 * sum(counts.values()) / max(self._mean_length, 1)
                    )
                    score += inverse_df * frequency * 2.2 / denominator
            scores.append(score)
        return scores

    @staticmethod
    def _normalize(vectors):
        values = np.asarray(vectors, dtype=float)
        if values.ndim != 2 or not np.isfinite(values).all():
            raise ValueError("encoder must return finite (N, D) vectors")
        lengths = np.linalg.norm(values, axis=1, keepdims=True)
        if np.any(lengths < 1e-12):
            raise ValueError("encoder returned a zero vector")
        return values / lengths

    def _scene_ids(self, scene):
        if scene is None:
            return set(self.memory.scenes)
        return {
            sid
            for sid, record in self.memory.scenes.items()
            if scene.lower() in {sid.lower(), record.label.lower()}
        }

    @staticmethod
    def _time_range(time_range):
        if time_range is None:
            return (-math.inf, math.inf)
        start, end = time_range
        if not np.isfinite([start, end]).all() or start > end:
            raise ValueError("time_range must be finite and ordered")
        return start, end

    def query_nlq(self, text: str, *, top_k=5, scene=None, time_range=None, object_id=None):
        start, end = self._time_range(time_range)
        scenes = self._scene_ids(scene)
        results = []
        for entry, score in zip(self._entries, self._scores(text), strict=True):
            if score <= 0 or entry["t_end"] < start or entry["t_start"] > end:
                continue
            if not scenes.intersection(entry["scene_ids"]) or (
                object_id and entry["object_id"] != object_id
            ):
                continue
            results.append({**entry, "score": float(score), "query_type": "nlq"})
        return sorted(results, key=lambda result: (-result["score"], result["t_start"]))[:top_k]

    def query_vq2d(
        self, feature, *, feature_space: str, top_k=5, threshold=0.7, scene=None, time_range=None
    ):
        if not self.memory.feature_space or feature_space != self.memory.feature_space:
            raise ValueError(
                f"feature space mismatch: memory={self.memory.feature_space!r}, query={feature_space!r}"
            )
        query = self._normalize([feature])[0]
        start, end = self._time_range(time_range)
        scenes = self._scene_ids(scene)
        results = []
        for anchor in self.views.images:
            if (
                anchor.feature is None
                or anchor.scene_id not in scenes
                or not start <= anchor.timestamp_s <= end
            ):
                continue
            if len(anchor.feature) != len(query):
                raise ValueError("visual query and memory feature dimensions differ")
            score = float(np.dot(query, self._normalize([anchor.feature])[0]))
            if score >= threshold:
                results.append(
                    {**anchor.model_dump(), "feature": None, "score": score, "query_type": "vq2d"}
                )
        return sorted(results, key=lambda result: (-result["score"], -result["timestamp_s"]))[
            :top_k
        ]

    def query_str(
        self, text: str = "", *, object_id=None, scene=None, time_range=None, region=None, top_k=5
    ):
        start, end = self._time_range(time_range)
        scenes = self._scene_ids(scene)
        interval_scores = {}
        if text:
            matches = self.query_nlq(
                text,
                top_k=len(self._entries),
                scene=scene,
                time_range=time_range,
                object_id=object_id,
            )
            interval_scores = {r["interval_id"]: r["score"] for r in matches if r["interval_id"]}
        region_ids = None
        if region is not None:
            region_ids = {
                a.object_id
                for a in self.spatial.query_box(*region)
                if start <= a.timestamp_s <= end and a.scene_id in scenes
            }
        results = []
        for oid, obj in self.memory.objects.items():
            if (object_id and oid != object_id) or (
                region_ids is not None and oid not in region_ids
            ):
                continue
            for interval in obj.intervals:
                if (
                    interval.t_end < start
                    or interval.t_start > end
                    or not scenes.intersection(interval.scene_ids)
                ):
                    continue
                if text and interval.interval_id not in interval_scores:
                    continue
                anchors = [
                    a.model_dump()
                    for a in interval.spatial_anchors
                    if start <= a.timestamp_s <= end
                ]
                if region is not None:
                    lower, upper = np.asarray(region[0]), np.asarray(region[1])
                    anchors = [
                        a
                        for a in anchors
                        if np.all(np.asarray(a["position_world"]) >= lower)
                        and np.all(np.asarray(a["position_world"]) <= upper)
                    ]
                    if not anchors:
                        continue
                results.append(
                    {
                        "query_type": "str",
                        "object_id": oid,
                        "label": obj.label,
                        "interval_id": interval.interval_id,
                        "t_start": max(start, interval.t_start),
                        "t_end": min(end, interval.t_end),
                        "kind": interval.kind,
                        "text": interval.caption,
                        "spatial_anchors": anchors,
                        "visual_anchor_ids": [
                            a.anchor_id
                            for a in obj.visual_anchors
                            if a.anchor_id in interval.visual_anchor_ids
                            and start <= a.timestamp_s <= end
                            and a.scene_id in scenes
                        ],
                        "score": float(interval_scores.get(interval.interval_id, 1.0)),
                    }
                )
        return sorted(results, key=lambda result: (-result["score"], result["t_start"]))[:top_k]

    def query_lor(
        self,
        label: str | None = None,
        *,
        object_id=None,
        lookback_seconds=None,
        as_of=None,
        scene=None,
        top_k=5,
    ):
        """Latest observed anchor within the window; sparse history for earlier as_of queries."""
        end = self.memory.t_end if as_of is None else float(as_of)
        if not np.isfinite(end) or (
            lookback_seconds is not None
            and (not np.isfinite(lookback_seconds) or lookback_seconds < 0)
        ):
            raise ValueError("as_of/lookback must be finite; lookback must be nonnegative")
        start = -math.inf if lookback_seconds is None else end - lookback_seconds
        scenes = self._scene_ids(scene)
        results = []
        for oid, obj in self.memory.objects.items():
            if (object_id and oid != object_id) or (label and label.lower() != obj.label.lower()):
                continue
            candidates = [
                a
                for a in obj.visual_anchors
                if start <= a.timestamp_s <= end and a.scene_id in scenes
            ]
            if candidates:
                last = max(candidates, key=lambda a: a.timestamp_s)
                results.append(
                    {
                        **last.model_dump(),
                        "feature": None,
                        "label": obj.label,
                        "query_type": "lor",
                        "evidence": "observed",
                    }
                )
        return sorted(results, key=lambda result: (-result["timestamp_s"], result["object_id"]))[
            :top_k
        ]
